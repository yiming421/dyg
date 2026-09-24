import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser("Export DTGB BERT entity/relation text features")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_root", default="../DyLink_Datasets")
    parser.add_argument("--pretrained_model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def export_text_features(
    *,
    csv_path: Path,
    output_path: Path,
    max_id: int,
    model,
    tokenizer,
    hidden_size: int,
    batch_size: int,
    max_length: int,
    precision: int,
    device: torch.device,
    output_dtype: np.dtype,
    overwrite: bool,
):
    if output_path.exists() and not overwrite:
        print(f"Keeping existing feature file: {output_path}", flush=True)
        return

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    features = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=output_dtype,
        shape=(max_id + 1, hidden_size),
    )
    features[0] = 0
    seen = np.zeros(max_id + 1, dtype=bool)
    seen[0] = True

    reader = pd.read_csv(csv_path, chunksize=batch_size)
    with tqdm(total=max_id, desc=output_path.name) as progress:
        for batch in reader:
            ids = batch["i"].to_numpy(dtype=np.int64, copy=False)
            texts = batch["text"].fillna("NULL").astype(str).tolist()
            keep = ids != 0
            ids = ids[keep]
            texts = [text for text, retain in zip(texts, keep) if retain]
            if len(ids) == 0:
                continue
            if ids.min() < 1 or ids.max() > max_id:
                raise ValueError(f"Out-of-range id in {csv_path}: [{ids.min()}, {ids.max()}]")
            if len(np.unique(ids)) != len(ids) or seen[ids].any():
                raise ValueError(f"Duplicate id in {csv_path}")

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model(**encoded)
                pooled = outputs.pooler_output
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0]
            values = np.round(pooled.float().cpu().numpy(), precision).astype(output_dtype, copy=False)
            features[ids] = values
            seen[ids] = True
            progress.update(len(ids))

    missing = np.flatnonzero(~seen)
    if len(missing):
        raise ValueError(f"Missing {len(missing)} ids in {csv_path}; first ids: {missing[:10].tolist()}")
    features.flush()
    del features
    os.replace(temporary_path, output_path)
    print(f"Saved {output_path} with shape {(max_id + 1, hidden_size)}", flush=True)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    dataset_root = Path(args.data_root).expanduser().resolve() / args.dataset_name
    edge_list_path = dataset_root / "edge_list.csv"
    if not edge_list_path.exists():
        raise FileNotFoundError(edge_list_path)

    edge_list = pd.read_csv(edge_list_path, usecols=["u", "i", "r"])
    max_node_id = int(max(edge_list["u"].max(), edge_list["i"].max()))
    max_relation_id = int(edge_list["r"].max())

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    model = AutoModel.from_pretrained(args.pretrained_model_name).to(device).eval()
    output_dtype = np.dtype(args.output_dtype)
    print(
        f"Model={args.pretrained_model_name} dataset={args.dataset_name} device={device} "
        f"dtype={output_dtype} batch_size={args.batch_size}",
        flush=True,
    )

    started_at = time.time()
    export_text_features(
        csv_path=dataset_root / "entity_text.csv",
        output_path=dataset_root / "e_feat.npy",
        max_id=max_node_id,
        model=model,
        tokenizer=tokenizer,
        hidden_size=config.hidden_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        precision=args.precision,
        device=device,
        output_dtype=output_dtype,
        overwrite=args.overwrite,
    )
    export_text_features(
        csv_path=dataset_root / "relation_text.csv",
        output_path=dataset_root / "r_feat.npy",
        max_id=max_relation_id,
        model=model,
        tokenizer=tokenizer,
        hidden_size=config.hidden_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        precision=args.precision,
        device=device,
        output_dtype=output_dtype,
        overwrite=args.overwrite,
    )
    print(f"Feature export completed in {time.time() - started_at:.2f}s", flush=True)


if __name__ == "__main__":
    main()
