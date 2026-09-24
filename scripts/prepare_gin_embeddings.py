#!/usr/bin/env python3
"""Build the dataset-specific E5 cache using the frozen trainer's preprocessing."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("GDELT", "ICEWS1819", "Enron", "Googlemap_CT"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="intfloat/e5-large-v2", help="Model identifier or local E5-large-v2 directory")
    args = parser.parse_args()
    if "e5" not in args.model.lower():
        parser.error("The frozen encoder recognizes E5 by its name; keep 'e5' in a local model directory name")
    output = args.output.resolve()
    metadata = output.with_suffix(output.suffix + ".json")
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if output.exists() or metadata.exists():
        raise FileExistsError("Refusing to overwrite an embedding cache or its metadata")
    source = args.data_root.resolve() / args.dataset / "entity_text.csv"
    if not source.is_file():
        raise FileNotFoundError(source)

    import numpy as np
    import pandas as pd
    from experiments.gin.protocol import sha256_file, verify_sources
    from experiments.modules.heuristic_models import precompute_entity_embeddings
    from experiments.modules.llm_lp.experiment import build_prompt_entity_map

    verify_sources(ROOT, ROOT / "configs/gin/source-pins.json")
    spec = json.loads((ROOT / "configs/gin" / f"{args.dataset}.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(source)
    if not {"i", "text"}.issubset(frame.columns) or frame["i"].duplicated().any():
        raise ValueError("entity_text.csv needs unique entity ids in 'i' and a 'text' column")
    entity_map = build_prompt_entity_map(args.dataset, dict(zip(frame["i"], frame["text"])),
                                         entity_name_mode=spec["embedding_entity_name_mode"])
    embeddings, mapping = precompute_entity_embeddings(entity_map, model_name=args.model, device=args.device)
    if embeddings.shape != (len(entity_map), 1024) or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite 1024-dimensional E5-large-v2 embeddings for every entity")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.save(stream, embeddings, allow_pickle=False)
    details = {"dataset": args.dataset, "model": args.model, "device": args.device,
               "entity_name_mode": spec["embedding_entity_name_mode"],
               "entity_text_sha256": sha256_file(source), "embedding_sha256": sha256_file(output),
               "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
               "entity_ids_in_row_order": sorted(mapping, key=mapping.get)}
    details["reference_cache_match"] = details["embedding_sha256"] == spec["reference_input_sha256"]["embedding_cache"]
    with metadata.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"output": str(output), "metadata": str(metadata),
                      "reference_cache_match": details["reference_cache_match"]}, indent=2))


if __name__ == "__main__":
    main()
