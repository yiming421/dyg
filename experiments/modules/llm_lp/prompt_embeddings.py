"""Utilities for persisting one contextual LLM vector per prediction prompt."""

import hashlib
import os
import uuid

import numpy as np
import torch


def _hidden_states_path(output):
    params = getattr(output, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    path = params.get("hidden_states_path")
    return os.path.abspath(path) if path else None


def discard_vllm_hidden_state_files(outputs):
    """Remove connector scratch files without loading their tensors."""
    removed = 0
    for output in outputs or []:
        path = _hidden_states_path(output)
        if path and os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def extract_prompt_embeddings(outputs, *, normalize=True, delete_sources=True):
    """Load selected-layer, last-prompt-token states returned by the vLLM connector."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required when --capture_prompt_embeddings is enabled."
        ) from exc

    vectors = []
    paths = [_hidden_states_path(output) for output in (outputs or [])]
    try:
        for index, path in enumerate(paths):
            if not path or not os.path.isfile(path):
                raise RuntimeError(
                    "vLLM did not return a prompt hidden-state file for output "
                    f"{index}. Check that vLLM supports extract_hidden_states and the "
                    "LastTokenHiddenStatesConnector."
                )
            tensors = load_file(path, device="cpu")
            hidden_states = tensors.get("hidden_states")
            if hidden_states is None or hidden_states.numel() == 0:
                raise RuntimeError(f"Missing hidden_states tensor in {path}")

            # The optimized connector stores [num_layers, hidden_size]. Keep
            # compatibility with vLLM's stock [tokens, num_layers, hidden_size]
            # connector so old scratch files remain readable.
            if hidden_states.ndim == 3:
                vector = hidden_states[-1, -1]
            elif hidden_states.ndim == 2:
                vector = hidden_states[-1]
            elif hidden_states.ndim == 1:
                vector = hidden_states
            else:
                raise RuntimeError(
                    f"Unexpected hidden_states shape {tuple(hidden_states.shape)} in {path}"
                )
            vector = vector.float()
            if normalize:
                norm = torch.linalg.vector_norm(vector)
                if torch.isfinite(norm) and float(norm) > 0.0:
                    vector = vector / norm
            vectors.append(vector.numpy())
    finally:
        if delete_sources:
            for path in paths:
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32, copy=False)


def save_prompt_embedding_shard(
    output_dir,
    embeddings,
    samples,
    prompts,
    *,
    eval_split,
    layer_id,
    normalize,
    save_dtype="float16",
    representation="final_rmsnorm_last_prompt_token_pre_lm_head",
    binary_scores=None,
    model_name_or_path=None,
):
    """Save aligned prompt vectors and stable link-prediction identifiers."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a rank-2 embedding matrix, got {embeddings.shape}")
    if not (len(samples) == len(prompts) == embeddings.shape[0]):
        raise ValueError(
            "Prompt embedding alignment mismatch: "
            f"samples={len(samples)}, prompts={len(prompts)}, vectors={embeddings.shape[0]}"
        )
    if binary_scores is not None and len(binary_scores) != embeddings.shape[0]:
        raise ValueError(
            "Binary-score alignment mismatch: "
            f"scores={len(binary_scores)}, vectors={embeddings.shape[0]}"
        )

    dtype = np.float16 if str(save_dtype).lower() == "float16" else np.float32
    os.makedirs(output_dir, exist_ok=True)
    safe_split = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(eval_split)
    )
    filename = f"prompt_embeddings_{safe_split}_{os.getpid()}_{uuid.uuid4().hex}.npz"
    path = os.path.abspath(os.path.join(output_dir, filename))
    tmp_path = path + ".tmp"

    def integer_field(sample, key, default=-1):
        value = sample.get(key, default)
        return int(default if value is None else value)

    prompt_hashes = [
        hashlib.sha256(str(prompt).encode("utf-8")).hexdigest() for prompt in prompts
    ]
    payload = {
        "embeddings": embeddings.astype(dtype, copy=False),
        "row_index": np.arange(len(samples), dtype=np.int64),
        "query_id": np.asarray(
            [integer_field(sample, "query_id", index) for index, sample in enumerate(samples)],
            dtype=np.int64,
        ),
        "source_id": np.asarray(
            [integer_field(sample, "source_id") for sample in samples], dtype=np.int64
        ),
        "target_id": np.asarray(
            [integer_field(sample, "target_id") for sample in samples], dtype=np.int64
        ),
        "relation_id": np.asarray(
            [integer_field(sample, "relation_id") for sample in samples], dtype=np.int64
        ),
        "timestamp": np.asarray(
            [
                float(
                    -1.0
                    if sample.get("timestamp", -1.0) is None
                    else sample.get("timestamp", -1.0)
                )
                for sample in samples
            ],
            dtype=np.float64,
        ),
        "dtgb_timestamp": np.asarray(
            [
                float(
                    sample.get(
                        "dtgb_timestamp",
                        -1.0 if sample.get("timestamp") is None else sample.get("timestamp", -1.0),
                    )
                )
                for sample in samples
            ],
            dtype=np.float64,
        ),
        "label": np.asarray(
            [integer_field(sample, "label") for sample in samples], dtype=np.int8
        ),
        "prompt_sha256": np.asarray(prompt_hashes, dtype="U64"),
        "layer_id": np.asarray(int(layer_id), dtype=np.int64),
        "normalized": np.asarray(bool(normalize), dtype=np.bool_),
        "representation": np.asarray(str(representation)),
        "eval_split": np.asarray(str(eval_split)),
        "model_name_or_path": np.asarray(str(model_name_or_path or "unknown")),
    }
    if binary_scores is not None:
        scores = np.asarray(binary_scores, dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Binary scores must all be finite.")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("Binary scores must lie in [0, 1].")
        clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
        payload["llm_binary_score"] = scores.astype(np.float32)
        payload["llm_binary_logit_margin"] = np.log(clipped / (1.0 - clipped)).astype(
            np.float32
        )
    with open(tmp_path, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp_path, path)

    return {
        "path": path,
        "count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "dtype": np.dtype(dtype).name,
        "layer_id": int(layer_id),
        "normalized": bool(normalize),
        "representation": str(representation),
        "eval_split": str(eval_split),
        "has_binary_score": binary_scores is not None,
        "model_name_or_path": str(model_name_or_path or "unknown"),
    }
