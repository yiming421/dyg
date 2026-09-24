"""
Model loading and evaluation helpers for LLM-based link prediction.
"""
import atexit
import json
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from experiments.modules.llm_lp.eval_helpers import (
    compute_binary_logprob_features,
    compute_binary_score_from_logprobs,
    extract_score_0_100_with_meta,
)
from experiments.modules.llm_lp.data_parallel import (
    data_parallel_run_dir,
    resolve_data_parallel_sync_dir,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics
from experiments.modules.llm_lp.prompt_template import (
    DEFAULT_KEY_SIGNAL_FIELDS,
    create_prompt,
)
from experiments.modules.llm_lp.prompt_embeddings import (
    discard_vllm_hidden_state_files,
    extract_prompt_embeddings,
    save_prompt_embedding_shard,
)

try:
    import vllm as _vllm_pkg
    from vllm import LLM, SamplingParams
except ImportError:
    _vllm_pkg = None
    LLM = None
    SamplingParams = None


def _llm_log(message):
    print(f"[LLM] {message}")


def _llm_warn(message):
    print(f"[LLM][WARN] {message}")


def _rrf_log(message):
    print(f"[RRF] {message}")


def _resolve_stop_token_ids(tokenizer):
    """
    Derive model-appropriate stop IDs instead of hard-coding Llama tokens.
    """
    stop_ids = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple, set)):
        stop_ids.extend(int(tok_id) for tok_id in eos_token_id if tok_id is not None)
    elif eos_token_id is not None:
        stop_ids.append(int(eos_token_id))

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token_text in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tok_id = tokenizer.convert_tokens_to_ids(token_text)
        except Exception:
            tok_id = None
        if tok_id is None or tok_id == unk_token_id or tok_id < 0:
            continue
        stop_ids.append(int(tok_id))

    return list(dict.fromkeys(stop_ids))


def load_model_transformers(model_path, quantization=None):
    """Load model using standard transformers (fallback)."""
    _llm_log(f"Loading Transformers model from {model_path}...")
    model_path_lower = str(model_path).lower()
    quantization_name = str(quantization).strip().lower() if quantization is not None else None
    is_qwen = "qwen" in model_path_lower
    use_bfloat16 = "llama-3.2-1b" in model_path_lower

    tokenizer_kwargs = {}
    if is_qwen:
        tokenizer_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if use_bfloat16 else torch.float16,
    }
    if quantization_name in {"bitsandbytes", "bnb", "4bit"}:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "bitsandbytes is not installed, but transformers 4-bit quantization was requested. "
                "Install bitsandbytes or rerun without --quantization bitsandbytes."
            )
        model_kwargs["load_in_4bit"] = True
        _llm_log("Using bitsandbytes 4-bit quantization for Transformers load")
    if is_qwen:
        model_kwargs["torch_dtype"] = "auto"
        model_kwargs["trust_remote_code"] = True
    elif use_bfloat16:
        _llm_log("Using bfloat16 for Llama-3.2-1B Transformers load")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    _llm_log("Model loaded (Transformers)")
    return model, tokenizer


def load_model_and_tokenizer(
    model_path,
    quantization=None,
    tensor_parallel_size=1,
    max_model_len=16384,
    gpu_utilization=0.8,
    enforce_eager=False,
    capture_prompt_embeddings=False,
    prompt_embedding_storage_dir=None,
    prompt_embedding_layer=-1,
):
    """Load vLLM model and tokenizer."""
    if LLM is None:
        raise ImportError(
            "vLLM is not installed. Install a recent vLLM build or rerun with --use_transformers."
        )
    _llm_log(f"Loading vLLM model from {model_path}...")
    _llm_log(
        f"Config: TP={tensor_parallel_size}, Max Context={max_model_len}, GPU Util={gpu_utilization}, "
        f"Eager={bool(enforce_eager)}"
    )
    model_path_lower = str(model_path).lower()
    configured_dtype = None
    config_path = os.path.join(str(model_path), "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                model_config_payload = json.load(handle)
            configured_dtype = str(
                model_config_payload.get("dtype")
                or model_config_payload.get("torch_dtype")
                or ""
            ).strip().lower()
        except Exception:
            configured_dtype = None
    llm_dtype = (
        "bfloat16"
        if configured_dtype in {"bfloat16", "bf16", "torch.bfloat16"}
        or "llama-3.2-1b" in model_path_lower
        else "float16"
    )
    if llm_dtype == "bfloat16":
        _llm_log("Using bfloat16 from the model configuration/path for vLLM load")

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype=llm_dtype,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_utilization,
        max_model_len=max_model_len,
        enforce_eager=bool(enforce_eager),
    )
    embedding_layer_id = None
    owned_storage_dir = None
    capture_bootstrap_environment = None
    if capture_prompt_embeddings:
        try:
            from vllm.config.kv_transfer import KVTransferConfig
        except ImportError as exc:
            raise RuntimeError(
                "Prompt embedding capture requires a vLLM build with "
                "KVTransferConfig and extract_hidden_states support (tested with vLLM 0.17)."
            ) from exc

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_config = model_config
        get_text_config = getattr(model_config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_config = get_text_config()
            except TypeError:
                text_config = get_text_config(decoder=True)
        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            for field_name in ("n_layer", "num_layers", "n_layers"):
                num_hidden_layers = getattr(text_config, field_name, None)
                if num_hidden_layers is not None:
                    break
        if num_hidden_layers is None:
            raise ValueError(
                "Could not determine num_hidden_layers from the model config; "
                "cannot select a prompt embedding layer."
            )
        num_hidden_layers = int(num_hidden_layers)
        model_architectures = tuple(getattr(text_config, "architectures", ()) or ())
        if "LlamaForCausalLM" not in model_architectures:
            raise RuntimeError(
                "Raw final-RMSNorm prompt capture is currently implemented only for "
                f"LlamaForCausalLM; model architectures are {model_architectures or '(unknown)'}."
            )
        # vLLM's ordinary auxiliary ids [0, N-1] denote block inputs.  The
        # custom connector uses N as a sentinel for the post-final-RMSNorm
        # state, which is the actual LM-head input.
        embedding_layer_id = (
            num_hidden_layers
            if int(prompt_embedding_layer) == -1
            else int(prompt_embedding_layer)
        )
        if not 0 <= embedding_layer_id <= num_hidden_layers:
            raise ValueError(
                "--prompt_embedding_layer must be -1 or between 0 and "
                f"{num_hidden_layers}; got {prompt_embedding_layer}."
            )
        embedding_representation = (
            "final_rmsnorm_last_prompt_token_pre_lm_head"
            if embedding_layer_id == num_hidden_layers
            else "pre_decoder_block_last_prompt_token"
        )

        if embedding_layer_id == num_hidden_layers:
            # vLLM imports/compiles the model before constructing the custom KV
            # connector.  A capture-only sitecustomize path ensures spawned
            # engine workers install the final-state shim before model loading.
            bootstrap_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "vllm_capture_bootstrap",
                )
            )
            repo_root = os.path.dirname(os.path.dirname(bootstrap_dir))
            capture_bootstrap_environment = {
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "DTGB_VLLM_CAPTURE_FINAL_HIDDEN": os.environ.get(
                    "DTGB_VLLM_CAPTURE_FINAL_HIDDEN"
                ),
                "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get(
                    "VLLM_WORKER_MULTIPROC_METHOD"
                ),
            }
            python_path_entries = [
                entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
            ]
            os.environ["PYTHONPATH"] = os.pathsep.join(
                [
                    entry
                    for entry in (bootstrap_dir, repo_root, *python_path_entries)
                    if entry not in {""}
                ]
            )
            os.environ["DTGB_VLLM_CAPTURE_FINAL_HIDDEN"] = "1"
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        if prompt_embedding_storage_dir:
            storage_dir = os.path.abspath(prompt_embedding_storage_dir)
            os.makedirs(storage_dir, exist_ok=True)
        else:
            ramdisk_root = "/dev/shm"
            temp_root = (
                ramdisk_root
                if os.path.isdir(ramdisk_root) and os.access(ramdisk_root, os.W_OK)
                else None
            )
            storage_dir = tempfile.mkdtemp(
                prefix=f"dtgb_prompt_hidden_{os.getpid()}_",
                dir=temp_root,
            )
            owned_storage_dir = storage_dir
            atexit.register(shutil.rmtree, owned_storage_dir, ignore_errors=True)

        llm_kwargs.update(
            {
                "enable_chunked_prefill": False,
                "speculative_config": {
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {
                            "eagle_aux_hidden_state_layer_ids": [embedding_layer_id],
                        }
                    },
                },
                "kv_transfer_config": KVTransferConfig(
                    kv_connector="LastTokenHiddenStatesConnector",
                    kv_role="kv_producer",
                    kv_connector_module_path=(
                        "experiments.modules.llm_lp.vllm_prompt_embedding_connector"
                    ),
                    kv_connector_extra_config={"shared_storage_path": storage_dir},
                ),
            }
        )
        _llm_log(
            "Prompt embedding capture enabled: "
            f"hidden_layer={embedding_layer_id}, representation={embedding_representation}, "
            f"scratch={storage_dir}. Chunked prefill is disabled by vLLM for this mode."
        )
    try:
        llm = LLM(**llm_kwargs)
    except KeyError as exc:
        missing_key = str(exc).strip("'\"")
        config_path = os.path.join(str(model_path), "config.json")
        rope_scaling = None
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    model_cfg = json.load(handle)
                rope_scaling = model_cfg.get("rope_scaling")
            except Exception:
                rope_scaling = None
        if (
            missing_key == "type"
            and isinstance(rope_scaling, dict)
            and "rope_type" in rope_scaling
            and "type" not in rope_scaling
        ):
            vllm_version = getattr(_vllm_pkg, "__version__", "unknown")
            raise RuntimeError(
                "vLLM failed to parse model rope_scaling because this environment uses an older "
                f"vLLM build (detected vLLM={vllm_version}). "
                "This Llama-3.x config uses rope_scaling.rope_type; older vLLM expects "
                "rope_scaling.type. Use a newer vLLM environment (e.g. your `vllm` conda env) "
                "or rerun with --use_transformers."
            ) from exc
        raise
    finally:
        if capture_bootstrap_environment is not None:
            for key, previous_value in capture_bootstrap_environment.items():
                if previous_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous_value

    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm._dtgb_capture_prompt_embeddings = bool(capture_prompt_embeddings)
    llm._dtgb_prompt_embedding_layer_id = embedding_layer_id
    llm._dtgb_prompt_embedding_representation = (
        embedding_representation if capture_prompt_embeddings else None
    )
    llm._dtgb_prompt_embedding_storage_dir = (
        owned_storage_dir or prompt_embedding_storage_dir
    )

    _llm_log("Model loaded")
    return llm, tokenizer


def sanitize_torch_distributed_env_for_vllm():
    """
    torchrun exports distributed env vars that can interfere with vLLM's
    internal TCPStore/bootstrap even when each process runs TP=1.
    """
    dist_env_keys = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    ]
    removed_keys = []
    for key in dist_env_keys:
        if key in os.environ:
            os.environ.pop(key, None)
            removed_keys.append(key)
    if removed_keys:
        _llm_log(f"Sanitized torch distributed env for vLLM: removed {removed_keys}")


def maybe_spawn_local_dp_workers(args, script_path, argv):
    """
    Convenience launcher:
    If --data_parallel_size > 1 and no rank env is present, spawn one local worker
    process per rank and let those workers run the actual evaluation.
    """
    if args.data_parallel_size <= 1:
        return False

    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None:
        return False

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
    else:
        device_count = torch.cuda.device_count()
        devices = [str(i) for i in range(device_count)]

    if len(devices) < args.data_parallel_size:
        raise RuntimeError(
            f"Need at least {args.data_parallel_size} visible GPUs, got {len(devices)}. "
            "Set CUDA_VISIBLE_DEVICES accordingly."
        )

    configured_sync_dir = getattr(args, "data_parallel_sync_dir", None)
    sync_dir = resolve_data_parallel_sync_dir(args.output, sync_dir=configured_sync_dir)
    run_id = str(os.environ.get("DATA_PARALLEL_RUN_ID", time.time_ns()))
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(sync_dir, exist_ok=True)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    cmd = [sys.executable, os.path.abspath(script_path)] + list(argv)

    print(
        f"Auto-launching {args.data_parallel_size} DP workers locally "
        f"(run_id={run_id}, devices={devices[:args.data_parallel_size]}, sync_dir={sync_dir})"
    )

    procs = []
    for rank in range(args.data_parallel_size):
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["WORLD_SIZE"] = str(args.data_parallel_size)
        env["DATA_PARALLEL_RUN_ID"] = run_id
        env["DATA_PARALLEL_SYNC_DIR"] = sync_dir
        env["CUDA_VISIBLE_DEVICES"] = devices[rank]
        env["PYTHONUNBUFFERED"] = "1"
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"Some DP workers failed: exit_codes={exit_codes}")

    print("All DP workers finished.")
    return True


def _select_key_signal_values(
    sample,
    use_raw_key_signals=False,
    use_percentile_key_signals=False,
):
    def _global_recency_raw_value():
        raw_value = sample.get("heuristic_global_recency_score")
        if raw_value is None:
            return "No prior target interactions"
        try:
            raw_value = float(raw_value)
        except (TypeError, ValueError):
            return raw_value
        if raw_value <= -1e14:
            return "No prior target interactions"
        return -raw_value

    if use_raw_key_signals:
        last_interaction_signal = sample.get("last_interaction_delta")
        if last_interaction_signal is None:
            last_interaction_signal = "No prior interactions"
        return {
            "source_popularity": sample.get("source_popularity_raw", sample.get("source_popularity", 0)),
            "target_popularity": sample.get("target_popularity_raw", sample.get("target_popularity", 0)),
            "past_interactions": sample.get(
                "num_past_interactions_raw",
                sample.get("num_past_interactions", 0),
            ),
            "recency": last_interaction_signal,
            "common_neighbor": sample.get(
                "common_neighbor_score",
                sample.get("common_neighbor_level", "Modest"),
            ),
            "recent_degree": sample.get("heuristic_recent_degree_score"),
            "global_recency": _global_recency_raw_value(),
            "itemcf": sample.get("heuristic_itemcf_score"),
            "usercf": sample.get("heuristic_usercf_score"),
        }

    if use_percentile_key_signals:
        return {
            "source_popularity": sample.get("source_popularity_pct", 50.0),
            "target_popularity": sample.get("target_popularity_pct", 50.0),
            "past_interactions": sample.get("num_past_interactions_pct", 50.0),
            "recency": sample.get("last_interaction_recency_pct", 0.0),
            "common_neighbor": sample.get("common_neighbor_score_pct", 50.0),
            "recent_degree": sample.get("recent_degree_pct", 50.0),
            "global_recency": sample.get("global_recency_pct", 0.0),
            "itemcf": sample.get("itemcf_pct", 50.0),
            "usercf": sample.get("usercf_pct", 50.0),
        }

    return {
        "source_popularity": sample.get("source_popularity", 0),
        "target_popularity": sample.get("target_popularity", 0),
        "past_interactions": sample.get("num_past_interactions", 0),
        "recency": sample.get("last_interaction_str"),
        "common_neighbor": sample.get("common_neighbor_level", "Modest"),
        "recent_degree": sample.get("recent_degree_level", "Low"),
        "global_recency": sample.get("global_recency_level", "No prior target interactions"),
        "itemcf": sample.get("itemcf_level", "Low"),
        "usercf": sample.get("usercf_level", "Low"),
    }


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_prediction_debug_fields(sample, llm_score, parse_method=None):
    recency_rank = _safe_int(sample.get("heuristic_recency_rank"))
    popularity_rank = _safe_int(sample.get("heuristic_popularity_rank"))
    past_rank = _safe_int(sample.get("heuristic_past_interactions_rank"))
    ra_rank = _safe_int(sample.get("heuristic_resource_allocation_rank"))
    global_recency_rank = _safe_int(sample.get("heuristic_global_recency_rank"))
    semantic_smoothing_rank = _safe_int(sample.get("heuristic_semantic_smoothing_rank"))
    rrf_rank = _safe_int(sample.get("rrf_rank"))
    rrf_score = _safe_float(sample.get("rrf_score")) if rrf_rank is not None else None

    return {
        "llm_prediction_confidence": float(llm_score),
        "llm_prediction_label": int(llm_score > 0.5),
        "llm_parse_method": parse_method,
        # Pre-LLM numeric fields used by lightweight routing experiments.  Keep
        # these in the debug trace so a small, explicitly requested LLM-labeled
        # calibration set can be retained without serializing the full prompt
        # sample bundle.
        "semantic_mlp_score": _safe_float(sample.get("semantic_mlp_score")),
        "common_neighbor_score": _safe_float(sample.get("common_neighbor_score")),
        "source_popularity_raw": _safe_float(sample.get("source_popularity_raw")),
        "target_popularity_raw": _safe_float(sample.get("target_popularity_raw")),
        "num_past_interactions_raw": _safe_float(sample.get("num_past_interactions_raw")),
        "last_interaction_delta": _safe_float(sample.get("last_interaction_delta")),
        "source_popularity_pct": _safe_float(sample.get("source_popularity_pct")),
        "target_popularity_pct": _safe_float(sample.get("target_popularity_pct")),
        "num_past_interactions_pct": _safe_float(sample.get("num_past_interactions_pct")),
        "common_neighbor_score_pct": _safe_float(sample.get("common_neighbor_score_pct")),
        "last_interaction_recency_pct": _safe_float(
            sample.get("last_interaction_recency_pct")
        ),
        "global_recency_pct": _safe_float(sample.get("global_recency_pct")),
        "itemcf_pct": _safe_float(sample.get("itemcf_pct")),
        "usercf_pct": _safe_float(sample.get("usercf_pct")),
        "rrf_prediction_score": rrf_score,
        "rrf_prediction_rank": rrf_rank,
        "rrf_top1_prediction": (1 if rrf_rank == 1 else 0) if rrf_rank is not None else None,
        "rrf_median_prediction": sample.get("expert_prediction", "Unknown"),
        "overall_structural_signal": sample.get("overall_structural_signal", None),
        "heuristic_recency_score": _safe_float(sample.get("heuristic_recency_score")),
        "heuristic_recency_rank": recency_rank,
        "heuristic_recency_top1_prediction": (1 if recency_rank == 1 else 0) if recency_rank is not None else None,
        "heuristic_popularity_score": _safe_float(sample.get("heuristic_popularity_score")),
        "heuristic_popularity_rank": popularity_rank,
        "heuristic_popularity_top1_prediction": (1 if popularity_rank == 1 else 0) if popularity_rank is not None else None,
        "heuristic_past_interactions_score": _safe_float(sample.get("heuristic_past_interactions_score")),
        "heuristic_past_interactions_rank": past_rank,
        "heuristic_past_interactions_top1_prediction": (1 if past_rank == 1 else 0) if past_rank is not None else None,
        "heuristic_resource_allocation_score": _safe_float(
            sample.get("heuristic_resource_allocation_score")
        ),
        "heuristic_resource_allocation_rank": ra_rank,
        "heuristic_resource_allocation_top1_prediction": (1 if ra_rank == 1 else 0) if ra_rank is not None else None,
        "heuristic_global_recency_score": _safe_float(sample.get("heuristic_global_recency_score")),
        "heuristic_global_recency_rank": global_recency_rank,
        "heuristic_global_recency_top1_prediction": (
            1 if global_recency_rank == 1 else 0
        ) if global_recency_rank is not None else None,
        "heuristic_semantic_smoothing_score": _safe_float(
            sample.get("heuristic_semantic_smoothing_score")
        ),
        "heuristic_semantic_smoothing_rank": semantic_smoothing_rank,
        "heuristic_semantic_smoothing_top1_prediction": (
            1 if semantic_smoothing_rank == 1 else 0
        ) if semantic_smoothing_rank is not None else None,
    }


def write_prediction_debug_log(path, detailed_results, trial_idx, eval_split):
    if not path:
        return 0
    if not detailed_results:
        return 0

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    keep_keys = [
        "sample_id",
        "query_id",
        "source_id",
        "target_id",
        "relation_id",
        "timestamp",
        "source_entity",
        "target_entity",
        "relation",
        "label",
        "prompt",
        "generated_text",
        "prediction_score",
        "llm_prediction_confidence",
        "llm_prediction_label",
        "llm_parse_method",
        "prompt_token_count",
        "binary_logprob_0",
        "binary_logprob_1",
        "binary_logit_margin",
        "binary_token_mass",
        "binary_entropy",
        "semantic_mlp_score",
        "common_neighbor_score",
        "source_popularity_raw",
        "target_popularity_raw",
        "num_past_interactions_raw",
        "last_interaction_delta",
        "source_popularity_pct",
        "target_popularity_pct",
        "num_past_interactions_pct",
        "common_neighbor_score_pct",
        "last_interaction_recency_pct",
        "global_recency_pct",
        "itemcf_pct",
        "usercf_pct",
        "rrf_prediction_score",
        "rrf_prediction_rank",
        "rrf_median_prediction",
        "heuristic_recency_score",
        "heuristic_recency_rank",
        "heuristic_popularity_score",
        "heuristic_popularity_rank",
        "heuristic_past_interactions_score",
        "heuristic_past_interactions_rank",
        "heuristic_resource_allocation_score",
        "heuristic_resource_allocation_rank",
        "heuristic_global_recency_score",
        "heuristic_global_recency_rank",
        "heuristic_semantic_smoothing_score",
        "heuristic_semantic_smoothing_rank",
    ]
    with open(path, "a") as handle:
        for row in detailed_results:
            if not isinstance(row, dict):
                continue
            payload = {
                "trial": int(trial_idx),
                "split": str(eval_split),
            }
            for key in keep_keys:
                payload[key] = row.get(key)
            handle.write(json.dumps(payload) + "\n")
            written += 1
    return written


def _build_prompt_for_sample(
    sample,
    tokenizer,
    entity_map,
    summary_map,
    summary_mode,
    summary_max_chars,
    relation_map,
    history_window,
    include_key_signals,
    include_expert_prediction,
    include_overall_structural_signal,
    overall_structural_signal_name,
    use_raw_key_signals,
    use_percentile_key_signals,
    include_edge_type,
    use_cot,
    mutual_timestamps_only,
    mutual_timestamps_dedup,
    mutual_summary_count_recency,
    common_neighbors_names_only,
    compact_common_neighbors_top_k,
    compact_common_neighbors_novel_only,
    history_table_aliases,
    natural_grouped_history,
    natural_activity_summary,
    natural_neighbor_names_only,
    natural_activity_compact_top3,
    natural_activity_top_k,
    ablate_reasoning_guidance,
    no_cot_output_0_100,
    ablate_mutual_history=False,
    ablate_common_neighbors=False,
    ablate_source_history=False,
    ablate_target_history=False,
    ablate_source_target_history=False,
    prompt_variant="gdelt",
    few_shot_examples=None,
    graph_prompt_special_token=None,
    graph_prompt_num_tokens=0,
    include_edge_type_except_target=False,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
    anonymous_entity_aliases=False,
):
    key_signal_values = _select_key_signal_values(
        sample=sample,
        use_raw_key_signals=use_raw_key_signals,
        use_percentile_key_signals=use_percentile_key_signals,
    )

    expert_prediction_value = sample.get("expert_prediction", "Unknown")
    include_expert_prediction_for_sample = bool(include_expert_prediction)
    if include_expert_prediction_for_sample and include_overall_structural_signal:
        overall_bucket = str(sample.get("overall_structural_signal", "Unknown")).strip().lower()
        if overall_bucket == "high":
            expert_prediction_value = "True"
        elif overall_bucket == "low":
            expert_prediction_value = "False"
        else:
            expert_prediction_value = "Neutral"

    return create_prompt(
        source_entity=sample["source_entity"],
        relation=sample["relation"],
        target_entity=sample["target_entity"],
        source_summary=(summary_map.get(int(sample["source_id"])) if summary_map else None),
        target_summary=(summary_map.get(int(sample["target_id"])) if summary_map else None),
        summary_mode=summary_mode,
        summary_max_chars=summary_max_chars,
        source_id=sample["source_id"],
        relation_id=sample["relation_id"],
        target_id=sample["target_id"],
        prediction_time=sample["timestamp"],
        source_history=sample["source_history"],
        target_history=sample["target_history"],
        source_history_entities=sample.get("source_history_entities", []),
        target_history_entities=sample.get("target_history_entities", []),
        mutual_history=sample.get("mutual_history", []),
        num_past_interactions=key_signal_values["past_interactions"],
        global_avg_interactions=sample.get("global_avg_interactions", 0),
        target_popularity=key_signal_values["target_popularity"],
        source_popularity=key_signal_values["source_popularity"],
        avg_node_popularity=sample.get("avg_node_popularity", 0.0),
        last_interaction_str=key_signal_values["recency"],
        common_neighbor_level=key_signal_values["common_neighbor"],
        recent_degree_signal=key_signal_values["recent_degree"],
        global_recency_signal=key_signal_values["global_recency"],
        itemcf_signal=key_signal_values["itemcf"],
        usercf_signal=key_signal_values["usercf"],
        expert_prediction=expert_prediction_value,
        include_overall_structural_signal=include_overall_structural_signal,
        overall_structural_signal=sample.get("overall_structural_signal", "Unknown"),
        overall_structural_signal_name=overall_structural_signal_name,
        compacted_history_text=sample.get("compacted_history_text", None),
        common_neighbors_desc=sample.get("common_neighbors_desc", "sorted by popularity"),
        common_neighbors=sample.get("common_neighbors", []),
        entity_map=entity_map,
        relation_map=relation_map,
        history_window=history_window,
        source_history_desc=sample.get("source_history_desc", "most recent"),
        target_history_desc=sample.get("target_history_desc", "most recent"),
        include_key_signals=include_key_signals,
        key_signal_fields=key_signal_fields,
        include_expert_prediction=include_expert_prediction_for_sample,
        use_raw_key_signals=use_raw_key_signals,
        use_percentile_key_signals=use_percentile_key_signals,
        include_edge_type=include_edge_type,
        include_edge_type_except_target=include_edge_type_except_target,
        prompt_variant=prompt_variant,
        use_chat_template=True,
        tokenizer=tokenizer,
        use_cot=use_cot,
        mutual_timestamps_only=mutual_timestamps_only,
        mutual_timestamps_dedup=mutual_timestamps_dedup,
        mutual_summary_count_recency=mutual_summary_count_recency,
        common_neighbors_names_only=common_neighbors_names_only,
        compact_common_neighbors_top_k=compact_common_neighbors_top_k,
        compact_common_neighbors_novel_only=compact_common_neighbors_novel_only,
        history_table_aliases=history_table_aliases,
        anonymous_entity_aliases=anonymous_entity_aliases,
        natural_grouped_history=natural_grouped_history,
        natural_activity_summary=natural_activity_summary,
        natural_neighbor_names_only=natural_neighbor_names_only,
        natural_activity_compact_top3=natural_activity_compact_top3,
        natural_activity_top_k=natural_activity_top_k,
        ablate_mutual_history=ablate_mutual_history,
        ablate_common_neighbors=ablate_common_neighbors,
        ablate_source_history=ablate_source_history,
        ablate_target_history=ablate_target_history,
        ablate_source_target_history=ablate_source_target_history,
        ablate_reasoning_guidance=ablate_reasoning_guidance,
        no_cot_output_0_100=no_cot_output_0_100,
        few_shot_examples=few_shot_examples,
        graph_prompt_special_token=graph_prompt_special_token,
        graph_prompt_num_tokens=graph_prompt_num_tokens,
    )


def _build_forced_binary_scoring_prompt(prompt):
    answer_prefix = "The answer is: "
    stripped = prompt.rstrip()
    if stripped.endswith(answer_prefix.rstrip()):
        return stripped + " "
    return prompt + answer_prefix


def build_link_prediction_prompts(
    samples,
    tokenizer,
    entity_map,
    relation_map,
    *,
    summary_map=None,
    summary_mode="off",
    summary_max_chars=120,
    history_window=10,
    include_key_signals=True,
    include_expert_prediction=True,
    include_overall_structural_signal=False,
    overall_structural_signal_name="Overall structural signal",
    use_raw_key_signals=False,
    use_percentile_key_signals=False,
    include_edge_type=False,
    include_edge_type_except_target=False,
    use_cot=False,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    mutual_summary_count_recency=False,
    common_neighbors_names_only=False,
    compact_common_neighbors_top_k=0,
    compact_common_neighbors_novel_only=False,
    history_table_aliases=False,
    anonymous_entity_aliases=False,
    natural_grouped_history=False,
    natural_activity_summary=False,
    natural_neighbor_names_only=False,
    natural_activity_compact_top3=False,
    natural_activity_top_k=3,
    ablate_mutual_history=False,
    ablate_common_neighbors=False,
    ablate_source_history=False,
    ablate_target_history=False,
    ablate_source_target_history=False,
    ablate_reasoning_guidance=False,
    no_cot_output_0_100=False,
    prompt_variant="gdelt",
    few_shot_examples=None,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
    force_binary_answer_prefix=False,
):
    """Render the canonical prompt text without running model inference."""
    prompts = []
    for sample in tqdm(samples, desc="[LLM] Formatting"):
        prompt = _build_prompt_for_sample(
            sample=sample,
            tokenizer=tokenizer,
            entity_map=entity_map,
            summary_map=summary_map,
            summary_mode=summary_mode,
            summary_max_chars=summary_max_chars,
            relation_map=relation_map,
            history_window=history_window,
            include_key_signals=include_key_signals,
            include_expert_prediction=include_expert_prediction,
            include_overall_structural_signal=include_overall_structural_signal,
            overall_structural_signal_name=overall_structural_signal_name,
            use_raw_key_signals=use_raw_key_signals,
            use_percentile_key_signals=use_percentile_key_signals,
            include_edge_type=include_edge_type,
            include_edge_type_except_target=include_edge_type_except_target,
            use_cot=use_cot,
            mutual_timestamps_only=mutual_timestamps_only,
            mutual_timestamps_dedup=mutual_timestamps_dedup,
            mutual_summary_count_recency=mutual_summary_count_recency,
            common_neighbors_names_only=common_neighbors_names_only,
            compact_common_neighbors_top_k=compact_common_neighbors_top_k,
            compact_common_neighbors_novel_only=compact_common_neighbors_novel_only,
            history_table_aliases=history_table_aliases,
            anonymous_entity_aliases=anonymous_entity_aliases,
            natural_grouped_history=natural_grouped_history,
            natural_activity_summary=natural_activity_summary,
            natural_neighbor_names_only=natural_neighbor_names_only,
            natural_activity_compact_top3=natural_activity_compact_top3,
            natural_activity_top_k=natural_activity_top_k,
            ablate_mutual_history=ablate_mutual_history,
            ablate_common_neighbors=ablate_common_neighbors,
            ablate_source_history=ablate_source_history,
            ablate_target_history=ablate_target_history,
            ablate_source_target_history=ablate_source_target_history,
            ablate_reasoning_guidance=ablate_reasoning_guidance,
            no_cot_output_0_100=no_cot_output_0_100,
            prompt_variant=prompt_variant,
            few_shot_examples=few_shot_examples,
            key_signal_fields=key_signal_fields,
        )
        if force_binary_answer_prefix:
            prompt = _build_forced_binary_scoring_prompt(prompt)
        prompts.append(prompt)
    return prompts


def export_link_prediction_prompt_dataset(
    output_dir,
    samples,
    prompts,
    *,
    eval_split,
):
    """Persist aligned prompt text and link identities as streaming JSONL."""
    import hashlib
    import json

    if len(samples) != len(prompts):
        raise ValueError(
            f"Prompt export alignment mismatch: samples={len(samples)}, prompts={len(prompts)}"
        )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(output_dir, f"prompts_{eval_split}.jsonl"))
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row_index, (sample, prompt) in enumerate(zip(samples, prompts)):
            row = {
                "row_index": int(row_index),
                "query_id": int(sample.get("query_id", row_index)),
                "source_id": int(sample.get("source_id", -1)),
                "target_id": int(sample.get("target_id", -1)),
                "relation_id": int(sample.get("relation_id", -1)),
                "timestamp": float(sample.get("timestamp", -1.0)),
                "dtgb_timestamp": float(
                    sample.get("dtgb_timestamp", sample.get("timestamp", -1.0))
                ),
                "label": int(sample["label"]),
                "eval_split": str(eval_split),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    return path


def _estimate_text_token_count(tokenizer, text):
    encoded = tokenizer.encode(str(text), add_special_tokens=False)
    return int(len(encoded))


def _extract_binary_text_score(text):
    stripped = (text or "").strip()
    if not stripped:
        return None, None
    lowered = stripped.lower()
    if "answer is: 1" in lowered or stripped == "1" or stripped.startswith("1"):
        return 1.0, "text_answer_phrase"
    if "answer is: 0" in lowered or stripped == "0" or stripped.startswith("0"):
        return 0.0, "text_answer_phrase"
    if stripped.endswith("1"):
        return 1.0, "text_suffix"
    if stripped.endswith("0"):
        return 0.0, "text_suffix"
    return None, None


def _score_no_cot_binary_forced(llm, tokenizer, prompts, stop_token_ids):
    scoring_prompts = [_build_forced_binary_scoring_prompt(prompt) for prompt in prompts]
    prompt_token_counts = [
        _estimate_text_token_count(tokenizer, prompt) for prompt in scoring_prompts
    ]
    prompt_tokens = sum(prompt_token_counts)

    scoring_params = SamplingParams(
        max_tokens=1,
        logprobs=20,
        stop_token_ids=stop_token_ids or None,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    _llm_log(f"Running deterministic forced-binary scoring on {len(scoring_prompts)} prompts...")
    outputs = llm.generate(
        scoring_prompts,
        scoring_params,
        tokenization_kwargs={"add_special_tokens": False},
    )

    scoring_results = []
    completion_tokens = 0
    for output_idx, output in enumerate(outputs):
        score = None
        parse_method = None
        binary_features = None
        generated_text = output.outputs[0].text if output.outputs else ""
        output_token_ids = getattr(output.outputs[0], "token_ids", None) if output.outputs else None
        if output_token_ids is not None:
            completion_tokens += int(len(output_token_ids))
        elif generated_text:
            completion_tokens += _estimate_text_token_count(tokenizer, generated_text)
        if output.outputs:
            seq_logprobs = output.outputs[0].logprobs or []
            if seq_logprobs:
                binary_features = compute_binary_logprob_features(
                    seq_logprobs[0], tokenizer
                )
                if binary_features is not None:
                    score = binary_features["score"]
                    parse_method = "nocot_forced_binary_logprob"
        if score is None:
            text_score, text_method = _extract_binary_text_score(generated_text)
            if text_score is not None:
                score = text_score
                parse_method = f"nocot_forced_binary_{text_method}"
        if score is None:
            score = 0.5
            parse_method = "nocot_forced_binary_default_0_5"
        result = {
            "score": float(score),
            "parse_method": parse_method,
            "generated_text": generated_text,
            "prompt_token_count": int(prompt_token_counts[output_idx]),
            "binary_logprob_0": None,
            "binary_logprob_1": None,
            "binary_logit_margin": None,
            "binary_token_mass": None,
            "binary_entropy": None,
        }
        if binary_features is not None:
            result.update(binary_features)
        scoring_results.append(result)
    return scoring_results, {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
    }, outputs, scoring_prompts


def _build_detail_row(sample, sample_id, prompt, score, label, generated_text, parse_method):
    detail = {
        "sample_id": sample_id,
        "query_id": int(sample.get("query_id", sample_id)),
        "source_id": int(sample.get("source_id", -1)),
        "target_id": int(sample.get("target_id", -1)),
        "relation_id": int(sample.get("relation_id", -1)),
        "timestamp": int(sample.get("timestamp", -1)),
        "source_entity": sample.get("source_entity"),
        "target_entity": sample.get("target_entity"),
        "relation": sample.get("relation"),
        "prompt": prompt,
        "prediction_score": float(score),
        "label": int(label),
        "generated_text": generated_text,
    }
    detail.update(_build_prediction_debug_fields(sample, score, parse_method=parse_method))
    return detail


def _finalize_prediction_results(
    predictions,
    labels,
    parse_stats,
    detailed_results,
    use_cot,
    no_cot_output_0_100,
    dtgb_eval_batch_size=None,
):
    metrics = compute_prediction_metrics(
        predictions,
        labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )

    if use_cot:
        abnormal = (
            parse_stats.get("cot_possible_clip_by_max_tokens", 0)
            + parse_stats.get("cot_clipped_by_max_tokens", 0)
            + parse_stats.get("cot_answer_phrase_fallback", 0)
            + parse_stats.get("cot_last_integer_fallback", 0)
            + parse_stats.get("cot_parse_failed_default_0_5", 0)
        )
    else:
        if no_cot_output_0_100:
            abnormal = (
                parse_stats.get("nocot_score_0_100_answer_phrase_fallback", 0)
                + parse_stats.get("nocot_score_0_100_last_integer_fallback", 0)
                + parse_stats.get("nocot_score_0_100_parse_failed_default_0_5", 0)
            )
        else:
            abnormal = (
                parse_stats.get("nocot_text_suffix", 0)
                + parse_stats.get("nocot_parse_failed_default_0", 0)
                + parse_stats.get("nocot_parse_failed_default_0_5", 0)
                + parse_stats.get("nocot_forced_binary_text_suffix", 0)
                + parse_stats.get("nocot_forced_binary_default_0_5", 0)
            )

    if abnormal > 0:
        _llm_warn(f"Abnormal parsing count: {abnormal}/{len(labels)}")
    _llm_log(f"Parse stats: {parse_stats}")

    metrics["parse_stats"] = parse_stats
    metrics["detailed_results"] = detailed_results
    return metrics


def evaluate_transformers(
    model,
    tokenizer,
    samples,
    entity_map,
    relation_map,
    summary_map=None,
    summary_mode="off",
    summary_max_chars=120,
    history_window=10,
    save_details=True,
    use_cot=True,
    include_key_signals=True,
    include_expert_prediction=True,
    include_overall_structural_signal=False,
    overall_structural_signal_name="Overall structural signal",
    use_raw_key_signals=False,
    use_percentile_key_signals=False,
    include_edge_type=False,
    include_edge_type_except_target=False,
    cot_max_tokens=2048,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    mutual_summary_count_recency=False,
    common_neighbors_names_only=False,
    compact_common_neighbors_top_k=0,
    compact_common_neighbors_novel_only=False,
    history_table_aliases=False,
    anonymous_entity_aliases=False,
    natural_grouped_history=False,
    natural_activity_summary=False,
    natural_neighbor_names_only=False,
    natural_activity_compact_top3=False,
    natural_activity_top_k=3,
    ablate_mutual_history=False,
    ablate_common_neighbors=False,
    ablate_source_history=False,
    ablate_target_history=False,
    ablate_source_target_history=False,
    ablate_reasoning_guidance=False,
    no_cot_output_0_100=False,
    no_cot_binary_score_mode="sample_logprob",
    eval_split="transductive",
    dtgb_eval_batch_size=None,
    few_shot_examples=None,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
):
    """Evaluate using standard transformers generate."""
    _llm_log(f"Starting evaluation ({eval_split.capitalize()}, Transformers Fallback) [CoT={use_cot}]")

    predictions = []
    labels = []
    detailed_results = []
    parse_stats = {
        "mode": "cot" if use_cot else "no_cot",
        "cot_possible_clip_by_max_tokens": 0,
        "cot_exact_phrase": 0,
        "cot_answer_phrase_fallback": 0,
        "cot_last_integer_fallback": 0,
        "cot_parse_failed_default_0_5": 0,
        "nocot_text_answer_phrase": 0,
        "nocot_text_suffix": 0,
        "nocot_parse_failed_default_0_5": 0,
        "nocot_score_0_100_exact_phrase": 0,
        "nocot_score_0_100_answer_phrase_fallback": 0,
        "nocot_score_0_100_last_integer_fallback": 0,
        "nocot_score_0_100_parse_failed_default_0_5": 0,
    }
    if (not use_cot) and (not no_cot_output_0_100) and no_cot_binary_score_mode != "sample_logprob":
        _llm_warn(
            "--no_cot_binary_score_mode is only implemented for the vLLM path; "
            "Transformers fallback will keep text parsing."
        )

    for idx, sample in enumerate(tqdm(samples, desc="[LLM] Evaluating")):
        prompt = _build_prompt_for_sample(
            sample=sample,
            tokenizer=tokenizer,
            entity_map=entity_map,
            summary_map=summary_map,
            summary_mode=summary_mode,
            summary_max_chars=summary_max_chars,
            relation_map=relation_map,
            history_window=history_window,
            include_key_signals=include_key_signals,
            include_expert_prediction=include_expert_prediction,
            include_overall_structural_signal=include_overall_structural_signal,
            overall_structural_signal_name=overall_structural_signal_name,
            use_raw_key_signals=use_raw_key_signals,
            use_percentile_key_signals=use_percentile_key_signals,
            include_edge_type=include_edge_type,
            include_edge_type_except_target=include_edge_type_except_target,
            use_cot=use_cot,
            mutual_timestamps_only=mutual_timestamps_only,
            mutual_timestamps_dedup=mutual_timestamps_dedup,
            mutual_summary_count_recency=mutual_summary_count_recency,
            common_neighbors_names_only=common_neighbors_names_only,
            compact_common_neighbors_top_k=compact_common_neighbors_top_k,
            compact_common_neighbors_novel_only=compact_common_neighbors_novel_only,
            history_table_aliases=history_table_aliases,
            anonymous_entity_aliases=anonymous_entity_aliases,
            natural_grouped_history=natural_grouped_history,
            natural_activity_summary=natural_activity_summary,
            natural_neighbor_names_only=natural_neighbor_names_only,
            natural_activity_compact_top3=natural_activity_compact_top3,
            natural_activity_top_k=natural_activity_top_k,
            ablate_mutual_history=ablate_mutual_history,
            ablate_common_neighbors=ablate_common_neighbors,
            ablate_source_history=ablate_source_history,
            ablate_target_history=ablate_target_history,
            ablate_source_target_history=ablate_source_target_history,
            ablate_reasoning_guidance=ablate_reasoning_guidance,
            no_cot_output_0_100=no_cot_output_0_100,
            few_shot_examples=few_shot_examples,
            key_signal_fields=key_signal_fields,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cot_max_tokens if use_cot else 32,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        if use_cot and generated_ids.shape[0] >= cot_max_tokens:
            parse_stats["cot_possible_clip_by_max_tokens"] += 1
            _llm_warn(f"Sample {idx} - Possible CoT clipping at max_new_tokens={cot_max_tokens}")

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        score = 0.5
        parse_method_used = None
        if use_cot:
            val, parse_method = extract_score_0_100_with_meta(generated_text)
            parse_method_used = f"cot_{parse_method}"
            if val is None:
                parse_stats["cot_parse_failed_default_0_5"] += 1
                _llm_warn(f"Sample {idx} - Could not parse 0-100 score. Defaulting to 0.5")
            else:
                score = val / 100.0
                if parse_method == "exact_phrase":
                    parse_stats["cot_exact_phrase"] += 1
                elif parse_method == "answer_phrase_fallback":
                    parse_stats["cot_answer_phrase_fallback"] += 1
                    _llm_warn(f"Sample {idx} - CoT parsed via fallback phrase matcher.")
                elif parse_method == "last_integer_fallback":
                    parse_stats["cot_last_integer_fallback"] += 1
                    _llm_warn(f"Sample {idx} - CoT parsed via last-integer fallback.")
        else:
            if no_cot_output_0_100:
                val, parse_method = extract_score_0_100_with_meta(generated_text)
                parse_method_used = f"nocot_0_100_{parse_method}"
                if val is None:
                    score = 0.5
                    parse_stats["nocot_score_0_100_parse_failed_default_0_5"] += 1
                    _llm_warn(f"Sample {idx} - No-CoT(0-100) parse failed. Defaulting to 0.5")
                else:
                    score = val / 100.0
                    if parse_method == "exact_phrase":
                        parse_stats["nocot_score_0_100_exact_phrase"] += 1
                    elif parse_method == "answer_phrase_fallback":
                        parse_stats["nocot_score_0_100_answer_phrase_fallback"] += 1
                        _llm_warn(f"Sample {idx} - No-CoT(0-100) parsed via fallback phrase matcher.")
                    elif parse_method == "last_integer_fallback":
                        parse_stats["nocot_score_0_100_last_integer_fallback"] += 1
                        _llm_warn(f"Sample {idx} - No-CoT(0-100) parsed via last-integer fallback.")
            else:
                text_lower = generated_text.lower()
                if "answer is: 1" in text_lower or generated_text.strip() == "1":
                    score = 1.0
                    parse_method_used = "nocot_text_answer_phrase"
                    parse_stats["nocot_text_answer_phrase"] += 1
                elif "answer is: 0" in text_lower or generated_text.strip() == "0":
                    score = 0.0
                    parse_method_used = "nocot_text_answer_phrase"
                    parse_stats["nocot_text_answer_phrase"] += 1
                elif generated_text.strip().endswith("1"):
                    score = 1.0
                    parse_method_used = "nocot_text_suffix"
                    parse_stats["nocot_text_suffix"] += 1
                    _llm_warn(f"Sample {idx} - No-CoT parsed via suffix fallback.")
                elif generated_text.strip().endswith("0"):
                    score = 0.0
                    parse_method_used = "nocot_text_suffix"
                    parse_stats["nocot_text_suffix"] += 1
                    _llm_warn(f"Sample {idx} - No-CoT parsed via suffix fallback.")
                else:
                    parse_method_used = "nocot_parse_failed_default_0_5"
                    parse_stats["nocot_parse_failed_default_0_5"] += 1
                    _llm_warn(f"Sample {idx} - No-CoT parse failed. Defaulting to 0.5")

        predictions.append(score)
        labels.append(sample["label"])

        if save_details:
            detailed_results.append(
                _build_detail_row(
                    sample=sample,
                    sample_id=idx,
                    prompt=prompt,
                    score=score,
                    label=sample["label"],
                    generated_text=generated_text,
                    parse_method=parse_method_used,
                )
            )

    return _finalize_prediction_results(
        predictions=predictions,
        labels=labels,
        parse_stats=parse_stats,
        detailed_results=detailed_results if save_details else [],
        use_cot=use_cot,
        no_cot_output_0_100=no_cot_output_0_100,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )


def evaluate(
    llm,
    tokenizer,
    samples,
    entity_map,
    relation_map,
    summary_map=None,
    summary_mode="off",
    summary_max_chars=120,
    history_window=10,
    save_details=True,
    use_cot=True,
    include_key_signals=True,
    include_expert_prediction=True,
    include_overall_structural_signal=False,
    overall_structural_signal_name="Overall structural signal",
    use_raw_key_signals=False,
    use_percentile_key_signals=False,
    include_edge_type=False,
    include_edge_type_except_target=False,
    cot_max_tokens=2048,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    mutual_summary_count_recency=False,
    common_neighbors_names_only=False,
    compact_common_neighbors_top_k=0,
    compact_common_neighbors_novel_only=False,
    history_table_aliases=False,
    anonymous_entity_aliases=False,
    natural_grouped_history=False,
    natural_activity_summary=False,
    natural_neighbor_names_only=False,
    natural_activity_compact_top3=False,
    natural_activity_top_k=3,
    ablate_mutual_history=False,
    ablate_common_neighbors=False,
    ablate_source_history=False,
    ablate_target_history=False,
    ablate_source_target_history=False,
    ablate_reasoning_guidance=False,
    no_cot_output_0_100=False,
    no_cot_binary_score_mode="sample_logprob",
    prompt_variant="gdelt",
    eval_split="transductive",
    dtgb_eval_batch_size=None,
    few_shot_examples=None,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
    capture_prompt_embeddings=False,
    prompt_embedding_output_dir=None,
    prompt_embedding_normalize=False,
    prompt_embedding_save_dtype="float16",
):
    """
    Evaluate link prediction using vLLM batching.
    """
    if SamplingParams is None:
        raise ImportError(
            "vLLM is not installed. Install a recent vLLM build or rerun with --use_transformers."
        )
    _llm_log(f"Starting evaluation ({eval_split.capitalize()} Setting) [CoT={use_cot}]")

    _llm_log("Generating prompts...")
    prompts = build_link_prediction_prompts(
        samples,
        tokenizer,
        entity_map,
        relation_map,
        summary_map=summary_map,
        summary_mode=summary_mode,
        summary_max_chars=summary_max_chars,
        history_window=history_window,
        include_key_signals=include_key_signals,
        include_expert_prediction=include_expert_prediction,
        include_overall_structural_signal=include_overall_structural_signal,
        overall_structural_signal_name=overall_structural_signal_name,
        use_raw_key_signals=use_raw_key_signals,
        use_percentile_key_signals=use_percentile_key_signals,
        include_edge_type=include_edge_type,
        include_edge_type_except_target=include_edge_type_except_target,
        use_cot=use_cot,
        mutual_timestamps_only=mutual_timestamps_only,
        mutual_timestamps_dedup=mutual_timestamps_dedup,
        mutual_summary_count_recency=mutual_summary_count_recency,
        common_neighbors_names_only=common_neighbors_names_only,
        compact_common_neighbors_top_k=compact_common_neighbors_top_k,
        compact_common_neighbors_novel_only=compact_common_neighbors_novel_only,
        history_table_aliases=history_table_aliases,
        anonymous_entity_aliases=anonymous_entity_aliases,
        natural_grouped_history=natural_grouped_history,
        natural_activity_summary=natural_activity_summary,
        natural_neighbor_names_only=natural_neighbor_names_only,
        natural_activity_compact_top3=natural_activity_compact_top3,
        natural_activity_top_k=natural_activity_top_k,
        ablate_mutual_history=ablate_mutual_history,
        ablate_common_neighbors=ablate_common_neighbors,
        ablate_source_history=ablate_source_history,
        ablate_target_history=ablate_target_history,
        ablate_source_target_history=ablate_source_target_history,
        ablate_reasoning_guidance=ablate_reasoning_guidance,
        no_cot_output_0_100=no_cot_output_0_100,
        prompt_variant=prompt_variant,
        few_shot_examples=few_shot_examples,
        key_signal_fields=key_signal_fields,
    )

    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen" in tokenizer_name:
        stop_token_ids = _resolve_stop_token_ids(tokenizer)
        stop_token_source = "tokenizer"
    else:
        stop_token_ids = [128001, 128008, 128009]
        stop_token_source = "llama-default"
    if stop_token_ids:
        _llm_log(f"Using stop token IDs ({stop_token_source}): {stop_token_ids}")

    is_qwen35 = any(tag in tokenizer_name for tag in ("qwen3.5", "qwen3_5"))
    sampling_kwargs = {
        "max_tokens": cot_max_tokens if use_cot else 32,
        "logprobs": 0 if use_cot else 5,
        "stop_token_ids": stop_token_ids or None,
    }
    if is_qwen35 and not use_cot:
        _llm_log("Using Qwen 3.5 no-CoT sampling params")
        sampling_kwargs.update(
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
            }
        )
    else:
        sampling_kwargs.update(
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.1,
            }
        )
    use_forced_binary_only = (
        (not use_cot)
        and (not no_cot_output_0_100)
        and no_cot_binary_score_mode == "forced_binary"
    )

    outputs = []
    forced_binary_results = None
    token_usage = {}
    embedding_outputs = []
    embedding_prompts = prompts
    if use_forced_binary_only:
        (
            forced_binary_results,
            token_usage,
            embedding_outputs,
            embedding_prompts,
        ) = _score_no_cot_binary_forced(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            stop_token_ids=stop_token_ids,
        )
    else:
        prompt_tokens = 0
        for prompt in prompts:
            prompt_tokens += _estimate_text_token_count(tokenizer, prompt)
        sampling_params = SamplingParams(**sampling_kwargs)
        _llm_log(f"Running inference on {len(prompts)} prompts...")
        outputs = llm.generate(
            prompts,
            sampling_params,
            tokenization_kwargs={"add_special_tokens": False},
        )
        embedding_outputs = outputs
        completion_tokens = 0
        for output in outputs:
            if not output.outputs:
                continue
            output_token_ids = getattr(output.outputs[0], "token_ids", None)
            if output_token_ids is not None:
                completion_tokens += int(len(output_token_ids))
                continue
            generated_text = output.outputs[0].text
            if generated_text:
                completion_tokens += _estimate_text_token_count(tokenizer, generated_text)
        token_usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        }

    prompt_embedding_manifest = None
    engine_capture_enabled = bool(
        getattr(llm, "_dtgb_capture_prompt_embeddings", False)
    )
    if capture_prompt_embeddings and not engine_capture_enabled:
        discard_vllm_hidden_state_files(embedding_outputs)
        raise RuntimeError(
            "Prompt embedding capture was requested for evaluate(), but the vLLM engine "
            "was not initialized with capture_prompt_embeddings=True."
        )
    if engine_capture_enabled and capture_prompt_embeddings:
        if not prompt_embedding_output_dir:
            discard_vllm_hidden_state_files(embedding_outputs)
            raise ValueError("prompt_embedding_output_dir is required for embedding capture.")
        capture_start = time.perf_counter()
        prompt_embedding_matrix = extract_prompt_embeddings(
            embedding_outputs,
            normalize=bool(prompt_embedding_normalize),
            delete_sources=True,
        )
        prompt_embedding_manifest = save_prompt_embedding_shard(
            prompt_embedding_output_dir,
            prompt_embedding_matrix,
            samples,
            embedding_prompts,
            eval_split=eval_split,
            layer_id=getattr(llm, "_dtgb_prompt_embedding_layer_id", -1),
            normalize=bool(prompt_embedding_normalize),
            save_dtype=prompt_embedding_save_dtype,
            representation=getattr(
                llm,
                "_dtgb_prompt_embedding_representation",
                "unknown_prompt_representation",
            ),
            binary_scores=(
                [result["score"] for result in forced_binary_results]
                if forced_binary_results is not None
                else None
            ),
            model_name_or_path=getattr(tokenizer, "name_or_path", None),
        )
        prompt_embedding_manifest["capture_and_save_seconds"] = float(
            time.perf_counter() - capture_start
        )
        _llm_log(
            "Saved prompt embeddings: "
            f"{prompt_embedding_manifest['path']} "
            f"({prompt_embedding_manifest['count']} x "
            f"{prompt_embedding_manifest['dimension']})"
        )
    elif engine_capture_enabled:
        # Calibration and auxiliary generation calls use the same engine but do
        # not need persistent vectors.
        discard_vllm_hidden_state_files(embedding_outputs)

    predictions = []
    labels = []
    detailed_results = []
    parse_stats = {
        "mode": "cot" if use_cot else "no_cot",
        "cot_clipped_by_max_tokens": 0,
        "cot_exact_phrase": 0,
        "cot_answer_phrase_fallback": 0,
        "cot_last_integer_fallback": 0,
        "cot_parse_failed_default_0_5": 0,
        "nocot_logprob": 0,
        "nocot_text_answer_phrase": 0,
        "nocot_text_suffix": 0,
        "nocot_parse_failed_default_0": 0,
        "nocot_forced_binary_logprob": 0,
        "nocot_forced_binary_text_answer_phrase": 0,
        "nocot_forced_binary_text_suffix": 0,
        "nocot_forced_binary_default_0_5": 0,
        "nocot_score_0_100_exact_phrase": 0,
        "nocot_score_0_100_answer_phrase_fallback": 0,
        "nocot_score_0_100_last_integer_fallback": 0,
        "nocot_score_0_100_parse_failed_default_0_5": 0,
    }
    if forced_binary_results is not None:
        generated_texts = [result.get("generated_text", "") for result in forced_binary_results]
        eval_items = enumerate(samples)
    else:
        generated_texts = [output.outputs[0].text if output.outputs else "" for output in outputs]
        eval_items = enumerate(outputs)

    for idx, item in eval_items:
        score = 0.5
        found_answer = False
        parse_method_used = None
        generated_text = generated_texts[idx]
        sample = samples[idx]
        output = None if forced_binary_results is not None else item
        finish_reason = None
        if output is not None and output.outputs:
            finish_reason = getattr(output.outputs[0], "finish_reason", None)
        if use_cot and finish_reason == "length":
            parse_stats["cot_clipped_by_max_tokens"] += 1
            _llm_warn(f"Sample {idx} - CoT clipped by max_tokens={cot_max_tokens}")

        if use_cot:
            val, parse_method = extract_score_0_100_with_meta(generated_text)
            parse_method_used = f"cot_{parse_method}"
            if val is None:
                parse_stats["cot_parse_failed_default_0_5"] += 1
                _llm_warn(f"Sample {idx} - Could not parse 0-100 score. Defaulting to 0.5")
                score = 0.5
            else:
                score = val / 100.0
                if parse_method == "exact_phrase":
                    parse_stats["cot_exact_phrase"] += 1
                elif parse_method == "answer_phrase_fallback":
                    parse_stats["cot_answer_phrase_fallback"] += 1
                    _llm_warn(f"Sample {idx} - CoT parsed via fallback phrase matcher.")
                elif parse_method == "last_integer_fallback":
                    parse_stats["cot_last_integer_fallback"] += 1
                    _llm_warn(f"Sample {idx} - CoT parsed via last-integer fallback.")
        else:
            if no_cot_output_0_100:
                val, parse_method = extract_score_0_100_with_meta(generated_text)
                parse_method_used = f"nocot_0_100_{parse_method}"
                if val is None:
                    parse_stats["nocot_score_0_100_parse_failed_default_0_5"] += 1
                    _llm_warn(f"Sample {idx} - No-CoT(0-100) parse failed. Defaulting to 0.5")
                    score = 0.5
                else:
                    score = val / 100.0
                    if parse_method == "exact_phrase":
                        parse_stats["nocot_score_0_100_exact_phrase"] += 1
                    elif parse_method == "answer_phrase_fallback":
                        parse_stats["nocot_score_0_100_answer_phrase_fallback"] += 1
                        _llm_warn(f"Sample {idx} - No-CoT(0-100) parsed via fallback phrase matcher.")
                    elif parse_method == "last_integer_fallback":
                        parse_stats["nocot_score_0_100_last_integer_fallback"] += 1
                        _llm_warn(f"Sample {idx} - No-CoT(0-100) parsed via last-integer fallback.")
            else:
                if forced_binary_results is not None:
                    forced_result = forced_binary_results[idx]
                    score = forced_result["score"]
                    parse_method_used = forced_result["parse_method"]
                    parse_stats[parse_method_used] += 1
                    found_answer = True
                elif output.outputs and output.outputs[0].logprobs:
                    seq_logprobs = output.outputs[0].logprobs
                    output_token_ids = getattr(output.outputs[0], "token_ids", None) or []
                    stop_id_set = set(stop_token_ids or [])

                    candidate_positions = []
                    if output_token_ids:
                        max_positions = min(len(seq_logprobs), len(output_token_ids))
                        for i in range(max_positions - 1, -1, -1):
                            chosen_token_id = output_token_ids[i]
                            if chosen_token_id in stop_id_set:
                                continue
                            chosen_token_text = tokenizer.decode([chosen_token_id]).strip()
                            if chosen_token_text in ("0", "1"):
                                candidate_positions.append(i)
                        for i in range(max_positions - 1, -1, -1):
                            chosen_token_id = output_token_ids[i]
                            if chosen_token_id in stop_id_set or i in candidate_positions:
                                continue
                            candidate_positions.append(i)
                    else:
                        candidate_positions = list(range(len(seq_logprobs) - 1, -1, -1))

                    for i in candidate_positions:
                        token_logprobs = seq_logprobs[i]
                        score = compute_binary_score_from_logprobs(token_logprobs, tokenizer)
                        if score is not None:
                            found_answer = True
                            parse_method_used = "nocot_logprob"
                            parse_stats["nocot_logprob"] += 1
                            break

                if not found_answer:
                    text_lower = generated_text.lower()
                    if "answer is: 1" in text_lower or generated_text.strip() == "1":
                        score = 1.0
                        parse_method_used = "nocot_text_answer_phrase"
                        parse_stats["nocot_text_answer_phrase"] += 1
                    elif "answer is: 0" in text_lower or generated_text.strip() == "0":
                        score = 0.0
                        parse_method_used = "nocot_text_answer_phrase"
                        parse_stats["nocot_text_answer_phrase"] += 1
                    elif generated_text.strip().endswith("1"):
                        score = 1.0
                        parse_method_used = "nocot_text_suffix"
                        parse_stats["nocot_text_suffix"] += 1
                        _llm_warn(f"Sample {idx} - No-CoT parsed via suffix fallback.")
                    elif generated_text.strip().endswith("0"):
                        score = 0.0
                        parse_method_used = "nocot_text_suffix"
                        parse_stats["nocot_text_suffix"] += 1
                        _llm_warn(f"Sample {idx} - No-CoT parsed via suffix fallback.")
                    else:
                        parse_stats["nocot_parse_failed_default_0"] += 1
                        parse_method_used = "nocot_parse_failed_default_0"
                        _llm_warn(f"Sample {idx} - No-CoT parse failed. Defaulting to 0.0")
                        score = 0.0

        predictions.append(score)
        labels.append(sample["label"])
        if save_details:
            detail = _build_detail_row(
                sample=sample,
                sample_id=idx,
                prompt=prompts[idx],
                score=score,
                label=sample["label"],
                generated_text=generated_text,
                parse_method=parse_method_used,
            )
            if prompt_embedding_manifest is not None:
                detail["prompt_embedding_shard"] = prompt_embedding_manifest["path"]
                detail["prompt_embedding_row"] = int(idx)
            if forced_binary_results is not None:
                forced_result = forced_binary_results[idx]
                for key in (
                    "prompt_token_count",
                    "binary_logprob_0",
                    "binary_logprob_1",
                    "binary_logit_margin",
                    "binary_token_mass",
                    "binary_entropy",
                ):
                    detail[key] = forced_result.get(key)
            detailed_results.append(detail)

    metrics = _finalize_prediction_results(
        predictions=predictions,
        labels=labels,
        parse_stats=parse_stats,
        detailed_results=detailed_results if save_details else [],
        use_cot=use_cot,
        no_cot_output_0_100=no_cot_output_0_100,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    if token_usage:
        metrics["token_usage"] = token_usage
    if prompt_embedding_manifest is not None:
        metrics["prompt_embedding_shards"] = [prompt_embedding_manifest]
    return metrics


def evaluate_rrf_only(
    samples,
    negative_ratio=1,
    save_details=True,
    dtgb_eval_batch_size=None,
):
    """
    Standalone RRF evaluation path (no LLM inference).
    Uses per-sample rrf_score as prediction confidence.
    """
    _rrf_log("Starting evaluation (RRF-Only, No LLM)")

    predictions = np.array([float(sample.get("rrf_score", 0.0)) for sample in samples], dtype=np.float64)
    labels = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)

    metrics = compute_prediction_metrics(
        predictions,
        labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    detailed_results = []
    if save_details:
        for idx, sample in enumerate(samples):
            score = float(predictions[idx])
            detailed_results.append(
                _build_detail_row(
                    sample=sample,
                    sample_id=idx,
                    prompt=None,
                    score=score,
                    label=sample["label"],
                    generated_text=None,
                    parse_method="rrf_only",
                )
            )

    metrics["parse_stats"] = {"mode": "rrf_only"}
    metrics["detailed_results"] = detailed_results if save_details else []
    return metrics


__all__ = [
    "evaluate",
    "evaluate_rrf_only",
    "evaluate_transformers",
    "load_model_and_tokenizer",
    "load_model_transformers",
    "maybe_spawn_local_dp_workers",
    "sanitize_torch_distributed_env_for_vllm",
    "write_prediction_debug_log",
]
