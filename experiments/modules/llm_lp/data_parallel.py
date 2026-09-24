"""
Helpers for simple data-parallel evaluation orchestration.

Each rank evaluates a disjoint subset of samples and writes one shard JSON per trial.
Rank 0 waits for all shard files, merges predictions back to global sample order,
and computes final metrics.
"""
import json
import os
import pickle
import re
import shutil
import time

from experiments.modules.prediction_metrics import compute_prediction_metrics


def _sanitize_path_token(value):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._")
    return token or "output"


def resolve_data_parallel_sync_dir(output_path, sync_dir=None):
    override = sync_dir or os.environ.get("DATA_PARALLEL_SYNC_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(str(override)))

    output_abs = os.path.abspath(str(output_path))
    output_dir = os.path.dirname(output_abs) or os.getcwd()
    output_name = _sanitize_path_token(os.path.basename(output_abs))
    return os.path.join(output_dir, ".dp_sync", output_name)


def data_parallel_run_dir(sync_dir, run_id):
    safe_run_id = str(run_id).replace(os.sep, "_")
    return os.path.join(sync_dir, f"run_{safe_run_id}")


def cleanup_data_parallel_run(sync_dir, run_id, remove_sync_root_if_empty=True):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    removed = False

    try:
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
            removed = True
    except OSError:
        return None

    if remove_sync_root_if_empty and os.path.isdir(sync_dir):
        try:
            os.rmdir(sync_dir)
        except OSError:
            pass

    return run_dir if removed else None


def write_rank_completion(sync_dir, run_id, rank):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    done_path = os.path.join(run_dir, f"rank_{int(rank)}.done")
    payload = {
        "rank": int(rank),
        "completed_at": time.time(),
    }
    with open(done_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return done_path


def wait_for_rank_completions(sync_dir, run_id, dp_size, timeout_sec):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    expected_paths = [
        os.path.join(run_dir, f"rank_{rank}.done")
        for rank in range(int(dp_size))
    ]
    deadline = time.time() + max(1, int(timeout_sec))
    while True:
        if all(os.path.exists(path) for path in expected_paths):
            return expected_paths
        now = time.time()
        if now >= deadline:
            missing = [path for path in expected_paths if not os.path.exists(path)]
            raise TimeoutError(
                "Timed out waiting for rank completion markers. "
                f"Missing {len(missing)}/{len(expected_paths)} markers."
            )
        time.sleep(0.5)


def init_data_parallel_context(output_path, dp_size, timeout_sec=3600, sync_dir=None):
    if dp_size < 1:
        raise ValueError("--data_parallel_size must be >= 1")
    if dp_size == 1:
        return {
            'enabled': False,
            'size': 1,
            'rank': 0,
            'sync_dir': None,
            'run_id': None,
            'timeout_sec': int(timeout_sec),
        }

    rank_env = os.environ.get("LOCAL_RANK", os.environ.get("RANK"))
    if rank_env is None:
        raise RuntimeError(
            "data_parallel_size > 1 requires launcher-provided rank env. "
            "Use torchrun so LOCAL_RANK/RANK is set."
        )
    rank = int(rank_env)
    if rank < 0 or rank >= dp_size:
        raise RuntimeError(f"Invalid rank {rank} for data_parallel_size={dp_size}.")

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is not None and int(world_size_env) != dp_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size_env} does not match --data_parallel_size={dp_size}."
        )

    sync_dir = resolve_data_parallel_sync_dir(output_path, sync_dir=sync_dir)
    os.makedirs(sync_dir, exist_ok=True)
    run_id = str(_build_run_id()).replace(os.sep, "_")
    os.makedirs(data_parallel_run_dir(sync_dir, run_id), exist_ok=True)

    return {
        'enabled': True,
        'size': int(dp_size),
        'rank': int(rank),
        'sync_dir': sync_dir,
        'run_id': run_id,
        'timeout_sec': int(timeout_sec),
    }


def _build_run_id():
    env_run_id = (
        os.environ.get("DATA_PARALLEL_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("RDZV_ID")
    )
    if env_run_id and env_run_id.lower() != "none":
        return env_run_id
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if master_addr and master_port:
        return f"{master_addr}_{master_port}"
    return f"manual_{os.getppid()}"


def split_samples_for_rank(samples, dp_size, dp_rank):
    query_groups = {}
    for sample_idx, sample in enumerate(samples):
        query_id = sample.get("query_id") if isinstance(sample, dict) else None
        if query_id is None:
            sample_indices = list(range(dp_rank, len(samples), dp_size))
            sample_subset = [samples[idx] for idx in sample_indices]
            return sample_indices, sample_subset
        query_groups.setdefault(int(query_id), []).append(sample_idx)

    sample_indices = []
    for group_idx, grouped_indices in enumerate(query_groups.values()):
        if group_idx % dp_size != dp_rank:
            continue
        sample_indices.extend(grouped_indices)

    sample_subset = [samples[idx] for idx in sample_indices]
    return sample_indices, sample_subset


def shared_samples_path(sync_dir, run_id, trial_idx, split_name):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    safe_split = str(split_name).replace(os.sep, "_")
    return os.path.join(run_dir, f"samples_trial_{trial_idx:04d}_{safe_split}.pkl")


def write_shared_samples(sync_dir, run_id, trial_idx, split_name, samples):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "trial_idx": int(trial_idx),
        "split_name": str(split_name),
        "num_samples": int(len(samples)),
        "samples": samples,
    }
    bundle_path = shared_samples_path(sync_dir, run_id, trial_idx, split_name)
    tmp_path = bundle_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, bundle_path)
    return bundle_path


def wait_for_shared_samples(sync_dir, run_id, trial_idx, split_name, timeout_sec):
    bundle_path = shared_samples_path(sync_dir, run_id, trial_idx, split_name)
    deadline = time.time() + float(timeout_sec)
    announced_wait = False

    while True:
        if os.path.exists(bundle_path):
            with open(bundle_path, "rb") as f:
                payload = pickle.load(f)
            if int(payload.get("trial_idx", -1)) != int(trial_idx):
                raise RuntimeError(
                    f"Shared sample bundle trial mismatch in {bundle_path}: "
                    f"expected {trial_idx}, got {payload.get('trial_idx')}"
                )
            if str(payload.get("split_name")) != str(split_name):
                raise RuntimeError(
                    f"Shared sample bundle split mismatch in {bundle_path}: "
                    f"expected {split_name}, got {payload.get('split_name')}"
                )
            return payload.get("samples", [])
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for shared sample bundle "
                f"(trial {trial_idx}, split={split_name}) at {bundle_path}"
            )
        if not announced_wait:
            print(
                f"Waiting for rank 0 sample bundle "
                f"(trial {trial_idx}, split={split_name})..."
            )
            announced_wait = True
        time.sleep(1.0)


def shared_payload_path(sync_dir, run_id, trial_idx, payload_name):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    safe_name = str(payload_name).replace(os.sep, "_")
    return os.path.join(run_dir, f"payload_trial_{trial_idx:04d}_{safe_name}.pkl")


def write_shared_payload(sync_dir, run_id, trial_idx, payload_name, payload):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    bundle = {
        "trial_idx": int(trial_idx),
        "payload_name": str(payload_name),
        "payload": payload,
    }
    bundle_path = shared_payload_path(sync_dir, run_id, trial_idx, payload_name)
    tmp_path = bundle_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, bundle_path)
    return bundle_path


def wait_for_shared_payload(sync_dir, run_id, trial_idx, payload_name, timeout_sec):
    bundle_path = shared_payload_path(sync_dir, run_id, trial_idx, payload_name)
    deadline = time.time() + float(timeout_sec)
    announced_wait = False

    while True:
        if os.path.exists(bundle_path):
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            if int(bundle.get("trial_idx", -1)) != int(trial_idx):
                raise RuntimeError(
                    f"Shared payload trial mismatch in {bundle_path}: "
                    f"expected {trial_idx}, got {bundle.get('trial_idx')}"
                )
            if str(bundle.get("payload_name")) != str(payload_name):
                raise RuntimeError(
                    f"Shared payload name mismatch in {bundle_path}: "
                    f"expected {payload_name}, got {bundle.get('payload_name')}"
                )
            return bundle.get("payload")
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for shared payload "
                f"(trial {trial_idx}, name={payload_name}) at {bundle_path}"
            )
        if not announced_wait:
            print(
                f"Waiting for rank 0 shared payload "
                f"(trial {trial_idx}, name={payload_name})..."
            )
            announced_wait = True
        time.sleep(1.0)


def hybrid_selection_path(sync_dir, run_id, trial_idx, split_name):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    safe_split = str(split_name).replace(os.sep, "_")
    return os.path.join(run_dir, f"hybrid_selection_trial_{trial_idx:04d}_{safe_split}.json")


def write_hybrid_selection(
    sync_dir,
    run_id,
    trial_idx,
    split_name,
    selected_sample_indices,
    selection_meta,
):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    payload = {
        "trial_idx": int(trial_idx),
        "split_name": str(split_name),
        "selected_sample_indices": [int(idx) for idx in selected_sample_indices],
        "selection_meta": dict(selection_meta) if isinstance(selection_meta, dict) else {},
    }
    selection_path = hybrid_selection_path(sync_dir, run_id, trial_idx, split_name)
    tmp_path = selection_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, selection_path)
    return selection_path


def wait_for_hybrid_selection(sync_dir, run_id, trial_idx, split_name, timeout_sec):
    selection_path = hybrid_selection_path(sync_dir, run_id, trial_idx, split_name)
    deadline = time.time() + float(timeout_sec)
    announced_wait = False

    while True:
        if os.path.exists(selection_path):
            with open(selection_path, "r") as f:
                payload = json.load(f)
            if int(payload.get("trial_idx", -1)) != int(trial_idx):
                raise RuntimeError(
                    f"Hybrid selection trial mismatch in {selection_path}: "
                    f"expected {trial_idx}, got {payload.get('trial_idx')}"
                )
            if str(payload.get("split_name")) != str(split_name):
                raise RuntimeError(
                    f"Hybrid selection split mismatch in {selection_path}: "
                    f"expected {split_name}, got {payload.get('split_name')}"
                )
            selected_sample_indices = payload.get("selected_sample_indices", [])
            selection_meta = payload.get("selection_meta", {})
            return [int(idx) for idx in selected_sample_indices], selection_meta

        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for hybrid selection "
                f"(trial {trial_idx}, split={split_name}) at {selection_path}"
            )
        if not announced_wait:
            print(
                f"Waiting for rank 0 hybrid selection "
                f"(trial {trial_idx}, split={split_name})..."
            )
            announced_wait = True
        time.sleep(1.0)


def merge_parse_stats(parse_stats_list):
    merged = {}
    mode = None
    for stats in parse_stats_list:
        if not isinstance(stats, dict):
            continue
        if mode is None and 'mode' in stats:
            mode = stats.get('mode')
        for key, value in stats.items():
            if key == 'mode':
                continue
            if isinstance(value, int):
                merged[key] = int(merged.get(key, 0) + value)
            elif isinstance(value, float):
                merged[key] = float(merged.get(key, 0.0) + value)
    if mode is not None:
        return {'mode': mode, **merged}
    return merged


def merge_token_usage(token_usage_list):
    merged = {}
    for usage in token_usage_list:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            merged[key] = int(merged.get(key, 0) + numeric)
    return merged


def trial_shard_path(sync_dir, run_id, trial_idx, rank):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    return os.path.join(run_dir, f"trial_{trial_idx:04d}_rank_{rank:04d}.json")


def write_trial_shard(
    sync_dir,
    run_id,
    trial_idx,
    rank,
    total_samples,
    sample_indices,
    trial_results,
):
    if len(sample_indices) != len(trial_results['predictions']):
        raise RuntimeError(
            f"Shard size mismatch: {len(sample_indices)} indices vs "
            f"{len(trial_results['predictions'])} predictions."
        )

    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    remapped_details = []
    raw_details = trial_results.get('detailed_results', [])
    for local_idx, detail in enumerate(raw_details):
        if not isinstance(detail, dict):
            continue
        detail_copy = dict(detail)
        if local_idx < len(sample_indices):
            detail_copy['sample_id'] = int(sample_indices[local_idx])
        remapped_details.append(detail_copy)

    payload = {
        'trial_idx': int(trial_idx),
        'rank': int(rank),
        'total_samples': int(total_samples),
        'sample_indices': [int(x) for x in sample_indices],
        'predictions': [float(x) for x in trial_results['predictions']],
        'labels': [int(x) for x in trial_results['labels']],
        'parse_stats': trial_results.get('parse_stats', {}),
        'token_usage': trial_results.get('token_usage', {}),
        'prompt_embedding_shards': trial_results.get('prompt_embedding_shards', []),
        'detailed_results': remapped_details,
    }

    shard_path = trial_shard_path(sync_dir, run_id, trial_idx, rank)
    tmp_path = shard_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, shard_path)
    return shard_path


def wait_for_trial_shards(sync_dir, run_id, trial_idx, dp_size, timeout_sec):
    expected_paths = [
        trial_shard_path(sync_dir, run_id, trial_idx, rank)
        for rank in range(dp_size)
    ]
    deadline = time.time() + float(timeout_sec)
    announced_wait = False

    while True:
        missing = [p for p in expected_paths if not os.path.exists(p)]
        if not missing:
            return expected_paths
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for DP shards (trial {trial_idx}). "
                f"Missing {len(missing)} files, e.g. {missing[0]}"
            )
        if not announced_wait:
            print(
                f"Waiting for DP shards (trial {trial_idx}): "
                f"{len(missing)}/{len(expected_paths)} missing..."
            )
            announced_wait = True
        time.sleep(1.0)


def hybrid_selected_shard_path(sync_dir, run_id, trial_idx, split_name, rank):
    run_dir = data_parallel_run_dir(sync_dir, run_id)
    safe_split = str(split_name).replace(os.sep, "_")
    return os.path.join(
        run_dir,
        f"hybrid_selected_trial_{trial_idx:04d}_{safe_split}_rank_{rank:04d}.json",
    )


def write_hybrid_selected_shard(
    sync_dir,
    run_id,
    trial_idx,
    split_name,
    rank,
    selected_count,
    selected_positions,
    trial_results,
):
    if len(selected_positions) != len(trial_results["predictions"]):
        raise RuntimeError(
            f"Hybrid shard size mismatch: {len(selected_positions)} indices vs "
            f"{len(trial_results['predictions'])} predictions."
        )

    run_dir = data_parallel_run_dir(sync_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    remapped_details = []
    raw_details = trial_results.get("detailed_results", [])
    for local_idx, detail in enumerate(raw_details):
        if not isinstance(detail, dict):
            continue
        detail_copy = dict(detail)
        if local_idx < len(selected_positions):
            detail_copy["sample_id"] = int(selected_positions[local_idx])
        remapped_details.append(detail_copy)

    payload = {
        "trial_idx": int(trial_idx),
        "split_name": str(split_name),
        "rank": int(rank),
        "selected_count": int(selected_count),
        "selected_positions": [int(x) for x in selected_positions],
        "predictions": [float(x) for x in trial_results["predictions"]],
        "labels": [int(x) for x in trial_results["labels"]],
        "parse_stats": trial_results.get("parse_stats", {}),
        "token_usage": trial_results.get("token_usage", {}),
        "prompt_embedding_shards": trial_results.get("prompt_embedding_shards", []),
        "detailed_results": remapped_details,
    }

    shard_path = hybrid_selected_shard_path(
        sync_dir=sync_dir,
        run_id=run_id,
        trial_idx=trial_idx,
        split_name=split_name,
        rank=rank,
    )
    tmp_path = shard_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, shard_path)
    return shard_path


def wait_for_hybrid_selected_shards(
    sync_dir,
    run_id,
    trial_idx,
    split_name,
    dp_size,
    timeout_sec,
):
    expected_paths = [
        hybrid_selected_shard_path(
            sync_dir=sync_dir,
            run_id=run_id,
            trial_idx=trial_idx,
            split_name=split_name,
            rank=rank,
        )
        for rank in range(dp_size)
    ]
    deadline = time.time() + float(timeout_sec)
    announced_wait = False

    while True:
        missing = [p for p in expected_paths if not os.path.exists(p)]
        if not missing:
            return expected_paths
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for hybrid selected shards "
                f"(trial {trial_idx}, split={split_name}). "
                f"Missing {len(missing)} files, e.g. {missing[0]}"
            )
        if not announced_wait:
            print(
                f"Waiting for hybrid selected shards (trial {trial_idx}, split={split_name}): "
                f"{len(missing)}/{len(expected_paths)} missing..."
            )
            announced_wait = True
        time.sleep(1.0)


def merge_hybrid_selected_shards(shard_paths, selected_count):
    merged_predictions = [None] * int(selected_count)
    merged_labels = [None] * int(selected_count)
    details_by_index = {}
    parse_stats_list = []
    token_usage_list = []
    prompt_embedding_shards = []

    for shard_path in shard_paths:
        with open(shard_path, "r") as f:
            shard = json.load(f)

        expected_selected_count = int(shard.get("selected_count", selected_count))
        if expected_selected_count != int(selected_count):
            raise RuntimeError(
                f"Hybrid shard selected_count mismatch in {shard_path}: "
                f"{expected_selected_count} != {selected_count}"
            )

        selected_positions = shard.get("selected_positions", [])
        predictions = shard.get("predictions", [])
        labels = shard.get("labels", [])
        if not (len(selected_positions) == len(predictions) == len(labels)):
            raise RuntimeError(f"Corrupt hybrid shard lengths in {shard_path}")

        for local_idx, selected_pos in enumerate(selected_positions):
            sidx = int(selected_pos)
            if sidx < 0 or sidx >= int(selected_count):
                raise RuntimeError(
                    f"Out-of-range selected position {sidx} in {shard_path}"
                )
            if merged_predictions[sidx] is not None:
                raise RuntimeError(f"Duplicate selected position {sidx} in hybrid shards")
            merged_predictions[sidx] = float(predictions[local_idx])
            merged_labels[sidx] = int(labels[local_idx])

        parse_stats_list.append(shard.get("parse_stats", {}))
        token_usage_list.append(shard.get("token_usage", {}))
        prompt_embedding_shards.extend(shard.get("prompt_embedding_shards", []))
        for detail in shard.get("detailed_results", []):
            if not isinstance(detail, dict):
                continue
            sample_id = detail.get("sample_id")
            if isinstance(sample_id, int) and 0 <= sample_id < int(selected_count):
                details_by_index[sample_id] = detail

    missing_indices = [idx for idx, pred in enumerate(merged_predictions) if pred is None]
    if missing_indices:
        raise RuntimeError(
            f"Missing merged hybrid selected predictions for {len(missing_indices)} samples. "
            f"First missing index: {missing_indices[0]}"
        )

    return {
        "predictions": merged_predictions,
        "labels": merged_labels,
        "parse_stats": merge_parse_stats(parse_stats_list),
        "token_usage": merge_token_usage(token_usage_list),
        "prompt_embedding_shards": prompt_embedding_shards,
        "detailed_results": [
            details_by_index[idx]
            for idx in range(int(selected_count))
            if idx in details_by_index
        ],
    }


def merge_trial_shards(shard_paths, total_samples, dtgb_eval_batch_size=None):
    merged_predictions = [None] * total_samples
    merged_labels = [None] * total_samples
    details_by_index = {}
    parse_stats_list = []
    token_usage_list = []
    prompt_embedding_shards = []

    for shard_path in shard_paths:
        with open(shard_path, 'r') as f:
            shard = json.load(f)

        sample_indices = shard.get('sample_indices', [])
        predictions = shard.get('predictions', [])
        labels = shard.get('labels', [])
        if not (len(sample_indices) == len(predictions) == len(labels)):
            raise RuntimeError(f"Corrupt shard lengths in {shard_path}")

        for local_idx, global_idx in enumerate(sample_indices):
            gidx = int(global_idx)
            if gidx < 0 or gidx >= total_samples:
                raise RuntimeError(f"Out-of-range sample index {gidx} in {shard_path}")
            if merged_predictions[gidx] is not None:
                raise RuntimeError(f"Duplicate sample index {gidx} in DP shards")
            merged_predictions[gidx] = float(predictions[local_idx])
            merged_labels[gidx] = int(labels[local_idx])

        parse_stats_list.append(shard.get('parse_stats', {}))
        token_usage_list.append(shard.get('token_usage', {}))
        prompt_embedding_shards.extend(shard.get('prompt_embedding_shards', []))
        for detail in shard.get('detailed_results', []):
            if not isinstance(detail, dict):
                continue
            sample_id = detail.get('sample_id')
            if isinstance(sample_id, int) and 0 <= sample_id < total_samples:
                details_by_index[sample_id] = detail

    missing_indices = [idx for idx, pred in enumerate(merged_predictions) if pred is None]
    if missing_indices:
        raise RuntimeError(
            f"Missing merged predictions for {len(missing_indices)} samples. "
            f"First missing index: {missing_indices[0]}"
        )

    results = compute_prediction_metrics(
        merged_predictions,
        merged_labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    results['parse_stats'] = merge_parse_stats(parse_stats_list)
    results['token_usage'] = merge_token_usage(token_usage_list)
    results['prompt_embedding_shards'] = prompt_embedding_shards
    results['detailed_results'] = [
        details_by_index[idx]
        for idx in range(total_samples)
        if idx in details_by_index
    ]
    return results
