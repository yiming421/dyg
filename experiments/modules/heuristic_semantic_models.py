#!/usr/bin/env python3
"""
Semantic embedding, smoothing, and semantic scoring helpers for heuristic baselines.
"""
import math
import time

import numpy as np
from tqdm import tqdm

from experiments.modules.heuristic_models import build_csr_from_neighbor_sampler


def build_source_history_mean_initialized_embeddings(
    base_embeddings,
    node_id_lookup,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
):
    """
    Replace each source-node embedding with the mean of its observed destination/item
    embeddings under the supplied edge set. Destination embeddings remain unchanged.
    """
    import torch

    if isinstance(base_embeddings, torch.Tensor):
        device = base_embeddings.device
        base = base_embeddings.detach().clone().float()
    else:
        device = None
        base = torch.from_numpy(np.asarray(base_embeddings, dtype=np.float32)).float()

    if isinstance(node_id_lookup, torch.Tensor):
        lookup = node_id_lookup.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        lookup = np.asarray(node_id_lookup, dtype=np.int64)

    src_arr = np.asarray(src_node_ids, dtype=np.int64)
    dst_arr = np.asarray(dst_node_ids, dtype=np.int64)

    sums = torch.zeros_like(base)
    counts = torch.zeros(base.size(0), dtype=torch.long, device=base.device)

    valid = (
        (src_arr >= 0) & (src_arr < len(lookup)) &
        (dst_arr >= 0) & (dst_arr < len(lookup))
    )
    if not np.any(valid):
        return (base.to(device) if device is not None else base.cpu().numpy()), 0

    src_emb_idx = lookup[src_arr[valid]]
    dst_emb_idx = lookup[dst_arr[valid]]
    valid_pairs = (src_emb_idx >= 0) & (dst_emb_idx >= 0)
    if not np.any(valid_pairs):
        return (base.to(device) if device is not None else base.cpu().numpy()), 0

    src_emb_idx_t = torch.from_numpy(src_emb_idx[valid_pairs]).long().to(base.device)
    dst_emb_idx_t = torch.from_numpy(dst_emb_idx[valid_pairs]).long().to(base.device)

    sums.index_add_(0, src_emb_idx_t, base.index_select(0, dst_emb_idx_t))
    counts.index_add_(0, src_emb_idx_t, torch.ones_like(src_emb_idx_t))

    updated = base.clone()
    replace_mask = counts > 0
    if replace_mask.any():
        updated[replace_mask] = sums[replace_mask] / counts[replace_mask].unsqueeze(1).float()
        updated[replace_mask] = torch.nn.functional.normalize(updated[replace_mask], p=2, dim=1)

    replaced_nodes = int(replace_mask.sum().item())
    if device is not None:
        return updated.to(device), replaced_nodes
    return updated.cpu().numpy(), replaced_nodes


def precompute_entity_embeddings(entity_texts, model_name='intfloat/e5-large-v2', device='cuda'):
    """
    Precompute text embeddings for all entities (cached for efficiency).

    Args:
        entity_texts: Dict mapping entity_id -> text
        model_name: Embedding model name or local path
        device: 'cuda', 'cuda:N', or 'cpu'

    Returns:
        embeddings: np.ndarray of shape (num_entities, embedding_dim)
        entity_id_to_idx: Dict mapping entity_id -> embedding index
    """
    import os

    def _infer_model_family(name: str) -> str:
        lowered = str(name).lower()
        if 'qwen3-embedding' in lowered or 'qwen-embedding' in lowered:
            return 'qwen3'
        if 'e5' in lowered:
            return 'e5'
        return 'generic'

    def _normalize_text(text) -> str:
        if text is None:
            return "unknown"
        if isinstance(text, str):
            stripped = text.strip()
            return stripped if stripped else "unknown"
        if isinstance(text, (float, np.floating)) and np.isnan(text):
            return "unknown"
        return str(text)

    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = bool((attention_mask[:, -1] == 1).all().item())
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    print(f"Precomputing embeddings with {model_name}...")
    start = time.time()
    model_family = _infer_model_family(model_name)
    model_is_local = os.path.isdir(model_name)

    try:
        from sentence_transformers import SentenceTransformer
        use_sentence_transformers = (model_family != 'qwen3')
    except ImportError:
        use_sentence_transformers = False

    from transformers import AutoTokenizer, AutoModel
    import torch

    entity_ids = sorted(entity_texts.keys())
    texts = [_normalize_text(entity_texts.get(eid, "unknown")) for eid in entity_ids]

    if model_family == 'e5':
        texts = [f"passage: {text}" for text in texts]

    batch_size = 8 if model_family == 'qwen3' else 32

    if use_sentence_transformers:
        model = SentenceTransformer(model_name, device=device, local_files_only=model_is_local)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    else:
        tokenizer_kwargs = {'local_files_only': model_is_local}
        if model_family == 'qwen3':
            tokenizer_kwargs['padding_side'] = 'left'
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        model_kwargs = {
            'local_files_only': model_is_local,
            'low_cpu_mem_usage': True,
        }
        if model_family == 'qwen3':
            model_kwargs['dtype'] = 'auto'
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        device_obj = torch.device(device if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        model = model.to(device_obj)
        model.eval()

        embeddings_list = []
        batch_starts = range(0, len(texts), batch_size)
        total_batches = int(math.ceil(len(texts) / float(batch_size))) if batch_size > 0 else 0
        for i in tqdm(
            batch_starts,
            total=total_batches,
            desc="Embedding batches",
            unit="batch",
        ):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {key: value.to(device_obj) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                if model_family == 'qwen3':
                    embeddings_batch = _last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                else:
                    embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                embeddings_list.append(embeddings_batch.float().cpu().numpy())

        embeddings = np.vstack(embeddings_list)

    entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    elapsed = time.time() - start
    print(f"✓ Computed {len(embeddings)} embeddings in {elapsed:.2f}s")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    return embeddings, entity_id_to_idx


def smooth_embeddings_by_time_window_torch(embeddings,
                                           src_node_ids: np.ndarray,
                                           dst_node_ids: np.ndarray,
                                           node_interact_times: np.ndarray,
                                           time_window: float,
                                           num_steps: int = 1,
                                           symmetric_norm: bool = True,
                                           decay_gamma: float = None,
                                           residual_alpha: float = 0.0,
                                           undirected: bool = False,
                                           layer_norm: str = None,
                                           pre_norm: str = None,
                                           device: str = None,
                                           debug: bool = False,
                                           log_dampen: bool = False,
                                           supernode_strength: float = 0.0,
                                           reference_time: float = None,
                                           hetero_highpass_coef: float = 0.0,
                                           hetero_tau: float = 0.0,
                                           hetero_filter_tau: float = None,
                                           hetero_filter_keep_weight: float = 1.0,
                                           endpoint_topk_recent: int = None,
                                           endpoint_topk_ensure_coverage: bool = False,
                                           endpoint_topk_mode: str = "union",
                                           return_norm_adj: bool = False,
                                           return_sum_adj: bool = False):
    """
    Fast, parameter-free GCN smoothing (A+I)-based over a recent time window using torch sparse ops.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if isinstance(embeddings, np.ndarray):
        emb = torch.from_numpy(embeddings)
    else:
        emb = embeddings
    emb = emb.to(device)

    if pre_norm == "zscore":
        mean = emb.mean(dim=0, keepdim=True)
        std = emb.std(dim=0, keepdim=True).clamp(min=1e-8)
        emb = (emb - mean) / std

    num_nodes = emb.size(0)
    ref_time = float(node_interact_times.max()) if reference_time is None else float(reference_time)
    mask = (node_interact_times <= ref_time) & (node_interact_times >= (ref_time - time_window))
    src_np_masked = src_node_ids[mask]
    dst_np_masked = dst_node_ids[mask]
    times_np_masked = node_interact_times[mask]
    src = torch.as_tensor(src_np_masked, dtype=torch.long, device=device)
    dst = torch.as_tensor(dst_np_masked, dtype=torch.long, device=device)
    times = torch.as_tensor(times_np_masked, dtype=torch.float32, device=device)

    valid = (src < num_nodes) & (dst < num_nodes)
    src, dst, times = src[valid], dst[valid], times[valid]

    endpoint_topk_mode = str(endpoint_topk_mode).strip().lower()
    if endpoint_topk_mode not in {"union", "per_node"}:
        raise ValueError(
            "endpoint_topk_mode must be one of {'union', 'per_node'}, "
            f"got {endpoint_topk_mode!r}."
        )
    do_endpoint_cap = endpoint_topk_recent is not None and endpoint_topk_recent > 0 and src.numel() > 0
    if do_endpoint_cap:
        src_np = src.detach().cpu().numpy()
        dst_np = dst.detach().cpu().numpy()
        times_np = times.detach().cpu().numpy()
        if endpoint_topk_mode == "per_node":
            if not undirected:
                raise ValueError(
                    "endpoint_topk_mode='per_node' requires undirected=True so "
                    "each endpoint can retain its own recent interaction history."
                )

            # Build one directed aggregation row per retained endpoint incidence.
            # For an event u-v, u keeps the row (u, v) when the event belongs to
            # u's latest K history, while v independently keeps (v, u) when it
            # belongs to v's latest K. This matches DyGFormer's per-query-node
            # undirected history cap and guarantees at most K interaction
            # contributions per real-node row before pair coalescing.
            edge_idx = np.arange(len(src_np), dtype=np.int64)
            incidence_nodes = np.concatenate([src_np, dst_np])
            incidence_neighbors = np.concatenate([dst_np, src_np])
            incidence_times = np.concatenate([times_np, times_np])
            incidence_edges = np.concatenate([edge_idx, edge_idx])
            order = np.lexsort(
                (-incidence_edges, -incidence_times, incidence_nodes)
            )
            sorted_nodes = incidence_nodes[order]
            group_start = np.empty(len(order), dtype=np.int64)
            is_start = np.empty(len(order), dtype=bool)
            is_start[0] = True
            is_start[1:] = sorted_nodes[1:] != sorted_nodes[:-1]
            start_indices = np.flatnonzero(is_start)
            group_sizes = np.diff(np.append(start_indices, len(order)))
            group_start[:] = np.repeat(start_indices, group_sizes)
            ranks = np.arange(len(order), dtype=np.int64) - group_start
            selected = order[ranks < int(endpoint_topk_recent)]

            src = torch.as_tensor(
                incidence_nodes[selected], dtype=torch.long, device=src.device
            )
            dst = torch.as_tensor(
                incidence_neighbors[selected], dtype=torch.long, device=dst.device
            )
            times = torch.as_tensor(
                incidence_times[selected], dtype=torch.float32, device=times.device
            )
            # The two endpoint directions were materialized independently above.
            undirected = False
        else:
            order = np.argsort(times_np, kind="mergesort")[::-1]

            keep_src = np.zeros(len(src_np), dtype=bool)
            cnt_src = np.zeros(num_nodes, dtype=np.int32)
            for idx in order:
                u = int(src_np[idx])
                if cnt_src[u] < int(endpoint_topk_recent):
                    keep_src[idx] = True
                    cnt_src[u] += 1

            keep_dst = np.zeros(len(src_np), dtype=bool)
            cnt_dst = np.zeros(num_nodes, dtype=np.int32)
            for idx in order:
                v = int(dst_np[idx])
                if cnt_dst[v] < int(endpoint_topk_recent):
                    keep_dst[idx] = True
                    cnt_dst[v] += 1

            keep_mask = keep_src | keep_dst

            if endpoint_topk_ensure_coverage:
                active = np.zeros(num_nodes, dtype=bool)
                active[src_np] = True
                active[dst_np] = True

                covered = np.zeros(num_nodes, dtype=bool)
                kept_idx = np.where(keep_mask)[0]
                if len(kept_idx) > 0:
                    covered[src_np[kept_idx]] = True
                    covered[dst_np[kept_idx]] = True

                missing = active & (~covered)
                missing_count = int(missing.sum())
                if missing_count > 0:
                    for idx in order:
                        u = int(src_np[idx])
                        v = int(dst_np[idx])
                        need_u = missing[u]
                        need_v = missing[v]
                        if not (need_u or need_v):
                            continue
                        keep_mask[idx] = True
                        if need_u:
                            missing[u] = False
                            missing_count -= 1
                        if need_v:
                            missing[v] = False
                            missing_count -= 1
                        if missing_count <= 0:
                            break

            keep_idx = np.where(keep_mask)[0]
            keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long, device=src.device)
            src = src.index_select(0, keep_idx_t)
            dst = dst.index_select(0, keep_idx_t)
            times = times.index_select(0, keep_idx_t)

    if undirected:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
        times = torch.cat([times, times])

    if decay_gamma is None:
        base_vals = torch.ones(len(src), dtype=torch.float32, device=src.device)
    else:
        base_vals = torch.exp(-decay_gamma * (ref_time - times))

    do_hetero_split = hetero_highpass_coef > 0.0
    do_hetero_filter = (hetero_filter_tau is not None) and (hetero_filter_keep_weight < 1.0)

    if (do_hetero_split or do_hetero_filter) and src.numel() > 0:
        src_feat = emb[src]
        dst_feat = emb[dst]
        src_feat = src_feat / src_feat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        dst_feat = dst_feat / dst_feat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        edge_cos = (src_feat * dst_feat).sum(dim=1)
        if do_hetero_filter:
            filter_mask = (edge_cos < hetero_filter_tau)
        else:
            filter_mask = torch.zeros(len(src), dtype=torch.bool, device=src.device)
        if do_hetero_split:
            hetero_mask = (edge_cos < hetero_tau)
        else:
            hetero_mask = torch.zeros(len(src), dtype=torch.bool, device=src.device)
    else:
        filter_mask = torch.zeros(len(src), dtype=torch.bool, device=src.device)
        hetero_mask = torch.zeros(len(src), dtype=torch.bool, device=src.device)

    if do_hetero_filter and filter_mask.any():
        if hetero_filter_keep_weight <= 0.0:
            keep = ~filter_mask
            src, dst, times = src[keep], dst[keep], times[keep]
            base_vals = base_vals[keep]
            hetero_mask = hetero_mask[keep]
        else:
            base_vals = base_vals.clone()
            base_vals[filter_mask] = base_vals[filter_mask] * float(hetero_filter_keep_weight)

    homo_src = src[~hetero_mask]
    homo_dst = dst[~hetero_mask]
    homo_vals = base_vals[~hetero_mask]
    hetero_src = src[hetero_mask]
    hetero_dst = dst[hetero_mask]
    hetero_vals = base_vals[hetero_mask]

    graph_num_nodes = num_nodes
    # A virtual node may aggregate only nodes observed in this history window.
    # The embedding table can also contain held-out or not-yet-observed nodes.
    supernode_active = supernode_strength > 0.0 and src.numel() > 0
    emb_for_prop = emb

    if supernode_active:
        graph_num_nodes = num_nodes + 1
        supernode_id = num_nodes
        all_nodes = torch.unique(torch.cat([src, dst]))
        super_ids = torch.full_like(all_nodes, supernode_id)

        homo_src = torch.cat([homo_src, all_nodes, super_ids])
        homo_dst = torch.cat([homo_dst, super_ids, all_nodes])
        super_vals = torch.full(
            (2 * all_nodes.numel(),),
            float(supernode_strength),
            dtype=torch.float32,
            device=homo_vals.device,
        )
        homo_vals = torch.cat([homo_vals, super_vals])

        super_emb = emb[all_nodes].mean(dim=0, keepdim=True)
        emb_for_prop = torch.cat([emb, super_emb], dim=0)

    def build_sparse_adj(chan_src, chan_dst, chan_vals):
        chan_indices = torch.stack([chan_src, chan_dst], dim=0)
        coalesced = torch.sparse_coo_tensor(
            chan_indices,
            chan_vals,
            (graph_num_nodes, graph_num_nodes),
        ).coalesce()
        indices = coalesced.indices().to(device)
        values = coalesced.values().to(device)
        if log_dampen:
            values = torch.log(1 + values)
        adj = torch.sparse_coo_tensor(indices, values, (graph_num_nodes, graph_num_nodes), device=device)
        return adj.coalesce(), indices, values

    sum_adj_homo, _, _ = build_sparse_adj(homo_src, homo_dst, homo_vals)

    loop_idx = torch.arange(graph_num_nodes, dtype=torch.long, device=homo_src.device)
    homo_src = torch.cat([homo_src, loop_idx])
    homo_dst = torch.cat([homo_dst, loop_idx])
    homo_vals = torch.cat(
        [homo_vals, torch.ones(len(loop_idx), dtype=torch.float32, device=homo_vals.device)]
    )

    def build_norm_adj(chan_src, chan_dst, chan_vals):
        adj, indices, values = build_sparse_adj(chan_src, chan_dst, chan_vals)
        s = indices[0]
        d = indices[1]
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        if symmetric_norm:
            deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
            norm_values = deg_inv_sqrt[s] * values * deg_inv_sqrt[d]
        else:
            deg_inv = deg.clamp(min=1.0).reciprocal()
            norm_values = deg_inv[s] * values
        norm_adj = torch.sparse_coo_tensor(indices, norm_values, (graph_num_nodes, graph_num_nodes), device=device)
        return norm_adj.coalesce(), values, deg

    norm_adj_homo, values_homo, deg_homo = build_norm_adj(homo_src, homo_dst, homo_vals)
    hetero_active = do_hetero_split and len(hetero_vals) > 0
    if hetero_active:
        norm_adj_hetero, values_hetero, deg_hetero = build_norm_adj(hetero_src, hetero_dst, hetero_vals)
    else:
        norm_adj_hetero, values_hetero = None, None
        deg_hetero = torch.zeros_like(deg_homo)

    deg = deg_homo + deg_hetero

    if debug:
        print("\n" + "="*80)
        print("GCN SMOOTHING DIAGNOSTICS")
        print("="*80)
        total_edges = len(values_homo) + (len(values_hetero) if values_hetero is not None else 0)
        print(f"Graph stats: {graph_num_nodes} graph nodes ({num_nodes} real), {total_edges} edges (after coalescing)")
        print(f"Time window: {time_window}, Decay gamma: {decay_gamma}, Residual alpha: {residual_alpha}")
        print(f"Symmetric norm: {symmetric_norm}, Undirected: {undirected}, Layer norm: {layer_norm}, Pre-norm: {pre_norm}, Steps: {num_steps}")
        print(f"Supernode strength: {supernode_strength}")
        if do_hetero_filter:
            num_filter = int(filter_mask.sum().item())
            num_interaction = int(len(filter_mask))
            ratio = (num_filter / max(1, num_interaction)) * 100.0
            print(f"Hetero filter: tau={hetero_filter_tau}, keep_weight={hetero_filter_keep_weight}, filtered_edges={num_filter}/{num_interaction} ({ratio:.1f}%)")
        if hetero_highpass_coef > 0.0:
            num_hetero = int(hetero_mask.sum().item())
            num_interaction = int(len(hetero_mask))
            ratio = (num_hetero / max(1, num_interaction)) * 100.0
            print(f"Hetero high-pass: coef={hetero_highpass_coef}, tau={hetero_tau}, hetero_edges={num_hetero}/{num_interaction} ({ratio:.1f}%)")
        if do_endpoint_cap:
            print(
                "Endpoint top-k recent cap: "
                f"k={endpoint_topk_recent}, mode={endpoint_topk_mode}, "
                f"ensure_coverage={endpoint_topk_ensure_coverage}"
            )

        deg_np = deg[:num_nodes].cpu().numpy() if supernode_active else deg.cpu().numpy()
        print(f"\nDegree distribution:")
        print(f"  Min/Med/Mean/Max: {deg_np.min():.1f} / {np.median(deg_np):.1f} / {deg_np.mean():.1f} / {deg_np.max():.1f}")
        print(f"  Isolated nodes (deg=1, self-loop only): {(deg_np == 1).sum()} ({(deg_np == 1).sum() / len(deg_np) * 100:.1f}%)")

        if values_homo is not None and len(values_homo) > 0:
            edge_weights_np = values_homo.cpu().numpy()
        elif values_hetero is not None and len(values_hetero) > 0:
            edge_weights_np = values_hetero.cpu().numpy()
        else:
            edge_weights_np = np.array([1.0], dtype=np.float32)
        print(f"\nEdge weight distribution (before normalization):")
        print(f"  Min/Med/Mean/Max: {edge_weights_np.min():.3e} / {np.median(edge_weights_np):.3e} / {edge_weights_np.mean():.3e} / {edge_weights_np.max():.3e}")

    def compute_smoothing_metrics(embeddings_tensor, step_name):
        norms = embeddings_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = embeddings_tensor / norms

        n_sample = min(1000, normed.size(0))
        if normed.size(0) > n_sample:
            indices = torch.randperm(normed.size(0), device=device)[:n_sample]
            sample = normed[indices]
        else:
            sample = normed

        sample_cpu = sample.cpu()
        sim_matrix = torch.mm(sample_cpu, sample_cpu.t())
        mask = ~torch.eye(len(sample_cpu), dtype=torch.bool)
        pairwise_sims = sim_matrix[mask]
        mean_sim = pairwise_sims.mean().item()
        std_sim = pairwise_sims.std().item()

        emb_mean = embeddings_tensor.mean(dim=0)
        emb_std = embeddings_tensor.std(dim=0).mean().item()
        emb_norm_mean = norms.mean().item()
        emb_norm_std = norms.std().item()

        centered = embeddings_tensor - emb_mean
        dist_from_mean = centered.norm(dim=1).mean().item()

        print(f"\n{step_name}:")
        print(f"  Mean pairwise cosine sim: {mean_sim:.4f} ± {std_sim:.4f}")
        print(f"  Embedding std (avg across dims): {emb_std:.4f}")
        print(f"  Embedding norm: {emb_norm_mean:.4f} ± {emb_norm_std:.4f}")
        print(f"  Mean distance from centroid: {dist_from_mean:.4f}")

        return mean_sim, emb_std

    def apply_layer_norm(x, norm_type):
        if norm_type == 'ln':
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
            return (x - mean) / std
        else:
            return x

    def real_nodes(x):
        return x[:num_nodes] if supernode_active else x

    steps = max(1, num_steps)
    out = emb_for_prop

    if debug:
        print("\n" + "-"*80)
        print("LAYER-BY-LAYER SMOOTHING ANALYSIS")
        print("-"*80)
        start_label = f"Layer 0 (pre-norm: {pre_norm})" if pre_norm else "Layer 0 (original)"
        compute_smoothing_metrics(real_nodes(out), start_label)

    for step in range(steps):
        out_homo = torch.sparse.mm(norm_adj_homo, out)
        if hetero_active:
            out_hetero_lp = torch.sparse.mm(norm_adj_hetero, out)
            out = out_homo + hetero_highpass_coef * (out - out_hetero_lp)
        else:
            out = out_homo
        if layer_norm is not None:
            out = apply_layer_norm(out, layer_norm)
        if debug:
            compute_smoothing_metrics(real_nodes(out), f"Layer {step + 1}")

    if residual_alpha > 0.0:
        out = residual_alpha * emb_for_prop + (1.0 - residual_alpha) * out
        if debug:
            compute_smoothing_metrics(real_nodes(out), f"After residual (alpha={residual_alpha})")

    if debug:
        print("="*80 + "\n")

    out_real = out[:num_nodes] if supernode_active else out
    if return_norm_adj or return_sum_adj:
        if hetero_active and hetero_highpass_coef > 0.0:
            raise ValueError(
                "Returning sparse operators is not supported with hetero_highpass_coef > 0."
            )
        outputs = [out_real]
        if return_norm_adj:
            outputs.append(norm_adj_homo)
        if return_sum_adj:
            outputs.append(sum_adj_homo)
        return tuple(outputs)
    return out_real


def score_links_by_semantic_similarity(neighbor_sampler,
                                        sources: np.ndarray,
                                        targets: np.ndarray,
                                        prediction_times: np.ndarray,
                                        embeddings: np.ndarray = None,
                                        entity_id_to_idx: dict = None) -> np.ndarray:
    """
    Score batch of links by semantic similarity (cosine) between entity embeddings
    """
    if embeddings is None:
        raise ValueError("Embeddings must be precomputed!")

    import torch

    if isinstance(embeddings, np.ndarray):
        emb_tensor = torch.from_numpy(embeddings).to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        emb_tensor = embeddings
    device = emb_tensor.device

    if isinstance(sources, np.ndarray):
        src_tensor = torch.from_numpy(sources).long().to(device)
        tgt_tensor = torch.from_numpy(targets).long().to(device)
    else:
        src_tensor = sources.long().to(device)
        tgt_tensor = targets.long().to(device)

    if isinstance(entity_id_to_idx, torch.Tensor):
        lookup = entity_id_to_idx.to(device)
        max_id = lookup.size(0)

        src_mask = (src_tensor < max_id)
        tgt_mask = (tgt_tensor < max_id)

        src_indices = torch.zeros_like(src_tensor)
        tgt_indices = torch.zeros_like(tgt_tensor)

        src_indices[src_mask] = lookup[src_tensor[src_mask]]
        tgt_indices[tgt_mask] = lookup[tgt_tensor[tgt_mask]]
    elif isinstance(entity_id_to_idx, dict):
        s_list = [entity_id_to_idx.get(int(x), -1) for x in sources]
        t_list = [entity_id_to_idx.get(int(x), -1) for x in targets]
        src_indices = torch.tensor(s_list, device=device)
        tgt_indices = torch.tensor(t_list, device=device)
    else:
        raise ValueError("entity_id_to_idx must be a dict or a torch.Tensor (lookup)")

    valid_mask = (src_indices != -1) & (tgt_indices != -1)
    scores = torch.zeros(len(src_tensor), device=device, dtype=torch.float32)

    if valid_mask.any():
        v_src = src_indices[valid_mask]
        v_tgt = tgt_indices[valid_mask]

        s_emb = emb_tensor[v_src]
        t_emb = emb_tensor[v_tgt]
        scores[valid_mask] = torch.sum(s_emb * t_emb, dim=1)

    return scores.cpu().numpy()


def score_links_by_semantic_history_mean(
                                            neighbor_sampler,
                                            sources: np.ndarray,
                                            targets: np.ndarray,
                                            prediction_times: np.ndarray,
                                            embeddings: np.ndarray = None,
                                            entity_id_to_idx: dict = None,
                                            fallback_to_raw_source: bool = True) -> np.ndarray:
    """
    Score links by cosine similarity between:
      - source profile = mean of source's historical neighbor/item embeddings before t
      - target profile = raw target embedding
    """
    if embeddings is None:
        raise ValueError("Embeddings must be precomputed!")

    import torch

    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = all_indices
        neighbor_sampler._csr_times = all_times
    indptr = neighbor_sampler._csr_indptr
    all_indices = neighbor_sampler._csr_indices
    all_times = neighbor_sampler._csr_times

    if isinstance(embeddings, np.ndarray):
        emb_tensor = torch.from_numpy(embeddings).to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        emb_tensor = embeddings
    device = emb_tensor.device

    if isinstance(sources, np.ndarray):
        src_arr = sources.astype(np.int64, copy=False)
        tgt_arr = targets.astype(np.int64, copy=False)
        time_arr = prediction_times.astype(np.float64, copy=False)
        src_tensor = torch.from_numpy(src_arr).long().to(device)
        tgt_tensor = torch.from_numpy(tgt_arr).long().to(device)
    else:
        src_tensor = sources.long().to(device)
        tgt_tensor = targets.long().to(device)
        src_arr = src_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        tgt_arr = tgt_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        time_arr = prediction_times.detach().cpu().numpy().astype(np.float64, copy=False)

    if isinstance(entity_id_to_idx, torch.Tensor):
        lookup_cpu = entity_id_to_idx.detach().cpu().numpy()
    elif isinstance(entity_id_to_idx, dict):
        max_node_id = max(
            int(max(src_arr)) if len(src_arr) > 0 else -1,
            int(max(tgt_arr)) if len(tgt_arr) > 0 else -1,
            int(indptr.shape[0] - 2),
        )
        lookup_cpu = np.full(max_node_id + 1, -1, dtype=np.int64)
        for node_id, emb_idx in entity_id_to_idx.items():
            node_id = int(node_id)
            if 0 <= node_id <= max_node_id:
                lookup_cpu[node_id] = int(emb_idx)
    else:
        raise ValueError("entity_id_to_idx must be a dict or a torch.Tensor (lookup)")

    max_lookup_id = int(len(lookup_cpu))
    scores = torch.zeros(len(src_arr), device=device, dtype=torch.float32)

    for row_idx, (src, tgt, pred_t) in enumerate(zip(src_arr, tgt_arr, time_arr)):
        if src < 0 or src + 1 >= len(indptr) or tgt < 0 or tgt >= max_lookup_id:
            continue

        tgt_emb_idx = int(lookup_cpu[tgt])
        if tgt_emb_idx < 0:
            continue

        start = int(indptr[src])
        end = int(indptr[src + 1])
        cutoff = start + int(np.searchsorted(all_times[start:end], pred_t, side='left'))
        hist_node_ids = all_indices[start:cutoff]

        src_profile = None
        if len(hist_node_ids) > 0:
            valid_hist_ids = hist_node_ids[(hist_node_ids >= 0) & (hist_node_ids < max_lookup_id)]
            if len(valid_hist_ids) > 0:
                hist_emb_idx = lookup_cpu[valid_hist_ids]
                hist_emb_idx = hist_emb_idx[hist_emb_idx >= 0]
                if len(hist_emb_idx) > 0:
                    hist_idx_tensor = torch.from_numpy(hist_emb_idx).long().to(device)
                    src_profile = emb_tensor.index_select(0, hist_idx_tensor).mean(dim=0)
                    src_norm = torch.norm(src_profile, p=2)
                    if float(src_norm.item()) > 0.0:
                        src_profile = src_profile / src_norm

        if src_profile is None and fallback_to_raw_source and src < max_lookup_id:
            src_emb_idx = int(lookup_cpu[src])
            if src_emb_idx >= 0:
                src_profile = emb_tensor[src_emb_idx]

        if src_profile is None:
            continue

        tgt_profile = emb_tensor[tgt_emb_idx]
        scores[row_idx] = torch.sum(src_profile * tgt_profile)

    return scores.cpu().numpy()


def score_links_by_semantic_history_query_conditioned(
                                                        neighbor_sampler,
                                                        sources: np.ndarray,
                                                        targets: np.ndarray,
                                                        prediction_times: np.ndarray,
                                                        embeddings: np.ndarray = None,
                                                        entity_id_to_idx: dict = None,
                                                        fallback_to_raw_source: bool = True,
                                                        attention_temperature: float = 0.2) -> np.ndarray:
    """
    Score links by cosine similarity between:
      - source profile = target-conditioned weighted average of source's historical
        neighbor/item embeddings before t
      - target profile = raw target embedding
    """
    if embeddings is None:
        raise ValueError("Embeddings must be precomputed!")

    import torch

    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = all_indices
        neighbor_sampler._csr_times = all_times
    indptr = neighbor_sampler._csr_indptr
    all_indices = neighbor_sampler._csr_indices
    all_times = neighbor_sampler._csr_times

    if isinstance(embeddings, np.ndarray):
        emb_tensor = torch.from_numpy(embeddings).to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        emb_tensor = embeddings
    device = emb_tensor.device

    if isinstance(sources, np.ndarray):
        src_arr = sources.astype(np.int64, copy=False)
        tgt_arr = targets.astype(np.int64, copy=False)
        time_arr = prediction_times.astype(np.float64, copy=False)
    else:
        src_arr = sources.detach().cpu().numpy().astype(np.int64, copy=False)
        tgt_arr = targets.detach().cpu().numpy().astype(np.int64, copy=False)
        time_arr = prediction_times.detach().cpu().numpy().astype(np.float64, copy=False)

    if isinstance(entity_id_to_idx, torch.Tensor):
        lookup_cpu = entity_id_to_idx.detach().cpu().numpy()
    elif isinstance(entity_id_to_idx, dict):
        max_node_id = max(
            int(max(src_arr)) if len(src_arr) > 0 else -1,
            int(max(tgt_arr)) if len(tgt_arr) > 0 else -1,
            int(indptr.shape[0] - 2),
        )
        lookup_cpu = np.full(max_node_id + 1, -1, dtype=np.int64)
        for node_id, emb_idx in entity_id_to_idx.items():
            node_id = int(node_id)
            if 0 <= node_id <= max_node_id:
                lookup_cpu[node_id] = int(emb_idx)
    else:
        raise ValueError("entity_id_to_idx must be a dict or a torch.Tensor (lookup)")

    max_lookup_id = int(len(lookup_cpu))
    temperature = max(float(attention_temperature), 1e-6)
    scores = torch.zeros(len(src_arr), device=device, dtype=torch.float32)

    for row_idx, (src, tgt, pred_t) in enumerate(zip(src_arr, tgt_arr, time_arr)):
        if src < 0 or src + 1 >= len(indptr) or tgt < 0 or tgt >= max_lookup_id:
            continue

        tgt_emb_idx = int(lookup_cpu[tgt])
        if tgt_emb_idx < 0:
            continue
        tgt_profile = emb_tensor[tgt_emb_idx]

        start = int(indptr[src])
        end = int(indptr[src + 1])
        cutoff = start + int(np.searchsorted(all_times[start:end], pred_t, side='left'))
        hist_node_ids = all_indices[start:cutoff]

        src_profile = None
        if len(hist_node_ids) > 0:
            valid_hist_ids = hist_node_ids[(hist_node_ids >= 0) & (hist_node_ids < max_lookup_id)]
            if len(valid_hist_ids) > 0:
                hist_emb_idx = lookup_cpu[valid_hist_ids]
                hist_emb_idx = hist_emb_idx[hist_emb_idx >= 0]
                if len(hist_emb_idx) > 0:
                    hist_idx_tensor = torch.from_numpy(hist_emb_idx).long().to(device)
                    hist_embs = emb_tensor.index_select(0, hist_idx_tensor)
                    attn_logits = hist_embs.matmul(tgt_profile) / temperature
                    attn_weights = torch.softmax(attn_logits, dim=0)
                    src_profile = torch.sum(hist_embs * attn_weights.unsqueeze(1), dim=0)
                    src_norm = torch.norm(src_profile, p=2)
                    if float(src_norm.item()) > 0.0:
                        src_profile = src_profile / src_norm

        if src_profile is None and fallback_to_raw_source and src < max_lookup_id:
            src_emb_idx = int(lookup_cpu[src])
            if src_emb_idx >= 0:
                src_profile = emb_tensor[src_emb_idx]

        if src_profile is None:
            continue

        scores[row_idx] = torch.sum(src_profile * tgt_profile)

    return scores.cpu().numpy()


def score_links_by_semantic_asymmetric_blend(neighbor_sampler,
                                             sources: np.ndarray,
                                             targets: np.ndarray,
                                             prediction_times: np.ndarray,
                                             raw_embeddings: np.ndarray = None,
                                             smoothed_embeddings: np.ndarray = None,
                                             entity_id_to_idx: dict = None,
                                             src_blend_a: float = 1.0,
                                             dst_blend_g: float = 1.0) -> np.ndarray:
    """
    Score links by asymmetric raw/smoothed blend:
      src_vec = (1-a) * raw_src + a * smooth_src
      dst_vec = (1-g) * raw_dst + g * smooth_dst
      score   = cosine(src_vec, dst_vec)
    """
    if raw_embeddings is None or smoothed_embeddings is None:
        raise ValueError("raw_embeddings and smoothed_embeddings must be provided!")

    import torch

    a = float(src_blend_a)
    g = float(dst_blend_g)

    raw_device = raw_embeddings.device if isinstance(raw_embeddings, torch.Tensor) else None
    smooth_device = smoothed_embeddings.device if isinstance(smoothed_embeddings, torch.Tensor) else None
    if raw_device is not None:
        device = raw_device
    elif smooth_device is not None:
        device = smooth_device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(raw_embeddings, np.ndarray):
        raw_tensor = torch.from_numpy(raw_embeddings).to(device)
    else:
        raw_tensor = raw_embeddings.to(device)
    if isinstance(smoothed_embeddings, np.ndarray):
        smooth_tensor = torch.from_numpy(smoothed_embeddings).to(device)
    else:
        smooth_tensor = smoothed_embeddings.to(device)

    if isinstance(sources, np.ndarray):
        src_tensor = torch.from_numpy(sources).long().to(device)
        tgt_tensor = torch.from_numpy(targets).long().to(device)
    else:
        src_tensor = sources.long().to(device)
        tgt_tensor = targets.long().to(device)

    if isinstance(entity_id_to_idx, torch.Tensor):
        lookup = entity_id_to_idx.to(device)
        max_id = lookup.size(0)
        src_mask = (src_tensor < max_id)
        tgt_mask = (tgt_tensor < max_id)
        src_indices = torch.zeros_like(src_tensor)
        tgt_indices = torch.zeros_like(tgt_tensor)
        src_indices[src_mask] = lookup[src_tensor[src_mask]]
        tgt_indices[tgt_mask] = lookup[tgt_tensor[tgt_mask]]
    elif isinstance(entity_id_to_idx, dict):
        s_list = [entity_id_to_idx.get(int(x), -1) for x in sources]
        t_list = [entity_id_to_idx.get(int(x), -1) for x in targets]
        src_indices = torch.tensor(s_list, device=device)
        tgt_indices = torch.tensor(t_list, device=device)
    else:
        raise ValueError("entity_id_to_idx must be a dict or a torch.Tensor (lookup)")

    valid_mask = (src_indices != -1) & (tgt_indices != -1)
    scores = torch.zeros(len(src_tensor), device=device, dtype=torch.float32)

    if valid_mask.any():
        v_src = src_indices[valid_mask]
        v_tgt = tgt_indices[valid_mask]

        raw_src = raw_tensor[v_src]
        raw_tgt = raw_tensor[v_tgt]
        sm_src = smooth_tensor[v_src]
        sm_tgt = smooth_tensor[v_tgt]

        src_vec = (1.0 - a) * raw_src + a * sm_src
        dst_vec = (1.0 - g) * raw_tgt + g * sm_tgt

        src_vec = src_vec / src_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        dst_vec = dst_vec / dst_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scores[valid_mask] = torch.sum(src_vec * dst_vec, dim=1)

    return scores.cpu().numpy()
