#!/usr/bin/env python3
"""
Heuristic baselines for link prediction
- Recency: Score by most recent interaction time between src and dst
- Popularity: Score by target node degree (with optional time decay)
- Past Interactions: Count of historical interactions between src and dst
- Global Recency: Last time target node had ANY interaction
- CN (Common Neighbors): Count of shared neighbors
- AA (Adamic-Adar): Weighted common neighbors by inverse log degree
- RA (Resource Allocation): Weighted common neighbors by inverse degree
- Semantic Similarity: Cosine similarity between entity text embeddings (E5-large)
Numba-optimized with CSR format (zero-copy, parallel)
"""
import numpy as np
import numba
from numba import cuda, float64, int32, int64
import math
import time
import warnings
from tqdm import tqdm

try:
    from numba.core.errors import NumbaPerformanceWarning
except Exception:  # pragma: no cover
    NumbaPerformanceWarning = None

if NumbaPerformanceWarning is not None:
    # Suppress repetitive low-occupancy warnings from small CUDA launches.
    warnings.filterwarnings(
        "ignore",
        message=r"Grid size \d+ will likely result in GPU under-utilization due to low occupancy\.",
        category=NumbaPerformanceWarning,
    )

# Check if CUDA is available
try:
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

# --- Device Helper Functions for GPU ---

@cuda.jit(device=True)
def _device_bisect_right(arr, val, start, end):
    """Binary search for temporal cutoff."""
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo

@cuda.jit(device=True)
def _get_temporal_degree(node, indptr, all_times, t):
    """Calculates degree of a node at time t."""
    start = indptr[node]
    end = indptr[node + 1]
    count_idx = _device_bisect_right(all_times, t, start, end)
    return count_idx - start

# --- Main Block-Parallel Kernel ---

@cuda.jit
def _gpu_dense_score_kernel(sources, targets, pred_times, 
                            indptr, all_indices, all_times, time_sorted_times,
                            mode, out_scores):
    """
    Block-Parallel Kernel: 
    - Grid size = Batch Size
    - Block size = 128 or 256 threads
    Each block computes the score for ONE pair (sources[bid], targets[bid]).
    """
    # 1. Identify which query this block is solving
    bid = cuda.blockIdx.x  
    tid = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    
    if bid >= sources.shape[0]:
        return

    # Initialize shared memory accumulator
    shared_score = cuda.shared.array(1, dtype=float64)
    if tid == 0:
        shared_score[0] = 0.0
    
    cuda.syncthreads()

    src = sources[bid]
    tgt = targets[bid]
    t_pred = pred_times[bid]
    
    num_nodes = len(indptr) - 1
    if src < 0 or src >= num_nodes or tgt < 0 or tgt >= num_nodes:
        return

    # Get rows from ID-SORTED CSR
    # Note: indptr is same for both (structure is same, just order permuted)
    src_start = indptr[src]
    src_end = indptr[src + 1]
    
    tgt_start = indptr[tgt]
    tgt_end = indptr[tgt + 1]
    
    src_len = src_end - src_start
    tgt_len = tgt_end - tgt_start

    if src_len > 0 and tgt_len > 0:
        # Optimization: Loop over smaller set, Binary Search in larger set
        # Since we use binary search, we want inner to be the larger one (log factor)?
        # No, iterating smaller loop is better.
        if src_len <= tgt_len:
            outer_start, outer_end = src_start, src_end
            inner_start, inner_end = tgt_start, tgt_end
        else:
            outer_start, outer_end = tgt_start, tgt_end
            inner_start, inner_end = src_start, src_end
            
        outer_len = outer_end - outer_start
        inner_len = inner_end - inner_start
        
        # Parallelize Outer Loop
        for i in range(tid, outer_len, block_dim):
            u_idx = outer_start + i
            u = all_indices[u_idx] # ID-sorted index
            u_t = all_times[u_idx] # Time for this interaction
            
            if u_t >= t_pred:
                continue # Edge too new
                
            # Binary Search for 'u' in 'inner' range
            # inner indices are sorted by ID
            # _device_bisect_right/left logic for finding ID
            
            # Custom binary search for value u
            lo = inner_start
            hi = inner_end
            found_idx = -1
            
            while lo < hi:
                mid = (lo + hi) // 2
                val = all_indices[mid]
                if val < u:
                    lo = mid + 1
                elif val > u:
                    hi = mid
                else:
                    found_idx = mid
                    break
            
            # Note: Binary search might find *one* instance.
            # If multiple interactions exist with same ID, we need to check them.
            # But duplicate IDs are adjacent.
            # We found one. Check its time. If invalid, check neighbors.
            
            if found_idx != -1:
                # Check validity of found match
                v_valid = False
                
                # Check exact match first
                if all_times[found_idx] < t_pred:
                    v_valid = True
                else:
                    # Scan backwards
                    k = found_idx - 1
                    while k >= inner_start and all_indices[k] == u:
                        if all_times[k] < t_pred:
                            v_valid = True
                            break
                        k -= 1
                    
                    if not v_valid:
                        # Scan forwards
                        k = found_idx + 1
                        while k < inner_end and all_indices[k] == u:
                            if all_times[k] < t_pred:
                                v_valid = True
                                break
                            k += 1
                            
                if v_valid:
                    val = 0.0
                    if mode == 0: # CN
                        val = 1.0
                    else:
                        if u >= 0 and u < num_nodes:
                            deg = _get_temporal_degree(u, indptr, time_sorted_times, t_pred)
                            if mode == 1 and deg > 1:
                                val = 1.0 / math.log(float(deg))
                            elif mode == 2 and deg > 0:
                                val = 1.0 / float(deg)
                    cuda.atomic.add(shared_score, 0, val)

    cuda.syncthreads()
    if tid == 0:
        out_scores[bid] = shared_score[0]

@numba.njit(cache=True)
def _binary_search_right_float64(arr, start, end, value):
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def _binary_search_left_float64(arr, start, end, value):
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.jit(nopython=True, parallel=True)
def _score_recency_kernel(sources, targets, pred_times,
                         indptr, all_indices, all_times):
    """
    Numba kernel for recency scoring (CSR format, parallel).
    Uses explicit binary search instead of slice-based np.searchsorted to avoid
    fragile interactions between prange and temporary slice views.
    """
    n = len(sources)
    scores = np.full(n, -1e15, dtype=np.float64)
    num_rows = len(indptr) - 1

    # Parallel loop over batch
    for i in numba.prange(n):
        src = sources[i]
        tgt = targets[i]
        t_limit = pred_times[i]

        if src < 0 or src >= num_rows:
            continue

        start = indptr[src]
        end = indptr[src + 1]

        if start >= end:
            continue

        cutoff = _binary_search_left_float64(all_times, start, end, t_limit)
        if cutoff <= start:
            continue

        for k in range(cutoff - 1, start - 1, -1):
            if all_indices[k] == tgt:
                scores[i] = all_times[k] - t_limit
                break

    return scores

def score_dense_gpu(neighbor_sampler, sources, targets, pred_times, mode=1):
    if not HAS_CUDA: raise RuntimeError("CUDA missing")
    batch_size = len(sources)
    if batch_size == 0: return np.zeros(0)

    # 1. Ensure ID-Sorted Graph on GPU
    if not hasattr(neighbor_sampler, '_d_csr_id_indices'):
        # Build CPU ID-sorted
        if not hasattr(neighbor_sampler, '_csr_id_sorted_indptr'):
            build_id_sorted_csr_from_neighbor_sampler(neighbor_sampler)
            
        print("Moving ID-Sorted CSR to GPU...")
        neighbor_sampler._d_csr_indptr = cuda.to_device(neighbor_sampler._csr_id_sorted_indptr)
        neighbor_sampler._d_csr_id_indices = cuda.to_device(neighbor_sampler._csr_id_sorted_indices)
        neighbor_sampler._d_csr_id_times = cuda.to_device(neighbor_sampler._csr_id_sorted_times)
        
        # Move time-sorted times for degree lookup (AA/RA)
        neighbor_sampler._d_csr_time_times = cuda.to_device(neighbor_sampler._csr_times)
    elif not hasattr(neighbor_sampler, '_d_csr_time_times'):
        # Backfill time-sorted times if only ID-sorted arrays were moved previously
        if not hasattr(neighbor_sampler, '_csr_times'):
            build_id_sorted_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._d_csr_time_times = cuda.to_device(neighbor_sampler._csr_times)

    # 2. Move Batch
    t_start = time.time()
    d_src = cuda.to_device(sources.astype(np.int64))
    d_tgt = cuda.to_device(targets.astype(np.int64))
    d_times = cuda.to_device(pred_times.astype(np.float64))
    # Invalid/unseen node IDs return from the kernel before its final write.
    # Start from deterministic zeros so those queries cannot expose uninitialized
    # device memory as heuristic scores.
    d_out = cuda.to_device(np.zeros(batch_size, dtype=np.float64))
    cuda.synchronize()
    t_transfer = time.time() - t_start

    # 3. Kernel
    threads_per_block = 128
    blocks_per_grid = batch_size 

    _gpu_dense_score_kernel[blocks_per_grid, threads_per_block](
        d_src, d_tgt, d_times,
        neighbor_sampler._d_csr_indptr,
        neighbor_sampler._d_csr_id_indices,
        neighbor_sampler._d_csr_id_times,
        neighbor_sampler._d_csr_time_times,
        mode, d_out
    )
    
    return d_out.copy_to_host()


def build_csr_from_neighbor_sampler(neighbor_sampler):
    """
    Convert DTGB's list-of-arrays format to CSR format (one-time preprocessing)

    Args:
        neighbor_sampler: DTGB NeighborSampler

    Returns:
        indptr: [num_nodes + 1] row pointers
        all_indices: [num_edges] flattened neighbor IDs
        all_times: [num_edges] flattened timestamps
    """
    print("Converting NeighborSampler to CSR format...")
    start = time.time()

    # Get the lists from DTGB sampler
    nodes_neighbor_ids = neighbor_sampler.nodes_neighbor_ids
    nodes_neighbor_times = neighbor_sampler.nodes_neighbor_times

    n_nodes = len(nodes_neighbor_ids)

    # Calculate indptr (cumulative sum of neighbor counts)
    lengths = np.array([len(nbrs) for nbrs in nodes_neighbor_ids], dtype=np.int64)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])

    # Flatten indices and timestamps
    all_indices = np.concatenate([arr for arr in nodes_neighbor_ids if len(arr) > 0]).astype(np.int64)
    all_times = np.concatenate([arr for arr in nodes_neighbor_times if len(arr) > 0]).astype(np.float64)

    elapsed = time.time() - start
    print(f"✓ CSR built in {elapsed:.2f}s")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Edges: {len(all_indices):,}")

    return indptr, all_indices, all_times


def build_directed_csr_from_edge_list(src_node_ids: np.ndarray,
                                      dst_node_ids: np.ndarray,
                                      node_interact_times: np.ndarray,
                                      num_nodes: int = None):
    """
    Build CSR adjacency for a *directed* temporal graph from edge list.

    Rows correspond to source nodes; row i stores (dst, time) for edges i -> dst,
    sorted by time (ascending) so searchsorted(time < t_pred) works.
    """
    print("Converting directed edge list to CSR format...")
    start = time.time()

    if num_nodes is None:
        num_nodes = int(max(src_node_ids.max(), dst_node_ids.max())) + 1

    # Collect outgoing neighbors per source node
    per_src_dst = [[] for _ in range(num_nodes)]
    per_src_t = [[] for _ in range(num_nodes)]

    for s, d, t in zip(src_node_ids, dst_node_ids, node_interact_times):
        s_i = int(s)
        if s_i < 0 or s_i >= num_nodes:
            continue
        per_src_dst[s_i].append(int(d))
        per_src_t[s_i].append(float(t))

    lengths = np.array([len(x) for x in per_src_dst], dtype=np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])

    total_edges = int(indptr[-1])
    all_indices = np.empty(total_edges, dtype=np.int64)
    all_times = np.empty(total_edges, dtype=np.float64)

    offset = 0
    for node_id in range(num_nodes):
        m = lengths[node_id]
        if m == 0:
            continue
        t_arr = np.asarray(per_src_t[node_id], dtype=np.float64)
        d_arr = np.asarray(per_src_dst[node_id], dtype=np.int64)
        order = np.argsort(t_arr, kind='mergesort')  # stable
        all_times[offset:offset + m] = t_arr[order]
        all_indices[offset:offset + m] = d_arr[order]
        offset += m

    elapsed = time.time() - start
    print(f"✓ Directed CSR built in {elapsed:.2f}s")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {total_edges:,}")

    return indptr, all_indices, all_times


def build_directed_id_sorted_csr_from_edge_list(src_node_ids: np.ndarray,
                                                dst_node_ids: np.ndarray,
                                                node_interact_times: np.ndarray,
                                                num_nodes: int = None):
    """
    Build directed CSR where each row is sorted by destination node ID instead of
    time. Timestamps are kept alongside edges so temporal validity can still be
    checked inside kernels.
    """
    print("Converting directed edge list to ID-sorted CSR format...")
    start = time.time()

    if num_nodes is None:
        num_nodes = int(max(src_node_ids.max(), dst_node_ids.max())) + 1

    per_src_dst = [[] for _ in range(num_nodes)]
    per_src_t = [[] for _ in range(num_nodes)]

    for s, d, t in zip(src_node_ids, dst_node_ids, node_interact_times):
        s_i = int(s)
        if s_i < 0 or s_i >= num_nodes:
            continue
        per_src_dst[s_i].append(int(d))
        per_src_t[s_i].append(float(t))

    lengths = np.array([len(x) for x in per_src_dst], dtype=np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])

    total_edges = int(indptr[-1])
    all_indices = np.empty(total_edges, dtype=np.int64)
    all_times = np.empty(total_edges, dtype=np.float64)

    offset = 0
    for node_id in range(num_nodes):
        m = lengths[node_id]
        if m == 0:
            continue
        d_arr = np.asarray(per_src_dst[node_id], dtype=np.int64)
        t_arr = np.asarray(per_src_t[node_id], dtype=np.float64)
        # _score_recency_kernel binary-searches this row by interaction time and
        # then scans backward for the requested destination.  Sorting by
        # destination here makes that cutoff undefined whenever times are not
        # already monotonic within destination order.
        order = np.argsort(t_arr, kind='mergesort')
        all_indices[offset:offset + m] = d_arr[order]
        all_times[offset:offset + m] = t_arr[order]
        offset += m

    elapsed = time.time() - start
    print(f"✓ Directed ID-sorted CSR built in {elapsed:.2f}s")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {total_edges:,}")

    return indptr, all_indices, all_times


def score_links_by_recency(neighbor_sampler,
                           sources: np.ndarray,
                           targets: np.ndarray,
                           prediction_times: np.ndarray,
                           directed: bool = False,
                           directed_src_node_ids: np.ndarray = None,
                           directed_dst_node_ids: np.ndarray = None,
                           directed_node_interact_times: np.ndarray = None) -> np.ndarray:
    """
    Score batch of links by recency (Numba-optimized)

    Args:
        neighbor_sampler: DTGB NeighborSampler
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        prediction_times: [batch_size] prediction times
        directed: if True, compute recency using only directed history (src -> dst) edges.
        directed_src_node_ids, directed_dst_node_ids, directed_node_interact_times:
            required when directed=True; used to build a directed CSR once and cache it.

    Returns:
        scores: [batch_size] last interaction time minus prediction time;
            -1e15 for pairs with no interaction strictly before prediction time.
    """
    # Build CSR format (cached after first call)
    if not directed:
        if not hasattr(neighbor_sampler, '_csr_indptr'):
            indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
            neighbor_sampler._csr_indptr = indptr
            neighbor_sampler._csr_indices = all_indices
            neighbor_sampler._csr_times = all_times
        indptr = neighbor_sampler._csr_indptr
        all_indices = neighbor_sampler._csr_indices
        all_times = neighbor_sampler._csr_times
    else:
        if directed_src_node_ids is None or directed_dst_node_ids is None or directed_node_interact_times is None:
            raise ValueError("directed=True requires directed_src_node_ids, directed_dst_node_ids, directed_node_interact_times")
        if not hasattr(neighbor_sampler, '_csr_indptr_directed'):
            indptr, all_indices, all_times = build_directed_csr_from_edge_list(
                src_node_ids=directed_src_node_ids,
                dst_node_ids=directed_dst_node_ids,
                node_interact_times=directed_node_interact_times
            )
            neighbor_sampler._csr_indptr_directed = indptr
            neighbor_sampler._csr_indices_directed = all_indices
            neighbor_sampler._csr_times_directed = all_times
        indptr = neighbor_sampler._csr_indptr_directed
        all_indices = neighbor_sampler._csr_indices_directed
        all_times = neighbor_sampler._csr_times_directed

    # Call Numba kernel (always uses row=source node; "directed" is controlled by CSR construction)
    scores = _score_recency_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        indptr,
        all_indices,
        all_times
    )

    return scores


@numba.jit(nopython=True, parallel=True)
def _score_popularity_kernel(sources, targets, pred_times, indptr, all_times, decay_lambda, mode):
    """
    Numba kernel for popularity scoring with optional time decay (CSR format, parallel)

    Args:
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        pred_times: [batch_size] prediction times
        indptr: [num_nodes + 1] CSR row pointers
        all_times: [num_edges] flattened timestamps (sorted per node)
        decay_lambda: Time decay factor (0.0 = no decay, >0 = exponential decay)
        mode: 0=target only (default), 1=source only, 2=source+target (sum)

    Returns:
        scores: [batch_size] popularity scores
                If decay_lambda=0: simple degree count
                If decay_lambda>0: Σ exp(-λ * (t_pred - t_i))
    """
    n = len(targets)
    scores = np.zeros(n, dtype=np.float64)
    num_rows = len(indptr) - 1

    # Parallel loop over batch
    for i in numba.prange(n):
        src = sources[i]
        tgt = targets[i]
        t_pred = pred_times[i]

        total_score = 0.0

        # helper inline (duplicated for numba simplicity)
        if (mode == 0 or mode == 2) and tgt >= 0 and tgt < num_rows:
            start = indptr[tgt]
            end = indptr[tgt + 1]
            if start != end:
                history_times = all_times[start:end]
                cutoff = np.searchsorted(history_times, t_pred)
                if cutoff > 0:
                    if decay_lambda == 0.0:
                        total_score += float(cutoff)
                    else:
                        total = 0.0
                        for j in range(start, start + cutoff):
                            delta_t = t_pred - all_times[j]
                            total += np.exp(-decay_lambda * delta_t)
                        total_score += total

        if (mode == 1 or mode == 2) and src >= 0 and src < num_rows:
            start = indptr[src]
            end = indptr[src + 1]
            if start != end:
                history_times = all_times[start:end]
                cutoff = np.searchsorted(history_times, t_pred)
                if cutoff > 0:
                    if decay_lambda == 0.0:
                        total_score += float(cutoff)
                    else:
                        total = 0.0
                        for j in range(start, start + cutoff):
                            delta_t = t_pred - all_times[j]
                            total += np.exp(-decay_lambda * delta_t)
                        total_score += total

        scores[i] = total_score

    return scores


def score_links_by_popularity(neighbor_sampler,
                                sources: np.ndarray,
                                targets: np.ndarray,
                                prediction_times: np.ndarray,
                                decay: float = 0.0,
                                mode: str = 'target') -> np.ndarray:
    """
    Score batch of links by target node popularity (degree with optional time decay)

    Args:
        neighbor_sampler: DTGB NeighborSampler
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        prediction_times: [batch_size] prediction times
        decay: Time decay factor (0.0 = no decay, >0 = exponential decay)
        mode: 'target' (default), 'source', or 'sum' (source+target)

    Returns:
        scores: [batch_size] popularity scores
    """
    # Build CSR format (cached after first call)
    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = all_indices
        neighbor_sampler._csr_times = all_times

    # One-time diagnostic on decay scale
    if decay > 0.0 and not hasattr(neighbor_sampler, '_popularity_decay_logged'):
        hist_times = neighbor_sampler._csr_times
        pred_min, pred_median, pred_max = np.min(prediction_times), np.median(prediction_times), np.max(prediction_times)
        hist_min, hist_median, hist_max = np.min(hist_times), np.median(hist_times), np.max(hist_times)
        span = pred_max - hist_min
        mid_delta = max(0.0, pred_median - hist_median)
        print("[Popularity decay] decay=%.3g hist[min/med/max]=[%.2f/%.2f/%.2f] "
              "pred[min/med/max]=[%.2f/%.2f/%.2f] span=%.2f "
              "exp(-λ*medianΔ)=%.3e exp(-λ*span)=%.3e" %
              (decay, hist_min, hist_median, hist_max,
               pred_min, pred_median, pred_max,
               span,
               np.exp(-decay * mid_delta) if mid_delta > 0 else 1.0,
               np.exp(-decay * span) if span > 0 else 1.0))
        neighbor_sampler._popularity_decay_logged = True

    # Call Numba kernel
    scores = _score_popularity_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr,
        neighbor_sampler._csr_times,
        float(decay),
        0 if mode == 'target' else (1 if mode == 'source' else 2)
    )

    return scores


@numba.jit(nopython=True, parallel=True)
def _score_recent_degree_kernel(sources, targets, pred_times, indptr, all_times, window, mode):
    """
    Count interactions within the hard recent window [t_pred - window, t_pred).

    mode: 0=target only, 1=source only, 2=source+target
    """
    n = len(targets)
    scores = np.zeros(n, dtype=np.float64)
    num_rows = len(indptr) - 1

    for i in numba.prange(n):
        src = sources[i]
        tgt = targets[i]
        t_pred = pred_times[i]
        t_start = t_pred - window

        total_score = 0.0

        if (mode == 0 or mode == 2) and tgt >= 0 and tgt < num_rows:
            start = indptr[tgt]
            end = indptr[tgt + 1]
            if start != end:
                history_times = all_times[start:end]
                hi = np.searchsorted(history_times, t_pred)
                lo = np.searchsorted(history_times, t_start)
                if hi > lo:
                    total_score += float(hi - lo)

        if (mode == 1 or mode == 2) and src >= 0 and src < num_rows:
            start = indptr[src]
            end = indptr[src + 1]
            if start != end:
                history_times = all_times[start:end]
                hi = np.searchsorted(history_times, t_pred)
                lo = np.searchsorted(history_times, t_start)
                if hi > lo:
                    total_score += float(hi - lo)

        scores[i] = total_score

    return scores


def score_links_by_recent_degree(neighbor_sampler,
                                 sources: np.ndarray,
                                 targets: np.ndarray,
                                 prediction_times: np.ndarray,
                                 window: float,
                                 mode: str = 'target') -> np.ndarray:
    """
    Score by recent degree: number of interactions in a hard lookback window.

    Args:
        neighbor_sampler: DTGB NeighborSampler
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        prediction_times: [batch_size] prediction times
        window: hard temporal lookback window
        mode: 'target' (default), 'source', or 'sum' (source+target)
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = all_indices
        neighbor_sampler._csr_times = all_times

    mode_map = {
        'target': 0,
        'source': 1,
        'sum': 2,
    }
    if mode not in mode_map:
        raise ValueError(f"Unknown mode '{mode}', expected one of {list(mode_map.keys())}")

    return _score_recent_degree_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr,
        neighbor_sampler._csr_times,
        float(window),
        mode_map[mode],
    )


@numba.jit(nopython=True, parallel=True)
def _score_past_interactions_kernel(sources, targets, pred_times,
                                     indptr, all_indices, all_times):
    """
    Numba kernel for past interactions counting (CSR format, parallel)

    Args:
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        pred_times: [batch_size] prediction times
        indptr: [num_nodes + 1] CSR row pointers
        all_indices: [num_edges] flattened neighbor IDs
        all_times: [num_edges] flattened timestamps (sorted per node)

    Returns:
        scores: [batch_size] count of past interactions between src and dst
    """
    n = len(sources)
    scores = np.zeros(n, dtype=np.float64)
    num_rows = len(indptr) - 1

    # Parallel loop over batch
    for i in numba.prange(n):
        src = sources[i]
        tgt = targets[i]
        t_limit = pred_times[i]

        if src < 0 or src >= num_rows:
            scores[i] = 0.0
            continue

        # Get slice coordinates for source node
        start = indptr[src]
        end = indptr[src + 1]

        if start == end:
            scores[i] = 0.0  # No neighbors
            continue

        # Find first index with time >= t_limit using explicit binary search on
        # the base array to avoid slice/searchsorted issues under prange.
        cutoff = _binary_search_left_float64(all_times, start, end, t_limit)

        if cutoff <= start:
            scores[i] = 0.0  # No history before t_limit
            continue

        # Count how many times target appears in history
        count = 0
        for j in range(start, cutoff):
            if all_indices[j] == tgt:
                count += 1

        scores[i] = float(count)

    return scores


def score_links_by_past_interactions(neighbor_sampler,
                                      sources: np.ndarray,
                                      targets: np.ndarray,
                                      prediction_times: np.ndarray) -> np.ndarray:
    """
    Score batch of links by count of past interactions between src and dst

    Args:
        neighbor_sampler: DTGB NeighborSampler
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        prediction_times: [batch_size] prediction times

    Returns:
        scores: [batch_size] count of past interactions
    """
    # Build CSR format (cached after first call)
    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = all_indices
        neighbor_sampler._csr_times = all_times

    # Call Numba kernel
    scores = _score_past_interactions_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr,
        neighbor_sampler._csr_indices,
        neighbor_sampler._csr_times
    )

    return scores


@numba.jit(nopython=True, parallel=True)
def _score_global_recency_kernel(targets, pred_times, indptr, all_times):
    """
    Numba kernel for global recency scoring (CSR format, parallel)

    Args:
        targets: [batch_size] target node IDs
        pred_times: [batch_size] prediction times
        indptr: [num_nodes + 1] CSR row pointers
        all_times: [num_edges] flattened timestamps (sorted per node)

    Returns:
        scores: [batch_size] most recent interaction time of target (-1 if never)
    """
    n = len(targets)
    scores = np.full(n, -1e15, dtype=np.float64)
    num_rows = len(indptr) - 1

    # Parallel loop over batch
    for i in numba.prange(n):
        tgt = targets[i]
        t_limit = pred_times[i]

        if tgt < 0 or tgt >= num_rows:
            continue

        # Get slice coordinates for target node
        start = indptr[tgt]
        end = indptr[tgt + 1]

        if start == end:
            scores[i] = -1e15  # No neighbors
            continue

        # Binary search: find last interaction before t_limit
        history_times = all_times[start:end]
        cutoff = np.searchsorted(history_times, t_limit)

        if cutoff == 0:
            scores[i] = -1e15  # No history before t_limit
        else:
            # Return the most recent timestamp as negative delta
            scores[i] = all_times[start + cutoff - 1] - t_limit

    return scores


def score_links_by_global_recency(neighbor_sampler,
                                   sources: np.ndarray,
                                   targets: np.ndarray,
                                   prediction_times: np.ndarray) -> np.ndarray:
    """
    Score batch of links by global recency (last time target had ANY interaction)

    Args:
        neighbor_sampler: DTGB NeighborSampler
        sources: [batch_size] source node IDs (not used, for API consistency)
        targets: [batch_size] target node IDs
        prediction_times: [batch_size] prediction times

    Returns:
        scores: [batch_size] most recent interaction time of target (-1 if never)
    """
    # Build CSR format (cached after first call)
    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, all_indices, all_times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = all_indices
        neighbor_sampler._csr_times = all_times

    # Call Numba kernel
    scores = _score_global_recency_kernel(
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr,
        neighbor_sampler._csr_times
    )

    return scores


@numba.jit(nopython=True, inline='always')
def _temporal_overlap_count_id_sorted(row_a, row_b, indptr, indices, times, t_limit):
    """
    Count shared neighbors between two ID-sorted CSR rows when each side has at
    least one interaction with that neighbor before t_limit. Duplicate neighbors
    within a row count once.
    """
    a_start = indptr[row_a]
    a_end = indptr[row_a + 1]
    b_start = indptr[row_b]
    b_end = indptr[row_b + 1]

    p_a = a_start
    p_b = b_start
    overlap = 0

    while p_a < a_end and p_b < b_end:
        a_node = indices[p_a]
        b_node = indices[p_b]

        if a_node < b_node:
            current = a_node
            while p_a < a_end and indices[p_a] == current:
                p_a += 1
        elif b_node < a_node:
            current = b_node
            while p_b < b_end and indices[p_b] == current:
                p_b += 1
        else:
            shared = a_node
            valid_a = False
            while p_a < a_end and indices[p_a] == shared:
                if times[p_a] < t_limit:
                    valid_a = True
                p_a += 1

            valid_b = False
            while p_b < b_end and indices[p_b] == shared:
                if times[p_b] < t_limit:
                    valid_b = True
                p_b += 1

            if valid_a and valid_b:
                overlap += 1

    return overlap


@numba.jit(nopython=True, parallel=True)
def _score_itemcf_cosine_kernel(
    sources,
    targets,
    pred_times,
    user_item_indptr,
    user_item_indices,
    user_item_times,
    item_user_indptr,
    item_user_indices,
    item_user_times,
):
    """
    Item-based CF score for bipartite user-item graphs:
      score(u, i, t) = sum_{j in H(u,t)} co_users(i,j) / sqrt(pop(i,t) * pop(j,t))
    where all counts only use interactions strictly before t.
    """
    n = len(sources)
    num_user_rows = len(user_item_indptr) - 1
    num_item_rows = len(item_user_indptr) - 1
    scores = np.zeros(n, dtype=np.float64)

    for idx in numba.prange(n):
        src = sources[idx]
        tgt = targets[idx]
        t_limit = pred_times[idx]

        if src < 0 or src >= num_user_rows or tgt < 0 or tgt >= num_item_rows:
            continue

        tgt_deg = _temporal_degree_id_sorted(
            tgt, item_user_indptr, item_user_indices, item_user_times, t_limit
        )
        if tgt_deg <= 0:
            continue

        start = user_item_indptr[src]
        end = user_item_indptr[src + 1]
        if start >= end:
            continue

        score = 0.0
        for pos in range(start, end):
            hist_time = user_item_times[pos]
            if hist_time >= t_limit:
                continue

            hist_item = user_item_indices[pos]
            if hist_item < 0 or hist_item >= num_item_rows:
                continue

            hist_deg = _temporal_degree_id_sorted(
                hist_item, item_user_indptr, item_user_indices, item_user_times, t_limit
            )
            if hist_deg <= 0:
                continue

            overlap = _temporal_overlap_count_id_sorted(
                tgt, hist_item, item_user_indptr, item_user_indices, item_user_times, t_limit
            )
            if overlap <= 0:
                continue

            score += float(overlap) / math.sqrt(float(tgt_deg) * float(hist_deg))

        scores[idx] = score

    return scores


def score_links_by_itemcf_cosine(neighbor_sampler,
                                 sources: np.ndarray,
                                 targets: np.ndarray,
                                 prediction_times: np.ndarray,
                                 directed_src_node_ids: np.ndarray,
                                 directed_dst_node_ids: np.ndarray,
                                 directed_node_interact_times: np.ndarray) -> np.ndarray:
    """
    Item-based collaborative filtering score for directed user->item interaction
    graphs. For query (source=user, target=item, time=t), this sums cosine-
    normalized co-user overlap between the candidate item and the user's
    historical items observed strictly before t.
    """
    if directed_src_node_ids is None or directed_dst_node_ids is None or directed_node_interact_times is None:
        raise ValueError(
            "ItemCF cosine requires directed_src_node_ids, directed_dst_node_ids, "
            "and directed_node_interact_times."
        )

    if not hasattr(neighbor_sampler, '_csr_indptr_user_item_directed'):
        num_nodes = int(max(directed_src_node_ids.max(), directed_dst_node_ids.max())) + 1

        ui_indptr, ui_indices, ui_times = build_directed_csr_from_edge_list(
            src_node_ids=directed_src_node_ids,
            dst_node_ids=directed_dst_node_ids,
            node_interact_times=directed_node_interact_times,
            num_nodes=num_nodes,
        )
        neighbor_sampler._csr_indptr_user_item_directed = ui_indptr
        neighbor_sampler._csr_indices_user_item_directed = ui_indices
        neighbor_sampler._csr_times_user_item_directed = ui_times

        iu_indptr, iu_indices, iu_times = build_directed_id_sorted_csr_from_edge_list(
            src_node_ids=directed_dst_node_ids,
            dst_node_ids=directed_src_node_ids,
            node_interact_times=directed_node_interact_times,
            num_nodes=num_nodes,
        )
        neighbor_sampler._csr_indptr_item_user_directed = iu_indptr
        neighbor_sampler._csr_indices_item_user_directed = iu_indices
        neighbor_sampler._csr_times_item_user_directed = iu_times

    return _score_itemcf_cosine_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr_user_item_directed,
        neighbor_sampler._csr_indices_user_item_directed,
        neighbor_sampler._csr_times_user_item_directed,
        neighbor_sampler._csr_indptr_item_user_directed,
        neighbor_sampler._csr_indices_item_user_directed,
        neighbor_sampler._csr_times_item_user_directed,
    )


@numba.njit(cache=True)
def _build_grouped_markov_transition_events(sorted_users, sorted_items, sorted_times):
    """
    Build first-order item transitions between adjacent, distinct timestamp
    buckets for each user.  Every unique item in the previous bucket connects
    to every unique item in the next bucket.  Inputs must be sorted by
    (user, time, item).
    """
    n = len(sorted_users)
    total = 0
    user_start = 0

    while user_start < n:
        user = sorted_users[user_start]
        user_end = user_start + 1
        while user_end < n and sorted_users[user_end] == user:
            user_end += 1

        prev_start = user_start
        prev_end = prev_start + 1
        while prev_end < user_end and sorted_times[prev_end] == sorted_times[prev_start]:
            prev_end += 1

        cur_start = prev_end
        while cur_start < user_end:
            cur_end = cur_start + 1
            while cur_end < user_end and sorted_times[cur_end] == sorted_times[cur_start]:
                cur_end += 1

            prev_unique = 0
            for pos in range(prev_start, prev_end):
                if pos == prev_start or sorted_items[pos] != sorted_items[pos - 1]:
                    prev_unique += 1

            cur_unique = 0
            for pos in range(cur_start, cur_end):
                if pos == cur_start or sorted_items[pos] != sorted_items[pos - 1]:
                    cur_unique += 1

            total += prev_unique * cur_unique
            prev_start = cur_start
            prev_end = cur_end
            cur_start = cur_end

        user_start = user_end

    previous_items = np.empty(total, dtype=np.int64)
    next_items = np.empty(total, dtype=np.int64)
    transition_times = np.empty(total, dtype=np.float64)
    write_pos = 0
    user_start = 0

    while user_start < n:
        user = sorted_users[user_start]
        user_end = user_start + 1
        while user_end < n and sorted_users[user_end] == user:
            user_end += 1

        prev_start = user_start
        prev_end = prev_start + 1
        while prev_end < user_end and sorted_times[prev_end] == sorted_times[prev_start]:
            prev_end += 1

        cur_start = prev_end
        while cur_start < user_end:
            cur_end = cur_start + 1
            while cur_end < user_end and sorted_times[cur_end] == sorted_times[cur_start]:
                cur_end += 1

            for prev_pos in range(prev_start, prev_end):
                if prev_pos > prev_start and sorted_items[prev_pos] == sorted_items[prev_pos - 1]:
                    continue
                previous_item = sorted_items[prev_pos]
                for cur_pos in range(cur_start, cur_end):
                    if cur_pos > cur_start and sorted_items[cur_pos] == sorted_items[cur_pos - 1]:
                        continue
                    previous_items[write_pos] = previous_item
                    next_items[write_pos] = sorted_items[cur_pos]
                    transition_times[write_pos] = sorted_times[cur_start]
                    write_pos += 1

            prev_start = cur_start
            prev_end = cur_end
            cur_start = cur_end

        user_start = user_end

    return previous_items, next_items, transition_times


@numba.njit(cache=True, inline='always')
def _binary_search_int64(arr, value):
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(arr) and arr[lo] == value:
        return lo
    return -1


@numba.njit(cache=True, parallel=True)
def _score_markov_transition_kernel(
    sources,
    targets,
    pred_times,
    num_nodes,
    user_indptr,
    user_items,
    user_times,
    pair_keys,
    pair_indptr,
    pair_times,
    outgoing_indptr,
    outgoing_times,
):
    """
    Leakage-safe first-order Markov score.

    For each query, use the user's most recent timestamp bucket strictly before
    prediction time.  The score is the mean row-normalized transition
    probability from that bucket's unique items to the candidate item, using
    only transition events strictly before prediction time.
    """
    n = len(sources)
    scores = np.zeros(n, dtype=np.float64)
    num_user_rows = len(user_indptr) - 1

    for idx in numba.prange(n):
        source = sources[idx]
        target = targets[idx]
        prediction_time = pred_times[idx]
        if source < 0 or source >= num_user_rows or target < 0 or target >= num_nodes:
            continue

        history_start = user_indptr[source]
        history_end = user_indptr[source + 1]
        cutoff = _binary_search_left_float64(
            user_times, history_start, history_end, prediction_time
        )
        if cutoff <= history_start:
            continue

        latest_time = user_times[cutoff - 1]
        bucket_start = _binary_search_left_float64(
            user_times, history_start, cutoff, latest_time
        )
        score_sum = 0.0
        bucket_size = 0
        previous_item_seen = -1

        for pos in range(bucket_start, cutoff):
            previous_item = user_items[pos]
            if previous_item == previous_item_seen:
                continue
            previous_item_seen = previous_item
            bucket_size += 1

            outgoing_start = outgoing_indptr[previous_item]
            outgoing_end = outgoing_indptr[previous_item + 1]
            outgoing_cutoff = _binary_search_left_float64(
                outgoing_times, outgoing_start, outgoing_end, prediction_time
            )
            outgoing_count = outgoing_cutoff - outgoing_start
            if outgoing_count <= 0:
                continue

            pair_key = previous_item * num_nodes + target
            pair_idx = _binary_search_int64(pair_keys, pair_key)
            if pair_idx < 0:
                continue
            pair_start = pair_indptr[pair_idx]
            pair_end = pair_indptr[pair_idx + 1]
            pair_cutoff = _binary_search_left_float64(
                pair_times, pair_start, pair_end, prediction_time
            )
            transition_count = pair_cutoff - pair_start
            if transition_count > 0:
                score_sum += float(transition_count) / float(outgoing_count)

        if bucket_size > 0:
            scores[idx] = score_sum / float(bucket_size)

    return scores


def _build_markov_transition_cache(
    directed_src_node_ids: np.ndarray,
    directed_dst_node_ids: np.ndarray,
    directed_node_interact_times: np.ndarray,
):
    sources = np.asarray(directed_src_node_ids, dtype=np.int64).reshape(-1)
    items = np.asarray(directed_dst_node_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(directed_node_interact_times, dtype=np.float64).reshape(-1)
    if not (len(sources) == len(items) == len(times)):
        raise ValueError("Markov transition history arrays must have matching lengths.")
    if len(sources) == 0:
        raise ValueError("Markov transition history cannot be empty.")

    num_nodes = int(max(sources.max(), items.max())) + 1
    original_order = np.arange(len(sources), dtype=np.int64)
    order = np.lexsort((original_order, items, times, sources))
    sorted_users = sources[order]
    sorted_items = items[order]
    sorted_times = times[order]

    user_counts = np.bincount(sorted_users, minlength=num_nodes).astype(np.int64)
    user_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(user_counts, out=user_indptr[1:])

    previous_items, next_items, transition_times = _build_grouped_markov_transition_events(
        sorted_users,
        sorted_items,
        sorted_times,
    )
    if len(previous_items) == 0:
        return {
            "num_nodes": num_nodes,
            "user_indptr": user_indptr,
            "user_items": sorted_items,
            "user_times": sorted_times,
            "pair_keys": np.empty(0, dtype=np.int64),
            "pair_indptr": np.zeros(1, dtype=np.int64),
            "pair_times": np.empty(0, dtype=np.float64),
            "outgoing_indptr": np.zeros(num_nodes + 1, dtype=np.int64),
            "outgoing_times": np.empty(0, dtype=np.float64),
        }

    pair_key_values = previous_items * np.int64(num_nodes) + next_items
    pair_order = np.lexsort((transition_times, pair_key_values))
    sorted_pair_keys = pair_key_values[pair_order]
    pair_times = transition_times[pair_order]
    is_new_pair = np.empty(len(sorted_pair_keys), dtype=bool)
    is_new_pair[0] = True
    is_new_pair[1:] = sorted_pair_keys[1:] != sorted_pair_keys[:-1]
    pair_starts = np.flatnonzero(is_new_pair).astype(np.int64)
    pair_keys = sorted_pair_keys[pair_starts]
    pair_indptr = np.empty(len(pair_starts) + 1, dtype=np.int64)
    pair_indptr[:-1] = pair_starts
    pair_indptr[-1] = len(sorted_pair_keys)

    outgoing_order = np.lexsort((transition_times, previous_items))
    sorted_previous_items = previous_items[outgoing_order]
    outgoing_times = transition_times[outgoing_order]
    outgoing_counts = np.bincount(
        sorted_previous_items, minlength=num_nodes
    ).astype(np.int64)
    outgoing_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(outgoing_counts, out=outgoing_indptr[1:])

    print(
        "Built grouped Markov transition cache: "
        f"events={len(previous_items):,}, pairs={len(pair_keys):,}, nodes={num_nodes:,}"
    )
    return {
        "num_nodes": num_nodes,
        "user_indptr": user_indptr,
        "user_items": sorted_items,
        "user_times": sorted_times,
        "pair_keys": pair_keys,
        "pair_indptr": pair_indptr,
        "pair_times": pair_times,
        "outgoing_indptr": outgoing_indptr,
        "outgoing_times": outgoing_times,
    }


def score_links_by_markov_transition(
    neighbor_sampler,
    sources: np.ndarray,
    targets: np.ndarray,
    prediction_times: np.ndarray,
    directed_src_node_ids: np.ndarray,
    directed_dst_node_ids: np.ndarray,
    directed_node_interact_times: np.ndarray,
) -> np.ndarray:
    """
    First-order Markov recommendation score for directed user->item histories.

    Repeated interactions at the same timestamp are treated as an unordered
    basket.  Transitions are built between adjacent timestamp baskets, and all
    query-time lookups use strict ``transition_time < prediction_time`` cutoffs.
    """
    cache_attr = "_grouped_markov_transition_cache"
    if not hasattr(neighbor_sampler, cache_attr):
        setattr(
            neighbor_sampler,
            cache_attr,
            _build_markov_transition_cache(
                directed_src_node_ids=directed_src_node_ids,
                directed_dst_node_ids=directed_dst_node_ids,
                directed_node_interact_times=directed_node_interact_times,
            ),
        )
    cache = getattr(neighbor_sampler, cache_attr)
    return _score_markov_transition_kernel(
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(prediction_times, dtype=np.float64),
        int(cache["num_nodes"]),
        cache["user_indptr"],
        cache["user_items"],
        cache["user_times"],
        cache["pair_keys"],
        cache["pair_indptr"],
        cache["pair_times"],
        cache["outgoing_indptr"],
        cache["outgoing_times"],
    )


@numba.jit(nopython=True, parallel=True)
def _score_usercf_cosine_kernel(
    sources,
    targets,
    pred_times,
    user_item_indptr,
    user_item_indices,
    user_item_times,
    item_user_indptr,
    item_user_indices,
    item_user_times,
    user_item_id_indptr,
    user_item_id_indices,
    user_item_id_times,
):
    """
    User-based CF score for bipartite user-item graphs:
      score(u, i, t) = sum_{v in U(i,t)} co_items(u,v) / sqrt(deg(u,t) * deg(v,t))
    where all counts only use interactions strictly before t.
    """
    n = len(sources)
    num_user_rows = len(user_item_indptr) - 1
    num_item_rows = len(item_user_indptr) - 1
    scores = np.zeros(n, dtype=np.float64)

    for idx in numba.prange(n):
        src = sources[idx]
        tgt = targets[idx]
        t_limit = pred_times[idx]

        if src < 0 or src >= num_user_rows or tgt < 0 or tgt >= num_item_rows:
            continue

        src_deg = _temporal_degree_id_sorted(
            src, user_item_id_indptr, user_item_id_indices, user_item_id_times, t_limit
        )
        if src_deg <= 0:
            continue

        start = item_user_indptr[tgt]
        end = item_user_indptr[tgt + 1]
        if start >= end:
            continue

        score = 0.0
        for pos in range(start, end):
            hist_time = item_user_times[pos]
            if hist_time >= t_limit:
                continue

            hist_user = item_user_indices[pos]
            if hist_user < 0 or hist_user >= num_user_rows:
                continue

            hist_deg = _temporal_degree_id_sorted(
                hist_user, user_item_id_indptr, user_item_id_indices, user_item_id_times, t_limit
            )
            if hist_deg <= 0:
                continue

            overlap = _temporal_overlap_count_id_sorted(
                src, hist_user, user_item_id_indptr, user_item_id_indices, user_item_id_times, t_limit
            )
            if overlap <= 0:
                continue

            score += float(overlap) / math.sqrt(float(src_deg) * float(hist_deg))

        scores[idx] = score

    return scores


def score_links_by_usercf_cosine(neighbor_sampler,
                                 sources: np.ndarray,
                                 targets: np.ndarray,
                                 prediction_times: np.ndarray,
                                 directed_src_node_ids: np.ndarray,
                                 directed_dst_node_ids: np.ndarray,
                                 directed_node_interact_times: np.ndarray) -> np.ndarray:
    """
    User-based collaborative filtering score for directed user->item interaction
    graphs. For query (source=user, target=item, time=t), this sums cosine-
    normalized co-item overlap between the source user and historical users who
    interacted with the candidate item strictly before t.
    """
    if directed_src_node_ids is None or directed_dst_node_ids is None or directed_node_interact_times is None:
        raise ValueError(
            "UserCF cosine requires directed_src_node_ids, directed_dst_node_ids, "
            "and directed_node_interact_times."
        )

    if not hasattr(neighbor_sampler, '_csr_indptr_user_item_directed'):
        num_nodes = int(max(directed_src_node_ids.max(), directed_dst_node_ids.max())) + 1

        ui_indptr, ui_indices, ui_times = build_directed_csr_from_edge_list(
            src_node_ids=directed_src_node_ids,
            dst_node_ids=directed_dst_node_ids,
            node_interact_times=directed_node_interact_times,
            num_nodes=num_nodes,
        )
        neighbor_sampler._csr_indptr_user_item_directed = ui_indptr
        neighbor_sampler._csr_indices_user_item_directed = ui_indices
        neighbor_sampler._csr_times_user_item_directed = ui_times

        iu_indptr, iu_indices, iu_times = build_directed_id_sorted_csr_from_edge_list(
            src_node_ids=directed_dst_node_ids,
            dst_node_ids=directed_src_node_ids,
            node_interact_times=directed_node_interact_times,
            num_nodes=num_nodes,
        )
        neighbor_sampler._csr_indptr_item_user_directed = iu_indptr
        neighbor_sampler._csr_indices_item_user_directed = iu_indices
        neighbor_sampler._csr_times_item_user_directed = iu_times

    if not hasattr(neighbor_sampler, '_csr_indptr_user_item_directed_id_sorted'):
        num_nodes = int(max(directed_src_node_ids.max(), directed_dst_node_ids.max())) + 1
        ui_id_indptr, ui_id_indices, ui_id_times = build_directed_id_sorted_csr_from_edge_list(
            src_node_ids=directed_src_node_ids,
            dst_node_ids=directed_dst_node_ids,
            node_interact_times=directed_node_interact_times,
            num_nodes=num_nodes,
        )
        neighbor_sampler._csr_indptr_user_item_directed_id_sorted = ui_id_indptr
        neighbor_sampler._csr_indices_user_item_directed_id_sorted = ui_id_indices
        neighbor_sampler._csr_times_user_item_directed_id_sorted = ui_id_times

    return _score_usercf_cosine_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr_user_item_directed,
        neighbor_sampler._csr_indices_user_item_directed,
        neighbor_sampler._csr_times_user_item_directed,
        neighbor_sampler._csr_indptr_item_user_directed,
        neighbor_sampler._csr_indices_item_user_directed,
        neighbor_sampler._csr_times_item_user_directed,
        neighbor_sampler._csr_indptr_user_item_directed_id_sorted,
        neighbor_sampler._csr_indices_user_item_directed_id_sorted,
        neighbor_sampler._csr_times_user_item_directed_id_sorted,
    )


@numba.jit(nopython=True, parallel=True)
def _score_personalized_location_kernel(
    sources,
    targets,
    pred_times,
    user_item_indptr,
    user_item_indices,
    user_item_times,
    node_location_ids,
):
    """
    Personalized location preference:
      score(u, i, t) = count(history items of u before t with location(i)) / |H(u,t)|
    """
    n = len(sources)
    num_user_rows = len(user_item_indptr) - 1
    num_location_rows = len(node_location_ids)
    scores = np.zeros(n, dtype=np.float64)

    for idx in numba.prange(n):
        src = sources[idx]
        tgt = targets[idx]
        t_limit = pred_times[idx]

        if src < 0 or src >= num_user_rows or tgt < 0 or tgt >= num_location_rows:
            continue

        target_location = node_location_ids[tgt]
        if target_location < 0:
            continue

        start = user_item_indptr[src]
        end = user_item_indptr[src + 1]
        if start >= end:
            continue

        same_location_count = 0
        history_count = 0
        for pos in range(start, end):
            if user_item_times[pos] >= t_limit:
                continue

            hist_item = user_item_indices[pos]
            if hist_item < 0 or hist_item >= num_location_rows:
                continue

            hist_location = node_location_ids[hist_item]
            if hist_location < 0:
                continue

            history_count += 1
            if hist_location == target_location:
                same_location_count += 1

        if history_count > 0:
            scores[idx] = float(same_location_count) / float(history_count)

    return scores


def score_links_by_personalized_location(neighbor_sampler,
                                         sources: np.ndarray,
                                         targets: np.ndarray,
                                         prediction_times: np.ndarray,
                                         directed_src_node_ids: np.ndarray,
                                         directed_dst_node_ids: np.ndarray,
                                         directed_node_interact_times: np.ndarray,
                                         node_location_ids: np.ndarray) -> np.ndarray:
    """
    Score directed user->item links by the source user's historical preference
    for the candidate item's coarse location.
    """
    if directed_src_node_ids is None or directed_dst_node_ids is None or directed_node_interact_times is None:
        raise ValueError(
            "Personalized location requires directed_src_node_ids, directed_dst_node_ids, "
            "and directed_node_interact_times."
        )
    if node_location_ids is None:
        raise ValueError("Personalized location requires node_location_ids.")

    if not hasattr(neighbor_sampler, '_csr_indptr_user_item_directed'):
        num_nodes = int(max(directed_src_node_ids.max(), directed_dst_node_ids.max())) + 1

        ui_indptr, ui_indices, ui_times = build_directed_csr_from_edge_list(
            src_node_ids=directed_src_node_ids,
            dst_node_ids=directed_dst_node_ids,
            node_interact_times=directed_node_interact_times,
            num_nodes=num_nodes,
        )
        neighbor_sampler._csr_indptr_user_item_directed = ui_indptr
        neighbor_sampler._csr_indices_user_item_directed = ui_indices
        neighbor_sampler._csr_times_user_item_directed = ui_times

    return _score_personalized_location_kernel(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        neighbor_sampler._csr_indptr_user_item_directed,
        neighbor_sampler._csr_indices_user_item_directed,
        neighbor_sampler._csr_times_user_item_directed,
        np.asarray(node_location_ids, dtype=np.int64),
    )


@numba.jit(nopython=True, inline='always')
def _temporal_degree(node, indptr, all_times, t_limit):
    """Get node degree at time t_limit (inline helper)"""
    start, end = indptr[node], indptr[node + 1]
    if start >= end:
        return 0
    return np.searchsorted(all_times[start:end], t_limit)


@numba.jit(nopython=True, inline='always')
def _get_temporal_neighbors(node, indptr, all_indices, all_times, t_limit):
    """Get neighbor indices before t_limit (inline helper)"""
    start, end = indptr[node], indptr[node + 1]
    if start >= end:
        return all_indices[0:0]  # Empty slice
    cutoff = start + np.searchsorted(all_times[start:end], t_limit)
    return all_indices[start:cutoff]


@numba.jit(nopython=True, parallel=True)
def _score_common_neighbors_kernel(sources, targets, pred_times,
                                    indptr, all_indices, all_times,
                                    mode):
    """
    Clean CN/AA/RA scoring with temporal degrees

    Args:
        sources: [batch_size] source node IDs
        targets: [batch_size] target node IDs
        pred_times: [batch_size] prediction times
        indptr: [num_nodes + 1] CSR row pointers
        all_indices: [num_edges] flattened neighbor IDs
        all_times: [num_edges] flattened timestamps (sorted per node)
        mode: 0=CN, 1=AA (Adamic-Adar), 2=RA (Resource Allocation)

    Returns:
        scores: [batch_size] common neighbor scores with temporal degrees
    """
    n = len(sources)
    num_nodes = len(indptr) - 1
    scores = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        src, tgt, t = sources[i], targets[i], pred_times[i]

        if src < 0 or src >= num_nodes or tgt < 0 or tgt >= num_nodes:
            continue

        # Get temporal neighbors
        src_nb = _get_temporal_neighbors(src, indptr, all_indices, all_times, t)
        tgt_nb = _get_temporal_neighbors(tgt, indptr, all_indices, all_times, t)

        if len(src_nb) == 0 or len(tgt_nb) == 0:
            continue

        # Intersection: use smaller set as hash
        if len(src_nb) <= len(tgt_nb):
            small_set = set(src_nb)
            large_arr = tgt_nb
        else:
            small_set = set(tgt_nb)
            large_arr = src_nb

        score = 0.0
        for neighbor in large_arr:
            if neighbor in small_set:
                if mode == 0:  # CN
                    score += 1.0
                else:
                    # Temporal degree of common neighbor
                    deg = _temporal_degree(neighbor, indptr, all_times, t)
                    if mode == 1 and deg > 1:  # AA
                        score += 1.0 / np.log(float(deg))
                    elif mode == 2 and deg > 0:  # RA
                        score += 1.0 / float(deg)

        scores[i] = score

    return scores


@numba.jit(nopython=True, parallel=True)
def _sort_csr_rows(indptr, src_indices, src_times, out_indices, out_times):
    n_nodes = len(indptr) - 1
    for i in numba.prange(n_nodes):
        start = indptr[i]
        end = indptr[i+1]
        if start == end: continue
        
        # Copy slice
        row_idx = src_indices[start:end].copy()
        row_t = src_times[start:end].copy()
        
        # Argsort by ID
        order = np.argsort(row_idx)
        
        # Write back sorted
        out_indices[start:end] = row_idx[order]
        out_times[start:end] = row_t[order]


def build_id_sorted_csr_from_neighbor_sampler(neighbor_sampler):
    """
    Build CSR where neighbors are sorted by Node ID (for fast intersection).
    """
    if hasattr(neighbor_sampler, '_csr_id_sorted_indptr'):
        return (neighbor_sampler._csr_id_sorted_indptr, 
                neighbor_sampler._csr_id_sorted_indices, 
                neighbor_sampler._csr_id_sorted_times)

    print("Building ID-sorted CSR for optimized intersection...")
    start = time.time()
    
    # 1. Get standard time-sorted CSR
    if not hasattr(neighbor_sampler, '_csr_indptr'):
        indptr, indices, times = build_csr_from_neighbor_sampler(neighbor_sampler)
        neighbor_sampler._csr_indptr = indptr
        neighbor_sampler._csr_indices = indices
        neighbor_sampler._csr_times = times
    else:
        indptr = neighbor_sampler._csr_indptr
        indices = neighbor_sampler._csr_indices
        times = neighbor_sampler._csr_times

    n_nodes = len(indptr) - 1
    
    # 2. Sort each row by ID
    # New arrays
    sorted_indices = np.empty_like(indices)
    sorted_times = np.empty_like(times)
    
    _sort_csr_rows(indptr, indices, times, sorted_indices, sorted_times)
    if len(sorted_indices) > 0:
        min_neighbor_id = int(np.min(sorted_indices))
        max_neighbor_id = int(np.max(sorted_indices))
    else:
        min_neighbor_id = -1
        max_neighbor_id = -1
    print(
        "ID-sorted CSR neighbor ID range: "
        f"[{min_neighbor_id}, {max_neighbor_id}] vs valid rows [0, {n_nodes - 1}]"
    )
    
    neighbor_sampler._csr_id_sorted_indptr = indptr
    neighbor_sampler._csr_id_sorted_indices = sorted_indices
    neighbor_sampler._csr_id_sorted_times = sorted_times
    
    print(f"✓ ID-sorted CSR built in {time.time() - start:.2f}s")
    return indptr, sorted_indices, sorted_times


@numba.jit(nopython=True, inline='always')
def _temporal_degree_id_sorted(node, indptr, indices, times, t_limit):
    """
    Get node degree at time t_limit given ID-sorted CSR.
    Must scan row because times are not sorted.
    """
    start, end = indptr[node], indptr[node+1]
    count = 0
    # Linear scan - slow for hubs, but usually degree is small?
    # Average degree < 50.
    for i in range(start, end):
        if times[i] < t_limit:
            count += 1
    return count


@numba.jit(nopython=True, parallel=True)
def _score_common_neighbors_sorted_kernel_fixed(sources, targets, pred_times,
                                          indptr, indices, times,
                                          mode):
    # Fixed version using correct degree lookup
    n = len(sources)
    num_nodes = len(indptr) - 1
    scores = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        src, tgt, t = sources[i], targets[i], pred_times[i]
        if src < 0 or src >= num_nodes or tgt < 0 or tgt >= num_nodes: continue

        u_start, u_end = indptr[src], indptr[src+1]
        v_start, v_end = indptr[tgt], indptr[tgt+1]
        
        p_u, p_v = u_start, v_start
        score = 0.0
        
        while p_u < u_end and p_v < v_end:
            u_node, v_node = indices[p_u], indices[p_v]
            
            if u_node < v_node:
                p_u += 1
            elif v_node < u_node:
                p_v += 1
            else:
                # Match z = u_node
                z = u_node
                
                # Check validity (any interaction < t)
                z_valid_u = False
                while p_u < u_end and indices[p_u] == z:
                    if times[p_u] < t: z_valid_u = True
                    p_u += 1
                
                z_valid_v = False
                while p_v < v_end and indices[p_v] == z:
                    if times[p_v] < t: z_valid_v = True
                    p_v += 1
                
                if z_valid_u and z_valid_v:
                    if mode == 0:
                        score += 1.0
                    else:
                        if z >= 0 and z < num_nodes:
                            # Use linear scan for degree on ID-sorted array
                            deg = _temporal_degree_id_sorted(z, indptr, indices, times, t)
                            if mode == 1 and deg > 1: score += 1.0 / np.log(float(deg))
                            elif mode == 2 and deg > 0: score += 1.0 / float(deg)
        
        scores[i] = score
    return scores


def score_links_by_common_neighbors(neighbor_sampler,
                                     sources: np.ndarray,
                                     targets: np.ndarray,
                                     prediction_times: np.ndarray,
                                     mode: str = 'cn',
                                     use_gpu: bool = False) -> np.ndarray:
    """
    Score batch of links by common neighbor metrics (Optimized)
    
    Args:
        ...
        use_gpu: If True and CUDA available, use Block-Parallel GPU kernel.
    """
    # Map mode to integer
    mode_map = {'cn': 0, 'aa': 1, 'ra': 2}
    mode_int = mode_map.get(mode.lower(), 0)

    if use_gpu and HAS_CUDA:
        return score_dense_gpu(neighbor_sampler, sources, targets, prediction_times, mode_int)

    # Fallback to CPU optimized ID-sorted intersection
    # Build ID-sorted CSR
    indptr, indices, times = build_id_sorted_csr_from_neighbor_sampler(neighbor_sampler)

    scores = _score_common_neighbors_sorted_kernel_fixed(
        sources.astype(np.int64),
        targets.astype(np.int64),
        prediction_times.astype(np.float64),
        indptr, indices, times,
        mode_int
    )

    return scores


def precompute_entity_embeddings(*args, **kwargs):
    from experiments.modules.heuristic_semantic_models import precompute_entity_embeddings as fn

    return fn(*args, **kwargs)


def smooth_embeddings_by_time_window_torch(*args, **kwargs):
    from experiments.modules.heuristic_semantic_models import smooth_embeddings_by_time_window_torch as fn

    return fn(*args, **kwargs)


def score_links_by_semantic_similarity(*args, **kwargs):
    from experiments.modules.heuristic_semantic_models import score_links_by_semantic_similarity as fn

    return fn(*args, **kwargs)


def score_links_by_semantic_history_mean(*args, **kwargs):
    from experiments.modules.heuristic_semantic_models import score_links_by_semantic_history_mean as fn

    return fn(*args, **kwargs)


def score_links_by_semantic_history_query_conditioned(*args, **kwargs):
    from experiments.modules.heuristic_semantic_models import (
        score_links_by_semantic_history_query_conditioned as fn,
    )

    return fn(*args, **kwargs)


def score_links_by_semantic_asymmetric_blend(*args, **kwargs):
    from experiments.modules.heuristic_semantic_models import (
        score_links_by_semantic_asymmetric_blend as fn,
    )

    return fn(*args, **kwargs)


def score_links_by_ppr(neighbor_sampler,
                       sources,
                       targets,
                       prediction_times,
                       alpha=0.15,
                       num_walks=None,
                       bfs_depth=3,
                       fanout=20):
    """
    Score links using approximate Personalized PageRank (PPR) from source to target.
    PPR(u -> v) estimated via multi-hop sampled BFS.
    """
    batch_size = len(sources)
    scores = np.zeros(batch_size, dtype=np.float32)
    
    # 1. Get Multi-Hop Neighbors
    try:
        ids_list, _, _ = neighbor_sampler.get_multi_hop_neighbors(
            num_hops=bfs_depth,
            node_ids=sources,
            node_interact_times=prediction_times,
            num_neighbors=fanout
        )
    except Exception as e:
        print(f"PPR Sampling failed: {e}")
        return scores

    current_targets = targets
    
    for k in range(bfs_depth):
        layer_ids = ids_list[k]
        
        if layer_ids.size != batch_size * (fanout**(k+1)):
            continue
            
        layer_ids_flat = layer_ids.reshape(batch_size, -1)
        
        matches = (layer_ids_flat == current_targets[:, None])
        match_counts = matches.sum(axis=1) # [Batch]
        
        hop_idx = k + 1
        term_prob = alpha * ((1 - alpha) ** hop_idx)
        spatial_prob = match_counts / (fanout ** hop_idx)
        
        scores += term_prob * spatial_prob

    return scores


def main():
    """Test and benchmark the recency baseline"""
    from utils.DataLoader import get_link_prediction_data
    from utils.utils import get_neighbor_sampler

    print("="*80)
    print("Recency Baseline - Numba CSR Optimized")
    print("="*80)

    # Load GDELT data
    print("\nLoading GDELT data...")

    class Args:
        use_feature = 'None'
        model_name = 'RecencyBaseline'

    args = Args()

    _, _, full_data, train_data, val_data, test_data, _, _, _ = get_link_prediction_data(
        dataset_name='GDELT',
        val_ratio=0.15,
        test_ratio=0.15,
        args=args
    )

    print(f"✓ Loaded data:")
    print(f"  Full: {full_data.num_interactions:,} edges")
    print(f"  Test: {test_data.num_interactions:,} edges")

    # Build neighbor sampler
    print("\nBuilding neighbor sampler...")
    start_build = time.time()
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy='recent',
        time_scaling_factor=0.0,
        seed=0
    )
    build_time = time.time() - start_build
    print(f"✓ Built in {build_time:.2f}s")

    # Warm-up (triggers CSR conversion + Numba JIT compilation)
    print("\nWarm-up (CSR conversion + Numba JIT compilation)...")
    warmup_start = time.time()
    warmup_batch = 100
    _ = score_links_by_recency(
        full_neighbor_sampler,
        test_data.src_node_ids[:warmup_batch],
        test_data.dst_node_ids[:warmup_batch],
        test_data.node_interact_times[:warmup_batch]
    )
    warmup_time = time.time() - warmup_start
    print(f"✓ Warm-up complete in {warmup_time:.2f}s")

    # Benchmark different batch sizes
    print("\n" + "="*80)
    print("Benchmarks")
    print("="*80)

    for batch_size in [100, 1000, 10000, 50000, 100000]:
        if batch_size > test_data.num_interactions:
            break

        batch_src = test_data.src_node_ids[:batch_size]
        batch_dst = test_data.dst_node_ids[:batch_size]
        batch_times = test_data.node_interact_times[:batch_size]

        start = time.time()
        scores = score_links_by_recency(full_neighbor_sampler, batch_src, batch_dst, batch_times)
        elapsed = time.time() - start

        throughput = batch_size / elapsed
        print(f"\nBatch size: {batch_size:>7,}")
        print(f"  Time: {elapsed:>8.4f}s")
        print(f"  Throughput: {throughput:>10,.0f} links/sec")

        # Statistics
        valid_scores = scores[scores >= 0]
        print(f"  Valid scores: {len(valid_scores):,} / {len(scores):,} ({len(valid_scores)/len(scores)*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ All benchmarks completed")
    print("="*80)


if __name__ == "__main__":
    main()
