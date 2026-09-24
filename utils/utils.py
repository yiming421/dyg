import random
import torch
import torch.nn as nn
import numpy as np

from utils.DataLoader import Data


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


class NeighborSampler:

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),
                                                                                                     node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                                                                                                     num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None, last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None, historical_min_gap: float = 0.0,
                 historical_sample_per_src: bool = False):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        :param historical_min_gap: float, minimum age gap required for historical negatives.
               Larger values make historical negatives easier/safer by excluding recently seen pairs.
        :param historical_sample_per_src: bool, when True historical sampling is source-conditioned:
               keep source node fixed and sample only destination negatives from source-specific history.
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.historical_min_gap = float(historical_min_gap)
        self.historical_sample_per_src = bool(historical_sample_per_src)
        if self.historical_min_gap < 0:
            raise ValueError('historical_min_gap must be >= 0.')
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.has_temporal_info = interact_times is not None
        if self.has_temporal_info:
            self.unique_interact_times = np.unique(interact_times)
            self.earliest_time = min(self.unique_interact_times)
        else:
            self.unique_interact_times = np.array([])
            self.earliest_time = 0.0
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            if not self.has_temporal_info:
                raise ValueError(f'negative_sample_strategy={self.negative_sample_strategy} requires interact_times.')

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        self.stream_ready = self.negative_sample_strategy in ['historical', 'inductive'] and self.has_temporal_info
        if self.stream_ready:
            sorted_order = np.argsort(self.interact_times, kind='mergesort')
            self.sorted_src_node_ids = self.src_node_ids[sorted_order]
            self.sorted_dst_node_ids = self.dst_node_ids[sorted_order]
            self.sorted_interact_times = self.interact_times[sorted_order]
            self._reset_temporal_state()

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def _reset_temporal_state(self):
        """
        Reset streaming temporal cache used by historical/inductive sampling.
        """
        self.hist_ptr = 0
        self.hist_edge_counts = {}
        self.hist_edge_last_time = {}
        self.hist_src_to_dst_last_time = {}
        self.hist_edges_list = []
        self.last_batch_start_time = None
        self.cached_batch_range = None
        self.cached_batch_edges = None

    def _advance_history_to(self, current_batch_start_time: float):
        """
        Incrementally add edges with time < current_batch_start_time into history cache.
        """
        if not self.stream_ready:
            return

        # Handle non-monotonic calls (e.g., a new epoch) safely.
        if self.last_batch_start_time is not None and current_batch_start_time < self.last_batch_start_time:
            self._reset_temporal_state()

        total = len(self.sorted_interact_times)
        while self.hist_ptr < total and self.sorted_interact_times[self.hist_ptr] < current_batch_start_time:
            src = int(self.sorted_src_node_ids[self.hist_ptr])
            dst = int(self.sorted_dst_node_ids[self.hist_ptr])
            edge = (src, dst)
            edge_time = float(self.sorted_interact_times[self.hist_ptr])
            if edge not in self.hist_edge_counts:
                self.hist_edge_counts[edge] = 1
                self.hist_edges_list.append(edge)
            else:
                self.hist_edge_counts[edge] += 1
            self.hist_edge_last_time[edge] = edge_time
            if src not in self.hist_src_to_dst_last_time:
                self.hist_src_to_dst_last_time[src] = {}
            self.hist_src_to_dst_last_time[src][dst] = edge_time
            self.hist_ptr += 1

        self.last_batch_start_time = current_batch_start_time

    def _get_edges_between_time(self, start_time: float, end_time: float):
        """
        Return unique edges with timestamps in [start_time, end_time].
        """
        if not self.stream_ready:
            return self.get_unique_edges_between_start_end_time(start_time=start_time, end_time=end_time)

        cache_key = (float(start_time), float(end_time))
        if self.cached_batch_range == cache_key and self.cached_batch_edges is not None:
            return self.cached_batch_edges

        left = np.searchsorted(self.sorted_interact_times, start_time, side='left')
        right = np.searchsorted(self.sorted_interact_times, end_time, side='right')
        if right <= left:
            edges = set()
            self.cached_batch_range = cache_key
            self.cached_batch_edges = edges
            return edges

        src_slice = self.sorted_src_node_ids[left:right]
        dst_slice = self.sorted_dst_node_ids[left:right]
        edges = set((int(src), int(dst)) for src, dst in zip(src_slice, dst_slice))
        self.cached_batch_range = cache_key
        self.cached_batch_edges = edges
        return edges

    def _passes_historical_gap(self, edge, current_batch_start_time: float):
        """
        Historical negatives can be filtered by how recently the pair last appeared.
        """
        if self.negative_sample_strategy != 'historical' or self.historical_min_gap <= 0.0:
            return True

        last_seen_time = self.hist_edge_last_time.get(edge)
        if last_seen_time is None:
            return False

        return (float(current_batch_start_time) - float(last_seen_time)) >= self.historical_min_gap

    def _fast_sample_from_history(self, size: int, excluded_edges: set, current_batch_start_time: float,
                                  extra_excluded_edges: set = None):
        """
        Fast rejection sampling from cached history edges without replacement.
        Returns sampled edge tuples, potentially fewer than size if candidates are scarce.
        """
        if size <= 0 or len(self.hist_edges_list) == 0:
            return []

        sampled_edges = []
        sampled_set = set()
        n_hist_edges = len(self.hist_edges_list)
        # When min-gap filtering is enabled, acceptance can drop sharply.
        # Cap attempts more aggressively and let caller fill remainder with random negatives.
        attempts_factor = 40 if self.historical_min_gap <= 0.0 else 8
        max_attempts = max(200, size * attempts_factor)
        attempts = 0

        while len(sampled_edges) < size and attempts < max_attempts:
            need = size - len(sampled_edges)
            draw_size = max(need * 2, 16)
            draw_indices = self.random_state.randint(0, n_hist_edges, size=draw_size)
            attempts += draw_size

            for idx in draw_indices:
                edge = self.hist_edges_list[int(idx)]
                if edge in excluded_edges:
                    continue
                if extra_excluded_edges is not None and edge in extra_excluded_edges:
                    continue
                if not self._passes_historical_gap(edge=edge, current_batch_start_time=current_batch_start_time):
                    continue
                if edge in sampled_set:
                    continue
                sampled_set.add(edge)
                sampled_edges.append(edge)
                if len(sampled_edges) >= size:
                    break

        return sampled_edges

    @staticmethod
    def _edges_to_arrays(edges):
        if len(edges) == 0:
            return np.array([], dtype=np.longlong), np.array([], dtype=np.longlong)
        return np.array([edge[0] for edge in edges], dtype=np.longlong), \
               np.array([edge[1] for edge in edges], dtype=np.longlong)

    def _random_dst_for_sources_with_collision_check(self, src_node_ids: np.ndarray, excluded_edges: set):
        """
        Source-conditioned random fill: keep source fixed, resample destination until it avoids excluded_edges.
        """
        src_node_ids = np.asarray(src_node_ids, dtype=np.longlong)
        if len(src_node_ids) == 0:
            return np.array([], dtype=np.longlong), np.array([], dtype=np.longlong)

        negative_dst_node_ids = np.empty(len(src_node_ids), dtype=np.longlong)
        for i, src_node_id in enumerate(src_node_ids):
            attempts = 0
            while True:
                if self.seed is None:
                    dst_idx = np.random.randint(0, len(self.unique_dst_node_ids))
                else:
                    dst_idx = self.random_state.randint(0, len(self.unique_dst_node_ids))
                dst_node_id = int(self.unique_dst_node_ids[dst_idx])
                if (int(src_node_id), dst_node_id) not in excluded_edges:
                    negative_dst_node_ids[i] = dst_node_id
                    break
                attempts += 1
                if attempts >= 1000:
                    negative_dst_node_ids[i] = dst_node_id
                    break

        return src_node_ids, negative_dst_node_ids

    def _sample_historical_edges_for_sources(self, src_node_ids: np.ndarray, current_batch_start_time: float,
                                             excluded_edges: set, extra_excluded_edges: set = None):
        """
        Source-conditioned historical sampling:
        for each requested source node, sample a destination from that source's historical interactions.
        Returns sampled historical edges and source ids that failed to find eligible historical destinations.
        """
        src_node_ids = np.asarray(src_node_ids, dtype=np.longlong)
        if len(src_node_ids) == 0:
            return np.array([], dtype=np.longlong), np.array([], dtype=np.longlong), np.array([], dtype=np.longlong)

        excluded_dst_by_src = {}
        for src_node_id, dst_node_id in excluded_edges:
            src_i = int(src_node_id)
            dst_i = int(dst_node_id)
            if src_i not in excluded_dst_by_src:
                excluded_dst_by_src[src_i] = set()
            excluded_dst_by_src[src_i].add(dst_i)

        if extra_excluded_edges is not None:
            for src_node_id, dst_node_id in extra_excluded_edges:
                src_i = int(src_node_id)
                dst_i = int(dst_node_id)
                if src_i not in excluded_dst_by_src:
                    excluded_dst_by_src[src_i] = set()
                excluded_dst_by_src[src_i].add(dst_i)

        cached_candidates = {}
        sampled_src = []
        sampled_dst = []
        missing_src = []

        for src_node_id in src_node_ids:
            src_i = int(src_node_id)
            if src_i not in cached_candidates:
                src_hist = self.hist_src_to_dst_last_time.get(src_i, None)
                if src_hist is None or len(src_hist) == 0:
                    cached_candidates[src_i] = np.array([], dtype=np.longlong)
                else:
                    excluded_dst = excluded_dst_by_src.get(src_i, None)
                    candidates = []
                    for dst_i, last_seen_time in src_hist.items():
                        if excluded_dst is not None and dst_i in excluded_dst:
                            continue
                        if self.historical_min_gap > 0.0 and (float(current_batch_start_time) - float(last_seen_time)) < self.historical_min_gap:
                            continue
                        candidates.append(int(dst_i))
                    cached_candidates[src_i] = np.asarray(candidates, dtype=np.longlong)

            candidates = cached_candidates[src_i]
            if len(candidates) == 0:
                missing_src.append(src_i)
                continue

            sampled_src.append(src_i)
            dst_idx = self.random_state.randint(0, len(candidates))
            sampled_dst.append(int(candidates[dst_idx]))

        return (
            np.asarray(sampled_src, dtype=np.longlong),
            np.asarray(sampled_dst, dtype=np.longlong),
            np.asarray(missing_src, dtype=np.longlong),
        )

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0,
               allow_random_fallback: bool = True):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time,
                                                                                  allow_random_fallback=allow_random_fallback)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids))
        negative_src_node_ids = np.empty(size, dtype=np.longlong)
        negative_dst_node_ids = np.empty(size, dtype=np.longlong)

        filled = 0
        while filled < size:
            need = size - filled
            draw_size = max(need * 2, 16)
            cand_src_node_ids, cand_dst_node_ids = self.random_sample(size=draw_size)
            for src_node_id, dst_node_id in zip(cand_src_node_ids, cand_dst_node_ids):
                if (src_node_id, dst_node_id) in batch_edges:
                    continue
                negative_src_node_ids[filled] = src_node_id
                negative_dst_node_ids[filled] = dst_node_id
                filled += 1
                if filled >= size:
                    break

        return negative_src_node_ids, negative_dst_node_ids

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float,
                          allow_random_fallback: bool = True):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None

        if self.stream_ready:
            self._advance_history_to(current_batch_start_time=current_batch_start_time)
            current_batch_edges = self._get_edges_between_time(start_time=current_batch_start_time, end_time=current_batch_end_time)

            if self.historical_sample_per_src and batch_src_node_ids is not None:
                requested_src = np.asarray(batch_src_node_ids, dtype=np.longlong)[:size]
                hist_src, hist_dst, missing_src = self._sample_historical_edges_for_sources(
                    src_node_ids=requested_src,
                    current_batch_start_time=current_batch_start_time,
                    excluded_edges=current_batch_edges,
                )
                if len(hist_src) >= size:
                    return hist_src[:size].astype(np.longlong), hist_dst[:size].astype(np.longlong)
                remain = size - len(hist_src)
                if remain > 0:
                    if not allow_random_fallback:
                        return hist_src.astype(np.longlong), hist_dst.astype(np.longlong)
                    fill_src = missing_src[:remain] if len(missing_src) > 0 else requested_src[:remain]
                    random_src, random_dst = self._random_dst_for_sources_with_collision_check(
                        src_node_ids=fill_src,
                        excluded_edges=current_batch_edges,
                    )
                    if len(hist_src) == 0:
                        return random_src.astype(np.longlong), random_dst.astype(np.longlong)
                    return np.concatenate([hist_src, random_src]).astype(np.longlong), \
                           np.concatenate([hist_dst, random_dst]).astype(np.longlong)
                return hist_src.astype(np.longlong), hist_dst.astype(np.longlong)
            else:
                sampled_edges = self._fast_sample_from_history(
                    size=size,
                    excluded_edges=current_batch_edges,
                    current_batch_start_time=current_batch_start_time,
                )
                if len(sampled_edges) >= size:
                    return self._edges_to_arrays(sampled_edges[:size])

                # Fast fallback: avoid O(|history|) exhaustive candidate scans on every batch.
                # Use accepted historical edges, then fill remainder with random negatives.
                hist_src, hist_dst = self._edges_to_arrays(sampled_edges)
                remain = size - len(sampled_edges)
                if remain > 0:
                    if not allow_random_fallback:
                        return hist_src.astype(np.longlong), hist_dst.astype(np.longlong)
                    random_src, random_dst = self.random_sample_with_collision_check(
                        size=remain,
                        batch_src_node_ids=batch_src_node_ids,
                        batch_dst_node_ids=batch_dst_node_ids,
                    )
                    if len(hist_src) == 0:
                        return random_src.astype(np.longlong), random_dst.astype(np.longlong)
                    return np.concatenate([hist_src, random_src]).astype(np.longlong), \
                           np.concatenate([hist_dst, random_dst]).astype(np.longlong)

                return hist_src.astype(np.longlong), hist_dst.astype(np.longlong)

        # Fallback to original full-scan behavior when stream cache is unavailable.
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        if self.historical_min_gap > 0.0:
            self._advance_history_to(current_batch_start_time=current_batch_start_time)
            unique_historical_edges = set(
                edge for edge in unique_historical_edges
                if self._passes_historical_gap(edge=edge, current_batch_start_time=current_batch_start_time)
            )
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            if not allow_random_fallback:
                return unique_historical_edges_src_node_ids.astype(np.longlong), unique_historical_edges_dst_node_ids.astype(np.longlong)
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None

        if self.stream_ready:
            self._advance_history_to(current_batch_start_time=current_batch_start_time)
            current_batch_edges = self._get_edges_between_time(start_time=current_batch_start_time, end_time=current_batch_end_time)

            sampled_edges = self._fast_sample_from_history(
                size=size,
                excluded_edges=current_batch_edges,
                current_batch_start_time=current_batch_start_time,
                extra_excluded_edges=self.observed_edges,
            )
            if len(sampled_edges) >= size:
                return self._edges_to_arrays(sampled_edges[:size])

            hist_src, hist_dst = self._edges_to_arrays(sampled_edges)
            remain = size - len(sampled_edges)
            if remain > 0:
                random_src, random_dst = self.random_sample_with_collision_check(
                    size=remain,
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                )
                if len(hist_src) == 0:
                    return random_src.astype(np.longlong), random_dst.astype(np.longlong)
                return np.concatenate([hist_src, random_src]).astype(np.longlong), \
                       np.concatenate([hist_dst, random_dst]).astype(np.longlong)

            return hist_src.astype(np.longlong), hist_dst.astype(np.longlong)

        # Fallback to original full-scan behavior when stream cache is unavailable.
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)
        if self.stream_ready:
            self._reset_temporal_state()
