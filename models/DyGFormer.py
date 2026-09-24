import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class DyGFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512,
                 attention_mode: str = 'learned', use_neighbor_co_occurrence: bool = True,
                 padding_mode: str = 'masked',
                 device: str = 'cpu'):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param use_neighbor_co_occurrence: whether to include DyGFormer's pairwise
            neighbor co-occurrence channel
        :param padding_mode: ``masked`` excludes padding from attention and
            endpoint pooling; ``legacy`` reproduces the original unmasked
            DyGFormer behavior for controlled experiments
        :param device: str, device
        """
        super(DyGFormer, self).__init__()

        #self.node_raw_features = nn.Parameter(torch.from_numpy(node_raw_features.astype(np.float32)), requires_grad = True).to(device)
        #self.edge_raw_features = nn.Parameter(torch.from_numpy(edge_raw_features.astype(np.float32)), requires_grad = True).to(device)

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.attention_mode = str(attention_mode)
        if self.attention_mode not in {'learned', 'uniform', 'mlp_mixer'}:
            raise ValueError(f'Unsupported DyGFormer attention mode: {self.attention_mode}')
        self.use_neighbor_co_occurrence = bool(use_neighbor_co_occurrence)
        self.padding_mode = str(padding_mode)
        if self.padding_mode not in {'masked', 'legacy'}:
            raise ValueError(f'Unsupported DyGFormer padding mode: {self.padding_mode}')
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = (
            NeighborCooccurrenceEncoder(
                neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
                device=self.device,
            )
            if self.use_neighbor_co_occurrence
            else None
        )

        projection_layers = {
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        }
        if self.use_neighbor_co_occurrence:
            projection_layers['neighbor_co_occurrence'] = nn.Linear(
                in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            )
        self.projection_layer = nn.ModuleDict(projection_layers)

        self.num_channels = 4 if self.use_neighbor_co_occurrence else 3

        # A canonical MLP-Mixer learns a dense map over the token axis and
        # therefore requires a fixed token count.  Each endpoint is padded to
        # the same configured cap only for this mode; masking keeps those extra
        # slots from changing the evidence seen by the model.
        fixed_endpoint_length = (
            ((self.max_input_sequence_length + self.patch_size - 1) // self.patch_size)
            * self.patch_size
        )
        self.mixer_num_tokens = (
            2 * (fixed_endpoint_length // self.patch_size)
            if self.attention_mode == 'mlp_mixer'
            else None
        )
        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads,
                               dropout=self.dropout, attention_mode=self.attention_mode,
                               num_tokens=self.mixer_num_tokens)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        if self.use_neighbor_co_occurrence:
            src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
                self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                    dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)
        else:
            src_padded_nodes_neighbor_co_occurrence_features = None
            dst_padded_nodes_neighbor_co_occurrence_features = None

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        if self.use_neighbor_co_occurrence:
            src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        if self.use_neighbor_co_occurrence:
            dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features]
        if self.use_neighbor_co_occurrence:
            patches_nodes_neighbor_co_occurrence_features = torch.cat(
                [src_patches_nodes_neighbor_co_occurrence_features,
                 dst_patches_nodes_neighbor_co_occurrence_features],
                dim=1,
            )
            patches_data.append(patches_nodes_neighbor_co_occurrence_features)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # A patch is padding only when all node ids inside it are padding.  Both
        # learned and fixed-uniform attention must exclude these positions so an
        # endpoint representation does not depend on unrelated batch padding.
        src_patch_valid = np.any(
            src_padded_nodes_neighbor_ids.reshape(batch_size, src_num_patches, self.patch_size) != 0,
            axis=2,
        )
        dst_patch_valid = np.any(
            dst_padded_nodes_neighbor_ids.reshape(batch_size, dst_num_patches, self.patch_size) != 0,
            axis=2,
        )
        patch_padding_mask = torch.from_numpy(
            ~np.concatenate([src_patch_valid, dst_patch_valid], axis=1)
        ).to(self.device)

        # The legacy branch is intentionally opt-in and exists only to measure
        # the original implementation's padding effect under matched settings.
        transformer_padding_mask = (
            patch_padding_mask if self.padding_mode == 'masked' else None
        )
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(
                patches_data,
                key_padding_mask=transformer_padding_mask,
            )

        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        if self.padding_mode == 'legacy':
            src_patches_data = torch.mean(src_patches_data, dim=1)
            dst_patches_data = torch.mean(dst_patches_data, dim=1)
        else:
            # Apply the same masked endpoint readout to both token mixers.  An
            # ordinary mean would dilute short histories according to the longest
            # sequence in the current batch and pool activations produced at padded
            # query positions.
            src_valid = torch.from_numpy(src_patch_valid).to(
                device=src_patches_data.device,
                dtype=src_patches_data.dtype,
            ).unsqueeze(dim=-1)
            dst_valid = torch.from_numpy(dst_patch_valid).to(
                device=dst_patches_data.device,
                dtype=dst_patches_data.dtype,
            ).unsqueeze(dim=-1)
            src_patches_data = (src_patches_data * src_valid).sum(dim=1) / \
                src_valid.sum(dim=1).clamp_min(1.0)
            dst_patches_data = (dst_patches_data * dst_valid).sum(dim=1) / \
                dst_valid.sum(dim=1).clamp_min(1.0)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if self.attention_mode == 'mlp_mixer':
            # Dense token mixing needs a stable token axis across batches.  The
            # valid prefix is unchanged; only padding is extended to the
            # configured endpoint cap.
            max_seq_length = max_input_sequence_length
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features = []
        patches_nodes_edge_raw_features = []
        patches_nodes_neighbor_time_features = []
        patches_nodes_neighbor_co_occurrence_features = (
            [] if padded_nodes_neighbor_co_occurrence_features is not None else None
        )

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            if patches_nodes_neighbor_co_occurrence_features is not None:
                patches_nodes_neighbor_co_occurrence_features.append(
                    padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :]
                )

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        if patches_nodes_neighbor_co_occurrence_features is not None:
            patches_nodes_neighbor_co_occurrence_features = torch.stack(
                patches_nodes_neighbor_co_occurrence_features, dim=1
            ).reshape(
                batch_size,
                num_patches,
                patch_size * self.neighbor_co_occurrence_feat_dim,
            )

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # Batched equality matrices exactly reproduce the original per-row
        # np.unique/dictionary implementation while avoiding Python callbacks for
        # every token. Sequence lengths are capped by DyGFormer, so the temporary
        # O(BL^2) boolean arrays remain small.
        src_counts_in_src = (
            src_padded_nodes_neighbor_ids[:, :, None]
            == src_padded_nodes_neighbor_ids[:, None, :]
        ).sum(axis=2)
        src_counts_in_dst = (
            src_padded_nodes_neighbor_ids[:, :, None]
            == dst_padded_nodes_neighbor_ids[:, None, :]
        ).sum(axis=2)
        dst_counts_in_src = (
            dst_padded_nodes_neighbor_ids[:, :, None]
            == src_padded_nodes_neighbor_ids[:, None, :]
        ).sum(axis=2)
        dst_counts_in_dst = (
            dst_padded_nodes_neighbor_ids[:, :, None]
            == dst_padded_nodes_neighbor_ids[:, None, :]
        ).sum(axis=2)

        src_appearances = np.stack(
            [src_counts_in_src, src_counts_in_dst], axis=2
        ).astype(np.float32, copy=False)
        dst_appearances = np.stack(
            [dst_counts_in_src, dst_counts_in_dst], axis=2
        ).astype(np.float32, copy=False)
        src_appearances[src_padded_nodes_neighbor_ids == 0] = 0.0
        dst_appearances[dst_padded_nodes_neighbor_ids == 0] = 0.0

        return (
            torch.from_numpy(src_appearances).to(self.device),
            torch.from_numpy(dst_appearances).to(self.device),
        )

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1,
                 attention_mode: str = 'learned', num_tokens: int = None,
                 token_dim_expansion_factor: float = 0.5):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param attention_mode: learned self-attention, fixed uniform mixing, or
            a padding-safe MLP-Mixer token MLP
        :param num_tokens: fixed number of joint source/destination tokens,
            required only by MLP-Mixer
        :param token_dim_expansion_factor: hidden-width multiplier for the
            MLP-Mixer token MLP, matching GraphMixer's default
        """
        super(TransformerEncoder, self).__init__()
        if attention_mode not in {'learned', 'uniform', 'mlp_mixer'}:
            raise ValueError(f'Unsupported DyGFormer attention mode: {attention_mode}')
        self.attention_mode = attention_mode
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        if self.attention_mode == 'mlp_mixer':
            if self.num_tokens is None or self.num_tokens <= 0:
                raise ValueError('MLP-Mixer requires a positive fixed num_tokens')
            token_hidden_dim = max(1, int(self.num_tokens * token_dim_expansion_factor))
            self.multi_head_attention = None
            self.token_linear_layers = nn.ModuleList([
                nn.Linear(in_features=self.num_tokens, out_features=token_hidden_dim),
                nn.Linear(in_features=token_hidden_dim, out_features=self.num_tokens),
            ])
        else:
            # use the MultiheadAttention implemented by PyTorch
            self.multi_head_attention = MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.token_linear_layers = None

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def _uniform_attention(self, normalized_inputs: torch.Tensor,
                           key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Fixed uniform token mixing with learned V and output projections."""
        attention = self.multi_head_attention
        embedding_dim = attention.embed_dim
        bias = attention.in_proj_bias
        if key_padding_mask is None:
            summary_inputs = normalized_inputs.mean(dim=0, keepdim=True)
        else:
            valid = (~key_padding_mask).transpose(0, 1).unsqueeze(-1).to(normalized_inputs.dtype)
            summary_inputs = (normalized_inputs * valid).sum(dim=0, keepdim=True) / valid.sum(
                dim=0, keepdim=True
            ).clamp_min(1.0)
        # Uniform averaging commutes with the affine value projection, so apply
        # V only to the single summary instead of redundantly to every token.
        summary = F.linear(
            summary_inputs,
            attention.in_proj_weight[2 * embedding_dim:],
            None if bias is None else bias[2 * embedding_dim:],
        )
        # The uniform message is identical for every query, so project it once
        # per example and broadcast afterward instead of repeating the same
        # O(D^2) output projection for every token.
        return attention.out_proj(summary).expand_as(normalized_inputs)

    def forward(self, inputs: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        if key_padding_mask is not None:
            if key_padding_mask.shape != inputs.shape[:2]:
                raise ValueError(
                    'key_padding_mask must have shape '
                    f'{tuple(inputs.shape[:2])}, got {tuple(key_padding_mask.shape)}'
                )
            masked_inputs = inputs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        else:
            masked_inputs = inputs

        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = masked_inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        if key_padding_mask is not None:
            transposed_inputs = transposed_inputs.masked_fill(
                key_padding_mask.transpose(0, 1).unsqueeze(-1),
                0.0,
            )
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        if self.attention_mode == 'uniform':
            hidden_states = self._uniform_attention(
                transposed_inputs,
                key_padding_mask=key_padding_mask,
            ).transpose(0, 1)
        elif self.attention_mode == 'learned':
            hidden_states = self.multi_head_attention(
                query=transposed_inputs,
                key=transposed_inputs,
                value=transposed_inputs,
                key_padding_mask=key_padding_mask,
            )[0].transpose(0, 1)
        else:
            if inputs.size(1) != self.num_tokens:
                raise ValueError(
                    f'MLP-Mixer expected {self.num_tokens} tokens, got {inputs.size(1)}'
                )
            # Canonical Mixer token MLP: normalize channels, transpose to
            # [batch, channels, tokens], mix along the fixed token axis, and
            # return to [batch, tokens, channels].
            hidden_states = transposed_inputs.permute(1, 2, 0)
            hidden_states = self.token_linear_layers[0](hidden_states)
            hidden_states = self.dropout(F.gelu(hidden_states))
            hidden_states = self.token_linear_layers[1](hidden_states)
            hidden_states = hidden_states.transpose(1, 2)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = masked_inputs + self.dropout(hidden_states)
        if key_padding_mask is not None:
            outputs = outputs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        if key_padding_mask is not None:
            # Prevent LayerNorm/linear biases at padding positions from becoming
            # inputs to a later Mixer block.
            outputs = outputs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return outputs
