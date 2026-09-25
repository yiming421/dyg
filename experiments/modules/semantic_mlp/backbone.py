from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiments.modules.heuristic_semantic_models import (
    build_source_history_mean_initialized_embeddings,
)
from experiments.modules.heuristic_models import smooth_embeddings_by_time_window_torch
from experiments.modules.llm_lp.cli import normalize_semantic_fusion_heuristics
from experiments.modules.prediction_metrics import compute_prediction_metrics
from experiments.modules.semantic_mlp.runtime import (
    HeuristicFeatureExtractor,
    HeuristicFusionHead,
    LearnableGCNEncoder,
    LearnableGINEncoder,
    MPLPExactFusionHead,
    RollingSmoothedEmbeddingProvider,
    apply_attention_pool_message_passing_for_nodes,
    build_binary_history_adj,
    build_semantic_mlp_auxiliary_features,
    build_two_hop_binary_adj,
    fuse_pos_neg_logits_with_mplp_exact,
    score_edge_batch,
)
from experiments.modules.semantic_mlp.models import (
    SemanticCrossAttention,
    SemanticDyGFormerLiteScorer,
    SemanticNCNScorer,
    SemanticSeqFilterScorer,
    SemanticMLP,
    TemporalNeighborIndex,
    TemporalSelfAttentionPooling,
    build_lookup_tensor,
    maybe_filter_train_edges,
    make_subset,
)
from experiments.modules.semantic_mlp.ridge import (
    FixedRandomProjection,
    SemanticRidgeScorer,
)
from experiments.modules.semantic_mlp.heuristic_components import (
    build_googlemap_city_zip_ids,
)
from experiments.modules.semantic_mlp.graph_components import (
    build_temporal_relational_context,
    resolve_checkpoint_semantic_source_init,
)
from utils.DataLoader import get_link_prediction_data
from utils.utils import get_neighbor_sampler
from utils.graph_history import TRAIN_HISTORY_POLICY, graph_history
from experiments.modules.llm_lp.training_protocol import sample_history_scope


class SemanticMLPHybridBackbone:
    score_field = "semantic_mlp_score"
    score_label = "Semantic MLP"

    def __init__(
        self,
        *,
        dataset_name,
        checkpoint_path,
        eval_positive_batch_size,
        device,
        model,
        gcn_encoder,
        heuristic_fusion,
        mplp_exact_fusion,
        heuristic_extractor,
        lookup,
        base_embeddings,
        static_embeddings,
        static_mp_adj,
        static_ncn_adj,
        static_mplp_exact_adj2,
        full_src,
        full_dst,
        full_times,
        full_edge_ids,
        rolling_provider_enabled,
        cross_attn_neighbor_index,
        mp_neighbor_index,
        model_config,
        gcn_config,
        heuristic_config,
        structural_config,
        smoothing_config,
        rolling_smoothing,
        temporal_mode,
    ):
        self.dataset_name = str(dataset_name)
        self.checkpoint_path = str(checkpoint_path)
        self.eval_positive_batch_size = (
            None
            if eval_positive_batch_size is None
            else max(1, int(eval_positive_batch_size))
        )
        self.device = device
        self.model = model
        self.gcn_encoder = gcn_encoder
        self.heuristic_fusion = heuristic_fusion
        self.mplp_exact_fusion = mplp_exact_fusion
        self.heuristic_extractor = heuristic_extractor
        self.lookup = lookup
        self.base_embeddings = base_embeddings
        self.static_embeddings = static_embeddings
        self.static_mp_adj = static_mp_adj
        self.static_ncn_adj = static_ncn_adj
        self.static_mplp_exact_adj2 = static_mplp_exact_adj2
        self.full_src = np.asarray(full_src, dtype=np.int64)
        self.full_dst = np.asarray(full_dst, dtype=np.int64)
        self.full_times = np.asarray(full_times, dtype=np.float64)
        self.full_edge_ids = np.asarray(full_edge_ids, dtype=np.int64)
        self.rolling_provider_enabled = bool(rolling_provider_enabled)
        self.cross_attn_neighbor_index = cross_attn_neighbor_index
        self.mp_neighbor_index = mp_neighbor_index
        self.model_config = dict(model_config or {})
        self.gcn_config = dict(gcn_config or {})
        self.heuristic_config = dict(heuristic_config or {})
        self.structural_config = dict(structural_config or {})
        self.smoothing_config = dict(smoothing_config or {})
        self.rolling_smoothing = bool(rolling_smoothing)
        self.temporal_mode = str(temporal_mode or "rolling_replay").strip().lower()
        if self.temporal_mode not in {"rolling_replay", "timestamp_rebuild"}:
            raise ValueError(
                "semantic backbone temporal_mode must be one of "
                "rolling_replay/timestamp_rebuild, "
                f"got {self.temporal_mode!r}."
            )

        self.scorer_type = str(self.model_config.get("scorer_type", "mlp"))
        self.cross_attn_num_neighbors = int(
            self.model_config.get("cross_attn_num_neighbors", 50)
        )
        self.ncn_num_neighbors = int(self.model_config.get("ncn_num_neighbors", 50))
        self.seqfilter_num_neighbors = int(
            self.model_config.get("seqfilter_num_neighbors", 32)
        )
        self.use_learnable_gcn = bool(self.gcn_config.get("use_learnable_gcn", False))
        self.learnable_mp_type = str(self.gcn_config.get("learnable_mp_type", "gcn"))
        self.attn_mp_num_neighbors = int(self.gcn_config.get("attn_mp_num_neighbors", 50))
        self.use_heuristic_features = bool(
            self.heuristic_config.get("use_heuristic_features", False)
        )
        self.use_mplp_exact_features = bool(
            self.structural_config.get("use_mplp_exact_features", False)
        )
        self.semantic_aux_fusion_mode = str(
            self.model_config.get(
                "semantic_aux_fusion_mode",
                self.heuristic_config.get("semantic_aux_fusion_mode", "residual"),
            )
        )
        self.mp_replaces_smoothing = self.use_learnable_gcn
        self.graph_mp_uses_smoothing_adj = (
            self.use_learnable_gcn and self.learnable_mp_type in {"gcn", "gin"}
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        dataset_name,
        checkpoint_path,
        embeddings,
        entity_ids_sorted,
        device="cpu",
        val_ratio=0.15,
        test_ratio=0.15,
        backbone_ablate_recency=False,
        eval_positive_batch_size=None,
        source_init_override="auto",
        temporal_mode="rolling_replay",
        history_scope="evaluation",
    ):
        if history_scope not in {"train", "evaluation"}:
            raise ValueError("Unknown graph history scope")
        if embeddings is None:
            raise ValueError("Semantic hybrid backbone requires preloaded embeddings.")

        device_obj = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=device_obj)
        if history_scope == "train" and ckpt.get("train_history_policy") != TRAIN_HISTORY_POLICY:
            raise ValueError("Training-context scoring requires an observed_train_only_v1 checkpoint")

        checkpoint_dataset_name = ckpt.get("dataset_name")
        if checkpoint_dataset_name and str(checkpoint_dataset_name) != str(dataset_name):
            raise ValueError(
                "Semantic hybrid backbone checkpoint dataset mismatch: "
                f"checkpoint={checkpoint_dataset_name}, args={dataset_name}."
            )

        model_config = dict(ckpt.get("model_config") or {})
        gcn_config = dict(ckpt.get("gcn_config") or {})
        heuristic_config = dict(ckpt.get("heuristic_config") or {})
        structural_config = dict(ckpt.get("structural_config") or {})
        heuristic_config["heuristic_feature_names"] = list(
            normalize_semantic_fusion_heuristics(
                heuristic_config.get("heuristic_feature_names")
            )
        )
        smoothing_config = dict(ckpt.get("smoothing") or {})
        rolling_smoothing = bool(ckpt.get("rolling_smoothing", True))
        strict_no_leakage = bool(ckpt.get("strict_no_leakage", True))
        resolved_source_init, source_init_resolution = (
            resolve_checkpoint_semantic_source_init(
                smoothing_config=smoothing_config,
                gcn_config=gcn_config,
                rolling_smoothing=rolling_smoothing,
                source_init_override=source_init_override,
            )
        )
        if source_init_resolution.startswith("legacy_"):
            print(
                "Semantic hybrid legacy checkpoint compatibility: using raw source "
                "embeddings because the old runtime ignored its recorded source-init "
                f"setting ({source_init_resolution})."
            )
        elif source_init_resolution == "missing_metadata_default":
            print(
                "Semantic hybrid checkpoint is missing source-init metadata; "
                "defaulting to raw embeddings."
            )
        smoothing_config["source_init"] = resolved_source_init
        smoothing_config["source_init_effective"] = resolved_source_init
        smoothing_config["source_init_resolution"] = source_init_resolution

        base_embeddings = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device_obj)
        base_embeddings = F.normalize(base_embeddings, dim=1)

        input_dim = int(model_config["input_dim"])
        raw_input_dim = int(model_config.get("raw_input_dim", input_dim))
        if int(base_embeddings.shape[1]) != raw_input_dim:
            raise ValueError(
                "Semantic hybrid backbone embedding dimension mismatch: "
                f"checkpoint={raw_input_dim}, embeddings={int(base_embeddings.shape[1])}."
            )

        scorer_type = str(model_config.get("scorer_type", "mlp"))
        if scorer_type == "mlp":
            model = SemanticMLP(
                input_dim=input_dim,
                auxiliary_dim=int(model_config.get("auxiliary_dim", 0)),
                pair_feature_mode=str(model_config.get("pair_feature_mode", "concat")),
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
                activation=str(model_config["activation"]),
                use_layernorm=bool(model_config.get("use_layernorm", True)),
            ).to(device_obj)
        elif scorer_type == "ridge":
            model = SemanticRidgeScorer(
                input_dim=input_dim,
                auxiliary_dim=int(model_config.get("auxiliary_dim", 0)),
                semantic_feature_mode=str(
                    model_config.get("ridge_semantic_feature_mode", "hadamard")
                ),
            ).to(device_obj)
        elif scorer_type == "cross_attention":
            model = SemanticCrossAttention(
                input_dim=input_dim,
                num_layers=int(model_config.get("cross_attn_num_layers", model_config["num_layers"])),
                num_heads=int(model_config.get("cross_attn_heads", 4)),
                hidden_dropout_prob=float(model_config.get("cross_attn_hidden_dropout", model_config["dropout"])),
                attn_dropout_prob=float(model_config.get("cross_attn_attn_dropout", model_config["dropout"])),
                emb_dropout_prob=float(model_config.get("cross_attn_emb_dropout", 0.1)),
                activation=str(model_config["activation"]),
                use_pos=bool(model_config.get("cross_attn_use_pos", False)),
                max_seq_length=int(model_config.get("cross_attn_num_neighbors", 50)),
                add_time_to_history=bool(
                    model_config.get("cross_attn_add_time_to_history", True)
                ),
                time_encoder_type=str(model_config.get("time_encoder_type", "mlp")),
                time_encoder_mask_padding=bool(
                    model_config.get("time_encoder_mask_padding", True)
                ),
                time_encoder_fourier_dim=int(
                    model_config.get("time_encoder_fourier_dim", 32)
                ),
                time_encoder_rbf_dim=int(model_config.get("time_encoder_rbf_dim", 32)),
                time_encoder_rbf_gamma=float(
                    model_config.get("time_encoder_rbf_gamma", 16.0)
                ),
            ).to(device_obj)
        elif scorer_type == "dygformer_lite":
            model = SemanticDyGFormerLiteScorer(
                input_dim=input_dim,
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config.get("dygformer_num_layers", 1)),
                head_num_layers=int(model_config.get("num_layers", 2)),
                num_neighbors=int(model_config.get("dygformer_num_neighbors", 20)),
                num_heads=int(model_config.get("dygformer_heads", 4)),
                dropout=float(model_config["dropout"]),
                activation=str(model_config["activation"]),
                use_layernorm=bool(model_config.get("use_layernorm", True)),
                add_time=bool(model_config.get("dygformer_add_time", True)),
                time_encoder_type=str(model_config.get("time_encoder_type", "mlp")),
                time_encoder_fourier_dim=int(
                    model_config.get("time_encoder_fourier_dim", 32)
                ),
                time_encoder_rbf_dim=int(model_config.get("time_encoder_rbf_dim", 32)),
                time_encoder_rbf_gamma=float(
                    model_config.get("time_encoder_rbf_gamma", 16.0)
                ),
                time_encoder_mask_padding=bool(
                    model_config.get("time_encoder_mask_padding", True)
                ),
            ).to(device_obj)
        elif scorer_type == "ncn":
            model = SemanticNCNScorer(
                input_dim=input_dim,
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
                activation=str(model_config["activation"]),
                use_layernorm=bool(model_config.get("use_layernorm", True)),
            ).to(device_obj)
        elif scorer_type == "seqfilter":
            model = SemanticSeqFilterScorer(
                input_dim=input_dim,
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                num_neighbors=int(model_config.get("seqfilter_num_neighbors", 32)),
                tau=float(model_config.get("seqfilter_tau", 0.2)),
                kernel_size=int(model_config.get("seqfilter_kernel_size", 3)),
                dropout=float(model_config["dropout"]),
                activation=str(model_config["activation"]),
                use_layernorm=bool(model_config.get("use_layernorm", True)),
                time_encoder_type=str(model_config.get("time_encoder_type", "mlp")),
                time_encoder_fourier_dim=int(
                    model_config.get("time_encoder_fourier_dim", 32)
                ),
                time_encoder_rbf_dim=int(model_config.get("time_encoder_rbf_dim", 32)),
                time_encoder_rbf_gamma=float(
                    model_config.get("time_encoder_rbf_gamma", 16.0)
                ),
                use_soft_mask=bool(model_config.get("seqfilter_use_soft_mask", True)),
            ).to(device_obj)
        else:
            raise ValueError(f"Unsupported semantic hybrid scorer type: {scorer_type}")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ridge_projection = None
        ridge_projection_dim = int(model_config.get("ridge_projection_dim", 0))
        if scorer_type == "ridge" and ridge_projection_dim > 0:
            ridge_projection = FixedRandomProjection(
                input_dim=raw_input_dim,
                output_dim=ridge_projection_dim,
                seed=int(model_config.get("ridge_projection_seed", 42)),
            ).to(device_obj)
            projection_state = ckpt.get("ridge_projection_state_dict")
            if projection_state is None:
                raise ValueError("Semantic ridge checkpoint is missing ridge_projection_state_dict.")
            ridge_projection.load_state_dict(projection_state)
            ridge_projection.eval()

        gcn_encoder = None
        if bool(gcn_config.get("use_learnable_gcn", False)):
            mp_type = str(gcn_config.get("learnable_mp_type", "gcn"))
            state_dict = ckpt.get("gcn_state_dict")
            if state_dict is None:
                raise ValueError("Semantic hybrid checkpoint is missing gcn_state_dict.")
            if mp_type == "gcn":
                relation_features = None
                if bool(gcn_config.get("use_temporal_relational_gcn", False)):
                    relation_features = state_dict.get("relation_features")
                    if relation_features is None:
                        raise ValueError(
                            "Temporal-relational checkpoint is missing its relation feature table."
                        )
                gcn_encoder = LearnableGCNEncoder(
                    input_dim=input_dim,
                    hidden_dim=int(gcn_config.get("gcn_hidden_dim", input_dim)),
                    num_layers=int(gcn_config.get("gcn_num_layers", 1)),
                    dropout=float(gcn_config.get("gcn_dropout", 0.0)),
                    activation=str(gcn_config.get("gcn_activation", "relu")),
                    use_layernorm=bool(gcn_config.get("gcn_use_layernorm", True)),
                    residual=bool(gcn_config.get("gcn_residual", True)),
                    use_linear_transform=bool(
                        gcn_config.get("gcn_use_linear_transform", True)
                    ),
                    relation_features=relation_features,
                    temporal_relational_rank=int(
                        gcn_config.get("temporal_relational_rank", 32)
                    ),
                    temporal_relational_time_basis_dim=int(
                        gcn_config.get("temporal_relational_time_basis_dim", 16)
                    ),
                ).to(device_obj)
            elif mp_type == "gin":
                gcn_encoder = LearnableGINEncoder(
                    input_dim=input_dim,
                    hidden_dim=int(gcn_config.get("gcn_hidden_dim", input_dim)),
                    num_layers=int(gcn_config.get("gcn_num_layers", 1)),
                    dropout=float(gcn_config.get("gcn_dropout", 0.0)),
                    activation=str(gcn_config.get("gcn_activation", "relu")),
                    use_layernorm=bool(gcn_config.get("gcn_use_layernorm", True)),
                    residual=bool(gcn_config.get("gcn_residual", True)),
                    nonparametric=bool(gcn_config.get("gin_nonparametric", False)),
                    nonparametric_norm=str(
                        gcn_config.get("gin_nonparametric_norm", "bn")
                    ),
                ).to(device_obj)
            elif mp_type == "attn_pool":
                gcn_encoder = TemporalSelfAttentionPooling(
                    input_dim=input_dim,
                    num_layers=int(gcn_config.get("attn_mp_layers", 1)),
                    num_heads=int(gcn_config.get("attn_mp_heads", 4)),
                    dropout=float(gcn_config.get("attn_mp_dropout", 0.0)),
                    activation=str(model_config.get("activation", "gelu")),
                    residual=bool(gcn_config.get("attn_mp_residual", True)),
                    time_encoder_type=str(gcn_config.get("time_encoder_type", "mlp")),
                    time_encoder_mask_padding=bool(
                        gcn_config.get("time_encoder_mask_padding", True)
                    ),
                    time_encoder_fourier_dim=int(
                        gcn_config.get("time_encoder_fourier_dim", 32)
                    ),
                    time_encoder_rbf_dim=int(gcn_config.get("time_encoder_rbf_dim", 32)),
                    time_encoder_rbf_gamma=float(
                        gcn_config.get("time_encoder_rbf_gamma", 16.0)
                    ),
                ).to(device_obj)
            else:
                raise ValueError(f"Unsupported semantic hybrid MP type: {mp_type}")

            gcn_encoder.load_state_dict(state_dict)
            gcn_encoder.eval()

        heuristic_fusion = None
        heuristic_fusion_mode = str(
            model_config.get(
                "semantic_aux_fusion_mode",
                heuristic_config.get("semantic_aux_fusion_mode", "residual"),
            )
        )
        if bool(heuristic_config.get("use_heuristic_features", False)) and heuristic_fusion_mode != "late_concat":
            heuristic_fusion = HeuristicFusionHead(
                feature_dim=len(heuristic_config["heuristic_feature_names"])
            ).to(device_obj)
            state_dict = ckpt.get("heuristic_fusion_state_dict")
            if state_dict is None:
                raise ValueError(
                    "Semantic hybrid checkpoint is missing heuristic_fusion_state_dict."
                )
            heuristic_fusion.load_state_dict(state_dict)
            heuristic_fusion.eval()

        mplp_exact_fusion = None
        structural_fusion_mode = str(
            model_config.get(
                "semantic_aux_fusion_mode",
                structural_config.get(
                    "semantic_aux_fusion_mode",
                    heuristic_config.get("semantic_aux_fusion_mode", "residual"),
                ),
            )
        )
        if bool(structural_config.get("use_mplp_exact_features", False)) and structural_fusion_mode != "late_concat":
            mplp_exact_fusion = MPLPExactFusionHead().to(device_obj)
            state_dict = ckpt.get("mplp_exact_fusion_state_dict")
            if state_dict is None:
                raise ValueError(
                    "Semantic hybrid checkpoint is missing mplp_exact_fusion_state_dict."
                )
            mplp_exact_fusion.load_state_dict(state_dict)
            mplp_exact_fusion.eval()

        class DataArgs:
            use_feature = "None"
            model_name = "SemanticMLP"

        _, _, full_data, train_data, _, test_data, _, _, _ = get_link_prediction_data(
            dataset_name=dataset_name,
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            args=DataArgs(),
        )

        max_node_id = int(max(full_data.src_node_ids.max(), full_data.dst_node_ids.max()))
        train_data, _, _ = maybe_filter_train_edges(
            train_data=train_data,
            cutoff_time=ckpt.get("train_edge_cutoff_time"),
            cutoff_ratio=float(ckpt.get("train_edge_cutoff_ratio", 1.0)),
        )
        holdout = int(ckpt.get("train_holdout_recent_edges") or 0)
        if holdout:
            if holdout >= len(train_data.src_node_ids):
                raise ValueError("Checkpoint holdout removes the entire training graph")
            keep = np.arange(len(train_data.src_node_ids)) < len(train_data.src_node_ids) - holdout
            train_data = make_subset(train_data, keep)
        if history_scope == "train":
            full_data = train_data

        full_src = np.asarray(full_data.src_node_ids, dtype=np.int64)
        full_dst = np.asarray(full_data.dst_node_ids, dtype=np.int64)
        full_times = np.asarray(full_data.node_interact_times, dtype=np.float64)
        full_edge_ids = np.asarray(full_data.edge_ids, dtype=np.int64)

        static_mask = np.ones(len(full_src), dtype=bool)
        if strict_no_leakage:
            test_start_time = float(test_data.node_interact_times[0])
            static_mask &= full_times < test_start_time
        smooth_cutoff_time = smoothing_config.get("smooth_cutoff_time")
        if smooth_cutoff_time is not None:
            static_mask &= full_times < float(smooth_cutoff_time)

        static_src = full_src[static_mask]
        static_dst = full_dst[static_mask]
        static_times = full_times[static_mask]
        if not rolling_smoothing and ckpt.get("train_history_policy") == TRAIN_HISTORY_POLICY:
            static_times = np.asarray(train_data.node_interact_times, dtype=np.float64)
            static_keep = np.ones(len(static_times), dtype=bool)
            if smooth_cutoff_time is not None:
                static_keep &= static_times < float(smooth_cutoff_time)
            static_src = np.asarray(train_data.src_node_ids)[static_keep]
            static_dst = np.asarray(train_data.dst_node_ids)[static_keep]
            static_times = static_times[static_keep]

        lookup = build_lookup_tensor(entity_ids_sorted, max_node_id, device_obj)

        if resolved_source_init == "history_mean" and not rolling_smoothing:
            base_embeddings, replaced_nodes = build_source_history_mean_initialized_embeddings(
                base_embeddings=base_embeddings,
                node_id_lookup=lookup,
                src_node_ids=static_src,
                dst_node_ids=static_dst,
            )
            print(
                "Semantic hybrid source init updated "
                f"{replaced_nodes:,} source/user embeddings from static history."
            )
        elif resolved_source_init == "history_mean":
            print(
                "Semantic hybrid source init: causal rolling history_mean "
                "(events strictly before each scoring batch)."
            )
        else:
            print("Semantic hybrid source init: raw base embeddings.")

        if ridge_projection is not None:
            with torch.inference_mode():
                base_embeddings = F.normalize(ridge_projection(base_embeddings), dim=1)
            if int(base_embeddings.shape[1]) != input_dim:
                raise ValueError(
                    "Semantic ridge projection width mismatch: "
                    f"checkpoint input_dim={input_dim}, projected={int(base_embeddings.shape[1])}."
                )

        temporal_history = graph_history(full_data, smooth_cutoff_time)
        cross_attn_neighbor_index = None
        if str(model_config.get("scorer_type", "mlp")) in {"cross_attention", "dygformer_lite", "ncn", "seqfilter"}:
            cross_attn_neighbor_index = TemporalNeighborIndex(
                src_node_ids=temporal_history.src_node_ids,
                dst_node_ids=temporal_history.dst_node_ids,
                node_interact_times=temporal_history.node_interact_times,
                max_node_id=max_node_id,
                undirected=bool(model_config.get("cross_attn_undirected_history", True)),
            )

        mp_neighbor_index = None
        if bool(gcn_config.get("use_learnable_gcn", False)) and str(
            gcn_config.get("learnable_mp_type", "gcn")
        ) == "attn_pool":
            mp_neighbor_index = TemporalNeighborIndex(
                src_node_ids=temporal_history.src_node_ids,
                dst_node_ids=temporal_history.dst_node_ids,
                node_interact_times=temporal_history.node_interact_times,
                max_node_id=max_node_id,
                undirected=bool(gcn_config.get("gcn_undirected", True)),
            )

        heuristic_extractor = None
        if bool(heuristic_config.get("use_heuristic_features", False)):
            node_city_ids = None
            node_zip_ids = None
            location_features = {"city_preference", "zip_preference"} & set(
                heuristic_config["heuristic_feature_names"]
            )
            if location_features:
                if str(dataset_name) != "Googlemap_CT":
                    raise ValueError(
                        "city_preference/zip_preference currently require Googlemap_CT."
                    )
                entity_text_path = heuristic_config.get("entity_text_path") or os.path.join(
                    "..", "DyLink_Datasets", str(dataset_name), "entity_text.csv"
                )
                if not os.path.exists(entity_text_path):
                    raise FileNotFoundError(
                        "Googlemap city/ZIP preference features require entity_text.csv at "
                        f"{entity_text_path}."
                    )
                import pandas as pd

                entity_text_df = pd.read_csv(entity_text_path)
                node_city_ids, node_zip_ids, _, _ = build_googlemap_city_zip_ids(
                    entity_text_df=entity_text_df,
                    max_node_id=max_node_id,
                )
            # Heuristic kernels enforce interaction_time < prediction_time.
            # Give them the authoritative stream so earlier test interactions
            # remain available without exposing future edges. Using static_src
            # here froze recency/popularity/past/RA at the test boundary.
            heuristic_graph_data = SimpleNamespace(
                src_node_ids=full_src,
                dst_node_ids=full_dst,
                edge_ids=np.arange(len(full_src), dtype=np.int64),
                node_interact_times=full_times,
            )
            heuristic_neighbor_sampler = get_neighbor_sampler(
                data=heuristic_graph_data,
                sample_neighbor_strategy="recent",
                seed=0,
            )
            heuristic_extractor = HeuristicFeatureExtractor(
                neighbor_sampler=heuristic_neighbor_sampler,
                directed_src_node_ids=full_src,
                directed_dst_node_ids=full_dst,
                directed_node_interact_times=full_times,
                use_gpu_heuristics=bool(heuristic_config.get("use_gpu_heuristics", True)),
                popularity_decay=float(
                    heuristic_config.get("heuristic_popularity_decay", 0.0)
                ),
                recent_degree_window=float(
                    heuristic_config.get("heuristic_recent_degree_window", 50.0)
                ),
                score_batch_size=int(
                    heuristic_config.get("backbone_score_batch_size", 50000)
                ),
                ablate_recency=bool(backbone_ablate_recency),
                recency_directed=bool(
                    heuristic_config.get("heuristic_recency_directed", False)
                ),
                feature_names=tuple(heuristic_config["heuristic_feature_names"]),
                node_city_ids=node_city_ids,
                node_zip_ids=node_zip_ids,
            )
            heuristic_extractor.load_normalization_state_dict(ckpt.get("heuristic_preprocessing"))
            print(
                "Semantic heuristic fusion config: "
                f"features={','.join(heuristic_extractor.feature_names)}, "
                f"use_gpu_heuristics={heuristic_extractor.use_gpu_heuristics}, "
                f"score_batch_size={heuristic_extractor.score_batch_size}, "
                f"popularity_decay={heuristic_extractor.popularity_decay}, "
                f"recent_degree_window={heuristic_extractor.recent_degree_window}, "
                f"ablate_recency={heuristic_extractor.ablate_recency}, "
                f"recency_directed={heuristic_extractor.recency_directed}"
            )

        static_embeddings = None
        static_mp_adj = None
        static_ncn_adj = None
        static_mplp_exact_adj2 = None
        use_learnable_gcn = bool(gcn_config.get("use_learnable_gcn", False))
        learnable_mp_type = str(gcn_config.get("learnable_mp_type", "gcn"))
        graph_mp_uses_smoothing_adj = use_learnable_gcn and learnable_mp_type in {
            "gcn",
            "gin",
        }
        rolling_provider_enabled = bool(
            rolling_smoothing
            and (
                (not use_learnable_gcn)
                or graph_mp_uses_smoothing_adj
                or resolved_source_init == "history_mean"
            )
        )
        if not rolling_smoothing:
            if str(model_config.get("scorer_type", "mlp")) == "ncn" or bool(
                structural_config.get("use_mplp_exact_features", False)
            ):
                static_ncn_adj = build_binary_history_adj(
                    src_node_ids=static_src,
                    dst_node_ids=static_dst,
                    lookup=lookup,
                    num_rows=int(base_embeddings.shape[0]),
                    undirected=True,
                )
                if bool(structural_config.get("use_mplp_exact_features", False)):
                    static_mplp_exact_adj2 = build_two_hop_binary_adj(static_ncn_adj)
            if use_learnable_gcn:
                if graph_mp_uses_smoothing_adj:
                    if learnable_mp_type == "gcn":
                        _, static_mp_adj = smooth_embeddings_by_time_window_torch(
                            embeddings=base_embeddings,
                            src_node_ids=static_src,
                            dst_node_ids=static_dst,
                            node_interact_times=static_times,
                            time_window=float(smoothing_config.get("time_window", 50.0)),
                            num_steps=1,
                            symmetric_norm=True,
                            decay_gamma=float(smoothing_config.get("decay_gamma", 0.1)),
                            undirected=bool(smoothing_config.get("undirected", True)),
                            log_dampen=bool(smoothing_config.get("log_dampen", True)),
                            supernode_strength=float(
                                smoothing_config.get("supernode_strength", 0.5)
                            ),
                            endpoint_topk_recent=int(
                                smoothing_config.get("endpoint_topk_recent", 0)
                            ),
                            endpoint_topk_mode=str(
                                smoothing_config.get("endpoint_topk_mode", "per_node")
                            ),
                            device=device_obj,
                            return_norm_adj=True,
                        )
                    else:
                        _, static_mp_adj = smooth_embeddings_by_time_window_torch(
                            embeddings=base_embeddings,
                            src_node_ids=static_src,
                            dst_node_ids=static_dst,
                            node_interact_times=static_times,
                            time_window=float(smoothing_config.get("time_window", 50.0)),
                            num_steps=1,
                            symmetric_norm=True,
                            decay_gamma=float(smoothing_config.get("decay_gamma", 0.1)),
                            undirected=bool(smoothing_config.get("undirected", True)),
                            log_dampen=bool(smoothing_config.get("log_dampen", True)),
                            supernode_strength=float(
                                smoothing_config.get("supernode_strength", 0.5)
                            ),
                            endpoint_topk_recent=int(
                                smoothing_config.get("endpoint_topk_recent", 0)
                            ),
                            endpoint_topk_mode=str(
                                smoothing_config.get("endpoint_topk_mode", "per_node")
                            ),
                            device=device_obj,
                            return_sum_adj=True,
                        )
            else:
                static_embeddings = smooth_embeddings_by_time_window_torch(
                    embeddings=base_embeddings,
                    src_node_ids=static_src,
                    dst_node_ids=static_dst,
                    node_interact_times=static_times,
                    time_window=float(smoothing_config.get("time_window", 50.0)),
                    num_steps=int(smoothing_config.get("steps", 1)),
                    symmetric_norm=True,
                    decay_gamma=float(smoothing_config.get("decay_gamma", 0.1)),
                    undirected=bool(smoothing_config.get("undirected", True)),
                    log_dampen=bool(smoothing_config.get("log_dampen", True)),
                    supernode_strength=float(smoothing_config.get("supernode_strength", 0.5)),
                    endpoint_topk_recent=int(
                        smoothing_config.get("endpoint_topk_recent", 0)
                    ),
                    endpoint_topk_mode=str(
                        smoothing_config.get("endpoint_topk_mode", "per_node")
                    ),
                    device=device_obj,
                )
                static_embeddings = F.normalize(static_embeddings.float(), dim=1)

        result = cls(
            dataset_name=dataset_name,
            checkpoint_path=checkpoint_path,
            eval_positive_batch_size=eval_positive_batch_size,
            device=device_obj,
            model=model,
            gcn_encoder=gcn_encoder,
            heuristic_fusion=heuristic_fusion,
            mplp_exact_fusion=mplp_exact_fusion,
            heuristic_extractor=heuristic_extractor,
            lookup=lookup,
            base_embeddings=base_embeddings,
            static_embeddings=static_embeddings,
            static_mp_adj=static_mp_adj,
            static_ncn_adj=static_ncn_adj,
            static_mplp_exact_adj2=static_mplp_exact_adj2,
            full_src=full_src,
            full_dst=full_dst,
            full_times=full_times,
            full_edge_ids=full_edge_ids,
            rolling_provider_enabled=rolling_provider_enabled,
            cross_attn_neighbor_index=cross_attn_neighbor_index,
            mp_neighbor_index=mp_neighbor_index,
            model_config=model_config,
            gcn_config=gcn_config,
            heuristic_config=heuristic_config,
            structural_config=structural_config,
            smoothing_config=smoothing_config,
            rolling_smoothing=rolling_smoothing,
            temporal_mode=temporal_mode,
        )
        result.history_scope = history_scope
        result._training_view = None
        result._reload_kwargs = dict(
            dataset_name=dataset_name, checkpoint_path=checkpoint_path,
            embeddings=embeddings, entity_ids_sorted=entity_ids_sorted, device=device,
            val_ratio=val_ratio, test_ratio=test_ratio,
            backbone_ablate_recency=backbone_ablate_recency,
            eval_positive_batch_size=eval_positive_batch_size,
            source_init_override=source_init_override, temporal_mode=temporal_mode,
        )
        return result

    def _scorer_for_samples(self, samples):
        scope = sample_history_scope(samples)
        if scope == "train":
            spec = samples[0].get("training_history_spec", {})
            if (spec.get("data_seed") != 2020
                    or spec.get("val_ratio") != self._reload_kwargs["val_ratio"]
                    or spec.get("test_ratio") != self._reload_kwargs["test_ratio"]):
                raise ValueError("Training samples and GNN use different reserved-node splits")
        if scope == "train" and self.history_scope != "train":
            if self._training_view is None:
                self._training_view = type(self).from_checkpoint(**self._reload_kwargs, history_scope="train")
            return self._training_view
        if scope != self.history_scope:
            raise ValueError("Evaluation samples cannot use a training-only scorer")
        return self

    def _dynamic_source_embeddings(self, query_time):
        if self.smoothing_config.get("source_init_effective", "raw") != "history_mean":
            return self.base_embeddings

        query_time = float(query_time)
        history_mask = self.full_times < query_time
        smooth_cutoff_time = self.smoothing_config.get("smooth_cutoff_time")
        if smooth_cutoff_time is not None:
            history_mask &= self.full_times < float(smooth_cutoff_time)
        initialized, _ = build_source_history_mean_initialized_embeddings(
            base_embeddings=self.base_embeddings,
            node_id_lookup=self.lookup,
            src_node_ids=self.full_src[history_mask],
            dst_node_ids=self.full_dst[history_mask],
        )
        return initialized

    def _dynamic_smoothing(self, query_time, *, return_norm_adj=False, return_sum_adj=False):
        query_time = float(query_time)
        time_mask = self.full_times < query_time
        smooth_cutoff_time = self.smoothing_config.get("smooth_cutoff_time")
        if smooth_cutoff_time is not None:
            time_mask &= self.full_times < float(smooth_cutoff_time)
        return smooth_embeddings_by_time_window_torch(
            embeddings=self._dynamic_source_embeddings(query_time),
            src_node_ids=self.full_src[time_mask],
            dst_node_ids=self.full_dst[time_mask],
            node_interact_times=self.full_times[time_mask],
            time_window=float(self.smoothing_config.get("time_window", 50.0)),
            num_steps=int(self.smoothing_config.get("steps", 1)),
            symmetric_norm=True,
            decay_gamma=float(self.smoothing_config.get("decay_gamma", 0.1)),
            undirected=bool(self.smoothing_config.get("undirected", True)),
            log_dampen=bool(self.smoothing_config.get("log_dampen", True)),
            supernode_strength=float(self.smoothing_config.get("supernode_strength", 0.5)),
            endpoint_topk_recent=int(
                self.smoothing_config.get("endpoint_topk_recent", 0)
            ),
            endpoint_topk_mode=str(
                self.smoothing_config.get("endpoint_topk_mode", "per_node")
            ),
            reference_time=query_time,
            device=self.device,
            return_norm_adj=bool(return_norm_adj),
            return_sum_adj=bool(return_sum_adj),
        )

    def _build_rolling_provider(self, split_start_time):
        if not self.rolling_provider_enabled:
            return None
        history_mask = np.ones(len(self.full_times), dtype=bool)
        smooth_cutoff_time = self.smoothing_config.get("smooth_cutoff_time")
        if smooth_cutoff_time is not None:
            history_mask &= self.full_times < float(smooth_cutoff_time)
        return RollingSmoothedEmbeddingProvider(
            base_embeddings=self.base_embeddings,
            lookup=self.lookup,
            init_src_node_ids=self.full_src[history_mask],
            init_dst_node_ids=self.full_dst[history_mask],
            init_node_interact_times=self.full_times[history_mask],
            smooth_time_window=float(self.smoothing_config.get("time_window", 50.0)),
            smooth_steps=int(self.smoothing_config.get("steps", 1)),
            smooth_decay_gamma=float(self.smoothing_config.get("decay_gamma", 0.1)),
            smooth_undirected=bool(self.smoothing_config.get("undirected", True)),
            smooth_log_dampen=bool(self.smoothing_config.get("log_dampen", True)),
            smooth_supernode_strength=float(
                self.smoothing_config.get("supernode_strength", 0.5)
            ),
            smooth_endpoint_topk_recent=int(
                self.smoothing_config.get("endpoint_topk_recent", 0)
            ),
            smooth_endpoint_topk_mode=str(
                self.smoothing_config.get("endpoint_topk_mode", "per_node")
            ),
            export_norm_adj=(self.use_learnable_gcn and self.learnable_mp_type == "gcn"),
            export_sum_adj=(self.use_learnable_gcn and self.learnable_mp_type == "gin"),
            export_binary_adj=(self.scorer_type == "ncn"),
            export_binary_two_hop_adj=self.use_mplp_exact_features,
            source_init=str(
                self.smoothing_config.get("source_init_effective", "raw")
            ),
            apply_smoothing=(
                (not self.use_learnable_gcn) or self.graph_mp_uses_smoothing_adj
            ),
            history_is_complete=True,
            materialize_smoothed_embeddings=not self.use_learnable_gcn,
            init_edge_ids=self.full_edge_ids[history_mask],
            export_temporal_relational_context=bool(
                self.gcn_config.get("use_temporal_relational_gcn", False)
            ),
            temporal_relational_num_relations=(
                int(self.gcn_encoder.relation_features.size(0))
                if bool(self.gcn_config.get("use_temporal_relational_gcn", False))
                else 0
            ),
            temporal_relational_time_basis_dim=int(
                self.gcn_config.get("temporal_relational_time_basis_dim", 16)
            ),
        )

    def _dynamic_temporal_relational_context(self, query_time):
        if not bool(self.gcn_config.get("use_temporal_relational_gcn", False)):
            return None
        time_mask = self.full_times < float(query_time)
        time_mask &= self.full_times >= (
            float(query_time) - float(self.smoothing_config.get("time_window", 50.0))
        )
        smooth_cutoff_time = self.smoothing_config.get("smooth_cutoff_time")
        if smooth_cutoff_time is not None:
            time_mask &= self.full_times < float(smooth_cutoff_time)
        return build_temporal_relational_context(
            src_node_ids=self.full_src[time_mask],
            dst_node_ids=self.full_dst[time_mask],
            edge_ids=self.full_edge_ids[time_mask],
            node_interact_times=self.full_times[time_mask],
            reference_time=float(query_time),
            time_window=float(self.smoothing_config.get("time_window", 50.0)),
            decay_gamma=float(self.smoothing_config.get("decay_gamma", 0.1)),
            lookup=self.lookup,
            num_rows=int(self.base_embeddings.size(0)),
            num_relations=int(self.gcn_encoder.relation_features.size(0)),
            time_basis_dim=int(
                self.gcn_config.get("temporal_relational_time_basis_dim", 16)
            ),
            output_dtype=self.base_embeddings.dtype,
        )

    def _get_current_embeddings(self, query_time, src_ids, dst_ids, *, rolling_provider=None):
        source_embeddings = self.base_embeddings
        if rolling_provider is not None:
            source_embeddings = rolling_provider.current_base_embeddings
        elif self.rolling_smoothing:
            source_embeddings = self._dynamic_source_embeddings(query_time)
        current_embeddings = source_embeddings

        if self.use_learnable_gcn:
            if self.learnable_mp_type == "gcn":
                current_mp_adj = self.static_mp_adj
                if rolling_provider is not None:
                    current_mp_adj = rolling_provider.current_norm_adj
                elif self.rolling_smoothing:
                    _, current_mp_adj = self._dynamic_smoothing(
                        query_time, return_norm_adj=True
                    )
                temporal_relational_context = (
                    rolling_provider.current_temporal_relational_context
                    if rolling_provider is not None
                    else self._dynamic_temporal_relational_context(query_time)
                )
                temporal_relational_query_rows = None
                if getattr(self.gcn_encoder, "use_temporal_relational", False):
                    query_node_ids = torch.as_tensor(
                        np.concatenate([src_ids, dst_ids]),
                        dtype=torch.long,
                        device=self.lookup.device,
                    )
                    valid_query_ids = (
                        (query_node_ids >= 0)
                        & (query_node_ids < self.lookup.numel())
                    )
                    temporal_relational_query_rows = torch.unique(
                        self.lookup[query_node_ids[valid_query_ids]]
                    )
                    temporal_relational_query_rows = temporal_relational_query_rows[
                        temporal_relational_query_rows >= 0
                    ]
                current_embeddings = self.gcn_encoder(
                    source_embeddings,
                    current_mp_adj,
                    temporal_relational_context=temporal_relational_context,
                    temporal_relational_query_rows=temporal_relational_query_rows,
                )
                current_embeddings = F.normalize(current_embeddings, dim=1)
            elif self.learnable_mp_type == "gin":
                current_mp_adj = self.static_mp_adj
                if rolling_provider is not None:
                    current_mp_adj = rolling_provider.current_sum_adj
                elif self.rolling_smoothing:
                    _, current_mp_adj = self._dynamic_smoothing(
                        query_time, return_sum_adj=True
                    )
                current_embeddings = self.gcn_encoder(source_embeddings, current_mp_adj)
                current_embeddings = F.normalize(current_embeddings, dim=1)
            elif self.learnable_mp_type == "attn_pool":
                query_nodes = np.concatenate([src_ids, dst_ids], axis=0)
                current_embeddings = apply_attention_pool_message_passing_for_nodes(
                    mp_encoder=self.gcn_encoder,
                    embeddings=source_embeddings,
                    lookup=self.lookup,
                    query_node_ids=query_nodes,
                    query_time=float(query_time),
                    neighbor_index=self.mp_neighbor_index,
                    num_neighbors=self.attn_mp_num_neighbors,
                )
            else:
                raise ValueError(f"Unsupported semantic hybrid MP type: {self.learnable_mp_type}")
        elif rolling_provider is not None:
            current_embeddings = rolling_provider.current_embeddings
        elif self.rolling_smoothing:
            current_embeddings = self._dynamic_smoothing(query_time)
            current_embeddings = F.normalize(current_embeddings.float(), dim=1)
        elif self.static_embeddings is not None:
            current_embeddings = self.static_embeddings

        return current_embeddings

    def _get_current_structural_adjs(self, query_time, *, rolling_provider=None):
        if rolling_provider is not None:
            batch_ncn_adj = (
                rolling_provider.current_binary_adj
                if rolling_provider is not None
                else self.static_ncn_adj
            )
            batch_mplp_exact_adj2 = (
                rolling_provider.current_binary_two_hop_adj
                if rolling_provider is not None
                else self.static_mplp_exact_adj2
            )
            return batch_ncn_adj, batch_mplp_exact_adj2

        needs_binary_adj = (
            self.scorer_type == "ncn" or self.use_mplp_exact_features
        )
        if not (self.rolling_smoothing and needs_binary_adj):
            return self.static_ncn_adj, self.static_mplp_exact_adj2

        time_mask = self.full_times < float(query_time)
        time_mask &= self.full_times >= (
            float(query_time) - float(self.smoothing_config.get("time_window", 50.0))
        )
        smooth_cutoff_time = self.smoothing_config.get("smooth_cutoff_time")
        if smooth_cutoff_time is not None:
            time_mask &= self.full_times < float(smooth_cutoff_time)

        batch_ncn_adj = build_binary_history_adj(
            src_node_ids=self.full_src[time_mask],
            dst_node_ids=self.full_dst[time_mask],
            lookup=self.lookup,
            num_rows=int(self.base_embeddings.size(0)),
            undirected=True,
        )
        if not self.use_mplp_exact_features:
            return batch_ncn_adj, None
        try:
            batch_mplp_exact_adj2 = build_two_hop_binary_adj(batch_ncn_adj)
        except torch.cuda.OutOfMemoryError:
            if batch_ncn_adj is not None and batch_ncn_adj.storage.row().device.type == "cuda":
                torch.cuda.empty_cache()
            batch_mplp_exact_adj2 = None
        return batch_ncn_adj, batch_mplp_exact_adj2

    def _encode_semantic_mlp_pairs(
        self,
        embeddings,
        src_ids,
        dst_ids,
        *,
        auxiliary_features=None,
    ):
        """Encode aligned links at the same graph state without changing checkpoint weights."""
        if not isinstance(self.model, SemanticMLP):
            raise TypeError(
                "Pair-representation extraction currently supports SemanticMLP checkpoints only; "
                f"got {type(self.model).__name__}."
            )
        device = embeddings.device
        src = torch.as_tensor(src_ids, dtype=torch.long, device=device)
        dst = torch.as_tensor(dst_ids, dtype=torch.long, device=device)
        max_id = int(self.lookup.size(0))
        in_range = (src >= 0) & (src < max_id) & (dst >= 0) & (dst < max_id)
        src_idx = torch.full_like(src, -1)
        dst_idx = torch.full_like(dst, -1)
        src_idx[in_range] = self.lookup[src[in_range]]
        dst_idx[in_range] = self.lookup[dst[in_range]]
        valid = in_range & (src_idx >= 0) & (dst_idx >= 0)
        output = torch.zeros(
            (int(src.numel()), int(self.model.pair_representation_dim)),
            dtype=embeddings.dtype,
            device=device,
        )
        if valid.any():
            valid_aux = auxiliary_features[valid] if auxiliary_features is not None else None
            output[valid] = self.model.encode_pair(
                embeddings[src_idx[valid]],
                embeddings[dst_idx[valid]],
                valid_aux,
            )
        return output, valid

    def _apply_heuristic_fusion(
        self,
        logits,
        src_ids,
        dst_ids,
        times,
        *,
        raw_features=None,
    ):
        if self.heuristic_extractor is None or self.heuristic_fusion is None:
            return logits

        if raw_features is None:
            raw_features = self.heuristic_extractor.get_raw_features(
                sources=src_ids,
                targets=dst_ids,
                prediction_times=times,
            )
        else:
            raw_features = np.asarray(raw_features, dtype=np.float32)
        features = self.heuristic_extractor.normalize_raw_features(raw_features)
        feature_tensor = torch.from_numpy(features).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        return self.heuristic_fusion(logits, feature_tensor)

    def _apply_mplp_exact_fusion(
        self,
        logits,
        src_ids,
        dst_ids,
        *,
        ncn_adj,
        mplp_exact_adj2,
    ):
        if self.mplp_exact_fusion is None:
            return logits
        fused_logits, _, _ = fuse_pos_neg_logits_with_mplp_exact(
            mplp_exact_fusion=self.mplp_exact_fusion,
            lookup=self.lookup,
            pos_logits=logits,
            neg_logits=logits[:0],
            pos_src=src_ids,
            pos_dst=dst_ids,
            neg_src=np.empty((0,), dtype=np.int64),
            neg_dst=np.empty((0,), dtype=np.int64),
            ncn_adj=ncn_adj,
            two_hop_adj=mplp_exact_adj2,
        )
        return fused_logits

    def forward_timestamp_group(
        self,
        src_ids,
        dst_ids,
        times,
        *,
        raw_heuristic_features=None,
        return_pair_features=False,
    ):
        """Differentiably score links sharing one timestamp.

        Unlike ``score_samples``, this method deliberately does not enter
        inference mode or detach its outputs.  It is the training entry point
        for objectives that need to fine-tune the live GNN/scorer checkpoint.
        The graph state is rebuilt from all real interactions strictly before
        the supplied timestamp, matching ``temporal_mode=timestamp_rebuild``.
        """
        src_ids = np.asarray(src_ids, dtype=np.int64)
        dst_ids = np.asarray(dst_ids, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        if not (len(src_ids) == len(dst_ids) == len(times)):
            raise ValueError("src_ids, dst_ids, and times must have equal length.")
        if len(times) == 0:
            empty_logits = self.base_embeddings.new_empty((0,))
            if not return_pair_features:
                return empty_logits
            if not isinstance(self.model, SemanticMLP):
                raise TypeError(
                    "Pair-representation extraction currently supports "
                    "SemanticMLP checkpoints only."
                )
            empty_pairs = self.base_embeddings.new_empty(
                (0, int(self.model.pair_representation_dim))
            )
            empty_valid = torch.empty(
                (0,), dtype=torch.bool, device=self.base_embeddings.device
            )
            return empty_logits, empty_pairs, empty_valid

        query_time = float(times[0])
        if not np.all(times == query_time):
            raise ValueError(
                "forward_timestamp_group requires every row to share one timestamp."
            )

        current_embeddings = self._get_current_embeddings(
            query_time=query_time,
            src_ids=src_ids,
            dst_ids=dst_ids,
            rolling_provider=None,
        )
        batch_ncn_adj, batch_mplp_exact_adj2 = self._get_current_structural_adjs(
            query_time=query_time,
            rolling_provider=None,
        )
        auxiliary_features = None
        if self.semantic_aux_fusion_mode == "late_concat":
            auxiliary_features = build_semantic_mlp_auxiliary_features(
                heuristic_extractor=self.heuristic_extractor,
                lookup=self.lookup,
                sources=src_ids,
                targets=dst_ids,
                prediction_times=times,
                raw_heuristic_features=raw_heuristic_features,
                ncn_adj=batch_ncn_adj if self.use_mplp_exact_features else None,
                two_hop_adj=(
                    batch_mplp_exact_adj2 if self.use_mplp_exact_features else None
                ),
                output_device=current_embeddings.device,
                output_dtype=current_embeddings.dtype,
                normalize_heuristics=not getattr(
                    self.model, "expects_raw_auxiliary_features", False
                ),
            )

        logits, valid = score_edge_batch(
            self.model,
            current_embeddings,
            self.lookup,
            src_ids,
            dst_ids,
            auxiliary_features=auxiliary_features,
            node_interact_times=times,
            neighbor_index=self.cross_attn_neighbor_index,
            ncn_adj=batch_ncn_adj,
            cross_attn_num_neighbors=self.cross_attn_num_neighbors,
            ncn_num_neighbors=self.ncn_num_neighbors,
            seqfilter_num_neighbors=self.seqfilter_num_neighbors,
        )
        pair_features = None
        if return_pair_features:
            pair_features, pair_valid = self._encode_semantic_mlp_pairs(
                current_embeddings,
                src_ids,
                dst_ids,
                auxiliary_features=auxiliary_features,
            )
            valid = valid & pair_valid

        if self.semantic_aux_fusion_mode != "late_concat":
            logits = self._apply_heuristic_fusion(
                logits,
                src_ids,
                dst_ids,
                times,
                raw_features=raw_heuristic_features,
            )
            logits = self._apply_mplp_exact_fusion(
                logits,
                src_ids,
                dst_ids,
                ncn_adj=batch_ncn_adj,
                mplp_exact_adj2=batch_mplp_exact_adj2,
            )

        if return_pair_features:
            return logits, pair_features, valid
        return logits

    @staticmethod
    def _build_query_spans(samples):
        if not samples:
            return []
        spans = []
        start = 0
        current_query_id = int(samples[0]["query_id"])
        for idx in range(1, len(samples)):
            query_id = int(samples[idx]["query_id"])
            if query_id != current_query_id:
                spans.append((start, idx))
                start = idx
                current_query_id = query_id
        spans.append((start, len(samples)))
        return spans

    def _score_samples_exact_timestamp(
        self,
        samples,
        sample_timestamps,
        precomputed_raw_features,
        *,
        use_rolling_provider=True,
        return_pair_features=False,
    ):
        timestamp_order = np.argsort(sample_timestamps, kind="mergesort")
        sorted_timestamps = sample_timestamps[timestamp_order]
        predictions = np.zeros(len(samples), dtype=np.float64)
        pair_features = None
        if return_pair_features:
            if not isinstance(self.model, SemanticMLP):
                raise TypeError(
                    "Pair-representation extraction currently supports SemanticMLP checkpoints only."
                )
            pair_features = np.zeros(
                (len(samples), int(self.model.pair_representation_dim)),
                dtype=np.float32,
            )
        debug_group_logs = 0
        total_groups = (
            int(1 + np.count_nonzero(np.diff(sorted_timestamps)))
            if len(sorted_timestamps) > 0
            else 0
        )
        progress_t0 = time.perf_counter()
        last_progress_t = progress_t0
        groups_processed = 0
        samples_processed = 0
        slow_group_threshold_s = 5.0

        print(
            "[SemanticBackbone] scoring start: "
            f"total_samples={len(samples)}, total_timestamp_groups={total_groups}, "
            "mode=exact_timestamp"
        )
        rolling_provider = (
            self._build_rolling_provider(float(np.min(sample_timestamps)))
            if use_rolling_provider
            else None
        )

        with torch.inference_mode():
            start = 0
            while start < len(timestamp_order):
                idx0 = int(timestamp_order[start])
                query_time = float(samples[idx0].get("dtgb_timestamp", samples[idx0]["timestamp"]))
                end = start + 1
                while end < len(timestamp_order):
                    next_idx = int(timestamp_order[end])
                    if float(samples[next_idx].get("dtgb_timestamp", samples[next_idx]["timestamp"])) != query_time:
                        break
                    end += 1

                batch_indices = timestamp_order[start:end]
                src_ids = np.asarray(
                    [int(samples[int(idx)]["source_id"]) for idx in batch_indices],
                    dtype=np.int64,
                )
                dst_ids = np.asarray(
                    [int(samples[int(idx)]["target_id"]) for idx in batch_indices],
                    dtype=np.int64,
                )
                times = np.asarray(
                    [float(samples[int(idx)].get("dtgb_timestamp", samples[int(idx)]["timestamp"])) for idx in batch_indices],
                    dtype=np.float64,
                )
                if debug_group_logs < 8:
                    print(
                        "[SemanticBackbone] timestamp group: "
                        f"group_idx={groups_processed}, time={query_time:.6f}, "
                        f"size={int(len(batch_indices))}, start={int(start)}, end={int(end)}"
                    )
                    debug_group_logs += 1

                group_t0 = time.perf_counter()
                if rolling_provider is not None:
                    rolling_provider.set_base_embeddings(self.base_embeddings)
                    rolling_provider.prepare_batch(times)
                current_embeddings = self._get_current_embeddings(
                    query_time=query_time,
                    src_ids=src_ids,
                    dst_ids=dst_ids,
                    rolling_provider=rolling_provider,
                )
                batch_ncn_adj, batch_mplp_exact_adj2 = (
                    self._get_current_structural_adjs(
                        query_time=query_time,
                        rolling_provider=rolling_provider,
                    )
                )
                embedding_elapsed = time.perf_counter() - group_t0
                score_t0 = time.perf_counter()
                batch_raw_features = (
                    precomputed_raw_features[batch_indices]
                    if precomputed_raw_features is not None
                    else None
                )
                auxiliary_features = None
                if self.semantic_aux_fusion_mode == "late_concat":
                    auxiliary_features = build_semantic_mlp_auxiliary_features(
                        heuristic_extractor=self.heuristic_extractor,
                        lookup=self.lookup,
                        sources=src_ids,
                        targets=dst_ids,
                        prediction_times=times,
                        raw_heuristic_features=batch_raw_features,
                        ncn_adj=batch_ncn_adj if self.use_mplp_exact_features else None,
                        two_hop_adj=batch_mplp_exact_adj2 if self.use_mplp_exact_features else None,
                        output_device=current_embeddings.device,
                        output_dtype=current_embeddings.dtype,
                        normalize_heuristics=not getattr(
                            self.model, "expects_raw_auxiliary_features", False
                        ),
                    )
                logits, _ = score_edge_batch(
                    self.model,
                    current_embeddings,
                    self.lookup,
                    src_ids,
                    dst_ids,
                    auxiliary_features=auxiliary_features,
                    node_interact_times=times,
                    neighbor_index=self.cross_attn_neighbor_index,
                    ncn_adj=batch_ncn_adj,
                    cross_attn_num_neighbors=self.cross_attn_num_neighbors,
                    ncn_num_neighbors=self.ncn_num_neighbors,
                    seqfilter_num_neighbors=self.seqfilter_num_neighbors,
                )
                batch_pair_features = None
                if return_pair_features:
                    batch_pair_features, _ = self._encode_semantic_mlp_pairs(
                        current_embeddings,
                        src_ids,
                        dst_ids,
                        auxiliary_features=auxiliary_features,
                    )
                model_elapsed = time.perf_counter() - score_t0
                heur_t0 = time.perf_counter()
                if self.semantic_aux_fusion_mode != "late_concat":
                    logits = self._apply_heuristic_fusion(
                        logits,
                        src_ids,
                        dst_ids,
                        times,
                        raw_features=batch_raw_features,
                    )
                    logits = self._apply_mplp_exact_fusion(
                        logits,
                        src_ids,
                        dst_ids,
                        ncn_adj=batch_ncn_adj,
                        mplp_exact_adj2=batch_mplp_exact_adj2,
                    )
                heuristic_elapsed = time.perf_counter() - heur_t0
                copy_t0 = time.perf_counter()
                logits_np = logits.detach().float().cpu().numpy().astype(np.float64, copy=False)
                copy_elapsed = time.perf_counter() - copy_t0

                for local_pos, sample_idx in enumerate(batch_indices):
                    predictions[int(sample_idx)] = float(logits_np[local_pos])
                    if pair_features is not None:
                        pair_features[int(sample_idx)] = (
                            batch_pair_features[local_pos].detach().float().cpu().numpy()
                        )
                groups_processed += 1
                samples_processed += int(len(batch_indices))
                group_elapsed = time.perf_counter() - group_t0
                if group_elapsed >= slow_group_threshold_s:
                    print(
                        "[SemanticBackbone] slow timestamp group: "
                        f"group_idx={groups_processed - 1}/{total_groups}, "
                        f"time={query_time:.6f}, size={int(len(batch_indices))}, "
                        f"embed_s={embedding_elapsed:.2f}, "
                        f"model_s={model_elapsed:.2f}, "
                        f"heur_s={heuristic_elapsed:.2f}, "
                        f"copy_s={copy_elapsed:.2f}, "
                        f"total_s={group_elapsed:.2f}"
                    )
                now = time.perf_counter()
                should_log_progress = (
                    groups_processed == total_groups
                    or (groups_processed % 25 == 0)
                    or (samples_processed == len(samples))
                    or (now - last_progress_t >= 30.0)
                )
                if should_log_progress:
                    elapsed = now - progress_t0
                    rate = samples_processed / max(elapsed, 1e-9)
                    print(
                        "[SemanticBackbone] scoring progress: "
                        f"groups={groups_processed}/{total_groups}, "
                        f"samples={samples_processed}/{len(samples)}, "
                        f"elapsed={elapsed:.1f}s, "
                        f"rate={rate:.1f} samples/s"
                    )
                    last_progress_t = now
                if rolling_provider is not None:
                    pos_indices = np.asarray(
                        [
                            int(sample_idx)
                            for sample_idx in batch_indices
                            if int(samples[int(sample_idx)]["label"]) == 1
                        ],
                        dtype=np.int64,
                    )
                    if len(pos_indices) > 0:
                        rolling_provider.commit_batch(
                            np.asarray(
                                [int(samples[idx]["source_id"]) for idx in pos_indices],
                                dtype=np.int64,
                            ),
                            np.asarray(
                                [int(samples[idx]["target_id"]) for idx in pos_indices],
                                dtype=np.int64,
                            ),
                            sample_timestamps[pos_indices],
                        )
                start = end

        if return_pair_features:
            return predictions, pair_features
        return predictions

    def _score_samples_training_batches(self, samples, sample_timestamps, precomputed_raw_features):
        query_spans = self._build_query_spans(samples)
        if not query_spans:
            return np.empty((0,), dtype=np.float64)

        predictions = np.zeros(len(samples), dtype=np.float64)
        batch_query_size = int(self.eval_positive_batch_size or 1)
        total_batches = (len(query_spans) + batch_query_size - 1) // batch_query_size
        progress_t0 = time.perf_counter()
        last_progress_t = progress_t0
        batches_processed = 0
        samples_processed = 0
        slow_batch_threshold_s = 5.0

        print(
            "[SemanticBackbone] scoring start: "
            f"total_samples={len(samples)}, total_queries={len(query_spans)}, "
            f"query_batch_size={batch_query_size}, total_query_batches={total_batches}, "
            "mode=training_eval_batches"
        )
        rolling_provider = self._build_rolling_provider(float(np.min(sample_timestamps)))

        with torch.inference_mode():
            for query_batch_start in range(0, len(query_spans), batch_query_size):
                query_batch_end = min(len(query_spans), query_batch_start + batch_query_size)
                sample_start = query_spans[query_batch_start][0]
                sample_end = query_spans[query_batch_end - 1][1]
                batch_indices = np.arange(sample_start, sample_end, dtype=np.int64)
                src_ids = np.asarray(
                    [int(samples[int(idx)]["source_id"]) for idx in batch_indices],
                    dtype=np.int64,
                )
                dst_ids = np.asarray(
                    [int(samples[int(idx)]["target_id"]) for idx in batch_indices],
                    dtype=np.int64,
                )
                times = sample_timestamps[batch_indices]
                query_time = float(np.min(times))

                if batches_processed < 8:
                    print(
                        "[SemanticBackbone] query batch: "
                        f"batch_idx={batches_processed}, "
                        f"query_range={query_batch_start}:{query_batch_end}, "
                        f"sample_range={sample_start}:{sample_end}, "
                        f"query_time={query_time:.6f}, size={int(len(batch_indices))}"
                    )

                batch_t0 = time.perf_counter()
                if rolling_provider is not None:
                    rolling_provider.set_base_embeddings(self.base_embeddings)
                    rolling_provider.prepare_batch(times)
                current_embeddings = self._get_current_embeddings(
                    query_time=query_time,
                    src_ids=src_ids,
                    dst_ids=dst_ids,
                    rolling_provider=rolling_provider,
                )
                batch_ncn_adj, batch_mplp_exact_adj2 = (
                    self._get_current_structural_adjs(
                        query_time=query_time,
                        rolling_provider=rolling_provider,
                    )
                )
                embedding_elapsed = time.perf_counter() - batch_t0
                score_t0 = time.perf_counter()
                batch_raw_features = (
                    precomputed_raw_features[batch_indices]
                    if precomputed_raw_features is not None
                    else None
                )
                auxiliary_features = None
                if self.semantic_aux_fusion_mode == "late_concat":
                    auxiliary_features = build_semantic_mlp_auxiliary_features(
                        heuristic_extractor=self.heuristic_extractor,
                        lookup=self.lookup,
                        sources=src_ids,
                        targets=dst_ids,
                        prediction_times=times,
                        raw_heuristic_features=batch_raw_features,
                        ncn_adj=batch_ncn_adj if self.use_mplp_exact_features else None,
                        two_hop_adj=batch_mplp_exact_adj2 if self.use_mplp_exact_features else None,
                        output_device=current_embeddings.device,
                        output_dtype=current_embeddings.dtype,
                        normalize_heuristics=not getattr(
                            self.model, "expects_raw_auxiliary_features", False
                        ),
                    )
                logits, _ = score_edge_batch(
                    self.model,
                    current_embeddings,
                    self.lookup,
                    src_ids,
                    dst_ids,
                    auxiliary_features=auxiliary_features,
                    node_interact_times=times,
                    neighbor_index=self.cross_attn_neighbor_index,
                    ncn_adj=batch_ncn_adj,
                    cross_attn_num_neighbors=self.cross_attn_num_neighbors,
                    ncn_num_neighbors=self.ncn_num_neighbors,
                    seqfilter_num_neighbors=self.seqfilter_num_neighbors,
                )
                model_elapsed = time.perf_counter() - score_t0
                heur_t0 = time.perf_counter()
                if self.semantic_aux_fusion_mode != "late_concat":
                    logits = self._apply_heuristic_fusion(
                        logits,
                        src_ids,
                        dst_ids,
                        times,
                        raw_features=batch_raw_features,
                    )
                    logits = self._apply_mplp_exact_fusion(
                        logits,
                        src_ids,
                        dst_ids,
                        ncn_adj=batch_ncn_adj,
                        mplp_exact_adj2=batch_mplp_exact_adj2,
                    )
                heuristic_elapsed = time.perf_counter() - heur_t0
                copy_t0 = time.perf_counter()
                logits_np = logits.detach().float().cpu().numpy().astype(np.float64, copy=False)
                copy_elapsed = time.perf_counter() - copy_t0
                predictions[batch_indices] = logits_np

                batches_processed += 1
                samples_processed += int(len(batch_indices))
                batch_elapsed = time.perf_counter() - batch_t0
                if batch_elapsed >= slow_batch_threshold_s:
                    print(
                        "[SemanticBackbone] slow query batch: "
                        f"batch_idx={batches_processed - 1}/{total_batches}, "
                        f"query_time={query_time:.6f}, size={int(len(batch_indices))}, "
                        f"embed_s={embedding_elapsed:.2f}, "
                        f"model_s={model_elapsed:.2f}, "
                        f"heur_s={heuristic_elapsed:.2f}, "
                        f"copy_s={copy_elapsed:.2f}, "
                        f"total_s={batch_elapsed:.2f}"
                    )
                now = time.perf_counter()
                should_log_progress = (
                    batches_processed == total_batches
                    or (batches_processed % 25 == 0)
                    or (samples_processed == len(samples))
                    or (now - last_progress_t >= 30.0)
                )
                if should_log_progress:
                    elapsed = now - progress_t0
                    rate = samples_processed / max(elapsed, 1e-9)
                    print(
                        "[SemanticBackbone] scoring progress: "
                        f"batches={batches_processed}/{total_batches}, "
                        f"samples={samples_processed}/{len(samples)}, "
                        f"elapsed={elapsed:.1f}s, "
                        f"rate={rate:.1f} samples/s"
                    )
                    last_progress_t = now
                if rolling_provider is not None:
                    pos_indices = np.asarray(
                        [query_spans[q][0] for q in range(query_batch_start, query_batch_end)],
                        dtype=np.int64,
                    )
                    rolling_provider.commit_batch(
                        np.asarray(
                            [int(samples[idx]["source_id"]) for idx in pos_indices],
                            dtype=np.int64,
                        ),
                        np.asarray(
                            [int(samples[idx]["target_id"]) for idx in pos_indices],
                            dtype=np.int64,
                        ),
                        sample_timestamps[pos_indices],
                    )

        return predictions

    def _score_samples_timestamp_rebuild_batches(
        self,
        samples,
        sample_timestamps,
        precomputed_raw_features,
    ):
        query_spans = self._build_query_spans(samples)
        if not query_spans:
            return np.empty((0,), dtype=np.float64)

        def _span_time(span):
            start, end = span
            return float(np.min(sample_timestamps[start:end]))

        ordered_spans = sorted(
            enumerate(query_spans),
            key=lambda item: (_span_time(item[1]), int(item[0])),
        )
        predictions = np.zeros(len(samples), dtype=np.float64)
        batch_query_size = int(self.eval_positive_batch_size or 1)
        total_batches = (len(ordered_spans) + batch_query_size - 1) // batch_query_size
        progress_t0 = time.perf_counter()
        last_progress_t = progress_t0
        batches_processed = 0
        samples_processed = 0
        slow_batch_threshold_s = 5.0

        print(
            "[SemanticBackbone] scoring start: "
            f"total_samples={len(samples)}, total_queries={len(query_spans)}, "
            f"query_batch_size={batch_query_size}, total_query_batches={total_batches}, "
            "mode=timestamp_rebuild_batches"
        )

        with torch.inference_mode():
            for query_batch_start in range(0, len(ordered_spans), batch_query_size):
                span_items = ordered_spans[
                    query_batch_start: query_batch_start + batch_query_size
                ]
                batch_indices = np.concatenate(
                    [
                        np.arange(start, end, dtype=np.int64)
                        for _, (start, end) in span_items
                    ],
                    axis=0,
                )
                src_ids = np.asarray(
                    [int(samples[int(idx)]["source_id"]) for idx in batch_indices],
                    dtype=np.int64,
                )
                dst_ids = np.asarray(
                    [int(samples[int(idx)]["target_id"]) for idx in batch_indices],
                    dtype=np.int64,
                )
                times = sample_timestamps[batch_indices]
                query_time = float(np.min(times))

                if batches_processed < 8:
                    print(
                        "[SemanticBackbone] timestamp-rebuild batch: "
                        f"batch_idx={batches_processed}, "
                        f"query_range={query_batch_start}:{query_batch_start + len(span_items)}, "
                        f"query_time={query_time:.6f}, size={int(len(batch_indices))}"
                    )

                batch_t0 = time.perf_counter()
                current_embeddings = self._get_current_embeddings(
                    query_time=query_time,
                    src_ids=src_ids,
                    dst_ids=dst_ids,
                    rolling_provider=None,
                )
                batch_ncn_adj, batch_mplp_exact_adj2 = self._get_current_structural_adjs(
                    query_time=query_time,
                    rolling_provider=None,
                )
                embedding_elapsed = time.perf_counter() - batch_t0
                score_t0 = time.perf_counter()
                batch_raw_features = (
                    precomputed_raw_features[batch_indices]
                    if precomputed_raw_features is not None
                    else None
                )
                auxiliary_features = None
                if self.semantic_aux_fusion_mode == "late_concat":
                    auxiliary_features = build_semantic_mlp_auxiliary_features(
                        heuristic_extractor=self.heuristic_extractor,
                        lookup=self.lookup,
                        sources=src_ids,
                        targets=dst_ids,
                        prediction_times=times,
                        raw_heuristic_features=batch_raw_features,
                        ncn_adj=batch_ncn_adj if self.use_mplp_exact_features else None,
                        two_hop_adj=batch_mplp_exact_adj2 if self.use_mplp_exact_features else None,
                        output_device=current_embeddings.device,
                        output_dtype=current_embeddings.dtype,
                        normalize_heuristics=not getattr(
                            self.model, "expects_raw_auxiliary_features", False
                        ),
                    )
                logits, _ = score_edge_batch(
                    self.model,
                    current_embeddings,
                    self.lookup,
                    src_ids,
                    dst_ids,
                    auxiliary_features=auxiliary_features,
                    node_interact_times=times,
                    neighbor_index=self.cross_attn_neighbor_index,
                    ncn_adj=batch_ncn_adj,
                    cross_attn_num_neighbors=self.cross_attn_num_neighbors,
                    ncn_num_neighbors=self.ncn_num_neighbors,
                    seqfilter_num_neighbors=self.seqfilter_num_neighbors,
                )
                model_elapsed = time.perf_counter() - score_t0
                heur_t0 = time.perf_counter()
                if self.semantic_aux_fusion_mode != "late_concat":
                    logits = self._apply_heuristic_fusion(
                        logits,
                        src_ids,
                        dst_ids,
                        times,
                        raw_features=batch_raw_features,
                    )
                    logits = self._apply_mplp_exact_fusion(
                        logits,
                        src_ids,
                        dst_ids,
                        ncn_adj=batch_ncn_adj,
                        mplp_exact_adj2=batch_mplp_exact_adj2,
                    )
                heuristic_elapsed = time.perf_counter() - heur_t0
                copy_t0 = time.perf_counter()
                logits_np = logits.detach().float().cpu().numpy().astype(np.float64, copy=False)
                copy_elapsed = time.perf_counter() - copy_t0
                predictions[batch_indices] = logits_np

                batches_processed += 1
                samples_processed += int(len(batch_indices))
                batch_elapsed = time.perf_counter() - batch_t0
                if batch_elapsed >= slow_batch_threshold_s:
                    print(
                        "[SemanticBackbone] slow timestamp-rebuild batch: "
                        f"batch_idx={batches_processed - 1}/{total_batches}, "
                        f"query_time={query_time:.6f}, size={int(len(batch_indices))}, "
                        f"embed_s={embedding_elapsed:.2f}, "
                        f"model_s={model_elapsed:.2f}, "
                        f"heur_s={heuristic_elapsed:.2f}, "
                        f"copy_s={copy_elapsed:.2f}, "
                        f"total_s={batch_elapsed:.2f}"
                    )
                now = time.perf_counter()
                should_log_progress = (
                    batches_processed == total_batches
                    or (batches_processed % 25 == 0)
                    or (samples_processed == len(samples))
                    or (now - last_progress_t >= 30.0)
                )
                if should_log_progress:
                    elapsed = now - progress_t0
                    rate = samples_processed / max(elapsed, 1e-9)
                    print(
                        "[SemanticBackbone] scoring progress: "
                        f"batches={batches_processed}/{total_batches}, "
                        f"samples={samples_processed}/{len(samples)}, "
                        f"elapsed={elapsed:.1f}s, "
                        f"rate={rate:.1f} samples/s"
                    )
                    last_progress_t = now

        return predictions

    def _precompute_sample_heuristics(self, samples, sample_timestamps):
        precomputed_raw_features = None
        if self.heuristic_extractor is not None and (
            self.heuristic_fusion is not None or self.semantic_aux_fusion_mode == "late_concat"
        ):
            heuristic_t0 = time.perf_counter()
            print(
                "[SemanticBackbone] collecting source/target arrays for heuristic precompute...",
                flush=True,
            )
            all_sources = np.asarray(
                [int(sample["source_id"]) for sample in samples],
                dtype=np.int64,
            )
            all_targets = np.asarray(
                [int(sample["target_id"]) for sample in samples],
                dtype=np.int64,
            )
            print(
                "[SemanticBackbone] precomputing heuristic features: "
                f"samples={len(samples)}, features={','.join(self.heuristic_extractor.feature_names)}",
                flush=True,
            )
            precomputed_raw_features = self.heuristic_extractor.precompute_raw_features(
                sources=all_sources,
                targets=all_targets,
                prediction_times=sample_timestamps,
                desc="Semantic backbone heuristics",
                store_in_cache=False,
            )
            heuristic_elapsed = time.perf_counter() - heuristic_t0
            print(
                "[SemanticBackbone] heuristic precompute ready: "
                f"samples={len(samples)}, elapsed={heuristic_elapsed:.1f}s",
                flush=True,
            )
        return precomputed_raw_features

    @staticmethod
    def _sample_timestamps(samples):
        return np.asarray(
            [
                float(sample.get("dtgb_timestamp", sample["timestamp"]))
                for sample in samples
            ],
            dtype=np.float64,
        )

    def score_samples(self, samples):
        if not samples:
            return np.empty((0,), dtype=np.float64)
        scorer = self._scorer_for_samples(samples)
        if scorer is not self:
            return scorer.score_samples(samples)

        print(
            "[SemanticBackbone] score_samples start: "
            f"samples={len(samples)}, temporal_mode={self.temporal_mode}, "
            f"eval_positive_batch_size={self.eval_positive_batch_size}, "
            f"heuristic_fusion={self.heuristic_extractor is not None and self.heuristic_fusion is not None}",
            flush=True,
        )
        sample_timestamps = self._sample_timestamps(samples)
        precomputed_raw_features = self._precompute_sample_heuristics(
            samples,
            sample_timestamps,
        )

        if self.temporal_mode == "timestamp_rebuild":
            if self.eval_positive_batch_size is not None:
                return self._score_samples_timestamp_rebuild_batches(
                    samples,
                    sample_timestamps,
                    precomputed_raw_features,
                )
            return self._score_samples_exact_timestamp(
                samples,
                sample_timestamps,
                precomputed_raw_features,
                use_rolling_provider=False,
            )

        if self.eval_positive_batch_size is not None:
            return self._score_samples_training_batches(
                samples,
                sample_timestamps,
                precomputed_raw_features,
            )
        return self._score_samples_exact_timestamp(
            samples,
            sample_timestamps,
            precomputed_raw_features,
        )

    def score_samples_with_pair_features(self, samples):
        """Return final checkpoint logits and the aligned pre-head graph pair vectors."""
        if samples:
            scorer = self._scorer_for_samples(samples)
            if scorer is not self:
                return scorer.score_samples_with_pair_features(samples)
        if not samples:
            if not isinstance(self.model, SemanticMLP):
                raise TypeError("Pair-representation extraction requires a SemanticMLP checkpoint.")
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0, int(self.model.pair_representation_dim)), dtype=np.float32),
            )
        sample_timestamps = self._sample_timestamps(samples)
        precomputed_raw_features = self._precompute_sample_heuristics(
            samples,
            sample_timestamps,
        )
        return self._score_samples_exact_timestamp(
            samples,
            sample_timestamps,
            precomputed_raw_features,
            use_rolling_provider=self.temporal_mode != "timestamp_rebuild",
            return_pair_features=True,
        )

    def annotate_samples(self, samples, *, score_field=None):
        target_field = str(score_field or self.score_field)
        scores = self.score_samples(samples)
        for sample, score in zip(samples, scores):
            sample[target_field] = float(score)
        return samples

    def evaluate_samples(self, samples, *, dtgb_eval_batch_size=None, score_field=None):
        target_field = str(score_field or self.score_field)
        self.annotate_samples(samples, score_field=target_field)
        predictions = np.asarray(
            [float(sample.get(target_field, 0.0)) for sample in samples],
            dtype=np.float64,
        )
        labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
        metrics = compute_prediction_metrics(
            predictions,
            labels,
            dtgb_eval_batch_size=dtgb_eval_batch_size,
        )
        metrics["parse_stats"] = {"mode": "semantic_mlp_backbone"}
        metrics["detailed_results"] = []
        return metrics


__all__ = ["SemanticMLPHybridBackbone"]
