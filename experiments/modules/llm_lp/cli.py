import argparse
import math
import os
import shutil

from utils.seed_runs import DEFAULT_SEEDS, add_seed_arguments

from experiments.modules.llm_lp.prompt_template import (
    DEFAULT_KEY_SIGNAL_FIELDS,
    SUPPORTED_KEY_SIGNAL_FIELDS,
    normalize_key_signal_fields,
)
from experiments.modules.rrf.scoring import (
    DEFAULT_RRF_HEURISTICS,
    SUPPORTED_RRF_HEURISTICS,
    normalize_rrf_heuristics,
)

SUPPORTED_SEMANTIC_FUSION_HEURISTICS = (
    "recency",
    "popularity",
    "recent_degree",
    "global_recency",
    "past",
    "ra",
    "itemcf",
    "usercf",
    "city_preference",
    "zip_preference",
)
DEFAULT_SEMANTIC_FUSION_HEURISTICS = (
    "recency",
    "popularity",
    "past",
    "ra",
)
SEMANTIC_FUSION_HEURISTIC_ALIASES = {
    "itemcf_cosine": "itemcf",
    "usercf_cosine": "usercf",
    "personalized_city": "city_preference",
    "personalized_zip": "zip_preference",
    "past_interaction": "past",
    "past_interactions": "past",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    text = str(v).strip().lower()
    if text in {"yes", "true", "t", "y", "1"}:
        return True
    if text in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def apply_compact_activity_prompt_recipe(args):
    """Expand the unified compact-activity recipe into canonical prompt flags."""
    top_k = int(getattr(args, "compact_activity_top_k", 3))
    if top_k < 1:
        raise ValueError("--compact_activity_top_k must be >= 1")
    neighbor_top_k = int(getattr(args, "natural_neighbor_top_k", 5))
    if neighbor_top_k < 1:
        raise ValueError("--natural_neighbor_top_k must be >= 1")
    args.natural_activity_top_k = (
        neighbor_top_k
        if bool(getattr(args, "natural_neighbor_names_only", False))
        and not bool(getattr(args, "compact_activity_prompt", False))
        else top_k
    )

    if bool(getattr(args, "mutual_summary_count_recency", False)):
        # The control is a strict compression of the compact baseline's
        # timestamp-only, consecutive-deduplicated mutual-history view.
        args.mutual_timestamps_only = True
        args.mutual_timestamps_dedup = True

    if not bool(getattr(args, "compact_activity_prompt", False)):
        return args

    conflicting_modes = [
        name
        for name in (
            "natural_grouped_history",
            "natural_activity_summary",
            "natural_neighbor_names_only",
        )
        if bool(getattr(args, name, False))
    ]
    if conflicting_modes:
        flags = ", ".join(f"--{name}" for name in conflicting_modes)
        raise ValueError(
            "--compact_activity_prompt cannot be combined with alternative "
            f"history renderers: {flags}"
        )

    args.natural_activity_compact_top3 = True
    args.mutual_timestamps_only = True
    args.mutual_timestamps_dedup = True
    # Plain compact prompts keep their historical no-CN behavior.  When the
    # caller explicitly requests semantic common neighbors, preserve that
    # section so compact activity and semantic-CN can be studied together.
    args.ablate_common_neighbors = bool(
        getattr(args, "ablate_common_neighbors", False)
    ) or not bool(getattr(args, "common_neighbors_semantic", False))
    args.hide_key_signals = True
    args.hide_expert_prediction = True
    args.ablate_reasoning_guidance = True
    return args


def add_heuristic_batch_size_arg(
    parser,
    *,
    flag="--rrf_batch_size",
    default=200000,
    help_text="Batch size for heuristic scoring",
):
    parser.add_argument(
        flag,
        type=int,
        default=int(default),
        help=help_text,
    )
    parser.set_defaults(history_direction="both", history_protocol="both_endpoints_recent_v1",
                        interaction_count_direction="source_to_target")
    return parser


def normalize_semantic_fusion_heuristics(value):
    if value is None:
        return DEFAULT_SEMANTIC_FUSION_HEURISTICS
    if isinstance(value, str):
        raw_tokens = [token.strip().lower() for token in value.split(",")]
    else:
        raw_tokens = [str(token).strip().lower() for token in value]
    tokens = [token for token in raw_tokens if token]
    if not tokens or tokens == ["all"]:
        return DEFAULT_SEMANTIC_FUSION_HEURISTICS
    tokens = [SEMANTIC_FUSION_HEURISTIC_ALIASES.get(token, token) for token in tokens]

    supported = set(SUPPORTED_SEMANTIC_FUSION_HEURISTICS)
    unknown = [token for token in tokens if token not in supported]
    if unknown:
        supported_text = ", ".join(SUPPORTED_SEMANTIC_FUSION_HEURISTICS)
        raise ValueError(
            "Unsupported semantic fusion heuristic(s): "
            f"{', '.join(unknown)}. Supported values: {supported_text}."
        )

    selected = tuple(
        name for name in SUPPORTED_SEMANTIC_FUSION_HEURISTICS if name in set(tokens)
    )
    if not selected:
        raise ValueError("At least one semantic fusion heuristic must be selected.")
    return selected


def build_semantic_mlp_arg_parser(str2bool):
    parser = argparse.ArgumentParser(description="Train semantic pair scorer on smoothed embeddings")

    parser.add_argument("--dataset_name", type=str, default="GDELT")
    parser.add_argument(
        "--embedding_entity_name_mode",
        type=str,
        default="raw",
        choices=["raw", "auto", "compressed", "compressed_profile"],
        help=(
            "Entity-text preprocessing used before computing base text embeddings. "
            "'compressed' applies dataset-specific compaction "
            "(ICEWS1819 body names, Googlemap_CT business cards); "
            "default 'raw' preserves the original behavior."
        ),
    )
    parser.add_argument(
        "--entity_text_path",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset entity_text CSV used to initialize text-derived "
            "semantic embeddings. When omitted, defaults to ../DyLink_Datasets/<dataset>/entity_text.csv."
        ),
    )
    parser.add_argument(
        "--use_learnable_entity_embeddings",
        type=str2bool,
        default=False,
        help=(
            "Bypass text-derived initialization and learn one embedding per raw node id from scratch. "
            "Useful for datasets without entity_text.csv."
        ),
    )
    parser.add_argument(
        "--learnable_entity_embedding_dim",
        type=int,
        default=256,
        help=(
            "Embedding dimension used when --use_learnable_entity_embeddings=true. "
            "Ignored when text-derived base embeddings are used."
        ),
    )
    parser.add_argument(
        "--learnable_entity_embedding_init",
        type=str,
        default="normal",
        choices=["normal", "orthogonal", "zero"],
        help=(
            "Initialization for raw-node-id embedding tables. "
            "'orthogonal' creates fixed random orthogonal-noise features when combined with "
            "--freeze_learnable_entity_embeddings=true; "
            "'zero' initializes every entity embedding to zero for collapse/bias-only ablations."
        ),
    )
    parser.add_argument(
        "--freeze_learnable_entity_embeddings",
        type=str2bool,
        default=False,
        help=(
            "Freeze the raw-node-id embedding table after initialization. "
            "Useful for testing fixed random-feature baselines."
        ),
    )
    parser.add_argument(
        "--use_structural_seed_features",
        type=str2bool,
        default=False,
        help=(
            "Add trainable projected per-node structural seed features before "
            "semantic smoothing/message passing. This gives the graph operator "
            "nonzero node signals to propagate even when entity embeddings are zero."
        ),
    )
    parser.add_argument(
        "--semantic_project_dim",
        type=int,
        default=0,
        help=(
            "Optional trainable down-projection dimension applied to materialized semantic "
            "embeddings before smoothing/message passing and scoring. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--semantic_smoothing_source_init",
        type=str,
        default="raw",
        choices=["raw", "history_mean"],
        help=(
            "How to initialize source/user embeddings before semantic smoothing. "
            "'history_mean' replaces each source/user node with the mean of its observed "
            "destination/item embeddings before smoothing."
        ),
    )
    add_seed_arguments(parser)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use; set -1 to force CPU",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Training batch size; defaults to --batch_size when not set",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Validation/test batch size; defaults to --batch_size when not set",
    )
    parser.add_argument(
        "--dtgb_eval_batch_size",
        type=int,
        default=None,
        help=(
            "Number of positive queries per DTGB metric batch for semantic scorer evaluation; "
            "defaults to --eval_batch_size when not set."
        ),
    )
    parser.add_argument(
        "--eval_time_buckets",
        type=int,
        default=0,
        help=(
            "Optional number of timestamp-ordered buckets to report during final evaluation. "
            "0 disables bucketed diagnostics."
        ),
    )
    parser.add_argument(
        "--report_train_metrics",
        type=str2bool,
        default=False,
        help=(
            "Evaluate and log train AP/AUC/MRR after each epoch. "
            "Disabled by default because it adds a full train-set evaluation pass."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of PyTorch DataLoader workers used for index loaders. "
            "Set to 0 to rule out worker deadlocks."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer used for the semantic pair scorer.",
    )
    parser.add_argument(
        "--negative_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive"],
    )
    parser.add_argument("--train_num_negatives", type=int, default=3)
    parser.add_argument(
        "--freeze_train_negatives_per_epoch",
        type=str2bool,
        default=True,
        help=(
            "Sample the full train negative pool once at the start of each epoch and reuse it "
            "for all train batches in that epoch; when heuristic fusion is enabled, negative "
            "heuristics are also precomputed once per epoch. Faster, but usually worse for "
            "training quality because negative diversity drops within each epoch."
        ),
    )
    parser.add_argument(
        "--train_rand_ratio",
        type=float,
        default=None,
        help=(
            "If set in [0,1], training negatives are mixed: "
            "random ratio=train_rand_ratio, historical ratio=1-ratio"
        ),
    )
    parser.add_argument(
        "--historical_neg_min_gap",
        type=float,
        default=0.0,
        help=(
            "Minimum time gap since last occurrence required for historical negatives; "
            "larger values make them easier/safer"
        ),
    )
    parser.add_argument(
        "--historical_neg_strict_gap",
        type=str2bool,
        default=False,
        help=(
            "When true, historical branch never backfills with random; "
            "missing historical quota is reassigned to random branch in mixer"
        ),
    )
    parser.add_argument("--val_num_negatives", type=int, default=1)
    parser.add_argument("--eval_num_negatives", type=int, default=1)

    parser.add_argument(
        "--scorer_type",
        type=str,
        default="mlp",
        choices=["mlp", "ridge", "cross_attention", "dygformer_lite", "ncn", "seqfilter"],
        help="Pair scorer architecture: plain MLP, closed-form semantic ridge, lightweight cross-attention, DyGFormer-lite pair transformer, simple NCN, or SeqFilter-style scorer",
    )
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=1.0,
        help="L2 regularization for the closed-form pairwise ridge scorer.",
    )
    parser.add_argument(
        "--ridge_projection_dim",
        type=int,
        default=256,
        help=(
            "Fixed random projection width applied before smoothing for scorer_type=ridge. "
            "Set 0 to keep the original semantic width."
        ),
    )
    parser.add_argument(
        "--ridge_semantic_feature_mode",
        type=str,
        default="hadamard",
        choices=["hadamard", "cosine"],
        help=(
            "Semantic part of the ridge edge vector: keep the projected element-wise "
            "product, or collapse it immediately to one cosine-similarity scalar."
        ),
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "silu"])
    parser.add_argument(
        "--use_layernorm",
        type=str2bool,
        default=True,
        help="Enable LayerNorm blocks: true/false",
    )
    parser.add_argument(
        "--cross_attn_num_neighbors",
        type=int,
        default=20,
        help="Number of recent source-history neighbors for CRAFT-style cross attention",
    )
    parser.add_argument(
        "--cross_attn_num_layers",
        type=int,
        default=2,
        help="Number of transformer layers inside the cross-attention scorer",
    )
    parser.add_argument(
        "--cross_attn_heads",
        type=int,
        default=2,
        help="Number of attention heads inside the cross-attention scorer",
    )
    parser.add_argument(
        "--cross_attn_use_pos",
        type=str2bool,
        default=True,
        help="Add learned positional embeddings to source-history tokens for cross attention",
    )
    parser.add_argument(
        "--cross_attn_add_time_to_history",
        type=str2bool,
        default=False,
        help="Add learned relative-time embeddings directly to source-history tokens for cross attention",
    )
    parser.add_argument(
        "--cross_attn_hidden_dropout",
        type=float,
        default=0.5,
        help="Hidden/output dropout used inside the cross-attention stack",
    )
    parser.add_argument(
        "--cross_attn_attn_dropout",
        type=float,
        default=0.5,
        help="Attention-probability dropout used inside the cross-attention stack",
    )
    parser.add_argument(
        "--cross_attn_emb_dropout",
        type=float,
        default=0.1,
        help="Embedding/input dropout applied to query and history tokens before cross attention",
    )
    parser.add_argument(
        "--cross_attn_use_raw_embeddings",
        type=str2bool,
        default=True,
        help="Use unsmoothed base embeddings for the CA scorer, matching the healthy CRAFT semantic run",
    )
    parser.add_argument(
        "--ncn_num_neighbors",
        type=int,
        default=50,
        help="Number of recent neighbors per endpoint used to form the NCN common-neighbor set",
    )
    parser.add_argument(
        "--dygformer_num_neighbors",
        type=int,
        default=20,
        help="Recent neighbors per endpoint used by the DyGFormer-lite pair transformer",
    )
    parser.add_argument(
        "--dygformer_num_layers",
        type=int,
        default=1,
        help="Number of self-attention layers in the DyGFormer-lite pair transformer",
    )
    parser.add_argument(
        "--dygformer_heads",
        type=int,
        default=4,
        help="Number of attention heads in the DyGFormer-lite pair transformer",
    )
    parser.add_argument(
        "--dygformer_add_time",
        type=str2bool,
        default=True,
        help="Add relative-time encodings to DyGFormer-lite history tokens: true/false",
    )
    parser.add_argument(
        "--seqfilter_num_neighbors",
        type=int,
        default=32,
        help="Recent neighbors per endpoint used by the SeqFilter scorer",
    )
    parser.add_argument(
        "--seqfilter_tau",
        type=float,
        default=0.2,
        help="Relative energy threshold for SeqFilter frequency masking",
    )
    parser.add_argument(
        "--seqfilter_kernel_size",
        type=int,
        default=3,
        help="Odd Conv1D kernel size used by SeqFilter rhythm and structure filters",
    )
    parser.add_argument(
        "--seqfilter_use_soft_mask",
        type=str2bool,
        default=True,
        help="Use a differentiable soft frequency mask in SeqFilter: true/false",
    )
    parser.add_argument(
        "--use_mplp_exact_features",
        type=str2bool,
        default=True,
        help=(
            "Fuse exact MPLP-style high-order structural overlap features into semantic logits. "
            "This uses the current binary temporal history graph and its exact 2-hop closure."
        ),
    )
    parser.add_argument(
        "--semantic_aux_fusion_mode",
        type=str,
        default="residual",
        choices=["residual", "late_concat"],
        help=(
            "How heuristic/MPLP auxiliary features are fused with semantic logits. "
            "'residual' uses semantic_logit + linear(features). "
            "'late_concat' is MLP-only and appends features to the selected "
            "semantic MLP pair representation."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_pair_feature_mode",
        type=str,
        default="concat",
        choices=["concat", "hadamard"],
        help=(
            "MLP-only edge representation before optional auxiliary input concat. "
            "'concat' uses [src_emb, dst_emb]. "
            "'hadamard' uses src_emb * dst_emb."
        ),
    )
    parser.add_argument(
        "--mplp_signature_dim",
        type=int,
        default=1024,
        choices=[512, 1024, 2048],
        help=(
            "Signature dimension for MPLP-style structural modes. "
            "Currently reserved for future approximate MPLP variants; "
            "the exact MPLP feature path does not consume it."
        ),
    )
    parser.add_argument(
        "--use_heuristic_features",
        type=str2bool,
        default=True,
        help="Fuse CRAFT-style heuristics into semantic logits",
    )
    parser.add_argument(
        "--heuristic_feature_names",
        type=str,
        default=",".join(DEFAULT_SEMANTIC_FUSION_HEURISTICS),
        help=(
            "Comma-separated heuristic subset to inject into semantic logits. "
            f"Supported values: {', '.join(SUPPORTED_SEMANTIC_FUSION_HEURISTICS)}. "
            "Use 'all' for the default full set."
        ),
    )
    parser.add_argument(
        "--heuristic_popularity_decay",
        type=float,
        default=0.0,
        help="Popularity decay lambda for heuristic feature extraction",
    )
    parser.add_argument(
        "--heuristic_recency_directed",
        type=str2bool,
        default=False,
        help=(
            "Compute the recency heuristic from directed source-to-target history. "
            "The default false preserves legacy undirected checkpoint behavior."
        ),
    )
    parser.add_argument(
        "--heuristic_recent_degree_window",
        type=float,
        default=50.0,
        help="Hard lookback window for the recent_degree heuristic feature",
    )
    parser.add_argument(
        "--use_gpu_heuristics",
        type=str2bool,
        default=True,
        help="Use GPU kernel for RA heuristic when CUDA is available",
    )
    add_heuristic_batch_size_arg(
        parser,
        flag="--heuristic_score_batch_size",
        default=200000,
        help_text="Chunk size for batched heuristic precompute/cache fills",
    )
    parser.add_argument(
        "--use_learnable_gcn",
        type=str2bool,
        default=True,
        help="Enable learnable message passing on top of current embeddings: true/false",
    )
    parser.add_argument(
        "--gcn_num_layers",
        type=int,
        default=1,
        help="Number of learnable GCN layers when --use_learnable_gcn=true",
    )
    parser.add_argument(
        "--gcn_use_linear_transform",
        type=str2bool,
        default=True,
        help=(
            "Apply a learned linear transformation inside each GCN layer: true/false. "
            "When false, GCN uses pure normalized propagation A_norm @ x; "
            "LayerNorm and residual connections remain enabled."
        ),
    )
    parser.add_argument(
        "--use_temporal_relational_gcn",
        type=str2bool,
        default=False,
        help=(
            "Augment rolling GCN with cached relation-text, relative-time, and "
            "directed-history summaries: true/false"
        ),
    )
    parser.add_argument(
        "--temporal_relational_rank",
        type=int,
        default=32,
        help="Low-rank width for the temporal-relational GCN residual branch",
    )
    parser.add_argument(
        "--temporal_relational_time_basis_dim",
        type=int,
        default=16,
        help="Compact fixed age-basis width before its learned projection",
    )
    parser.add_argument(
        "--temporal_relational_edge_feature_path",
        type=str,
        default=None,
        help=(
            "Optional relation feature .npy path; defaults to "
            "../DyLink_Datasets/<dataset>/r_feat.npy"
        ),
    )
    parser.add_argument(
        "--gin_nonparametric",
        type=str2bool,
        default=False,
        help=(
            "Use parameter-free GIN-style self-plus-neighbor-sum pooling: true/false. "
            "When true, the operator is norm(x + sum_neighbors), with fixed self "
            "coefficient 1 and no learned epsilon, MLP transformation, affine norm, "
            "or outer residual."
        ),
    )
    parser.add_argument(
        "--gin_nonparametric_norm",
        type=str,
        default="bn",
        choices=["ln", "bn"],
        help=(
            "Normalization for --gin_nonparametric=true: ln applies affine-free "
            "LayerNorm per node; bn applies affine-free BatchNorm across graph nodes."
        ),
    )
    parser.add_argument(
        "--learnable_mp_type",
        type=str,
        default="gin",
        choices=["gcn", "gin", "attn_pool"],
        help="Learnable message-passing operator when --use_learnable_gcn=true",
    )
    parser.add_argument(
        "--attn_mp_heads",
        type=int,
        default=4,
        help="Number of heads for attention-pooling message passing",
    )
    parser.add_argument(
        "--attn_mp_num_neighbors",
        type=int,
        default=50,
        help="Recent neighbors per node for attention-pooling message passing",
    )
    parser.add_argument(
        "--time_encoder_type",
        type=str,
        default="mlp",
        choices=["mlp", "fourier", "rbf"],
        help="Relative-time encoder style for CA scorer and attention-pooling MP",
    )
    parser.add_argument(
        "--time_encoder_mask_padding",
        type=str2bool,
        default=True,
        help="Mask padded history positions before adding time encoding: true/false",
    )
    parser.add_argument(
        "--time_encoder_fourier_dim",
        type=int,
        default=32,
        help="Basis dimension for fourier relative-time encoder",
    )
    parser.add_argument(
        "--time_encoder_rbf_dim",
        type=int,
        default=32,
        help="Number of RBF kernels for rbf relative-time encoder",
    )

    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Global grad-norm clipping value; <=0 disables clipping",
    )
    parser.add_argument("--scheduler", type=str, default="linear", choices=["none", "cosine", "linear"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Binary label smoothing in [0, 0.5)",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop if no val AP improvement for N epochs; <=0 disables",
    )
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)

    parser.add_argument(
        "--train_edge_cutoff_time",
        type=float,
        default=None,
        help="Recency cutoff for training: keep only edges with time >= cutoff_time",
    )
    parser.add_argument(
        "--train_edge_cutoff_ratio",
        type=float,
        default=1.0,
        help="Keep most recent ratio of train edges in (0,1]",
    )
    parser.add_argument(
        "--train_holdout_recent_edges",
        type=int,
        default=0,
        help=(
            "Exclude the N most-recent observed train edges from GNN parameter "
            "optimization. Intended for a labeled calibration/ICL tail."
        ),
    )

    parser.add_argument("--smooth_time_window", type=float, default=50)
    parser.add_argument(
        "--use_semantic_smoothing",
        type=str2bool,
        default=True,
        help=(
            "Apply graph-based semantic smoothing to base entity embeddings: true/false. "
            "Set false to feed raw base semantic embeddings directly to the pair scorer."
        ),
    )
    parser.add_argument("--smooth_steps", type=int, default=1)
    parser.add_argument(
        "--smooth_decay_gamma",
        type=float,
        default=0.1,
        help="Decay gamma for semantic smoothing edge weights in the semantic MLP pipeline",
    )
    parser.add_argument(
        "--smooth_undirected",
        type=str2bool,
        default=True,
        help="Use undirected smoothing graph: true/false",
    )
    parser.add_argument(
        "--smooth_log_dampen",
        type=str2bool,
        default=True,
        help="Apply log dampening on edge weights: true/false",
    )
    parser.add_argument("--smooth_supernode_strength", type=float, default=0.5)
    parser.add_argument(
        "--smooth_endpoint_topk_recent",
        type=int,
        default=0,
        help=(
            "If >0, restrict semantic smoothing/message passing to each node's "
            "latest K interactions inside --smooth_time_window."
        ),
    )
    parser.add_argument(
        "--smooth_endpoint_topk_mode",
        type=str,
        default="per_node",
        choices=["per_node", "union"],
        help=(
            "per_node gives each aggregation row its own latest-K history; union "
            "retains the legacy symmetric union of endpoint selections."
        ),
    )
    parser.add_argument(
        "--smooth_cutoff_time",
        type=float,
        default=None,
        help="If set, smoothing graph keeps edges with time < cutoff_time",
    )
    parser.add_argument(
        "--rolling_smoothing",
        type=str2bool,
        default=True,
        help="Rolling smoothing per batch: true/false",
    )
    parser.add_argument(
        "--strict_no_leakage",
        type=str2bool,
        default=True,
        help="Use only train+val history for smoothing: true/false",
    )
    parser.add_argument(
        "--profile_runtime",
        type=str2bool,
        default=False,
        help="Enable lightweight batch-stage profiling: true/false",
    )
    parser.add_argument(
        "--debug_negative_precompute",
        type=str2bool,
        default=False,
        help=(
            "Print timestamped boundary logs around precomputed negative-query loops "
            "to identify whether a stall occurs on DataLoader fetch or negative sampling."
        ),
    )
    parser.add_argument(
        "--debug_negative_precompute_every",
        type=int,
        default=25,
        help="When debug_negative_precompute=true, print every N batches after the first five.",
    )
    parser.add_argument(
        "--profile_batches",
        type=int,
        default=20,
        help="Number of batches to profile per split when --profile_runtime=true",
    )
    parser.add_argument(
        "--profile_skip_batches",
        type=int,
        default=20,
        help="Skip first N warmup batches before profiling each split",
    )
    parser.add_argument(
        "--torch_profile_steps",
        type=int,
        default=0,
        help="If > 0, run torch.profiler on the first training epoch for this many active steps.",
    )
    parser.add_argument(
        "--torch_profile_warmup_steps",
        type=int,
        default=1,
        help="Warmup steps before active capture when --torch_profile_steps > 0.",
    )
    parser.add_argument(
        "--torch_profile_dir",
        type=str,
        default="result/torch_profiles",
        help="Directory for torch.profiler traces and summary tables.",
    )
    parser.add_argument(
        "--torch_profile_row_limit",
        type=int,
        default=50,
        help="Row limit for the exported torch.profiler operator summary table.",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="intfloat/e5-large-v2",
        help="Hugging Face/SentenceTransformers model used to compute entity text embeddings.",
    )
    parser.add_argument("--embedding_cache", type=str, default=None)
    parser.add_argument("--smoothed_embedding_cache", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="best_semantic_mlp.pt")
    parser.set_defaults(history_direction="both", history_protocol="both_endpoints_recent_v1",
                        interaction_count_direction="source_to_target")
    return parser


def validate_semantic_mlp_args(args):
    if not (0.0 < args.train_edge_cutoff_ratio <= 1.0):
        raise ValueError("--train_edge_cutoff_ratio must be in (0, 1].")
    if args.train_holdout_recent_edges < 0:
        raise ValueError("--train_holdout_recent_edges must be >= 0.")
    if args.smooth_endpoint_topk_recent < 0:
        raise ValueError("--smooth_endpoint_topk_recent must be >= 0.")
    if args.smooth_endpoint_topk_recent > 0 and args.smooth_endpoint_topk_mode == "per_node" and not args.smooth_undirected:
        raise ValueError(
            "--smooth_endpoint_topk_mode=per_node requires --smooth_undirected=true."
        )
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup_ratio must be in [0, 1).")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min_lr_ratio must be in [0, 1].")
    if not (0.0 <= args.label_smoothing < 0.5):
        raise ValueError("--label_smoothing must be in [0, 0.5).")
    if args.train_rand_ratio is not None and not (0.0 <= args.train_rand_ratio <= 1.0):
        raise ValueError("--train_rand_ratio must be in [0, 1].")
    if args.historical_neg_min_gap < 0.0:
        raise ValueError("--historical_neg_min_gap must be >= 0.")
    if args.heuristic_popularity_decay < 0.0:
        raise ValueError("--heuristic_popularity_decay must be >= 0.")
    if args.heuristic_recent_degree_window <= 0.0:
        raise ValueError("--heuristic_recent_degree_window must be > 0.")
    if args.heuristic_score_batch_size <= 0:
        raise ValueError("--heuristic_score_batch_size must be > 0.")
    if args.semantic_aux_fusion_mode == "late_concat" and args.scorer_type not in {"mlp", "ridge"}:
        raise ValueError("--semantic_aux_fusion_mode=late_concat is supported only with --scorer_type=mlp/ridge.")
    if args.semantic_mlp_pair_feature_mode != "concat" and args.scorer_type not in {"mlp", "ridge"}:
        raise ValueError("--semantic_mlp_pair_feature_mode is supported only with --scorer_type=mlp/ridge.")
    if args.ridge_lambda <= 0.0:
        raise ValueError("--ridge_lambda must be > 0.")
    if args.ridge_projection_dim < 0:
        raise ValueError("--ridge_projection_dim must be >= 0.")
    if args.smooth_decay_gamma < 0.0:
        raise ValueError("--smooth_decay_gamma must be >= 0.")
    args.heuristic_feature_names = normalize_semantic_fusion_heuristics(
        getattr(args, "heuristic_feature_names", None)
    )
    if args.train_batch_size is None:
        args.train_batch_size = args.batch_size
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.dtgb_eval_batch_size is None:
        args.dtgb_eval_batch_size = args.eval_batch_size
    if args.train_batch_size <= 0:
        raise ValueError("--train_batch_size must be > 0.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be > 0.")
    if args.dtgb_eval_batch_size <= 0:
        raise ValueError("--dtgb_eval_batch_size must be > 0.")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader_num_workers must be >= 0.")
    if args.debug_negative_precompute_every <= 0:
        raise ValueError("--debug_negative_precompute_every must be > 0.")
    if args.profile_batches <= 0:
        raise ValueError("--profile_batches must be > 0.")
    if args.profile_skip_batches < 0:
        raise ValueError("--profile_skip_batches must be >= 0.")
    if args.cross_attn_num_neighbors <= 0:
        raise ValueError("--cross_attn_num_neighbors must be > 0.")
    if args.dygformer_num_neighbors <= 0:
        raise ValueError("--dygformer_num_neighbors must be > 0.")
    if args.dygformer_num_layers <= 0:
        raise ValueError("--dygformer_num_layers must be > 0.")
    if args.dygformer_heads <= 0:
        raise ValueError("--dygformer_heads must be > 0.")
    if args.ncn_num_neighbors <= 0:
        raise ValueError("--ncn_num_neighbors must be > 0.")
    if args.seqfilter_num_neighbors <= 0:
        raise ValueError("--seqfilter_num_neighbors must be > 0.")
    if args.seqfilter_tau < 0.0:
        raise ValueError("--seqfilter_tau must be >= 0.")
    if args.seqfilter_kernel_size <= 0 or args.seqfilter_kernel_size % 2 == 0:
        raise ValueError("--seqfilter_kernel_size must be a positive odd integer.")
    if args.cross_attn_num_layers <= 0:
        raise ValueError("--cross_attn_num_layers must be > 0.")
    if args.cross_attn_heads <= 0:
        raise ValueError("--cross_attn_heads must be > 0.")
    if args.gcn_num_layers <= 0:
        raise ValueError("--gcn_num_layers must be > 0.")
    if args.temporal_relational_rank <= 0:
        raise ValueError("--temporal_relational_rank must be > 0.")
    if args.temporal_relational_time_basis_dim < 2:
        raise ValueError("--temporal_relational_time_basis_dim must be >= 2.")
    if args.use_temporal_relational_gcn:
        if not args.use_learnable_gcn or args.learnable_mp_type != "gcn":
            raise ValueError(
                "--use_temporal_relational_gcn=true requires "
                "--use_learnable_gcn=true and --learnable_mp_type=gcn."
            )
        if not args.rolling_smoothing:
            raise ValueError(
                "--use_temporal_relational_gcn=true requires --rolling_smoothing=true."
            )
    if args.attn_mp_heads <= 0:
        raise ValueError("--attn_mp_heads must be > 0.")
    if args.attn_mp_num_neighbors <= 0:
        raise ValueError("--attn_mp_num_neighbors must be > 0.")
    if args.time_encoder_fourier_dim <= 0:
        raise ValueError("--time_encoder_fourier_dim must be > 0.")
    if args.time_encoder_rbf_dim <= 1:
        raise ValueError("--time_encoder_rbf_dim must be > 1.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be > 0.")
    if args.learnable_entity_embedding_dim <= 0:
        raise ValueError("--learnable_entity_embedding_dim must be > 0.")
    if args.semantic_project_dim < 0:
        raise ValueError("--semantic_project_dim must be >= 0.")
    if (
        not args.use_semantic_smoothing
        and args.use_learnable_gcn
        and args.learnable_mp_type in ("gcn", "gin")
    ):
        raise ValueError(
            "--use_semantic_smoothing=false is incompatible with "
            "--learnable_mp_type=gcn/gin because those modes require the smoothing graph operator."
        )
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.cross_attn_hidden_dropout < 0.0 or args.cross_attn_hidden_dropout >= 1.0:
        raise ValueError("--cross_attn_hidden_dropout must be in [0, 1).")
    if args.cross_attn_attn_dropout < 0.0 or args.cross_attn_attn_dropout >= 1.0:
        raise ValueError("--cross_attn_attn_dropout must be in [0, 1).")
    if args.cross_attn_emb_dropout < 0.0 or args.cross_attn_emb_dropout >= 1.0:
        raise ValueError("--cross_attn_emb_dropout must be in [0, 1).")
    return args


def build_arg_parser():
    parser = argparse.ArgumentParser(description="LLM Link Prediction Evaluation on DTGB datasets")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GDELT",
        help="DTGB dataset folder name under ../DyLink_Datasets or ./DyLink_Datasets",
    )
    parser.add_argument(
        "--entity_name_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "compressed", "compressed_profile"],
        help=(
            "Prompt-side entity display mode. "
            "'auto' uses dataset defaults (currently compress ICEWS1819 only). "
            "'compressed_profile' enables compressed endpoint names plus a one-time compact "
            "endpoint profile block for supported datasets."
        ),
    )
    parser.add_argument(
        "--disable_stack_elec_prompt_cleaning",
        action="store_true",
        help=(
            "Disable the main-pipeline Stack_elec prompt-side entity cleaning/compression. "
            "When enabled, Stack_elec prompt rendering uses the raw entity text from the "
            "loaded entity CSV even if --entity_name_mode=auto/compressed."
        ),
    )
    parser.add_argument(
        "--entity_text_path",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset entity_text CSV. "
            "Useful for entity-name ablations such as scrambled Enron identities."
        ),
    )
    parser.add_argument(
        "--relation_text_path",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset relation_text CSV. "
            "Useful for compacted Enron message-topic relation text."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local model directory or Hugging Face model identifier",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of positive samples to test",
    )
    parser.add_argument(
        "--positive_sample_mode",
        type=str,
        default="random",
        choices=["random", "contiguous", "most_recent"],
        help=(
            "How to subsample positive evaluation edges when --num_samples is smaller "
            "than the split. 'random' preserves previous behavior; 'contiguous' selects "
            "many chronological blocks, keeping each sampled block locally continuous; "
            "'most_recent' selects the latest edges and restores chronological order."
        ),
    )
    parser.add_argument(
        "--positive_sample_block_size",
        type=int,
        default=None,
        help=(
            "Positive-query block size for --positive_sample_mode contiguous. "
            "Defaults to --dtgb_eval_batch_size so each sampled metric batch is a "
            "local chronological chunk."
        ),
    )
    parser.add_argument(
        "--dtgb_eval_batch_size",
        type=int,
        default=256,
        help="Fixed positive-query batch size for DTGB-style AP/AUC averaging",
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=1,
        help="Number of negative samples per positive",
    )
    parser.add_argument(
        "--negative_sampling_mode",
        type=str,
        default="dtgb_sampler",
        choices=["dtgb_sampler", "pool"],
        help=(
            "Negative sampling for LLM sample construction. "
            "'dtgb_sampler' uses the repo DTGB NegativeEdgeSampler seeds "
            "(validation=1, transductive=2, inductive=3); "
            "'pool' preserves the older split destination-pool draw."
        ),
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="both",
        choices=["train", "transductive", "inductive", "validation", "both"],
        help=(
            "Evaluation split: train, transductive, inductive, validation, or both "
            "(run transductive+inductive). The train option is primarily intended "
            "for exporting a sparse, causally constructed prompt-embedding cache."
        ),
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (DTGB default: 0.15)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (DTGB default: 0.15)",
    )
    parser.add_argument(
        "--history_window",
        type=int,
        default=47,
        help="Number of historical events to include in prompt",
    )
    parser.add_argument(
        "--vllm_prompt_variant",
        type=str,
        default="gdelt",
        choices=["gdelt", "temporal_link_prediction"],
        help=(
            "Prompt framing for the local vLLM path. "
            "'temporal_link_prediction' removes GDELT/political-specific wording."
        ),
    )
    parser.add_argument(
        "--mutual_timestamps_only",
        action="store_true",
        help="Render mutual history as timestamps-only time series (token-efficient)",
    )
    parser.add_argument(
        "--mutual_timestamps_dedup",
        action="store_true",
        help="When --mutual_timestamps_only is enabled, collapse consecutive duplicate timestamps",
    )
    parser.add_argument(
        "--mutual_summary_count_recency",
        action="store_true",
        help=(
            "Replace the retained mutual timestamp list with only direct-history "
            "existence, distinct displayed-time count, and latest-event recency. "
            "This implies timestamp-only consecutive-deduplicated processing."
        ),
    )
    parser.add_argument(
        "--common_neighbors_names_only",
        action="store_true",
        help="Render common neighbors as names only (no edge direction/time details)",
    )
    parser.add_argument(
        "--compact_common_neighbors_top_k",
        type=int,
        default=0,
        help=(
            "Compact semantic common neighbors as a pool-size summary plus at most "
            "K ordered neighbors with endpoint-relative direction/time (0 disables)."
        ),
    )
    parser.add_argument(
        "--compact_common_neighbors_novel_only",
        action="store_true",
        help=(
            "Exclude semantic common-neighbor IDs already named in either rendered "
            "compact endpoint history, then backfill from later candidates up to K."
        ),
    )
    parser.add_argument(
        "--history_table_aliases",
        action="store_true",
        help=(
            "Render repeated endpoint mentions inside history event rows as the aliases "
            "'Source' and 'Target' instead of repeating the full entity text."
        ),
    )
    parser.add_argument(
        "--anonymous_entity_aliases",
        action="store_true",
        help=(
            "Replace query endpoints with Source/Target and all other visible "
            "entity names with prompt-local N1, N2, ... aliases."
        ),
    )
    parser.add_argument(
        "--natural_grouped_history",
        action="store_true",
        help=(
            "Group source/target history by named interaction partner and list all "
            "timestamps, preserving direction and duplicate events."
        ),
    )
    parser.add_argument(
        "--natural_activity_summary",
        action="store_true",
        help=(
            "Summarize source/target history with natural interaction-count, "
            "partner-diversity, time-span, frequent/recent-partner, and tail statistics."
        ),
    )
    parser.add_argument(
        "--natural_neighbor_names_only",
        action="store_true",
        help=(
            "Render only the deduplicated union of frequent and recent source/target "
            "neighbor names, omitting counts, timestamps, spans, and tail statistics."
        ),
    )
    parser.add_argument(
        "--natural_neighbor_top_k",
        type=int,
        default=5,
        help=(
            "For --natural_neighbor_names_only, retain the union of the top-K "
            "frequent and top-K recent partners per activity direction (default: 5)."
        ),
    )
    parser.add_argument(
        "--natural_activity_compact_top3",
        action="store_true",
        help=(
            "Deprecated compatibility flag for compact activity rendering. "
            "Prefer --compact_activity_prompt."
        ),
    )
    parser.add_argument(
        "--compact_activity_prompt",
        action="store_true",
        help=(
            "Unified compact prompt recipe: per-direction compact activity, "
            "timestamp-only deduplicated mutual history, no common-neighbor section "
            "unless --common_neighbors_semantic is set, no key-signal/expert sections, "
            "and minimal output guidance."
        ),
    )
    parser.add_argument(
        "--compact_activity_top_k",
        type=int,
        default=3,
        help=(
            "For --compact_activity_prompt, retain the union of the top-K frequent "
            "and top-K recent partners per activity direction (default: 3)."
        ),
    )
    parser.add_argument(
        "--ablate_mutual_history",
        action="store_true",
        help="Remove the MUTUAL HISTORY section from the prompt body",
    )
    parser.add_argument(
        "--ablate_common_neighbors",
        action="store_true",
        help="Remove the COMMON NEIGHBORS section from the prompt body",
    )
    parser.add_argument(
        "--ablate_source_history",
        action="store_true",
        help="Remove the SOURCE HISTORY section from the prompt body",
    )
    parser.add_argument(
        "--ablate_target_history",
        action="store_true",
        help="Remove the TARGET HISTORY section from the prompt body",
    )
    parser.add_argument(
        "--ablate_source_target_history",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--common_neighbors_semantic",
        action="store_true",
        help="Select top semantic common neighbors by target similarity, then present them in destination-recency order",
    )
    parser.add_argument(
        "--ablate_reasoning_guidance",
        action="store_true",
        help="Ablation: remove reasoning guidance text from prompt, keep only output-format constraints",
    )
    parser.add_argument(
        "--semantic_history",
        action="store_true",
        help="Select history events by semantic similarity (default: query-time smoothed E5 cosine)",
    )
    parser.add_argument(
        "--semantic_history_entity_mode",
        action="store_true",
        help="Entity-centric semantic history: rank counterpart entities by recency (semantic tie-break) and show grouped timestamps",
    )
    parser.add_argument(
        "--semantic_history_no_smoothing",
        action="store_true",
        help="A/B mode: disable query-time smoothing and use original raw E5 cosine for semantic history",
    )
    parser.add_argument(
        "--smooth_time_window",
        type=float,
        default=50.0,
        help="Time window for query-time smoothing features used by semantic prompt/context and calibration flows",
    )
    parser.add_argument(
        "--smooth_steps",
        type=int,
        default=1,
        help="Number of smoothing propagation steps for semantic prompt/context and calibration flows",
    )
    parser.add_argument(
        "--smooth_decay_gamma",
        type=float,
        default=0.1,
        help="Decay gamma for query-time smoothing features used by semantic prompt/context and calibration flows",
    )
    parser.add_argument(
        "--smooth_undirected",
        type=str2bool,
        default=True,
        help="Use undirected smoothing graph for semantic prompt/context and calibration flows",
    )
    parser.add_argument(
        "--semantic_hub_penalty_alpha",
        type=float,
        default=0.0,
        help="Subtract alpha * normalized log(dst-popularity-before-t) when selecting source semantic history (0 disables)",
    )
    parser.add_argument(
        "--semantic_fusion_alpha",
        type=float,
        default=1.0,
        help="Fusion weight for similarity rank in semantic-history event mode (0=recency only, 1=similarity only)",
    )
    parser.add_argument(
        "--semantic_fusion_tau",
        type=float,
        default=None,
        help="Recency decay tau for semantic-history event mode; default auto uses per-query median age",
    )
    parser.add_argument(
        "--semantic_fusion_recency_speed",
        type=float,
        default=1.0,
        help="Recency drop speed in semantic-history event mode (>1 steeper, <1 slower)",
    )
    parser.add_argument(
        "--semantic_topk",
        type=int,
        default=None,
        help="Top-K semantic events to keep (defaults to history_window)",
    )
    parser.add_argument(
        "--history_pool_size",
        type=int,
        default=None,
        help="Pool size to consider before semantic selection (defaults to 5*history_window)",
    )
    parser.add_argument(
        "--history_pool_window",
        type=int,
        default=None,
        help="Time window (in ts units) to consider before semantic selection",
    )
    parser.add_argument(
        "--history_preserve_recent_k",
        type=int,
        default=10,
        help="Always preserve the most recent K source/target history events inside the final semantic history window (event mode only)",
    )
    parser.add_argument(
        "--sample_creation_monitor_every",
        type=int,
        default=0,
        help="If >0, print periodic timing breakdown every N positive queries during sample creation",
    )
    parser.add_argument(
        "--sample_creation_profile",
        action="store_true",
        help="Enable cProfile for the Creating samples tqdm loop",
    )
    parser.add_argument(
        "--sample_creation_profile_sort",
        type=str,
        default="cumtime",
        choices=["cumtime", "tottime", "calls", "ncalls", "time"],
        help="Sort key for sample-creation cProfile summary output",
    )
    parser.add_argument(
        "--sample_creation_profile_top_n",
        type=int,
        default=40,
        help="How many rows to print from sample-creation cProfile summary",
    )
    parser.add_argument(
        "--sample_creation_profile_output",
        type=str,
        default=None,
        help=(
            "Optional base output path for raw cProfile stats; each phase writes "
            "to '<base>.<phase>.prof' unless '{phase}' placeholder is provided"
        ),
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="intfloat/e5-large-v2",
        help="Embedding model for semantic selection",
    )
    parser.add_argument(
        "--embedding_cache",
        type=str,
        default=None,
        help="Path to load/save E5 embeddings (.npy)",
    )
    parser.add_argument(
        "--semantic_mlp_embedding_model",
        type=str,
        default=None,
        help=(
            "Optional embedding model used only by the semantic-MLP scorer. "
            "Defaults to --embedding_model."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_embedding_cache",
        type=str,
        default=None,
        help=(
            "Optional embedding cache used only by the semantic-MLP scorer. "
            "This allows prompt semantic selection and the structural checkpoint "
            "to use different embedding dimensions."
        ),
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF k parameter",
    )
    add_heuristic_batch_size_arg(
        parser,
        flag="--rrf_batch_size",
        default=200000,
        help_text="Batch size for RRF heuristic scoring",
    )
    parser.add_argument(
        "--rrf_mode",
        type=str,
        default="train_pool_pointwise",
        choices=["query_local", "sequential_pointwise", "train_pool_pointwise"],
        help="RRF scoring mode: query-local, sequential pointwise, or train-pool pointwise",
    )
    parser.add_argument(
        "--rrf_heuristics",
        type=str,
        default=",".join(DEFAULT_RRF_HEURISTICS),
        help=(
            "Comma-separated RRF heuristic subset. "
            f"Supported: {', '.join(SUPPORTED_RRF_HEURISTICS)}"
        ),
    )
    parser.add_argument(
        "--sequential_rank_bins",
        type=int,
        default=1024,
        help="Pseudo-rank bin count used by --rrf_mode sequential_pointwise",
    )
    parser.add_argument(
        "--rrf_pointwise_pool_size",
        type=int,
        default=256,
        help="Candidate pool size for --rrf_mode train_pool_pointwise",
    )
    parser.add_argument(
        "--rrf_pointwise_num_pools",
        type=int,
        default=4,
        help="Number of sampled train pools per sample for --rrf_mode train_pool_pointwise",
    )
    parser.add_argument(
        "--expert_prediction_mode",
        type=str,
        default="fixed_threshold",
        choices=[
            "global_median",
            "sequential_running_median",
            "fixed_threshold",
            "validation_sampled_threeway",
            "validation_sampled_binary",
        ],
        help=(
            "How to convert the selected structural score source into the prompt PRIOR SIGNAL. "
            "This controls only the single prior cue and does not recalibrate the selected KEY SIGNALS."
        ),
    )
    parser.add_argument(
        "--expert_prediction_source",
        type=str,
        default="rrf",
        choices=["rrf", "semantic_mlp"],
        help=(
            "Structural score source used for PRIOR SIGNAL and overall structural buckets. "
            "'rrf' preserves the original behavior; "
            "'semantic_mlp' uses --semantic_mlp_checkpoint instead."
        ),
    )
    parser.add_argument(
        "--expert_prediction_fixed_threshold",
        type=float,
        default=0.05,
        help=(
            "Fixed structural-score threshold used when --expert_prediction_mode fixed_threshold "
            "(default: 0.05)"
        ),
    )
    parser.add_argument(
        "--validation_calibration_num_samples",
        type=int,
        default=0,
        help=(
            "Positive validation queries used to calibrate "
            "--expert_prediction_mode validation_sampled_threeway or validation_sampled_binary "
            "(<=0 reuses --num_samples)."
        ),
    )
    parser.add_argument(
        "--validation_calibration_negative_ratio",
        type=int,
        default=None,
        help=(
            "Negative ratio for validation-sampled calibration "
            "(default: reuse --negative_ratio)."
        ),
    )
    parser.add_argument(
        "--validation_calibration_low_neg_quantile",
        type=float,
        default=0.75,
        help=(
            "Low-band threshold uses this quantile of validation negative structural scores "
            "for --expert_prediction_mode validation_sampled_threeway."
        ),
    )
    parser.add_argument(
        "--validation_calibration_high_pos_quantile",
        type=float,
        default=0.25,
        help=(
            "High-band threshold uses this quantile of validation positive structural scores "
            "for --expert_prediction_mode validation_sampled_threeway."
        ),
    )
    parser.add_argument(
        "--enable_rrf_validation_band_debug",
        action="store_true",
        help=(
            "Print and save post-hoc validation-derived structural/expert discrimination debug "
            "on the evaluated split. This is a debug switch only; "
            "--validation_calibration_* remain calibration parameters."
        ),
    )
    parser.add_argument(
        "--force_compute_rrf_scores",
        action="store_true",
        help="Force RRF/heuristic scoring even when expert prediction and hybrid mode are disabled",
    )
    parser.add_argument(
        "--hide_key_signals",
        action="store_true",
        help="Hide the selected KEY SIGNALS lines from the LLM prompt",
    )
    parser.add_argument(
        "--key_signal_fields",
        type=str,
        default=",".join(DEFAULT_KEY_SIGNAL_FIELDS),
        help=(
            "Comma-separated prompt key-signal subset. "
            f"Supported: {', '.join(SUPPORTED_KEY_SIGNAL_FIELDS)}"
        ),
    )
    parser.add_argument(
        "--heuristic_recent_degree_window",
        type=float,
        default=30.0,
        help="Hard lookback window for recent_degree heuristic values used in LLM-side prompt signals/RRF.",
    )
    parser.add_argument(
        "--hide_expert_prediction",
        action="store_true",
        help="Hide the PRIOR SIGNAL line from the LLM prompt",
    )
    parser.add_argument(
        "--include_overall_structural_signal",
        action="store_true",
        help=(
            "Also compute a bucketed overall structural prior (Low/Modest/High). "
            "This affects the prior-signal control flow only and does not recalibrate the selected KEY SIGNALS."
        ),
    )
    parser.add_argument(
        "--overall_signal_name",
        type=str,
        default="Overall structural signal",
        help="Reserved label for backward compatibility (no longer rendered in KEY SIGNALS)",
    )
    parser.add_argument(
        "--overall_signal_low_threshold",
        type=float,
        default=0.0475,
        help="Low/Modest boundary for overall structural signal (default: 0.0475)",
    )
    parser.add_argument(
        "--overall_signal_high_threshold",
        type=float,
        default=0.0510,
        help="Modest/High boundary for overall structural signal (default: 0.0510)",
    )
    parser.add_argument(
        "--include_edge_type",
        action="store_true",
        help="Include relation/edge-type text in prompts (default: off)",
    )
    parser.add_argument(
        "--include_edge_type_except_target",
        action="store_true",
        help=(
            "vLLM-only prompt mode: include relation/edge-type text in observed context, "
            "but hide the queried target relation text."
        ),
    )
    parser.add_argument(
        "--summary_entity_text_path",
        type=str,
        default=None,
        help="Optional summary CSV (columns i,text) used for endpoint profile context in LLM prompts.",
    )
    parser.add_argument(
        "--summary_mode",
        type=str,
        default="off",
        choices=["off", "entity_meaning", "full"],
        help="Prompt-side summary mode: off, entity_meaning-only, or full summary text.",
    )
    parser.add_argument(
        "--summary_max_chars",
        type=int,
        default=120,
        help="Max characters per endpoint summary snippet injected into prompts.",
    )
    parser.add_argument(
        "--few_shot_path",
        type=str,
        default=None,
        help=(
            "Optional path to a text/markdown file containing summary-only few-shot examples. "
            "Blocks are split on 'ID N' or 'Example N' headers."
        ),
    )
    parser.add_argument(
        "--few_shot_max_examples",
        type=int,
        default=None,
        help="Optional cap on the number of few-shot examples loaded from --few_shot_path.",
    )
    parser.add_argument(
        "--key_signal_mode",
        type=str,
        default="bucket",
        choices=["bucket", "raw", "percentile"],
        help="Key-signal rendering mode (default: bucket)",
    )
    parser.add_argument(
        "--key_signal_reference",
        type=str,
        default="contextual",
        choices=["sequential_global", "contextual", "validation_sampled_global"],
        help=(
            "Reference population for calibrating the selected KEY SIGNALS only "
            "(default prompt set: target popularity, past interactions, recency, common neighbors). "
            "'validation_sampled_global' freezes that calibration from a sampled validation subset."
        ),
    )
    parser.add_argument(
        "--hybrid_uncertain_topk_queries",
        type=int,
        default=0,
        help=(
            "If >0, run budgeted hybrid: LLM on selected middle-band samples, metrics on full set. "
            "Interpretation depends on --hybrid_selection_mode."
        ),
    )
    parser.add_argument(
        "--hybrid_backbone",
        type=str,
        default="rrf",
        choices=["rrf", "semantic_mlp"],
        help=(
            "Structural backbone used for hybrid routing/fallback/merge-base. "
            "'rrf' preserves the original behavior; "
            "'semantic_mlp' uses a trained checkpoint from train_semantic_mlp_pipeline.py."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path for semantic structural scoring: "
            "--hybrid_backbone semantic_mlp or --expert_prediction_source semantic_mlp."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_score_python",
        type=str,
        default=None,
        help=(
            "Optional Python executable for out-of-process semantic MLP scoring. "
            "When omitted, semantic scoring runs in the current process."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_source_init_override",
        type=str,
        default="auto",
        choices=["auto", "raw", "history_mean"],
        help=(
            "Override semantic source/user embedding initialization when loading a "
            "semantic MLP checkpoint for evaluation. 'auto' uses checkpoint metadata "
            "when available; older checkpoints without this field fall back to 'raw'."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_temporal_mode",
        type=str,
        default="timestamp_rebuild",
        choices=["timestamp_rebuild", "rolling_replay"],
        help=(
            "Temporal replay mode for semantic MLP scoring inside the LLM evaluator. "
            "'timestamp_rebuild' rebuilds the allowed history from full graph timestamps "
            "for each semantic scoring batch and is order-invariant for sampled rows. "
            "'rolling_replay' is faster but assumes samples are processed in chronological order."
        ),
    )
    parser.add_argument(
        "--semantic_backbone_ablate_recency",
        action="store_true",
        help=(
            "Runtime debug ablation for semantic structural scoring: "
            "replace heuristic recency values with a constant sentinel instead of "
            "running the recency kernel."
        ),
    )
    parser.add_argument(
        "--hybrid_selection_mode",
        type=str,
        default="per_dtgb_batch_middle_band",
        choices=[
            "per_dtgb_batch_middle_band",
            "pointwise_fixed_threshold_band",
            "random_sample",
            "learned_router_top_fraction",
            "tabicl_router",
            "validation_fitted_tabicl_router",
            "validation_sampled_uncertainty_band",
            "validation_sampled_gmm_overlap_band",
        ],
        help=(
            "Hybrid selected-slice construction mode: "
            "per_dtgb_batch_middle_band uses DTGB test-batch grouping; "
            "pointwise_fixed_threshold_band routes by fixed score thresholds only; "
            "random_sample routes a uniformly random sample slice at --hybrid_validation_target_fraction; "
            "learned_router_top_fraction globally routes the highest predicted-utility slice from --hybrid_router_checkpoint; "
            "tabicl_router fits grouped-OOF utility routing from a most-recent labeled train slice; "
            "validation_fitted_tabicl_router is a deprecated alias for tabicl_router; "
            "validation_sampled_uncertainty_band learns a frozen pointwise middle band from sampled validation backbone scores; "
            "validation_sampled_gmm_overlap_band fits a 2-component GMM on sampled validation backbone scores and routes the frozen overlap region."
        ),
    )
    parser.add_argument(
        "--hybrid_router_checkpoint",
        type=str,
        default=None,
        help=(
            "Fitted joblib router used by --hybrid_selection_mode "
            "learned_router_top_fraction."
        ),
    )
    parser.add_argument(
        "--tabicl_router_python",
        type=str,
        default=None,
        help=(
            "Python executable containing TabICL for automatic train-context "
            "routing and final fusion."
        ),
    )
    parser.add_argument(
        "--tabicl_router_device",
        type=str,
        default="cuda:0",
        help="Torch device inside the TabICL subprocess (default: cuda:0).",
    )
    parser.add_argument(
        "--tabicl_router_cuda_visible_devices",
        type=str,
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES override for the TabICL subprocess, "
            "allowing it to use a GPU different from vLLM."
        ),
    )
    parser.add_argument(
        "--tabicl_router_artifact_dir",
        type=str,
        default=None,
        help=(
            "Directory for automatic router tables, logs, predictions, and "
            "exact-split route checkpoints (default: derived from --output)."
        ),
    )
    parser.add_argument("--tabicl_router_folds", type=int, default=5)
    parser.add_argument("--tabicl_router_n_estimators", type=int, default=4)
    parser.add_argument("--tabicl_router_batch_size", type=int, default=8)
    parser.add_argument(
        "--tabicl_alignment_variant",
        type=str,
        default="score_fusion_no_heuristics",
        choices=["score_fusion_no_heuristics", "score_engineered"],
        help=(
            "Feature set for automatic TabICL final fusion. The default uses "
            "score/routing features only; 'score_engineered' additionally uses "
            "timestamp, node popularity, pair history, recency, and common-neighbor "
            "features."
        ),
    )
    parser.add_argument(
        "--tabicl_context_sampling",
        type=str,
        default="most_recent",
        choices=["most_recent", "uniform_recent_pool", "uniform_all_train"],
        help=(
            "Sampling policy for labeled train-context queries. "
            "'uniform_recent_pool' first takes the most-recent pool configured "
            "by --tabicl_context_pool_positive_queries, then uniformly samples "
            "the requested alignment queries from that pool; "
            "'uniform_all_train' samples them uniformly from every eligible "
            "observed train positive."
        ),
    )
    parser.add_argument(
        "--tabicl_context_pool_positive_queries",
        type=int,
        default=0,
        help=(
            "Positive-query pool size for --tabicl_context_sampling "
            "uniform_recent_pool. Must be at least the requested alignment "
            "positive-query count."
        ),
    )
    parser.add_argument(
        "--hybrid_pointwise_low_threshold",
        type=float,
        default=0.0475,
        help=(
            "Low threshold for --hybrid_selection_mode pointwise_fixed_threshold_band. "
            "Route to LLM when low <= rrf_score < high."
        ),
    )
    parser.add_argument(
        "--hybrid_pointwise_high_threshold",
        type=float,
        default=0.0510,
        help=(
            "High threshold for --hybrid_selection_mode pointwise_fixed_threshold_band. "
            "Route to LLM when low <= rrf_score < high."
        ),
    )
    parser.add_argument(
        "--hybrid_validation_target_fraction",
        type=float,
        default=0.0,
        help=(
            "Approximate sample fraction to route under "
            "--hybrid_selection_mode validation_sampled_uncertainty_band. "
            "Uses a frozen validation-derived middle band around the validation decision boundary."
        ),
    )
    parser.add_argument(
        "--hybrid_debug_max_routed_samples",
        type=int,
        default=0,
        help=(
            "Debug-only cap on the number of routed samples after normal hybrid selection. "
            "(<=0 disables). When enabled, keeps the router's highest-priority routed samples only."
        ),
    )
    parser.add_argument(
        "--hybrid_merge_method",
        type=str,
        default="rank_graft",
        choices=["rank_graft", "alpha", "raw_alpha", "llm_only"],
        help="Hybrid merge on selected middle-band samples (default: rank_graft)",
    )
    parser.add_argument(
        "--hybrid_score_alignment_mode",
        type=str,
        default="off",
        choices=[
            "off",
            "tabicl",
            "validation_selected_quantile_match",
            "validation_selected_isotonic_regression",
        ],
        help=(
            "Optional score-alignment stage before hybrid merge. "
            "'tabicl' fits a full-score TabICL corrector on the same most-recent "
            "train context used by tabicl_router; "
            "'validation_selected_quantile_match' runs LLM on a sampled validation calibration slice, "
            "then maps raw LLM scores onto the selected-slice backbone score distribution by quantile matching. "
            "'validation_selected_isotonic_regression' fits a monotone isotonic remap from raw LLM scores "
            "to paired backbone scores on that same calibration slice."
        ),
    )
    parser.add_argument(
        "--hybrid_score_alignment_fit_scope",
        type=str,
        default="validation_all",
        choices=["validation_all", "validation_selected"],
        help=(
            "Validation slice used to fit --hybrid_score_alignment_mode. "
            "'validation_all' fits on the full sampled validation calibration slice; "
            "'validation_selected' first applies the active hybrid routing rule to that slice, "
            "then fits alignment only on the routed validation subset."
        ),
    )
    parser.add_argument(
        "--hybrid_alignment_num_samples",
        type=int,
        default=0,
        help=(
            "Positive calibration queries used to fit --hybrid_score_alignment_mode "
            "(most-recent train queries for 'tabicl', validation queries otherwise) "
            "(<=0 reuses --validation_calibration_num_samples, then --num_samples)."
        ),
    )
    parser.add_argument(
        "--hybrid_alignment_negative_ratio",
        type=int,
        default=None,
        help=(
            "Negative ratio for hybrid score-alignment calibration "
            "(default: reuse --validation_calibration_negative_ratio, then --negative_ratio)."
        ),
    )
    parser.add_argument(
        "--hybrid_fusion_alpha",
        type=float,
        default=1.0,
        help="Fusion alpha when --hybrid_merge_method is alpha/raw_alpha",
    )
    parser.add_argument(
        "--hybrid_backbone_score_space",
        type=str,
        default="minmax",
        choices=["minmax", "raw"],
        help=(
            "Score space used for hybrid fallback/merge base and alignment target. "
            "'minmax' preserves current behavior; 'raw' uses the backbone's raw score scale."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=len(DEFAULT_SEEDS),
        help="Evaluation trials with seeds seed+i (default: five); does not retrain the model",
    )
    parser.add_argument(
        "--no_cot",
        action="store_true",
        help="Disable Chain-of-Thought reasoning (Direct Answer mode)",
    )
    parser.add_argument(
        "--no_cot_output_0_100",
        action="store_true",
        help="In no-CoT mode, request and parse score X in [0,100] instead of binary 0/1",
    )
    parser.add_argument(
        "--no_cot_binary_score_mode",
        type=str,
        default="forced_binary",
        choices=["sample_logprob", "forced_binary"],
        help="How to score no-CoT binary 0/1 outputs in vLLM mode (default: sample_logprob)",
    )
    parser.add_argument(
        "--cot_max_tokens",
        type=int,
        default=2048,
        help="Max generated tokens in CoT mode (warns if clipped)",
    )
    parser.add_argument(
        "--enable_history_compaction_llm_only",
        action="store_true",
        help="Enable local history compaction before prediction (default vLLM backend only)",
    )
    parser.add_argument(
        "--history_compaction_max_tokens",
        type=int,
        default=256,
        help="Max generation tokens for local history compaction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_eval_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--debug_prediction_log",
        type=str,
        default=None,
        help="Optional JSONL path for per-sample debug trace (LLM/RRF/heuristic predictions)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., awq, gptq, bitsandbytes)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        "-tp",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=10000,
        help="Maximum context length (to prevent OOM)",
    )
    parser.add_argument(
        "--gpu_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--enforce_eager_vllm",
        action="store_true",
        help="Disable vLLM torch.compile and cudagraphs. Useful for torch/vLLM compatibility issues.",
    )
    parser.add_argument(
        "--capture_prompt_embeddings",
        action="store_true",
        help=(
            "Capture one contextual deep-layer, last-prompt-token vector per evaluated "
            "link-prediction prompt. Uses vLLM extract_hidden_states and saves NPZ shards."
        ),
    )
    parser.add_argument(
        "--export_prompt_dataset_dir",
        type=str,
        default=None,
        help=(
            "Export the canonical forced-binary prompt text and aligned link identities "
            "to prompts_<split>.jsonl without running model inference."
        ),
    )
    parser.add_argument(
        "--prompt_embedding_output_dir",
        type=str,
        default=None,
        help=(
            "Directory for captured prompt-embedding NPZ shards. Defaults to "
            "<output-without-.json>_prompt_embeddings."
        ),
    )
    parser.add_argument(
        "--prompt_embedding_storage_dir",
        type=str,
        default=None,
        help=(
            "Optional vLLM connector scratch directory. Defaults to a temporary /dev/shm "
            "directory when available; scratch tensors are deleted after each batch."
        ),
    )
    parser.add_argument(
        "--prompt_embedding_layer",
        type=int,
        default=-1,
        help=(
            "Hidden-state location to capture. -1 selects the raw post-final-RMSNorm "
            "LM-head input (recommended); explicit decoder indices select vLLM's "
            "pre-block auxiliary states."
        ),
    )
    parser.add_argument(
        "--prompt_embedding_save_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for prompt-embedding NPZ shards.",
    )
    prompt_embedding_norm_group = parser.add_mutually_exclusive_group()
    prompt_embedding_norm_group.add_argument(
        "--prompt_embedding_normalize",
        dest="prompt_embedding_normalize",
        action="store_true",
        help=(
            "L2-normalize captured prompt vectors. Off by default so the saved final "
            "state remains the exact LM-head input."
        ),
    )
    prompt_embedding_norm_group.add_argument(
        "--no_prompt_embedding_normalize",
        dest="prompt_embedding_normalize",
        action="store_false",
        help="Deprecated compatibility alias; raw prompt states are now the default.",
    )
    parser.set_defaults(prompt_embedding_normalize=False)
    parser.add_argument(
        "--use_transformers",
        action="store_true",
        help="Use standard transformers instead of vLLM (debug)",
    )
    parser.add_argument(
        "--use_openai_api",
        action="store_true",
        help="Use OpenAI Chat Completions API backend (separate module path)",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model name when --use_openai_api is enabled",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL (e.g. proxy/local gateway)",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional if env var is set)",
    )
    parser.add_argument(
        "--openai_api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name to read API key from",
    )
    parser.add_argument(
        "--openai_reasoning_effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help="reasoning_effort sent to OpenAI Chat Completions",
    )
    parser.add_argument(
        "--openai_timeout_sec",
        type=float,
        default=120.0,
        help="HTTP timeout (seconds) for OpenAI calls",
    )
    parser.add_argument(
        "--openai_max_retries",
        type=int,
        default=3,
        help="Max client-level retries for OpenAI calls",
    )
    parser.add_argument(
        "--openai_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for OpenAI backend",
    )
    parser.add_argument(
        "--openai_top_p",
        type=float,
        default=1.0,
        help="Top-p for OpenAI backend",
    )
    parser.add_argument(
        "--openai_concurrency",
        type=int,
        default=1,
        help="Max in-flight OpenAI API requests (async semaphore cap)",
    )
    parser.add_argument(
        "--openai_max_requests_per_sec",
        type=float,
        default=0.0,
        help="Optional client-side request rate cap for OpenAI backend (0 disables)",
    )
    parser.add_argument(
        "--openai_no_cot_max_tokens",
        type=int,
        default=32,
        help="max_tokens for no-CoT OpenAI calls",
    )
    parser.add_argument(
        "--rrf_only",
        action="store_true",
        help="Skip model loading and evaluate standalone RRF baseline only",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Number of DP workers (auto-spawn local workers if rank env is absent)",
    )
    parser.add_argument(
        "--data_parallel_sync_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for DP sync artifacts. Default is "
            "'<output_dir>/.dp_sync/<output_basename>'."
        ),
    )
    parser.add_argument(
        "--keep_data_parallel_sync",
        action="store_true",
        help="Keep DP sync shard files after successful run (default: auto-clean current run dir).",
    )
    parser.add_argument(
        "--legacy_interleaved_split_processing",
        action="store_true",
        help="Restore the original split loop: build/evaluate each split before moving to the next one",
    )
    parser.set_defaults(history_direction="both", history_protocol="both_endpoints_recent_v1",
                        interaction_count_direction="source_to_target")
    return parser


def validate_args(args):
    apply_compact_activity_prompt_recipe(args)
    if int(args.compact_common_neighbors_top_k) < 0:
        raise ValueError("--compact_common_neighbors_top_k must be >= 0")
    if args.compact_common_neighbors_top_k and not args.common_neighbors_semantic:
        raise ValueError(
            "--compact_common_neighbors_top_k requires --common_neighbors_semantic"
        )
    if args.compact_common_neighbors_top_k and args.common_neighbors_names_only:
        raise ValueError(
            "--compact_common_neighbors_top_k cannot be combined with "
            "--common_neighbors_names_only"
        )
    if args.compact_common_neighbors_novel_only:
        if not args.compact_common_neighbors_top_k:
            raise ValueError(
                "--compact_common_neighbors_novel_only requires "
                "--compact_common_neighbors_top_k"
            )
        if not args.natural_activity_compact_top3:
            raise ValueError(
                "--compact_common_neighbors_novel_only requires compact activity histories"
            )
        if args.anonymous_entity_aliases:
            raise ValueError(
                "--compact_common_neighbors_novel_only cannot be combined with "
                "--anonymous_entity_aliases"
            )
    if getattr(args, "ablate_source_target_history", False):
        args.ablate_source_history = True
        args.ablate_target_history = True
    args.rrf_heuristics = normalize_rrf_heuristics(getattr(args, "rrf_heuristics", None))
    args.key_signal_fields = normalize_key_signal_fields(getattr(args, "key_signal_fields", None))
    hybrid_enabled = bool(
        args.hybrid_uncertain_topk_queries > 0
        or str(args.hybrid_selection_mode).strip().lower()
        in {
            "pointwise_fixed_threshold_band",
            "random_sample",
            "learned_router_top_fraction",
            "tabicl_router",
            "validation_fitted_tabicl_router",
            "validation_sampled_uncertainty_band",
            "validation_sampled_gmm_overlap_band",
        }
    )
    if args.dtgb_eval_batch_size < 1:
        raise ValueError("--dtgb_eval_batch_size must be >= 1")
    if getattr(args, "positive_sample_block_size", None) is None:
        args.positive_sample_block_size = int(args.dtgb_eval_batch_size)
    elif args.positive_sample_block_size < 1:
        raise ValueError("--positive_sample_block_size must be >= 1 when provided")
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")
    if args.negative_ratio < 1:
        raise ValueError("--negative_ratio must be >= 1")
    if getattr(args, "heuristic_recent_degree_window", 30.0) <= 0.0:
        raise ValueError("--heuristic_recent_degree_window must be > 0.")
    if args.rrf_only and hybrid_enabled:
        raise ValueError("--rrf_only does not support hybrid LLM evaluation mode.")
    if args.expert_prediction_source == "semantic_mlp" and not args.semantic_mlp_checkpoint:
        raise ValueError(
            "--semantic_mlp_checkpoint is required when "
            "--expert_prediction_source semantic_mlp."
        )
    if args.hybrid_backbone == "semantic_mlp":
        if not args.semantic_mlp_checkpoint:
            raise ValueError(
                "--semantic_mlp_checkpoint is required when --hybrid_backbone semantic_mlp."
            )
        if hybrid_enabled and (
            str(args.hybrid_selection_mode).strip().lower()
            not in {
                "pointwise_fixed_threshold_band",
                "random_sample",
                "learned_router_top_fraction",
                "tabicl_router",
                "validation_fitted_tabicl_router",
                "validation_sampled_uncertainty_band",
                "validation_sampled_gmm_overlap_band",
            }
        ):
            raise ValueError(
                "--hybrid_backbone semantic_mlp currently supports only "
                "--hybrid_selection_mode pointwise_fixed_threshold_band, "
                "--hybrid_selection_mode random_sample, "
                "--hybrid_selection_mode learned_router_top_fraction, "
                "--hybrid_selection_mode tabicl_router, "
                "--hybrid_selection_mode validation_fitted_tabicl_router, "
                "--hybrid_selection_mode validation_sampled_uncertainty_band "
                "or validation_sampled_gmm_overlap_band."
            )
    if (
        args.validation_calibration_negative_ratio is not None
        and args.validation_calibration_negative_ratio < 1
    ):
        raise ValueError("--validation_calibration_negative_ratio must be >= 1 when provided")
    if (
        getattr(args, "hybrid_alignment_negative_ratio", None) is not None
        and args.hybrid_alignment_negative_ratio < 1
    ):
        raise ValueError("--hybrid_alignment_negative_ratio must be >= 1 when provided")
    if not (0.0 <= args.validation_calibration_low_neg_quantile <= 1.0):
        raise ValueError("--validation_calibration_low_neg_quantile must be in [0, 1]")
    if not (0.0 <= args.validation_calibration_high_pos_quantile <= 1.0):
        raise ValueError("--validation_calibration_high_pos_quantile must be in [0, 1]")
    if args.semantic_hub_penalty_alpha < 0.0:
        raise ValueError("--semantic_hub_penalty_alpha must be >= 0.0")
    if not (0.0 <= float(args.hybrid_validation_target_fraction) <= 1.0):
        raise ValueError("--hybrid_validation_target_fraction must be in [0, 1]")
    if int(getattr(args, "hybrid_debug_max_routed_samples", 0)) < 0:
        raise ValueError("--hybrid_debug_max_routed_samples must be >= 0")
    if (
        args.hybrid_selection_mode in {
            "random_sample",
            "learned_router_top_fraction",
            "tabicl_router",
            "validation_fitted_tabicl_router",
            "validation_sampled_uncertainty_band",
            "validation_sampled_gmm_overlap_band",
        }
        and float(args.hybrid_validation_target_fraction) <= 0.0
    ):
        raise ValueError(
            "--hybrid_validation_target_fraction must be > 0 for "
            "--hybrid_selection_mode random_sample, "
            "--hybrid_selection_mode learned_router_top_fraction, "
            "--hybrid_selection_mode tabicl_router, "
            "--hybrid_selection_mode validation_fitted_tabicl_router, "
            "--hybrid_selection_mode validation_sampled_uncertainty_band "
            "or validation_sampled_gmm_overlap_band"
        )
    if not (0.0 <= args.semantic_fusion_alpha <= 1.0):
        raise ValueError("--semantic_fusion_alpha must be in [0, 1]")
    if args.semantic_fusion_tau is not None and args.semantic_fusion_tau <= 0.0:
        raise ValueError("--semantic_fusion_tau must be > 0 or omitted for auto")
    if args.semantic_fusion_recency_speed <= 0.0:
        raise ValueError("--semantic_fusion_recency_speed must be > 0")
    if args.history_preserve_recent_k < 0:
        raise ValueError("--history_preserve_recent_k must be >= 0")
    if args.sample_creation_monitor_every < 0:
        raise ValueError("--sample_creation_monitor_every must be >= 0")
    if args.sample_creation_profile_top_n < 1:
        raise ValueError("--sample_creation_profile_top_n must be >= 1")
    if args.rrf_mode == "sequential_pointwise" and args.sequential_rank_bins < 2:
        raise ValueError("--sequential_rank_bins must be >= 2 for --rrf_mode sequential_pointwise")
    if args.rrf_mode == "train_pool_pointwise":
        if args.rrf_pointwise_pool_size < 2:
            raise ValueError("--rrf_pointwise_pool_size must be >= 2 for --rrf_mode train_pool_pointwise")
        if args.rrf_pointwise_num_pools < 1:
            raise ValueError("--rrf_pointwise_num_pools must be >= 1 for --rrf_mode train_pool_pointwise")
    if args.use_openai_api and args.use_transformers:
        raise ValueError("Use either --use_openai_api or --use_transformers, not both.")
    if args.capture_prompt_embeddings:
        if args.rrf_only or args.use_transformers or args.use_openai_api:
            raise ValueError(
                "--capture_prompt_embeddings is supported only on the default local vLLM backend."
            )
        if args.prompt_embedding_layer < -1:
            raise ValueError("--prompt_embedding_layer must be -1 or a non-negative index.")
        if args.prompt_embedding_output_dir is None:
            output_stem = os.path.splitext(os.path.abspath(args.output))[0]
            args.prompt_embedding_output_dir = output_stem + "_prompt_embeddings"
        if os.path.isfile(args.prompt_embedding_output_dir):
            raise ValueError(
                "--prompt_embedding_output_dir points to an existing file "
                "(expected a directory path)."
            )
    if args.export_prompt_dataset_dir:
        if args.capture_prompt_embeddings:
            raise ValueError(
                "Use either --export_prompt_dataset_dir or --capture_prompt_embeddings."
            )
        if args.data_parallel_size != 1:
            raise ValueError("Prompt dataset export currently requires --data_parallel_size 1.")
        if os.path.isfile(args.export_prompt_dataset_dir):
            raise ValueError(
                "--export_prompt_dataset_dir points to an existing file "
                "(expected a directory path)."
            )
        if (
            args.prompt_embedding_storage_dir
            and os.path.isfile(args.prompt_embedding_storage_dir)
        ):
            raise ValueError(
                "--prompt_embedding_storage_dir points to an existing file "
                "(expected a directory path)."
            )
    if args.enable_history_compaction_llm_only and (
        args.rrf_only or args.use_transformers or args.use_openai_api
    ):
        raise ValueError(
            "--enable_history_compaction_llm_only is supported only on the default local vLLM backend."
        )
    if (
        args.hybrid_selection_mode == "learned_router_top_fraction"
        and not args.hybrid_router_checkpoint
    ):
        raise ValueError(
            "--hybrid_router_checkpoint is required for "
            "--hybrid_selection_mode learned_router_top_fraction"
        )
    tabicl_router_enabled = args.hybrid_selection_mode in {
        "tabicl_router",
        "validation_fitted_tabicl_router",
    }
    tabicl_alignment_enabled = args.hybrid_score_alignment_mode == "tabicl"
    if tabicl_router_enabled and not tabicl_alignment_enabled:
        raise ValueError(
            "tabicl_router requires --hybrid_score_alignment_mode tabicl "
            "so routing and final fusion share the train context"
        )
    if tabicl_alignment_enabled:
        supported_selection_modes = {
            "tabicl_router",
            "validation_fitted_tabicl_router",
            "validation_sampled_uncertainty_band",
        }
        if args.hybrid_selection_mode not in supported_selection_modes:
            raise ValueError(
                "--hybrid_score_alignment_mode tabicl requires tabicl_router "
                "or validation_sampled_uncertainty_band selection"
            )
        if not args.tabicl_router_python:
            raise ValueError(
                "--tabicl_router_python is required for TabICL alignment"
            )
        if args.data_parallel_size != 1:
            raise ValueError(
                "TabICL routing/alignment currently requires --data_parallel_size 1"
            )
        effective_negative_ratio = args.hybrid_alignment_negative_ratio
        if effective_negative_ratio is None:
            effective_negative_ratio = args.validation_calibration_negative_ratio
        if effective_negative_ratio is None:
            effective_negative_ratio = args.negative_ratio
        if int(effective_negative_ratio) != 1:
            raise ValueError(
                "TabICL routing/alignment currently requires a 1:1 train-context slice"
            )
        if tabicl_router_enabled and args.tabicl_router_folds < 2:
            raise ValueError("--tabicl_router_folds must be at least 2")
        if args.tabicl_router_n_estimators < 1:
            raise ValueError("--tabicl_router_n_estimators must be positive")
        if args.tabicl_router_batch_size < 1:
            raise ValueError("--tabicl_router_batch_size must be positive")
    if args.include_edge_type_except_target:
        if args.use_openai_api or args.use_transformers or args.rrf_only:
            raise ValueError(
                "--include_edge_type_except_target is supported only on the default local vLLM backend."
            )
        if args.enable_history_compaction_llm_only:
            raise ValueError(
                "--include_edge_type_except_target is not supported with --enable_history_compaction_llm_only."
            )
        if args.include_edge_type:
            raise ValueError(
                "Use either --include_edge_type or --include_edge_type_except_target, not both."
            )
    if args.history_compaction_max_tokens < 1:
        raise ValueError("--history_compaction_max_tokens must be >= 1")
    if args.openai_timeout_sec <= 0:
        raise ValueError("--openai_timeout_sec must be > 0")
    if args.openai_max_retries < 0:
        raise ValueError("--openai_max_retries must be >= 0")
    if not (0.0 <= args.openai_top_p <= 1.0):
        raise ValueError("--openai_top_p must be in [0, 1]")
    if args.openai_temperature < 0.0:
        raise ValueError("--openai_temperature must be >= 0")
    if args.openai_concurrency < 1:
        raise ValueError("--openai_concurrency must be >= 1")
    if args.openai_max_requests_per_sec < 0.0:
        raise ValueError("--openai_max_requests_per_sec must be >= 0")
    if args.openai_no_cot_max_tokens < 1:
        raise ValueError("--openai_no_cot_max_tokens must be >= 1")
    if not (
        math.isfinite(args.overall_signal_low_threshold)
        and math.isfinite(args.overall_signal_high_threshold)
    ):
        raise ValueError("--overall_signal_* thresholds must be finite floats")
    if args.overall_signal_low_threshold >= args.overall_signal_high_threshold:
        raise ValueError(
            "--overall_signal_low_threshold must be < --overall_signal_high_threshold"
        )
    if not (
        math.isfinite(args.hybrid_pointwise_low_threshold)
        and math.isfinite(args.hybrid_pointwise_high_threshold)
    ):
        raise ValueError("--hybrid_pointwise_*_threshold must be finite floats")
    if args.hybrid_pointwise_low_threshold >= args.hybrid_pointwise_high_threshold:
        raise ValueError("--hybrid_pointwise_low_threshold must be < --hybrid_pointwise_high_threshold")
    if (
        args.summary_mode != "off"
        and not args.summary_entity_text_path
        and str(args.dataset_name).strip().upper() != "ICEWS1819"
    ):
        raise ValueError(
            "--summary_entity_text_path is required when --summary_mode is not off "
            "(except ICEWS1819, which can auto-generate compact endpoint profiles)"
        )
    if args.summary_max_chars <= 0:
        raise ValueError("--summary_max_chars must be > 0")
    if args.few_shot_max_examples is not None and args.few_shot_max_examples < 1:
        raise ValueError("--few_shot_max_examples must be >= 1 when provided")
    if args.few_shot_path and not os.path.isfile(args.few_shot_path):
        raise ValueError(f"--few_shot_path not found: {args.few_shot_path}")
    if args.entity_text_path and not os.path.isfile(args.entity_text_path):
        raise ValueError(f"--entity_text_path not found: {args.entity_text_path}")
    if args.relation_text_path and not os.path.isfile(args.relation_text_path):
        raise ValueError(f"--relation_text_path not found: {args.relation_text_path}")
    if args.semantic_mlp_checkpoint and not os.path.isfile(args.semantic_mlp_checkpoint):
        raise ValueError(f"--semantic_mlp_checkpoint not found: {args.semantic_mlp_checkpoint}")
    if (
        args.semantic_mlp_score_python
        and not os.path.isfile(args.semantic_mlp_score_python)
        and shutil.which(args.semantic_mlp_score_python) is None
    ):
        raise ValueError(f"--semantic_mlp_score_python not found: {args.semantic_mlp_score_python}")
    if args.data_parallel_sync_dir and os.path.isfile(args.data_parallel_sync_dir):
        raise ValueError("--data_parallel_sync_dir points to an existing file (expected a directory path).")
