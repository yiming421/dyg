"""Simple, deterministic sampling helpers for direct PEFT training queries."""

from __future__ import annotations

from typing import Optional

import pandas as pd


PROMPT_MULTIPLICITY_COLUMN = "prompt_multiplicity"
POSITIVE_SAMPLING_GROUP_COLUMN = "positive_sampling_group"
SOURCE_FIRST_TOUCH_COLUMN = "source_first_touch"
TARGET_FIRST_TOUCH_COLUMN = "target_first_touch"


def prompt_identity_columns(*, include_relation: bool) -> list[str]:
    """Return the edge columns that determine the rendered query identity."""
    if include_relation:
        return ["u", "r", "i", "ts"]
    return ["u", "i", "ts"]


def _validate_edge_columns(edges_df: pd.DataFrame) -> None:
    required = {"u", "r", "i", "ts"}
    missing = sorted(required - set(edges_df.columns))
    if missing:
        raise ValueError(f"Edge table is missing required columns: {missing}")


def _first_seen_time_by_node(edges_df: pd.DataFrame) -> pd.Series:
    """Find each node's first timestamp in either endpoint role."""
    source_rows = edges_df[["u", "ts"]].rename(columns={"u": "node_id"})
    target_rows = edges_df[["i", "ts"]].rename(columns={"i": "node_id"})
    endpoint_rows = pd.concat([source_rows, target_rows], axis=0, ignore_index=True)
    return endpoint_rows.groupby("node_id", sort=False)["ts"].min()


def _prepare_prompt_pool(
    split_edges: pd.DataFrame,
    *,
    collapse_prompt_duplicates: bool,
    prompt_identity_includes_relation: bool,
) -> tuple[pd.DataFrame, list[str], int]:
    identity_columns = prompt_identity_columns(
        include_relation=prompt_identity_includes_relation
    )
    pool = split_edges.copy()
    if collapse_prompt_duplicates:
        pool[PROMPT_MULTIPLICITY_COLUMN] = pool.groupby(
            identity_columns,
            sort=False,
            dropna=False,
        )[identity_columns[0]].transform("size")
        before = len(pool)
        pool = pool.drop_duplicates(identity_columns, keep="first").copy()
        collapsed_count = before - len(pool)
    else:
        pool[PROMPT_MULTIPLICITY_COLUMN] = 1
        collapsed_count = 0
    return pool, identity_columns, int(collapsed_count)


def select_training_positive_edges(
    split_edges: pd.DataFrame,
    *,
    num_samples: int,
    sampling_strategy: str,
    random_seed: int,
    sampling_skip_recent: int = 0,
    collapse_prompt_duplicates: bool = False,
    prompt_identity_includes_relation: bool = False,
    inductive_num_samples: int = 0,
    first_seen_edges: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """Select a fixed-budget mix of ordinary and natural first-touch queries.

    ``inductive_num_samples`` reserves up to that many slots inside ``num_samples``
    for queries whose source or target has no interaction at an earlier timestamp.
    The remaining slots use the existing ``most_recent`` or ``random`` policy.
    """
    _validate_edge_columns(split_edges)
    if first_seen_edges is None:
        first_seen_edges = split_edges
    _validate_edge_columns(first_seen_edges)

    num_samples = int(num_samples)
    inductive_num_samples = int(inductive_num_samples)
    sampling_skip_recent = max(0, int(sampling_skip_recent))
    if num_samples < 0:
        raise ValueError("num_samples must be >= 0")
    if inductive_num_samples < 0:
        raise ValueError("inductive_num_samples must be >= 0")
    if inductive_num_samples > num_samples:
        raise ValueError(
            "inductive_num_samples cannot exceed the total positive sample budget: "
            f"{inductive_num_samples} > {num_samples}"
        )

    sampling_strategy = str(sampling_strategy).strip().lower()
    if sampling_strategy not in {"most_recent", "random"}:
        raise ValueError(
            f"Unsupported sampling_strategy={sampling_strategy!r}. "
            "Use 'most_recent' or 'random'."
        )

    pool, identity_columns, collapsed_count = _prepare_prompt_pool(
        split_edges,
        collapse_prompt_duplicates=collapse_prompt_duplicates,
        prompt_identity_includes_relation=prompt_identity_includes_relation,
    )
    if inductive_num_samples > 0:
        first_seen = _first_seen_time_by_node(first_seen_edges)
        pool[SOURCE_FIRST_TOUCH_COLUMN] = pool["ts"].eq(pool["u"].map(first_seen))
        pool[TARGET_FIRST_TOUCH_COLUMN] = pool["ts"].eq(pool["i"].map(first_seen))
    else:
        # Avoid a full endpoint group-by in unchanged/default runs.
        pool[SOURCE_FIRST_TOUCH_COLUMN] = False
        pool[TARGET_FIRST_TOUCH_COLUMN] = False

    # Preserve the old skip semantics: it only applies to a truncated
    # most-recent selection. With prompt collapse enabled, it counts prompt
    # queries rather than duplicate edge rows.
    eligible_pool = pool
    if (
        sampling_strategy == "most_recent"
        and len(pool) > num_samples
        and sampling_skip_recent > 0
    ):
        eligible_pool = (
            pool.sort_values("ts", ascending=False, kind="mergesort")
            .iloc[sampling_skip_recent:]
            .copy()
        )

    if num_samples == 0 or len(eligible_pool) == 0:
        empty = eligible_pool.head(0).copy()
        empty[POSITIVE_SAMPLING_GROUP_COLUMN] = pd.Series(dtype="object")
        return empty, {
            "input_positive_edges": int(len(split_edges)),
            "eligible_prompt_queries": int(len(eligible_pool)),
            "selected_positive_queries": 0,
            "collapse_prompt_duplicates": bool(collapse_prompt_duplicates),
            "prompt_identity_columns": identity_columns,
            "collapsed_duplicate_rows": collapsed_count,
            "first_touch_computed": bool(inductive_num_samples > 0),
            "inductive_requested": inductive_num_samples,
            "inductive_reserved_selected": 0,
            "first_touch_selected_total": 0,
            "base_selected": 0,
        }

    inductive_mask = (
        eligible_pool[SOURCE_FIRST_TOUCH_COLUMN]
        | eligible_pool[TARGET_FIRST_TOUCH_COLUMN]
    )
    inductive_candidates = eligible_pool[inductive_mask]
    inductive_take = min(inductive_num_samples, len(inductive_candidates), num_samples)
    if inductive_take > 0:
        if sampling_strategy == "most_recent":
            inductive_selected = (
                inductive_candidates.sort_values(
                    "ts", ascending=False, kind="mergesort"
                )
                .head(inductive_take)
                .copy()
            )
        else:
            inductive_selected = inductive_candidates.sample(
                n=inductive_take,
                replace=False,
                random_state=int(random_seed),
            ).copy()
    else:
        inductive_selected = eligible_pool.head(0).copy()
    inductive_selected[POSITIVE_SAMPLING_GROUP_COLUMN] = "pseudo_inductive"

    # Remove every row with a reserved prompt identity, not just the retained
    # representative row. This prevents a duplicate prompt from re-entering
    # through the ordinary branch when collapse is disabled.
    if len(inductive_selected) > 0:
        selected_identity = pd.MultiIndex.from_frame(
            inductive_selected[identity_columns]
        )
        pool_identity = pd.MultiIndex.from_frame(eligible_pool[identity_columns])
        base_pool = eligible_pool[~pool_identity.isin(selected_identity)].copy()
    else:
        base_pool = eligible_pool

    base_take = min(num_samples - len(inductive_selected), len(base_pool))
    if base_take <= 0:
        base_selected = base_pool.head(0).copy()
    elif len(base_pool) <= base_take:
        base_selected = base_pool.copy()
    elif sampling_strategy == "most_recent":
        base_selected = (
            base_pool.sort_values("ts", ascending=False, kind="mergesort")
            .head(base_take)
            .copy()
        )
    else:
        # Offset the seed so the base draw is independent of the reserved draw.
        base_selected = base_pool.sample(
            n=base_take,
            replace=False,
            random_state=int(random_seed) + (1 if len(inductive_selected) > 0 else 0),
        ).copy()
    base_selected[POSITIVE_SAMPLING_GROUP_COLUMN] = "base"

    selected = pd.concat(
        [inductive_selected, base_selected], axis=0, ignore_index=False
    )
    selected = selected.sort_values("ts", ascending=True, kind="mergesort").copy()
    first_touch_selected = (
        selected[SOURCE_FIRST_TOUCH_COLUMN] | selected[TARGET_FIRST_TOUCH_COLUMN]
    )

    def _time_bound(frame: pd.DataFrame, reducer: str):
        if len(frame) == 0:
            return None
        value = getattr(frame["ts"], reducer)()
        return value.item() if hasattr(value, "item") else value

    stats = {
        "input_positive_edges": int(len(split_edges)),
        "prompt_pool_queries": int(len(pool)),
        "eligible_prompt_queries": int(len(eligible_pool)),
        "selected_positive_queries": int(len(selected)),
        "sampling_strategy": sampling_strategy,
        "sampling_skip_recent": sampling_skip_recent,
        "collapse_prompt_duplicates": bool(collapse_prompt_duplicates),
        "prompt_identity_columns": identity_columns,
        "collapsed_duplicate_rows": collapsed_count,
        "first_touch_computed": bool(inductive_num_samples > 0),
        "inductive_requested": inductive_num_samples,
        "inductive_candidates": int(len(inductive_candidates)),
        "inductive_reserved_selected": int(len(inductive_selected)),
        "first_touch_selected_total": int(first_touch_selected.sum()),
        "source_first_touch_selected": int(selected[SOURCE_FIRST_TOUCH_COLUMN].sum()),
        "target_first_touch_selected": int(selected[TARGET_FIRST_TOUCH_COLUMN].sum()),
        "base_selected": int(len(base_selected)),
        "selected_time_min": _time_bound(selected, "min"),
        "selected_time_max": _time_bound(selected, "max"),
        "inductive_time_min": _time_bound(inductive_selected, "min"),
        "inductive_time_max": _time_bound(inductive_selected, "max"),
        "base_time_min": _time_bound(base_selected, "min"),
        "base_time_max": _time_bound(base_selected, "max"),
        "selected_prompt_multiplicity_sum": int(
            selected[PROMPT_MULTIPLICITY_COLUMN].sum()
        ),
    }
    return selected, stats


__all__ = [
    "POSITIVE_SAMPLING_GROUP_COLUMN",
    "PROMPT_MULTIPLICITY_COLUMN",
    "SOURCE_FIRST_TOUCH_COLUMN",
    "TARGET_FIRST_TOUCH_COLUMN",
    "prompt_identity_columns",
    "select_training_positive_edges",
]
