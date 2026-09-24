#!/usr/bin/env python3
"""
LLM-based Link Prediction Evaluation for DTGB datasets.
Following DTGB standard protocol: 70% train, 15% val, 15% test
Metrics: AP (Average Precision) and AUC-ROC
"""
import logging
import os
import re
import warnings

import pandas as pd
from experiments.modules.amazon_movies_entity_text import (
    build_compact_amazon_movies_profile,
    compress_amazon_movies_entity_text,
)
from experiments.modules.llm_lp.experiment_calibration import (
    _apply_validation_sampled_binary_labels,
    _apply_validation_sampled_key_signal_reference,
    _apply_validation_sampled_threeway_labels,
    _build_expert_prediction_discriminative_debug,
    _build_key_signal_discriminative_debug,
    _build_lightweight_rrf_samples_for_split,
    _build_realized_discriminative_debug,
    _calibrate_validation_sampled_hybrid_gmm_overlap_band,
    _calibrate_validation_sampled_hybrid_uncertainty_band,
    _calibrate_validation_sampled_key_signal_reference,
    _calibrate_validation_sampled_threeway_thresholds,
    _resolve_hybrid_alignment_calibration_sizes,
    _resolve_validation_calibration_sizes,
    _uses_validation_sampled_binary,
    _uses_validation_sampled_key_signal_reference,
    _uses_validation_sampled_threeway,
    dataset_uses_dtgb_time_bucket,
)


def configure_runtime_logging():
    # Numba's CUDA driver logger can emit per-deallocation INFO spam during
    # GPU heuristic scoring, which corrupts tqdm progress bars.
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
    try:
        from numba.core.errors import NumbaPerformanceWarning

        warnings.filterwarnings(
            "ignore",
            message=r"Grid size .* under-utilization due to low occupancy",
            category=NumbaPerformanceWarning,
        )
    except Exception:
        pass


def _build_selection_summary(selection):
    if not isinstance(selection, dict):
        return None

    summary = {}
    for key in (
        "enabled",
        "selection_mode",
        "selection_rule",
        "requested_top_k",
        "requested_fraction",
        "requested_percent",
        "selected_count",
        "selected_fraction_realized",
        "selected_batches",
        "available_batches",
        "dtgb_eval_batch_size",
        "num_candidates",
        "center_threshold",
        "low_threshold",
        "high_threshold",
        "validation_requested_target_fraction",
        "validation_requested_target_count",
        "validation_realized_count",
        "validation_realized_fraction",
        "ambiguity_threshold",
        "gmm_weights",
        "gmm_means",
        "gmm_variances",
        "debug_route_cap_enabled",
        "debug_route_cap",
        "debug_route_cap_reason",
        "selected_count_before_debug_cap",
        "selected_fraction_before_debug_cap",
    ):
        if key in selection:
            summary[key] = selection[key]
    return summary


def _build_metric_snapshot(metrics):
    if not isinstance(metrics, dict):
        return None

    summary = {}
    for key in (
        "ap",
        "auc",
        "accuracy",
        "ap_global",
        "auc_global",
    ):
        if key in metrics:
            summary[key] = metrics[key]
    return summary


def _build_hybrid_summary(hybrid):
    if not isinstance(hybrid, dict):
        return None

    summary = {}
    for key in (
        "enabled",
        "mode",
        "backbone_name",
        "backbone_score_field",
        "backbone_score_space",
        "selection_mode",
        "merge_method",
        "score_alignment",
        "fusion_alpha",
        "num_selected_batches",
        "num_selected_samples",
    ):
        if key in hybrid:
            summary[key] = hybrid[key]

    summary["selection"] = _build_selection_summary(hybrid.get("selection"))
    summary["backbone_baseline_fullset"] = _build_metric_snapshot(
        hybrid.get("backbone_baseline_fullset", hybrid.get("rrf_baseline_fullset"))
    )
    summary["uplift_vs_backbone"] = _build_metric_snapshot(
        hybrid.get("uplift_vs_backbone", hybrid.get("uplift_vs_rrf"))
    )
    return summary


def _build_oracle_budget_summary(oracle_curve):
    if not isinstance(oracle_curve, dict):
        return None

    summary = {}
    for key in ("enabled", "selection_mode", "merge_method", "fusion_alpha"):
        if key in oracle_curve:
            summary[key] = oracle_curve[key]

    summary["rrf_baseline_fullset"] = _build_metric_snapshot(oracle_curve.get("rrf_baseline_fullset"))
    budget_rows = []
    for row in oracle_curve.get("budget_curve", []):
        if not isinstance(row, dict):
            continue
        budget_rows.append(
            {
                "requested_fraction": row.get("requested_fraction"),
                "requested_percent": row.get("requested_percent"),
                "selection": _build_selection_summary(row.get("selection")),
                "metrics": _build_metric_snapshot(row.get("metrics")),
                "uplift_vs_rrf": _build_metric_snapshot(row.get("uplift_vs_rrf")),
            }
        )
    summary["budget_curve"] = budget_rows
    return summary


def build_trial_result_summary(result):
    summary = {}
    for key in (
        "ap",
        "auc",
        "accuracy",
        "ap_global",
        "auc_global",
        "num_samples",
        "num_positive",
        "num_negative",
        "dtgb_eval_batch_size",
        "dtgb_num_metric_batches",
        "dtgb_metric_aggregation",
        "parse_stats",
        "token_usage",
        "prompt_embedding_shards",
    ):
        if key in result:
            summary[key] = result[key]

    if "hybrid" in result:
        summary["hybrid"] = _build_hybrid_summary(result.get("hybrid"))

    if "oracle_rerank_upper_bound" in result:
        summary["oracle_rerank_upper_bound"] = _build_oracle_budget_summary(
            result.get("oracle_rerank_upper_bound")
        )

    if "rrf_uncertainty_proxy" in result:
        summary["rrf_uncertainty_proxy"] = {
            "selection": _build_selection_summary(
                result.get("rrf_uncertainty_proxy", {}).get("selection")
            )
        }

    if "validation_sampled_threeway_debug" in result:
        summary["validation_sampled_threeway_debug"] = result.get(
            "validation_sampled_threeway_debug"
        )
    if "rrf_validation_band_debug" in result:
        summary["rrf_validation_band_debug"] = result.get("rrf_validation_band_debug")
    if "expert_discrimination_debug" in result:
        summary["expert_discrimination_debug"] = result.get("expert_discrimination_debug")
    if "hybrid_routing_debug" in result:
        summary["hybrid_routing_debug"] = result.get("hybrid_routing_debug")
    if "hybrid_alignment_debug" in result:
        summary["hybrid_alignment_debug"] = result.get("hybrid_alignment_debug")
    if "key_signal_discrimination_debug" in result:
        summary["key_signal_discrimination_debug"] = result.get(
            "key_signal_discrimination_debug"
        )

    return summary


def normalize_hybrid_selection(selected_sample_indices, selection_meta, total_samples):
    if selected_sample_indices is None:
        selected_sample_indices = []
    normalized_indices = [int(idx) for idx in selected_sample_indices]
    if len(normalized_indices) != len(set(normalized_indices)):
        raise RuntimeError("Hybrid selection contains duplicate sample indices.")

    max_index = int(total_samples) - 1
    for idx in normalized_indices:
        if idx < 0 or idx > max_index:
            raise RuntimeError(
                f"Hybrid selection index {idx} is out of range for {total_samples} samples."
            )
    normalized_indices.sort()

    normalized_meta = dict(selection_meta) if isinstance(selection_meta, dict) else {}
    if "selected_count" in normalized_meta:
        if int(normalized_meta["selected_count"]) != len(normalized_indices):
            raise RuntimeError(
                "Hybrid selection metadata count does not match selected index list."
            )
    meta_indices = normalized_meta.get("selected_sample_indices")
    if meta_indices is not None:
        meta_norm = [int(idx) for idx in meta_indices]
        if len(meta_norm) != len(set(meta_norm)):
            raise RuntimeError("Hybrid selection metadata has duplicate sample indices.")
        meta_norm.sort()
        if meta_norm != normalized_indices:
            raise RuntimeError(
                "Hybrid selection metadata indices do not match selected index list."
            )

    normalized_meta["selected_sample_indices"] = list(normalized_indices)
    normalized_meta["selected_count"] = int(len(normalized_indices))
    return normalized_indices, normalized_meta


def apply_hybrid_debug_route_cap(
    samples,
    selected_sample_indices,
    selection_meta,
    *,
    score_field="rrf_score",
    max_routed_samples=0,
):
    cap = int(max_routed_samples)
    if cap <= 0:
        return selected_sample_indices, selection_meta

    selected_indices = [int(idx) for idx in (selected_sample_indices or [])]
    if len(selected_indices) <= cap:
        return selected_indices, selection_meta

    normalized_meta = dict(selection_meta) if isinstance(selection_meta, dict) else {}
    selection_mode = str(normalized_meta.get("selection_mode", "")).strip().lower()

    center_threshold = normalized_meta.get("center_threshold")
    if center_threshold is None:
        low = normalized_meta.get("low_threshold")
        high = normalized_meta.get("high_threshold")
        if low is not None and high is not None:
            center_threshold = 0.5 * (float(low) + float(high))

    if center_threshold is not None:
        center = float(center_threshold)
        ranked = sorted(
            selected_indices,
            key=lambda idx: (
                abs(float(samples[idx].get(score_field, 0.0)) - center),
                -float(samples[idx].get(score_field, 0.0)),
                int(idx),
            ),
        )
        cap_reason = "closest_to_routing_center"
    else:
        ranked = list(selected_indices)
        cap_reason = "selection_order"

    kept = ranked[:cap]
    kept_sorted = sorted(int(idx) for idx in kept)

    normalized_meta["debug_route_cap_enabled"] = True
    normalized_meta["debug_route_cap"] = int(cap)
    normalized_meta["debug_route_cap_reason"] = str(cap_reason)
    normalized_meta["selected_count_before_debug_cap"] = int(len(selected_indices))
    normalized_meta["selected_fraction_before_debug_cap"] = (
        float(len(selected_indices)) / float(len(samples)) if samples else 0.0
    )
    normalized_meta["debug_priority_center_threshold"] = (
        float(center_threshold) if center_threshold is not None else None
    )
    normalized_meta["selected_sample_indices_before_debug_cap"] = [
        int(idx) for idx in selected_indices
    ]
    normalized_meta["selected_sample_indices"] = kept_sorted
    normalized_meta["selected_count"] = int(len(kept_sorted))
    normalized_meta["selected_fraction_realized"] = (
        float(len(kept_sorted)) / float(len(samples)) if samples else 0.0
    )
    if selection_mode in {
        "validation_sampled_uncertainty_band",
        "validation_sampled_gmm_overlap_band",
    }:
        normalized_meta["validation_realized_count_after_debug_cap"] = int(len(kept_sorted))
        normalized_meta["validation_realized_fraction_after_debug_cap"] = (
            float(len(kept_sorted)) / float(len(samples)) if samples else 0.0
        )
    return kept_sorted, normalized_meta


def maybe_print_token_usage_summary(results, trial_idx, eval_split):
    if not isinstance(results, dict):
        return

    token_usage = results.get("token_usage")
    if not isinstance(token_usage, dict) or not token_usage:
        return

    prompt_tokens = token_usage.get("prompt_tokens")
    completion_tokens = token_usage.get("completion_tokens")
    total_tokens = token_usage.get("total_tokens")

    details = []
    if prompt_tokens is not None:
        details.append(f"prompt={int(prompt_tokens)}")
    if completion_tokens is not None:
        details.append(f"completion={int(completion_tokens)}")
    if total_tokens is not None:
        details.append(f"total={int(total_tokens)}")

    if not details:
        return

    scope = "selected-slice LLM calls" if results.get("hybrid", {}).get("enabled") else "LLM calls"
    print(
        f"  -> Trial {trial_idx} [{eval_split}] Token usage ({scope}): "
        + ", ".join(details)
    )


def dataset_is_icews1819(dataset_name):
    return str(dataset_name).strip().upper() == "ICEWS1819"


def dataset_is_googlemap_ct(dataset_name):
    return str(dataset_name).strip().upper() == "GOOGLEMAP_CT"


def dataset_is_amazon_movies(dataset_name):
    return str(dataset_name).strip().upper() == "AMAZON_MOVIES"


def dataset_is_stack_elec(dataset_name):
    return str(dataset_name).strip().upper() == "STACK_ELEC"


def _normalize_entity_text(text):
    if text is None:
        return ""
    if pd.isna(text):
        return ""
    return " ".join(str(text).split()).strip()


def _compress_icews_entity_text(text):
    clean = _normalize_entity_text(text)
    if not clean:
        return clean

    country_match = re.search(r"Country is\s+(.+?)(?:[.;]|$)", clean, flags=re.IGNORECASE)
    country = _normalize_entity_text(country_match.group(1)) if country_match else ""

    name = clean
    for marker in (": Sector is", ": sector is", ":Country is", ": country is"):
        idx = name.find(marker)
        if idx != -1:
            name = name[:idx]
            break
    if name == clean and ":" in clean:
        name = clean.split(":", 1)[0]
    name = _normalize_entity_text(name)

    if name and country:
        lowered_name = name.lower()
        lowered_country = country.lower()
        if lowered_country in lowered_name:
            return name
        return f"{name} ({country})"
    if name:
        return name
    return clean


def _truncate_words(text, max_words):
    clean = _normalize_entity_text(text)
    if not clean:
        return ""
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _stack_elec_extract_field(clean, label, next_labels):
    if not clean:
        return ""
    pattern = rf"{re.escape(label)}\s*:\s*(.+?)(?=(?:\.\s*(?:{'|'.join(re.escape(x) for x in next_labels)})\s*:)|$)"
    match = re.search(pattern, clean, flags=re.IGNORECASE)
    return _normalize_entity_text(match.group(1)) if match else ""


def _looks_like_stack_elec_user_profile(clean):
    lowered = clean.lower()
    return (
        lowered.startswith("name:")
        and "location:" in lowered
        and "introduction:" in lowered
    )


def _looks_like_stack_elec_post_text(clean):
    lowered = clean.lower()
    return lowered.startswith("title:") and "post:" in lowered


def _select_stack_elec_primary_clause(text):
    clean = _normalize_entity_text(text)
    if not clean:
        return ""

    candidates = [
        piece.strip(" -|,;:.")
        for piece in re.split(r"(?<=[.!?])\s+|\s+[|/]\s+|\s+-\s+|\s{2,}", clean)
        if piece.strip(" -|,;:.")
    ]
    if not candidates:
        candidates = [clean]

    for candidate in candidates:
        letters = sum(ch.isalpha() for ch in candidate)
        nonspace = sum(not ch.isspace() for ch in candidate)
        if nonspace >= 12 and letters / max(nonspace, 1) < 0.45:
            continue
        lowered = candidate.lower()
        if lowered in {"unknown", "none", "n/a"}:
            continue
        return candidate
    return ""


def _clean_stack_elec_intro(text):
    clean = _normalize_entity_text(text)
    if not clean or clean.lower() == "unknown":
        return ""

    clean = re.sub(r"https?://\S+|www\.\S+", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"&(?:quot|amp|lt|gt|nbsp);", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", clean)
    clean = re.sub(r"\([^)]*https?://[^)]*\)", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(" -|,;:.")
    if not clean:
        return ""
    return _select_stack_elec_primary_clause(clean)


def _clean_stack_elec_post_fragment(text, max_words):
    clean = _normalize_entity_text(text)
    if not clean or clean.lower() == "unknown":
        return ""
    clean = re.sub(r"https?://\S+|www\.\S+", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"&(?:quot|amp|lt|gt|nbsp);", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(" -|,;:.")
    if not clean:
        return ""
    return _select_stack_elec_primary_clause(clean)


def _compress_stack_elec_entity_text(text):
    clean = _normalize_entity_text(text)
    if not clean:
        return clean

    if _looks_like_stack_elec_post_text(clean):
        title = _stack_elec_extract_field(clean, "Title", ("Post",))
        body = _stack_elec_extract_field(clean, "Post", ())
        title = _clean_stack_elec_post_fragment(title, 12)
        body = _clean_stack_elec_post_fragment(body, 16)

        if title and body:
            body_lower = body.lower()
            title_lower = title.lower()
            if title_lower not in body_lower:
                return f"{title} | {body}"
            return title
        if title:
            return title
        if body:
            return body
        return ""

    # Already compacted post labels should pass through unchanged.
    if not _looks_like_stack_elec_user_profile(clean):
        return clean

    name = _stack_elec_extract_field(clean, "Name", ("Location", "Introduction"))
    location = _stack_elec_extract_field(clean, "Location", ("Introduction",))
    intro = _stack_elec_extract_field(clean, "Introduction", ())
    intro = _clean_stack_elec_intro(intro)

    parts = []
    if name:
        parts.append(f"Name: {name}")
    if location and location.lower() != "unknown":
        parts.append(f"Location: {location}")
    if intro:
        parts.append(f"Profile: {intro}")

    compact = "; ".join(parts) if parts else clean
    return _normalize_entity_text(compact)


def _extract_googlemap_field(clean, label, next_labels):
    if not clean:
        return ""
    if str(label).lower() == "address":
        label_prefix = rf"(?:^|(?:\.\s+)){re.escape(label)}\s*:?\s*"
    else:
        label_prefix = rf"(?:^|(?:\.\s+)){re.escape(label)}\s*:\s*"
    if next_labels:
        boundary = rf"(?=(?:\.\s*(?:{'|'.join(re.escape(x) for x in next_labels)})\s*:?)|$)"
    else:
        boundary = r"$"
    pattern = rf"{label_prefix}(.+?){boundary}"
    match = re.search(pattern, clean, flags=re.IGNORECASE)
    return _normalize_entity_text(match.group(1)) if match else ""


def _parse_googlemap_entity_text(text):
    clean = _normalize_entity_text(text)
    if not clean:
        return {
            "clean": "",
            "name": "",
            "location": "",
            "categories": [],
            "description": "",
        }

    name = _extract_googlemap_field(clean, "Name", ("Address", "Category", "Description"))
    address = _extract_googlemap_field(clean, "Address", ("Category", "Description"))
    raw_category_text = _extract_googlemap_field(clean, "Category", ("Description",))
    description = _extract_googlemap_field(clean, "Description", ())

    categories = []
    seen_categories = set()
    for piece in raw_category_text.split(","):
        token = _normalize_entity_text(piece.strip(" ."))
        if not token:
            continue
        key = token.lower()
        if key in seen_categories:
            continue
        seen_categories.add(key)
        categories.append(token)

    specific_categories = [
        token for token in categories
        if token.lower() not in {"restaurant", "food", "point of interest", "establishment"}
    ]
    if specific_categories:
        categories = specific_categories

    location = ""
    if address:
        zip_match = re.search(
            r",\s*([^,]+),\s*([A-Z]{2})(?:\s+\d{5}(?:-\d{4})?)?\s*$",
            address,
        )
        if zip_match:
            city = _normalize_entity_text(zip_match.group(1))
            state = _normalize_entity_text(zip_match.group(2))
            if city and state:
                location = f"{city}, {state}"
        else:
            parts = [part.strip() for part in address.split(",") if _normalize_entity_text(part)]
            if len(parts) >= 2:
                location = _normalize_entity_text(", ".join(parts[-2:]))

    if description.lower() == "unknown":
        description = ""

    return {
        "clean": clean,
        "name": name,
        "location": location,
        "categories": categories,
        "description": description,
    }


def _compress_googlemap_entity_text(text):
    parsed = _parse_googlemap_entity_text(text)
    clean = parsed["clean"]
    if not clean:
        return clean

    # User nodes are often just person names; keep them intact.
    if not parsed["name"] and not parsed["categories"] and not parsed["location"]:
        return clean

    parts = []
    if parsed["name"]:
        parts.append(parsed["name"])
    if parsed["categories"]:
        parts.append(", ".join(parsed["categories"][:3]))
    if parsed["location"]:
        parts.append(parsed["location"])
    if parsed["description"]:
        parts.append(_truncate_words(parsed["description"], 12))
    return " | ".join(part for part in parts if part) or clean


def _build_compact_googlemap_profile(text):
    parsed = _parse_googlemap_entity_text(text)
    clean = parsed["clean"]
    if not clean:
        return ""

    if not parsed["name"] and not parsed["categories"] and not parsed["location"]:
        return ""

    parts = []
    if parsed["location"]:
        parts.append(f"Area: {parsed['location']}")
    if parsed["categories"]:
        parts.append("Categories: " + ", ".join(parsed["categories"][:4]))
    if parsed["description"]:
        parts.append("Summary: " + _truncate_words(parsed["description"], 18))
    return "; ".join(parts)


def build_prompt_entity_map(
    dataset_name,
    entity_map,
    entity_name_mode="auto",
    disable_stack_elec_prompt_cleaning=False,
):
    mode = str(entity_name_mode).strip().lower()
    if mode not in {"auto", "raw", "compressed", "compressed_profile"}:
        raise ValueError(
            "entity_name_mode must be one of "
            "{'auto', 'raw', 'compressed', 'compressed_profile'}"
        )

    if mode == "raw":
        return entity_map

    if dataset_is_icews1819(dataset_name):
        compressed = {}
        changed = 0
        for entity_id, text in entity_map.items():
            short_text = _compress_icews_entity_text(text)
            compressed[entity_id] = short_text
            if short_text != text:
                changed += 1
        print(
            "Prompt entity display map: "
            f"compressed {changed}/{len(entity_map)} entity strings "
            f"(dataset={dataset_name}, mode={mode}; keeping raw text for embeddings)."
        )
        return compressed

    if dataset_is_googlemap_ct(dataset_name) and mode in {"compressed", "compressed_profile"}:
        compressed = {}
        changed = 0
        for entity_id, text in entity_map.items():
            short_text = _compress_googlemap_entity_text(text)
            compressed[entity_id] = short_text
            if short_text != text:
                changed += 1
        print(
            "Prompt entity display map: "
            f"compressed {changed}/{len(entity_map)} entity strings "
            f"(dataset={dataset_name}, mode={mode}; keeping name/category/location summaries for Googlemap_CT)."
        )
        return compressed

    if dataset_is_amazon_movies(dataset_name) and mode in {"compressed", "compressed_profile"}:
        compressed = {}
        changed = 0
        for entity_id, text in entity_map.items():
            short_text = compress_amazon_movies_entity_text(text)
            compressed[entity_id] = short_text
            if short_text != text:
                changed += 1
        print(
            "Prompt entity display map: "
            f"compressed {changed}/{len(entity_map)} entity strings "
            f"(dataset={dataset_name}, mode={mode}; trimming Amazon_movies metadata to title/category/summary)."
        )
        return compressed

    if (
        dataset_is_stack_elec(dataset_name)
        and mode in {"auto", "compressed", "compressed_profile"}
        and disable_stack_elec_prompt_cleaning
    ):
        print(
            "Prompt entity display map: "
            f"Stack_elec prompt-side cleaning disabled "
            f"(dataset={dataset_name}, mode={mode}; using raw entity text for prompt display)."
        )
        return entity_map

    if dataset_is_stack_elec(dataset_name) and mode in {"auto", "compressed", "compressed_profile"}:
        compressed = {}
        changed = 0
        for entity_id, text in entity_map.items():
            short_text = _compress_stack_elec_entity_text(text)
            compressed[entity_id] = short_text
            if short_text != text:
                changed += 1
        print(
            "Prompt entity display map: "
            f"compressed {changed}/{len(entity_map)} entity strings "
            f"(dataset={dataset_name}, mode={mode}; shortening Stack_elec user bios for prompt display)."
        )
        return compressed

    if dataset_is_googlemap_ct(dataset_name):
        return entity_map

    if dataset_is_amazon_movies(dataset_name):
        return entity_map

    if mode in {"compressed", "compressed_profile"}:
        print(
            "Prompt entity display map: "
            f"no dataset-specific compression applied for dataset={dataset_name}."
        )
    return entity_map


def entity_name_mode_uses_compact_profile(dataset_name, entity_name_mode):
    mode = str(entity_name_mode).strip().lower()
    return mode == "compressed_profile" and (
        dataset_is_icews1819(dataset_name)
        or dataset_is_googlemap_ct(dataset_name)
        or dataset_is_amazon_movies(dataset_name)
    )


def _extract_icews_sectors(text):
    clean = _normalize_entity_text(text)
    if not clean:
        return []
    sector_match = re.search(
        r"Sector is\s+(.+?)(?:\.\s*Country is|;\s*Country is|$)",
        clean,
        flags=re.IGNORECASE,
    )
    if not sector_match:
        return []
    raw_sector_text = _normalize_entity_text(sector_match.group(1))
    if not raw_sector_text or raw_sector_text.lower() == "none":
        return []
    sectors = []
    seen = set()
    for piece in raw_sector_text.split(","):
        token = _normalize_entity_text(piece)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        sectors.append(token)
    return sectors


def _infer_icews_profile_tags(text):
    sectors = _extract_icews_sectors(text)
    if not sectors:
        return []

    lower_sectors = [sector.lower() for sector in sectors]

    def has_any(*keywords):
        for sector in lower_sectors:
            for keyword in keywords:
                if keyword in sector:
                    return True
        return False

    tags = []
    tag_rules = [
        ("government", ("government", "ministry", "cabinet", "executive office")),
        ("executive", ("executive",)),
        ("party", ("party", "parties")),
        ("legislative", ("legislative", "parliament", "lower house", "upper house", "unicameral")),
        ("security/military", ("military", "defense", "security")),
        ("media", ("media",)),
        ("business/economic", ("business", "finance", "commerce", "trade", "industrial", "enterprise")),
        ("civil society", ("nongovernmental", "activist", "human rights", "social")),
        ("international/igo", ("international government organization", "igo", "regional diplomatic", "regional defense", "global")),
        ("religious", ("religious", "muslim", "catholic", "sunni")),
        ("insurgent/extremist", ("insurgent", "terrorist", "dissident", "radical", "extremist", "fundamentalist")),
        ("ethnic", ("ethnic", "bantu", "pashtun")),
    ]

    for label, keywords in tag_rules:
        if has_any(*keywords):
            tags.append(label)

    if not tags:
        tags = sectors[:3]
    return tags[:4]


def _build_compact_icews_profile(text):
    clean = _normalize_entity_text(text)
    if not clean:
        return ""

    country_match = re.search(r"Country is\s+(.+?)(?:[.;]|$)", clean, flags=re.IGNORECASE)
    country = _normalize_entity_text(country_match.group(1)) if country_match else ""
    tags = _infer_icews_profile_tags(clean)

    parts = []
    if country:
        parts.append(f"Country: {country}")
    if tags:
        parts.append("Profile: " + ", ".join(tags))
    return "; ".join(parts)


def build_auto_summary_entity_map(dataset_name, entity_map):
    if dataset_is_icews1819(dataset_name):
        summary_map = {}
        non_empty = 0
        for entity_id, text in entity_map.items():
            summary_text = _build_compact_icews_profile(text)
            summary_map[entity_id] = summary_text
            if summary_text:
                non_empty += 1
        print(
            "Auto summary map: "
            f"built {non_empty}/{len(entity_map)} compact ICEWS1819 endpoint profiles."
        )
        return summary_map

    if dataset_is_googlemap_ct(dataset_name):
        summary_map = {}
        non_empty = 0
        for entity_id, text in entity_map.items():
            summary_text = _build_compact_googlemap_profile(text)
            summary_map[entity_id] = summary_text
            if summary_text:
                non_empty += 1
        print(
            "Auto summary map: "
            f"built {non_empty}/{len(entity_map)} compact Googlemap_CT endpoint profiles."
        )
        return summary_map

    if dataset_is_amazon_movies(dataset_name):
        summary_map = {}
        non_empty = 0
        for entity_id, text in entity_map.items():
            summary_text = build_compact_amazon_movies_profile(text)
            summary_map[entity_id] = summary_text
            if summary_text:
                non_empty += 1
        print(
            "Auto summary map: "
            f"built {non_empty}/{len(entity_map)} compact Amazon_movies endpoint profiles."
        )
        return summary_map

    return None


def _resolve_dataset_root(dataset_name):
    candidates = [
        os.path.join("..", "DyLink_Datasets", dataset_name),
        os.path.join(".", "DyLink_Datasets", dataset_name),
    ]
    for root in candidates:
        if os.path.isdir(root):
            return root
    raise FileNotFoundError(
        f"Dataset folder not found for '{dataset_name}'. Checked: {candidates}"
    )


def load_dataset_data(dataset_name, entity_text_path=None, relation_text_path=None):
    dataset_root = _resolve_dataset_root(dataset_name)
    print(f"\nLoading dataset '{dataset_name}' from {dataset_root}...")
    edges = pd.read_csv(os.path.join(dataset_root, "edge_list.csv"))
    resolved_entity_text_path = entity_text_path or os.path.join(dataset_root, "entity_text.csv")
    resolved_relation_text_path = relation_text_path or os.path.join(dataset_root, "relation_text.csv")
    entities = pd.read_csv(resolved_entity_text_path)
    relations = pd.read_csv(resolved_relation_text_path)

    entity_map = dict(zip(entities["i"], entities["text"]))
    relation_map = dict(zip(relations["i"], relations["text"]))

    if entity_text_path:
        print(f"✓ Using custom entity text from {resolved_entity_text_path}")
    if relation_text_path:
        print(f"✓ Using custom relation text from {resolved_relation_text_path}")
    print(f"✓ Loaded {len(edges)} edges, {len(entities)} entities, {len(relations)} relations")
    return edges, entity_map, relation_map


def load_summary_entity_map(summary_csv_path):
    summary_df = pd.read_csv(summary_csv_path)
    if "i" not in summary_df.columns or "text" not in summary_df.columns:
        raise ValueError(
            f"Summary CSV must contain columns 'i' and 'text': {summary_csv_path}"
        )
    summary_map = {}
    for row in summary_df[["i", "text"]].itertuples(index=False):
        node_id = int(row[0])
        text = "" if pd.isna(row[1]) else str(row[1]).strip()
        summary_map[node_id] = text
    return summary_map


_FEW_SHOT_BLOCK_HEADER_RE = re.compile(
    r"^\s*(?:##\s*)?(?:ID|Example)\s+\d+\b.*$",
    flags=re.MULTILINE,
)


def load_few_shot_examples(few_shot_path, max_examples=None):
    path = os.path.abspath(str(few_shot_path))
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().replace("\r\n", "\n")

    matches = list(_FEW_SHOT_BLOCK_HEADER_RE.finditer(text))
    examples = []
    if matches:
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if not block:
                continue
            if block.startswith("```"):
                lines = block.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                block = "\n".join(lines).strip()
            if block:
                examples.append(block)
    else:
        block = text.strip()
        if block:
            examples = [block]

    if max_examples is not None:
        examples = examples[: int(max_examples)]

    if not examples:
        raise ValueError(f"No usable few-shot examples found in {path}")
    return examples


def print_run_config(
    args,
    dp_ctx,
    semantic_use_smoothing,
    use_raw_key_signals,
    use_percentile_key_signals,
    few_shot_examples_count,
):
    prior_signal_visible = not args.hide_expert_prediction
    key_signals_visible = not args.hide_key_signals

    if args.export_prompt_dataset_dir:
        backend = "prompt_dataset_export"
    elif args.rrf_only:
        backend = "rrf_only"
    elif args.use_openai_api:
        backend = "openai"
    elif args.use_transformers:
        backend = "transformers"
    else:
        backend = "vllm"

    print("=" * 80)
    print(
        f"LLM Link Prediction Evaluation - {args.dataset_name} "
        f"({args.eval_split.capitalize()})"
    )
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Inference backend: {backend}")
    print(f"Prompt entity name mode: {args.entity_name_mode}")
    if dataset_is_stack_elec(args.dataset_name):
        print(
            "Stack_elec prompt-side cleaning: "
            f"{'disabled' if args.disable_stack_elec_prompt_cleaning else 'enabled'}"
        )
    if entity_name_mode_uses_compact_profile(args.dataset_name, args.entity_name_mode):
        print("Prompt entity profile mode: compact one-time ICEWS endpoint profiles")
    if args.use_openai_api:
        print(f"OpenAI model: {args.openai_model}")
        if args.openai_base_url:
            print(f"OpenAI base URL: {args.openai_base_url}")
        print(f"OpenAI reasoning effort: {args.openai_reasoning_effort}")
        print(f"OpenAI async concurrency: {args.openai_concurrency}")
        if args.openai_max_requests_per_sec > 0:
            print(f"OpenAI client-side RPS cap: {args.openai_max_requests_per_sec:g}")
    print(f"Samples: {args.num_samples} positive, {args.num_samples * args.negative_ratio} negative")
    print(
        "Split: "
        f"{(1 - args.val_ratio - args.test_ratio) * 100:.0f}% train, "
        f"{args.val_ratio * 100:.0f}% val, {args.test_ratio * 100:.0f}% test"
    )
    print(
        "DTGB metric aggregation: "
        f"batch-mean over {args.dtgb_eval_batch_size} positive queries per evaluation batch"
    )
    if dp_ctx["enabled"]:
        print(
            f"Data Parallel: size={dp_ctx['size']}, rank={dp_ctx['rank']}, "
            f"run_id={dp_ctx['run_id']}"
        )
        print(f"Data Parallel sync dir: {dp_ctx['sync_dir']}")
        print(
            "Data Parallel sync cleanup: "
            f"{'keep' if args.keep_data_parallel_sync else 'auto-clean run dir on success'}"
        )

    if args.include_edge_type:
        print("Edge-type prompt mode: full")
    elif getattr(args, "include_edge_type_except_target", False):
        print("Edge-type prompt mode: observed-only (target relation hidden)")
    else:
        print("Edge-type prompt mode: off")
    if (args.semantic_history or args.semantic_history_entity_mode) and not args.rrf_only:
        fusion_tau_desc = "auto" if args.semantic_fusion_tau is None else f"{float(args.semantic_fusion_tau):g}"
        hist_mode = (
            "entity-centric (recency-first, semantic tie-break)"
            if args.semantic_history_entity_mode
            else "event-centric (similarity-recency rank-fusion)"
        )
        print(
            "Semantic-history mode: "
            f"{'smoothed' if semantic_use_smoothing else 'raw-no-smoothing'} ({hist_mode})"
        )
        print(
            "Semantic event-fusion: "
            f"alpha={args.semantic_fusion_alpha:g}, "
            f"tau={fusion_tau_desc}, "
            f"recency_speed={args.semantic_fusion_recency_speed:g}"
        )
        if args.history_preserve_recent_k > 0:
            if args.semantic_history_entity_mode:
                print(
                    "Recent-history preservation is ignored in semantic entity mode "
                    f"(requested K={args.history_preserve_recent_k})."
                )
            else:
                print(
                    "Recent-history preservation inside semantic window: "
                    f"K={args.history_preserve_recent_k}"
                )
        if args.semantic_hub_penalty_alpha > 0.0:
            print(f"Semantic source-history hub penalty alpha: {args.semantic_hub_penalty_alpha}")
    if args.mutual_summary_count_recency:
        print(
            "Mutual-history prompt mode: existence + retained distinct-time "
            "count + latest-event recency"
        )
    elif args.mutual_timestamps_only:
        mode = "dedup-consecutive" if args.mutual_timestamps_dedup else "keep-duplicates"
        print(f"Mutual-history prompt mode: timestamps-only ({mode})")
    if args.common_neighbors_names_only:
        print("Common-neighbors prompt mode: names-only")
    if args.compact_common_neighbors_top_k:
        novelty_suffix = (
            ", excluding entities already shown in endpoint histories"
            if args.compact_common_neighbors_novel_only
            else ""
        )
        print(
            "Common-neighbors prompt mode: compact natural activity "
            f"(top_k={args.compact_common_neighbors_top_k}{novelty_suffix})"
        )
    if args.common_neighbors_semantic:
        print(
            "Common-neighbor selection mode: semantic top-K to target, "
            "then destination-recency ordering"
        )
    if args.sample_creation_monitor_every > 0:
        print(
            "Sample-creation monitor: "
            f"enabled (every {args.sample_creation_monitor_every} positive queries)"
        )
    if args.sample_creation_profile:
        profile_output = args.sample_creation_profile_output or "(stdout-only summary)"
        print(
            "Sample-creation profiler: "
            f"enabled (sort={args.sample_creation_profile_sort}, "
            f"top_n={args.sample_creation_profile_top_n}, output={profile_output})"
        )
    if args.ablate_reasoning_guidance:
        print("Prompt ablation mode: reasoning guidance removed")
    print(f"Key-signal mode: {args.key_signal_mode} (default is bucket)")
    print(f"Key-signal fields: {', '.join(args.key_signal_fields)}")
    if _uses_validation_sampled_key_signal_reference(args.key_signal_reference):
        calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
        print(
            "Key-signal reference: validation_sampled_global "
            f"(validation positives={calibration_num_samples}, "
            f"negative_ratio={calibration_negative_ratio})"
        )
    else:
        print(f"Key-signal reference: {args.key_signal_reference}")
    if args.rrf_mode == "sequential_pointwise":
        print(
            f"RRF mode: {args.rrf_mode} "
            f"(k={args.rrf_k}, sequential_rank_bins={args.sequential_rank_bins})"
        )
    elif args.rrf_mode == "train_pool_pointwise":
        print(
            f"RRF mode: {args.rrf_mode} "
            f"(k={args.rrf_k}, pool_size={args.rrf_pointwise_pool_size}, "
            f"num_pools={args.rrf_pointwise_num_pools})"
        )
    else:
        print(f"RRF mode: {args.rrf_mode} (k={args.rrf_k})")
    expert_source = getattr(args, "expert_prediction_source", "rrf")
    if args.expert_prediction_mode == "fixed_threshold":
        print(
            f"Prior-signal mode: {args.expert_prediction_mode} "
            f"(source={expert_source}, "
            f"threshold={args.expert_prediction_fixed_threshold:.6f})"
        )
    elif _uses_validation_sampled_threeway(args.expert_prediction_mode):
        calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
        print(
            "Prior-signal mode: validation_sampled_threeway "
            f"(source={expert_source}, "
            f"validation positives={calibration_num_samples}, "
            f"negative_ratio={calibration_negative_ratio}, "
            f"low_neg_q={args.validation_calibration_low_neg_quantile:.2f}, "
            f"high_pos_q={args.validation_calibration_high_pos_quantile:.2f})"
        )
    else:
        print(f"Prior-signal mode: {args.expert_prediction_mode} (source={expert_source})")
    print(
        "Prompt signal plan: "
        f"prior_signal={'shown' if prior_signal_visible else 'hidden'}, "
        f"key_signals={'shown' if key_signals_visible else 'hidden'}"
    )
    print(
        "Prompt signal wiring: "
        "prior_signal follows --expert_prediction_mode and --expert_prediction_source; "
        "the selected KEY SIGNALS follow --key_signal_mode, --key_signal_fields, "
        "and --key_signal_reference."
    )
    print(
        "RRF/expert debug: "
        f"{'enabled' if (args.enable_rrf_validation_band_debug or _uses_validation_sampled_threeway(args.expert_prediction_mode)) else 'disabled'}"
    )
    if _uses_validation_sampled_threeway(args.expert_prediction_mode):
        print(
            "Prompt signal note: validation_sampled_threeway recalibrates only the prior signal "
            "from validation RRF scores. It does not recalibrate the selected KEY SIGNALS."
        )
    if _uses_validation_sampled_key_signal_reference(args.key_signal_reference):
        print(
            "Prompt signal note: validation_sampled_global freezes the selected KEY SIGNALS "
            "from a sampled validation subset instead of calibrating them on the eval stream."
        )
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
    if hybrid_enabled:
        if args.hybrid_selection_mode == "pointwise_fixed_threshold_band":
            print(
                "Hybrid selection mode: "
                f"{args.hybrid_selection_mode} "
                f"(low={args.hybrid_pointwise_low_threshold:.6f}, "
                f"high={args.hybrid_pointwise_high_threshold:.6f}; top_k ignored)"
            )
        elif args.hybrid_selection_mode == "random_sample":
            print(
                "Hybrid selection mode: "
                f"{args.hybrid_selection_mode} "
                f"(target_fraction={float(args.hybrid_validation_target_fraction):.3f}; "
                "top_k ignored, validation calibration skipped)"
            )
        elif args.hybrid_selection_mode == "learned_router_top_fraction":
            print(
                "Hybrid selection mode: "
                f"{args.hybrid_selection_mode} "
                f"(target_fraction={float(args.hybrid_validation_target_fraction):.3f}, "
                f"checkpoint={args.hybrid_router_checkpoint}; top_k ignored)"
            )
        elif args.hybrid_selection_mode in {
            "tabicl_router",
            "validation_fitted_tabicl_router",
        }:
            router_support_positives, _ = (
                _resolve_hybrid_alignment_calibration_sizes(args)
            )
            context_sampling = getattr(
                args, "tabicl_context_sampling", "most_recent"
            )
            print(
                "Hybrid selection mode: "
                f"{args.hybrid_selection_mode} "
                f"(target_fraction="
                f"{float(args.hybrid_validation_target_fraction):.3f}, "
                f"train positives={router_support_positives}, "
                f"context_sampling={context_sampling}, "
                "automatic grouped-OOF TabICL fit; top_k ignored)"
            )
        elif args.hybrid_selection_mode in {
            "validation_sampled_uncertainty_band",
            "validation_sampled_gmm_overlap_band",
        }:
            calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
            print(
                "Hybrid selection mode: "
                f"{args.hybrid_selection_mode} "
                f"(validation positives={calibration_num_samples}, "
                f"negative_ratio={calibration_negative_ratio}, "
                f"target_fraction={float(args.hybrid_validation_target_fraction):.3f}; top_k ignored)"
            )
        else:
            print(
                "Hybrid selection mode: "
                f"{args.hybrid_selection_mode} "
                f"(requested_top_k={args.hybrid_uncertain_topk_queries} per DTGB batch)"
            )
        if int(getattr(args, "hybrid_debug_max_routed_samples", 0)) > 0:
            print(
                "Hybrid debug routed-sample cap: "
                f"{int(args.hybrid_debug_max_routed_samples)}"
            )
        if args.hybrid_score_alignment_mode == "tabicl":
            context_sampling = getattr(
                args, "tabicl_context_sampling", "most_recent"
            )
            print(
                "Hybrid score alignment: tabicl "
                f"(fit_scope=train_{context_sampling}, full-score fusion, "
                f"variant={args.tabicl_alignment_variant}, "
                f"context_sampling={context_sampling})"
            )
        else:
            print(
                "Hybrid score alignment: "
                f"{args.hybrid_score_alignment_mode} "
                f"(fit_scope={getattr(args, 'hybrid_score_alignment_fit_scope', 'validation_all')})"
            )
        print(f"Hybrid backbone score space: {getattr(args, 'hybrid_backbone_score_space', 'minmax')}")
        if args.hybrid_score_alignment_mode != "off":
            alignment_num_samples, alignment_negative_ratio = _resolve_hybrid_alignment_calibration_sizes(args)
            calibration_split = (
                f"train/{getattr(args, 'tabicl_context_sampling', 'most_recent')}"
                if args.hybrid_score_alignment_mode == "tabicl"
                else "validation"
            )
            print(
                "Hybrid alignment calibration slice: "
                f"{calibration_split} positives={alignment_num_samples}, "
                f"negative_ratio={alignment_negative_ratio}"
            )
    if args.include_overall_structural_signal:
        print(
            "Overall structural signal: "
            "extra three-way prior bucket (High->leans positive, Low->leans negative, Modest->neutral) "
            "used only in prior-signal control flow, not for the selected KEY SIGNALS "
            f"(low<{args.overall_signal_low_threshold:.6f}, "
            f"high>={args.overall_signal_high_threshold:.6f})"
        )
    if use_raw_key_signals:
        print("Prompt ablation mode: raw numeric key signals enabled")
    if use_percentile_key_signals:
        print("Prompt ablation mode: percentile key signals enabled")
    if args.no_cot_output_0_100:
        if args.no_cot:
            print("No-CoT output mode: scalar 0-100")
            if args.no_cot_binary_score_mode != "sample_logprob":
                print(
                    "WARNING: --no_cot_binary_score_mode has no effect when "
                    "--no_cot_output_0_100 is enabled."
                )
        else:
            print("WARNING: --no_cot_output_0_100 is set but CoT mode is active; flag has no effect.")
    elif args.no_cot:
        print(f"No-CoT binary scoring: {args.no_cot_binary_score_mode}")
    elif args.no_cot_binary_score_mode != "sample_logprob":
        print("WARNING: --no_cot_binary_score_mode is set but CoT mode is active; flag has no effect.")
    if args.enable_history_compaction_llm_only:
        print(
            "History compaction mode: local-vllm-only "
            f"(max_tokens={int(args.history_compaction_max_tokens)})"
        )
    if few_shot_examples_count > 0:
        print(
            "Few-shot prompt mode: "
            f"summary-only ({few_shot_examples_count} examples, path={args.few_shot_path})"
        )


def build_output_data(
    args,
    eval_splits,
    aggregated_metrics_by_split,
    all_trial_results,
    validation_sampled_calibration_by_trial=None,
):
    calibration_payload = validation_sampled_calibration_by_trial or []
    if args.rrf_only:
        backend = "rrf_only"
    elif args.use_openai_api:
        backend = "openai"
    elif args.use_transformers:
        backend = "transformers"
    else:
        backend = "vllm"

    if len(eval_splits) == 1:
        split_name = eval_splits[0]
        return {
            "model": args.model_path,
            "backend": backend,
            "openai_model": args.openai_model if args.use_openai_api else None,
            "dataset": args.dataset_name,
            "entity_name_mode": args.entity_name_mode,
            "setting": split_name,
            "num_samples_per_trial": args.num_samples,
            "dtgb_eval_batch_size": args.dtgb_eval_batch_size,
            "num_trials": args.num_trials,
            "base_seed": args.seed,
            "hybrid_uncertain_topk_queries": int(args.hybrid_uncertain_topk_queries),
            "hybrid_backbone": getattr(args, "hybrid_backbone", "rrf"),
            "semantic_mlp_checkpoint": getattr(args, "semantic_mlp_checkpoint", None),
            "semantic_mlp_device": (
                "cuda" if getattr(args, "hybrid_backbone", "rrf") == "semantic_mlp" else None
            ),
            "hybrid_selection_mode": args.hybrid_selection_mode,
            "hybrid_router_checkpoint": getattr(args, "hybrid_router_checkpoint", None),
            "tabicl_router_python": getattr(args, "tabicl_router_python", None),
            "tabicl_router_device": getattr(args, "tabicl_router_device", None),
            "tabicl_router_cuda_visible_devices": getattr(
                args, "tabicl_router_cuda_visible_devices", None
            ),
            "tabicl_router_artifact_dir": getattr(
                args, "tabicl_router_artifact_dir", None
            ),
            "tabicl_router_folds": int(getattr(args, "tabicl_router_folds", 0)),
            "tabicl_router_n_estimators": int(
                getattr(args, "tabicl_router_n_estimators", 0)
            ),
            "tabicl_router_batch_size": int(
                getattr(args, "tabicl_router_batch_size", 0)
            ),
            "tabicl_alignment_variant": getattr(
                args,
                "tabicl_alignment_variant",
                "score_fusion_no_heuristics",
            ),
            "tabicl_context_sampling": getattr(
                args,
                "tabicl_context_sampling",
                "most_recent",
            ),
            "tabicl_context_pool_positive_queries": int(
                getattr(args, "tabicl_context_pool_positive_queries", 0)
            ),
            "hybrid_merge_method": args.hybrid_merge_method,
            "hybrid_score_alignment_mode": args.hybrid_score_alignment_mode,
            "hybrid_score_alignment_fit_scope": getattr(
                args,
                "hybrid_score_alignment_fit_scope",
                "validation_all",
            ),
            "hybrid_backbone_score_space": getattr(
                args,
                "hybrid_backbone_score_space",
                "minmax",
            ),
            "hybrid_alignment_num_samples": int(getattr(args, "hybrid_alignment_num_samples", 0)),
            "hybrid_alignment_negative_ratio": getattr(
                args,
                "hybrid_alignment_negative_ratio",
                None,
            ),
            "hybrid_fusion_alpha": float(args.hybrid_fusion_alpha),
            "hybrid_pointwise_low_threshold": float(args.hybrid_pointwise_low_threshold),
            "hybrid_pointwise_high_threshold": float(args.hybrid_pointwise_high_threshold),
            "hybrid_validation_target_fraction": float(args.hybrid_validation_target_fraction),
            "hybrid_debug_max_routed_samples": int(
                getattr(args, "hybrid_debug_max_routed_samples", 0)
            ),
            "expert_prediction_source": getattr(args, "expert_prediction_source", "rrf"),
            "key_signal_mode": args.key_signal_mode,
            "key_signal_reference": args.key_signal_reference,
            "key_signal_fields": list(args.key_signal_fields),
            "few_shot_path": args.few_shot_path,
            "few_shot_max_examples": args.few_shot_max_examples,
            "no_cot_binary_score_mode": args.no_cot_binary_score_mode,
            "include_edge_type": bool(args.include_edge_type),
            "include_edge_type_except_target": bool(
                getattr(args, "include_edge_type_except_target", False)
            ),
            "history_preserve_recent_k": int(args.history_preserve_recent_k),
            "sample_creation_profile": bool(args.sample_creation_profile),
            "sample_creation_profile_sort": args.sample_creation_profile_sort,
            "sample_creation_profile_top_n": int(args.sample_creation_profile_top_n),
            "sample_creation_profile_output": args.sample_creation_profile_output,
            "enable_history_compaction_llm_only": bool(args.enable_history_compaction_llm_only),
            "history_compaction_max_tokens": int(args.history_compaction_max_tokens),
            "data_parallel_sync_dir": args.data_parallel_sync_dir,
            "keep_data_parallel_sync": bool(args.keep_data_parallel_sync),
            "expert_prediction_mode": args.expert_prediction_mode,
            "validation_calibration_num_samples": int(args.validation_calibration_num_samples),
            "validation_calibration_negative_ratio": args.validation_calibration_negative_ratio,
            "validation_calibration_low_neg_quantile": float(args.validation_calibration_low_neg_quantile),
            "validation_calibration_high_pos_quantile": float(args.validation_calibration_high_pos_quantile),
            "validation_sampled_calibration_by_trial": calibration_payload,
            "aggregated_metrics": aggregated_metrics_by_split.get(split_name, {}),
            "trials": all_trial_results.get(split_name, []),
        }

    return {
        "model": args.model_path,
        "backend": backend,
        "openai_model": args.openai_model if args.use_openai_api else None,
        "dataset": args.dataset_name,
        "entity_name_mode": args.entity_name_mode,
        "setting": "both",
        "num_samples_per_trial": args.num_samples,
        "dtgb_eval_batch_size": args.dtgb_eval_batch_size,
        "num_trials": args.num_trials,
        "base_seed": args.seed,
        "hybrid_uncertain_topk_queries": int(args.hybrid_uncertain_topk_queries),
        "hybrid_backbone": getattr(args, "hybrid_backbone", "rrf"),
        "semantic_mlp_checkpoint": getattr(args, "semantic_mlp_checkpoint", None),
        "semantic_mlp_device": (
            "cuda" if getattr(args, "hybrid_backbone", "rrf") == "semantic_mlp" else None
        ),
        "hybrid_selection_mode": args.hybrid_selection_mode,
        "hybrid_router_checkpoint": getattr(args, "hybrid_router_checkpoint", None),
        "tabicl_router_python": getattr(args, "tabicl_router_python", None),
        "tabicl_router_device": getattr(args, "tabicl_router_device", None),
        "tabicl_router_cuda_visible_devices": getattr(
            args, "tabicl_router_cuda_visible_devices", None
        ),
        "tabicl_router_artifact_dir": getattr(
            args, "tabicl_router_artifact_dir", None
        ),
        "tabicl_router_folds": int(getattr(args, "tabicl_router_folds", 0)),
        "tabicl_router_n_estimators": int(
            getattr(args, "tabicl_router_n_estimators", 0)
        ),
        "tabicl_router_batch_size": int(
            getattr(args, "tabicl_router_batch_size", 0)
        ),
        "tabicl_alignment_variant": getattr(
            args,
            "tabicl_alignment_variant",
            "score_fusion_no_heuristics",
        ),
        "tabicl_context_sampling": getattr(
            args,
            "tabicl_context_sampling",
            "most_recent",
        ),
        "tabicl_context_pool_positive_queries": int(
            getattr(args, "tabicl_context_pool_positive_queries", 0)
        ),
        "hybrid_merge_method": args.hybrid_merge_method,
        "hybrid_score_alignment_mode": args.hybrid_score_alignment_mode,
        "hybrid_score_alignment_fit_scope": getattr(
            args,
            "hybrid_score_alignment_fit_scope",
            "validation_all",
        ),
        "hybrid_backbone_score_space": getattr(
            args,
            "hybrid_backbone_score_space",
            "minmax",
        ),
        "hybrid_alignment_num_samples": int(getattr(args, "hybrid_alignment_num_samples", 0)),
        "hybrid_alignment_negative_ratio": getattr(
            args,
            "hybrid_alignment_negative_ratio",
            None,
        ),
        "hybrid_fusion_alpha": float(args.hybrid_fusion_alpha),
        "hybrid_pointwise_low_threshold": float(args.hybrid_pointwise_low_threshold),
        "hybrid_pointwise_high_threshold": float(args.hybrid_pointwise_high_threshold),
        "hybrid_validation_target_fraction": float(args.hybrid_validation_target_fraction),
        "hybrid_debug_max_routed_samples": int(
            getattr(args, "hybrid_debug_max_routed_samples", 0)
        ),
        "expert_prediction_source": getattr(args, "expert_prediction_source", "rrf"),
        "key_signal_mode": args.key_signal_mode,
        "key_signal_reference": args.key_signal_reference,
        "key_signal_fields": list(args.key_signal_fields),
        "few_shot_path": args.few_shot_path,
        "few_shot_max_examples": args.few_shot_max_examples,
        "no_cot_binary_score_mode": args.no_cot_binary_score_mode,
        "include_edge_type": bool(args.include_edge_type),
        "include_edge_type_except_target": bool(
            getattr(args, "include_edge_type_except_target", False)
        ),
        "history_preserve_recent_k": int(args.history_preserve_recent_k),
        "sample_creation_profile": bool(args.sample_creation_profile),
        "sample_creation_profile_sort": args.sample_creation_profile_sort,
        "sample_creation_profile_top_n": int(args.sample_creation_profile_top_n),
        "sample_creation_profile_output": args.sample_creation_profile_output,
        "enable_history_compaction_llm_only": bool(args.enable_history_compaction_llm_only),
        "history_compaction_max_tokens": int(args.history_compaction_max_tokens),
        "data_parallel_sync_dir": args.data_parallel_sync_dir,
        "keep_data_parallel_sync": bool(args.keep_data_parallel_sync),
        "expert_prediction_mode": args.expert_prediction_mode,
        "validation_calibration_num_samples": int(args.validation_calibration_num_samples),
        "validation_calibration_negative_ratio": args.validation_calibration_negative_ratio,
        "validation_calibration_low_neg_quantile": float(args.validation_calibration_low_neg_quantile),
        "validation_calibration_high_pos_quantile": float(args.validation_calibration_high_pos_quantile),
        "validation_sampled_calibration_by_trial": calibration_payload,
        "split_results": {
            split_name: {
                "aggregated_metrics": aggregated_metrics_by_split.get(split_name, {}),
                "trials": all_trial_results.get(split_name, []),
            }
            for split_name in eval_splits
        },
    }
