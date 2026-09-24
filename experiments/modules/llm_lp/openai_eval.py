"""
OpenAI chat-completions backend for link-prediction evaluation.
Keeps API-based inference isolated from local vLLM/Transformers paths.
"""

import asyncio
import math
import os
import time

from tqdm import tqdm

from experiments.modules.llm_lp.eval_helpers import extract_score_0_100_with_meta
from experiments.modules.llm_lp.eval import (
    _build_prediction_debug_fields,
    _finalize_prediction_results,
)
from experiments.modules.llm_lp.prompt_template import (
    DEFAULT_KEY_SIGNAL_FIELDS,
    create_prompt,
)


def _openai_log(message):
    print(f"[OpenAI] {message}")


def _openai_warn(message):
    print(f"[OpenAI][WARN] {message}")


_AUTO_PIN_OPENROUTER_LOGPROB_PROFILES = {
    "deepseek/deepseek-chat-v3.1": {
        "provider": {
            "order": ["fireworks"],
            "require_parameters": True,
            "allow_fallbacks": False,
        },
        "disable_reasoning": True,
    },
    "openai/gpt-oss-120b": {
        "provider": {
            "order": ["crusoe/bf16"],
            "require_parameters": True,
            "allow_fallbacks": False,
        },
        "disable_reasoning": False,
        "include_reasoning": False,
    },
}


def build_openai_client(
    api_key=None,
    api_key_env="OPENAI_API_KEY",
    base_url=None,
    timeout_sec=120.0,
    max_retries=3,
):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. Install it in this environment first."
        ) from exc

    resolved_key = api_key or os.environ.get(str(api_key_env))
    if not resolved_key:
        raise ValueError(
            f"Missing OpenAI API key. Provide --openai_api_key or set env var {api_key_env}."
        )

    kwargs = {
        "api_key": resolved_key,
        "timeout": float(timeout_sec),
        "max_retries": int(max_retries),
    }
    if base_url:
        kwargs["base_url"] = str(base_url)
    return OpenAI(**kwargs)


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


def _build_openai_messages_for_sample(
    sample,
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
    anonymous_entity_aliases,
    natural_grouped_history,
    natural_activity_summary,
    natural_neighbor_names_only,
    natural_activity_compact_top3,
    natural_activity_top_k,
    ablate_mutual_history,
    ablate_common_neighbors,
    ablate_source_history,
    ablate_target_history,
    ablate_source_target_history,
    ablate_reasoning_guidance,
    no_cot_output_0_100,
    few_shot_examples=None,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
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

    user_content = create_prompt(
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
        itemcf_signal=key_signal_values["itemcf"],
        usercf_signal=key_signal_values["usercf"],
        global_recency_signal=key_signal_values["global_recency"],
        expert_prediction=expert_prediction_value,
        include_overall_structural_signal=include_overall_structural_signal,
        overall_structural_signal=sample.get("overall_structural_signal", "Unknown"),
        overall_structural_signal_name=overall_structural_signal_name,
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
        use_chat_template=False,
        tokenizer=None,
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
    )

    if use_cot:
        system_msg = (
            "You are a political event analyst. Analyze the history step-by-step and conclude with "
            'exactly: "Therefore, the answer is: X" (where X is an integer from 0 to 100).'
        )
    else:
        system_msg = (
            "You are a political event analyst. Output only the final answer in the format: "
            + (
                '"The answer is: X" (where X is an integer from 0 to 100).'
                if no_cot_output_0_100
                else '"The answer is: X" (where X is 0 or 1).'
            )
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


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


def _extract_message_text(response):
    if not getattr(response, "choices", None):
        return ""
    message = getattr(response.choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "".join(chunks)
    return str(content or "")


def _obj_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _maybe_add_binary_logprob_candidate(bucket, token_text, logprob_value):
    if token_text is None or logprob_value is None:
        return
    normalized = str(token_text).strip()
    if normalized not in ("0", "1"):
        return
    try:
        lp = float(logprob_value)
    except (TypeError, ValueError):
        return
    previous = bucket.get(normalized)
    if previous is None or lp > previous:
        bucket[normalized] = lp


def _extract_response_token_usage(response):
    usage = _obj_get(response, "usage")
    if usage is None:
        return None

    merged = {}
    field_aliases = (
        ("prompt_tokens", ("prompt_tokens", "input_tokens")),
        ("completion_tokens", ("completion_tokens", "output_tokens")),
        ("total_tokens", ("total_tokens",)),
    )
    for canonical_key, aliases in field_aliases:
        for key in aliases:
            value = _obj_get(usage, key)
            if value is None:
                continue
            try:
                merged[canonical_key] = int(value)
                break
            except (TypeError, ValueError):
                continue

    if (
        ("total_tokens" not in merged)
        and ("prompt_tokens" in merged)
        and ("completion_tokens" in merged)
    ):
        merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]

    return merged or None


def _merge_token_usage(accumulator, usage):
    if not isinstance(accumulator, dict):
        accumulator = {}
    if not isinstance(usage, dict):
        return accumulator

    for key, value in usage.items():
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        accumulator[key] = int(accumulator.get(key, 0) + numeric)
    return accumulator


def _extract_binary_score_from_chat_logprobs(response):
    choices = _obj_get(response, "choices")
    if not choices:
        return None, None

    first_choice = choices[0]
    logprobs = _obj_get(first_choice, "logprobs")
    content = _obj_get(logprobs, "content")
    if not content:
        return None, None

    for token_info in reversed(content):
        candidates = {}
        _maybe_add_binary_logprob_candidate(
            candidates,
            _obj_get(token_info, "token"),
            _obj_get(token_info, "logprob"),
        )
        for alt in _obj_get(token_info, "top_logprobs", []) or []:
            _maybe_add_binary_logprob_candidate(
                candidates,
                _obj_get(alt, "token"),
                _obj_get(alt, "logprob"),
            )

        if ("0" in candidates) and ("1" in candidates):
            lp0 = candidates["0"]
            lp1 = candidates["1"]
            pivot = max(lp0, lp1)
            p0 = math.exp(lp0 - pivot)
            p1 = math.exp(lp1 - pivot)
            denom = p0 + p1
            if denom > 0:
                return float(p1 / denom), "paired_0_1"

        if ("1" in candidates) and ("0" not in candidates):
            return 1.0, "one_sided_only_1"

        if ("0" in candidates) and ("1" not in candidates):
            return 0.0, "one_sided_only_0"

    return None, None


def _build_detail_row(sample, sample_id, messages, score, label, generated_text, parse_method):
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
        "prompt": messages,
        "prediction_score": float(score),
        "label": int(label),
        "generated_text": generated_text,
    }
    detail.update(_build_prediction_debug_fields(sample, score, parse_method=parse_method))
    return detail


class _FixedRateLimiter:
    """Simple async rate limiter to cap request start rate (requests/sec)."""

    def __init__(self, requests_per_sec=0.0):
        self.requests_per_sec = float(requests_per_sec or 0.0)
        self._interval_sec = (1.0 / self.requests_per_sec) if self.requests_per_sec > 0 else 0.0
        self._next_ts = 0.0
        self._lock = asyncio.Lock()

    async def wait_turn(self):
        if self._interval_sec <= 0.0:
            return

        sleep_sec = 0.0
        async with self._lock:
            now = time.monotonic()
            if self._next_ts <= now:
                self._next_ts = now + self._interval_sec
            else:
                sleep_sec = self._next_ts - now
                self._next_ts += self._interval_sec

        if sleep_sec > 0.0:
            await asyncio.sleep(sleep_sec)


def _score_openai_output(
    generated_text,
    use_cot,
    no_cot_output_0_100,
    finish_reason,
    prefer_logprob=False,
    binary_logprob_score=None,
    binary_logprob_method=None,
):
    parse_delta = {}
    score = 0.5
    parse_method_used = None

    if use_cot and finish_reason == "length":
        parse_delta["cot_clipped_by_max_tokens"] = 1

    if use_cot:
        val, parse_method = extract_score_0_100_with_meta(generated_text)
        parse_method_used = f"cot_{parse_method}"
        if val is None:
            parse_delta["cot_parse_failed_default_0_5"] = 1
            score = 0.5
        else:
            score = val / 100.0
            if parse_method == "exact_phrase":
                parse_delta["cot_exact_phrase"] = 1
            elif parse_method == "answer_phrase_fallback":
                parse_delta["cot_answer_phrase_fallback"] = 1
            elif parse_method == "last_integer_fallback":
                parse_delta["cot_last_integer_fallback"] = 1
        return score, parse_method_used, parse_delta

    if no_cot_output_0_100:
        val, parse_method = extract_score_0_100_with_meta(generated_text)
        parse_method_used = f"nocot_0_100_{parse_method}"
        if val is None:
            parse_delta["nocot_score_0_100_parse_failed_default_0_5"] = 1
            score = 0.5
        else:
            score = val / 100.0
            if parse_method == "exact_phrase":
                parse_delta["nocot_score_0_100_exact_phrase"] = 1
            elif parse_method == "answer_phrase_fallback":
                parse_delta["nocot_score_0_100_answer_phrase_fallback"] = 1
            elif parse_method == "last_integer_fallback":
                parse_delta["nocot_score_0_100_last_integer_fallback"] = 1
        return score, parse_method_used, parse_delta

    if prefer_logprob and binary_logprob_score is not None:
        score = float(binary_logprob_score)
        parse_method_used = (
            f"nocot_logprob_{binary_logprob_method}"
            if binary_logprob_method
            else "nocot_logprob"
        )
        parse_delta["nocot_logprob"] = 1
        if binary_logprob_method == "paired_0_1":
            parse_delta["nocot_logprob_paired_0_1"] = 1
        elif binary_logprob_method == "one_sided_only_1":
            parse_delta["nocot_logprob_one_sided_only_1"] = 1
        elif binary_logprob_method == "one_sided_only_0":
            parse_delta["nocot_logprob_one_sided_only_0"] = 1
        return score, parse_method_used, parse_delta

    if prefer_logprob and binary_logprob_score is None:
        parse_delta["nocot_logprob_unavailable_fallback_text"] = 1

    text_score, text_method = _extract_binary_text_score(generated_text)
    if text_score is not None:
        score = text_score
        parse_method_used = f"nocot_{text_method}"
        if text_method == "text_answer_phrase":
            parse_delta["nocot_text_answer_phrase"] = 1
        else:
            parse_delta["nocot_text_suffix"] = 1
    else:
        score = 0.0
        parse_method_used = "nocot_parse_failed_default_0"
        parse_delta["nocot_parse_failed_default_0"] = 1
    return score, parse_method_used, parse_delta


async def _openai_chat_completion_create(
    client,
    request_kwargs,
    reasoning_effort,
):
    try:
        return await asyncio.to_thread(client.chat.completions.create, **request_kwargs)
    except Exception as exc:
        exc_text = str(exc).lower()
        if reasoning_effort and (
            ("reasoning_effort" in exc_text)
            or ("no endpoints found" in exc_text and "requested parameters" in exc_text)
        ):
            _openai_warn("Model endpoint rejected reasoning_effort; retrying request without it.")
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("reasoning_effort", None)
            return await asyncio.to_thread(client.chat.completions.create, **retry_kwargs)
        raise


async def _evaluate_openai_chat_async(
    client,
    model_name,
    samples,
    entity_map,
    relation_map,
    summary_map,
    summary_mode,
    summary_max_chars,
    history_window,
    save_details,
    use_cot,
    include_key_signals,
    include_expert_prediction,
    include_overall_structural_signal,
    overall_structural_signal_name,
    use_raw_key_signals,
    use_percentile_key_signals,
    include_edge_type,
    cot_max_tokens,
    no_cot_max_tokens,
    mutual_timestamps_only,
    mutual_timestamps_dedup,
    mutual_summary_count_recency,
    common_neighbors_names_only,
    compact_common_neighbors_top_k,
    compact_common_neighbors_novel_only,
    history_table_aliases,
    anonymous_entity_aliases,
    natural_grouped_history,
    natural_activity_summary,
    natural_neighbor_names_only,
    natural_activity_compact_top3,
    natural_activity_top_k,
    ablate_mutual_history,
    ablate_common_neighbors,
    ablate_source_history,
    ablate_target_history,
    ablate_source_target_history,
    ablate_reasoning_guidance,
    no_cot_output_0_100,
    no_cot_binary_score_mode,
    reasoning_effort,
    temperature,
    top_p,
    openai_concurrency,
    openai_max_requests_per_sec,
    few_shot_examples=None,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
):
    semaphore = asyncio.Semaphore(max(1, int(openai_concurrency)))
    rate_limiter = _FixedRateLimiter(openai_max_requests_per_sec)
    prefer_logprob = (
        (not use_cot)
        and (not no_cot_output_0_100)
        and (no_cot_binary_score_mode == "sample_logprob")
    )
    warning_state = {
        "logprob_unsupported": False,
        "logprob_unusable": False,
        "logprob_one_sided_heuristic": False,
    }
    warning_lock = asyncio.Lock()
    results = [None] * len(samples)
    progress = tqdm(total=len(samples), desc="[OpenAI] Evaluating")

    async def _warn_once(flag, message):
        should_warn = False
        async with warning_lock:
            if not warning_state[flag]:
                warning_state[flag] = True
                should_warn = True
        if should_warn:
            _openai_warn(message)

    async def _run_one(idx, sample):
        messages = _build_openai_messages_for_sample(
            sample=sample,
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

        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(cot_max_tokens if use_cot else no_cot_max_tokens),
        }
        model_name_normalized = str(model_name).strip()
        auto_logprob_profile = (
            _AUTO_PIN_OPENROUTER_LOGPROB_PROFILES.get(model_name_normalized)
            if prefer_logprob
            else None
        )
        if auto_logprob_profile:
            extra_body = request_kwargs.get("extra_body") or {}
            extra_body["provider"] = dict(auto_logprob_profile["provider"])
            # Keep no-CoT binary logprob mode deterministic and free of extra reasoning tokens.
            if auto_logprob_profile.get("disable_reasoning"):
                extra_body["reasoning"] = {"enabled": False}
            if auto_logprob_profile.get("include_reasoning") is not None:
                extra_body["include_reasoning"] = bool(auto_logprob_profile["include_reasoning"])
            request_kwargs["extra_body"] = extra_body
        normalized_reasoning_effort = str(reasoning_effort or "").strip().lower()
        if (
            normalized_reasoning_effort
            and normalized_reasoning_effort != "none"
            and not auto_logprob_profile
        ):
            request_kwargs["reasoning_effort"] = normalized_reasoning_effort
        logprob_enabled_for_request = False
        if prefer_logprob:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = 5
            logprob_enabled_for_request = True

        await rate_limiter.wait_turn()
        async with semaphore:
            try:
                response = await _openai_chat_completion_create(
                    client=client,
                    request_kwargs=request_kwargs,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:
                if logprob_enabled_for_request and "logprob" in str(exc).lower():
                    await _warn_once(
                        "logprob_unsupported",
                        "Model endpoint rejected logprobs; falling back to text parsing.",
                    )
                    request_kwargs.pop("logprobs", None)
                    request_kwargs.pop("top_logprobs", None)
                    logprob_enabled_for_request = False
                    response = await _openai_chat_completion_create(
                        client=client,
                        request_kwargs=request_kwargs,
                        reasoning_effort=reasoning_effort,
                    )
                else:
                    raise

        generated_text = _extract_message_text(response)
        finish_reason = None
        if getattr(response, "choices", None):
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        binary_logprob_score = None
        binary_logprob_method = None
        if prefer_logprob and logprob_enabled_for_request:
            binary_logprob_score, binary_logprob_method = _extract_binary_score_from_chat_logprobs(
                response
            )
            if binary_logprob_score is None:
                await _warn_once(
                    "logprob_unusable",
                    "Response did not include usable 0/1 logprobs; using text parsing fallback.",
                )
            elif binary_logprob_method in ("one_sided_only_1", "one_sided_only_0"):
                await _warn_once(
                    "logprob_one_sided_heuristic",
                    (
                        "Using one-sided 0/1 logprob heuristic because only one binary token "
                        "was present in top_logprobs."
                    ),
                )

        score, parse_method_used, parse_delta = _score_openai_output(
            generated_text=generated_text,
            use_cot=use_cot,
            no_cot_output_0_100=no_cot_output_0_100,
            finish_reason=finish_reason,
            prefer_logprob=prefer_logprob,
            binary_logprob_score=binary_logprob_score,
            binary_logprob_method=binary_logprob_method,
        )

        detail = None
        if save_details:
            detail = _build_detail_row(
                sample=sample,
                sample_id=idx,
                messages=messages,
                score=score,
                label=sample["label"],
                generated_text=generated_text,
                parse_method=parse_method_used,
            )

        return {
            "idx": idx,
            "score": score,
            "label": sample["label"],
            "detail": detail,
            "parse_delta": parse_delta,
            "token_usage": _extract_response_token_usage(response),
        }

    tasks = [asyncio.create_task(_run_one(idx, sample)) for idx, sample in enumerate(samples)]
    try:
        for task in asyncio.as_completed(tasks):
            item = await task
            results[item["idx"]] = item
            progress.update(1)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()

    return results


def evaluate_openai_chat(
    client,
    model_name,
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
    cot_max_tokens=2048,
    no_cot_max_tokens=32,
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
    reasoning_effort="none",
    temperature=0.0,
    top_p=1.0,
    openai_concurrency=1,
    openai_max_requests_per_sec=0.0,
    few_shot_examples=None,
    key_signal_fields=DEFAULT_KEY_SIGNAL_FIELDS,
):
    _openai_log(f"Starting evaluation ({eval_split.capitalize()}, OpenAI Chat) [CoT={use_cot}]")

    if (not use_cot) and (not no_cot_output_0_100) and no_cot_binary_score_mode != "sample_logprob":
        _openai_warn(
            "--no_cot_binary_score_mode is vLLM-specific. OpenAI backend will use text parsing."
        )

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
        "nocot_text_answer_phrase": 0,
        "nocot_text_suffix": 0,
        "nocot_parse_failed_default_0": 0,
        "nocot_logprob": 0,
        "nocot_logprob_paired_0_1": 0,
        "nocot_logprob_one_sided_only_1": 0,
        "nocot_logprob_one_sided_only_0": 0,
        "nocot_logprob_unavailable_fallback_text": 0,
        "nocot_score_0_100_exact_phrase": 0,
        "nocot_score_0_100_answer_phrase_fallback": 0,
        "nocot_score_0_100_last_integer_fallback": 0,
        "nocot_score_0_100_parse_failed_default_0_5": 0,
    }
    token_usage = {}

    openai_concurrency = max(1, int(openai_concurrency))
    openai_max_requests_per_sec = float(openai_max_requests_per_sec or 0.0)
    auto_logprob_profile = (
        (not use_cot)
        and (not no_cot_output_0_100)
        and (no_cot_binary_score_mode == "sample_logprob")
        and _AUTO_PIN_OPENROUTER_LOGPROB_PROFILES.get(str(model_name).strip())
    )
    if openai_concurrency > 1:
        _openai_log(f"Async concurrency enabled: {openai_concurrency} in-flight requests.")
    if openai_max_requests_per_sec > 0.0:
        _openai_log(f"Client-side rate limit: {openai_max_requests_per_sec:g} requests/sec.")
    if auto_logprob_profile:
        _openai_log(
            "Auto-pinning provider route for logprob mode "
            f"(model={str(model_name).strip()})."
        )

    ordered = asyncio.run(
        _evaluate_openai_chat_async(
            client=client,
            model_name=model_name,
            samples=samples,
            entity_map=entity_map,
            relation_map=relation_map,
            summary_map=summary_map,
            summary_mode=summary_mode,
            summary_max_chars=summary_max_chars,
            history_window=history_window,
            save_details=save_details,
            use_cot=use_cot,
            include_key_signals=include_key_signals,
            include_expert_prediction=include_expert_prediction,
            include_overall_structural_signal=include_overall_structural_signal,
            overall_structural_signal_name=overall_structural_signal_name,
            use_raw_key_signals=use_raw_key_signals,
            use_percentile_key_signals=use_percentile_key_signals,
            include_edge_type=include_edge_type,
            cot_max_tokens=cot_max_tokens,
            no_cot_max_tokens=no_cot_max_tokens,
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
            no_cot_binary_score_mode=no_cot_binary_score_mode,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            top_p=top_p,
            openai_concurrency=openai_concurrency,
            openai_max_requests_per_sec=openai_max_requests_per_sec,
            few_shot_examples=few_shot_examples,
            key_signal_fields=key_signal_fields,
        )
    )

    for item in ordered:
        if item is None:
            continue
        predictions.append(float(item["score"]))
        labels.append(item["label"])
        for key, value in item["parse_delta"].items():
            parse_stats[key] += int(value)
        if save_details and item["detail"] is not None:
            detailed_results.append(item["detail"])
        if item.get("token_usage"):
            token_usage = _merge_token_usage(token_usage, item["token_usage"])

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
    return metrics


__all__ = [
    "build_openai_client",
    "evaluate_openai_chat",
]
