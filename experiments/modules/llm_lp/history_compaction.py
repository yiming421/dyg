"""
Minimal history-compaction helpers for two-stage link prediction.

This module only supports local chat-style LLM compaction in the initial
implementation. It does not include OpenAI/API plumbing or endpoint-profile
summary handling.
"""

from tqdm import tqdm

from experiments.modules.llm_lp.prompt_embeddings import (
    discard_vllm_hidden_state_files,
)

from experiments.modules.llm_lp.prompt_template import (
    format_common_neighbors,
    format_entity_timestamp_series,
    format_historical_events,
    format_mutual_timestamps_only,
)

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None


HISTORY_COMPACTION_SYSTEM_PROMPT = (
    "You compress a temporal history window into a short factual memory block for a "
    "later prediction model. Do not predict whether the event will happen."
)


HISTORY_COMPACTION_USER_TEMPLATE = """
Query
Source: {source_entity}
Target: {target_entity}
Prediction Time: {prediction_time}

Raw History Window

Mutual history:
{formatted_mutual_history}

Common neighbors:
{formatted_common_neighbors}

Source history:
{formatted_source_history}

Target history:
{formatted_target_history}

Task
Summarize the history window into a short factual memory block for a later model.

Rules:
1. Do not make the final prediction.
2. Keep only evidence useful for predicting the candidate event.
3. Focus on the semantic content of the history and the notable entity interactions.
4. The summary should be concise.

Output exactly:
COMPACTED HISTORY
- Mutual history: ...
- Common neighbors: ...
- Source history: ...
- Target history: ...
""".strip()


def _history_log(message):
    print(f"[HistoryCompaction] {message}")


def _tail_window(items, window_size):
    if items is None:
        return []
    if window_size is None:
        return list(items)
    window_size = int(window_size)
    if window_size <= 0:
        return []
    return list(items[-window_size:])


def build_history_compaction_prompt_for_sample(
    sample,
    entity_map,
    relation_map,
    history_window=10,
    include_edge_type=False,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    common_neighbors_names_only=False,
):
    """
    Build a raw compaction prompt from an already-materialized sample.
    """
    source_history_entities = sample.get("source_history_entities", [])
    target_history_entities = sample.get("target_history_entities", [])
    source_history = sample.get("source_history", [])
    target_history = sample.get("target_history", [])
    mutual_history = sample.get("mutual_history", [])
    common_neighbors = sample.get("common_neighbors", [])

    if source_history_entities:
        formatted_source_history = format_entity_timestamp_series(
            source_history_entities[:history_window],
            entity_map,
        )
    else:
        formatted_source_history = format_historical_events(
            _tail_window(source_history, history_window),
            entity_map,
            relation_map,
            sample["timestamp"],
            include_edge_type=include_edge_type,
        )

    if target_history_entities:
        formatted_target_history = format_entity_timestamp_series(
            target_history_entities[:history_window],
            entity_map,
        )
    else:
        formatted_target_history = format_historical_events(
            _tail_window(target_history, history_window),
            entity_map,
            relation_map,
            sample["timestamp"],
            include_edge_type=include_edge_type,
        )

    mutual_window = _tail_window(mutual_history, history_window)
    if mutual_timestamps_only:
        formatted_mutual_history = format_mutual_timestamps_only(
            mutual_window,
            deduplicate=mutual_timestamps_dedup,
        )
    else:
        formatted_mutual_history = format_historical_events(
            mutual_window,
            entity_map,
            relation_map,
            sample["timestamp"],
            include_edge_type=include_edge_type,
        )

    formatted_common_neighbors = format_common_neighbors(
        common_neighbors[:history_window],
        entity_map,
        relation_map,
        sample["timestamp"],
        names_only=common_neighbors_names_only,
        include_edge_type=include_edge_type,
    )

    if include_edge_type:
        event_line = f"{sample['source_entity']} --[{sample['relation']}]--> {sample['target_entity']}"
    else:
        event_line = f"{sample['source_entity']} --> {sample['target_entity']}"

    return HISTORY_COMPACTION_USER_TEMPLATE.format(
        source_entity=sample["source_entity"],
        target_entity=sample["target_entity"],
        prediction_time=sample["timestamp"],
        event_line=event_line,
        formatted_mutual_history=formatted_mutual_history,
        formatted_common_neighbors=formatted_common_neighbors,
        formatted_source_history=formatted_source_history,
        formatted_target_history=formatted_target_history,
    )


def build_history_compaction_messages_for_sample(
    sample,
    entity_map,
    relation_map,
    history_window=10,
    include_edge_type=False,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    common_neighbors_names_only=False,
):
    return [
        {"role": "system", "content": HISTORY_COMPACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_history_compaction_prompt_for_sample(
                sample=sample,
                entity_map=entity_map,
                relation_map=relation_map,
                history_window=history_window,
                include_edge_type=include_edge_type,
                mutual_timestamps_only=mutual_timestamps_only,
                mutual_timestamps_dedup=mutual_timestamps_dedup,
                common_neighbors_names_only=common_neighbors_names_only,
            ),
        },
    ]


def build_history_compaction_chat_prompt_for_sample(
    sample,
    tokenizer,
    entity_map,
    relation_map,
    history_window=10,
    include_edge_type=False,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    common_neighbors_names_only=False,
):
    if tokenizer is None:
        raise ValueError("tokenizer is required for local history compaction.")

    messages = build_history_compaction_messages_for_sample(
        sample=sample,
        entity_map=entity_map,
        relation_map=relation_map,
        history_window=history_window,
        include_edge_type=include_edge_type,
        mutual_timestamps_only=mutual_timestamps_only,
        mutual_timestamps_dedup=mutual_timestamps_dedup,
        common_neighbors_names_only=common_neighbors_names_only,
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_compacted_history_text(text):
    clean = str(text or "").replace("\r\n", "\n").strip()
    if not clean:
        return ""
    if clean.startswith("```") and clean.endswith("```"):
        lines = clean.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    return clean


def attach_compacted_history(sample, compacted_history_text, *, field_name="compacted_history_text"):
    sample[field_name] = normalize_compacted_history_text(compacted_history_text)
    return sample


def compact_history_local(
    llm,
    tokenizer,
    samples,
    entity_map,
    relation_map,
    history_window=10,
    max_tokens=256,
    include_edge_type=False,
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    common_neighbors_names_only=False,
    field_name="compacted_history_text",
):
    """
    Run local batched history compaction with vLLM and attach results in-place.
    """
    if SamplingParams is None:
        raise ImportError("vLLM is required for local history compaction.")
    if not samples:
        return samples

    prompts = [
        build_history_compaction_chat_prompt_for_sample(
            sample=sample,
            tokenizer=tokenizer,
            entity_map=entity_map,
            relation_map=relation_map,
            history_window=history_window,
            include_edge_type=include_edge_type,
            mutual_timestamps_only=mutual_timestamps_only,
            mutual_timestamps_dedup=mutual_timestamps_dedup,
            common_neighbors_names_only=common_neighbors_names_only,
        )
        for sample in tqdm(samples, desc="[HistoryCompaction] Formatting")
    ]

    _history_log(f"Running local history compaction on {len(samples)} samples.")
    outputs = llm.generate(
        prompts,
        SamplingParams(
            max_tokens=int(max_tokens),
            temperature=0.0,
        ),
        tokenization_kwargs={"add_special_tokens": False},
    )

    # A capture-enabled vLLM engine produces connector scratch output for every
    # generate() call. History compaction does not use those vectors.
    discard_vllm_hidden_state_files(outputs)

    for sample, output in zip(samples, outputs):
        generated_text = output.outputs[0].text if output.outputs else ""
        attach_compacted_history(
            sample,
            generated_text,
            field_name=field_name,
        )

    return samples


__all__ = [
    "HISTORY_COMPACTION_SYSTEM_PROMPT",
    "HISTORY_COMPACTION_USER_TEMPLATE",
    "attach_compacted_history",
    "build_history_compaction_chat_prompt_for_sample",
    "build_history_compaction_messages_for_sample",
    "build_history_compaction_prompt_for_sample",
    "compact_history_local",
    "normalize_compacted_history_text",
]
