"""
Structured prompt templates for link prediction.
CoT: 0-100 score; No-CoT: binary 0/1
"""

import re

DEFAULT_KEY_SIGNAL_FIELDS = (
    "target_popularity",
    "past_interactions",
    "recency",
    "common_neighbor",
)
SUPPORTED_KEY_SIGNAL_FIELDS = DEFAULT_KEY_SIGNAL_FIELDS + (
    "recent_degree",
    "global_recency",
    "itemcf",
    "usercf",
)


def normalize_key_signal_fields(key_signal_fields=None):
    if key_signal_fields is None:
        return DEFAULT_KEY_SIGNAL_FIELDS

    if isinstance(key_signal_fields, str):
        raw_items = [item.strip().lower() for item in key_signal_fields.split(",")]
    else:
        raw_items = []
        for item in key_signal_fields:
            if item is None:
                continue
            raw_items.append(str(item).strip().lower())

    normalized = []
    seen = set()
    for item in raw_items:
        if not item or item in seen:
            continue
        if item not in SUPPORTED_KEY_SIGNAL_FIELDS:
            raise ValueError(
                f"Unsupported key signal field '{item}'. "
                f"Use a subset of {list(SUPPORTED_KEY_SIGNAL_FIELDS)}."
            )
        normalized.append(item)
        seen.add(item)

    if not normalized:
        raise ValueError(
            "At least one key signal field must be selected. "
            f"Supported values: {list(SUPPORTED_KEY_SIGNAL_FIELDS)}."
        )
    return tuple(normalized)

LINK_PREDICTION_TEMPLATE = """
{task_intro}

ENTITY INFORMATION
Source: {source_entity_line}
Target: {target_entity_line}
Prediction Time: {prediction_time}

{entity_profiles_section}{expert_prediction_section}{key_signals_section}

{history_context_section}

{graph_prompt_section}{few_shot_section}

PREDICTION TASK
Question: {prediction_question}
{event_type_section}{task_item_label}: {event_line}
Prediction Time: {prediction_time}


{analysis_instructions}

0 = {negative_outcome_text} (definitely no)
100 = {positive_outcome_text} (definitely yes)

Response:
""".strip()

LINK_PREDICTION_TEMPLATE_NO_COT = """
{task_intro}

ENTITY INFORMATION
Source: {source_entity_line}
Target: {target_entity_line}
Prediction Time: {prediction_time}

{entity_profiles_section}{expert_prediction_section}{key_signals_section}

{history_context_section}

{graph_prompt_section}{few_shot_section}

PREDICTION TASK
Question: {prediction_question}
{event_type_section}{task_item_label}: {event_line}
Prediction Time: {prediction_time}

Instruction:
{direct_instructions}

{no_cot_score_legend}

Response:
""".strip()


def format_timedelta(delta_ts):
    """
    Format time difference into years/months/days
    Assuming ts is in DAYS.
    """
    if delta_ts < 0:
        return "future"
    
    if delta_ts == 0:
        return "today"

    years = delta_ts // 365
    remaining_days = delta_ts % 365
    months = remaining_days // 30
    days = remaining_days % 30
    
    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years!=1 else ''}")
    if months > 0:
        parts.append(f"{months} month{'s' if months!=1 else ''}")
    if days > 0:
        parts.append(f"{days} day{'s' if days!=1 else ''}")
        
    if not parts:
        return "0 days"
        
    return ", ".join(parts)


# Formatting functions
def _history_entity_label(entity_id, entity_map, source_id=None, target_id=None, use_aliases=False):
    entity_id = int(entity_id)
    if use_aliases:
        if source_id is not None and entity_id == int(source_id):
            return "Source"
        if target_id is not None and entity_id == int(target_id):
            return "Target"
    return entity_map.get(entity_id, f"entity_{entity_id}")


def build_prompt_local_entity_aliases(
    *,
    source_id,
    target_id,
    source_history,
    target_history,
    mutual_history,
    source_history_entities,
    target_history_entities,
    common_neighbors,
    history_window,
    include_mutual_history=True,
    include_common_neighbors=True,
    include_source_history=True,
    include_target_history=True,
):
    """Build prompt-local anonymous entity labels in visible-order.

    Query endpoints are always ``Source`` and ``Target``. Every other entity is
    assigned ``N1``, ``N2``, ... on first appearance in the rendered history.
    The mapping is rebuilt for every prompt, so aliases carry no identity across
    examples.
    """
    source_id = int(source_id)
    target_id = int(target_id)
    aliases = {source_id: "Source"}
    if target_id != source_id:
        aliases[target_id] = "Target"
    next_alias = 1

    def add_entity(entity_id):
        nonlocal next_alias
        entity_id = int(entity_id)
        if entity_id not in aliases:
            aliases[entity_id] = f"N{next_alias}"
            next_alias += 1

    def add_events(events):
        for u, _r, i, _ts in events:
            add_entity(u)
            add_entity(i)

    window = max(0, int(history_window))
    if include_mutual_history:
        add_events(mutual_history[-window:] if window else [])
    if include_common_neighbors:
        for mid, _src_evt, _tar_evt in (common_neighbors or [])[:window]:
            add_entity(mid)
    if include_source_history:
        if source_history_entities:
            for entity_id, _timestamps in source_history_entities[:window]:
                add_entity(entity_id)
        else:
            add_events(source_history[-window:] if window else [])
    if include_target_history:
        if target_history_entities:
            for entity_id, _timestamps in target_history_entities[:window]:
                add_entity(entity_id)
        else:
            add_events(target_history[-window:] if window else [])
    return aliases


def format_historical_events(
    events_list,
    entity_map,
    relation_map,
    prediction_time,
    include_edge_type=False,
    source_id=None,
    target_id=None,
    use_aliases=False,
):
    """
    Format historical events into readable list

    Args:
        events_list: List of tuples (u, r, i, ts)
        entity_map: Dict mapping entity_id -> entity_name
        relation_map: Dict mapping relation_id -> relation_text
        prediction_time: The timestamp of the prediction target (to calculate delta)

    Returns:
        Formatted string
    """
    if not events_list:
        return "- No historical events available"

    formatted = []
    for u, r, i, ts in events_list:
        source = _history_entity_label(
            u,
            entity_map,
            source_id=source_id,
            target_id=target_id,
            use_aliases=use_aliases,
        )
        target = _history_entity_label(
            i,
            entity_map,
            source_id=source_id,
            target_id=target_id,
            use_aliases=use_aliases,
        )
        if include_edge_type:
            relation = relation_map.get(r, f"relation_{r}")
            formatted.append(f"- t={ts}: {source} --[{relation}]--> {target}")
        else:
            formatted.append(f"- t={ts}: {source} --> {target}")

    return "\n".join(formatted)


def format_natural_grouped_history(
    events_list,
    entity_map,
    relation_map,
    *,
    endpoint_id,
    history_role,
    include_edge_type=False,
):
    """Group history by counterpart while preserving timestamps and direction.

    The representation remains natural-language/table-like: counterpart names are
    written once and followed by every observed timestamp. Duplicate timestamps
    are intentionally retained because they encode repeated events.
    """
    if not events_list:
        return "- No historical events available"

    endpoint_id = int(endpoint_id)
    grouped_by_direction = {}
    for u, r, i, ts in events_list:
        u_id, i_id = int(u), int(i)
        if history_role == "source":
            direction = "interaction partners"
            counterpart_id = i_id if u_id == endpoint_id else u_id
        elif i_id == endpoint_id:
            direction = "incoming interactions"
            counterpart_id = u_id
        else:
            direction = "outgoing interactions"
            counterpart_id = i_id

        relation_key = int(r) if include_edge_type else None
        key = (counterpart_id, relation_key)
        grouped_by_direction.setdefault(direction, {}).setdefault(key, []).append(ts)

    lines = [
        "Each line lists an interaction partner followed by all interaction times:"
    ]
    for direction, grouped in grouped_by_direction.items():
        if history_role == "target":
            label = "Incoming interactions:" if direction == "incoming interactions" else "Outgoing interactions:"
            lines.append(label)
        for (counterpart_id, relation_key), timestamps in grouped.items():
            counterpart = entity_map.get(counterpart_id, f"entity_{counterpart_id}")
            if relation_key is not None:
                relation = relation_map.get(relation_key, f"relation_{relation_key}")
                counterpart = f"{counterpart} ({relation})"
            timestamp_text = ", ".join(str(int(ts)) for ts in timestamps)
            lines.append(f"- {counterpart}: {timestamp_text}")

    return "\n".join(lines)


def format_natural_activity_summary(
    events_list,
    entity_map,
    *,
    endpoint_id,
    history_role,
    top_frequent=5,
    top_recent=5,
):
    """Summarize endpoint activity using natural recurrence/recency statistics.

    Exact mutual history is rendered separately. For source/target activity this
    keeps totals, diversity, time span, and the union of frequent and recent
    partners, while representing the remaining long tail as aggregate counts.
    """
    if not events_list:
        return "- No historical events available"

    grouped_by_direction = _group_natural_activity_partners(
        events_list,
        endpoint_id=endpoint_id,
        history_role=history_role,
    )

    lines = []
    for direction, partners in grouped_by_direction.items():
        all_timestamps = [ts for values in partners.values() for ts in values]
        total_events = len(all_timestamps)
        total_partners = len(partners)
        lines.append(
            f"{direction}: {total_events} interactions with {total_partners} partners "
            f"from time {min(all_timestamps)} to {max(all_timestamps)}."
        )

        selected = _select_salient_activity_partners(
            partners,
            entity_map,
            top_frequent=top_frequent,
            top_recent=top_recent,
        )

        for partner_id in selected:
            timestamps = partners[partner_id]
            name = entity_map.get(partner_id, f"entity_{partner_id}")
            count = len(timestamps)
            if count == 1:
                detail = f"1 interaction at time {timestamps[0]}"
            elif min(timestamps) == max(timestamps):
                detail = f"{count} interactions at time {timestamps[0]}"
            else:
                detail = f"{count} interactions from time {min(timestamps)} to {max(timestamps)}"
            lines.append(f"- {name}: {detail}.")

        remaining = [partner_id for partner_id in partners if partner_id not in selected]
        if remaining:
            remaining_timestamps = [
                ts for partner_id in remaining for ts in partners[partner_id]
            ]
            lines.append(
                f"- Other partners: {len(remaining)} partners and "
                f"{len(remaining_timestamps)} interactions; latest at time "
                f"{max(remaining_timestamps)}."
            )

    return "\n".join(lines)


def _select_salient_activity_partners(
    partners,
    entity_map,
    *,
    top_frequent=5,
    top_recent=5,
):
    """Select the union of frequent and recent partners deterministically."""
    frequent = sorted(
        partners,
        key=lambda partner_id: (
            -len(partners[partner_id]),
            -max(partners[partner_id]),
            entity_map.get(partner_id, f"entity_{partner_id}"),
        ),
    )[: int(top_frequent)]
    recent = sorted(
        partners,
        key=lambda partner_id: (
            -max(partners[partner_id]),
            -len(partners[partner_id]),
            entity_map.get(partner_id, f"entity_{partner_id}"),
        ),
    )[: int(top_recent)]
    return list(dict.fromkeys(frequent + recent))


def _group_natural_activity_partners(events_list, *, endpoint_id, history_role):
    """Group endpoint history by direction and counterpart entity ID."""
    endpoint_id = int(endpoint_id)
    grouped_by_direction = {}
    for u, _r, i, ts in events_list:
        u_id, i_id, timestamp = int(u), int(i), int(ts)
        if history_role == "source":
            direction = "Source activity"
            counterpart_id = i_id if u_id == endpoint_id else u_id
        elif i_id == endpoint_id:
            direction = "Incoming activity"
            counterpart_id = u_id
        else:
            direction = "Outgoing activity"
            counterpart_id = i_id
        grouped_by_direction.setdefault(direction, {}).setdefault(counterpart_id, []).append(timestamp)
    return grouped_by_direction


def _compact_activity_partner_ids(
    events_list,
    entity_map,
    *,
    endpoint_id,
    history_role,
    top_k,
):
    """Return exactly the entity IDs named by the compact activity formatter."""
    selected_ids = []
    grouped_by_direction = _group_natural_activity_partners(
        events_list,
        endpoint_id=endpoint_id,
        history_role=history_role,
    )
    for partners in grouped_by_direction.values():
        selected_ids.extend(
            _select_salient_activity_partners(
                partners,
                entity_map,
                top_frequent=top_k,
                top_recent=top_k,
            )
        )
    return set(selected_ids)


def format_natural_neighbor_names_only(
    events_list,
    entity_map,
    *,
    endpoint_id,
    history_role,
    top_frequent=5,
    top_recent=5,
):
    """Render only deduplicated salient neighbor names for endpoint activity.

    Selection matches ``format_natural_activity_summary`` so this is a clean
    ablation of the numeric count, recency, span, diversity, and tail fields.
    Names are alphabetized after selection to avoid exposing their rank.
    """
    if not events_list:
        return "- No historical neighbors available"

    grouped_by_direction = _group_natural_activity_partners(
        events_list,
        endpoint_id=endpoint_id,
        history_role=history_role,
    )

    lines = []
    for direction, partners in grouped_by_direction.items():
        selected = _select_salient_activity_partners(
            partners,
            entity_map,
            top_frequent=top_frequent,
            top_recent=top_recent,
        )
        names = sorted(
            (entity_map.get(partner_id, f"entity_{partner_id}") for partner_id in selected),
            key=lambda value: str(value).lower(),
        )
        lines.append(f"{direction} neighbors: {', '.join(names)}.")

    return "\n".join(lines)


def format_natural_activity_compact(
    events_list,
    entity_map,
    *,
    endpoint_id,
    history_role,
    top_k=3,
):
    """Compact natural activity statistics with top-K frequent/recent names.

    Keeps direction-level activity, diversity, and latest time plus each
    selected partner's recurrence and latest time. It drops earliest-time
    spans, tail statistics, and the fourth/fifth frequent/recent partners.
    """
    if not events_list:
        return "- No historical events available"

    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    grouped_by_direction = _group_natural_activity_partners(
        events_list,
        endpoint_id=endpoint_id,
        history_role=history_role,
    )

    lines = []
    for direction, partners in grouped_by_direction.items():
        all_timestamps = [ts for values in partners.values() for ts in values]
        lines.append(
            f"{direction}: {len(all_timestamps)} interactions, {len(partners)} partners; "
            f"latest time {max(all_timestamps)}."
        )
        selected = _select_salient_activity_partners(
            partners,
            entity_map,
            top_frequent=top_k,
            top_recent=top_k,
        )
        details = []
        for partner_id in selected:
            timestamps = partners[partner_id]
            name = entity_map.get(partner_id, f"entity_{partner_id}")
            details.append(f"{name} ({len(timestamps)}, {max(timestamps)})")
        lines.append(
            "Key partners (interaction count, latest time): "
            + ", ".join(details)
            + "."
        )

    return "\n".join(lines)


def format_natural_activity_compact_top3(
    events_list,
    entity_map,
    *,
    endpoint_id,
    history_role,
):
    """Backward-compatible top-3 wrapper for the generic compact formatter."""
    return format_natural_activity_compact(
        events_list,
        entity_map,
        endpoint_id=endpoint_id,
        history_role=history_role,
        top_k=3,
    )


def format_entity_timestamp_series(entity_history, entity_map):
    """
    Format entity-centric history:
    [(entity_id, [ts1, ts2, ...]), ...] sorted by selection score upstream.
    """
    if not entity_history:
        return "- No historical entities available"

    lines = []
    for entity_id, ts_list in entity_history:
        name = entity_map.get(int(entity_id), f"entity_{entity_id}")
        if ts_list:
            ts_text = ", ".join([str(int(t)) for t in ts_list])
            lines.append(f"- {name}: t={ts_text}")
        else:
            lines.append(f"- {name}: t=None")
    return "\n".join(lines)


def format_mutual_timestamps_only(events_list, deduplicate=False):
    """
    Format mutual history as a compact timestamp-only sequence.
    """
    if not events_list:
        return "- No historical events available"

    ts = [int(evt[3]) for evt in events_list]
    if deduplicate:
        dedup_ts = []
        prev = None
        for t in ts:
            if t != prev:
                dedup_ts.append(t)
            prev = t
        ts = dedup_ts

    ts_text = ", ".join([str(t) for t in ts])
    return (
        "- Previous interaction timestamps (between Source and Target, oldest -> latest):\n"
        f"  t={ts_text}"
    )


def format_mutual_existence_count_recency(
    events_list,
    prediction_time,
    *,
    deduplicate=True,
):
    """Retain only the simple statistics exposed by the timestamp sequence.

    The count is computed after the same history-window truncation performed by
    ``create_prompt`` and, by default, after the same consecutive-timestamp
    deduplication used by the compact K=10 baseline.  This makes the summary a
    strict compression of the baseline prompt rather than a source of extra
    full-history or same-timestamp multiplicity information.
    """
    timestamps = [int(evt[3]) for evt in events_list]
    if deduplicate:
        deduplicated = []
        previous = None
        for timestamp in timestamps:
            if timestamp != previous:
                deduplicated.append(timestamp)
            previous = timestamp
        timestamps = deduplicated

    if not timestamps:
        return (
            "- Prior direct interaction exists: No\n"
            "- Retained distinct interaction-time count: 0\n"
            "- Time since most recent direct interaction: None"
        )

    recency = max(0, int(prediction_time) - timestamps[-1])
    return (
        "- Prior direct interaction exists: Yes\n"
        f"- Retained distinct interaction-time count: {len(timestamps)}\n"
        f"- Time since most recent direct interaction: {recency} time units"
    )


def format_common_neighbors(
    common_neighbors_info,
    entity_map,
    relation_map,
    prediction_time,
    names_only=False,
    include_edge_type=False,
):
    """
    Format common neighbors list of tuples (neighbor_id, src_evt, tar_evt)
    """
    if not common_neighbors_info:
        return "None"

    if names_only:
        names = []
        seen = set()
        for mid, _, _ in common_neighbors_info:
            mid = int(mid)
            if mid in seen:
                continue
            seen.add(mid)
            names.append(entity_map.get(mid, f"entity_{mid}"))
        return ", ".join(names) if names else "None"

    formatted = []
    for mid, src_evt, tar_evt in common_neighbors_info:
        neighbor_name = entity_map.get(mid, f"entity_{mid}")
        
        # src_evt / tar_evt = (u, r, i, ts)
        s_u, s_r, s_i, s_ts = src_evt
        t_u, t_r, t_i, t_ts = tar_evt

        if include_edge_type:
            s_rel = relation_map.get(s_r, f"relation_{s_r}")
            t_rel = relation_map.get(t_r, f"relation_{t_r}")
            if s_u == mid:
                s_str = f"{neighbor_name} --[{s_rel}]--> Source"
            else:
                s_str = f"Source --[{s_rel}]--> {neighbor_name}"
            if t_u == mid:
                t_str = f"{neighbor_name} --[{t_rel}]--> Target"
            else:
                t_str = f"Target --[{t_rel}]--> {neighbor_name}"
        else:
            if s_u == mid:
                s_str = f"{neighbor_name} --> Source"
            else:
                s_str = f"Source --> {neighbor_name}"
            if t_u == mid:
                t_str = f"{neighbor_name} --> Target"
            else:
                t_str = f"Target --> {neighbor_name}"
        formatted.append(f"- {neighbor_name}: {s_str} (t={s_ts}), {t_str} (t={t_ts})")
        
    return "\n" + "\n".join(formatted)


def format_common_neighbors_compact(
    common_neighbors_info,
    entity_map,
    relation_map,
    *,
    top_k,
    include_edge_type=False,
    selected_total=None,
    novel_only=False,
):
    """Render semantic common neighbors as a bounded natural activity summary.

    The input ordering is preserved: semantic common neighbors are already
    selected by target similarity and ordered by target-side recency upstream.
    Each displayed neighbor keeps both endpoint-relative directions and latest
    timestamps while mentioning the (potentially long) entity name only once.
    """
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("compact common-neighbor top_k must be >= 1")

    available = len(common_neighbors_info or [])
    total = available if selected_total is None else int(selected_total)
    shown = list(common_neighbors_info or [])[:top_k]
    if novel_only:
        neighbor_word = "neighbor" if total == 1 else "neighbors"
        novel_word = "neighbor" if available == 1 else "neighbors"
        lines = [
            f"{total} selected semantic {neighbor_word}; {available} {novel_word} "
            f"not already shown in endpoint histories; {len(shown)} shown."
        ]
    else:
        neighbor_word = "neighbor" if total == 1 else "neighbors"
        lines = [f"{total} semantic {neighbor_word}; {len(shown)} shown."]
    if not shown:
        return "\n".join(lines)

    if include_edge_type:
        lines.append(
            "Key neighbors (source direction/relation/time; "
            "target direction/relation/time):"
        )
    else:
        lines.append("Key neighbors (source direction/time; target direction/time):")

    details = []
    for mid, src_evt, tar_evt in shown:
        mid = int(mid)
        neighbor_name = entity_map.get(mid, f"entity_{mid}")
        s_u, s_r, _s_i, s_ts = src_evt
        t_u, t_r, _t_i, t_ts = tar_evt
        source_direction = "in" if int(s_u) == mid else "out"
        target_direction = "in" if int(t_u) == mid else "out"

        if include_edge_type:
            source_relation = relation_map.get(s_r, f"relation_{s_r}")
            target_relation = relation_map.get(t_r, f"relation_{t_r}")
            source_detail = f"{source_direction}/{source_relation}, {int(s_ts)}"
            target_detail = f"{target_direction}/{target_relation}, {int(t_ts)}"
        else:
            source_detail = f"{source_direction}, {int(s_ts)}"
            target_detail = f"{target_direction}, {int(t_ts)}"
        details.append(f"{neighbor_name} ({source_detail}; {target_detail})")

    lines.append("; ".join(details) + ".")
    return "\n".join(lines)



def get_entity_type(entity_name):
    """
    Infer entity type from name (simple heuristic for GDELT)
    """
    entity_lower = str(entity_name).lower()

    if any(word in entity_lower for word in ['president', 'minister', 'governor', 'official']):
        return "Government Official"
    elif any(word in entity_lower for word in ['government', 'administration', 'parliament']):
        return "Government Institution"
    elif any(word in entity_lower for word in ['terrorist', 'insurgent', 'militant', 'rebel']):
        return "Armed Group"
    elif any(word in entity_lower for word in ['citizen', 'protester', 'activist', 'demonstrator']):
        return "Civilian Actor"
    elif any(word in entity_lower for word in ['country', 'nation', 'state']) or entity_lower in ['egypt', 'nigeria', 'israel', 'america', 'china']:
        return "Nation/Country"
    else:
        return "Political Actor"


def _normalize_profile_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def _short_profile_text(text, max_chars):
    clean = _normalize_profile_text(text)
    if max_chars is None or max_chars <= 0 or len(clean) <= int(max_chars):
        return clean
    keep = max(1, int(max_chars) - 3)
    return clean[:keep].rstrip() + "..."


def _extract_entity_meaning(summary_text):
    clean = _normalize_profile_text(summary_text)
    if not clean:
        return ""
    if clean.lower().startswith("summary:"):
        clean = clean.split(":", 1)[1].strip()
    match = re.search(
        r"EntityMeaning\s*=\s*(.*?)(?:;\s*InteractionSummary\s*=|$)",
        clean,
        flags=re.IGNORECASE,
    )
    if match:
        return _normalize_profile_text(match.group(1))
    return clean


def create_prompt(
    source_entity,
    relation,
    target_entity,
    source_id,
    relation_id,
    target_id,
    prediction_time,
    source_history,
    target_history,
    entity_map,
    relation_map,
    source_summary=None,
    target_summary=None,
    summary_mode="off",
    summary_max_chars=120,
    source_history_entities=None,
    target_history_entities=None,
    mutual_history=None,
    num_past_interactions=0,
    global_avg_interactions=0.0,
    source_popularity=0,
    target_popularity=0,
    avg_node_popularity=0.0,
    common_neighbors=None,
    history_window=47,
    include_key_signals=True,
    key_signal_fields=None,
    use_raw_key_signals=False,
    use_percentile_key_signals=False,
    include_expert_prediction=True,
    source_history_desc="most recent",
    target_history_desc="most recent",
    use_chat_template=True,
    tokenizer=None,
    use_cot=True,
    last_interaction_str=None,
    common_neighbor_level="Modest",
    recent_degree_signal=None,
    global_recency_signal=None,
    itemcf_signal=None,
    usercf_signal=None,
    expert_prediction="Unknown",
    common_neighbors_desc="sorted by popularity",
    mutual_timestamps_only=False,
    mutual_timestamps_dedup=False,
    mutual_summary_count_recency=False,
    common_neighbors_names_only=False,
    compact_common_neighbors_top_k=0,
    compact_common_neighbors_novel_only=False,
    ablate_mutual_history=False,
    ablate_common_neighbors=False,
    ablate_source_history=False,
    ablate_target_history=False,
    ablate_source_target_history=False,
    ablate_reasoning_guidance=False,
    no_cot_output_0_100=False,
    include_edge_type=False,
    include_edge_type_except_target=False,
    prompt_variant="gdelt",
    include_overall_structural_signal=False,
    overall_structural_signal="Unknown",
    overall_structural_signal_name="Overall structural signal",
    compacted_history_text=None,
    few_shot_examples=None,
    graph_prompt_special_token=None,
    graph_prompt_num_tokens=0,
    history_table_aliases=False,
    anonymous_entity_aliases=False,
    natural_grouped_history=False,
    natural_activity_summary=False,
    natural_neighbor_names_only=False,
    natural_activity_compact_top3=False,
    natural_activity_top_k=3,
):
    """
    Create a structured prompt for link prediction

    Args:
        source_entity: Source entity name (str)
        relation: Relation name (str)
        target_entity: Target entity name (str)
        source_id: Source entity ID (int)
        relation_id: Relation ID (int)
        target_id: Target entity ID (int)
        prediction_time: Timestamp to predict (int)
        source_history: List of past incoming and outgoing events involving source entity [(u, r, i, ts), ...]
        target_history: List of past events involving target entity [(u, r, i, ts), ...]
        source_history_entities: Optional entity-centric source history [(entity_id, [ts, ...]), ...]
        target_history_entities: Optional entity-centric target history [(entity_id, [ts, ...]), ...]
        entity_map: Dict mapping entity_id -> entity_name
        relation_map: Dict mapping relation_id -> relation_text
        source_summary: Optional summary text for source endpoint
        target_summary: Optional summary text for target endpoint
        summary_mode: off | entity_meaning | full
        summary_max_chars: Max chars per endpoint profile snippet
        mutual_history: List of past events between source and target [(u, r, i, ts), ...]
        history_window: Number of historical events to include
        include_key_signals: Whether to include the KEY SIGNALS section
        key_signal_fields: Which prompt key-signal lines to render
        use_raw_key_signals: If True, show raw numeric values for key signals instead of bucket labels
        use_percentile_key_signals: If True, show percentile key signals instead of bucket labels
        include_expert_prediction: Whether to include expert prediction in the prompt
        source_history_desc: Descriptor for how source history is selected (default: "most recent")
        target_history_desc: Descriptor for how target history is selected (default: "most recent")
        use_chat_template: Whether to use Llama chat template format (default: True)
        tokenizer: Tokenizer for applying chat template (required if use_chat_template=True)
        use_cot: Whether to use Chain-of-Thought (True) or Direct Answer (False)
        last_interaction_str: Optional precomputed interaction recency string
        common_neighbor_level: Bucketed RA level for common neighbors
        recent_degree_signal: Prompt-facing recent-degree signal
        global_recency_signal: Prompt-facing global-recency signal
        itemcf_signal: Prompt-facing item-based collaborative filtering signal
        usercf_signal: Prompt-facing user-based collaborative filtering signal
        expert_prediction: Expert prediction string (e.g., "True", "False")
        common_neighbors_desc: Descriptor for common-neighbor selection rule
        mutual_timestamps_only: If True, mutual history is rendered as timestamps only
        mutual_timestamps_dedup: If True, collapse consecutive duplicate timestamps
        mutual_summary_count_recency: If True, replace the mutual timestamp list with
            existence, retained distinct-timestamp count, and latest-event recency
        common_neighbors_names_only: If True, only show common-neighbor entity names
        compact_common_neighbors_top_k: If positive, summarize the selected semantic
            common-neighbor pool and show at most this many ordered neighbors
        compact_common_neighbors_novel_only: Exclude common-neighbor entity IDs already
            named in either rendered endpoint history, then backfill up to compact K
        ablate_mutual_history: If True, remove the MUTUAL HISTORY section
        ablate_common_neighbors: If True, remove the COMMON NEIGHBORS section
        ablate_source_history: If True, remove the SOURCE HISTORY section
        ablate_target_history: If True, remove the TARGET HISTORY section
        ablate_source_target_history: Backward-compatible alias; if True, remove both
            SOURCE HISTORY and TARGET HISTORY sections
        ablate_reasoning_guidance: If True, remove reasoning guidance and keep only output-format constraints
        no_cot_output_0_100: If True and use_cot=False, request scalar score X in [0, 100]
        include_edge_type: If True, include relation/edge-type labels in event descriptions
        include_edge_type_except_target: If True, include relation labels in observed context
            but hide the queried target relation text
        prompt_variant: Prompt framing variant ("gdelt" or "temporal_link_prediction")
        include_overall_structural_signal: If True, enable overall-signal control flow for expert prediction (handled upstream)
        overall_structural_signal: Bucketed value for overall structural signal (Low/Modest/High/Unknown)
        overall_structural_signal_name: Reserved display label for backward compatibility
        compacted_history_text: Optional condensed history block. When provided, it replaces raw history sections.
        few_shot_examples: Optional compact few-shot summary blocks (list[str])
        graph_prompt_special_token: Optional dedicated graph placeholder token
        graph_prompt_num_tokens: Number of repeated graph placeholder tokens to insert
        history_table_aliases: If True, replace repeated endpoint names inside
            history event rows with the literal aliases "Source" and "Target"
        anonymous_entity_aliases: If True, replace endpoint names with Source/Target
            and every other visible entity with prompt-local N1, N2, ... aliases.
            Aliases are consistent only within one prompt and reset across prompts.
        natural_grouped_history: If True, group source/target events by named
            counterpart and list every timestamp in a natural table-like format
        natural_activity_summary: If True, summarize source/target histories with
            natural recurrence, recency, diversity, and long-tail statistics
        natural_neighbor_names_only: If True, retain only deduplicated salient
            source/target neighbor names, without numeric history statistics
        natural_activity_compact_top3: If True, retain aggregate activity and
            the top-3 frequent/recent partners with count and latest time
        natural_activity_top_k: Number of frequent and recent partners selected
            by the compact activity or names-only formatter

    Returns:
        Formatted prompt string (or chat-formatted string if use_chat_template=True)
    """
    if mutual_history is None:
        mutual_history = []
    if common_neighbors is None:
        common_neighbors = []
    if source_history_entities is None:
        source_history_entities = []
    if target_history_entities is None:
        target_history_entities = []
    if ablate_source_target_history:
        ablate_source_history = True
        ablate_target_history = True
    compact_common_neighbors_top_k = int(compact_common_neighbors_top_k or 0)
    if compact_common_neighbors_top_k < 0:
        raise ValueError("compact_common_neighbors_top_k must be >= 0")
    if compact_common_neighbors_top_k and common_neighbors_names_only:
        raise ValueError(
            "compact common-neighbor rendering and names-only rendering are mutually exclusive"
        )
    if compact_common_neighbors_novel_only:
        if not compact_common_neighbors_top_k:
            raise ValueError(
                "compact_common_neighbors_novel_only requires compact_common_neighbors_top_k"
            )
        if not natural_activity_compact_top3:
            raise ValueError(
                "compact_common_neighbors_novel_only requires compact natural activity histories"
            )
        if anonymous_entity_aliases:
            raise ValueError(
                "compact_common_neighbors_novel_only is incompatible with anonymous entity aliases"
            )
        if str(compacted_history_text or "").strip():
            raise ValueError(
                "compact_common_neighbors_novel_only cannot be used with compacted_history_text"
            )
    key_signal_fields = normalize_key_signal_fields(key_signal_fields)

    prompt_variant_norm = str(prompt_variant or "gdelt").strip().lower()
    if prompt_variant_norm not in {"gdelt", "temporal_link_prediction"}:
        raise ValueError(
            "prompt_variant must be one of {'gdelt', 'temporal_link_prediction'}"
        )
    include_context_edge_type = bool(include_edge_type or include_edge_type_except_target)
    natural_history_modes = sum(
        bool(value)
        for value in (
            natural_grouped_history,
            natural_activity_summary,
            natural_neighbor_names_only,
            natural_activity_compact_top3,
        )
    )
    if natural_history_modes > 1:
        raise ValueError(
            "natural_grouped_history, natural_activity_summary, and "
            "natural_neighbor_names_only/natural_activity_compact_top3 are mutually exclusive"
        )

    selected_common_neighbors = list(common_neighbors or [])[: int(history_window)]
    selected_common_neighbor_count = len(selected_common_neighbors)
    if compact_common_neighbors_novel_only:
        rendered_history_ids = set()
        if not ablate_source_history:
            if source_history_entities:
                rendered_history_ids.update(
                    int(entity_id)
                    for entity_id, _timestamps in source_history_entities[: int(history_window)]
                )
            else:
                recent_source_history_for_filter = list(source_history or [])[-int(history_window):]
                rendered_history_ids.update(
                    _compact_activity_partner_ids(
                        recent_source_history_for_filter,
                        entity_map,
                        endpoint_id=source_id,
                        history_role="source",
                        top_k=natural_activity_top_k,
                    )
                )
        if not ablate_target_history:
            if target_history_entities:
                rendered_history_ids.update(
                    int(entity_id)
                    for entity_id, _timestamps in target_history_entities[: int(history_window)]
                )
            else:
                recent_target_history_for_filter = list(target_history or [])[-int(history_window):]
                rendered_history_ids.update(
                    _compact_activity_partner_ids(
                        recent_target_history_for_filter,
                        entity_map,
                        endpoint_id=target_id,
                        history_role="target",
                        top_k=natural_activity_top_k,
                    )
                )
        common_neighbors = [
            neighbor
            for neighbor in selected_common_neighbors
            if int(neighbor[0]) not in rendered_history_ids
        ]
    else:
        common_neighbors = selected_common_neighbors

    if anonymous_entity_aliases:
        entity_map = build_prompt_local_entity_aliases(
            source_id=source_id,
            target_id=target_id,
            source_history=source_history,
            target_history=target_history,
            mutual_history=mutual_history,
            source_history_entities=source_history_entities,
            target_history_entities=target_history_entities,
            common_neighbors=common_neighbors,
            history_window=history_window,
            include_mutual_history=not ablate_mutual_history,
            include_common_neighbors=not ablate_common_neighbors,
            include_source_history=not ablate_source_history,
            include_target_history=not ablate_target_history,
        )
        source_entity = "Source"
        target_entity = "Target"
        source_summary = None
        target_summary = None
        summary_mode = "off"
        history_table_aliases = True

    # Get entity types
    source_type = get_entity_type(source_entity)
    target_type = get_entity_type(target_entity)
    if prompt_variant_norm == "temporal_link_prediction":
        source_entity_line = str(source_entity)
        target_entity_line = str(target_entity)
        task_intro = (
            "You are analyzing a temporal interaction graph. Your task is to predict "
            "whether a future link between the source entity and the target entity "
            "will occur at the prediction time using only historical interaction "
            "patterns observed before that time."
        )
        prediction_question = "Will a link between the following entities occur at the prediction time?"
        task_item_label = "Link"
        negative_outcome_text = "Link will NOT occur"
        positive_outcome_text = "Link WILL occur"
    else:
        source_entity_line = f"{source_entity} ({source_type})"
        target_entity_line = f"{target_entity} ({target_type})"
        task_intro = (
            "You are analyzing political events from the GDELT (Global Database of "
            "Events, Language, and Tone) database. Your task is to predict whether a "
            "specific event will occur based on historical patterns."
        )
        prediction_question = "Will the following event occur?"
        task_item_label = "Event"
        negative_outcome_text = "Event will NOT occur"
        positive_outcome_text = "Event WILL occur"

    summary_mode_norm = str(summary_mode or "off").strip().lower()
    source_profile_text = ""
    target_profile_text = ""
    if summary_mode_norm in {"entity_meaning", "full"}:
        if summary_mode_norm == "entity_meaning":
            source_profile_text = _extract_entity_meaning(source_summary)
            target_profile_text = _extract_entity_meaning(target_summary)
        else:
            source_profile_text = _normalize_profile_text(source_summary)
            target_profile_text = _normalize_profile_text(target_summary)
        source_profile_text = _short_profile_text(source_profile_text, summary_max_chars)
        target_profile_text = _short_profile_text(target_profile_text, summary_max_chars)

    entity_profiles_section = ""
    if summary_mode_norm != "off" and (source_profile_text or target_profile_text):
        source_value = source_profile_text if source_profile_text else "N/A"
        target_value = target_profile_text if target_profile_text else "N/A"
        entity_profiles_section = (
            "ENTITY PROFILES\n"
            f"Source profile: {source_value}\n"
            f"Target profile: {target_value}\n\n"
        )

    # Format source history (event-centric by default, entity-centric if provided)
    if source_history_entities:
        recent_source_entities = source_history_entities[:history_window]
        formatted_source_history = format_entity_timestamp_series(recent_source_entities, entity_map)
    else:
        recent_source_history = source_history[-history_window:] if len(source_history) > history_window else source_history
        if natural_activity_compact_top3:
            formatted_source_history = format_natural_activity_compact(
                recent_source_history,
                entity_map,
                endpoint_id=source_id,
                history_role="source",
                top_k=natural_activity_top_k,
            )
        elif natural_neighbor_names_only:
            formatted_source_history = format_natural_neighbor_names_only(
                recent_source_history,
                entity_map,
                endpoint_id=source_id,
                history_role="source",
                top_frequent=natural_activity_top_k,
                top_recent=natural_activity_top_k,
            )
        elif natural_activity_summary:
            formatted_source_history = format_natural_activity_summary(
                recent_source_history,
                entity_map,
                endpoint_id=source_id,
                history_role="source",
            )
        elif natural_grouped_history:
            formatted_source_history = format_natural_grouped_history(
                recent_source_history,
                entity_map,
                relation_map,
                endpoint_id=source_id,
                history_role="source",
                include_edge_type=include_context_edge_type,
            )
        else:
            formatted_source_history = format_historical_events(
                recent_source_history,
                entity_map,
                relation_map,
                prediction_time,
                include_edge_type=include_context_edge_type,
                source_id=source_id,
                target_id=target_id,
                use_aliases=history_table_aliases,
            )

    # Format target history (event-centric by default, entity-centric if provided)
    if target_history_entities:
        recent_target_entities = target_history_entities[:history_window]
        formatted_target_history = format_entity_timestamp_series(recent_target_entities, entity_map)
    else:
        recent_target_history = target_history[-history_window:] if len(target_history) > history_window else target_history
        if natural_activity_compact_top3:
            formatted_target_history = format_natural_activity_compact(
                recent_target_history,
                entity_map,
                endpoint_id=target_id,
                history_role="target",
                top_k=natural_activity_top_k,
            )
        elif natural_neighbor_names_only:
            formatted_target_history = format_natural_neighbor_names_only(
                recent_target_history,
                entity_map,
                endpoint_id=target_id,
                history_role="target",
                top_frequent=natural_activity_top_k,
                top_recent=natural_activity_top_k,
            )
        elif natural_activity_summary:
            formatted_target_history = format_natural_activity_summary(
                recent_target_history,
                entity_map,
                endpoint_id=target_id,
                history_role="target",
            )
        elif natural_grouped_history:
            formatted_target_history = format_natural_grouped_history(
                recent_target_history,
                entity_map,
                relation_map,
                endpoint_id=target_id,
                history_role="target",
                include_edge_type=include_context_edge_type,
            )
        else:
            formatted_target_history = format_historical_events(
                recent_target_history,
                entity_map,
                relation_map,
                prediction_time,
                include_edge_type=include_context_edge_type,
                source_id=source_id,
                target_id=target_id,
                use_aliases=history_table_aliases,
            )

    # Format mutual history
    recent_mutual_history = mutual_history[-history_window:] if len(mutual_history) > history_window else mutual_history
    if mutual_summary_count_recency:
        formatted_mutual_history = format_mutual_existence_count_recency(
            recent_mutual_history,
            prediction_time,
            deduplicate=True,
        )
    elif mutual_timestamps_only:
        formatted_mutual_history = format_mutual_timestamps_only(
            recent_mutual_history,
            deduplicate=mutual_timestamps_dedup
        )
    else:
        formatted_mutual_history = format_historical_events(
            recent_mutual_history,
            entity_map,
            relation_map,
            prediction_time,
            include_edge_type=include_context_edge_type,
            source_id=source_id,
            target_id=target_id,
            use_aliases=history_table_aliases,
        )
    
    # Format common neighbors
    if compact_common_neighbors_top_k:
        formatted_common_neighbors = format_common_neighbors_compact(
            common_neighbors,
            entity_map,
            relation_map,
            top_k=compact_common_neighbors_top_k,
            include_edge_type=include_context_edge_type,
            selected_total=selected_common_neighbor_count,
            novel_only=compact_common_neighbors_novel_only,
        )
    else:
        formatted_common_neighbors = format_common_neighbors(
            common_neighbors,
            entity_map,
            relation_map,
            prediction_time,
            names_only=common_neighbors_names_only,
            include_edge_type=include_context_edge_type,
        )

    compacted_history_text = str(compacted_history_text or "").strip()
    if compacted_history_text:
        history_context_section = (
            "COMPACTED HISTORY\n"
            "The following condensed context was produced from the raw history window:\n"
            f"{compacted_history_text}\n"
        )
    else:
        history_sections = []
        if not ablate_mutual_history:
            history_sections.append(
                "MUTUAL HISTORY\n"
                f"The most recent interactions BETWEEN {source_entity} and {target_entity} "
                f"(up to {history_window} events):\n"
                f"{formatted_mutual_history}"
            )
        if not ablate_common_neighbors:
            if compact_common_neighbors_top_k:
                history_sections.append(
                    "COMMON NEIGHBORS\n"
                    f"{formatted_common_neighbors}"
                )
            else:
                history_sections.append(
                    "COMMON NEIGHBORS\n"
                    f"Common neighbors (up to {history_window}, {common_neighbors_desc}): "
                    f"{formatted_common_neighbors}"
                )
        if not ablate_source_history:
            source_history_heading = (
                "Recent interactions involving the source entity (incoming and outgoing):"
                if (natural_grouped_history or natural_activity_summary or natural_neighbor_names_only or natural_activity_compact_top3) and not source_history_entities
                else f"The {source_history_desc} actions received BY {source_entity} or performed BY them "
                     f"(up to {history_window} events):"
            )
            history_sections.append(
                "SOURCE HISTORY\n"
                f"{source_history_heading}\n"
                f"{formatted_source_history}"
            )
        if not ablate_target_history:
            target_history_heading = (
                "Recent interactions involving the target entity:"
                if (natural_grouped_history or natural_activity_summary or natural_neighbor_names_only or natural_activity_compact_top3) and not target_history_entities
                else f"The {target_history_desc} actions received BY {target_entity} or performed BY them "
                     f"(up to {history_window} events):"
            )
            history_sections.append(
                "TARGET HISTORY\n"
                f"{target_history_heading}\n"
                f"{formatted_target_history}"
            )
        if history_sections:
            history_context_section = "\n\n".join(history_sections) + "\n"
        else:
            history_context_section = (
                "STRUCTURAL HISTORY\n"
                "All structural history sections are ablated for this prompt.\n"
            )

    if include_edge_type:
        event_line = f"{source_entity} --[{relation}]--> {target_entity}"
        event_type_section = f"Event Type (Relation): {relation}\n"
    else:
        event_line = f"{source_entity} --> {target_entity}"
        event_type_section = ""

    graph_prompt_section = ""
    graph_prompt_special_token = str(graph_prompt_special_token or "").strip()
    if graph_prompt_special_token and int(graph_prompt_num_tokens) > 0:
        repeated_graph_tokens = " ".join(
            [graph_prompt_special_token] * int(graph_prompt_num_tokens)
        )
        graph_prompt_section = (
            "GRAPH TOKENS\n"
            "Dedicated structural graph tokens for the current source-target pair:\n"
            f"{repeated_graph_tokens}\n\n"
        )

    # Calculate time since last interaction
    if last_interaction_str is None:
        if mutual_history:
            # mutual_history is sorted by time
            last_ts = mutual_history[-1][3]
            delta_last = prediction_time - last_ts
            time_str = format_timedelta(delta_last)
            last_interaction_str = f"The most recent interaction occurred {time_str} ago."
        else:
            last_interaction_str = "No prior interactions."

    # Select template
    template = LINK_PREDICTION_TEMPLATE if use_cot else LINK_PREDICTION_TEMPLATE_NO_COT

    # Build key signals / expert prediction sections (optional)
    expert_prediction_section = ""
    expert_prediction_is_neutral = False
    if include_expert_prediction:
        expert_prediction_text = str(expert_prediction).strip()
        if not expert_prediction_text:
            expert_prediction_text = "Unknown"
        expert_prediction_is_neutral = (
            expert_prediction_text.lower() in {"neutral", "unknown", "modest", "n/a", "na"}
        )
        expert_prediction_section = f"Prior Signal: {expert_prediction_text}\n\n"

    key_signals_section = ""
    if include_key_signals:
        legend_lines = []

        def _fmt_pct(value):
            if value is None:
                return "N/A"
            if isinstance(value, str):
                return value
            try:
                return f"{float(value):.1f}%"
            except (TypeError, ValueError):
                return str(value)

        if use_raw_key_signals:
            recency_value = (
                "No prior interactions"
                if last_interaction_str in (None, "No prior interactions")
                else last_interaction_str
            )
            key_line_map = {
                "target_popularity": f"Target popularity (raw dynamic degree): {target_popularity}",
                "past_interactions": f"Past source-to-target interactions (raw count): {num_past_interactions}",
                "recency": f"Interaction recency (raw delta in ts units): {recency_value}",
                "common_neighbor": f"Common neighbor (raw RA score): {common_neighbor_level}",
                "recent_degree": (
                    "Recent degree (raw recent count): "
                    f"{recent_degree_signal}"
                ),
                "global_recency": (
                    "Global recency (raw delta in ts units): "
                    f"{global_recency_signal}"
                ),
                "itemcf": f"ItemCF cosine (raw collaborative score): {itemcf_signal}",
                "usercf": f"UserCF cosine (raw collaborative score): {usercf_signal}",
            }
        elif use_percentile_key_signals:
            legend_lines = [
                "Legend: Percentiles are rank signals in [0, 100], not probabilities.",
                "Legend: Higher percentile means stronger support for a future link."
            ]
            key_line_map = {
                "target_popularity": f"Target popularity percentile: {_fmt_pct(target_popularity)}",
                "past_interactions": f"Past source-to-target interactions percentile: {_fmt_pct(num_past_interactions)}",
                "recency": f"Interaction recency percentile: {_fmt_pct(last_interaction_str)}",
                "common_neighbor": f"Common neighbor percentile: {_fmt_pct(common_neighbor_level)}",
                "recent_degree": f"Recent degree percentile: {_fmt_pct(recent_degree_signal)}",
                "global_recency": f"Global recency percentile: {_fmt_pct(global_recency_signal)}",
                "itemcf": f"ItemCF cosine percentile: {_fmt_pct(itemcf_signal)}",
                "usercf": f"UserCF cosine percentile: {_fmt_pct(usercf_signal)}",
            }
        else:
            key_line_map = {
                "target_popularity": f"Target popularity level: {target_popularity}",
                "past_interactions": f"Past source-to-target interactions level: {num_past_interactions}",
                "recency": f"Interaction recency level: {last_interaction_str}",
                "common_neighbor": f"Common neighbor level: {common_neighbor_level}",
                "recent_degree": f"Recent degree level: {recent_degree_signal}",
                "global_recency": f"Global recency level: {global_recency_signal}",
                "itemcf": f"ItemCF cosine level: {itemcf_signal}",
                "usercf": f"UserCF cosine level: {usercf_signal}",
            }
        key_lines = [key_line_map[field] for field in key_signal_fields if field in key_line_map]
        section_lines = legend_lines + key_lines
        # Overall structural signal is intentionally not rendered as an extra
        # KEY SIGNALS line. It is consumed upstream to gate/override expert
        # prediction behavior.
        key_signals_section = "KEY SIGNALS\n" + "\n".join(section_lines) + "\n\n"

    has_entity_profiles = bool(entity_profiles_section)
    has_few_shot_examples = bool(few_shot_examples)
    has_compacted_history = bool(compacted_history_text)
    has_mutual_history_section = (not has_compacted_history) and (not ablate_mutual_history)
    has_common_neighbors_section = (not has_compacted_history) and (not ablate_common_neighbors)
    has_source_history_section = (not has_compacted_history) and (not ablate_source_history)
    has_target_history_section = (not has_compacted_history) and (not ablate_target_history)
    mode_guidance_lines = []
    if has_few_shot_examples:
        mode_guidance_lines.append(
            "Use REFERENCE EXAMPLES as few-shot guidance for your decision while still grounding the decision in the current query evidence."
        )

    if prompt_variant_norm == "temporal_link_prediction":
        compacted_history_guidance = (
            "Examine the COMPACTED HISTORY for temporal, neighborhood, and behavioral patterns relevant to whether the future link will occur."
        )
        mutual_history_guidance = (
            "Analyze the MUTUAL HISTORY for temporal patterns. Higher frequency and more recent interactions often indicate a stronger chance of a future link."
        )
        common_neighbors_guidance = (
            "Evaluate the influence of COMMON NEIGHBORS. Shared neighbors can signal proximity in the temporal graph and increase the probability of a future link."
        )
        source_history_guidance = (
            "Examine the SOURCE HISTORY for behavioral trends. Repeated interactions with similar entities or neighborhoods can indicate a higher chance of this link recurring or emerging."
        )
        target_history_guidance = (
            "Examine the TARGET HISTORY for behavioral trends. Repeated interactions with similar entities or neighborhoods can indicate a higher chance of this link recurring or emerging."
        )
    else:
        compacted_history_guidance = (
            "Examine the COMPACTED HISTORY for temporal, neighborhood, and behavioral patterns relevant to whether the event will occur."
        )
        mutual_history_guidance = (
            "Analyze the MUTUAL HISTORY for temporal patterns. High frequency and recency of past interactions often indicate a strong likelihood of future events."
        )
        common_neighbors_guidance = (
            "Evaluate the influence of COMMON NEIGHBORS. Significant shared connections can bridge entities and increase the probability of interaction."
        )
        source_history_guidance = (
            "Examine the SOURCE HISTORY for behavioral trends. If the source entity frequently initiates events toward similar actors, it suggests a propensity to interact here."
        )
        target_history_guidance = (
            "Examine the TARGET HISTORY for behavioral trends. If the target entity frequently receives or initiates events with similar actors, it suggests a propensity to interact here."
        )

    history_guidance_lines = []
    if has_compacted_history:
        history_guidance_lines.append(compacted_history_guidance)
    else:
        if has_mutual_history_section:
            history_guidance_lines.append(mutual_history_guidance)
        if has_common_neighbors_section:
            history_guidance_lines.append(common_neighbors_guidance)
        if has_source_history_section:
            history_guidance_lines.append(source_history_guidance)
        if has_target_history_section:
            history_guidance_lines.append(target_history_guidance)

    # Build instruction blocks
    if ablate_reasoning_guidance:
        analysis_lines = [
            'YOU MUST END YOUR RESPONSE with the exact phrase: "Therefore, the answer is: X" (where X is an integer from 0 to 100).'
        ]
    else:
        analysis_lines = []
        if include_expert_prediction:
            if expert_prediction_is_neutral:
                analysis_lines.append(
                    "Consider the PRIOR SIGNAL provided as a initial cue. If it is Neutral/Unknown, treat it as non-directional context."
                )
            else:
                analysis_lines.append(
                    "Consider the PRIOR SIGNAL provided as a initial cue."
                )
        if include_key_signals:
            if prompt_variant_norm == "temporal_link_prediction":
                analysis_lines.append(
                    "Consider the KEY SIGNALS provided. Stronger KEY SIGNALS increase the likelihood that the queried link will occur at the prediction time."
                )
            else:
                analysis_lines.append(
                    "Consider the KEY SIGNALS provided, high KEY SIGNALS increase the likelihood of the event occurring."
                )
        analysis_lines.extend(mode_guidance_lines)
        if prompt_variant_norm == "temporal_link_prediction":
            plausibility_line = (
                f"Assess the structural plausibility of a future link between {source_entity} and {target_entity} based on the ENTITY PROFILES."
                if has_entity_profiles
                else f"Assess the structural plausibility of a future link between {source_entity} and {target_entity}."
            )
            analysis_lines.extend(history_guidance_lines)
            analysis_lines.extend([
                plausibility_line,
                "Reason step-by-step to derive the likelihood that this link will occur at the prediction time.",
                'YOU MUST END YOUR RESPONSE with the exact phrase: "Therefore, the answer is: X" (where X is an integer from 0 to 100).'
            ])
        else:
            plausibility_line = (
                f"Assess the logical plausibility of an interaction between {source_entity} ({source_type}) and {target_entity} ({target_type}) based on the ENTITY PROFILES."
                if has_entity_profiles
                else f"Assess the logical plausibility of an interaction between {source_entity} ({source_type}) and {target_entity} ({target_type})."
            )
            analysis_lines.extend(history_guidance_lines)
            analysis_lines.extend([
                plausibility_line,
                "Reason step-by-step to derive the likelihood of this specific event occurring.",
                'YOU MUST END YOUR RESPONSE with the exact phrase: "Therefore, the answer is: X" (where X is an integer from 0 to 100).'
            ])
    analysis_instructions = "\n".join([f"{i+1}. {line}" for i, line in enumerate(analysis_lines)])

    if ablate_reasoning_guidance:
        direct_lines = [
            (
                'The format MUST be exactly: "The answer is: X" '
                '(where X is an integer from 0 to 100).'
                if no_cot_output_0_100 else
                'The format MUST be exactly: "The answer is: X" (where X is 0 or 1).'
            )
        ]
    else:
        direct_lines = []
        if include_expert_prediction:
            if expert_prediction_is_neutral:
                direct_lines.append(
                    "Consider the PRIOR SIGNAL provided as a initial cue. If it is Neutral/Unknown, treat it as non-directional context."
                )
            else:
                direct_lines.append(
                    "Consider the PRIOR SIGNAL provided as a initial cue."
                )
        if include_key_signals:
            if prompt_variant_norm == "temporal_link_prediction":
                direct_lines.append(
                    "Consider the KEY SIGNALS provided. Stronger KEY SIGNALS increase the likelihood that the queried link will occur at the prediction time."
                )
            else:
                direct_lines.append(
                    "Consider the KEY SIGNALS provided, high KEY SIGNALS increase the likelihood of the event occurring."
                )
        direct_lines.extend(mode_guidance_lines)
        if prompt_variant_norm == "temporal_link_prediction":
            direct_plausibility_line = (
                f"Consider the structural plausibility of a future link between {source_entity} and {target_entity} based on the ENTITY PROFILES."
                if has_entity_profiles
                else f"Consider the structural plausibility of a future link between {source_entity} and {target_entity}."
            )
            direct_lines.extend([line.replace("Analyze", "Consider", 1).replace("Evaluate", "Consider", 1).replace("Examine", "Consider", 1) for line in history_guidance_lines])
            direct_lines.extend([
                direct_plausibility_line,
                (
                    'The format MUST be exactly: "The answer is: X" '
                    '(where X is an integer from 0 to 100).'
                    if no_cot_output_0_100 else
                    'The format MUST be exactly: "The answer is: X" (where X is 0 or 1).'
                )
            ])
        else:
            direct_plausibility_line = (
                f"Consider the logical plausibility of an interaction between {source_entity} ({source_type}) and {target_entity} ({target_type}) based on the ENTITY PROFILES."
                if has_entity_profiles
                else f"Consider the logical plausibility of an interaction between {source_entity} ({source_type}) and {target_entity} ({target_type})."
            )
            direct_lines.extend([line.replace("Analyze", "Consider", 1).replace("Evaluate", "Consider", 1).replace("Examine", "Consider", 1) for line in history_guidance_lines])
            direct_lines.extend([
                direct_plausibility_line,
                (
                    'The format MUST be exactly: "The answer is: X" '
                    '(where X is an integer from 0 to 100).'
                    if no_cot_output_0_100 else
                    'The format MUST be exactly: "The answer is: X" (where X is 0 or 1).'
                )
            ])
    direct_instructions = "\n".join([f"{i+1}. {line}" for i, line in enumerate(direct_lines)])

    no_cot_score_legend = (
        f"0 = {negative_outcome_text} (definitely no)\n"
        f"100 = {positive_outcome_text} (definitely yes)"
        if no_cot_output_0_100 else
        f"0 = {negative_outcome_text}\n"
        f"1 = {positive_outcome_text}"
    )

    few_shot_section = ""
    if few_shot_examples:
        demo_lines = [
            "REFERENCE EXAMPLES (GLOBAL, SUMMARY-ONLY)",
            "Use these as calibration only. Do NOT repeat the examples.",
            "Answer only for the current query in the Response section below.",
            "",
        ]
        for idx, demo in enumerate(few_shot_examples, start=1):
            demo_text = str(demo).strip()
            if not demo_text:
                continue
            demo_lines.append(f"Example {idx}")
            demo_lines.append(demo_text)
            demo_lines.append("")
        few_shot_section = "\n".join(demo_lines).rstrip() + "\n"

    # Create the content message
    content = template.format(
        task_intro=task_intro,
        source_entity_line=source_entity_line,
        target_entity_line=target_entity_line,
        source_entity=source_entity,
        source_type=source_type,
        target_entity=target_entity,
        target_type=target_type,
        prediction_time=prediction_time,
        prediction_question=prediction_question,
        task_item_label=task_item_label,
        negative_outcome_text=negative_outcome_text,
        positive_outcome_text=positive_outcome_text,
        entity_profiles_section=entity_profiles_section,
        history_context_section=history_context_section,
        event_type_section=event_type_section,
        event_line=event_line,
        expert_prediction_section=expert_prediction_section,
        key_signals_section=key_signals_section,
        analysis_instructions=analysis_instructions,
        direct_instructions=direct_instructions,
        no_cot_score_legend=no_cot_score_legend,
        formatted_source_history=formatted_source_history,
        formatted_target_history=formatted_target_history,
        formatted_mutual_history=formatted_mutual_history,
        formatted_common_neighbors=formatted_common_neighbors,
        common_neighbors_desc=common_neighbors_desc,
        num_past_interactions=num_past_interactions,
        global_avg_interactions=global_avg_interactions,
        source_popularity=source_popularity,
        target_popularity=target_popularity,
        avg_node_popularity=avg_node_popularity,
        last_interaction_str=last_interaction_str,
        relation=relation,
        common_neighbor_level=common_neighbor_level,
        expert_prediction=expert_prediction,
        history_window=history_window,
        source_history_desc=source_history_desc,
        target_history_desc=target_history_desc,
        graph_prompt_section=graph_prompt_section,
        few_shot_section=few_shot_section,
    )

    # Use chat template if requested
    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_chat_template=True")
        
        if prompt_variant_norm == "temporal_link_prediction":
            if use_cot:
                system_msg = (
                    "You are a temporal link prediction analyst. Analyze the history "
                    "step-by-step and conclude with exactly: "
                    "\"Therefore, the answer is: X\" (where X is an integer from 0 to 100)."
                )
            else:
                system_msg = (
                    "You are a temporal link prediction analyst. "
                    "Output only the final answer in the format: "
                    + (
                        "\"The answer is: X\" (where X is an integer from 0 to 100)."
                        if no_cot_output_0_100 else
                        "\"The answer is: X\" (where X is 0 or 1)."
                    )
                )
        else:
            if use_cot:
                system_msg = "You are a political event analyst. Analyze the history step-by-step and conclude with exactly: \"Therefore, the answer is: X\" (where X is an integer from 0 to 100)."
            else:
                system_msg = (
                    "You are a political event analyst. "
                    "Output only the final answer in the format: "
                    + (
                        "\"The answer is: X\" (where X is an integer from 0 to 100)."
                        if no_cot_output_0_100 else
                        "\"The answer is: X\" (where X is 0 or 1)."
                    )
                )

        messages = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": content
            }
        ]

        apply_chat_template_kwargs = {}
        tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
        if (not use_cot) and "qwen3" in tokenizer_name:
            apply_chat_template_kwargs["enable_thinking"] = False

        # Apply chat template. Older Transformers builds may not support
        # Qwen's enable_thinking kwarg, so fall back gracefully.
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **apply_chat_template_kwargs
            )
        except TypeError:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        # If No-CoT, we want to force the answer prefix if not already generated by the template
        # But Llama 3 template just adds <|start_header_id|>assistant<|end_header_id|>\n\n
        # We can append "The answer is:" to guide it if needed, but "The answer is:" is already at the end of the prompt text in NO_COT template.
        return formatted_prompt
    else:
        # Return plain template
        return content


# Example usage
if __name__ == "__main__":
    # Example data
    entity_map = {
        4: "president",
        8: "school",
        10: "government",
        12: "citizen",
        66: "nigeria"
    }

    relation_map = {
        1: "make an appeal or request",
        7: "praise or endorse",
        14: "threaten",
        4: "make a visit"
    }

    # Source history: events initiated BY "president" (entity_id=4)
    source_history = [
        (4, 1, 66, 5),   # president --[appeal]--> nigeria @ t=5
        (4, 7, 10, 12),  # president --[praise]--> government @ t=12
        (4, 14, 8, 18),  # president --[threaten]--> school @ t=18
        (4, 4, 10, 23),  # president --[visit]--> government @ t=23
    ]

    # Target history: events involving "citizen" (entity_id=12)
    target_history = [
        (10, 1, 12, 27), # government --[appeal]--> citizen @ t=27
        (12, 14, 10, 29), # citizen --[threaten]--> government @ t=29
        (8, 7, 12, 30),  # school --[praise]--> citizen @ t=30
    ]

    # Mutual history: events between "president" (4) and "citizen" (12)
    mutual_history = [
        (4, 1, 12, 10), # president --[appeal]--> citizen @ t=10
        (12, 7, 4, 15), # citizen --[praise]--> president @ t=15
    ]
    
    # Common Neighbors Example: government (10) connects them
    # president --[visit (4)]--> government @ t=23
    # government --[appeal (1)]--> citizen @ t=27
    common_neighbors = [
        (10, (4, 4, 10, 23), (10, 1, 12, 27))
    ]

    # Prediction query
    prompt = create_prompt(
        source_entity="president",
        relation="make an appeal or request",
        target_entity="citizen",
        source_id=4,
        relation_id=1,
        target_id=12,
        prediction_time=31,
        source_history=source_history,
        target_history=target_history,
        mutual_history=mutual_history,
        common_neighbors=common_neighbors,
        entity_map=entity_map,
        relation_map=relation_map,
        history_window=47,
        use_chat_template=False
    )

    print(prompt)
