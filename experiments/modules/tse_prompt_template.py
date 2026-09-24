"""
Prompt template for temporal semantics extraction on temporal text-attributed graphs.
"""

from typing import Iterable, List


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _format_timestamp(timestamp: float) -> str:
    if float(timestamp).is_integer():
        return str(int(timestamp))
    return f"{timestamp:.6f}".rstrip("0").rstrip(".")


def build_temporal_semantics_prompt(
    node_id: int,
    node_text: str,
    current_timestamp: float,
    historical_interactions: Iterable[str],
    max_history: int,
    summary_style: str = "full",
) -> str:
    """
    Build one prompt for a node at one reasoning timestamp.
    """
    node_text_clean = _normalize_whitespace(node_text) if node_text else "unknown entity"
    history_list: List[str] = list(historical_interactions)
    if max_history > 0:
        history_list = history_list[-max_history:]

    if history_list:
        history_block = "\n".join(f"{idx}. {item}" for idx, item in enumerate(history_list, start=1))
    else:
        history_block = "None"

    style = str(summary_style or "full").strip().lower()

    if style == "interaction_only":
        return f"""
# Goal #
Write one concise sentence summarizing the historical neighborhood into a short thematic description

# Focal node description #
{node_text_clean}

# Current time index #
{_format_timestamp(current_timestamp)}

# Historical interactions (oldest -> newest) #
{history_block}

# Constraints #
- Treat all time values as abstract dataset indices, NOT real-world dates or years.
- Do NOT enumerate names or produce long entity lists.
- Keep the summary to one short sentence.
- If historical interactions are None, output exactly:
  Summary: No historical interaction evidence before current time index.

Provide the output STRICTLY in this format:
Summary: <one short interaction summary sentence>.
""".strip()

    return f"""
# Goal #
Produce two concise parts for the focal node at the current time index:
1) provide a concise semantic identity for the focal node
2) summarize the historical neighborhood into a short thematic description

# Node description #
{node_text_clean}

# Current time index #
{_format_timestamp(current_timestamp)}

# Historical interactions (oldest -> newest) #
{history_block}

# Constraints #
- Treat all time values as abstract dataset indices, NOT real-world dates or years.
- If the node description already has clear semantic meaning, EntityMeaning may keep it.
- For InteractionSummary, Do NOT enumerate names or produce long entity lists
- Do NOT include additional explanations unrelated to the two summary parts, like "Note: ...".
- If historical interactions are None, set InteractionSummary to:
  "No historical interaction evidence before current time index."

Provide the output STRICTLY in this format:
Summary: EntityMeaning=<...>; InteractionSummary=<...>.
""".strip()
