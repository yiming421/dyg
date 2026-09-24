#!/usr/bin/env python3
"""
Deterministic cleaners and compactors for Amazon_movies entity_text rows.

These helpers target the main failure mode in the raw item metadata:
long HTML/XML/Word-export artifacts and low-signal package boilerplate that
inflate text length without helping link prediction.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List


_FIELD_MARKER_RE = re.compile(r"\b(Title|Category|Rank|Description)\s*:", flags=re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")

_GENERIC_CATEGORY_TOKENS = {
    "all titles",
    "movies & tv",
    "genre for featured categories",
    "featured categories",
    "special interests",
    "specials",
}

_GENERIC_CATEGORY_PATTERNS = (
    re.compile(r"^all .+ titles?$", flags=re.IGNORECASE),
    re.compile(r"^studio specials?$", flags=re.IGNORECASE),
    re.compile(r"^customers .* viewed items$", flags=re.IGNORECASE),
)

_LOW_SIGNAL_MARKERS = (
    "track listing",
    "bonus feature",
    "bonus features",
    "special feature",
    "special features",
    "disc 1",
    "disc 2",
    "disc 3",
    "disc 4",
    "episode list",
    "episodes include",
    "chapter listing",
    "note on boxed sets",
    "during shipping",
    "if you are not completely satisfied",
    "refund or replace your purchase",
    "includes dolby atmos",
    "the contents in disc",
)

_LOW_SIGNAL_SENTENCE_PATTERNS = (
    re.compile(r"^\d+\.\s+\S"),
    re.compile(r"^(disc|episode|chapter|side)\s+\d+\b", flags=re.IGNORECASE),
    re.compile(r"^(track|song|bonus)\s*[:\-]", flags=re.IGNORECASE),
)

_NULLISH_SUMMARY_TEXTS = {
    "unknown",
    "n/a",
    "none",
    "not available",
}

_FORMAT_ONLY_SUMMARY_TEXTS = {
    "dvd",
    "dvd video",
    "vhs",
    "vhs tape",
    "vhs video",
    "blu-ray",
}

_SELLER_NOISE_MARKERS = (
    "quick shipping",
    "ships fast",
    "new and sealed",
    "brand new sealed",
    "factory sealed",
    "sealed",
    "great condition",
    "in great shape",
    "as is",
)


def _normalize_text(text: str) -> str:
    clean = html.unescape(str(text or ""))
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.replace("\x00", " ")
    clean = _WHITESPACE_RE.sub(" ", clean)
    return clean.strip()


def _strip_markup(text: str) -> str:
    clean = html.unescape(str(text or ""))
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.replace("\x00", " ")
    clean = clean.replace("<![CDATA[", " ").replace("]]>", " ")
    clean = _HTML_COMMENT_RE.sub(" ", clean)
    clean = _TAG_RE.sub(" ", clean)
    clean = _WHITESPACE_RE.sub(" ", clean)
    return clean.strip()


def _is_generic_category(token: str) -> bool:
    lowered = _normalize_text(token).strip(" .").lower()
    if not lowered:
        return True
    if lowered in _GENERIC_CATEGORY_TOKENS:
        return True
    return any(pattern.fullmatch(lowered) for pattern in _GENERIC_CATEGORY_PATTERNS)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        clean = _normalize_text(value).strip(" .")
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _parse_labeled_fields(text: str) -> Dict[str, str]:
    clean = _strip_markup(text)
    if not clean:
        return {}

    matches = list(_FIELD_MARKER_RE.finditer(clean))
    if not matches:
        return {}

    fields: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        value = _normalize_text(clean[start:end]).strip(" .")
        fields[key] = value
    return fields


def _clean_categories(raw_category_text: str) -> List[str]:
    if not raw_category_text:
        return []
    categories = []
    for piece in raw_category_text.split(","):
        token = _normalize_text(piece).strip(" .")
        if not token or _is_generic_category(token):
            continue
        categories.append(token)
    return _dedupe_preserve_order(categories)


def _cut_low_signal_tail(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    cut_positions = [lowered.find(marker) for marker in _LOW_SIGNAL_MARKERS if lowered.find(marker) != -1]
    if not cut_positions:
        return text
    cut_at = min(cut_positions)
    if cut_at < 80:
        return text
    return text[:cut_at].rstrip(" ,;:-")


def _is_low_signal_sentence(sentence: str) -> bool:
    clean = _normalize_text(sentence)
    if not clean:
        return True
    lowered = clean.lower()
    if any(marker in lowered for marker in _LOW_SIGNAL_MARKERS):
        return True
    if clean.count("|") >= 2:
        return True
    return any(pattern.search(clean) for pattern in _LOW_SIGNAL_SENTENCE_PATTERNS)


def _truncate_words(text: str, max_words: int) -> str:
    clean = _normalize_text(text)
    if not clean:
        return ""
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words]).rstrip(" ,;:-") + "..."


def is_low_signal_amazon_movies_summary(text: str) -> bool:
    clean = _normalize_text(text).strip(" .").lower()
    if not clean:
        return False
    if clean in _NULLISH_SUMMARY_TEXTS:
        return True
    if clean in _FORMAT_ONLY_SUMMARY_TEXTS:
        return True
    return any(marker in clean for marker in _SELLER_NOISE_MARKERS)


def _summarize_description(text: str, *, max_sentences: int = 2, max_chars: int = 280) -> str:
    clean = _cut_low_signal_tail(_strip_markup(text))
    if not clean:
        return ""

    sentences = []
    for piece in _SENTENCE_SPLIT_RE.split(clean):
        sentence = _normalize_text(piece).strip(" .")
        if not sentence or _is_low_signal_sentence(sentence):
            continue
        sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break

    if not sentences:
        return _truncate_words(clean, 40)

    summary = ". ".join(sentences).strip()
    if summary and not summary.endswith((".", "!", "?")):
        summary += "."
    if len(summary) <= max_chars:
        return summary

    clipped = summary[:max_chars].rstrip(" ,;:-")
    last_space = clipped.rfind(" ")
    if last_space >= 80:
        clipped = clipped[:last_space]
    return clipped.rstrip(" ,;:-") + "..."


def parse_amazon_movies_entity_text(text: str) -> Dict[str, object]:
    normalized = _normalize_text(text)
    fields = _parse_labeled_fields(text)
    is_item = bool(fields.get("title") or fields.get("category") or fields.get("description"))

    title = _normalize_text(fields.get("title", ""))
    categories = _clean_categories(fields.get("category", ""))
    rank = _normalize_text(fields.get("rank", ""))
    description = _summarize_description(fields.get("description", ""))
    description_low_signal = is_low_signal_amazon_movies_summary(description)

    return {
        "clean": normalized,
        "is_item": is_item,
        "title": title,
        "categories": categories,
        "rank": rank,
        "description": description,
        "description_low_signal": description_low_signal,
    }


def compress_amazon_movies_entity_text(text: str) -> str:
    parsed = parse_amazon_movies_entity_text(text)
    clean = str(parsed["clean"])
    if not clean:
        return ""

    if not parsed["is_item"]:
        return clean

    parts = []
    if parsed["title"]:
        parts.append(str(parsed["title"]))
    categories = parsed["categories"]
    if categories:
        parts.append(", ".join(categories[:3]))
    if parsed["description"]:
        parts.append(str(parsed["description"]))
    return " | ".join(part for part in parts if part) or clean


def build_clean_amazon_movies_entity_text(
    text: str,
    *,
    drop_low_signal_summary: bool = False,
    drop_all_summaries: bool = False,
) -> str:
    parsed = parse_amazon_movies_entity_text(text)
    clean = str(parsed["clean"])
    if not clean:
        return ""

    if not parsed["is_item"]:
        return clean

    parts = []
    if parsed["title"]:
        parts.append(f"Title: {parsed['title']}.")
    categories = parsed["categories"]
    if categories:
        parts.append(f"Categories: {', '.join(categories[:4])}.")
    should_drop_summary = bool(drop_all_summaries) or (
        drop_low_signal_summary and bool(parsed.get("description_low_signal"))
    )
    if parsed["description"] and not should_drop_summary:
        parts.append(f"Summary: {parsed['description']}")
    return " ".join(parts).strip() or clean


def build_compact_amazon_movies_profile(text: str) -> str:
    parsed = parse_amazon_movies_entity_text(text)
    if not parsed["is_item"]:
        return ""

    parts = []
    categories = parsed["categories"]
    if categories:
        parts.append("Categories: " + ", ".join(categories[:4]))
    if parsed["description"]:
        parts.append("Summary: " + _truncate_words(str(parsed["description"]), 24))
    return "; ".join(parts)
