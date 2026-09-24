"""
Shared LLM-side helper functions for link prediction evaluation.
"""
import math
import os
import re

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ANSWER_PHRASE = "Therefore, the answer is:"
ANSWER_PHRASE_SHORT = "answer is:"


def compute_binary_logprob_features(token_logprobs, tokenizer):
    """Extract stable binary diagnostics from one next-token distribution.

    ``binary_token_mass`` retains information discarded by the normalized
    0-vs-1 probability: it measures how much of the model's next-token mass is
    assigned to either valid answer token.  The remaining fields make the raw
    evidence available to lightweight downstream fusion models without another
    LLM forward pass.
    """
    if not token_logprobs:
        return None
    p0 = 0.0
    p1 = 0.0
    for token_id, logprob_obj in token_logprobs.items():
        token_str = tokenizer.decode([token_id]).strip()
        val = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
        if token_str == '0':
            p0 += math.exp(val)
        elif token_str == '1':
            p1 += math.exp(val)
    if p0 + p1 == 0:
        return None
    score = p1 / (p0 + p1)
    clipped_score = min(max(score, 1e-12), 1.0 - 1e-12)
    return {
        "score": float(score),
        "binary_logprob_0": float(math.log(p0)) if p0 > 0.0 else None,
        "binary_logprob_1": float(math.log(p1)) if p1 > 0.0 else None,
        "binary_logit_margin": float(
            math.log(clipped_score) - math.log1p(-clipped_score)
        ),
        "binary_token_mass": float(p0 + p1),
        "binary_entropy": float(
            -clipped_score * math.log(clipped_score)
            - (1.0 - clipped_score) * math.log1p(-clipped_score)
        ),
    }


def compute_binary_score_from_logprobs(token_logprobs, tokenizer):
    features = compute_binary_logprob_features(token_logprobs, tokenizer)
    return None if features is None else features["score"]


def extract_score_0_100_with_meta(text):
    """
    Parse 0-100 score and return (value, method) for parsing diagnostics.
    method in: exact_phrase, answer_phrase_fallback, last_integer_fallback, parse_failed
    """
    if not text:
        return None, "parse_failed"
    idx = text.find(ANSWER_PHRASE)
    if idx != -1:
        after = text[idx + len(ANSWER_PHRASE):]
        match = re.search(r"([0-9]{1,3})", after)
        if match:
            val = int(match.group(1))
            return max(0, min(100, val)), "exact_phrase"
    # Fallback: any "answer is:" phrase
    match = re.search(rf"{ANSWER_PHRASE_SHORT}\s*([0-9]{{1,3}})", text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return max(0, min(100, val)), "answer_phrase_fallback"
    # Fallback: last standalone integer 0-100
    candidates = re.findall(r"\b([0-9]{1,3})\b", text)
    if candidates:
        val = int(candidates[-1])
        return max(0, min(100, val)), "last_integer_fallback"
    return None, "parse_failed"


def extract_score_0_100(text):
    val, _ = extract_score_0_100_with_meta(text)
    return val


def load_or_compute_embeddings(entity_map, embedding_model, embedding_cache):
    """
    Load cached embeddings if available; otherwise compute E5 embeddings.
    Expects cache to be a .npy file aligned with sorted(entity_map.keys()).
    """
    if embedding_cache and os.path.exists(embedding_cache):
        loaded = np.load(embedding_cache)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            embeddings = loaded['arr_0']
        else:
            embeddings = loaded
        entity_ids = sorted(entity_map.keys())
        entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}
        return embeddings, entity_id_to_idx

    if torch is None:
        raise ImportError(
            "torch is required to compute embeddings. Install torch or provide --embedding_cache."
        )

    from experiments.modules.heuristic_models import precompute_entity_embeddings

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings, entity_id_to_idx = precompute_entity_embeddings(
        entity_map, model_name=embedding_model, device=device
    )
    if embedding_cache:
        np.save(embedding_cache, embeddings)
    return embeddings, entity_id_to_idx
__all__ = [
    "compute_binary_logprob_features",
    "compute_binary_score_from_logprobs",
    "extract_score_0_100",
    "extract_score_0_100_with_meta",
    "load_or_compute_embeddings",
]
