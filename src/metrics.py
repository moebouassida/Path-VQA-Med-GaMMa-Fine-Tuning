"""
metrics.py — VQA evaluation metrics for PathVQA.

PathVQA has two question types:
    - Yes/No questions (~50%): binary exact match
    - Open-ended questions (~50%): BLEU score

Standard PathVQA evaluation follows this split.
"""

import re
import string
from collections import Counter


# ── Text Normalization ─────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_yes_no(answer: str) -> bool:
    """Check if answer is a yes/no question."""
    return normalize(answer) in {"yes", "no"}


# ── Exact Match ───────────────────────────────────────────────────────────────


def exact_match(pred: str, target: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize(pred) == normalize(target))


# ── BLEU Score ────────────────────────────────────────────────────────────────


def bleu_score(pred: str, target: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU score (no external dependencies).
    Uses modified n-gram precision with brevity penalty.
    """
    pred_tokens = normalize(pred).split()
    target_tokens = normalize(target).split()

    if not pred_tokens or not target_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        target_ngrams = _get_ngrams(target_tokens, n)

        if not pred_ngrams:
            scores.append(0.0)
            continue

        clipped = sum(
            min(count, target_ngrams.get(ngram, 0))
            for ngram, count in pred_ngrams.items()
        )
        precision = clipped / sum(pred_ngrams.values())
        scores.append(precision)

    if not any(s > 0 for s in scores):
        return 0.0

    import math

    log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)
    bp = min(1.0, math.exp(1 - len(target_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(log_avg)


def _get_ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ── VQA Score ─────────────────────────────────────────────────────────────────


def vqa_score(pred: str, target: str) -> dict:
    """
    Compute both exact match and BLEU for a single prediction.
    Returns dict with keys: exact_match, bleu, is_yes_no
    """
    yn = is_yes_no(target)
    em = exact_match(pred, target)
    bl = bleu_score(pred, target) if not yn else em  # use EM for yes/no

    return {
        "exact_match": em,
        "bleu": bl,
        "is_yes_no": yn,
    }


def aggregate_scores(results: list[dict]) -> dict:
    """
    Aggregate per-sample scores into dataset-level metrics.

    Args:
        results: list of dicts from vqa_score()

    Returns:
        dict with overall, yes_no, and open_ended metrics
    """
    if not results:
        return {}

    yn_results = [r for r in results if r["is_yes_no"]]
    open_results = [r for r in results if not r["is_yes_no"]]

    def mean(lst, key):
        return sum(r[key] for r in lst) / len(lst) if lst else 0.0

    return {
        "overall_exact_match": mean(results, "exact_match"),
        "overall_bleu": mean(results, "bleu"),
        "yes_no_accuracy": mean(yn_results, "exact_match"),
        "open_ended_bleu": mean(open_results, "bleu"),
        "yes_no_count": len(yn_results),
        "open_ended_count": len(open_results),
        "total": len(results),
    }


def check_quality_gates(scores: dict, cfg: dict) -> tuple:
    """
    Check if model meets minimum quality thresholds.

    Returns:
        (passed, failures)
    """
    gates = {
        "yes_no_accuracy": cfg.get("gate_exact_match", 0.55),
        "open_ended_bleu": cfg.get("gate_bleu", 0.20),
    }

    failures = []
    for metric, threshold in gates.items():
        value = scores.get(metric, 0.0)
        if value < threshold:
            failures.append(f"{metric}={value:.4f} < {threshold}")

    return len(failures) == 0, failures
