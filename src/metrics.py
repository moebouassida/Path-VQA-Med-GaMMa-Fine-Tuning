import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_yes_no(answer: str) -> bool:
    return normalize(answer) in {"yes", "no"}


def exact_match(pred: str, target: str) -> float:
    return float(normalize(pred) == normalize(target))


def token_f1(pred: str, target: str) -> float:
    pred_tokens = normalize(pred).split()
    target_tokens = normalize(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(target_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(pred: str, target: str, max_n: int = 4) -> float:
    """Sentence BLEU-4, no external deps."""
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
        clipped = sum(min(c, target_ngrams.get(ng, 0)) for ng, c in pred_ngrams.items())
        scores.append(clipped / sum(pred_ngrams.values()))

    if not any(s > 0 for s in scores):
        return 0.0

    import math
    log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)
    bp = min(1.0, math.exp(1 - len(target_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(log_avg)


def _get_ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def vqa_score(pred: str, target: str) -> dict:
    # for yes/no, BLEU is aliased to exact_match per standard PathVQA protocol
    yn = is_yes_no(target)
    em = exact_match(pred, target)
    f1 = token_f1(pred, target)
    bl = em if yn else bleu_score(pred, target)
    return {
        "exact_match": em,
        "token_f1": f1,
        "bleu": bl,
        "is_yes_no": yn,
    }


def aggregate_scores(results: list[dict]) -> dict:
    if not results:
        return {}
    yn = [r for r in results if r["is_yes_no"]]
    oe = [r for r in results if not r["is_yes_no"]]

    def mean(lst, key):
        return sum(r[key] for r in lst) / len(lst) if lst else 0.0

    return {
        "overall_exact_match": mean(results, "exact_match"),
        "overall_token_f1": mean(results, "token_f1"),
        "overall_bleu": mean(results, "bleu"),
        "yes_no_accuracy": mean(yn, "exact_match"),
        "yes_no_f1": mean(yn, "token_f1"),
        "open_ended_bleu": mean(oe, "bleu"),
        "open_ended_f1": mean(oe, "token_f1"),
        "yes_no_count": len(yn),
        "open_ended_count": len(oe),
        "total": len(results),
    }


def check_quality_gates(scores: dict, cfg: dict) -> tuple:
    """Returns (passed: bool, failures: list[str])."""
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
