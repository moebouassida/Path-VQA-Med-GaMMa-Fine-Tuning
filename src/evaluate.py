"""
evaluate.py — Standalone evaluation with quality gates.

Usage:
    python src/evaluate.py --model outputs/final --config config/config.yaml
    python src/evaluate.py --model outputs/final --config config/config.yaml --max-samples 200

Exit codes:
    0 — all quality gates passed
    1 — one or more gates failed
"""
import os
import sys
import json
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import main as load_dataset
from metrics import vqa_score, aggregate_scores, check_quality_gates
from inference import load_model, load_image, predict


def evaluate(cfg: dict, model_path: str, max_samples: int = None, output_json: str = None) -> bool:
    from datasets import load_dataset as hf_load

    print(f"\n{'='*62}")
    print("  Path-VQA Med-GaMMa — Evaluation")
    print(f"{'='*62}")
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {cfg['dataset_name']}")
    print(f"{'='*62}\n")

    # Load model
    model, processor = load_model(model_path)

    # Load test split directly
    print("[eval] Loading test split...")
    dataset = hf_load(cfg["dataset_name"])
    test_split = dataset["test"]

    if max_samples:
        test_split = test_split.select(range(min(max_samples, len(test_split))))

    print(f"[eval] Evaluating {len(test_split)} samples...")

    results = []
    for i, sample in enumerate(test_split):
        if i % 50 == 0:
            print(f"  [{i}/{len(test_split)}]")

        try:
            image  = sample["image"].convert("RGB")
            answer = predict(model, processor, image, sample["question"])
            target = sample["answer"]

            score = vqa_score(answer, target)
            results.append(score)

        except Exception as e:
            print(f"  [warn] Sample {i} failed: {e}")
            continue

    scores = aggregate_scores(results)
    passed, failures = check_quality_gates(scores, cfg)

    # Print results
    print(f"\n  {'Metric':<30} {'Value':>8}   {'Threshold':>10}   {'Status'}")
    print(f"  {'-'*65}")

    gates = {
        "yes_no_accuracy": cfg.get("gate_exact_match", 0.55),
        "open_ended_bleu":  cfg.get("gate_bleu",        0.20),
    }

    for metric, threshold in gates.items():
        value  = scores.get(metric, 0.0)
        ok     = value >= threshold
        status = "PASS" if ok else "FAIL"
        note   = "  <- BELOW THRESHOLD" if not ok else ""
        print(f"  {metric:<30} {value:>8.4f}   {threshold:>10.4f}   {status}{note}")

    print(f"\n  {'overall_exact_match':<30} {scores.get('overall_exact_match', 0):>8.4f}")
    print(f"  {'overall_bleu':<30} {scores.get('overall_bleu', 0):>8.4f}")
    print(f"  {'yes_no_count':<30} {scores.get('yes_no_count', 0):>8}")
    print(f"  {'open_ended_count':<30} {scores.get('open_ended_count', 0):>8}")
    print(f"\n{'='*62}")

    if passed:
        print("\n  All quality gates passed — ready for deployment.\n")
    else:
        print(f"\n  Quality gate failed: {failures}\n")

    # Save results
    output = {
        "scores":   scores,
        "gates":    gates,
        "passed":   passed,
        "failures": failures,
        "model":    model_path,
        "samples":  len(results),
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved -> {output_json}")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="outputs/final")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", default="metrics/eval_results.json")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    passed = evaluate(cfg, args.model, args.max_samples, args.output_json)
    sys.exit(0 if passed else 1)