"""Evaluate the frozen synthetic v0.9 one-shot concept study.

No network, GPU, Ollama or external package is required.

    python -m examples.concept_one_shot --split dev --summary
    python -m examples.concept_one_shot --split test
"""
import argparse
import json
from pathlib import Path

from acm.concept_induction import read_concept_episodes, evaluate_concept_episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACM v0.9 synthetic one-positive-example concept ambiguity study"
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[1] / "benchmarks" / "concept_v09.jsonl"),
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument(
        "--jaccard-threshold", type=float, default=0.5,
        help="Fixed similarity baseline threshold; changing it is a new experiment",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    report = evaluate_concept_episodes(
        read_concept_episodes(args.dataset),
        split=args.split,
        jaccard_threshold=args.jaccard_threshold,
    )
    if args.summary:
        report = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps({
        "dataset": args.dataset,
        "report": report,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
