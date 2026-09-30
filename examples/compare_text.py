"""Compare v0.3 rules to optional local Ollama on curated, held-out examples.

No model is downloaded or API is called by default.

python -m examples.compare_text --extractor rules --split test
python -m examples.compare_text --extractor hybrid --model gemma3:4b --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acm.text_benchmark import read_cases, run_benchmark
from acm.text_extraction import HybridExtractor, OllamaExtractor, RuleExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.4 controlled vs flexible text extraction")
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parents[1] / "benchmarks" / "text_v04.jsonl"))
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--extractor", choices=["rules", "ollama", "hybrid"], default="rules")
    parser.add_argument("--model", help="Existing local Ollama model name (required for ollama or hybrid)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--summary", action="store_true", help="Hide individual predictions")
    args = parser.parse_args()

    if args.extractor == "rules":
        extractor = RuleExtractor()
    else:
        if not args.model:
            parser.error("--model is required for ollama or hybrid; no model is auto-installed")
        local = OllamaExtractor(model=args.model, endpoint=args.endpoint)
        extractor = local if args.extractor == "ollama" else HybridExtractor(local)
    cases = read_cases(args.dataset)
    report = run_benchmark(cases, extractor, split=args.split)
    if args.summary:
        report = {key: val for key, val in report.items() if key != "results"}
    print(json.dumps({
        "extractor": args.extractor,
        "model": args.model if args.extractor != "rules" else None,
        "dataset": str(args.dataset),
        "report": report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
