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
    parser.add_argument("--think", choices=["default", "on", "off"], default="default",
                        help="Ollama thinking mode; default preserves prior behavior")
    parser.add_argument("--prompt", choices=["strict", "literal"], default="strict",
                        help="Local LLM prompt mode; default preserves v0.4 behavior")
    parser.add_argument("--summary", action="store_true", help="Hide individual predictions")
    parser.add_argument("--case", action="append", default=[],
                        help="Run only an individual case ID, e.g. --case t01; repeatable")
    args = parser.parse_args()

    if args.extractor == "rules":
        extractor = RuleExtractor()
    else:
        if not args.model:
            parser.error("--model is required for ollama or hybrid; no model is auto-installed")
        think = {"default": None, "on": True, "off": False}[args.think]
        local = OllamaExtractor(model=args.model, endpoint=args.endpoint,
                                think=think, prompt_mode=args.prompt)
        extractor = local if args.extractor == "ollama" else HybridExtractor(local)
    cases = read_cases(args.dataset)
    report = run_benchmark(cases, extractor, split=args.split,
                           case_ids=set(args.case) if args.case else None)
    if args.summary:
        report = {key: val for key, val in report.items() if key != "results"}
    print(json.dumps({
        "extractor": args.extractor,
        "model": args.model if args.extractor != "rules" else None,
        "think": args.think if args.extractor != "rules" else None,
        "prompt": args.prompt if args.extractor != "rules" else None,
        "dataset": str(args.dataset),
        "report": report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
