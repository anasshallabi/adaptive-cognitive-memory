"""Small local probe to separate direct assertions from attributed/hedged claims.

No accuracy score is reported: these contrasts were designed AFTER inspecting
the t03 failure, and they are NOT an unseen scientific test.

Run:
    python -m examples.probe_text --model qwen3:14b --case direct --case attributed
    python -m examples.probe_text --model qwen3:14b --case hedged
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

from acm.text_extraction import FactExtractor, OllamaExtractor, RuleExtractor


PROBES = {
    "direct": "Zentra is a car brand.",
    "attributed": "Zentra is considered a car brand.",
    "hedged": "Zentra might be a car brand.",
    "reported": "According to a reviewer, Zentra is a car brand.",
    "past": "Zentra used to be a car brand.",
    "fr_direct": "Zentra est une marque automobile.",
    "fr_attributed": "Zentra est considéré comme une marque automobile.",
    "fr_hedged": "Zentra pourrait être une marque automobile.",
}


@dataclass(frozen=True)
class ProbeResult:
    sentence: str
    rule_result: dict | None
    model_result: dict | None
    model_status: str
    model_latency_ms: float | None
    model_error: str | None
    model_telemetry: dict | None

    def as_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "rule_result": self.rule_result,
            "model_result": self.model_result,
            "model_status": self.model_status,
            "model_latency_ms": self.model_latency_ms,
            "model_error": self.model_error,
            "model_telemetry": self.model_telemetry,
        }


def _claim_dict(claim) -> dict | None:
    if claim is None:
        return None
    return {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
        "positive": claim.positive,
        "source": claim.source,
    }


def run_probes(
    sentences: Iterable[str],
    *,
    model: FactExtractor | None = None,
    rules: FactExtractor | None = None,
) -> list[dict]:
    """Compare raw rule parsing and *model-only* extraction independently.

    The model is NOT a rules+fallback hybrid here, so it always sees the sentence.
    This is only diagnosis; outcomes are not rated as factual truth.
    """
    rules = rules if rules is not None else RuleExtractor()
    report = []
    for index, sentence in enumerate(sentences, 1):
        source = f"probe:{index}"
        rule = rules.extract(sentence, source=source)
        model_claim = None
        latency_ms = None
        error = None
        status = "not_run"
        if model is not None:
            start = perf_counter()
            try:
                model_claim = model.extract(sentence, source=source)
                status = "extracted" if model_claim is not None else "abstained"
            except (ValueError, RuntimeError) as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((perf_counter() - start) * 1000, 3)
        report.append(ProbeResult(
            sentence=sentence,
            rule_result=_claim_dict(rule),
            model_result=_claim_dict(model_claim),
            model_status=status,
            model_latency_ms=latency_ms,
            model_error=error,
            model_telemetry=(dict(getattr(model, "last_metadata", {})) if model is not None else None),
        ).as_dict())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local FR/EN epistemic-modality probe (diagnostic, not benchmark)"
    )
    parser.add_argument("--model", help="Existing Ollama model; omitted means rules only")
    parser.add_argument("--think", choices=["default", "on", "off"], default="default",
                        help="Ollama thinking mode: default/explicit on/explicit off; off if supported")
    parser.add_argument("--prompt", choices=["strict", "literal"], default="strict",
                        help="Diagnostic prompt mode; strict preserves baseline behavior")
    parser.add_argument("--case", choices=sorted(PROBES), action="append", default=[],
                        help="Predefined diagnostic case (repeatable)")
    parser.add_argument("--sentence", action="append", default=[],
                        help="Additional sentence to probe (repeatable)")
    args = parser.parse_args()
    sentences = [PROBES[c] for c in args.case] + args.sentence
    if not sentences:
        parser.error("Provide --case or --sentence")
    if args.model:
        thinking = {"default": None, "on": True, "off": False}[args.think]
        extractor = OllamaExtractor(model=args.model, think=thinking,
                                   prompt_mode=args.prompt)
    else:
        extractor = None
    print(json.dumps({
        "model": args.model,
        "think": args.think,
        "prompt": args.prompt,
        "results": run_probes(sentences, model=extractor),
        "interpretation": (
            "Post-hoc diagnostic only. 'Considered', 'might', 'according to', and "
            "'used to' do not logically entail an unconditional present-day is_a fact. "
            "No generalization or accuracy conclusions can be drawn from these probes."
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
