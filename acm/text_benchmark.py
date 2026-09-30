"""Small audit-friendly French/English fact extraction benchmark.

This is a *synthetic challenge set*, not representative of real-world language.
No tuning/training with held-out test cases is done by this harness.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .text_extraction import FactExtractor
from .text_memory import Claim


_ALLOWED_PREDICATES = frozenset({"is_a", "makes", "has", "uses"})


@dataclass(frozen=True)
class TextCase:
    case_id: str
    split: str
    pattern_family: str
    sentence: str
    expected: dict[str, Any] | None


def _normalize(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def read_cases(path: str | Path) -> list[TextCase]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSON object required")
                expected = row["expected"]
                if expected is not None:
                    keys = {"subject", "predicate", "object", "positive"}
                    if set(expected) != keys:
                        raise ValueError("Expected claim must have exactly four fields")
                    if expected["predicate"] not in _ALLOWED_PREDICATES:
                        raise ValueError("Unsupported expected predicate")
                    if not isinstance(expected["positive"], bool):
                        raise ValueError("Expected polarity must be boolean")
                    if any(not isinstance(expected[k], str) or not expected[k].strip()
                           for k in ("subject", "predicate", "object")):
                        raise ValueError("Expected claim spans must be nonempty strings")
                rows.append(TextCase(
                    case_id=row["id"], split=row["split"],
                    pattern_family=row["pattern_family"],
                    sentence=row["sentence"], expected=expected,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed benchmark row {index}") from exc
    validate_cases(rows)
    return rows


def validate_cases(cases: list[TextCase]) -> None:
    if not cases:
        raise ValueError("Benchmark cannot be empty")
    if {c.split for c in cases} != {"dev", "test"}:
        raise ValueError("Benchmark requires both dev and held-out test")
    if any(not all((c.case_id.strip(), c.pattern_family.strip(), c.sentence.strip()))
           for c in cases):
        raise ValueError("Case id, template family and text required")
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate benchmark IDs")
    if len({_normalize(c.sentence) for c in cases}) != len(cases):
        raise ValueError("Duplicate benchmark sentences")
    dev = {c.pattern_family for c in cases if c.split == "dev"}
    test = {c.pattern_family for c in cases if c.split == "test"}
    overlap = dev & test
    if overlap:
        raise ValueError(f"Template family leakage between dev/test: {sorted(overlap)}")
    for case in cases:
        if case.expected is not None and not all(
            _normalize(case.expected[k]) in _normalize(case.sentence)
            for k in ("subject", "object")
        ):
            raise ValueError(f"Expected span not in text: {case.case_id}")


def _matches(claim: Claim | None, expected: dict | None) -> bool:
    if claim is None:
        return expected is None
    if expected is None:
        return False
    return (
        _normalize(claim.subject) == _normalize(expected["subject"])
        and claim.predicate == expected["predicate"]
        and _normalize(claim.object) == _normalize(expected["object"])
        and claim.positive is expected["positive"]
    )


def run_benchmark(cases: list[TextCase], extractor: FactExtractor, *, split: str = "test") -> dict:
    validate_cases(cases)
    if split not in {"dev", "test"}:
        raise ValueError("Choose a single split; do not mix validation with test")
    subset = [c for c in cases if c.split == split]
    tp = fp = fn = correct = 0
    unknown_total = unknown_correct = 0
    errors = []
    output = []
    for case in subset:
        start = perf_counter()
        try:
            claim = extractor.extract(case.sentence, source=f"benchmark:{case.case_id}")
            error = None
        except (ValueError, RuntimeError) as exc:
            claim, error = None, type(exc).__name__
            errors.append({"id": case.case_id, "type": error})
        elapsed_ms = (perf_counter() - start) * 1000
        right = error is None and _matches(claim, case.expected)
        correct += int(right)
        if case.expected is None:
            unknown_total += 1
            unknown_correct += int(claim is None and error is None)
        else:
            fn += int(not right)
        if claim is not None:
            if right:
                tp += 1
            else:
                fp += 1
        output.append({
            "id": case.case_id, "sentence": case.sentence,
            "expected": case.expected,
            "predicted": None if claim is None else {
                "subject": claim.subject, "predicate": claim.predicate,
                "object": claim.object, "positive": claim.positive,
            },
            "exact": right, "error": error,
            "latency_ms": round(elapsed_ms, 3),
        })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "split": split,
        "count": len(subset),
        "exact_accuracy": correct / len(subset),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "abstention_on_nonfacts": unknown_correct / unknown_total if unknown_total else None,
        "error_count": len(errors),
        "total_latency_ms": round(sum(p["latency_ms"] for p in output), 3),
        "results": output,
        "warning": "Hand-written synthetic set. No proof of general language understanding.",
    }
