"""Matched one-shot vision comparison on the same frozen embeddings.

Purpose: determine whether ACM contributes anything beyond cosine
nearest-neighbor label binding when both systems receive the exact same
pretrained image representation and rejection threshold.

This module does NOT train OpenCLIP or learn visual features. With exactly
one support embedding per label, current CognitiveMemory is mathematically
a cosine nearest-neighbor memory; agreement with a direct NN baseline is
therefore expected, not a new capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

from .benchmark import ImageSample, validate_manifest
from .memory import CognitiveMemory
from .vision import ImageEncoder


Vector = tuple[float, ...]


def _vector(values: Iterable[float]) -> Vector:
    result = tuple(float(v) for v in values)
    if not result:
        raise ValueError("Embedding must be nonempty")
    return result


def cosine(a: Vector, b: Vector) -> float:
    if not a or len(a) != len(b):
        raise ValueError("Embeddings must be nonempty and have equal dimensions")
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@dataclass(frozen=True)
class EncodedSample:
    sample: ImageSample
    embedding: Vector


class DirectNearestNeighbor:
    """Transparent cosine 1-NN with the same threshold semantics as ACM."""

    def __init__(self) -> None:
        self._supports: list[tuple[str, Vector]] = []

    def add(self, label: str, embedding: Iterable[float]) -> None:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Nonempty label required")
        vector = _vector(embedding)
        if self._supports and len(vector) != len(self._supports[0][1]):
            raise ValueError("Embedding dimensions must remain consistent")
        self._supports.append((label.strip(), vector))

    def recognize(self, embedding: Iterable[float], *, threshold: float) -> dict:
        vector = _vector(embedding)
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be in [-1,1]")
        if not self._supports:
            return {"label": None, "score": None, "status": "unknown"}
        label, score = max(
            ((label, cosine(vector, support)) for label, support in self._supports),
            key=lambda item: item[1],
        )
        if score < threshold:
            return {"label": None, "score": round(score, 6), "status": "unknown"}
        return {"label": label, "score": round(score, 6), "status": "matched"}


def encode_samples(samples: list[ImageSample], encoder: ImageEncoder) -> list[EncodedSample]:
    """Encode each unique image exactly once for a matched comparison."""
    validate_manifest(samples)
    cache: dict[Path, Vector] = {}
    encoded = []
    for sample in samples:
        if sample.path not in cache:
            cache[sample.path] = _vector(encoder.embed(sample.path))
        encoded.append(EncodedSample(sample, cache[sample.path]))
    return encoded


def _metrics(rows: list[dict]) -> dict:
    known = [r for r in rows if r["split"] == "known"]
    unknown = [r for r in rows if r["split"] == "unknown"]
    return {
        "known_count": len(known),
        "known_accuracy": (
            sum(r["predicted"] == r["expected"] for r in known) / len(known)
            if known else None
        ),
        "unknown_count": len(unknown),
        "unknown_rejection_rate": (
            sum(r["predicted"] == "unknown" for r in unknown) / len(unknown)
            if unknown else None
        ),
    }


def evaluate_matched_one_shot(
    samples: list[ImageSample],
    encoder: ImageEncoder,
    *,
    threshold: float,
) -> dict:
    """Compare ACM against direct cosine 1-NN on *identical* embeddings.

    One support image per known label is required by validate_manifest().
    Arbitrary opaque labels are also evaluated to show that label naming itself
    contributes no pretrained semantic information.
    """
    encoded = encode_samples(samples, encoder)
    support = [row for row in encoded if row.sample.split == "support"]
    queries = [row for row in encoded if row.sample.split != "support"]

    acm = CognitiveMemory()
    nn = DirectNearestNeighbor()

    label_map = {
        row.sample.label: f"concept-{index:03d}"
        for index, row in enumerate(sorted(support, key=lambda r: r.sample.label), 1)
    }
    opaque_acm = CognitiveMemory()

    for row in support:
        acm.learn(row.sample.label, row.embedding, source=str(row.sample.path))
        nn.add(row.sample.label, row.embedding)
        opaque_acm.learn(
            label_map[row.sample.label], row.embedding, source=str(row.sample.path)
        )

    details = []
    agreement = 0
    score_agreement = 0
    opaque_behavior_agreement = 0
    for row in queries:
        expected = row.sample.label if row.sample.split == "known" else "unknown"
        acm_result = acm.recognize(row.embedding, threshold=threshold)
        nn_result = nn.recognize(row.embedding, threshold=threshold)
        opaque_result = opaque_acm.recognize(row.embedding, threshold=threshold)

        acm_label = acm_result["label"] or "unknown"
        nn_label = nn_result["label"] or "unknown"
        if row.sample.split == "known":
            opaque_expected = label_map[row.sample.label]
        else:
            opaque_expected = "unknown"
        opaque_label = opaque_result["label"] or "unknown"

        same_prediction = acm_label == nn_label and acm_result["status"] == nn_result["status"]
        same_score = acm_result["score"] == nn_result["score"]
        # Map opaque result back only for behavior-equivalence check.
        reverse = {v: k for k, v in label_map.items()}
        opaque_back = reverse.get(opaque_label, "unknown")
        same_opaque_behavior = opaque_back == acm_label

        agreement += int(same_prediction)
        score_agreement += int(same_score)
        opaque_behavior_agreement += int(same_opaque_behavior)

        details.append({
            "image": row.sample.path.name,
            "split": row.sample.split,
            "expected": expected,
            "acm_predicted": acm_label,
            "nn_predicted": nn_label,
            "score": acm_result["score"],
            "opaque_expected": opaque_expected,
            "opaque_acm_predicted": opaque_label,
            "acm_nn_same_prediction": same_prediction,
            "acm_nn_same_score": same_score,
            "opaque_label_behavior_same": same_opaque_behavior,
        })

    count = len(queries)
    acm_rows = [
        {"split": d["split"], "expected": d["expected"], "predicted": d["acm_predicted"]}
        for d in details
    ]
    nn_rows = [
        {"split": d["split"], "expected": d["expected"], "predicted": d["nn_predicted"]}
        for d in details
    ]
    opaque_rows = [
        {
            "split": d["split"],
            "expected": d["opaque_expected"],
            "predicted": d["opaque_acm_predicted"],
        }
        for d in details
    ]

    return {
        "support_count": len(support),
        "query_count": count,
        "threshold": threshold,
        "embedding_count": len(encoded),
        "acm": _metrics(acm_rows),
        "direct_nearest_neighbor": _metrics(nn_rows),
        "opaque_label_acm": _metrics(opaque_rows),
        "acm_nn_prediction_agreement": agreement / count,
        "acm_nn_score_agreement": score_agreement / count,
        "opaque_label_behavior_agreement": opaque_behavior_agreement / count,
        "details": details,
        "interpretation": (
            "With one support embedding per label, current ACM uses cosine "
            "nearest-neighbor retrieval. Equality with matched direct 1-NN is "
            "expected and means performance should be attributed primarily to "
            "the frozen representation plus threshold, not a new visual learner."
        ),
    }
