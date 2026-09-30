"""A transparent one-shot association baseline, NOT a trained visual model."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Feature vectors must be nonempty and have identical dimensions")
    norm_a = sqrt(sum(v*v for v in a))
    norm_b = sqrt(sum(v*v for v in b))
    if not norm_a or not norm_b:
        return 0.0
    return sum(x*y for x,y in zip(a,b)) / (norm_a * norm_b)


@dataclass(frozen=True)
class Observation:
    label: str
    features: tuple[float, ...]
    concepts: tuple[str, ...] = ()
    source: str = "unspecified"


@dataclass
class CognitiveMemory:
    """Store observations, concepts and explicit associations without retraining.

    The caller supplies numerical features (e.g. from a separately evaluated encoder).
    These toy features are not proof of visual comprehension or concept formation.
    """
    observations: list[Observation] = field(default_factory=list)
    links: dict[str, set[str]] = field(default_factory=dict)

    def learn(self, label: str, features: Iterable[float], *, concepts: Iterable[str] = (), source: str = "unspecified") -> None:
        vector = tuple(float(x) for x in features)
        if not label.strip():
            raise ValueError("A nonempty label is required")
        if not vector:
            raise ValueError("A nonempty feature vector is required")
        if self.observations and len(vector) != len(self.observations[0].features):
            raise ValueError("Feature dimensions must remain consistent")
        concepts_tuple = tuple(dict.fromkeys(concepts))
        self.observations.append(Observation(label, vector, concepts_tuple, source))
        self.links.setdefault(label, set()).update(concepts_tuple)

    def recognize(self, features: Iterable[float], *, threshold: float = 0.85) -> dict:
        vector = tuple(float(x) for x in features)
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be in [-1, 1]")
        if not self.observations:
            return {"label": None, "score": None, "status": "unknown"}
        candidates = [(obs, _cosine(vector, obs.features)) for obs in self.observations]
        winner, score = max(candidates, key=lambda result: result[1])
        if score < threshold:
            return {"label": None, "score": round(score, 6), "status": "unknown"}
        return {"label": winner.label, "score": round(score, 6), "status": "matched", "concepts": sorted(self.links.get(winner.label, set()))}

    def related_concepts(self, label: str) -> list[str]:
        return sorted(self.links.get(label, set()))
