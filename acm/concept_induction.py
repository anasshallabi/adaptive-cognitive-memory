"""Transparent toy study of one-positive-example concept induction.

This module deliberately separates:
- an arbitrary new *label* (easy to bind in one shot), from
- identifying the *rule* that defines the concept (often underdetermined).

Objects are sets of opaque binary features (f01, f02, ...). A hidden concept
is a conjunction of one or two features. The learner never receives the hidden
rule; it receives one positive support example and, optionally, contrastive
negative examples.

This is a formal synthetic diagnostic, NOT a model of natural vision/language
and NOT evidence of human-like concept formation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable


FeatureSet = frozenset[str]
Hypothesis = frozenset[str]


def _features(values: Iterable[str]) -> FeatureSet:
    result = frozenset(v.strip() for v in values if isinstance(v, str) and v.strip())
    if not result:
        raise ValueError("Feature set must be nonempty")
    return result


@dataclass(frozen=True)
class ConceptEpisode:
    episode_id: str
    split: str
    label: str
    support: FeatureSet
    hidden_rule: Hypothesis
    contrast_negatives: tuple[FeatureSet, ...]
    queries: tuple[FeatureSet, ...]

    def expected(self, query: FeatureSet) -> bool:
        return self.hidden_rule.issubset(query)


def read_concept_episodes(path: str | Path) -> list[ConceptEpisode]:
    rows: list[ConceptEpisode] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episode = ConceptEpisode(
                    episode_id=str(row["id"]).strip(),
                    split=str(row["split"]).strip(),
                    label=str(row["label"]).strip(),
                    support=_features(row["support"]),
                    hidden_rule=_features(row["hidden_rule"]),
                    contrast_negatives=tuple(_features(x) for x in row["contrast_negatives"]),
                    queries=tuple(_features(x) for x in row["queries"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed concept episode at line {line_no}") from exc
            rows.append(episode)
    validate_concept_episodes(rows)
    return rows


def validate_concept_episodes(episodes: list[ConceptEpisode]) -> None:
    if not episodes:
        raise ValueError("At least one concept episode is required")
    if {e.split for e in episodes} != {"dev", "test"}:
        raise ValueError("Concept benchmark requires dev and test splits")
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode ID")
    if len({e.label for e in episodes}) != len(episodes):
        raise ValueError("Labels must be arbitrary and unique per episode")
    for episode in episodes:
        if not episode.episode_id or not episode.label:
            raise ValueError("Episode id and label are required")
        if len(episode.hidden_rule) not in (1, 2):
            raise ValueError("Hidden rule must use one or two features")
        if not episode.hidden_rule.issubset(episode.support):
            raise ValueError(f"Hidden rule absent from positive support: {episode.episode_id}")
        if not episode.contrast_negatives or not episode.queries:
            raise ValueError("Each episode requires contrasts and queries")
        if any(episode.hidden_rule.issubset(n) for n in episode.contrast_negatives):
            raise ValueError(f"Contrast labeled negative actually matches hidden rule: {episode.episode_id}")
        labels = [episode.expected(q) for q in episode.queries]
        if not any(labels) or all(labels):
            raise ValueError("Queries must contain both positives and negatives")


class VersionSpaceLearner:
    """All 1-2 feature conjunctions compatible with observed labels.

    The learner uses only the positive support and optional negative contrasts.
    It does not access the hidden rule or query labels.
    """

    def __init__(self, support: Iterable[str], *, max_rule_size: int = 2):
        self.support = _features(support)
        if max_rule_size < 1:
            raise ValueError("max_rule_size must be positive")
        self.max_rule_size = min(max_rule_size, len(self.support))
        self._negatives: list[FeatureSet] = []
        self._hypotheses = self._generate()

    def _generate(self) -> set[Hypothesis]:
        ordered = sorted(self.support)
        return {
            frozenset(parts)
            for size in range(1, self.max_rule_size + 1)
            for parts in combinations(ordered, size)
        }

    @property
    def hypotheses(self) -> set[Hypothesis]:
        return set(self._hypotheses)

    def add_negative(self, features: Iterable[str]) -> None:
        negative = _features(features)
        self._negatives.append(negative)
        self._hypotheses = {
            h for h in self._hypotheses
            if not h.issubset(negative)
        }
        if not self._hypotheses:
            raise ValueError("Evidence eliminated every candidate concept")

    def predict(self, features: Iterable[str]) -> str:
        """Return positive, negative, or unknown under all surviving rules."""
        query = _features(features)
        predictions = [h.issubset(query) for h in self._hypotheses]
        if predictions and all(predictions):
            return "positive"
        if predictions and not any(predictions):
            return "negative"
        return "unknown"


class ExactExemplarBaseline:
    """Simple episodic memorization: only an identical feature set is positive."""

    def __init__(self, support: Iterable[str]):
        self.support = _features(support)

    def predict(self, features: Iterable[str]) -> bool:
        return _features(features) == self.support


class JaccardBaseline:
    """Fixed similarity baseline; threshold is predeclared, not learned per episode."""

    def __init__(self, support: Iterable[str], *, threshold: float = 0.5):
        self.support = _features(support)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        self.threshold = threshold

    def score(self, features: Iterable[str]) -> float:
        query = _features(features)
        return len(self.support & query) / len(self.support | query)

    def predict(self, features: Iterable[str]) -> bool:
        return self.score(features) >= self.threshold


def _binary_accuracy(predictions: list[bool], expected: list[bool]) -> float:
    return sum(a == b for a, b in zip(predictions, expected)) / len(expected)


def evaluate_concept_episodes(
    episodes: list[ConceptEpisode], *, split: str = "test",
    jaccard_threshold: float = 0.5,
) -> dict:
    validate_concept_episodes(episodes)
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    selected = [e for e in episodes if e.split == split]
    results = []
    totals = {
        "queries": 0,
        "exact_correct": 0,
        "jaccard_correct": 0,
        "oracle_correct": 0,
        "one_shot_certain_correct": 0,
        "one_shot_decided": 0,
        "contrast_certain_correct": 0,
        "contrast_decided": 0,
        "unique_after_one_shot": 0,
        "unique_after_contrasts": 0,
    }
    for episode in selected:
        expected = [episode.expected(q) for q in episode.queries]
        exact = ExactExemplarBaseline(episode.support)
        jaccard = JaccardBaseline(episode.support, threshold=jaccard_threshold)
        exact_pred = [exact.predict(q) for q in episode.queries]
        jaccard_pred = [jaccard.predict(q) for q in episode.queries]

        one = VersionSpaceLearner(episode.support)
        one_count = len(one.hypotheses)
        one_pred = [one.predict(q) for q in episode.queries]

        contrasted = VersionSpaceLearner(episode.support)
        for negative in episode.contrast_negatives:
            contrasted.add_negative(negative)
        contrast_count = len(contrasted.hypotheses)
        contrast_pred = [contrasted.predict(q) for q in episode.queries]

        one_decided = [p != "unknown" for p in one_pred]
        contrast_decided = [p != "unknown" for p in contrast_pred]
        one_correct = sum(
            decided and ((pred == "positive") == truth)
            for decided, pred, truth in zip(one_decided, one_pred, expected)
        )
        contrast_correct = sum(
            decided and ((pred == "positive") == truth)
            for decided, pred, truth in zip(contrast_decided, contrast_pred, expected)
        )

        totals["queries"] += len(expected)
        totals["exact_correct"] += sum(a == b for a, b in zip(exact_pred, expected))
        totals["jaccard_correct"] += sum(a == b for a, b in zip(jaccard_pred, expected))
        totals["oracle_correct"] += len(expected)  # evaluator uses frozen hidden rule
        totals["one_shot_certain_correct"] += one_correct
        totals["one_shot_decided"] += sum(one_decided)
        totals["contrast_certain_correct"] += contrast_correct
        totals["contrast_decided"] += sum(contrast_decided)
        totals["unique_after_one_shot"] += int(one_count == 1)
        totals["unique_after_contrasts"] += int(contrast_count == 1)

        results.append({
            "id": episode.episode_id,
            "label": episode.label,
            "support_features": sorted(episode.support),
            "hidden_rule_size": len(episode.hidden_rule),
            "one_shot_candidate_rules": one_count,
            "after_contrast_candidate_rules": contrast_count,
            "exact_accuracy": _binary_accuracy(exact_pred, expected),
            "jaccard_accuracy": _binary_accuracy(jaccard_pred, expected),
            "one_shot_decided": sum(one_decided),
            "one_shot_correct_when_decided": one_correct,
            "after_contrast_decided": sum(contrast_decided),
            "after_contrast_correct_when_decided": contrast_correct,
            "query_count": len(expected),
        })

    q = totals["queries"]
    one_d = totals["one_shot_decided"]
    contrast_d = totals["contrast_decided"]
    return {
        "split": split,
        "episodes": len(selected),
        "queries": q,
        "jaccard_threshold": jaccard_threshold,
        "exact_exemplar_accuracy": totals["exact_correct"] / q,
        "jaccard_accuracy": totals["jaccard_correct"] / q,
        "oracle_rule_accuracy": totals["oracle_correct"] / q,
        "one_shot_unique_rule_rate": totals["unique_after_one_shot"] / len(selected),
        "one_shot_certain_coverage": one_d / q,
        "one_shot_accuracy_when_certain": (
            totals["one_shot_certain_correct"] / one_d if one_d else None
        ),
        "after_contrast_unique_rule_rate": (
            totals["unique_after_contrasts"] / len(selected)
        ),
        "after_contrast_certain_coverage": contrast_d / q,
        "after_contrast_accuracy_when_certain": (
            totals["contrast_certain_correct"] / contrast_d if contrast_d else None
        ),
        "results": results,
        "warning": (
            "Synthetic conjunction task with opaque features. It measures "
            "underdetermination and simple baselines, not natural concept learning."
        ),
    }
