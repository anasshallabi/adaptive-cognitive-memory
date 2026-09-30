"""Small one-shot image benchmark. Evaluation classes must be curated externally.

No neural packages needed to run these controls with a test encoder.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .vision import ImageEncoder, VisionMemory


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str
    split: str  # support, known, unknown
    vehicle_id: str
    capture_group: str


def load_manifest(csv_path: str | Path) -> list[ImageSample]:
    """Paths are relative to the manifest directory, not the current shell dir."""
    manifest = Path(csv_path).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "split", "vehicle_id", "capture_group"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        samples = []
        for number, row in enumerate(reader, start=2):
            if any(not (row.get(key) or "").strip() for key in required):
                raise ValueError(f"Row {number} has empty required fields")
            split = row["split"].strip()
            if split not in {"support", "known", "unknown"}:
                raise ValueError(f"Row {number}: invalid split {split!r}")
            image_path = (manifest.parent / row["path"].strip()).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Row {number}: image not found: {image_path}")
            samples.append(ImageSample(
                image_path, row["label"].strip(), split,
                row["vehicle_id"].strip(), row["capture_group"].strip()
            ))
    return samples


def validate_manifest(samples: list[ImageSample]) -> None:
    """Prevent obvious leakage and guarantee a true one-support example per label."""
    if not samples:
        raise ValueError("Manifest has no examples")
    support = [s for s in samples if s.split == "support"]
    known = [s for s in samples if s.split == "known"]
    unknown = [s for s in samples if s.split == "unknown"]
    if not support or not known or not unknown:
        raise ValueError("Manifest needs support, known and unknown rows")
    labels = [s.label for s in support]
    if len(labels) != len(set(labels)):
        raise ValueError("Exactly one support image is allowed per label")
    support_labels = set(labels)
    if any(s.label not in support_labels for s in known):
        raise ValueError("Known queries must be labels learned in support")
    if any(s.label in support_labels for s in unknown):
        raise ValueError("Unknown queries must use disjoint labels")
    if len({str(s.path) for s in samples}) != len(samples):
        raise ValueError("Image paths must not be duplicated")
    for attr in ("vehicle_id", "capture_group"):
        groups: dict[str, set[str]] = {}
        for sample in samples:
            groups.setdefault(getattr(sample, attr), set()).add(sample.split)
        leaked = sorted(k for k, splits in groups.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"{attr} reused between splits (possible leakage): {leaked[:5]}")


def evaluate_one_shot(
    samples: list[ImageSample],
    encoder: ImageEncoder,
    *,
    threshold: float,
) -> dict:
    """Return observed results; no model accuracy is implied before actual data runs."""
    validate_manifest(samples)
    memory = VisionMemory(encoder)
    for sample in samples:
        if sample.split == "support":
            memory.learn_image(sample.path, sample.label)
    known_correct = known_total = unknown_rejected = unknown_total = 0
    details = []
    for sample in samples:
        if sample.split == "support":
            continue
        result = memory.recognize_image(sample.path, threshold=threshold)
        if sample.split == "known":
            known_total += 1
            known_correct += result["label"] == sample.label
        else:
            unknown_total += 1
            unknown_rejected += result["status"] == "unknown"
        details.append({
            "image": sample.path.name,
            "expected": sample.label if sample.split == "known" else "unknown",
            "predicted": result["label"] if result["label"] is not None else "unknown",
            "score": result["score"],
            "split": sample.split,
        })
    return {
        "support_count": len(memory.memory.observations),
        "known_count": known_total,
        "known_accuracy": known_correct / known_total,
        "unknown_count": unknown_total,
        "unknown_rejection_rate": unknown_rejected / unknown_total,
        "threshold": threshold,
        "predictions": details,
    }
