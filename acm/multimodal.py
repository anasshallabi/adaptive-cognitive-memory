"""Minimal v0.3 bridge: exact shared labels between text and image memory.

The shared label is supplied by a person; it is NOT learned visual grounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .text_memory import TextMemory
from .vision import ImagePath, VisionMemory


@dataclass
class MultimodalMemory:
    vision: VisionMemory
    text: TextMemory = field(default_factory=TextMemory)

    def learn_image(self, image: ImagePath, label: str, *, concepts: Iterable[str] = ()) -> None:
        self.vision.learn_image(image, label, concepts=concepts)

    def learn_text(self, sentence: str, *, source: str = "user") -> dict:
        return self.text.learn_text(sentence, source=source)

    def inspect_image(self, image: ImagePath, *, threshold: float = 0.85) -> dict:
        result = self.vision.recognize_image(image, threshold=threshold)
        # Never attach knowledge if recognition rejected.
        facts = self.text.about(result["label"]) if result["status"] == "matched" else []
        return {"recognition": result, "linked_text_claims": facts}
