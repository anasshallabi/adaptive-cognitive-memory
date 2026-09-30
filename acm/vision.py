"""Optional real-image perception for ACM.

Uses a *frozen pretrained* OpenCLIP encoder. The association is new,
not the underlying ability to recognize images (acquired by pretraining).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .memory import CognitiveMemory


ImagePath = str | Path


class ImageEncoder(Protocol):
    """Minimal interface used to decouple tests from heavyweight model weights."""

    def embed(self, image: ImagePath) -> tuple[float, ...]: ...


class OpenCLIPEncoder:
    """Wrap a frozen OpenCLIP image encoder (optional dependencies).

    The first initialization may download pretrained checkpoint weights.
    Exact model, pretrained tag, device and versions should be logged.
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
    ) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "The vision adapter requires torch, Pillow and open_clip_torch. "
                "See docs/VISION.md for installation."
            ) from exc

        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.model_name = model_name
        self.pretrained = pretrained
        self._torch = torch
        self._image_class = Image
        self._open_clip = open_clip
        model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = model.to(self.device).eval()
        # Ensure all parameters are frozen, even if an external caller reuses this model.
        for param in self._model.parameters():
            param.requires_grad_(False)

    def embed(self, image: ImagePath) -> tuple[float, ...]:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with self._image_class.open(path) as opened:
            rgb = opened.convert("RGB")
            try:
                batch = self._preprocess(rgb).unsqueeze(0).to(self.device)
            finally:
                rgb.close()
        with self._torch.inference_mode():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return tuple(float(v) for v in features.squeeze(0).cpu().tolist())

    def metadata(self) -> dict[str, str]:
        return {
            "encoder": "OpenCLIP",
            "model": self.model_name,
            "pretrained": self.pretrained,
            "device": self.device,
            "torch_version": str(self._torch.__version__),
            "open_clip_version": str(getattr(self._open_clip, "__version__", "unknown")),
        }


@dataclass
class VisionMemory:
    """A label-memory coupled to any frozen image encoder.

    This is still cosine-nearest-neighbor retrieval, not autonomous understanding.
    """

    encoder: ImageEncoder
    memory: CognitiveMemory = field(default_factory=CognitiveMemory)

    def learn_image(
        self,
        image: ImagePath,
        label: str,
        *,
        concepts: Iterable[str] = (),
    ) -> None:
        self.memory.learn(
            label,
            self.encoder.embed(image),
            concepts=concepts,
            source=str(image),
        )

    def recognize_image(self, image: ImagePath, *, threshold: float = 0.85) -> dict:
        return self.memory.recognize(self.encoder.embed(image), threshold=threshold)
