# Adaptive Cognitive Memory (ACM)

**Research prototype v0.3 — rapid association baseline; no claim of human-level intelligence or novel general learning.**

[Français](docs/README.fr.md) · [Text learning](docs/TEXT.md) · [Vision](docs/VISION.md) · [Research](docs/RESEARCH.md) · [Evaluation](docs/EVALUATION.md)

ACM investigates whether observations can be bound to memory after one exposure and linked to existing knowledge without globally retraining model weights.

## Implemented

- **v0.1**: In-memory numerical vector associations with cosine nearest neighbor and explicit concept labels.
- **v0.2**: Optional frozen pretrained OpenCLIP image encoder, real-image CLI, benchmark CSV checks. **Actual photos have not yet been benchmarked**.
- **v0.3**: Dependency-free, deliberately limited **French/English controlled-text parser**; subject–predicate–object fact store with source tracking; bounded `is_a` inference; positive, negative, conflicting or unknown evidence; simple label-based bridge to image memory. This is **symbolic graph traversal, not general-language comprehension or autonomous concept discovery**.

## Quickstart (Python 3.10+; no dependencies for text and unit tests)

```bash
python -m unittest discover -s tests -v
python -m examples.text_one_shot
python -m examples.one_shot
```

For image experiments install PyTorch, Pillow and `open_clip_torch` per [docs/VISION.md](docs/VISION.md). First execution may download model weights.

## Research standards

We distinguish instant storage of a novel **fact** from learning a novel **concept**, inference from truth, and using a pretrained perceptual encoder from learning perception from scratch. The text parser accepts only documented sentence structures and never evaluates credibility. No cost savings or real-world accuracy gains have been demonstrated.

## Roadmap

- [x] Numeric association and source-free vector memory
- [x] Optional real-image interface and split guards
- [x] Controlled-language memory with provenance, explicit conflicts, and an image/text label bridge
- [ ] Reproducible empirical benchmark on licensed, held-out real photos
- [ ] Independent text generalization test beyond known grammar
- [ ] Durable memory, revisions, provenance confidence evaluation
- [ ] Test truly new concept induction against matched symbolic and neural baselines

MIT licensed — see [CONTRIBUTING.md](CONTRIBUTING.md).
