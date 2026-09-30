# Adaptive Cognitive Memory (ACM)

**Status: experimental v0.2 baseline. No breakthrough, human-like concept formation, real-image accuracy, or compute savings claimed.**

[Documentation française](docs/README.fr.md) · [Real-image guide](docs/VISION.md) · [Research](docs/RESEARCH.md) · [Evaluation](docs/EVALUATION.md) · [Architecture](docs/ARCHITECTURE.md)

ACM studies (1) one-exposure association and (2) reuse through existing representations without retraining the complete model. Inspired by learning a new car marque from seeing its name once.

## Implemented
- v0.1 pure Python vector-based associative memory and cosine nearest-neighbor search with manually supplied concept links.
- v0.2 optional **frozen pretrained OpenCLIP** image encoder, `VisionMemory.learn_image()`, `recognize_image()`, real-photo CLI, and an initial benchmark runner for one support image per known brand plus unknown brands with split leakage checks.
- **15 dependency-free automated tests** using *synthetic fake encoders*, **not** real-image accuracy measurements.

## Run
```bash
python -m unittest discover -s tests -v
python -m examples.one_shot
```

For real-image installation and PowerShell commands, read [docs/VISION.md](docs/VISION.md). Quick example after installing PyTorch, Pillow and `open_clip_torch`:

```bash
python -m examples.vision_one_shot --support data/toyota_a.jpg --label Toyota --query data/toyota_b.jpg data/honda.jpg --device auto
```

The first execution may download pretrained weights. It uses a **non-calibrated example threshold** (not a probability). No local photos are sent to a hosted inference API by this program. Images and weights are not included in the repo.

## Research roadmap
- [x] Inspectable synthetic vector-memory baseline
- [x] Optional frozen image encoder and first-pass dataset split checks
- [ ] **Run and publish real-image measurements** and calibrate unknown rejection
- [ ] Test logo/text leakage versus genuine unseen-body generalization
- [ ] Add persistence, provenance, concept induction and continual-learning experiments
- [ ] Quantify accuracy, compute, VRAM and latency against identical-backbone baselines

This is currently **frozen-encoder nearest-neighbor**, not a novel learner. Record negative results and statistical uncertainty. MIT license; contributions welcome via [CONTRIBUTING.md](CONTRIBUTING.md).
