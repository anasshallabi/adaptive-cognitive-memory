# Adaptive Cognitive Memory (ACM)

**Status: research prototype / baseline 0.1 — no claim of human-like learning or a breakthrough.**

[Documentation française](docs/README.fr.md) · [Research plan](docs/RESEARCH.md) · [Protocol](docs/EVALUATION.md) · [Architecture](docs/ARCHITECTURE.md)

ACM investigates whether a system can (1) record new information from a single exposure and (2) reuse it through explicit associations, without globally retraining its parameters. Inspired by the question: *Why can a person read the name of a new car brand once and later connect it to a different vehicle?*

## What exists today

A minimal **CPU-only, standard-library Python baseline** that stores labeled numerical vectors, links labels to provided concepts, and retrieves the nearest observation by cosine similarity. The sample vectors are **synthetic**; there is **no image encoder, no automatic discovery of concepts, no autonomous learning, no persistence, and no LLM** in version 0.1. This demonstrates a test harness, not the research hypothesis.

## Quickstart

Python 3.10+ required, no dependencies:

```bash
python -m unittest discover -s tests -v
python -m examples.one_shot
```

## Research questions

- RQ1: How much information can be acquired after one exposure, conditional on existing pretrained representations?
- RQ2: Can learned knowledge transfer to genuinely novel observations without manual concept labels?
- RQ3: Can the system learn continuously while limiting forgetting, false matches and compute usage?

We will compare fairly against nearest-neighbor and fixed-encoder baselines; details in `docs/EVALUATION.md`. No improvements are claimed before controlled experiments.

## Roadmap

- [x] Minimal associative store and transparent tests
- [ ] Real image-embedding adapter with explicitly documented pretrained dependencies
- [ ] Public data protocol with train/test identity separation and leakage checks
- [ ] Open-set calibration, precision/recall, false acceptance and forgetting benchmarks
- [ ] Hypothesis-driven experiments in incremental concept formation

## License / contributions

MIT licensed. See [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Never commit real private images, personal data, tokens, or copyrighted datasets without permission.
