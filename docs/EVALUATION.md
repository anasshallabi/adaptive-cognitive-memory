# Reproducible evaluation plan

## Stage 0 — engineering smoke tests

Run `python -m unittest discover -s tests -v`. Assertions only demonstrate deterministic behavior on fabricated vector examples. They **do not** measure model accuracy on real-world images.

## Stage 1 — image tasks (not implemented)

1. Specify dataset and licenses; group images by physical vehicle/source to prevent near-duplicate leakage.
2. Use a frozen, documented pretrained image encoder and record weights/version/hash and preprocessing.
3. For each episode, show exactly one labeled support image for each new brand. Hold out different photos, vehicles, viewpoints and backgrounds for evaluation.
4. Include disjoint unknown brands to test rejection.
5. Tune thresholds only on validation brands, never on final test brands.
6. Repeat episodes across seeds and report confidence intervals.

## Measurements

Top-1 accuracy on known brands, macro-F1, unknown-class AUROC / false accept rate, per-class confusion matrix, accuracy after subsequent learning, RAM/VRAM peak, latency and energy where measurable; report hardware and software versions.

## Baselines

(A) frozen encoder + one-shot cosine nearest neighbor; (B) nearest class prototype; (C) random or most-frequent prediction as sanity check; (D) candidate ACM memory enhancements with precisely the same encoder and evaluation data. Separate encoder pretraining cost from incremental learning cost.

## Acceptance / honesty

Do not announce scientific progress solely from a functional program. Publish full configuration, negative results, statistical uncertainty, dataset provenance, and test-time compute. Human-like learning is **not** an established outcome.
