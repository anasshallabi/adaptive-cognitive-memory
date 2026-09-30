# ACM v0.2 — Evaluation protocol

## Engineering stage

Run `python -m unittest discover -s tests -v` without installing neural dependencies. The deterministic tests use synthetic vectors/fake encoders and establish code behavior only — **not** accuracy on real photographs.

## Real image stage (runner exists; licensed dataset/results pending)

1. Obtain licensed test photos; document provenance, exact physical vehicle identity and capture sessions.
2. Freeze a documented OpenCLIP model and preprocessing, logging weight name, versions, GPU/CPU.
3. Give exactly **one labeled support photo per known brand**, then test different cars, camera angles and sessions.
4. Include distinct unseen brands; tune similarity rejection threshold only on **separate validation** brands, never final test data.
5. Evaluate with visible logos/text vs masked logos and body appearance separately, to detect simple text recognition.
6. Measure known-class top-1, false-accept/rejection of unknowns, predictions and failure cases; subsequently macro-F1, AUROC, confidence intervals, accuracy after continued learning, RAM/VRAM and latency.
7. Compare fairly with frozen-encoder cosine nearest-neighbor (already this implementation); maintain the same model/data/settings for any future architectural enhancement.

See [VISION.md](VISION.md) for manifest format and runnable commands. Manifest checks prevent repeated paths, vehicle IDs and capture groups across splits, but cannot guarantee labels are correct or the pretrained model never saw the photos.

**No real-image measurement or scientific breakthrough is claimed.** Publish failures and uncertainties.
