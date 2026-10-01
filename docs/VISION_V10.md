# ACM v0.10 — Matched real-image one-shot baseline

## Research question

When ACM recognizes a new visual label after seeing one image, is ACM adding
anything beyond the **frozen pretrained representation + cosine
nearest-neighbor**?

v0.10 is designed so that both systems receive the **exact same image
embeddings, one support image per label, and the exact same rejection
threshold**. If their predictions are identical, that is the expected result:
current ACM visual memory is a cosine nearest-neighbor label-binding mechanism.

This experiment therefore tries to *remove credit* from ACM that properly
belongs to the pretrained encoder.

## What is being compared

1. **ACM CognitiveMemory**
   - one support embedding per known label;
   - cosine similarity;
   - fixed rejection threshold.

2. **Direct cosine 1-nearest-neighbor**
   - same support embeddings;
   - same query embeddings;
   - same threshold.

3. **Opaque-label ACM control**
   - support labels are replaced internally by `concept-001`,
     `concept-002`, etc.;
   - embeddings and threshold are unchanged.

The opaque-label control tests whether the *name* of the label affects the
memory stage. It does **not** remove semantic information already encoded in
OpenCLIP's visual representation.

OpenCLIP is pretrained. This experiment cannot demonstrate learning visual
features from one image.

## Data collection: validation and final test must be separate

Use photos you own or are allowed to use. Avoid faces, private documents,
license plates, addresses, or other sensitive content.

Create **two independent CSV manifests** with the existing format:

```csv
path,label,split,vehicle_id,capture_group
images/a_support.jpg,CLASS_A,support,object_a1,session_01
images/a_query_1.jpg,CLASS_A,known,object_a2,session_02
images/a_query_2.jpg,CLASS_A,known,object_a3,session_03
images/u_query.jpg,UNKNOWN_X,unknown,object_u1,session_04
```

Despite the historical column name `vehicle_id`, it can identify any
physical object. In v0.10 it means **physical-instance ID**.

### Recommended validation manifest

At minimum:

- 3 known classes;
- exactly 1 support image per known class;
- 2–4 known query images per class, preferably different physical instances;
- at least 3 unknown classes;
- 1–3 query images per unknown class;
- support/query capture groups must differ.

Use validation only to select the threshold. You may inspect validation
predictions.

### Recommended final test manifest

Use **different classes from validation** whenever practical. At minimum:

- 3 new known classes;
- exactly 1 support image for each;
- 2–4 known queries per class from distinct physical objects/sessions;
- at least 3 additional unknown classes;
- no file, physical-instance ID, capture group, or calibration image reused.

Do **not** inspect test scores and then change the threshold while continuing
to call the result held-out.

## Good first photo categories

The purpose is not to prove that OpenCLIP knows familiar words. Labels are
remapped to opaque names internally. Prefer categories for which you can
photograph multiple physical examples, for example:

- mugs / shoes / remote controls;
- toy vehicles of several types;
- different packaging or bottle categories;
- cars only if support and query are genuinely different physical vehicles.

Do not use two crops of the same photograph as independent support/query
examples.

A later experiment can test masked logos/text to measure how much the frozen
representation depends on those visual shortcuts.

## Setup

The core repository needs no ML dependencies, but this experiment does.

Use a Python environment supported by your installed PyTorch build. Follow the
official PyTorch installation selector, then install:

```powershell
python -m pip install open_clip_torch Pillow
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The first OpenCLIP run can download pretrained weights. Local images are read
from disk; ACM does not intentionally upload them to a hosted inference API.

## Phase 1 — calibrate on validation only

```powershell
python -m examples.calibrate_vision_threshold ^
  --manifest data/vision_v10_validation.csv ^
  --device auto
```

PowerShell also accepts the command on one line.

The calibration objective is:

```
0.5 * known_top1_accuracy + 0.5 * unknown_rejection_rate
```

When several thresholds tie, the code selects the higher (more conservative)
threshold.

**Write down the returned threshold before opening final test predictions.**

## Phase 2 — final matched comparison

Suppose validation returned `0.842731`:

```powershell
python -m examples.compare_vision_baseline ^
  --manifest data/vision_v10_test.csv ^
  --threshold 0.842731 ^
  --device auto
```

Report:

- known-query top-1 accuracy;
- unknown rejection rate;
- ACM/direct-NN prediction agreement;
- ACM/direct-NN score agreement;
- opaque-label behavior agreement;
- exact encoder model, pretrained tag, device and library versions;
- number of supports and queries.

## Predeclared interpretation

### If ACM and direct 1-NN agree 100%

This is **expected** under current implementation. Attribute recognition
performance to:

1. OpenCLIP's pretrained representation;
2. the selected support example;
3. cosine nearest-neighbor;
4. the rejection threshold.

Do **not** call it new visual concept induction by ACM.

### If they disagree

Treat this first as an **implementation bug or unmatched condition**, because
the algorithms are intended to be equivalent with one support per label.
Inspect embeddings, thresholding and tie behavior before making any research
claim.

### If opaque labels perform identically

That shows the memory stage does not need semantic label names. It still does
not prove the image representation learned the category after one example.

### If known accuracy is high but unknown rejection is poor

The representation/rejection rule is not safe enough for open-set use.

### If both are poor

Publish the negative result. It means one-shot nearest-neighbor on this frozen
representation is insufficient for the selected data.

## Why this matters for ACM

v0.9 showed that one positive example does not, in general, uniquely identify
a hidden concept rule. v0.10 tests the corresponding practical visual
baseline: perhaps a powerful pretrained representation can nevertheless make
one-shot *recognition* look good.

If so, the correct interpretation is **new label binding over old features**.
A future ACM contribution must beat or qualitatively differ from that matched
baseline without secretly adding extra supervision or pretrained knowledge.

## Limitations

- OpenCLIP pretraining can contain related categories, products, brands or
  visual patterns.
- A few locally collected images are not population-representative.
- Physical-instance and capture-group separation relies on honest metadata.
- Threshold calibration can overfit a small validation set.
- Similarity scores are not calibrated probabilities.
- This experiment tests recognition, not autonomous discovery of the latent
  rule that defines a concept.
