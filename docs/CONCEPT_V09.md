# ACM v0.9 — Does one positive example identify a new concept?

**Synthetic falsification experiment.** This study asks a narrower question
than ordinary one-shot recognition: after seeing one positive example with a
new arbitrary label, can the system determine *which rule defines the concept*?

The answer can be **no even when the label is memorized perfectly**.

## Why this experiment exists

Earlier ACM versions demonstrated immediate association, retrieval, reviewed
evidence and persistence. Those capabilities do not establish *concept
induction*. A single object can support many explanations.

Example (opaque features):

- New label: `concept-nur`
- One positive support object: `{f17, f18, f19, f20}`
- Hidden rule used by the evaluator: `{f17, f19}`

After one positive exposure, all of these are compatible hypotheses:

- `f17`
- `f18`
- `f17 AND f19`
- `f18 AND f20`
- ...and other one-/two-feature conjunctions contained in the support.

A learner that immediately selects one of them is using an inductive bias.
It has not logically identified the rule from the single example alone.

## Frozen synthetic task

`benchmarks/concept_v09.jsonl` contains 12 episodes:

- 4 development episodes
- 8 test episodes
- 4 opaque support features in every episode
- one arbitrary nonce label per episode
- a hidden evaluator rule containing one or two support features
- contrastive negative examples
- eight held-out query objects per episode, with both positives and negatives

Feature tokens (`f01`, `f02`, ...) and labels
(`concept-nur`, `concept-bex`, ...) intentionally carry no natural
semantics. There is no image encoder, language model, world knowledge or
pretraining in this first experiment.

The hidden rule is present in the benchmark file because the evaluator needs
it to score predictions. **Learner classes never receive it.**

## Compared baselines

### 1. Exact exemplar memory

Memorize the one support feature set. A query is positive only if its entire
feature set is identical.

This is the simplest "I remember what I saw" baseline.

### 2. Jaccard similarity

Compare the query to the one positive example using set overlap. The threshold
is fixed at **0.5** before the test run.

This can generalize beyond exact identity, but similarity is not necessarily
the same as the hidden concept rule.

### 3. Conservative version space

Enumerate every one- or two-feature conjunction compatible with the observed
positive example. Do **not** choose one arbitrarily.

For a query:

- `positive` only when **all** surviving hypotheses predict positive;
- `negative` only when **all** predict negative;
- otherwise `unknown`.

This measures what is justified by the evidence, at the cost of coverage.

### 4. Hidden-rule oracle

The evaluator applies the known hidden rule directly. This is an upper-bound
control, **not a learnable baseline**.

## Contrastive evidence control

After the strict one-positive condition, the same version-space learner is
given designed negative examples. This tests whether additional evidence can
reduce ambiguity.

For a two-feature hidden rule, the contrasts in this fixture can eliminate all
alternative conjunctions. For a one-feature rule, one positive plus negative
contrasts still cannot distinguish that singleton rule from conjunctions that
contain it. This is deliberate: it exposes a real identifiability issue rather
than forcing every episode to "succeed."

## Metrics

- exact-exemplar query accuracy
- fixed Jaccard query accuracy
- fraction of episodes where the hidden rule is uniquely identified
- conservative prediction coverage
- accuracy conditional on the learner being certain
- the same uniqueness/coverage metrics after contrastive negatives
- oracle accuracy as a sanity check

A conservative learner can obtain high conditional accuracy by returning
`unknown` often. Therefore **coverage must always be reported alongside
accuracy**.

## Run

No network, GPU, Ollama or external package is needed.

```powershell
git pull origin main
python -m unittest discover -s tests -v
python -m examples.concept_one_shot --split dev --summary
python -m examples.concept_one_shot --split test
```

Changing `--jaccard-threshold` after looking at test results is a new
post-hoc experiment and must not replace the fixed 0.5 result.

## What would count as evidence?

This benchmark can demonstrate **underdetermination** and compare transparent
toy baselines. It cannot establish visual or linguistic concept learning.

A meaningful later experiment would require:

1. a representation learned independently of the test concepts;
2. truly new categories with multiple physical/source instances;
3. one support observation followed by distinct query observations;
4. unknown/rejection cases;
5. matched nearest-neighbor, retrieval, pretrained-model and oracle baselines;
6. frozen thresholds before final test;
7. reporting compute as well as accuracy;
8. independent data collection/annotation where possible.

If a frozen pretrained representation plus nearest-neighbor matches ACM, then
the experiment demonstrates **new label binding on an old representation**,
not autonomous new concept formation. That negative or mundane result should
be reported plainly.
