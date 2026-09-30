# ACM — Direct assertions, attribution, uncertainty and time

**Design note written after reviewing local result t03, on 2026-09-30.**
This is an **exploratory post-hoc diagnostic**, *not* a held-out benchmark or
proof of generalization. The original benchmark and its results are frozen.

## Triggering observation

The local `qwen3:14b` hybrid run for `t03` received:

> Zentra is considered a car brand.

The expected row is `(Zentra, is_a, car brand, positive=True)`, but the
local model returned an abstention. There was no network/model error and
inference took approximately 9.81 seconds. We **cannot tell from an
abstention alone why the model chose this result**.

The annotation itself is worth examining. "X is considered a Y" reports
a classification or perception, but does not necessarily establish the
unconditional fact "X is a Y". The current ACM data model represents only
four categorical/property relations and their polarity, so it cannot
faithfully express *who considers X a Y*, or whether the claim is an
assertion, a report, an uncertain possibility or a past fact.

## Proposed distinction

| Utterance | Semantic status | What can be recorded without assuming extra facts? |
| --- | --- | --- |
| `Zentra is a car brand.` | Direct positive assertion | Store the speaker's `is_a` claim with source; **not independently verified truth**. |
| `Zentra is considered a car brand.` | Attributed/consensus-style statement | An attribution about classification; no unconditional `is_a` deduction without an explicit policy. |
| `According to a reviewer, Zentra is a car brand.` | Reported assertion | Reviewer is the attributed source, not necessarily the primary speaker's personal assertion. |
| `Zentra might be a car brand.` | Modal possibility | Uncertain proposition, not a categorical membership fact. |
| `Zentra used to be a car brand.` | Past membership | A time-qualified statement, not automatically current membership. |
| `Zentra is not a car brand.` | Direct negative assertion | Store direct negative claim, with provenance; conflict detection if another source disagrees. |

**A conservative extractor may reasonably abstain when the limited schema
cannot encode an utterance faithfully.** Other representation choices are
possible; they must be decided explicitly in an annotation policy, not
retroactively chosen to make one model's score look better.

## Small local contrast probe

A separate CLI submits selected sentences to the *rules* and directly to
the Ollama model independently (not via hybrid), so a successful rule
match cannot conceal how the model actually handles direct assertions.

Run only two or three sentences to avoid unnecessary local GPU usage:

```powershell
git pull origin main
python -m examples.probe_text --model qwen3:14b --case direct --case attributed
python -m examples.probe_text --model qwen3:14b --case fr_direct --case fr_attributed
```

Other available probes: `hedged`, `reported`, `past`, `fr_hedged`,
and `--sentence "your own fictional example"`. The command reports
parsed claims, abstentions, failures and per-sentence model latency;
it does **not** request or expose the model's private reasoning.
`--model` is optional: without it, only deterministic rules run.

All examples use an invented brand and do not require a paid inference
API. Model execution occurs through the configured *loopback-only*
Ollama endpoint. These probes were designed **after observing t03**;
their scores are not suitable as fresh generalization evidence.

## Scientific protocol correction needed

1. Keep `benchmarks/text_v04.jsonl` unchanged, retaining its original
   baseline and local Qwen3 numbers as a transparent historical record.
2. Design a **new annotation policy before collecting future test data**.
   Distinguish direct, negated, attributed, modal and time-qualified
   propositions. Record speaker, attributed source, time and provenance.
3. Evaluate both extraction quality and **faithful abstention**. Never
   penalize a model merely for not fabricating an unconditional fact
   from a non-categorical statement; still measure missed unambiguous
   claims separately.
4. Create a substantially larger, independently designed test split
   (preferably with independent annotator agreement and multiple entity
   types), and avoid tuning based on that test.
5. Compare the same pretrained LLM, rules and symbolic memory under
   fixed settings and hardware before interpreting results.

## Research gap

An evidence-aware memory would preserve a claim's content, who asserted
it, confidence/uncertainty **about the assertion**, and time. This is
different from simply storing more triples. The future system should
also represent retractions and revisions without rewriting history.

This is a proposed direction, not yet implemented. [First local Qwen3
results](RESULTS_V04_2026-09-30.md).
