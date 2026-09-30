# ACM v0.5 — Predeclared evaluation of *claims* and abstention

**Frozen protocol authored 2026-09-30, before any local Ollama run against this new dataset.** The benchmark is a deliberately small original synthetic research fixture, **not an independently annotated representative corpus** and not an assessment of human-like concept induction.

## Motivation / observed evidence

On the earlier v0.4 set, the user's Windows/Ollama `qwen3:14b`
hybrid obtained 4/12 exact outcomes, equal to deterministic rules but
far slower. After *post-hoc* debugging, an alternative `literal`
instruction enabled model-only extraction of a directly asserted
English claim, then French extraction and two conservative abstentions.
Those results cannot establish generalization: the examples had been
observed during development.

This **new frozen set** is intended to detect failure modes under a
predeclared protocol. Its 40 cases were written after viewing v0.4
results, by the same project assistant, so they are **not genuinely
independent of the researchers' hypotheses**, and could overlap the
pretraining of any chosen model.

## Explicit annotation policy

We score whether the sentence **asserts** one extractable
subject–predicate–object relation, not whether the proposition is true
in the external world. All named entities and scenarios are
synthetic/fictional.

Accepted relations: `is_a`, `makes`, `has`, `uses`. All output
subjects/objects must be exact contiguous spans of the input sentence
(ignoring only whitespace and capitalization).

- **Positive direct statements**: annotate exactly one direct present
  claim with `positive=true`. Examples include is, fabrique, builds,
  owns, utilise and equivalents; a prewritten grammar is **not** used
  to label model success.
- **Direct negative statements**: annotate the explicit negation with
  `positive=false`; do not treat a negative sentence as absence of
  information.
- **Attributed statements** (e.g. `According to`, `Selon un magazine`,
  `is considered`): `expected=null` because the present restricted
  schema has no slot for attribution. This does not deny that a
  *reported claim* can be stored in a richer future schema.
- **Uncertain possibilities** (`might`, `may`, `probably`):
  `expected=null`. They should not be upgraded to categorical facts.
- **Historical claims** (`formerly`, `était`, dated past-tense
  assertions): `expected=null` as the current schema lacks time.
- **Questions, commands, prompt injections, multi-claim sentences and
  unsupported predicates**: `expected=null`.

We do **not** infer implied claims from external commonsense or accept
an unconditional `is_a` for something merely considered, reported or
possible. A faithful refusal to flatten modality is a **correct
abstention** under this policy. This choice is a measurable policy,
not the only possible knowledge representation.

## Dataset split and controls

`benchmarks/text_v05.jsonl`: 12 `dev` + 28 `test`, of which
18 test examples have an expected claim and 10 require abstention.
Rows have IDs `v05-dNN` and `v05-tNN` with named syntactic pattern
families. Validation checks distinct IDs, entire sentences,
dev/test pattern-family overlap, valid predicates, polarity, and
literal source spans.

Examples intentionally cover both French and English, positive and
negative facts, unfamiliar names, synonyms not in v0.3, reported,
modal, historical and multi-clause statements. **It is too small for a
population accuracy claim.** The dataset is public, so the test split is
a transparent benchmark, not a secrecy-based blind challenge.

## Fixed model comparison (no tuning against test)

1. Document the local environment: Python version, Ollama model ID
   and quantization, processor, VRAM, Ollama version and device load.
   Do not disclose secrets or private data.
2. Keep the current `literal` prompt, `think=off`, JSON schema,
   default temperature zero and existing extraction validation **fixed**.
3. Run baseline rules on `dev`; run local `hybrid` on `dev` and
   inspect all errors before freezing any decisions. If any changes
   are made, create and freeze an additional never-observed test set.
4. Run `rules`, `ollama` and `hybrid` against `test` with the
   same model and settings, recording every prediction/error.
   Only report a comparison against these fixed settings; if
   the researcher tunes on test, explicitly label the result exploratory.
5. Report exact match, precision/recall/F1, refusal quality for
   annotated nonclaims, each error, routes and timing. Measure cold
   vs warm load separately; do **not** call a warm-up difference
   an architectural speedup.
6. Publish negative results, errors and disagreements. Do not
   reinterpret test labels after model outputs.

## Windows PowerShell commands

All are local; no paid API or model download is required with an
already installed `qwen3:14b` in Ollama. The rules baseline does not
require Ollama at all.

```powershell
git pull origin main

# Smoke test and zero-LLM baseline (dev)
python -m unittest discover -s tests -v
python -m examples.compare_text --dataset benchmarks/text_v05.jsonl --extractor rules --split dev --summary

# Model-assisted validation; use the exact settings previously probed
python -m examples.compare_text --dataset benchmarks/text_v05.jsonl --extractor hybrid --model qwen3:14b --think off --prompt literal --split dev --summary

# After fixed settings are recorded, run the 28 unseen test examples once:
python -m examples.compare_text --dataset benchmarks/text_v05.jsonl --extractor hybrid --model qwen3:14b --think off --prompt literal --split test
```

For a *matched* model-only comparison, replace `hybrid` with `ollama`;
for the zero-cost baseline, replace it with `rules`. The
`--summary` flag hides per-case outputs; omit it to inspect failures.
For expensive local diagnostics, `--case v05-tNN` selects specific
rows, but this **uses the held-out data** and any subsequent tuning
must be reported as post-hoc.

## Current status

**No v0.5 Ollama accuracy, inference speed, recall or generalization
measurements exist yet.** Only schema/unit tests and rules-only
engineering CI checks are eligible to run on GitHub, as that CI does
not host the user's locally installed language model.

Converting what a person or model *asserted* into memory is one-shot
**fact registration**, not proof of one-shot **concept formation**.
Versioned evidence and richer epistemic semantics remain work to do
(issue [#8](https://github.com/anasshallabi/adaptive-cognitive-memory/issues/8)).
