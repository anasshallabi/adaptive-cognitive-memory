# ACM experimental conservative claim gate

Created after reviewing the first local v0.5 benchmark results.
This is an exploratory engineering safeguard, **not** an independently
validated improvement.

## Motivation

The original hybrid used broad grammar rules before the local LLM.
On the v0.5 test, some rules accepted attributions, instructions and
multiple propositions as unconditional facts. The pipeline could
therefore store incorrect triples without a model call.

## Opt-in design

The optional `ConservativeHybridExtractor` first checks for a limited
set of questions, directives, attributions, uncertainty markers,
historical forms, and recognizable multi-clause structures.
Risky inputs are rejected with an auditable reason; they are not
forwarded to the rules or local LLM. Other sentences use the original
grammar if the subject appears canonical, otherwise the existing
local model handles them.

The old `HybridExtractor` and historical benchmark remain unchanged.

## Run locally

```powershell
git pull origin main
python -m unittest discover -s tests -v
python -m examples.compare_text --dataset benchmarks/text_v05.jsonl --extractor conservative --model qwen3:14b --think off --prompt literal --split test --summary
```

This dataset was already inspected during development. Any new score
on it is **post-hoc debugging**, not fresh generalization evidence.

## Limitations and next steps

The gate is based on finite lexical patterns. It may reject valid
statements, miss dangerous forms, and does not independently verify
facts or guarantee safe interpretation. Model fallback can still err.

Next, design evidence states (candidate, supported, disputed),
provenance and revision history. Freeze a new independently annotated
corpus before comparing false acceptance, false refusal, accuracy and
CPU/GPU cost. Neither this gate nor existing pretrained LLM extraction
demonstrates novel concept formation.

See [original v0.5 results](RESULTS_V05_2026-09-30.md).
