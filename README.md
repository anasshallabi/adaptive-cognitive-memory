# Adaptive Cognitive Memory (ACM)

**Research prototype v0.10 matched real-image baseline — open-source experiments on memory, language, and vision. No human-level intelligence or novel general learning has been demonstrated.**

[Français](docs/README.fr.md) · [Matched vision baseline v0.10](docs/VISION_V10.md) · [One-shot concept study v0.9](docs/CONCEPT_V09.md) · [Persistent evidence v0.8](docs/DURABLE_EVIDENCE.md) · [Reviewed evidence v0.7](docs/EVIDENCE_LEDGER.md) · [Frozen v0.5 evaluation](docs/EVALUATION_V05.md) · [Text v0.4 experiment](docs/EXTRACTION.md) · [Controlled language v0.3](docs/TEXT.md) · [Vision](docs/VISION.md) · [Evaluation](docs/EVALUATION.md) · [Research](docs/RESEARCH.md)

ACM investigates whether an artificial system can memorize new facts from a single observation, connect them to earlier knowledge and reuse them without globally retraining a large neural model.

## Existing implementations

- **v0.1** — in-memory numeric vector association; cosine nearest-neighbor.
- **v0.2** — optional frozen pretrained OpenCLIP image encoder; real-image CLI and split guards (no real-photo benchmark published yet).
- **v0.3** — controlled French/English triple extraction with source-tracked facts, `is_a` inference and contradiction reporting.
- **v0.4** — *optional* loopback-only Ollama structured text extractor behind the same memory interface, deterministic rules as cheap first pass, 20-item synthetic FR/EN benchmark with held-out paraphrase families, and unit tests with mocked model replies.

- **v0.5 evaluation** — 40 synthetic FR/EN claims with dev/test partitions. The [first local Qwen3 result](docs/RESULTS_V05_2026-09-30.md) was **12/12 on dev but only 13/28 (46.4%) on test**; only 4/10 required abstentions were correct. Rule overmatching caused dangerous false facts.
- **v0.6 experimental safety gate** — an **opt-in** conservative heuristic prefilter that rejects recognized uncertain, attributed, historical, multi-clause or instruction-like statements before memory binding and defers complex entity subjects to the model. [Method and limitations](docs/CONSERVATIVE_GATE.md). No independent performance result yet.

- **v0.7 reviewed evidence ledger** — a **separate, in-process** candidate/accepted/rejected/retracted record store with explicit review, immutable audit-event objects, conflict handling, and a projection of reviewed, unopposed evidence into the graph. Nothing automatically becomes reviewed truth after one extraction. [Protocol and limitations](docs/EVIDENCE_LEDGER.md).

- **v0.8 local SQLite evidence** — atomic review-event records, reopening and replay, conflict-aware approved graph after restart, standard-library-only implementation, no model required. [Limitations and demo](docs/DURABLE_EVIDENCE.md).

- **v0.9 one-shot concept ambiguity study** — opaque binary features and arbitrary new labels test whether one positive example actually identifies a hidden concept rule. Includes exact-exemplar, fixed Jaccard, conservative version-space and oracle controls. No pretrained model or natural semantics are used in this first falsification experiment. [Protocol](docs/CONCEPT_V09.md).

- **v0.10 matched real-image baseline** — compare ACM against direct cosine 1-NN using the exact same frozen OpenCLIP embeddings, support images and rejection threshold. Includes a separate validation-only threshold calibration and opaque-label control. [Protocol](docs/VISION_V10.md).

**Important:** v0.4 uses a **pretrained** model for flexible language extraction; it doesn't train that model from one sentence or discover concepts autonomously. Storing a claim is not fact-checking. A [first local Qwen3 14B run](docs/RESULTS_V04_2026-09-30.md) matched rules-only accuracy at **4/12** on a small hand-written test and took much longer. A [follow-up attribution/uncertainty analysis](docs/MODALITY.md) warns that some original benchmark labels may conflate a reported classification with a direct fact.

## Quickstart (Python 3.10+)

```bash
python -m unittest discover -s tests -v
python -m examples.text_one_shot
python -m examples.evidence_review
python -m examples.durable_review
python -m examples.concept_one_shot --split test --summary
# Real-image v0.10 requires OpenCLIP/PyTorch and local photos; see docs/VISION_V10.md
python -m examples.compare_text --extractor rules --split test
```

### Optional local Ollama model

With [Ollama](https://ollama.com/) installed and a model already available locally (e.g. `gemma3:4b`):

```bash
python -m examples.compare_text --extractor hybrid --model gemma3:4b --split test
python -m examples.probe_text --model gemma3:4b --case direct --case attributed
```

To run the **new** local evaluation with an already installed `qwen3:14b`, see [v0.5 instructions](docs/EVALUATION_V05.md). Keep the v0.4 benchmark unchanged and do not treat post-hoc debugging successes as a new blind test.

The default rule-based mode needs **no external Python dependencies, model downloads or paid AI API**. The optional hybrid calls a local Ollama server and may consume CPU/GPU/RAM. See [the experiment protocol](docs/EXTRACTION.md). For Windows NVIDIA vision experiments see [docs/VISION.md](docs/VISION.md).

## Status and research roadmap

- [x] Minimal association store and basic reproducible tests
- [x] Vision adapter (real-image benchmarks still pending)
- [x] Source-preserving text memory and explicit contradictions
- [x] Optional flexible text extraction + a small hand-written benchmark harness
- [x] Publish first local Qwen3 results **including failure cases and semantic ambiguity caveats**
- [x] Freeze a separately authored protocol with direct, negated, attributed, modal and historical examples
- [x] Publish **negative v0.5 local-model findings** with per-route error analysis (13/28 exact test accuracy)
- [x] Archive **post-hoc conservative result** (21/28 on the same known v0.5 examples; not a held-out gain)
- [x] Add an explicitly reviewed in-process candidate evidence ledger
- [ ] Evaluate conservative screening on a fresh independently annotated test set
- [x] Persist review events with local SQLite and recover after simulated restarts
- [ ] Extend evidence to modality, time and robust entity resolution
- [x] Add a transparent one-positive concept-identifiability experiment against exemplar/similarity/oracle controls
- [x] Add a matched real-image OpenCLIP vs direct cosine-NN evaluator with separate threshold calibration
- [ ] Collect independent local validation/test images and run the frozen v0.10 protocol

Our objective is to **test hypotheses**, not to rebrand classic symbolic graphs or pretrained LLM capabilities as a new learning breakthrough. MIT license, community contributions welcome.
