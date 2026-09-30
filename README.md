# Adaptive Cognitive Memory (ACM)

**Research prototype v0.5 evaluation — open-source experiments on memory, language, and vision. No human-level intelligence or novel general learning has been demonstrated.**

[Français](docs/README.fr.md) · [Frozen v0.5 evaluation](docs/EVALUATION_V05.md) · [Text v0.4 experiment](docs/EXTRACTION.md) · [Controlled language v0.3](docs/TEXT.md) · [Vision](docs/VISION.md) · [Evaluation](docs/EVALUATION.md) · [Research](docs/RESEARCH.md)

ACM investigates whether an artificial system can memorize new facts from a single observation, connect them to earlier knowledge and reuse them without globally retraining a large neural model.

## Existing implementations

- **v0.1** — in-memory numeric vector association; cosine nearest-neighbor.
- **v0.2** — optional frozen pretrained OpenCLIP image encoder; real-image CLI and split guards (no real-photo benchmark published yet).
- **v0.3** — controlled French/English triple extraction with source-tracked facts, `is_a` inference and contradiction reporting.
- **v0.4** — *optional* loopback-only Ollama structured text extractor behind the same memory interface, deterministic rules as cheap first pass, 20-item synthetic FR/EN benchmark with held-out paraphrase families, and unit tests with mocked model replies.

- **v0.5 evaluation protocol** — a **new 40-sentence FR/EN synthetic corpus**, with predeclared claim/abstention labeling, independently tracked dev/test partitions, and CLI flags for fixed local LLM settings. **No new model-based accuracy has been measured.**

**Important:** v0.4 uses a **pretrained** model for flexible language extraction; it doesn't train that model from one sentence or discover concepts autonomously. Storing a claim is not fact-checking. A [first local Qwen3 14B run](docs/RESULTS_V04_2026-09-30.md) matched rules-only accuracy at **4/12** on a small hand-written test and took much longer. A [follow-up attribution/uncertainty analysis](docs/MODALITY.md) warns that some original benchmark labels may conflate a reported classification with a direct fact.

## Quickstart (Python 3.10+)

```bash
python -m unittest discover -s tests -v
python -m examples.text_one_shot
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
- [ ] Run the **new** v0.5 local-model comparison and publish all outcomes and errors
- [ ] Build durable evidence-aware memory, revisions and corrections
- [ ] Evaluate concept induction versus matched graph, retrieval and pretrained-model baselines

Our objective is to **test hypotheses**, not to rebrand classic symbolic graphs or pretrained LLM capabilities as a new learning breakthrough. MIT license, community contributions welcome.
