# Adaptive Cognitive Memory (ACM)

**Research prototype v0.4 — open-source experiments on memory, language, and vision. No human-level intelligence or novel general learning has been demonstrated.**

[Français](docs/README.fr.md) · [Text v0.4 experiment](docs/EXTRACTION.md) · [Controlled language v0.3](docs/TEXT.md) · [Vision](docs/VISION.md) · [Evaluation](docs/EVALUATION.md) · [Research](docs/RESEARCH.md)

ACM investigates whether an artificial system can memorize new facts from a single observation, connect them to earlier knowledge and reuse them without globally retraining a large neural model.

## Existing implementations

- **v0.1** — in-memory numeric vector association; cosine nearest-neighbor.
- **v0.2** — optional frozen pretrained OpenCLIP image encoder; real-image CLI and split guards (no real-photo benchmark published yet).
- **v0.3** — controlled French/English triple extraction with source-tracked facts, `is_a` inference and contradiction reporting.
- **v0.4** — *optional* loopback-only Ollama structured text extractor behind the same memory interface, deterministic rules as cheap first pass, 20-item synthetic FR/EN benchmark with held-out paraphrase families, and unit tests with mocked model replies.

**Important:** v0.4 uses a **pretrained** model for flexible language extraction; it doesn't train that model from one sentence or discover concepts autonomously. Storing a claim is not fact-checking. No improved benchmark results are asserted until a real local model has been run and compared.

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
```

The default rule-based mode needs **no external Python dependencies, model downloads or paid AI API**. The optional hybrid calls a local Ollama server and may consume CPU/GPU/RAM. See [the experiment protocol](docs/EXTRACTION.md). For Windows NVIDIA vision experiments see [docs/VISION.md](docs/VISION.md).

## Status and research roadmap

- [x] Minimal association store and basic reproducible tests
- [x] Vision adapter (real-image benchmarks still pending)
- [x] Source-preserving text memory and explicit contradictions
- [x] Optional flexible text extraction + held-out benchmark harness (real-model results pending)
- [ ] Publish measured extraction results on local hardware, including failed cases
- [ ] Build durable evidence-aware memory, revisions and corrections
- [ ] Evaluate concept induction versus matched graph, retrieval and pretrained-model baselines

Our objective is to **test hypotheses**, not to rebrand classic symbolic graphs or pretrained LLM capabilities as a new learning breakthrough. MIT license, community contributions welcome.
