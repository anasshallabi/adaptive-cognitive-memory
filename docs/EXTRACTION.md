# ACM v0.4 — Evaluating text extraction beyond fixed grammar

**Status: engineering implementation with an optional pretrained local LLM. No new concept-learning claim.**

## Why this experiment?

The v0.3 parser reliably stores simple French/English statements matching a few
hand-written templates. It cannot interpret novel paraphrases. Before changing
ACM's architecture, measure how much an existing small language model can help
extract new facts — while keeping the storage and reasoning graph unchanged.

This is **not** training a model from one sentence; it is one-shot **memory
binding using a previously pretrained language model**.

## Architecture

```
one French/English sentence
    |
    +--> RuleExtractor (v0.3, 0 ML cost) -----------+
    |                                              |
    +--> OllamaExtractor (optional, local only) ---+--> validated Claim
                     |                                         |
                     v                                         v
       schema + source substring guard                 TextMemory.remember_claim
                                                                |
                                                        provenance + contradiction
```

`HybridExtractor` tries the rules first and uses a local Ollama model only
when those rules abstain. Only the existing four relations are accepted:
`is_a`, `makes`, `has`, `uses`. Unsupported or ambiguous input should be
abstained from (not guessed). Rule-based extraction remains the comparison
baseline. Output constraints **reduce** hallucination risks but are not a
guarantee of correctness or immunity to prompt injection.

## How to run (Windows / PowerShell)

Python 3.10+; the rule baseline needs **no pip dependencies**.

```powershell
git clone https://github.com/anasshallabi/adaptive-cognitive-memory.git
cd adaptive-cognitive-memory
python -m unittest discover -s tests -v
python -m examples.compare_text --extractor rules --split test
```

Optional: install Ollama on your own machine (https://ollama.com), and
download a local model if it is not already available (model size and system
memory requirements vary). For example, if you have `gemma3:4b` locally:

```powershell
ollama pull gemma3:4b
python -m examples.compare_text --extractor hybrid --model gemma3:4b --split dev
python -m examples.compare_text --extractor hybrid --model gemma3:4b --split test
```

Use the **same settings**, do not tune after seeing held-out results.
`--extractor ollama` is available to measure model-only extraction
without the rules. Device allocation is managed by Ollama; our code does
not force GPU settings. The default endpoint is
`http://127.0.0.1:11434/api/chat`. Non-loopback endpoints are rejected.
The HTTP request has a timeout. Never expose the local Ollama endpoint to
untrusted networks, and do not submit personal or private texts to untrusted
models without understanding their handling.

[Ollama documentation: structured output schema](https://docs.ollama.com/capabilities/structured-outputs)

## Curated challenge set

`benchmarks/text_v04.jsonl` contains **20 small hand-written synthetic
sentences**, 8 `dev` and 12 `test`, in French and English. A row includes
`id`, `split`, `pattern_family`, `sentence`, and either a single explicit
expected `{subject,predicate,object,positive}` or `null` for inputs that
should be rejected.

Validation prevents duplicate sentence IDs, repeated full sentences and shared
**annotated** template families across dev and test. It cannot establish true
independence of linguistic patterns, external model pretraining exposure, or
representativeness of real human language. **The dataset must be replaced or
expanded before scientific claims.** It contains no private photos, user
accounts, or downloaded copyrighted texts.

Benchmark outputs:
- Exact-match accuracy (including appropriate abstentions)
- Relation-extraction precision / recall / F1
- Abstention rate for annotated non-facts
- Per-example errors and elapsed extraction time

Do not compare model inference costs with the rules unless hardware, model
weights, startup costs, RAM/VRAM and batching conditions are documented.
An LLM's prior training is not counted as a zero-cost new concept.

## Initial reproducible reference result (rules only)

GitHub Actions run [36727510772](https://github.com/anasshallabi/adaptive-cognitive-memory/actions/runs/36727510772)
on 2026-09-30 ran `python -m examples.compare_text --extractor rules --split test --summary`.

| Metric | Result |
| --- | ---: |
| Exact-match accuracy | 4/12 (33.3%) |
| Extraction precision / recall / F1 | 0 / 0 / 0 |
| Correct abstention on annotated non-facts | 4/5 (80%) |
| Runtime errors | 0 |
| Engineering unit tests | 61 passing |

This is a purposely challenging tiny synthetic **rules-only** baseline.
All seven annotated positive test facts use unfamiliar phrasing; none was
extracted exactly. It is neither a representative population estimate nor
a result for a local LLM. Speed was measured on GitHub's runner and should
not be generalized to an NVIDIA desktop or compared directly against
model inference on different hardware.

**The real v0.4 model comparison has not been executed yet.**

## Diagnose model fallback and avoid unnecessary GPU work

A first local run of `qwen3:14b` found **no test accuracy improvement** over rules (both 4/12 exact), with substantial added latency. See [the recorded negative result](RESULTS_V04_2026-09-30.md). It is essential to inspect **which route** produced each answer before changing the prompt.

The CLI now reports `route_stats`: counts, correct predictions, abstentions, errors and extraction latency broken down by `rules`, `guard` (question rejected before model), and `fallback` (local LLM). The full report includes per-item `route` and `outcome`.

To debug **one** failed test without running 12 expensive queries:

```powershell
python -m examples.compare_text --extractor hybrid --model qwen3:14b --split test --case t03
```

`--case` can be repeated. This diagnostic does not alter the benchmark's dataset or test labels. After studying held-out errors, create a fresh final evaluation set before reporting gains from any prompt or code change. No cloud model or API credits are used by the local Ollama route.

## Safeguards and limitations

- Source sentence and caller-supplied provenance are preserved in `Claim`.
- A claimed subject/object must appear as a textual span in the input.
- Invalid JSON, invalid predicates, impossible spans or missing fields
  are rejected and **not memorized**.
- One proposition at a time; multi-claim sentences are an abstention target.
- `status=abstain` means no claim is stored.
- Contradictory stored claims remain visible via `TextMemory`.

Known weaknesses: lexical span checks do not establish factual grounding,
negation may be parsed incorrectly, grammar and extraction are language-dependent,
the local model can hallucinate allowed facts, and there is no built-in truth
verification, persistence, revision policy or calibration.

## What would actually be novel?

A subsequent method that induces a previously unseen *concept*, identifies
its application to new examples, updates memory with evidence and corrects it
without expensive global retraining — and reproducibly outperforms standard
frozen-encoder retrieval / symbolic graph / local-LLM extraction baselines
for comparable compute and quality. No such result is claimed yet.
