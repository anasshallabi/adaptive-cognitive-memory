# ACM — Evaluation protocol

## Engineering smoke tests

Run `python -m unittest discover -s tests -v`. Mock image encoders and fake local LLM replies verify program behavior, input validation and provenance. **They are not scientific measures of model accuracy.**

## Text v0.4 (benchmark harness implemented)

See [EXTRACTION.md](EXTRACTION.md) and `benchmarks/text_v04.jsonl`.
- Eight `dev` plus twelve `test` synthetic sentences (no shared annotated template families).
- Baselines: `RuleExtractor` v0.3, optional `OllamaExtractor`, optional `HybridExtractor` rules-then-Ollama.
- Evaluate identical sentences/split, record model identifier, temperature, prompt, PC hardware, output JSON, abstentions, exact extraction precision/recall/F1 and latency.
- Refuse unsupported/multi-claim statements where possible; flag malformed/hallucinated source spans.
- Tune on `dev` only and report `test` once without post-hoc cherry picking.
- No measured real-model advantage currently claimed. Dataset is synthetic, small and not representative.

## Vision v0.2 (benchmark harness implemented, real experiments pending)

1. Obtain licensed images with provenance, physical vehicle IDs and capture sessions.
2. Use frozen OpenCLIP with exact weight, device and preprocessing versions recorded.
3. One labeled `support` photo per known brand; evaluate distinct cars, angles, and sessions; include disjoint unknown brands.
4. Calibrate rejection thresholds only on validation brands and not final test brands.
5. Compare visible logo/text vs logo-masked cases to detect leakage.
6. Measure known top-1 accuracy, unknown false acceptance/rejection, memory retention, latency, RAM/VRAM, and uncertainty over repeated episodes.

## Research standards

Compare with strong, equally resourced baselines; track learned facts separately from pretrained capabilities. Publish negative results, dataset limitations and test-time compute. **No claim of human-like learning is currently supported.**
