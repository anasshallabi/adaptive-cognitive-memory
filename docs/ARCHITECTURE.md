# ACM v0.3 — Architecture

```text
Numeric vectors -> CognitiveMemory (v0.1) -> cosine match / unknown

Photos -> OpenCLIPEncoder (frozen pretrained; optional v0.2)
                        |
                        v
                   VisionMemory -> label / unknown

Controlled FR/EN sentence -> parse_claim (explicit grammar)
                        |
                        v
                    TextMemory -> facts + provenance
                        |
            direct / bounded is_a inference
                        |
                        v
            supported / negated / conflict / unknown

MultimodalMemory: exact explicit image label -> TextMemory.about(label)
```

The perception/representation layer and memory are intentionally separate. Graph facts do not rewrite pretrained neural weights; the bridge matches user-supplied labels, not jointly learned perceptual concepts.

**Implemented:** direct fact retention, source identifiers, duplicate suppression per source, limited syntax, explicit negative evidence, positive `is_a` transitive closure with configurable hop bound. Results include source paths.

**Missing:** persistence, temporal/versioned claims, truth verification, contradiction resolution, general language parsing, automatic concept formation, independent semantic cross-modal grounding, causal reasoning.

**Complexity:** vector nearest-neighbor O(Nd) for N vectors of length d; current text inference uses repeated scans of the claim list (small-scale baseline; not an indexed graph engine). Large cyclic graphs with many paths may be expensive; max_hops mitigates but does not provide global complexity bounds. See [TEXT.md](TEXT.md) and [VISION.md](VISION.md).
