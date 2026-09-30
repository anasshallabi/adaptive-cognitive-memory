# ACM v0.2 — Architecture

```text
Image path -> OpenCLIPEncoder.embed() [frozen pretrained network]
                          |
                          v
            CognitiveMemory.learn()
             /                     \
     embedding stored         label -> supplied concepts
             |
             v
       cosine nearest neighbor + threshold
             |
             v
           label / unknown
```

The v0.1 numeric feature API remains available. `acm/vision.py` provides a mockable image encoder interface, a frozen pretrained OpenCLIP implementation with CPU/CUDA selection and official RGB preprocessing, and a `VisionMemory` wrapper. The `acm/benchmark.py` runner validates one-shot support and split metadata and reports basic known-vs-unknown performance. Image encoder dependencies are optional for lightweight tests.

Memory use scales O(Nd) for N observations of dimensionality d, lookup O(Nd) (plus image-encoding inference); image encoder pretraining and inference costs are not avoided. Stored associations do not persist after process exit. A cosine threshold is not a calibrated probability.

**Not implemented:** visual concept induction, pretrained-free perception, reasoning, causal world models, contradiction handling, persistence, and measured accuracy/efficiency improvements. Generalization from one exemplar is a testable hypothesis, not a demonstrated result. Read [the detailed vision guide](VISION.md).
