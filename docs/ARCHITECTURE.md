# Architecture — baseline 0.1

```
 numerical observation (provided externally)
                   |
                   v
        CognitiveMemory.learn()
             |          |
             v          v
      stored vectors   label -> provided concepts
             |
             v
       cosine retrieval + open-set threshold
             |
             v
       match / unknown + linked concepts
```

**Current:** in-memory list of `Observation` instances; dictionary of concept links; cosine nearest exemplar; thresholded rejection. Complexity: insertion O(d), exact lookup O(Nd) for N stored observations of dimension d; memory O(Nd + E), E link count. No optimized index, updates, inference engine or persistence. Repeated labels can have multiple exemplars.

**Future components are proposals, not implemented:** visual encoder, online concept induction, confidence calibration, contradiction and provenance tracking, episodic vs semantic memory, consolidation policy, and continual-learning evaluators.

**Known limitations:** a similarity threshold does not guarantee calibrated confidence; one observation may not transfer to different car designs; embeddings inherit pretrained learning and biases; manually supplied concepts are not inferred by the prototype; synthetic vectors cannot validate real-world cognition.
