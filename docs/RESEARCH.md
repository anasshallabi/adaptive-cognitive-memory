# ACM research notebook — 2026-09-30

## Original question

Why can a person see a new car brand once, remember its name, and connect it to previously learned concepts, while current machine learning often depends on large-scale pretraining? We distinguish **single-exposure fact binding** from **new concept induction**.

## Testable questions

1. Can a memory store facts immediately without changing neural weights?
2. Can image recognition transfer across different vehicles and sources from one labeled example?
3. Can text facts preserve provenance, derive transparent relations, and expose contradictions?
4. Does structured extraction by an existing local pretrained model expand coverage of unseen sentence patterns beyond a fixed grammar?
5. Can we actually induce new concepts and adjust them after contradictory observations with superior compute/accuracy tradeoffs vs matched baselines?

## Progress

- v0.5: frozen synthetic text evaluation; local hybrid result 13/28 exact on test.
- v0.6: post-hoc conservative guard; 21/28 on the same known test, explicitly not an independent gain.
- v0.7: explicit candidate/review/reject/retract evidence states.
- v0.8: local SQLite persistence and replay of reviewed evidence across restarts.
- v0.9: formal synthetic test of whether one positive example uniquely identifies a new opaque concept rule; matched exemplar/similarity/version-space/oracle baselines.

- v0.1: synthetic vector memory (nearest-neighbor).
- v0.2: optional frozen OpenCLIP image front-end and split checks; real-photo empirical results pending.
- v0.3: deterministic FR/EN controlled-grammar fact memory and bounded graph traversal.
- v0.4: optional local pretrained Ollama structured extractor, source-span constraints, same fact memory, synthetic held-out paraphrase test suite. Tests validate engineering behavior only; **there are no claimed local-model accuracy numbers yet**.

## Current concept-induction hypothesis

Binding a new arbitrary label after one exposure is easy. Inferring the
latent rule that defines a concept is a different problem and can be
logically underdetermined from one positive example. v0.9 makes this
ambiguity explicit instead of counting successful nearest-neighbor retrieval
as concept formation.

## v0.10 matched real representation control

The next visual experiment gives current ACM and a direct cosine
nearest-neighbor baseline the **same frozen OpenCLIP embeddings** and threshold.
With one support vector per label they are expected to agree exactly. If they
do, successful recognition is evidence for label binding over a pretrained
representation, not a distinct ACM visual learner. Threshold calibration is
restricted to separate validation images.

## Scientific honesty / falsification

At present, ACM is a combination of classic associative retrieval and symbolic graph operations. If a simple knowledge graph or pretrained model with retrieval matches its results, there is no demonstrated advantage. If model-assisted paraphrase extraction improves accuracy at substantial GPU cost, report both quality and total incremental cost. If it does not, publish that too.

A small hand-written synthetic benchmark is for debugging methodology and will not establish general language competence. Robust experiments will need larger independent datasets, sources, revisions, ablations, uncertainty, and reproducible measurements.
