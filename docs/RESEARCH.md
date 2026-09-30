# Research notebook — 2026-09-30

## Origin

Observation: a person can encounter the name of an unfamiliar car brand once, remember the name and recognize some future instances by reusing existing knowledge of text, logos, brands, and vehicles. Hypothesis: combine fast episodic binding with structured semantic relations and selective computation, then experimentally test whether this improves knowledge acquisition and transfer.

## Definitions

- **One-shot acquisition:** new association learned after one labeled observation, while holding any pretrained encoder fixed.
- **Association:** an explicit retrievable link; not evidence of autonomous reasoning.
- **Generalization:** success on distinct observations (different cars/lighting/angles), not memorization of the input.
- **Continual learning:** acquire multiple concepts over time without unacceptable degradation of earlier concepts.
- **Open-set recognition:** reject new/unseen classes instead of always returning the nearest known label.

## Existing lines of work to study

Few-shot learning, metric learning and nearest-neighbor retrieval; embedding encoders such as CLIP; complementary learning systems, continual learning, retrieval-augmented generation, semantic graphs, online clustering and model-based inference. These are inspirations, not ACM inventions. Seek peer-reviewed references and evaluate novelty carefully before scientific claims.

## Milestones

M0: implement small transparent baseline (done). M1: replace synthetic features with reproducible image embeddings. M2: build carefully separated test episodes; M3: evaluate unknown-class detection and memory retention; M4: investigate concept formation without hand-provided links; M5: report negative as well as positive results.

## Falsification

The strong hypothesis is **not supported** if performance on held-out examples, unknown classes, compute, or forgetting is no better than a simple pretrained encoder with nearest-neighbor retrieval under fair comparison. Results must be logged, not selectively reported.
