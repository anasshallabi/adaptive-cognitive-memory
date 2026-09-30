# ACM research notebook — 2026-09-30

## Original observation

A human can see the name of a previously unknown car brand once and sometimes recognize it later and associate it with existing concepts (brand, logo, car, manufacturer). Learning new *facts* should be distinguished from acquiring new *concepts* or perceptual skills.

## Research questions

- RQ1: How quickly can a previously unseen association be encoded while keeping a perceptual representation fixed?
- RQ2: How far does one-example recognition generalize to different cars, viewpoints and unseen brands?
- RQ3: Can a system reliably build inferences and revise conflicting information with traceable provenance?
- RQ4: Can **new concepts** be acquired from very few exposures **without supplying the ontology or inference rules by hand**?
- RQ5: Can such mechanisms improve verified reasoning and reduce computation against *matched baselines*?

## Progress and limitations

- v0.1: synthetic vector nearest-neighbor memory.
- v0.2: frozen pretrained OpenCLIP wrapper, image dataset split validations (not yet measured on real photos).
- v0.3: deterministic constrained-language triple extraction; fact history and limited inheritance with evidence paths. This is a classic symbolic graph operation and is **not a novel cognitive architecture**.

## Falsification criteria

If a simple knowledge graph and a frozen embedding/nearest-neighbor baseline perform as well as ACM at comparable accuracy and compute cost, no new architecture advantage has been established. If unseen phrasings fail, we cannot claim general natural-language learning. If real-image tests fail, we report them. All sources and negative results should remain available.
