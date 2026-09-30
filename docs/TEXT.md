# ACM v0.3 — Controlled-text memory (FR / EN)

**Status: experimental rule-based baseline; not an LLM, a human-like learner, automatic fact checking, nor general natural-language understanding.**

## Goal

Test three separately measurable capabilities:
1. **Immediate association:** store a new subject–predicate–object claim after one supplied sentence.
2. **Transparent link reuse:** traverse `is_a` relations to answer a narrow class-membership question, showing the path of supporting claims.
3. **Uncertainty and contradiction:** distinguish unsupported, negated and contradictory evidence without silently selecting a winner.

These are classic symbolic knowledge-graph operations. Their presence is not itself novel AI.

## Accepted grammar (one sentence at a time)

Examples supported:

| Language | Example | Stored relation |
| --- | --- | --- |
| French | `Zentra est une marque automobile.` | `(Zentra, is_a, marque automobile)` |
| French | `Une marque automobile est une organisation.` | `(marque automobile, is_a, organisation)` |
| French | `Zentra n'est pas une banque.` | Negative `is_a` |
| French | `Zentra fabrique des voitures électriques.` | `makes` |
| French | `Zentra possède un logo.` | `has` |
| French | `Zentra utilise des batteries.` | `uses` |
| English | `Zentra is a car brand.` | `is_a` |
| English | `Zentra is not a bank.` | Negative `is_a` |
| English | `Zentra makes electric vehicles.` | `makes` |

It **does not** understand unrestricted prose, synonyms, entity disambiguation, plurals, time, modality, sarcasm, or unstated commonsense rules. Unsupported inputs produce a `ValueError` (fail closed rather than inventing an extraction).

The two supported forms of question are:
- `Est-ce que Zentra est une organisation ?` / `Is Zentra an organization?`
- `Que sais-tu de Zentra ?` / `What do you know about Zentra?`

## Run (Python 3.10+, no additional packages)

```bash
python -m examples.text_one_shot
python -m examples.text_one_shot --fact "Zentra est une marque automobile." --fact "Une marque automobile est une organisation." --ask "Est-ce que Zentra est une organisation ?"
python -m unittest discover -s tests -v
```

The example teaches two separate claims, then follows a two-hop `is_a` chain. It has **not** learned the concept of an organization from a single experience; the inference works because both links and the transitivity rule are already provided.

## JSON outcomes

`TextMemory.query(subject, predicate, object)` returns:
- `supported`: one or more positive stored or inferred evidence paths;
- `negated`: one or more explicit negative claims;
- `conflict`: both positive and negative evidence paths;
- `unknown`: nothing supports or explicitly negates the queried relation.

Every path contains raw source strings and original texts. **Stored evidence is not verified truth.** Contradictions are retained and surfaced; no automatic resolution or confidence calibration is attempted. Negation is not treated as transitive. Only positive `is_a` uses bounded transitive closure (`max_hops`).

Memory is in-process only; source IDs are provided by the caller and do not attest to source authority.

## First image/text connection

`acm/multimodal.py` joins a `VisionMemory` with `TextMemory` using an **explicit matching label**. If an image is recognized as "Zentra", previously stored claims about that label become available. If rejected as unknown, no claims are returned. This is **manual grounding through label identity**, not cross-modal semantic learning.

## What comes next

- Add persistence, provenance and revisions without losing a claim's history.
- Evaluate held-out linguistic variations against this deliberately simple grammar baseline.
- Compare with independently validated structured extractors, controlling inference cost.
- Design experiments to acquire *new* concepts without prewritten parsing or reasoning rules.

See [research design](RESEARCH.md) and [evaluation protocol](EVALUATION.md).
