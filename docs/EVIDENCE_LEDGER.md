# ACM v0.7 — Reviewable evidence ledger (prototype)

## Research motivation

The first v0.5 local Qwen3 experiment produced **13/28** exactly
correct answers and accepted some attribution, instruction, and
multi-claim text as categorical facts. The post-hoc conservative
extractor then produced **21/28** exact answers on those **same**
28 examples, and rejected all ten annotated nonclaims. That is a
regression improvement, **not** an independent proof of generalization.

Extracting a well-formed relation, even from a reliable-looking
sentence, does not make that relation true. We therefore separate
**source assertions** from **reviewed knowledge**.

## New explicitly reviewed workflow

```
new observed sentence
      |
      v
chosen extractor (rules / optional local Ollama)
      |
      +-- abstain --> no record
      |
      v
EvidenceLedger.observe(...) --> CANDIDATE
      |                              |
      |  decision="accept"           | decision="reject"
      v                              v
   ACCEPTED                       REJECTED
      |                              |
      | decision="retract"           | (audit history kept)
      v
   RETRACTED

ACCEPTED + pending/accepted opposite-polarity claim => UNDER_REVIEW / DISPUTED
```

The ledger has four per-record statuses: `candidate`, `accepted`,
`rejected`, and `retracted`. A record starts as a candidate. The
review call requires an explicit actor and a written reason. Existing
approved claims can be retracted without deleting their history.
Rejecting a claim means *rejecting the proposed extraction*, whereas
accepting a `positive=False` claim is explicit negative evidence.

### Query interpretation

| Query status | Interpretation |
| --- | --- |
| `unknown` | No active reviewed or pending evidence for the exact relation |
| `candidate` | Only pending candidate claim(s) exist |
| `supported` | At least one positive reviewed source claim, no opposing active evidence |
| `negated` | At least one negative reviewed source claim, no opposing active evidence |
| `under_review` | A reviewed claim has an opposite-polarity candidate awaiting review |
| `disputed` | Reviewed sources assert both polarities of the same relation |

This is a status **of the stored source claims and reviewers' actions**,
not a factual truth score or probability.

`approved_memory()` constructs a **fresh, in-memory** `TextMemory`
from accepted claims while excluding contradictory or pending-opposition
relations. Only this projected subset can participate in the existing
limited `is_a` graph traversal. A later retraction changes the
projected graph; original audit events remain available.

## Run the no-LLM demonstration

```powershell
git pull origin main
python -m unittest discover -s tests -v
python -m examples.evidence_review
```

This example uses invented brands, explicit reviews and opposite
claims to illustrate initial candidate state, acceptance, pending
conflict, rejection, later retraction and no active evidence.

### Library usage

```python
from acm.evidence_ledger import EvidenceLedger
from acm.text_extraction import RuleExtractor

ledger = EvidenceLedger()
entry_id = ledger.observe(
    "Zentra est une marque.", RuleExtractor(), source="fictional-source-1"
)
assert ledger.query("Zentra", "is_a", "marque")["status"] == "candidate"
assert ledger.approved_memory().claims == []

# This action must reflect actual deliberate review of the source.
ledger.decide(
    entry_id, decision="accept", actor="reviewer",
    reason="Confirmed source-1 directly asserts this classification",
)
assert ledger.query("Zentra", "is_a", "marque")["status"] == "supported"
```

The ledger checks that extracted `Claim.source` and `Claim.text`
match the caller-supplied source/sentence. This helps preserve
provenance, but **does not verify meaning or accuracy**.

## Limitations — important

- **In-memory only.** No disk persistence, concurrency, auth,
  timestamped events, event-signing, tamper-resistance or durable audit.
  IDs are sequential only within each process.
- An explicit reviewer decision is an administrative review action,
  **not external verification of truth**. Fake or negligent reviews
  can promote false claims. Automatic acceptance based on an LLM
  score is deliberately not implemented.
- Inference and contradiction resolution remain simplified. Two
  different phrasings of the same underlying entity may not compare
  as identical; resolution is literal/casefold-based.
- Claims currently support four relations and polarity, not full
  modal, temporal, attributed, probabilistic or source-trust semantics.
- Source-span validity cannot ensure the correct entity boundary or
  predicate. The conservative extractor is heuristic and can still err.
- Rejection and retraction never delete the source observation,
  which remains visible in `history()`.
- Neither single-exposure memory binding nor LLM extraction proves
  new concept induction or human-like reasoning.

## Experimental next steps

1. Build a versioned durable, replayable event store with atomic write
   semantics and deliberate migration. Validate recovery after process
   restart, duplicate IDs, malformed events, and partial writes.
2. Extend attributed, modal and time-qualified evidence rather than
   silently collapsing it into current categorical `is_a` claims.
3. Re-evaluate against independently labeled, permissioned new data,
   including recall and false refusal cost, reviewed/contested graph
   inference and correction scenarios.
4. Compare model-only, existing hybrid and the post-hoc conservative
   gate under matched hardware, confidence and computation conditions.

The [v0.5 local report](RESULTS_V05_2026-09-30.md) remains the
historical record; later heuristic changes must not be rebranded as
independent success on the same examples.
