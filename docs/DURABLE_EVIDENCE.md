# ACM v0.8: persistent, reviewable evidence with SQLite

This prototype adds durable local event storage to the in-process
[review ledger](EVIDENCE_LEDGER.md). The earlier, memory-only API remains
available and unchanged.

## Principle

Observation creates a **candidate**, not an accepted truth. Explicit
decisions `accept`, `reject` and `retract` are recorded in order,
with a reviewing actor, a reason, source text and a UTC timestamp.
Only accepted, non-opposed relations enter the approved knowledge graph.

The ledger uses Python's bundled `sqlite3`; no extra database server
or API key is required. Each write acquires a SQLite transaction,
replays the current event state and commits the new event atomically.
On reopening a database, events are replayed and review status,
contradictions and the approved graph are reconstructed.

## Commands on Windows

```powershell
git pull origin main
python -m unittest discover -s tests -v

# Restart/review example, SQLite file automatically discarded afterward:
python -m examples.durable_review

# Keep the database on your PC for inspection and reopening:
python -m examples.durable_review --db review.sqlite
```

If `--db` is used repeatedly, each run adds a new illustrative
sequence. SQLite files are excluded from Git by `.gitignore`.
Use distinct paths for independent experiments.

## API example

```python
from acm.durable_evidence import SQLiteEvidenceLedger
from acm.text_extraction import RuleExtractor

ledger = SQLiteEvidenceLedger("research.sqlite")
record_id = ledger.observe(
    "Zentra est une marque.", RuleExtractor(), source="catalog-1"
)
assert ledger.record(record_id)["status"] == "candidate"

# This is an intentional reviewer decision, not a truth prediction.
ledger.decide(record_id, decision="accept",
              actor="reviewer", reason="The catalog explicitly asserts this")
restarted = SQLiteEvidenceLedger("research.sqlite")
assert restarted.record(record_id)["status"] == "accepted"
assert restarted.query("Zentra", "is_a", "marque")["status"] == "supported"
```

## Integrity and threat model

- SQLite transaction commits protect against partially written events
  during ordinary program interruption. Rejected transitions do not
  write a partial event.
- Each operation reloads the current committed state; two instances
  do not rely on stale in-memory decisions.
- The loader checks stored claim fields, event order, review transitions
  and schema version. Malformed event histories cause an error rather
  than silently rebuilding different knowledge.
- **Not tamper-proof:** a person with write access to the database
  can alter it. This is not a secure event journal or signed record.
- No user authentication, access control, schema migrations,
  backups or remote synchronization. Review actor names are supplied
  by the caller and are **not** verified.
- No automatic evidence credibility or real-world truth checking.
  An accepted statement means its source assertion was reviewed.
- Conflicting claims temporarily disappear from the approved
  reasoning graph; all prior events remain in the audit history.
- The existing four predicates, entity labels and basic `is_a`
  transitivity remain a deliberately limited knowledge representation.
- A SQLite database can contain sensitive source text: keep it
  on a controlled local path and out of public Git repositories.

## Research status

v0.6 produced **21/28** exact answers on the previously studied
v0.5 synthetic test examples, compared with **13/28** in the original
hybrid. Because the guard was developed after inspecting errors
on that very set, this was a *post-hoc regression*, **not a blind
generalization result**. v0.8 concerns **persistence and correction
engineering**, not better NLP extraction or new concept induction.

Future studies need independent evidence sources, explicit source
credibility policies, temporal propositions and controlled comparisons.
