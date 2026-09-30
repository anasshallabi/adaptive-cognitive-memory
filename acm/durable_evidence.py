"""SQLite-backed, replayable version of ACM's explicit evidence ledger.

Only reviewed, unopposed claims enter the approved graph. Events are
committed atomically to a local SQLite file and replayed on each access.
This uses the Python standard library; no database server or paid API.

Not a secure audit service: anyone who can edit the SQLite file can
alter history. No login/auth, semantic verification or model training.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .evidence_ledger import EvidenceLedger, Extractor
from .text_memory import Claim, TextMemory


_SCHEMA_VERSION = 1
_CLAIM_FIELDS = frozenset({"subject", "predicate", "object", "positive", "source", "text"})
_SCHEMA = """
CREATE TABLE evidence_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER NOT NULL CHECK (record_id > 0),
    action TEXT NOT NULL CHECK (action IN ('propose','accept','reject','retract')),
    actor TEXT NOT NULL CHECK (length(trim(actor)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    claim_json TEXT,
    timestamp_utc TEXT NOT NULL,
    CHECK ((action = 'propose' AND claim_json IS NOT NULL)
        OR (action != 'propose' AND claim_json IS NULL))
)
"""


def _normalize(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty")
    return " ".join(value.split())


class SQLiteEvidenceLedger:
    """Explicit review with durable events and conflict-aware graph projection.

    Every operation uses a new SQLite connection. Writes acquire an IMMEDIATE
    transaction and replay the latest state *under the write lock*, preventing
    stale-state decisions between independent processes.
    """

    def __init__(self, database: str | Path):
        raw = str(database)
        if not raw.strip() or raw == ":memory:":
            raise ValueError("Use a nonempty persistent SQLite file path")
        self.database = Path(raw)
        if not self.database.parent.is_dir():
            raise ValueError("Parent directory of SQLite database must exist")
        self._initialize()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database), timeout=10, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize(self) -> None:
        with closing(self._open()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='evidence_events'"
                ).fetchone() is not None
                if version == 0 and not exists:
                    conn.execute(_SCHEMA)
                    conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                elif version != _SCHEMA_VERSION or not exists:
                    raise RuntimeError("Unrecognized evidence database schema version")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            self._replay(conn)  # Fail fast on malformed/inconsistent audit events.

    @staticmethod
    def _rows(conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute(
            "SELECT sequence, record_id, action, actor, reason, claim_json, "
            "timestamp_utc FROM evidence_events ORDER BY sequence"
        ).fetchall()

    @classmethod
    def _replay(cls, conn: sqlite3.Connection) -> tuple[EvidenceLedger, list[tuple]]:
        rows = cls._rows(conn)
        ledger = EvidenceLedger()
        for expected_sequence, row in enumerate(rows, start=1):
            seq, record_id, action, actor, reason, claim_json, stamp = row
            try:
                if seq != expected_sequence or not stamp:
                    raise ValueError("Noncontiguous event sequence or missing timestamp")
                if action == "propose":
                    data = json.loads(claim_json)
                    if not isinstance(data, dict) or set(data) != _CLAIM_FIELDS:
                        raise ValueError("Unexpected stored Claim fields")
                    if type(data["positive"]) is not bool:
                        raise ValueError("Invalid stored claim polarity")
                    claim = Claim(**data)
                    if reason != "unverified extraction":
                        raise ValueError("Invalid proposal reason")
                    created_id = ledger.propose(claim, actor=actor)
                    if created_id != record_id:
                        raise ValueError("Inconsistent record ID")
                else:
                    if claim_json is not None:
                        raise ValueError("Unexpected claim on review event")
                    ledger.decide(record_id, decision=action, actor=actor, reason=reason)
                if len(ledger.history()) != expected_sequence:
                    raise ValueError("Event history mismatch")
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Invalid stored evidence event at sequence {expected_sequence}"
                ) from exc
        return ledger, rows

    @staticmethod
    def _insert(conn: sqlite3.Connection, record_id: int, action: str,
                actor: str, reason: str, claim: Claim | None = None) -> None:
        claim_json = (
            json.dumps(claim.as_dict(), ensure_ascii=False, sort_keys=True)
            if claim is not None else None
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO evidence_events "
            "(record_id,action,actor,reason,claim_json,timestamp_utc) "
            "VALUES (?,?,?,?,?,?)",
            (record_id, action, actor, reason, claim_json, timestamp),
        )

    def propose(self, claim: Claim, *, actor: str = "extractor") -> int:
        with closing(self._open()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                ledger, _ = self._replay(conn)
                record_id = ledger.propose(claim, actor=actor)
                self._insert(conn, record_id, "propose", actor,
                             "unverified extraction", claim)
                conn.commit()
                return record_id
            except Exception:
                conn.rollback()
                raise

    def observe(self, sentence: str, extractor: Extractor, *,
                source: str = "user", actor: str = "extractor") -> int | None:
        # Extraction is outside the database transaction, so local inference
        # does not hold a lock for seconds. Only validated candidates persist.
        original = _normalize(sentence, "sentence")
        src = _normalize(source, "source")
        claim = extractor.extract(original, source=src)
        if claim is None:
            return None
        if not isinstance(claim, Claim) or claim.source != src or claim.text != original:
            raise ValueError("Extractor changed input text/source provenance")
        return self.propose(claim, actor=actor)

    def decide(self, record_id: int, *, decision: str, actor: str,
               reason: str) -> None:
        with closing(self._open()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                ledger, _ = self._replay(conn)
                ledger.decide(record_id, decision=decision, actor=actor, reason=reason)
                self._insert(conn, record_id, decision, actor, reason)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _snapshot(self) -> tuple[EvidenceLedger, list[tuple]]:
        with closing(self._open()) as conn:
            return self._replay(conn)

    def record(self, record_id: int) -> dict:
        ledger, rows = self._snapshot()
        data = ledger.record(record_id)
        stamps = {row[0]: row[6] for row in rows}
        for event in data["events"]:
            event["timestamp_utc"] = stamps[event["sequence"]]
        return data

    def history(self) -> list[dict]:
        ledger, rows = self._snapshot()
        events = ledger.history()
        for event, row in zip(events, rows):
            event["timestamp_utc"] = row[6]
        return events

    def query(self, subject: str, predicate: str, obj: str) -> dict:
        ledger, _ = self._snapshot()
        return ledger.query(subject, predicate, obj)

    def approved_memory(self) -> TextMemory:
        ledger, _ = self._snapshot()
        return ledger.approved_memory()
