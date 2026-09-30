"""In-process append-only evidence review, separate from language extraction.

A claim enters as a CANDIDATE. It cannot become an accepted graph edge
without an explicit review decision. Approval means "reviewed by an actor",
not "objectively verified". Rejected/retracted proposals remain in an audit
log and are excluded from graph inference.

This is a tiny research prototype, NOT a durable/secure audit system:
no disk persistence, authentication, concurrency, or independent fact checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .text_memory import Claim, TextMemory


_PREDICATES = frozenset({"is_a", "makes", "has", "uses"})


class Extractor(Protocol):
    def extract(self, text: str, *, source: str = "user") -> Claim | None: ...


def _key(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return " ".join(value.split())


@dataclass(frozen=True)
class EvidenceEvent:
    sequence: int
    record_id: int
    action: str
    actor: str
    reason: str
    claim: Claim | None = None

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "record_id": self.record_id,
            "action": self.action,
            "actor": self.actor,
            "reason": self.reason,
            "claim": self.claim.as_dict() if self.claim is not None else None,
        }


class EvidenceLedger:
    """Reviewable claim registry; all IDs and event ordering are local-only."""

    def __init__(self) -> None:
        self._events: list[EvidenceEvent] = []
        self._claims: dict[int, Claim] = {}
        self._states: dict[int, str] = {}
        self._proposers: dict[int, str] = {}
        self._next_id = 1

    def _append(self, record_id: int, action: str, actor: str,
                reason: str, claim: Claim | None = None) -> None:
        self._events.append(EvidenceEvent(
            len(self._events) + 1, record_id, action, actor, reason, claim,
        ))

    def propose(self, claim: Claim, *, actor: str = "extractor") -> int:
        """Register one *unverified* extracted assertion for later review."""
        actor = _nonempty(actor, "actor")
        if not isinstance(claim, Claim):
            raise ValueError("Expected a Claim instance")
        if any(not isinstance(s, str) or not s.strip() for s in
               (claim.subject, claim.object, claim.source, claim.text)):
            raise ValueError("Claim fields must be nonempty strings")
        if claim.predicate not in _PREDICATES or type(claim.positive) is not bool:
            raise ValueError("Unsupported predicate or polarity")
        record_id = self._next_id
        self._next_id += 1
        self._claims[record_id] = claim
        self._states[record_id] = "candidate"
        self._proposers[record_id] = actor
        self._append(record_id, "propose", actor, "unverified extraction", claim)
        return record_id

    def observe(self, sentence: str, extractor: Extractor, *,
                source: str = "user", actor: str = "extractor") -> int | None:
        """Extract into a candidate, never directly into the trusted graph.

        Returns None for abstention; propagates extraction errors.
        This checks text/source correspondence, not semantic correctness.
        """
        original = _nonempty(sentence, "sentence")
        src = _nonempty(source, "source")
        claim = extractor.extract(original, source=src)
        if claim is None:
            return None
        if not isinstance(claim, Claim) or claim.source != src or claim.text != original:
            raise ValueError("Extractor changed input text/source provenance")
        return self.propose(claim, actor=actor)

    def decide(self, record_id: int, *, decision: str, actor: str, reason: str) -> None:
        """Accept or reject a pending candidate, or retract accepted evidence.

        Rejecting a candidate is different from recording a negative fact.
        To record negative evidence, propose a Claim with positive=False.
        """
        actor = _nonempty(actor, "actor")
        reason = _nonempty(reason, "reason")
        if type(record_id) is not int or record_id not in self._states:
            raise KeyError("Unknown record ID")
        current = self._states[record_id]
        if decision in ("accept", "reject"):
            if current != "candidate":
                raise ValueError("Only candidate records may be reviewed")
            self._states[record_id] = "accepted" if decision == "accept" else "rejected"
        elif decision == "retract":
            if current != "accepted":
                raise ValueError("Only accepted records may be retracted")
            self._states[record_id] = "retracted"
        else:
            raise ValueError("Decision must be accept, reject or retract")
        self._append(record_id, decision, actor, reason)

    def record(self, record_id: int) -> dict:
        if type(record_id) is not int or record_id not in self._claims:
            raise KeyError("Unknown record ID")
        return {
            "record_id": record_id,
            "status": self._states[record_id],
            "proposed_by": self._proposers[record_id],
            "claim": self._claims[record_id].as_dict(),
            "events": [e.as_dict() for e in self._events if e.record_id == record_id],
        }

    def history(self) -> list[dict]:
        """Return snapshots of every event, including retracted evidence."""
        return [event.as_dict() for event in self._events]

    def _matching(self, subject: str, predicate: str, obj: str) -> list[int]:
        return [record_id for record_id, claim in self._claims.items()
                if (_key(claim.subject), claim.predicate, _key(claim.object)) ==
                (_key(subject), predicate, _key(obj))]

    def query(self, subject: str, predicate: str, obj: str) -> dict:
        """Summarize *review state*, not objective truth or transitive inference.

        A pending opposite claim blocks a definitive reviewed status.
        A rejected or retracted claim provides no active evidence.
        """
        subject = _nonempty(subject, "subject")
        predicate = _nonempty(predicate, "predicate")
        obj = _nonempty(obj, "object")
        matching = self._matching(subject, predicate, obj)
        active = [(i, self._claims[i]) for i in matching
                  if self._states[i] == "accepted"]
        pending = [(i, self._claims[i]) for i in matching
                   if self._states[i] == "candidate"]
        positive = [i for i, c in active if c.positive]
        negative = [i for i, c in active if not c.positive]
        if positive and negative:
            status = "disputed"
        elif positive and any(not c.positive for _, c in pending):
            status = "under_review"
        elif negative and any(c.positive for _, c in pending):
            status = "under_review"
        elif positive:
            status = "supported"
        elif negative:
            status = "negated"
        elif pending:
            status = "candidate"
        else:
            status = "unknown"
        return {
            "status": status,
            "accepted_positive_ids": positive,
            "accepted_negative_ids": negative,
            "candidate_ids": [i for i, _ in pending],
            "note": "Review status of source claims, not verified external truth.",
        }

    def approved_memory(self) -> TextMemory:
        """Build a *new* graph with unopposed accepted claims only.

        If a candidate/accepted opposite polarity exists for a triple,
        neither polarity enters the inference graph until review resolves.
        This is conservative, not sound general epistemic reasoning.
        """
        approved = TextMemory()
        for record_id, claim in self._claims.items():
            if self._states[record_id] != "accepted":
                continue
            opposite = any(
                i != record_id and self._states[i] in ("candidate", "accepted")
                and c.positive != claim.positive
                for i, c in ((j, self._claims[j]) for j in
                             self._matching(claim.subject, claim.predicate, claim.object))
            )
            if not opposite:
                approved.remember_claim(claim)
        return approved
