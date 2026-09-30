"""Experimental v0.3 textual fact binding, provenance and transparent inference.

This is a small *controlled-language parser*, NOT general language understanding.
It never decides whether a source is trustworthy or a claim is true.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re


def _space(text: str) -> str:
    return " ".join(text.strip().split())


def _key(text: str) -> str:
    return _space(text).casefold()


def _strip_indefinite_subject(text: str) -> str:
    """Normalize generic class phrases: 'Une marque' -> 'marque'.

    Do NOT strip definite articles ('La Poste' is potentially a proper name).
    """
    return re.sub(r"^(?:un|une|a|an)\s+", "", _space(text), flags=re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    object: str
    positive: bool
    source: str
    text: str

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "positive": self.positive,
            "source": self.source,
            "text": self.text,
        }


# Intentionally narrow supported patterns; no hidden LLM inference.
_IS_A_FR = re.compile(
    r"^(?P<subject>.+?)\s+(?P<verb>n['’]est\s+pas|est)\s+(?:un|une)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_IS_A_EN = re.compile(
    r"^(?P<subject>.+?)\s+(?P<verb>is\s+not|is)\s+(?:a|an)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_PROPERTY_FR = re.compile(
    r"^(?P<subject>.+?)\s+(?:(?P<ne>ne)\s+)?"
    r"(?P<verb>fabrique|possède|utilise)\s+(?(ne)pas\s+)?(?P<object>.+)$",
    re.IGNORECASE,
)
_PROPERTY_EN = re.compile(
    r"^(?P<subject>.+?)\s+(?:(?P<negative>does\s+not)\s+)?"
    r"(?P<verb>makes|has|uses|make|have|use)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_PREDICATES_FR = {"fabrique": "makes", "possède": "has", "utilise": "uses"}
_PREDICATES_EN = {
    "makes": "makes", "make": "makes",
    "has": "has", "have": "has",
    "uses": "uses", "use": "uses",
}


def parse_claim(text: str, *, source: str = "user") -> Claim:
    """Extract ONE statement from the documented FR/EN controlled grammar.

    Unsupported expressions raise ValueError rather than guessing a fact.
    """
    original = _space(text)
    if not original or not _space(source):
        raise ValueError("A nonempty sentence and source are required")
    if original.endswith("?") or "\n" in text:
        raise ValueError("Only a single declarative sentence is supported")
    sentence = original.rstrip(".").strip()
    if not sentence:
        raise ValueError("Empty sentence")
    for pattern, predicate in ((_IS_A_FR, "is_a"), (_IS_A_EN, "is_a")):
        match = pattern.fullmatch(sentence)
        if match:
            subject = _strip_indefinite_subject(match["subject"])
            obj = _space(match["object"])
            if not subject or not obj:
                raise ValueError("Subject and object are required")
            negative = "pas" in match["verb"].casefold() or "not" in match["verb"].casefold()
            return Claim(subject, predicate, obj, not negative, _space(source), original)
    for pattern, names in ((_PROPERTY_FR, _PREDICATES_FR), (_PROPERTY_EN, _PREDICATES_EN)):
        match = pattern.fullmatch(sentence)
        if match:
            subject = _strip_indefinite_subject(match["subject"])
            obj = _space(match["object"])
            if not subject or not obj:
                raise ValueError("Subject and object are required")
            negative = bool(match.groupdict().get("ne") or match.groupdict().get("negative"))
            return Claim(
                subject, names[match["verb"].casefold()], obj,
                not negative, _space(source), original
            )
    raise ValueError("Unsupported syntax; see docs/TEXT.md for the controlled grammar")


@dataclass
class TextMemory:
    """A small auditable claim graph with source records and limited inference.

    Queries report support/negation *in stored claims*, never objective truth.
    """
    claims: list[Claim] = field(default_factory=list)

    def learn_text(self, sentence: str, *, source: str = "user") -> dict:
        claim = parse_claim(sentence, source=source)
        # Duplicate from the *same* source adds no further evidence.
        signature = (claim.source, _key(claim.subject), claim.predicate,
                     _key(claim.object), claim.positive)
        duplicate = any(
            (c.source, _key(c.subject), c.predicate, _key(c.object), c.positive)
            == signature for c in self.claims
        )
        if not duplicate:
            self.claims.append(claim)
        return {"status": "duplicate" if duplicate else "stored", "claim": claim.as_dict()}

    def about(self, subject: str) -> list[dict]:
        """Direct stored claims only, including contradictions and sources."""
        return [c.as_dict() for c in self.claims if _key(c.subject) == _key(subject)]

    def query(self, subject: str, predicate: str, obj: str, *, max_hops: int = 8) -> dict:
        """Evidence for a claim; 'is_a' supports positive multi-hop transitivity.

        Only direct negative claims are supported. Other predicates are direct.
        All inferences preserve their complete supporting claim path.
        """
        if not _key(subject) or not _key(obj) or not predicate:
            raise ValueError("Nonempty subject, predicate and object required")
        if max_hops < 1:
            raise ValueError("max_hops must be positive")
        target_subject, target_obj = _key(subject), _key(obj)
        negatives = [
            {"path": [c.as_dict()], "inferred": False}
            for c in self.claims
            if _key(c.subject) == target_subject
            and c.predicate == predicate and _key(c.object) == target_obj
            and not c.positive
        ]
        positive_paths: list[list[Claim]] = []
        if predicate == "is_a":
            # Simple BFS bounded by max_hops, preventing cycles per path.
            queue = deque([(target_subject, [], {target_subject})])
            while queue:
                node, path, visited = queue.popleft()
                if len(path) >= max_hops:
                    continue
                for c in self.claims:
                    if c.predicate != "is_a" or not c.positive or _key(c.subject) != node:
                        continue
                    next_key = _key(c.object)
                    new_path = path + [c]
                    if next_key == target_obj:
                        positive_paths.append(new_path)
                    if next_key not in visited:
                        queue.append((next_key, new_path, visited | {next_key}))
        else:
            positive_paths = [
                [c] for c in self.claims
                if _key(c.subject) == target_subject and c.predicate == predicate
                and _key(c.object) == target_obj and c.positive
            ]
        positives = [
            {"path": [c.as_dict() for c in path], "inferred": len(path) > 1}
            for path in positive_paths
        ]
        if positives and negatives:
            status = "conflict"
        elif positives:
            status = "supported"
        elif negatives:
            status = "negated"
        else:
            status = "unknown"
        return {
            "subject": subject, "predicate": predicate, "object": obj,
            "status": status, "positive_evidence": positives,
            "negative_evidence": negatives,
            "note": "Evidence is from stored claims, not verified external truth.",
        }

    def ask(self, question: str) -> dict:
        """Small FR/EN question interface; unsupported questions are rejected."""
        q = _space(question).rstrip("?").strip()
        for pattern in (
            r"^Est-ce que\s+(.+?)\s+est\s+(?:un|une)\s+(.+)$",
            r"^Is\s+(.+?)\s+(?:a|an)\s+(.+)$",
        ):
            m = re.fullmatch(pattern, q, flags=re.IGNORECASE)
            if m:
                return self.query(m.group(1), "is_a", m.group(2))
        for pattern in (
            r"^Que sais-tu de\s+(.+)$",
            r"^What do you know about\s+(.+)$",
        ):
            m = re.fullmatch(pattern, q, flags=re.IGNORECASE)
            if m:
                return {"subject": m.group(1), "facts": self.about(m.group(1))}
        raise ValueError("Unsupported question; see docs/TEXT.md")
