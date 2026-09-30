"""Conservative pre-memory screening for simple asserted claims.

A deliberately small heuristic *safety baseline*, NOT semantic understanding
or a guarantee that accepted statements are factual. Reject high-risk
epistemic/temporal/multi-clause expressions before they become stored triples.

Applied to an *optional new extractor*, never to historical v0.5 baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .text_memory import Claim


@dataclass(frozen=True)
class SafetyDecision:
    outcome: str  # "allow" or "abstain"
    reason: str

    @property
    def allowed(self) -> bool:
        return self.outcome == "allow"


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


# Cautiously reject anything our present source/polarity-only Claim schema
# cannot faithfully represent. Patterns are limited and can miss variants.
_QUESTIONS = (r"[?？]",)
_DIRECTIVES = (
    r"^(?:ignore|imagine|suppose|pretend|please|write|tell|imaginez|"
    r"ignorez|imagine|supposons|suppose|pense\s+à|réponds|répondez)\b",
)
_ATTRIBUTION = (
    r"^(?:according to|as reported by|selon|d['’]après)\b",
    r"\b(?:is considered|est considéré(?:e)?\s+comme|"
    r"est réputé(?:e)?\s+être)\b",
)
_UNCERTAINTY = (
    r"\b(?:might|may|could|perhaps|probably|possibly|"
    r"peut-être|pourrait|pourraient|probablement|éventuellement)\b",
)
_HISTORY = (
    r"\b(?:formerly|previously|used to|in the past|"
    r"était|étaient|autrefois|auparavant|jadis)\b",
    r"\b(?:in|en)\s+(?:19|20)\d{2}\b",
    r"\b(?:was|were)\s+(?:a|an)\b",
)
_MULTI_CLAUSE = (
    r"\b(?:and|et)\s+(?:(?:it|elle|il|ils|elles|they|"
    r"he|she|on)\s+)?(?:is|are|has|have|makes|make|uses|use|"
    r"builds|produces|est|sont|a|ont|fabrique|"
    r"fabriquent|utilise|utilisent|possède|possèdent)\b",
)


def assess_sentence(text: str) -> SafetyDecision:
    """Return a conservative structural decision, NOT external verification."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty sentence required")
    sentence = " ".join(text.strip().split())
    for reason, patterns in (
        ("question", _QUESTIONS),
        ("directive", _DIRECTIVES),
        ("attribution", _ATTRIBUTION),
        ("uncertainty", _UNCERTAINTY),
        ("historical", _HISTORY),
        ("multiple_propositions", _MULTI_CLAUSE),
    ):
        if _matches_any(sentence, patterns):
            return SafetyDecision("abstain", reason)
    return SafetyDecision("allow", "no_known_epistemic_warning")


def rule_is_canonical(sentence: str, claim: Claim) -> bool:
    """Narrow rule acceptance for unqualified subjects.

    Avoid silently storing subjects like "La société Alice", "Bob currently",
    "Une agence nommée Bob" as distinct entities. If uncertain, use the
    optional local extractor instead; that extractor can *also* be wrong.
    Generic class statements beginning 'Un(e)/A(n)' remain available.
    """
    if not claim.subject.strip():
        return False
    raw = sentence.strip()
    if re.search(r"\b(?:named|called|nommé(?:e)?|appelé(?:e)?)\b", raw, re.IGNORECASE):
        return False
    if re.match(r"^(?:un|une|a|an)\s+", raw, re.IGNORECASE):
        return True
    return len(claim.subject.split()) == 1
