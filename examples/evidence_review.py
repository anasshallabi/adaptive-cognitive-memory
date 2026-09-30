"""Demonstrate a one-exposure candidate and subsequent explicit review.

Run with no dependencies, API, Ollama, GPU or model downloads:
    python -m examples.evidence_review
"""
import json

from acm.evidence_ledger import EvidenceLedger
from acm.text_extraction import RuleExtractor


def main() -> None:
    ledger = EvidenceLedger()
    extractor = RuleExtractor()
    events = []

    first = ledger.observe("Zentra est une marque.", extractor,
                           source="fictional-catalog-1")
    events.append({"step": "one exposure", "record": ledger.record(first),
                   "review_status": ledger.query("Zentra", "is_a", "marque")})

    ledger.decide(first, decision="accept", actor="human-reviewer",
                  reason="Reviewed the wording of catalog-1's assertion")
    events.append({"step": "explicit acceptance",
                   "review_status": ledger.query("Zentra", "is_a", "marque"),
                   "approved_graph": ledger.approved_memory().about("Zentra")})

    second = ledger.observe("Zentra n'est pas une marque.", extractor,
                            source="fictional-catalog-2")
    events.append({"step": "contradictory candidate",
                   "review_status": ledger.query("Zentra", "is_a", "marque"),
                   "approved_graph": ledger.approved_memory().about("Zentra")})

    ledger.decide(second, decision="reject", actor="human-reviewer",
                  reason="Second source is a mistaken assertion")
    ledger.decide(first, decision="retract", actor="human-reviewer",
                  reason="First source later corrected its own assertion")
    events.append({"step": "correction and retraction",
                   "review_status": ledger.query("Zentra", "is_a", "marque"),
                   "approved_graph": ledger.approved_memory().about("Zentra")})

    print(json.dumps({
        "demonstration": events,
        "event_history": ledger.history(),
        "limitations": (
            "Illustrative human-reviewed status only; no world truth "
            "verification, durable storage, or autonomous concept induction."
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
