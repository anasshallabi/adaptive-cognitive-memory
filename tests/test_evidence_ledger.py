"""Evidence-ledger review and correction tests (no local LLM needed)."""
import unittest

from acm.evidence_ledger import EvidenceLedger
from acm.text_memory import Claim
from acm.text_extraction import RuleExtractor


def claim(positive=True, source="source-a", *, subject="Zentra",
          predicate="is_a", obj="car brand", sentence=None):
    return Claim(subject, predicate, obj, positive, source,
                 sentence or f"{subject} is a car brand.")


class BrokenExtractor:
    def extract(self, text, *, source="user"):
        return Claim("Zentra", "is_a", "brand", True, "invented-source", text)


class AbstainingExtractor:
    def extract(self, text, *, source="user"):
        return None


class TestEvidenceLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger()

    def test_new_candidate_is_not_approved_automatically(self):
        record_id = self.ledger.propose(claim())
        self.assertEqual(record_id, 1)
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"], "candidate")
        self.assertEqual(self.ledger.approved_memory().claims, [])
        self.assertEqual(self.ledger.record(record_id)["status"], "candidate")

    def test_accept_requires_reason_and_explicit_reviewer(self):
        record_id = self.ledger.propose(claim())
        with self.assertRaises(ValueError):
            self.ledger.decide(record_id, decision="accept", actor="reviewer", reason=" ")
        self.ledger.decide(record_id, decision="accept", actor="reviewer",
                           reason="Verified that this claim is what the source asserted")
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"], "supported")
        self.assertEqual(self.ledger.approved_memory().query(
            "Zentra", "is_a", "car brand")["status"], "supported")
        self.assertEqual(self.ledger.record(record_id)["events"][1]["actor"], "reviewer")

    def test_rejection_does_not_mean_negative_evidence(self):
        record_id = self.ledger.propose(claim())
        self.ledger.decide(record_id, decision="reject", actor="reviewer",
                           reason="Source is a speculation")
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"], "unknown")
        self.assertEqual(self.ledger.approved_memory().claims, [])
        self.assertEqual(self.ledger.record(record_id)["status"], "rejected")

    def test_negative_evidence_is_different_from_reject(self):
        rec = self.ledger.propose(claim(positive=False))
        self.ledger.decide(rec, decision="accept", actor="reviewer",
                           reason="Source clearly asserts negation")
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"], "negated")
        self.assertEqual(self.ledger.approved_memory().query(
            "Zentra", "is_a", "car brand")["status"], "negated")

    def test_pending_opposition_blocks_graph_approval(self):
        pos = self.ledger.propose(claim(positive=True, source="A"))
        self.ledger.decide(pos, decision="accept", actor="reviewer",
                           reason="Literal source assertion")
        neg = self.ledger.propose(claim(positive=False, source="B"))
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"],
                         "under_review")
        self.assertEqual(self.ledger.approved_memory().claims, [])
        self.ledger.decide(neg, decision="reject", actor="reviewer",
                           reason="Contradictory extract not grounded")
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"],
                         "supported")

    def test_accepted_contradiction_is_disputed_not_silently_overwritten(self):
        a = self.ledger.propose(claim(positive=True, source="report-a"))
        b = self.ledger.propose(claim(positive=False, source="report-b"))
        self.ledger.decide(a, decision="accept", actor="reader",
                           reason="Reporter A's assertion")
        self.ledger.decide(b, decision="accept", actor="reader",
                           reason="Reporter B's contradiction")
        result = self.ledger.query("Zentra", "is_a", "car brand")
        self.assertEqual(result["status"], "disputed")
        self.assertEqual(result["accepted_positive_ids"], [a])
        self.assertEqual(result["accepted_negative_ids"], [b])
        self.assertEqual(self.ledger.approved_memory().claims, [])

    def test_retraction_preserves_audit_trail(self):
        first = self.ledger.propose(claim(source="legacy"))
        self.ledger.decide(first, decision="accept", actor="reviewer",
                           reason="Originally accepted")
        self.ledger.decide(first, decision="retract", actor="reviewer",
                           reason="Later source correction")
        self.assertEqual(self.ledger.query("Zentra", "is_a", "car brand")["status"],
                         "unknown")
        self.assertEqual([e["action"] for e in self.ledger.record(first)["events"]],
                         ["propose", "accept", "retract"])
        self.assertEqual([e["sequence"] for e in self.ledger.history()], [1, 2, 3])
        self.assertEqual(self.ledger.approved_memory().claims, [])

    def test_review_state_transitions_are_strict(self):
        rec = self.ledger.propose(claim())
        with self.assertRaises(ValueError):
            self.ledger.decide(rec, decision="retract", actor="r", reason="not accepted")
        self.ledger.decide(rec, decision="reject", actor="r", reason="not a claim")
        with self.assertRaises(ValueError):
            self.ledger.decide(rec, decision="accept", actor="r", reason="too late")
        with self.assertRaises(KeyError):
            self.ledger.decide(9999, decision="accept", actor="r", reason="unknown")
        with self.assertRaises(ValueError):
            self.ledger.decide(rec, decision="forget", actor="r", reason="unknown decision")

    def test_observe_from_grammar_creates_candidate(self):
        rec = self.ledger.observe(
            "Zentra is a car brand.", RuleExtractor(), source="receipt-007",
        )
        self.assertEqual(self.ledger.record(rec)["claim"]["source"], "receipt-007")
        self.assertEqual(self.ledger.record(rec)["status"], "candidate")
        self.assertEqual(self.ledger.approved_memory().claims, [])

    def test_observe_abstention_creates_no_event(self):
        self.assertIsNone(self.ledger.observe(
            "A complicated unsupported sentence", AbstainingExtractor()))
        self.assertEqual(self.ledger.history(), [])

    def test_extractors_cannot_spoof_provenance(self):
        with self.assertRaises(ValueError):
            self.ledger.observe("Zentra is a brand.", BrokenExtractor(),
                                source="original")
        self.assertEqual(self.ledger.history(), [])

    def test_inference_restricted_to_accepted_edges(self):
        first = self.ledger.propose(claim(subject="Zentra", obj="car brand"))
        second = self.ledger.propose(claim(
            subject="car brand", obj="organization", source="source-b",
            sentence="A car brand is an organization.",
        ))
        self.ledger.decide(first, decision="accept", actor="reviewer",
                           reason="Reviewed first claim")
        self.assertEqual(self.ledger.approved_memory().query(
            "Zentra", "is_a", "organization")["status"], "unknown")
        self.ledger.decide(second, decision="accept", actor="reviewer",
                           reason="Reviewed parent category")
        self.assertEqual(self.ledger.approved_memory().query(
            "Zentra", "is_a", "organization")["status"], "supported")

    def test_pending_opposition_blocks_inferred_chain(self):
        a = self.ledger.propose(claim(subject="Zentra", obj="car brand"))
        b = self.ledger.propose(claim(
            subject="car brand", obj="organization", source="b",
            sentence="A car brand is an organization.",
        ))
        self.ledger.decide(a, decision="accept", actor="r", reason="reviewed a")
        self.ledger.decide(b, decision="accept", actor="r", reason="reviewed b")
        opposite = self.ledger.propose(claim(
            positive=False, subject="Zentra", obj="car brand", source="c",
            sentence="Zentra is not a car brand.",
        ))
        self.assertEqual(self.ledger.approved_memory().query(
            "Zentra", "is_a", "organization")["status"], "unknown")
        self.ledger.decide(opposite, decision="reject", actor="r",
                           reason="Incorrect negation")
        self.assertEqual(self.ledger.approved_memory().query(
            "Zentra", "is_a", "organization")["status"], "supported")

    def test_invalid_claim_rejected_at_ledger_boundary(self):
        with self.assertRaises(ValueError):
            self.ledger.propose(claim(predicate="surfs"))
        with self.assertRaises(ValueError):
            self.ledger.propose(Claim("Zentra", "is_a", "brand", "yes", "s", "t"))
        with self.assertRaises(ValueError):
            self.ledger.propose(claim(source=""))

    def test_public_history_is_snapshot_not_mutable_internal_state(self):
        rec = self.ledger.propose(claim())
        history = self.ledger.history()
        history[0]["action"] = "accept"
        history.clear()
        self.assertEqual(self.ledger.record(rec)["events"][0]["action"], "propose")


if __name__ == "__main__":
    unittest.main()
