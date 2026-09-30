"""Durable ACM event ledger: replay, restart, transactions and corruption guards."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from acm.durable_evidence import SQLiteEvidenceLedger
from acm.evidence_ledger import EvidenceLedger
from acm.text_extraction import RuleExtractor
from acm.text_memory import Claim


def assertion(*, positive=True, source="record-a", subject="Zentra",
              obj="marque") -> Claim:
    sentence = (
        f"{subject} est une {obj}." if positive
        else f"{subject} n'est pas une {obj}."
    )
    return Claim(subject, "is_a", obj, positive, source, sentence)


class WrongProvenance:
    def extract(self, sentence, *, source="user"):
        return Claim("Zentra", "is_a", "marque", True,
                     "spoofed-source", sentence)


class NoEvidence:
    def extract(self, sentence, *, source="user"):
        return None


class TestDurableEvidence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = Path(self.dir.name) / "review.sqlite"
        self.ledger = SQLiteEvidenceLedger(self.file)

    def reopen(self):
        return SQLiteEvidenceLedger(self.file)

    def test_candidate_survives_restart_but_not_in_approved_graph(self):
        record_id = self.ledger.propose(assertion())
        reopened = self.reopen()
        self.assertEqual(reopened.record(record_id)["status"], "candidate")
        self.assertEqual(reopened.query("Zentra", "is_a", "marque")["status"],
                         "candidate")
        self.assertEqual(reopened.approved_memory().claims, [])
        self.assertEqual(len(reopened.history()), 1)

    def test_acceptance_and_review_reopen_without_ollama(self):
        record_id = self.ledger.observe(
            "Zentra est une marque.", RuleExtractor(), source="newspaper"
        )
        self.assertIsNotNone(record_id)
        reopened = self.reopen()
        reopened.decide(record_id, decision="accept", actor="reviewer",
                        reason="Checked original source assertion")
        again = self.reopen()
        self.assertEqual(again.record(record_id)["status"], "accepted")
        self.assertEqual(again.query("Zentra", "is_a", "marque")["status"],
                         "supported")
        self.assertEqual(again.approved_memory().about("Zentra")[0]["source"],
                         "newspaper")
        self.assertEqual([e["action"] for e in again.history()],
                         ["propose", "accept"])
        self.assertTrue(all(e["timestamp_utc"].endswith("+00:00")
                            for e in again.history()))

    def test_pending_contradiction_and_correction_survive_restart(self):
        first = self.ledger.propose(assertion(source="A"))
        self.ledger.decide(first, decision="accept", actor="reviewer",
                           reason="Asserted in source A")
        second = self.reopen().propose(assertion(positive=False, source="B"))
        reviewing = self.reopen()
        self.assertEqual(reviewing.query("Zentra", "is_a", "marque")["status"],
                         "under_review")
        self.assertEqual(reviewing.approved_memory().claims, [])
        reviewing.decide(second, decision="accept", actor="reviewer",
                         reason="Contradictory source B")
        self.assertEqual(self.reopen().query(
            "Zentra", "is_a", "marque")["status"], "disputed")
        self.assertEqual(self.reopen().approved_memory().claims, [])
        self.reopen().decide(second, decision="retract", actor="reviewer",
                             reason="B's publisher issued a correction")
        self.assertEqual(self.reopen().query(
            "Zentra", "is_a", "marque")["status"], "supported")
        self.reopen().decide(first, decision="retract", actor="reviewer",
                             reason="A also corrected their assertion")
        self.assertEqual(self.reopen().query(
            "Zentra", "is_a", "marque")["status"], "unknown")
        self.assertEqual([event["action"] for event in self.reopen().history()],
                         ["propose", "accept", "propose", "accept",
                          "retract", "retract"])

    def test_existing_instances_reload_state_not_stale(self):
        left = self.ledger
        right = self.reopen()
        first = left.propose(assertion(source="A"))
        right.decide(first, decision="accept", actor="r",
                     reason="Review on the other instance")
        self.assertEqual(left.record(first)["status"], "accepted")
        # Both writers now see the latest candidate ID and status.
        second = left.propose(assertion(source="B"))
        self.assertEqual(second, 2)
        right.decide(second, decision="reject", actor="r",
                     reason="Not grounded")
        with self.assertRaises(ValueError):
            left.decide(second, decision="accept", actor="r",
                        reason="Must not accept already rejected claim")
        self.assertEqual(left.record(second)["status"], "rejected")

    def test_invalid_transition_does_not_write_event(self):
        first = self.ledger.propose(assertion())
        with self.assertRaises(ValueError):
            self.ledger.decide(first, decision="retract", actor="reviewer",
                               reason="Not yet accepted")
        with self.assertRaises(ValueError):
            self.ledger.decide(first, decision="accept", actor=" ",
                               reason="Missing actor")
        with self.assertRaises(KeyError):
            self.ledger.decide(555, decision="accept", actor="reviewer",
                               reason="Unknown record")
        self.assertEqual(len(self.reopen().history()), 1)
        self.assertEqual(self.reopen().record(first)["status"], "candidate")

    def test_rejected_claim_not_converted_to_negative(self):
        first = self.ledger.propose(assertion())
        self.ledger.decide(first, decision="reject", actor="reviewer",
                           reason="Not supported")
        self.assertEqual(self.reopen().query("Zentra", "is_a", "marque")["status"],
                         "unknown")
        self.assertEqual(self.reopen().approved_memory().claims, [])

    def test_inference_rebuilt_after_restart_and_retraction(self):
        a = self.ledger.propose(assertion(subject="Zentra", obj="marque"))
        b = self.ledger.propose(assertion(subject="marque", obj="organisation"))
        self.ledger.decide(a, decision="accept", actor="r",
                           reason="Reviewed a")
        self.ledger.decide(b, decision="accept", actor="r",
                           reason="Reviewed b")
        self.assertEqual(self.reopen().approved_memory().query(
            "Zentra", "is_a", "organisation")["status"], "supported")
        self.reopen().decide(a, decision="retract", actor="r",
                             reason="Correction of earlier classification")
        self.assertEqual(self.reopen().approved_memory().query(
            "Zentra", "is_a", "organisation")["status"], "unknown")

    def test_refused_extraction_is_not_persisted(self):
        self.assertIsNone(self.ledger.observe(
            "No supported sentence here", NoEvidence()))
        self.assertEqual(self.reopen().history(), [])

    def test_provenance_spoof_does_not_touch_disk(self):
        with self.assertRaises(ValueError):
            self.ledger.observe("Zentra est une marque.", WrongProvenance(),
                                source="genuine-user")
        self.assertEqual(self.reopen().history(), [])

    def test_malformed_event_is_detected_on_restart(self):
        # Deliberate direct DB corruption: no model content is needed to test it.
        with sqlite3.connect(self.file) as conn:
            conn.execute(
                "INSERT INTO evidence_events "
                "(record_id, action, actor, reason, claim_json, timestamp_utc)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (1, "propose", "tester", "unverified extraction",
                 '{"not-a-claim": 1}', "2026-09-30T00:00:00+00:00"),
            )
        with self.assertRaisesRegex(RuntimeError, "Invalid stored evidence event"):
            self.reopen()

    def test_schema_version_mismatch_is_rejected(self):
        with sqlite3.connect(self.file) as conn:
            conn.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(RuntimeError, "schema version"):
            self.reopen()

    def test_failed_creation_does_not_commit_partial_history(self):
        with self.assertRaises(ValueError):
            self.ledger.propose(assertion(source=" "))
        self.assertEqual(len(self.reopen().history()), 0)

    def test_database_created_as_local_file(self):
        self.assertTrue(self.file.is_file())
        with sqlite3.connect(self.file) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
        with self.assertRaises(ValueError):
            SQLiteEvidenceLedger(":memory:")

    def test_snapshot_is_copy_and_recorded_utc_times(self):
        first = self.ledger.propose(assertion())
        h = self.ledger.history()
        h[0]["action"] = "accept"
        h.clear()
        self.assertEqual(self.reopen().record(first)["status"], "candidate")
        self.assertIn("timestamp_utc", self.reopen().record(first)["events"][0])


if __name__ == "__main__":
    unittest.main()
