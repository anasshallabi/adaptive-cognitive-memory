"""Unit tests for modality probes, with no LLM or API dependency."""
import unittest

from acm.text_extraction import ExtractionError, RuleExtractor
from acm.text_memory import Claim
from examples.probe_text import PROBES, run_probes


class StubExtractor:
    def extract(self, sentence: str, *, source: str = "user"):
        if "considered" in sentence:
            return None
        if "might" in sentence:
            raise ExtractionError("test failure")
        return Claim("Zentra", "is_a", "car brand", True, source, sentence)


class TestModalityProbe(unittest.TestCase):
    def test_rule_only(self):
        results = run_probes([PROBES["direct"], PROBES["attributed"]])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rule_result"]["predicate"], "is_a")
        self.assertIsNone(results[1]["rule_result"])
        self.assertEqual(results[0]["model_status"], "not_run")
        self.assertIsNone(results[0]["model_latency_ms"])

    def test_model_is_independently_run_on_direct_claim(self):
        results = run_probes([PROBES["direct"]], model=StubExtractor())
        self.assertEqual(results[0]["model_status"], "extracted")
        self.assertEqual(results[0]["model_result"]["source"], "probe:1")
        self.assertGreaterEqual(results[0]["model_latency_ms"], 0)

    def test_model_abstention_is_recorded(self):
        result = run_probes([PROBES["attributed"]], model=StubExtractor())[0]
        self.assertEqual(result["model_status"], "abstained")
        self.assertIsNone(result["model_result"])
        self.assertIsNone(result["model_error"])

    def test_model_failure_is_not_reported_as_abstention(self):
        result = run_probes([PROBES["hedged"]], model=StubExtractor())[0]
        self.assertEqual(result["model_status"], "error")
        self.assertIn("ExtractionError", result["model_error"])
        self.assertIsNone(result["model_result"])

    def test_existing_baseline_unchanged(self):
        self.assertIsNone(RuleExtractor().extract(PROBES["attributed"]))
        self.assertEqual(RuleExtractor().extract(PROBES["direct"]).predicate, "is_a")


if __name__ == "__main__":
    unittest.main()
