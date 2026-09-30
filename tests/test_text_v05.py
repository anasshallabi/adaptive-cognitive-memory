"""Frozen v0.5 synthetic claim-evaluation protocol checks (no live Ollama).

These tests ensure structure and provenance, not LLM accuracy.
"""
import unittest
from pathlib import Path

from acm.text_benchmark import read_cases, run_benchmark
from acm.text_extraction import OllamaExtractor, RuleExtractor


MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks" / "text_v05.jsonl"


class TestFrozenNewEvaluation(unittest.TestCase):
    def test_new_protocol_has_separate_splits(self):
        cases = read_cases(MANIFEST)
        self.assertEqual(len(cases), 40)
        self.assertEqual(sum(c.split == "dev" for c in cases), 12)
        self.assertEqual(sum(c.split == "test" for c in cases), 28)
        self.assertEqual(sum(c.split == "test" and c.expected is not None
                             for c in cases), 18)
        self.assertEqual(sum(c.split == "test" and c.expected is None
                             for c in cases), 10)

    def test_all_examples_have_unique_fictional_case_ids(self):
        cases = read_cases(MANIFEST)
        self.assertEqual(len({c.case_id for c in cases}), len(cases))
        self.assertTrue(all(c.case_id.startswith("v05-") for c in cases))

    def test_reported_and_modal_cases_must_abstain(self):
        cases = read_cases(MANIFEST)
        ids = {"v05-t19", "v05-t20", "v05-t21", "v05-t25",
               "v05-t26", "v05-t27"}
        selected = [c for c in cases if c.case_id in ids]
        self.assertEqual(len(selected), len(ids))
        self.assertTrue(all(c.expected is None for c in selected))

    def test_rules_smoke_run_does_not_raise(self):
        cases = read_cases(MANIFEST)
        result = run_benchmark(cases, RuleExtractor(), split="dev")
        self.assertEqual(result["count"], 12)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["route_stats"]["rules"]["count"], 12)

    def test_optional_literal_mode_keeps_existing_schema(self):
        requests = []
        def mocked_ollama(payload):
            requests.append(payload)
            return {"message": {"content": '{"status":"extracted","subject":"Nuvora",'
                                           '"predicate":"is_a","object":"robotics company",'
                                           '"positive":true}'}}
        extractor = OllamaExtractor(
            "qwen3:14b", transport=mocked_ollama, think=False,
            prompt_mode="literal",
        )
        result = extractor.extract("Nuvora is a robotics company.")
        self.assertEqual(result.object, "robotics company")
        self.assertEqual(requests[0]["think"], False)
        self.assertIn("status", requests[0]["format"]["properties"])
        self.assertEqual(extractor.last_metadata["structured_status"], "extracted")


if __name__ == "__main__":
    unittest.main()
