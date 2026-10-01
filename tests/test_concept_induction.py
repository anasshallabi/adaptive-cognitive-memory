import unittest
from pathlib import Path

from acm.concept_induction import (
    ExactExemplarBaseline, JaccardBaseline, VersionSpaceLearner,
    evaluate_concept_episodes, read_concept_episodes,
)


DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "concept_v09.jsonl"


class TestConceptInduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = read_concept_episodes(DATASET)

    def test_frozen_dataset_shape(self):
        self.assertEqual(len(self.episodes), 12)
        self.assertEqual(sum(e.split == "dev" for e in self.episodes), 4)
        self.assertEqual(sum(e.split == "test" for e in self.episodes), 8)
        self.assertTrue(all(len(e.support) == 4 for e in self.episodes))
        self.assertTrue(all(len(e.queries) == 8 for e in self.episodes))

    def test_one_positive_example_leaves_ten_rules(self):
        for episode in self.episodes:
            learner = VersionSpaceLearner(episode.support)
            self.assertEqual(len(learner.hypotheses), 10)
            self.assertIn(episode.hidden_rule, learner.hypotheses)

    def test_pair_rule_becomes_identifiable_with_designed_negatives(self):
        pair_episodes = [e for e in self.episodes if len(e.hidden_rule) == 2]
        self.assertTrue(pair_episodes)
        for episode in pair_episodes:
            learner = VersionSpaceLearner(episode.support)
            for negative in episode.contrast_negatives:
                learner.add_negative(negative)
            self.assertEqual(learner.hypotheses, {episode.hidden_rule})

    def test_singleton_rule_remains_ambiguous_after_negative_contrasts(self):
        singleton_episodes = [e for e in self.episodes if len(e.hidden_rule) == 1]
        self.assertTrue(singleton_episodes)
        for episode in singleton_episodes:
            learner = VersionSpaceLearner(episode.support)
            for negative in episode.contrast_negatives:
                learner.add_negative(negative)
            self.assertEqual(len(learner.hypotheses), 4)
            self.assertIn(episode.hidden_rule, learner.hypotheses)

    def test_conservative_version_space_can_say_unknown(self):
        learner = VersionSpaceLearner(["f01", "f02", "f03", "f04"])
        self.assertEqual(learner.predict(["f01"]), "unknown")
        self.assertEqual(
            learner.predict(["f01", "f02", "f03", "f04"]), "positive"
        )
        self.assertEqual(learner.predict(["f99"]), "negative")

    def test_simple_baselines(self):
        exact = ExactExemplarBaseline(["f01", "f02"])
        self.assertTrue(exact.predict(["f01", "f02"]))
        self.assertFalse(exact.predict(["f01"]))
        similarity = JaccardBaseline(["f01", "f02", "f03", "f04"], threshold=0.5)
        self.assertTrue(similarity.predict(["f01", "f02"]))
        self.assertFalse(similarity.predict(["f01"]))

    def test_test_split_frozen_aggregate_results(self):
        report = evaluate_concept_episodes(self.episodes, split="test")
        self.assertEqual(report["episodes"], 8)
        self.assertEqual(report["queries"], 64)
        self.assertEqual(report["exact_exemplar_accuracy"], 0.625)
        self.assertEqual(report["jaccard_accuracy"], 0.5625)
        self.assertEqual(report["oracle_rule_accuracy"], 1.0)
        self.assertEqual(report["one_shot_unique_rule_rate"], 0.0)
        self.assertEqual(report["one_shot_certain_coverage"], 0.1875)
        self.assertEqual(report["one_shot_accuracy_when_certain"], 1.0)
        self.assertEqual(report["after_contrast_unique_rule_rate"], 0.5)
        self.assertEqual(report["after_contrast_certain_coverage"], 0.8125)
        self.assertEqual(report["after_contrast_accuracy_when_certain"], 1.0)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            JaccardBaseline(["f01"], threshold=1.1)


if __name__ == "__main__":
    unittest.main()
