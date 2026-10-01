import unittest
from pathlib import Path

from acm.benchmark import ImageSample
from acm.vision_compare import calibrate_rejection_threshold


class FakeEncoder:
    vectors = {
        "a_support.jpg": (1.0, 0.0),
        "a_known.jpg": (0.9, 0.1),
        "u1.jpg": (0.6, 0.8),
        "u2.jpg": (0.0, 1.0),
    }

    def embed(self, image):
        return self.vectors[Path(image).name]


def validation_fixture():
    return [
        ImageSample(Path("a_support.jpg"), "A", "support", "a1", "g1"),
        ImageSample(Path("a_known.jpg"), "A", "known", "a2", "g2"),
        ImageSample(Path("u1.jpg"), "B", "unknown", "b1", "g3"),
        ImageSample(Path("u2.jpg"), "C", "unknown", "c1", "g4"),
    ]


class TestVisionCalibration(unittest.TestCase):
    def test_calibration_returns_valid_threshold_and_metrics(self):
        report = calibrate_rejection_threshold(validation_fixture(), FakeEncoder())
        self.assertGreaterEqual(report["threshold"], -1)
        self.assertLessEqual(report["threshold"], 1)
        self.assertEqual(report["known_accuracy"], 1.0)
        self.assertEqual(report["unknown_rejection_rate"], 1.0)
        self.assertEqual(report["balanced_objective"], 1.0)
        self.assertGreater(report["candidate_thresholds_evaluated"], 1)

    def test_tie_break_prefers_more_conservative_threshold(self):
        report = calibrate_rejection_threshold(validation_fixture(), FakeEncoder())
        # Known cosine is very high; u1 is lower. Any separator works.
        # Higher tied separator is intentionally preferred.
        self.assertGreater(report["threshold"], 0.6)


if __name__ == "__main__":
    unittest.main()
