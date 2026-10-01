import unittest
from pathlib import Path

from acm.benchmark import ImageSample
from acm.vision_compare import DirectNearestNeighbor, evaluate_matched_one_shot


class FakeEncoder:
    vectors = {
        "a_support.jpg": (1.0, 0.0, 0.0),
        "a_known.jpg": (0.98, 0.02, 0.0),
        "b_support.jpg": (0.0, 1.0, 0.0),
        "b_known.jpg": (0.05, 0.95, 0.0),
        "unknown.jpg": (0.0, 0.0, 1.0),
    }

    def __init__(self):
        self.calls = []

    def embed(self, image):
        name = Path(image).name
        self.calls.append(name)
        return self.vectors[name]


def fixture():
    return [
        ImageSample(Path("a_support.jpg"), "ALPHA", "support", "a1", "s1"),
        ImageSample(Path("b_support.jpg"), "BETA", "support", "b1", "s2"),
        ImageSample(Path("a_known.jpg"), "ALPHA", "known", "a2", "s3"),
        ImageSample(Path("b_known.jpg"), "BETA", "known", "b2", "s4"),
        ImageSample(Path("unknown.jpg"), "GAMMA", "unknown", "g1", "s5"),
    ]


class TestMatchedVisionComparison(unittest.TestCase):
    def test_acm_and_direct_nn_are_identical(self):
        encoder = FakeEncoder()
        report = evaluate_matched_one_shot(fixture(), encoder, threshold=0.85)
        self.assertEqual(report["support_count"], 2)
        self.assertEqual(report["query_count"], 3)
        self.assertEqual(report["acm_nn_prediction_agreement"], 1.0)
        self.assertEqual(report["acm_nn_score_agreement"], 1.0)
        self.assertEqual(report["acm"], report["direct_nearest_neighbor"])
        self.assertEqual(report["acm"]["known_accuracy"], 1.0)
        self.assertEqual(report["acm"]["unknown_rejection_rate"], 1.0)

    def test_opaque_labels_do_not_change_embedding_behavior(self):
        report = evaluate_matched_one_shot(fixture(), FakeEncoder(), threshold=0.85)
        self.assertEqual(report["opaque_label_behavior_agreement"], 1.0)
        self.assertEqual(report["opaque_label_acm"]["known_accuracy"], 1.0)
        self.assertEqual(report["opaque_label_acm"]["unknown_rejection_rate"], 1.0)

    def test_each_image_encoded_once(self):
        encoder = FakeEncoder()
        evaluate_matched_one_shot(fixture(), encoder, threshold=0.85)
        self.assertEqual(len(encoder.calls), len(fixture()))
        self.assertEqual(len(set(encoder.calls)), len(fixture()))

    def test_direct_nn_threshold_and_dimensions(self):
        nn = DirectNearestNeighbor()
        nn.add("A", [1, 0])
        self.assertEqual(nn.recognize([1, 0], threshold=0.9)["label"], "A")
        self.assertEqual(nn.recognize([0, 1], threshold=0.9)["status"], "unknown")
        with self.assertRaises(ValueError):
            nn.recognize([1, 0], threshold=2)
        with self.assertRaises(ValueError):
            nn.add("B", [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
