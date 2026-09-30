import tempfile
import unittest
from pathlib import Path

from acm.benchmark import ImageSample, evaluate_one_shot, load_manifest, validate_manifest


class FakeEncoder:
    def embed(self, image):
        name = Path(image).name
        return {
            "s.jpg": (1, 0, 0),
            "k.jpg": (0.99, 0.01, 0),
            "u.jpg": (0, 1, 0),
        }[name]


def samples():
    return [
        ImageSample(Path("s.jpg"), "ZENTRA", "support", "car1", "shoot1"),
        ImageSample(Path("k.jpg"), "ZENTRA", "known", "car2", "shoot2"),
        ImageSample(Path("u.jpg"), "NOVEX", "unknown", "car3", "shoot3"),
    ]


class TestBenchmark(unittest.TestCase):
    def test_one_shot_metrics(self):
        report = evaluate_one_shot(samples(), FakeEncoder(), threshold=0.85)
        self.assertEqual(report["known_accuracy"], 1.0)
        self.assertEqual(report["unknown_rejection_rate"], 1.0)
        self.assertEqual(report["support_count"], 1)

    def test_reject_leaked_vehicle(self):
        rows = samples()
        rows[1] = ImageSample(Path("k.jpg"), "ZENTRA", "known", "car1", "shoot2")
        with self.assertRaisesRegex(ValueError, "vehicle_id"):
            validate_manifest(rows)

    def test_reject_leaked_capture(self):
        rows = samples()
        rows[1] = ImageSample(Path("k.jpg"), "ZENTRA", "known", "car2", "shoot1")
        with self.assertRaisesRegex(ValueError, "capture_group"):
            validate_manifest(rows)

    def test_reject_unknown_brand_in_support(self):
        rows = samples()
        rows[2] = ImageSample(Path("u.jpg"), "ZENTRA", "unknown", "car3", "shoot3")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_manifest(rows)

    def test_reject_duplicate_support(self):
        rows = samples() + [ImageSample(Path("extra.jpg"), "ZENTRA", "support", "car4", "shoot4")]
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            validate_manifest(rows)

    def test_manifest_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "picture.jpg").write_bytes(b"dummy")
            (folder / "images.csv").write_text(
                "path,label,split,vehicle_id,capture_group\n"
                "picture.jpg,ZENTRA,support,v1,g1\n", encoding="utf-8"
            )
            rows = load_manifest(folder / "images.csv")
            self.assertEqual(rows[0].path, folder / "picture.jpg")

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = folder / "images.csv"
            manifest.write_text(
                "path,label,split,vehicle_id,capture_group\n"
                "missing.jpg,ZENTRA,support,v1,g1\n", encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
