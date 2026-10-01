import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestVisionManifestChecker(unittest.TestCase):
    def test_checker_reports_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("s.jpg", "k.jpg", "u.jpg"):
                (root / name).write_bytes(b"x")
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "label", "split", "vehicle_id", "capture_group"])
                writer.writerow(["s.jpg", "A", "support", "a1", "g1"])
                writer.writerow(["k.jpg", "A", "known", "a2", "g2"])
                writer.writerow(["u.jpg", "B", "unknown", "b1", "g3"])

            result = subprocess.run(
                [sys.executable, "-m", "examples.check_vision_manifest",
                 "--manifest", str(manifest)],
                capture_output=True, text=True, check=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["support_count"], 1)
            self.assertEqual(report["known_query_count"], 1)
            self.assertEqual(report["unknown_query_count"], 1)


if __name__ == "__main__":
    unittest.main()
