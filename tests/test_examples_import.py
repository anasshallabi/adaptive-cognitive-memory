"""Smoke test the executable examples are importable as ACM modules."""
import importlib
import unittest
from pathlib import Path


class TestExamplesAreImportable(unittest.TestCase):
    def test_compare_text_resolves_to_our_repository(self):
        module = importlib.import_module("examples.compare_text")
        expected = Path(__file__).resolve().parents[1] / "examples" / "compare_text.py"
        self.assertEqual(Path(module.__file__).resolve(), expected.resolve())

    def test_text_demo_resolves_to_our_repository(self):
        module = importlib.import_module("examples.text_one_shot")
        expected = Path(__file__).resolve().parents[1] / "examples" / "text_one_shot.py"
        self.assertEqual(Path(module.__file__).resolve(), expected.resolve())


if __name__ == "__main__":
    unittest.main()
