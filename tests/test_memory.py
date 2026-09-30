import unittest
from acm import CognitiveMemory


class TestMemory(unittest.TestCase):
    def test_one_shot_association(self):
        m = CognitiveMemory()
        m.learn("ZENTRA", (1, 0, 0), concepts=("car brand", "electric"))
        result = m.recognize((0.99, 0.1, 0))
        self.assertEqual(result["label"], "ZENTRA")
        self.assertEqual(result["concepts"], ["car brand", "electric"])

    def test_unknown_rejection(self):
        m = CognitiveMemory()
        m.learn("ZENTRA", (1, 0, 0))
        self.assertEqual(m.recognize((0, 1, 0))["status"], "unknown")

    def test_empty_memory(self):
        self.assertEqual(CognitiveMemory().recognize((1, 2))["status"], "unknown")

    def test_dimension_mismatch(self):
        m = CognitiveMemory()
        m.learn("ZENTRA", (1, 0))
        with self.assertRaises(ValueError):
            m.recognize((1, 0, 0))

    def test_invalid_inputs(self):
        m = CognitiveMemory()
        with self.assertRaises(ValueError):
            m.learn(" ", (1, 0))
        with self.assertRaises(ValueError):
            m.learn("X", ())
        with self.assertRaises(ValueError):
            m.recognize((1,), threshold=2)

    def test_memory_retention(self):
        m = CognitiveMemory()
        m.learn("ZENTRA", (1, 0))
        m.learn("NOVEX", (0, 1))
        self.assertEqual(m.recognize((1, 0))["label"], "ZENTRA")
        self.assertEqual(m.recognize((0, 1))["label"], "NOVEX")


if __name__ == "__main__":
    unittest.main()
