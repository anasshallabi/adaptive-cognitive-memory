import unittest
from acm.vision import VisionMemory


class FakeEncoder:
    """Deterministic test doubles; NOT a real image encoder."""
    vectors = {
        "zentra_1.jpg": (1, 0, 0),
        "zentra_2.jpg": (0.95, 0.05, 0),
        "novex_1.jpg": (0, 1, 0),
        "unknown.jpg": (0, 0, 1),
    }

    def embed(self, image):
        return self.vectors[str(image)]


class TestVisionMemory(unittest.TestCase):
    def test_one_shot_and_unknown_rejection(self):
        memory = VisionMemory(FakeEncoder())
        memory.learn_image("zentra_1.jpg", "ZENTRA", concepts=["car brand"])
        self.assertEqual(memory.recognize_image("zentra_2.jpg")["label"], "ZENTRA")
        self.assertEqual(memory.recognize_image("unknown.jpg")["status"], "unknown")
        self.assertEqual(memory.memory.related_concepts("ZENTRA"), ["car brand"])

    def test_more_learning_does_not_erase_old_association(self):
        memory = VisionMemory(FakeEncoder())
        memory.learn_image("zentra_1.jpg", "ZENTRA")
        memory.learn_image("novex_1.jpg", "NOVEX")
        self.assertEqual(memory.recognize_image("zentra_2.jpg")["label"], "ZENTRA")
        self.assertEqual(memory.recognize_image("novex_1.jpg")["label"], "NOVEX")


if __name__ == "__main__":
    unittest.main()
