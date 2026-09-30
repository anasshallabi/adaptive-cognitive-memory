import unittest

from acm.multimodal import MultimodalMemory
from acm.vision import VisionMemory


class FakeEncoder:
    vectors = {"seen.jpg": (1., 0., 0.), "other.jpg": (.99, .01, 0.), "stranger.jpg": (0., 1., 0.)}

    def embed(self, image):
        return self.vectors[str(image)]


class TestSharedLabelBridge(unittest.TestCase):
    def test_image_recognition_links_explicit_text_claims(self):
        acm = MultimodalMemory(VisionMemory(FakeEncoder()))
        acm.learn_image("seen.jpg", "Zentra")
        acm.learn_text("Zentra est une marque automobile.", source="manual note")
        result = acm.inspect_image("other.jpg")
        self.assertEqual(result["recognition"]["label"], "Zentra")
        self.assertEqual(result["linked_text_claims"][0]["source"], "manual note")

    def test_unknown_image_does_not_leak_facts(self):
        acm = MultimodalMemory(VisionMemory(FakeEncoder()))
        acm.learn_image("seen.jpg", "Zentra")
        acm.learn_text("Zentra est une marque automobile.")
        result = acm.inspect_image("stranger.jpg")
        self.assertEqual(result["recognition"]["status"], "unknown")
        self.assertEqual(result["linked_text_claims"], [])


if __name__ == "__main__":
    unittest.main()
