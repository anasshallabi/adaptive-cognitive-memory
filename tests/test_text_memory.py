import unittest

from acm.text_memory import TextMemory, parse_claim


class TestTextParsing(unittest.TestCase):
    def test_fr_is_a(self):
        claim = parse_claim("Zentra est une marque automobile.", source="observation-1")
        self.assertEqual((claim.subject, claim.predicate, claim.object, claim.positive),
                         ("Zentra", "is_a", "marque automobile", True))
        self.assertEqual(claim.source, "observation-1")

    def test_fr_indefinite_generic_subject(self):
        claim = parse_claim("Une marque automobile est une organisation.")
        self.assertEqual(claim.subject, "marque automobile")

    def test_fr_negative(self):
        claim = parse_claim("Zentra n'est pas une banque.")
        self.assertFalse(claim.positive)

    def test_fr_apostrophe_unicode(self):
        claim = parse_claim("Zentra n’est pas une banque.")
        self.assertFalse(claim.positive)

    def test_en_is_a_and_negative(self):
        yes = parse_claim("Zentra is a car brand.")
        no = parse_claim("Zentra is not a bank.")
        self.assertEqual((yes.predicate, yes.object), ("is_a", "car brand"))
        self.assertFalse(no.positive)

    def test_property_fact(self):
        claim = parse_claim("Zentra fabrique des voitures électriques.")
        self.assertEqual((claim.predicate, claim.object), ("makes", "des voitures électriques"))

    def test_unsupported_sentence_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_claim("Zentra roule vite.")

    def test_question_not_accepted_as_fact(self):
        with self.assertRaises(ValueError):
            parse_claim("Zentra est une voiture ?")

    def test_source_mandatory(self):
        with self.assertRaises(ValueError):
            parse_claim("Zentra est une marque.", source="")


class TestTextMemory(unittest.TestCase):
    def test_one_exposure_direct_recall(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque automobile.", source="user-1")
        self.assertEqual(
            memory.query("ZENTRA", "is_a", "marque automobile")["status"], "supported"
        )
        self.assertEqual(len(memory.about("zentra")), 1)

    def test_two_hop_inference_with_provenance(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque automobile.", source="article-1")
        memory.learn_text("Une marque automobile est une organisation.", source="dictionary")
        answer = memory.ask("Est-ce que Zentra est une organisation ?")
        self.assertEqual(answer["status"], "supported")
        self.assertTrue(answer["positive_evidence"][0]["inferred"])
        self.assertEqual([x["source"] for x in answer["positive_evidence"][0]["path"]],
                         ["article-1", "dictionary"])

    def test_direct_negative(self):
        memory = TextMemory()
        memory.learn_text("Zentra n'est pas une banque.")
        self.assertEqual(memory.query("Zentra", "is_a", "banque")["status"], "negated")

    def test_conflict_exposed_not_overwritten(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une organisation.", source="A")
        memory.learn_text("Zentra n'est pas une organisation.", source="B")
        result = memory.ask("Est-ce que Zentra est une organisation ?")
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(len(result["positive_evidence"]), 1)
        self.assertEqual(len(result["negative_evidence"]), 1)

    def test_conflicting_inferred_and_direct_negative(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque automobile.", source="A")
        memory.learn_text("Une marque automobile est une organisation.", source="B")
        memory.learn_text("Zentra n'est pas une organisation.", source="C")
        self.assertEqual(memory.query("Zentra", "is_a", "organisation")["status"], "conflict")

    def test_negative_edges_are_not_transitive(self):
        memory = TextMemory()
        memory.learn_text("Zentra n'est pas une marque.")
        memory.learn_text("Une marque est une organisation.")
        self.assertEqual(memory.query("Zentra", "is_a", "organisation")["status"], "unknown")

    def test_no_invented_knowledge(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque.")
        self.assertEqual(memory.query("Zentra", "is_a", "organisation")["status"], "unknown")

    def test_duplicate_same_source_no_double_count(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque.", source="a")
        response = memory.learn_text("ZENTRA est une marque", source="a")
        self.assertEqual(response["status"], "duplicate")
        self.assertEqual(len(memory.claims), 1)

    def test_same_fact_from_distinct_sources_preserved(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque.", source="a")
        memory.learn_text("Zentra est une marque.", source="b")
        self.assertEqual(len(memory.query("Zentra", "is_a", "marque")["positive_evidence"]), 2)

    def test_cycle_safe(self):
        memory = TextMemory()
        memory.learn_text("A est un B.")
        memory.learn_text("B est un A.")
        memory.learn_text("B est un C.")
        self.assertEqual(memory.query("A", "is_a", "C")["status"], "supported")

    def test_max_hops(self):
        memory = TextMemory()
        memory.learn_text("A est un B.")
        memory.learn_text("B est un C.")
        self.assertEqual(memory.query("A", "is_a", "C", max_hops=1)["status"], "unknown")
        self.assertEqual(memory.query("A", "is_a", "C", max_hops=2)["status"], "supported")

    def test_property_not_inferred(self):
        memory = TextMemory()
        memory.learn_text("Zentra fabrique des voitures.")
        self.assertEqual(memory.query("Zentra", "makes", "des voitures")["status"], "supported")
        self.assertEqual(memory.query("des voitures", "makes", "Zentra")["status"], "unknown")

    def test_english_question(self):
        memory = TextMemory()
        memory.learn_text("Zentra is a car brand.")
        self.assertEqual(memory.ask("Is Zentra a car brand?")["status"], "supported")

    def test_about_question(self):
        memory = TextMemory()
        memory.learn_text("Zentra est une marque.")
        response = memory.ask("Que sais-tu de Zentra ?")
        self.assertEqual(len(response["facts"]), 1)

    def test_unsupported_question_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            TextMemory().ask("Pourquoi les voitures existent-elles ?")


if __name__ == "__main__":
    unittest.main()
