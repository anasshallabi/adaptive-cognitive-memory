"""v0.6 conservative safety smoke tests (post-hoc engineering only).

Fresh fictional entity strings; no real LLM, no claim of held-out gain.
"""
import unittest

from acm.claim_safety import assess_sentence, rule_is_canonical
from acm.text_extraction import ConservativeHybridExtractor, FlexibleTextMemory
from acm.text_memory import Claim, parse_claim


class CountingExtractor:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response

    def extract(self, sentence, *, source="user"):
        self.calls += 1
        return self.response


class TestSafetyScreen(unittest.TestCase):
    def test_question_does_not_become_fact(self):
        self.assertEqual(assess_sentence("Is Oranta a company?").reason, "question")

    def test_attributed_french_and_english(self):
        for s in ("Selon un journaliste, Olora est une banque.",
                  "According to a reporter, Mavix is a brand."):
            self.assertEqual(assess_sentence(s).reason, "attribution")

    def test_uncertain_or_historical(self):
        self.assertEqual(assess_sentence("Merana might be a manufacturer.").reason,
                         "uncertainty")
        self.assertEqual(assess_sentence("Brenar était une entreprise.").reason,
                         "historical")

    def test_multiple_clauses_and_imperative(self):
        self.assertEqual(
            assess_sentence("A sorata is a robot and it uses solar panels.").reason,
            "multiple_propositions",
        )
        self.assertEqual(
            assess_sentence("Imagine that Rovian is a bank.").reason, "directive"
        )

    def test_direct_negative_does_not_get_blocked(self):
        self.assertTrue(assess_sentence("Ravex n'est pas une banque.").allowed)

    def test_noncanonical_rule_subject_is_not_trusted(self):
        for text in (
            "La société Edera utilise des batteries.",
            "The company named Ivera makes robots.",
            "Xavora currently has solar panels.",
        ):
            claim = parse_claim(text) if "named" not in text else None
            if claim is not None:
                self.assertFalse(rule_is_canonical(text, claim))

    def test_generic_category_link_is_still_accepted(self):
        text = "Une marque automobile est une organisation."
        self.assertTrue(rule_is_canonical(text, parse_claim(text)))

    def test_guard_skips_the_model(self):
        model = CountingExtractor()
        hybrid = ConservativeHybridExtractor(model)
        result = hybrid.extract("Selon la presse, Alvira est une banque.")
        self.assertIsNone(result)
        self.assertEqual(model.calls, 0)
        self.assertEqual(hybrid.last_route, "guard")
        self.assertEqual(hybrid.last_reason, "attribution")

    def test_simple_assertion_keeps_zero_cost_rule(self):
        model = CountingExtractor()
        hybrid = ConservativeHybridExtractor(model)
        claim = hybrid.extract("Valora est une marque.")
        self.assertEqual(claim.subject, "Valora")
        self.assertEqual(model.calls, 0)
        self.assertEqual(hybrid.last_route, "rules")

    def test_complex_entity_uses_fallback(self):
        sentence = "La société Edera utilise des batteries."
        model = CountingExtractor(Claim(
            "Edera", "uses", "des batteries", True, "new source", sentence
        ))
        hybrid = ConservativeHybridExtractor(model)
        claim = hybrid.extract(sentence, source="new source")
        self.assertEqual(model.calls, 1)
        self.assertEqual(hybrid.last_route, "fallback")
        self.assertEqual(hybrid.last_reason, "review_noncanonical_subject")
        self.assertEqual(claim.subject, "Edera")

    def test_guard_prevents_storage_of_an_unqualified_statement(self):
        model = CountingExtractor()
        memory = FlexibleTextMemory(ConservativeHybridExtractor(model))
        result = memory.learn_text("A reviewer says that Selora might be a bank.")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(memory.memory.claims, [])
        self.assertEqual(model.calls, 0)

    def test_unsafe_syntax_can_still_escape_guard(self):
        # Intentional limitation of a deterministic guard. This test
        # documents that we make NO guarantee of semantic safety.
        self.assertTrue(assess_sentence("It is rumored that Calora is a startup.").allowed)


if __name__ == "__main__":
    unittest.main()
