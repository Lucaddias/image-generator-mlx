import unittest

import content_safety as cs


class FindViolationsTest(unittest.TestCase):
    def assertBlocked(self, text, rule_fragment):
        violations = cs.find_violations(text)
        self.assertTrue(
            any(rule_fragment in v.rule for v in violations),
            f"esperava bloqueio '{rule_fragment}' para {text!r}, veio {violations}",
        )

    def test_normal_prompt_passes(self):
        text = (
            "RAW photo, an adult woman with red hair, 30 years old, baby blue dress, "
            "85mm lens, f/1.8, Kodak Portra 400, 24 fps, mulher de 25 anos, ela bebe vinho"
        )
        self.assertEqual(cs.find_violations(text), [])

    def test_ages_below_minimum(self):
        for text in ("a 19-year-old woman", "17yo", "16 y/o", "aged 15", "mulher de 20 anos",
                     "eighteen year old", "dezenove anos"):
            with self.subTest(text=text):
                self.assertBlocked(text, "idade abaixo")

    def test_minor_terms_including_leetspeak_and_portuguese(self):
        for text in ("teen", "t33n", "sch00lgirl", "school uniform", "loli", "novinha", "colegial",
                     "menina", "criança", "bebê", "baby face", "fifteen", "a minor"):
            with self.subTest(text=text):
                self.assertBlocked(text, "menor de idade")

    def test_portuguese_false_positives_avoided(self):
        for text in ("cintura menor", "ela bebe água", "eighteen-wheeler truck"):
            with self.subTest(text=text):
                self.assertFalse(
                    any("menor" in v.rule for v in cs.find_violations(text)), text
                )

    def test_real_person_terms(self):
        for text in ("celebrity", "lookalike of", "face-swap", "famosa", "sósia",
                     "parecida com", "troca de rosto", "real person"):
            with self.subTest(text=text):
                self.assertBlocked(text, "pessoa real")

    def test_blocked_names_list(self):
        violations = cs.find_violations("portrait of J0hn   Smith", ["John Smith"])
        self.assertTrue(any("lista" in v.rule for v in violations))
        self.assertEqual(cs.find_violations("portrait of Johnny", ["John"]), [])

    def test_check_text_raises(self):
        with self.assertRaises(cs.SafetyError):
            cs.check_text("teen", where="teste")


class SubjectAgeTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(cs.check_subject_age(21, where="x"), 21)

    def test_invalid(self):
        for age in (None, "30", 30.0, True, 20, 18, 0, 130):
            with self.subTest(age=age), self.assertRaises(cs.SafetyError):
                cs.check_subject_age(age, where="x")


if __name__ == "__main__":
    unittest.main()
