import tempfile
import textwrap
import unittest
from pathlib import Path

import prompt_builder as pb

REPO = Path(__file__).resolve().parent.parent


def write_batch(tmp: Path, body: str) -> Path:
    path = tmp / "lote.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class LoadBatchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_example_batch(self):
        batch = pb.load_batch(REPO / "jobs" / "exemplo.yaml")
        self.assertEqual(len(batch.jobs), 5)
        self.assertEqual(batch.rejected, [])
        self.assertEqual([j.id for j in batch.jobs][-1], "lara-variacao-003")

    def test_generated_jobs_are_stable_when_count_grows(self):
        body = """
        characters:
          ana: {description: woman with black hair, age: 30}
        generate:
          count: COUNT
          seed: 7
          base: {subjects: [ana]}
          pick:
            scene: [beach, forest, loft, desert]
            lighting: [soft light, hard light, neon]
        """
        small = pb.load_batch(write_batch(self.tmp, body.replace("COUNT", "3")))
        big = pb.load_batch(write_batch(self.tmp, body.replace("COUNT", "6")))
        self.assertEqual([j.to_dict() for j in small.jobs], [j.to_dict() for j in big.jobs[:3]])
        self.assertEqual(len(big.jobs), 6)

    def test_unsafe_job_is_rejected_not_fatal(self):
        path = write_batch(self.tmp, """
        jobs:
          - id: ok
            subjects: [{description: woman, age: 30}]
          - id: ruim
            subjects: [{description: teen girl, age: 30}]
          - id: idade
            subjects: [{description: woman, age: 19}]
          - id: famosa
            subjects: [{description: woman, age: 30}]
            scene: celebrity lookalike
        """)
        batch = pb.load_batch(path)
        self.assertEqual([j.id for j in batch.jobs], ["ok"])
        self.assertEqual([r[0] for r in batch.rejected], ["ruim", "idade", "famosa"])

    def test_structural_errors_raise(self):
        cases = {
            "campo desconhecido": "jobs:\n  - {id: a, subjects: [{description: w, age: 30}], cenario: x}\n",
            "personagem inexistente": "jobs:\n  - {id: a, subjects: [ninguem]}\n",
            "sem subjects": "jobs:\n  - {id: a}\n",
            "id inválido": "jobs:\n  - {id: 'a b', subjects: [{description: w, age: 30}]}\n",
            "chave do topo": "jobz: []\n",
            "style": "jobs:\n  - {id: a, style: anime, subjects: [{description: w, age: 30}]}\n",
        }
        for name, body in cases.items():
            with self.subTest(name), self.assertRaises(pb.BatchError):
                pb.load_batch(write_batch(self.tmp, body))

    def test_character_name_is_checked_but_not_in_prompt(self):
        path = write_batch(self.tmp, """
        characters:
          zara_x: {description: woman with silver hair, age: 40}
        jobs:
          - {id: a, subjects: [zara_x]}
        """)
        job = pb.load_batch(path).jobs[0]
        self.assertNotIn("zara", pb.draft_image_prompt(job).lower())
        blocked = write_batch(self.tmp, """
        characters:
          celebridade: {description: woman, age: 40}
        jobs:
          - {id: a, subjects: [celebridade]}
        """)
        with self.assertRaises(pb.cs.SafetyError):
            pb.load_batch(blocked)


class PromptTest(unittest.TestCase):
    def job(self, **kw):
        base = dict(id="j", subjects=[pb.Character("a woman with red hair", 30)], seed=1)
        base.update(kw)
        return pb.JobSpec(**base)

    def test_subjects_phrase_anchors_adult_age(self):
        self.assertEqual(pb.subjects_phrase([pb.Character("a woman with red hair.", 30)]),
                         "an adult woman with red hair, 30 years old")
        two = pb.subjects_phrase([pb.Character("adult man", 40), pb.Character("woman", 35)])
        self.assertEqual(two, "two adults: an adult man, 40 years old; an adult woman, 35 years old")

    def test_video_prompt_order(self):
        prompt = pb.draft_video_prompt(self.job(motion="she waves", camera="slow pan", sound="rain",
                                                wardrobe="wearing a coat"))
        self.assertTrue(prompt.startswith("She waves."))
        self.assertIn("wearing a coat", prompt)
        self.assertNotIn("wearing wearing", prompt)
        self.assertLess(prompt.index("Camera:"), prompt.index("Audio:"))

    def test_clean_llm_output(self):
        raw = '<think>hmm</think>\nHere is the prompt:\n"An adult woman, 30 years old, on a beach."\n\nNotes: x'
        self.assertEqual(pb.clean_llm_output(raw), "An adult woman, 30 years old, on a beach.")

    def test_finalize_prompt_falls_back_when_llm_misbehaves(self):
        job = self.job()
        draft = pb.draft_image_prompt(job)

        class Fake:
            def __init__(self, out):
                self.out = out

            def generate(self, draft, kind, seed):
                return self.out

        good = "An adult woman, 30 years old, red hair, golden light."
        self.assertEqual(pb.finalize_prompt(draft, "image", job, Fake(good)), (good, "llm"))
        no_age = pb.finalize_prompt(draft, "image", job, Fake("A woman with red hair."))
        self.assertEqual(no_age[0], draft)
        unsafe = pb.finalize_prompt(draft, "image", job, Fake("An adult woman, 30 years old, celebrity lookalike."))
        self.assertEqual(unsafe[0], draft)
        self.assertIn("llm recusado", unsafe[1])


if __name__ == "__main__":
    unittest.main()
