from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
from run_qwen35_baselines import baseline_prompt, parse
from experiment_io import build_chain_prompt


class BaselineTests(unittest.TestCase):
    def test_chain_removes_only_grammar_context_and_reference_instruction(self):
        support = [{"source":"TRAIN_SOURCE", "reference":"TRAIN_TRANSLATION", "gloss":""}]
        system, user = baseline_prompt("chain_gloss", support, "Tsez", "TEST_SOURCE")
        expected_system, expected = build_chain_prompt(support, "Tsez", "TEST_SOURCE")
        expected = expected.replace("Here is a grammar reference summary for Tsez:\n\n", "", 1)
        expected = expected.replace("Use the grammar reference to produce the gloss first", "Produce the gloss first", 1)
        self.assertEqual(system, expected_system)
        self.assertEqual(user, expected)
        self.assertNotIn("grammar reference", user)
        self.assertIn("TRAIN_TRANSLATION", user)

    def test_no_thinking_or_truncated_outputs_accepted(self):
        item = {"raw":"FINAL_TRANSLATION: A translation.", "raw_with_special_tokens":"", "truncated":False,
                "repetition_warning":False}
        self.assertEqual(parse("shot", "sampled_v1", item), ("", "A translation."))
        for change in ({"truncated":True}, {"raw_with_special_tokens":"<think>x</think>"}, {"repetition_warning":True}):
            with self.assertRaises(ValueError):
                parse("shot", "sampled_v1", dict(item, **change))


if __name__ == "__main__":
    unittest.main()
