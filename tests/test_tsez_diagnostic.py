from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
from diagnose_tsez_chain import PresencePenalty, aligned_support, generation_policy, repetition_warning, select_dev


class DiagnosticTests(unittest.TestCase):
    def test_presence_is_generated_only_and_not_frequency(self):
        import torch
        scores = torch.tensor([[2., 3., 4., 5.], [2., 3., 4., 5.]])
        ids = torch.tensor([[0, 1, 2, 2], [0, 1, 3, 3]])
        result = PresencePenalty(2, 1.5)(ids, scores)
        torch.testing.assert_close(result, torch.tensor([[2., 3., 2.5, 5.], [2., 3., 4., 3.5]]))
        torch.testing.assert_close(PresencePenalty(4, 1.5)(ids, scores), scores)
        torch.testing.assert_close(scores, torch.tensor([[2., 3., 4., 5.], [2., 3., 4., 5.]]))

    def test_support_alignment_fail_closed(self):
        rows = [{"source":"x", "reference":"X", "gloss":""}]
        gold = [{"source":"x", "reference":"X", "gloss":"meaning"}]
        self.assertEqual(aligned_support(rows, gold), gold)
        for bad in ([], [{"source":"y", "reference":"X", "gloss":"meaning"}], rows):
            with self.assertRaises(ValueError):
                aligned_support(rows, bad)

    def test_selection_disjoint_stable_and_reference_independent(self):
        rows = [{"source":"x"*(i+1), "reference":"a"} for i in range(30)]
        exclude = {rows[-1]["source"], rows[0]["source"]}
        chosen = select_dev(rows, exclude)
        self.assertEqual(len(chosen), 8)
        self.assertTrue({26, 27, 28}.issubset(chosen))
        self.assertFalse(set(chosen) & {0, 29})
        changed = [dict(r, reference="other", gloss="other") for r in rows]
        self.assertEqual(chosen, select_dev(changed, exclude))

    def test_policies_and_repetition_flag(self):
        self.assertEqual(generation_policy(False), {"do_sample":False})
        self.assertEqual(generation_policy(True)["temperature"], 0.7)
        self.assertTrue(repetition_warning("the man came " * 4))
        self.assertFalse(repetition_warning("The man came home and ate dinner."))


if __name__ == "__main__":
    unittest.main()
