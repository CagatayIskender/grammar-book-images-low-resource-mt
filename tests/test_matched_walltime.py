import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from matched_walltime import choose_hours


class WalltimeTests(unittest.TestCase):
    def test_continuations(self):
        self.assertEqual(choose_hours('lezgi', 'chain_gloss', ['baseline'], 12), 1)
        self.assertEqual(choose_hours('lezgi', 'chain_gloss', ['baseline'], 1044), 6)

    def test_large_tsez_is_not_shortened(self):
        self.assertEqual(choose_hours('tsez', 'shot', ['baseline', 'cheatsheet_jpg'], 1335), 10)
        self.assertEqual(choose_hours('tsez', 'shot', ['summary_text_txt'], 445), 3)

    def test_other_groups(self):
        self.assertEqual(choose_hours('gitksan', 'modelgloss', ['baseline'], 481), 4)
        self.assertEqual(choose_hours('natugu', 'shot', ['baseline'], 693), 6)
