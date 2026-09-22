"""Check publication navigation without loading models or changing artifacts."""
from pathlib import Path
import re
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_single_metric_location(self):
        for name in ('gold_gloss_supports', 'historical_studies', 'pilot_tests'):
            self.assertFalse((ROOT / 'metrics' / name).exists())
        self.assertFalse(any(p.is_symlink() for p in (ROOT / 'metrics').rglob('*')))
        for name in ('matched_v1', 'matched_gold_v2/lexical_and_xcomet_xl', 'matched_gold_v2/xcomet_xxl'):
            self.assertTrue((ROOT / 'metrics' / name).is_dir())
        self.assertFalse((ROOT / 'metrics/matched_gold_v2_xcomet_xxl').exists())

    def test_report_has_one_location(self):
        self.assertFalse((ROOT / 'grammar_context_evaluation').exists())
        self.assertTrue((ROOT / 'reports/matched_gold_v2/audit.json').is_file())
        self.assertFalse((ROOT / 'scripts/organize_metric_views.py').exists())

    def test_reader_facing_links(self):
        paths = [ROOT / p for p in (
            'README.md', 'THESIS_GUIDE.md', 'metrics/README.md',
            'results/README.md', 'docs/REPOSITORY_STRUCTURE.md', 'reports/README.md')]
        paths += list((ROOT / 'reports/matched_gold_v2').glob('*.md'))
        for path in paths:
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if '://' in link or link.startswith('#'):
                    continue
                target = unquote(link.split('#', 1)[0].strip('<>'))
                with self.subTest(document=str(path.relative_to(ROOT)), link=link):
                    self.assertTrue((path.parent / target).exists())


if __name__ == '__main__':
    unittest.main()
