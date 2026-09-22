import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metric_views", ROOT / "scripts/organize_metric_views.py")
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


class MetricViewTests(unittest.TestCase):
    def test_relative_views_are_idempotent_and_preserve_canonical_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for target in views.VIEWS.values():
                (root / "metrics" / target).mkdir(parents=True)
            with contextlib.redirect_stdout(io.StringIO()):
                views.create_views(root)
                views.create_views(root)
            for name, target in views.VIEWS.items():
                link = root / "metrics" / name
                self.assertTrue(link.is_symlink())
                self.assertFalse(link.readlink().is_absolute())
                self.assertEqual(link.resolve(), root / "metrics" / target)

    def test_existing_real_directory_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics/matched_gold_v2").mkdir(parents=True)
            (root / "metrics/gold_gloss_supports/bleu_chrf_xcomet_xl").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                views.create_views(root)

    def test_missing_source_is_not_hidden_by_a_broken_link(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                views.create_views(Path(temp))


if __name__ == "__main__":
    unittest.main()
