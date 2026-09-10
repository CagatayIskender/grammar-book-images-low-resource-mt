
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))
import os
_layout_sys.path.insert(0, str(_LAYOUT_ROOT / "runners"))
from experiment_io import dataset

BASE = str(_LAYOUT_ROOT.parent / "Database/2023glossingST/data")

languages = {
    "Gitksan": ("Gitksan", "git-train-track1-covered.txt", "git-test-track1-uncovered.txt"),
    "Natugu": ("Natugu", "ntu-train-track1-covered", "ntu-test-track1-uncovered"),
    "Lezgi": ("Lezgi", "lez-train-track1-covered", "lez-test-track1-uncovered"),
    "Tsez": ("Tsez", "ddo-train-track1-covered", "ddo-test-track1-uncovered"),
}

print(f"{'Language':<10} | {'Train size':<10} | {'Test size':<10}")
print("-" * 36)

for lang, (folder, train_file, test_file) in languages.items():
    train_path = os.path.join(BASE, folder, train_file)
    test_path = os.path.join(BASE, folder, test_file)

    train_size = len(dataset(lang, "train"))
    test_size = len(dataset(lang))

    print(f"{lang:<10} | {train_size:<10} | {test_size:<10}")
