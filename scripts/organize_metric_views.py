"""Create readable metric views without changing frozen paths or file hashes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {
    "gold_gloss_supports/bleu_chrf_xcomet_xl": "matched_gold_v2",
    "gold_gloss_supports/xcomet_xxl": "matched_gold_v2_xcomet_xxl",
    "historical_studies/empty_support_glosses": "matched_v1",
    "historical_studies/earlier_grammar_materials": "curated_v1",
    "historical_studies/earlier_chain_gloss": "chain_gloss_v2",
    "historical_studies/earlier_baselines": "baseline",
    "historical_studies/legacy_grammar_runs": "legacy",
    "pilot_tests/api_smoke_tests": "smoke",
}


def create_views(root=ROOT):
    metrics = root / "metrics"
    for name, source in VIEWS.items():
        target = metrics / source
        link = metrics / name
        if not target.is_dir():
            raise FileNotFoundError(target)
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise FileExistsError(f"Refusing to replace {link}")
        else:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(Path("..") / source, target_is_directory=True)
        print(f"{name} -> {source}")


if __name__ == "__main__":
    create_views()
