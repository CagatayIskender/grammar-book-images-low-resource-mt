"""Create a new gold-support matrix without changing historical artifacts."""
import argparse
import json
import re
import sys
from pathlib import Path

from protocol import ROOT, REVISION, digest, verify_inputs, prompt
sys.path.insert(0, str(ROOT / "runners"))
from experiment_io import atomic_json, sha256, PREFIX

MODELS = ("qwen3", "qwen35", "gemini25flashlite")
DOCS = ROOT / "docs" / REVISION
DATA = ROOT / "inputs/support_gold_v2"
UPSTREAM = "190689ac81935359c69a46463c48e25e63e601f7"


def parse_igt(text):
    rows = []
    for block in re.split(r"\n\s*\n", text):
        row = {}
        for line in block.splitlines():
            for marker, key in (("\\t", "source"), ("\\g", "gloss"), ("\\l", "reference")):
                if line.startswith(marker):
                    row[key] = line[2:].strip()
        if row.get("source") and row.get("reference"):
            row["reference"] = row["reference"].split("||")[0].strip()
            rows.append(row)
    return rows


def historical_hashes():
    paths = []
    for base in ("configs/matched_v1", "results/matched_v1", "metrics/matched_v1",
                 "experiments/mtob/results", "experiments/mtob/metrics"):
        paths.extend(p for p in (ROOT / base).rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def source(language, download):
    if language != "Gitksan":
        path = ROOT.parent / "Database/2023glossingST/data" / language / f"{PREFIX[language]}-train-track1-uncovered"
        return path, {"kind": "local_authoritative_training_data", "path": str(path)}
    path = DATA / "gitksan/git-train-track1-uncovered"
    url = f"https://raw.githubusercontent.com/sigmorphon/2023glossingST/{UPSTREAM}/data_v1/Gitksan/git-train-track1-uncovered"
    if not path.exists():
        if not download:
            raise ValueError("Gitksan source missing; explicitly run with --download-gitksan")
        import requests
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        license_url = url.rsplit("/", 1)[0] + "/LICENSE"
        license_response = requests.get(license_url, timeout=60)
        license_response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        path.with_name("LICENSE").write_bytes(license_response.content)
    return path, {"kind": "official_sigmorphon_training_data", "url": url, "commit": UPSTREAM}


def prepare(download=False):
    old_paths = [p for p in sorted((ROOT / "configs/matched_v1").glob("*.json"))
                 if p.name.startswith(tuple(m + "_" for m in MODELS))]
    if len(old_paths) != 351:
        raise ValueError(f"Expected 351 historical configurations, found {len(old_paths)}")
    historical = historical_hashes()
    preserved = DOCS / "historical_hashes.json"
    if preserved.exists() and json.loads(preserved.read_text()) != historical:
        raise ValueError("Historical artifacts changed since correction preparation")
    atomic_json(preserved, historical)
    support, manifest_paths = {}, {}
    for language in PREFIX:
        cp = ROOT / f"configs/matched_v1/qwen3_{language.lower()}_grammar_original_shot_baseline.json"
        prior = json.loads(cp.read_text())
        old = prior["support"]
        path, origin = source(language, download)
        training = parse_igt(path.read_text(encoding="utf-8"))
        rows = training[:21]
        if len(rows) != 21 or any(not r.get("gloss", "").strip() for r in rows):
            raise ValueError(f"Missing gold glosses: {language}")
        if [(r["source"], r["reference"]) for r in rows] != [(r["source"], r["reference"]) for r in old]:
            raise ValueError(f"Gold support source/reference/order mismatch: {language}")
        test_sources = {r['source'] for r in prior['test']}
        removed = [i for i, r in enumerate(rows) if r['source'] in test_sources]
        indices = [i for i, r in enumerate(training) if r['source'] not in test_sources][:21]
        rows = [training[i] for i in indices]
        if len(rows) != 21 or any(not r.get('gloss', '').strip() for r in rows):
            raise ValueError(f'Insufficient non-test gold training supports: {language}')
        mp = DOCS / "support" / f"{language.lower()}.json"
        atomic_json(mp, {"language": language, "origin": origin, "training_file": str(path),
                         "training_sha256": sha256(path), "support": rows, "support_sha256": digest(rows),
                         "alignment": "First 21 covered records match gold training; test-overlapping sources are excluded, then filled in training order",
                         "training_indices": indices, "removed_original_support_indices": removed,
                         "added_training_indices": [i for i in indices if i >= 21],
                         "excluded_test_sources_sha256": digest(sorted(test_sources))})
        support[language], manifest_paths[language] = rows, str(mp.relative_to(ROOT))
    code = sorted((ROOT / "runners" / REVISION).glob("*.py"))
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in code}
    catalog, mapping = [], {}
    for path in old_paths:
        old = json.loads(path.read_text())
        cfg = json.loads(path.read_text())
        cfg.pop("fingerprint")
        cfg["support"] = support[cfg["language"]]
        cfg["family"] = REVISION
        cfg["protocol"] = "matched_gold_v2: same fixed prompts/policy/cohorts; 21 aligned gold TRAIN support glosses; one semantic attempt"
        cfg["previous_config"] = str(path.relative_to(ROOT))
        cfg["previous_config_sha256"] = sha256(path)
        cfg["gold_support_manifest"] = manifest_paths[cfg["language"]]
        cfg["gold_support_sha256"] = digest(cfg["support"])
        cfg["code_hashes"].update(code_hashes)
        mp = ROOT / cfg["gold_support_manifest"]
        cfg["context_hashes"][str(mp.relative_to(ROOT))] = sha256(mp)
        training = Path(json.loads(mp.read_text())["training_file"])
        cfg["context_hashes"][str(training)] = sha256(training)
        for key in ("results", "metrics"):
            cfg[key] = cfg[key].replace("/matched_v1/", f"/{REVISION}/")
        if cfg["test"] != old["test"] or cfg["policy"] != old["policy"] or cfg.get("predicted_glosses") != old.get("predicted_glosses"):
            raise ValueError("Unintended experimental change")
        cfg["fingerprint"] = digest(cfg)
        new = ROOT / "configs" / REVISION / path.name
        if new.exists() and json.loads(new.read_text()) != cfg and (ROOT / cfg["results"]).exists():
            raise ValueError("Cannot alter a configuration after generation")
        atomic_json(new, cfg)
        verify_inputs(cfg)
        active = dict(cfg, current_gloss=cfg.get("predicted_glosses", [""])[0])
        system, user = prompt(active, cfg["test"][0]["source"])
        atomic_json(DOCS / "prompt_examples" / path.name, {"config": str(new.relative_to(ROOT)),
                    "fingerprint": cfg["fingerprint"], "index": 0, "system": system, "user": user,
                    "support_nonempty_gold_glosses": 21, "user_prompt_sha256": digest([system, user])})
        mapping[str(path.relative_to(ROOT))] = str(new.relative_to(ROOT))
        catalog.append({k: cfg[k] for k in ("id", "model", "language", "source", "variant", "method", "material", "results", "metrics")} |
                       {"config": str(new.relative_to(ROOT)), "expected": len(cfg["test"])})
    jobs, covered = [], []
    for old_job in sorted((ROOT / "scripts/jobs/matched_v1").glob("*.sh")):
        if not old_job.name.startswith(tuple(m + "_" for m in MODELS)):
            continue
        text = old_job.read_text()
        group = re.search(r'--group\s+"?([^"\s]+)', text).group(1)
        paths = json.loads((ROOT / group).read_text())
        if any(p not in mapping for p in paths):
            raise ValueError(f"Unexpected configuration in {group}")
        new_group = group.replace("/matched_v1/", f"/{REVISION}/")
        atomic_json(ROOT / new_group, [mapping[p] for p in paths])
        covered.extend(paths)
        text = text.replace("matched_v1", REVISION).replace("--job-name=matched_", "--job-name=goldv2_")
        text = text.replace("runners/matched/run.py", f"runners/{REVISION}/run.py")
        dest = ROOT / "scripts/jobs" / REVISION / old_job.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        jobs.append(str(dest.relative_to(ROOT)))
    if len(covered) != 351 or set(covered) != set(mapping):
        raise ValueError("Duplicate or missing submitted conditions")
    atomic_json(ROOT / f"configs/{REVISION}/catalog.json", catalog)
    atomic_json(DOCS / "generation_jobs.json", jobs)
    if historical_hashes() != historical:
        raise ValueError("Historical files changed")
    print(f"PREPARED: {len(catalog)} conditions in {len(jobs)} full generation jobs; 21 gold supports each; history unchanged")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download-gitksan", action="store_true")
    prepare(p.parse_args().download_gitksan)
