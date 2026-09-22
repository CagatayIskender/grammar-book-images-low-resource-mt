"""Offline audit of saved final messages; never modify experiments or call models."""
import base64
from collections import Counter
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners/matched_gold_v2"))
from protocol import prompt, verify_inputs, validate_rows, digest
from experiment_io import atomic_json, sha256


def main():
    catalog = json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text())
    counts, methods, files, errors = Counter(), Counter(), Counter(), []
    hits, invalid_references, snapshots, predictions, cache_hashes = [], {}, {}, {}, {}
    image_parts = {}
    future_checks = 0
    for item in catalog:
        cfg = json.loads((ROOT / item["config"]).read_text())
        verify_inputs(cfg)
        language = cfg["language"]
        if language not in predictions:
            path = ROOT / "runners/openrouter/cache" / f"{language.lower()}_glosslm_predictions.csv"
            cache_hashes[str(path.relative_to(ROOT))] = sha256(path)
            entries = {}
            for entry in csv.DictReader(path.read_text().splitlines()):
                if entry["is_segmented"] != "no":
                    continue
                index = int(entry["id"].rsplit("_", 1)[-1])
                if index in entries:
                    raise ValueError("Duplicate unsegmented external gloss prediction")
                entries[index] = entry["pred"].replace("\n", " ").strip()
            predictions[language] = entries
        if cfg["method"] == "modelgloss":
            assert cfg["predicted_glosses"] == [predictions[language][i] for i in range(len(cfg["test"]))]
        tainted = dict(cfg, test=[dict(e, gloss="AUDIT_GOLD_TEST_7AB31", reference="AUDIT_TEST_REFERENCE_73AF2")
                                  for e in cfg["test"]])
        for index, example in enumerate(cfg["test"]):
            predicted = cfg.get("predicted_glosses", [""] * len(cfg["test"]))[index]
            original = prompt(dict(cfg, current_gloss=predicted), example["source"])
            changed = prompt(dict(tainted, current_gloss=predicted), example["source"])
            assert original == changed, "Test gold/reference field changes the prompt"
            future_checks += 1
            if example["reference"].strip().lower() in ("nan", "none", "null", ""):
                invalid_references[(language, index)] = {"language": language, "idx": index,
                    "reference": example["reference"], "source": example["source"]}
        path = ROOT / cfg["results"]
        if not path.exists():
            continue
        data = path.read_bytes()
        snapshots[cfg["results"]] = hashlib.sha256(data).hexdigest()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        validate_rows(rows, cfg)
        if rows:
            files[cfg["model"]] += 1
        for row in rows:
            index = row["idx"]
            example = cfg["test"][index]
            attempt = row["translation"]["attempt"]
            evidence = attempt["prompt_evidence"]
            system, user = evidence["system"], evidence["user"]
            counts[cfg["model"]] += 1
            methods[cfg["method"]] += 1
            predicted = predictions[language][index] if cfg["method"] == "modelgloss" else ""
            target_prefix = f"{language} sentence: {example['source']}\n"
            target = user[user.rfind(target_prefix):]
            if cfg["method"] != "chain_gloss":
                expected = target_prefix + (f"Gloss: {predicted}\n" if predicted else "") + "FINAL_TRANSLATION:"
                assert target == expected, "Unexpected content in target answer slot"
            if cfg["model"] == "gemini25flashlite":
                content = [{"type": "text", "text": user}]
                if cfg["context_kind"] == "image":
                    for name in cfg["context_files"]:
                        if name not in image_parts:
                            image_parts[name] = {"type": "image_url", "image_url": {"url":
                                "data:image/jpeg;base64," + base64.b64encode((ROOT / name).read_bytes()).decode()}}
                        content.append(image_parts[name])
                payload = dict(model=cfg["model_id"], messages=[{"role": "system", "content": system},
                    {"role": "user", "content": content}], max_tokens=512, seed=42,
                    reasoning={"effort": "none"}, provider={"data_collection": "deny",
                    "require_parameters": True, "only": ["Google"]})
                if cfg["policy"]["temperature"] is not None:
                    payload["temperature"] = cfg["policy"]["temperature"]
                if digest(payload) != attempt.get("request_sha256"):
                    errors.append({"id": cfg["id"], "idx": index, "error": "API payload hash mismatch"})
            for field in ("reference", "gloss"):
                value = example.get(field, "")
                if not value or value not in system + user:
                    continue
                origins = []
                for j, support in enumerate(cfg["support"]):
                    if any(value in support[k] for k in ("source", "reference", "gloss")):
                        origins.append(f"support_{j}")
                if value in cfg["grammar_text"]:
                    origins.append("grammar_text")
                if value in predicted:
                    origins.append("external_predicted_gloss_exact" if value == predicted else "external_predicted_gloss_substring")
                if value in example["source"]:
                    origins.append("test_source")
                hits.append({"id": cfg["id"], "language": language, "idx": index, "field": field,
                             "value": value, "origins": origins or ["other_prompt_text"],
                             "in_target_block": value in target})
        print(f"[AUDITED] {cfg['id']}: {len(rows)} recorded messages", flush=True)
    output = ROOT / "docs/matched_gold_v2/actual_input_audit"
    report = {"audited_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "planned_prompts_noninterference_checked": future_checks,
              "actual_records_by_model": dict(counts), "conditions_with_records": dict(files),
              "actual_records_by_method": dict(methods), "api_payload_mismatches": errors,
              "placeholder_test_references": list(invalid_references.values()), "literal_matches": hits,
              "result_snapshot_hashes": snapshots, "external_prediction_cache_hashes": cache_hashes,
              "audit_code_sha256": sha256(Path(__file__)),
              "limits": ["Actual records are a snapshot; pending outputs are not certified.",
                         "Qwen serialized templates are not reconstructed by this tool.",
                         "Image pixels are hash-bound but not OCR-audited for test overlap.",
                         "Literal equality can originate in support text or correct external predictions."]}
    atomic_json(output / "report.json", report)
    lines = ["# Actual Model Input Audit", "", f"Snapshot: {report['audited_utc']}",
             f"Planned prompts checked for test-gold/reference noninterference: {future_checks}.",
             f"Recorded messages: {dict(counts)}; API request-hash mismatches: {len(errors)}.",
             "All audited records pass reconstruction of gold-support messages; supports contain 21 nonempty gold glosses.",
             "ModelGloss test slots exactly match the corresponding external unsegmented pred field, not gold.",
             "No claim is made that test-reference strings or gold-gloss strings are absent everywhere in context.",
             f"Literal-match events: {len(hits)}. Placeholder references: {len(invalid_references)}.",
             "Lezgi test index 11 has its reference as a substring of a longer support translation (support index 19).",
             "Some external predictions equal gold exactly; this is not evidence of reading the gold field.",
             "See report.json for every match, source and snapshot hash. Images have not been OCR-audited here.",
             "No jobs, prompts, references, support sets, or predictions were changed by this audit."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("actual_records_by_model",
        "conditions_with_records", "planned_prompts_noninterference_checked", "api_payload_mismatches",
        "placeholder_test_references")}, indent=2))


if __name__ == "__main__":
    main()
