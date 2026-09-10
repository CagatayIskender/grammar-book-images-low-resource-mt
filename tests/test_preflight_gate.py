import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import submit_jobs
from experiment_io import CHAIN_PROMPT_REVISION, preflight_stem, sha256
from fp32_attention import ATTENTION_TAG


class PreflightGateTests(unittest.TestCase):
    def test_complete_current_records_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runners").mkdir()
            (root / "results/preflight").mkdir(parents=True)
            cfg = {"id":"probe", "language":"Tsez", "condition":"chain_gloss", "context_files":["context.txt"], "prediction_key":"chain"}
            (root / "context.txt").write_text("unchanged material")
            provenance = {"config":cfg, "attention":ATTENTION_TAG, "dtype":"float32", "prompt_module_hashes":{}, "chain_prompt_revision":CHAIN_PROMPT_REVISION}
            for key, name in (("runner_sha256", "run_audited_context.py"), ("io_sha256", "experiment_io.py"), ("attention_sha256", "fp32_attention.py")):
                path = root / "runners" / name
                path.write_text(name)
                provenance[key] = sha256(path)
            provenance["context_hashes"] = {"context.txt":sha256(root / "context.txt")}
            examples = [{"source":f"source{i}", "reference":f"reference{i}"} for i in range(3)]
            provenance.update(test=examples, support=examples)
            fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            result = root / "results/preflight" / (preflight_stem(cfg, ATTENTION_TAG) + ".jsonl")
            result.with_suffix(".provenance.json").write_text(json.dumps(provenance))
            result.with_suffix(".status.json").write_text(json.dumps({"status":"complete", "fingerprint":fingerprint, "records":3, "expected":3}))
            rows = [dict(ex, idx=i, fingerprint=fingerprint, chain={"prediction":"Translation.", "error":None}) for i, ex in enumerate(examples)]
            original = "".join(json.dumps(r)+"\n" for r in rows)
            result.write_text(original)
            with patch.object(submit_jobs, "ROOT", root), patch.object(submit_jobs, "dataset", return_value=examples):
                self.assertTrue(submit_jobs.preflight_ready(cfg))
                result.write_text(json.dumps(rows[0])+"\n")
                self.assertFalse(submit_jobs.preflight_ready(cfg))
                rows[1]["chain"]["error"] = "invalid_format"
                result.write_text("".join(json.dumps(r)+"\n" for r in rows))
                self.assertFalse(submit_jobs.preflight_ready(cfg))
                result.write_text(original)
                (root / "runners/fp32_attention.py").write_text("changed code")
                self.assertFalse(submit_jobs.preflight_ready(cfg))

    def test_old_failed_preflight_does_not_pass(self):
        cfg = next(c for c in json.loads((ROOT / "configs/experiment_catalog.json").read_text()) if c["condition"] == "chain_gloss")
        with tempfile.TemporaryDirectory() as folder, patch.object(submit_jobs, "ROOT", Path(folder)):
            self.assertFalse(submit_jobs.preflight_ready(cfg))

    def test_revision_separates_chain_only(self):
        cfg = {"id":"probe", "condition":"chain_gloss"}
        self.assertEqual(preflight_stem(cfg, ATTENTION_TAG), f"probe_{ATTENTION_TAG}_{CHAIN_PROMPT_REVISION}")
        for condition in ("shot", "modelgloss"):
            cfg["condition"] = condition
            self.assertEqual(preflight_stem(cfg, ATTENTION_TAG), f"probe_{ATTENTION_TAG}")


if __name__ == "__main__":
    unittest.main()
