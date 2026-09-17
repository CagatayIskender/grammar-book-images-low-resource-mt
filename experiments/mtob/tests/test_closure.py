import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from mtob_grammar.closure_audit import (OUT, ROOT, SETTINGS, digest, offline_encoding, save, validate_rows)
from mtob_grammar.closure_evaluation import holm, paired_test, reusable, verify_score_inputs
from mtob_grammar.data import TranslationExample
from mtob_grammar.evaluation import evaluate, paper_bleu, paper_clean
from mtob_grammar.response_views import response_views


class ResponseTests(unittest.TestCase):
    def view(self, text, status="ok"):
        return response_views({"initial_response": text, "status": status})

    def test_plain_multisentence(self):
        text = "He left. She stayed.\n\nThey returned later."
        self.assertEqual(self.view(text)["extracted"], text)

    def test_labels_and_spans(self):
        for label in ("English translation:", "Translation:", "**English translation:**", "### Translation:"):
            text = label + " A translation.\nA second sentence.\n\n**Explanation:**\nExtra text."
            view = self.view(text)
            a, b = view["extracted_span"]
            self.assertEqual(view["extracted"], "A translation.\nA second sentence.")
            self.assertEqual(text[a:b], view["extracted"])

    def test_heading_variants(self):
        for heading in ("Explanation:", "### Breakdown:", "**Notes:**", "## Translation notes", "**Breakdown of the translation:**", "Notes"):
            self.assertEqual(self.view("A sentence.\n" + heading + "\nExtra.")["extracted"], "A sentence.")

    def test_explanation_first(self):
        v = self.view("**Explanation:**\nSome explanation.\nTranslation: It happened.\nNotes:\nOther.")
        self.assertEqual(v["extracted"], "It happened.")
        v = self.view("Explanation:\nSome explanation without translation.")
        self.assertTrue(v["ambiguous_extraction"])
        self.assertEqual(v["raw"], v["extracted"])

    def test_multiple_candidates(self):
        v = self.view("Translation: One.\nTranslation: Two.")
        self.assertTrue(v["ambiguous_extraction"])
        self.assertEqual(v["raw"], v["extracted"])
        v = self.view("An initial translation.\nExplanation:\nDetails.\nTranslation: Another translation.")
        self.assertTrue(v["ambiguous_extraction"])
        self.assertEqual(v["raw"], v["extracted"])

    def test_unclear_prefix_and_inline_explanation(self):
        v = self.view("Some preface.\nTranslation: One.")
        self.assertTrue(v["ambiguous_extraction"])
        text = "She wrote Explanation: in her notebook."
        self.assertEqual(self.view(text)["extracted"], text)

    def test_failures_blank_without_dropping(self):
        for status in ("refusal", "empty", "reasoning_violation"):
            v = self.view("Saved non-translation response.", status)
            self.assertEqual((v["raw"], v["extracted"]), ("", ""))
            self.assertEqual(v["original_response"], "Saved non-translation response.")

    def test_explanation_not_reasoning(self):
        self.assertEqual(self.view("Hello.\nExplanation:\nDetails.")["extracted"], "Hello.")

    def test_last_response_and_unknown_truncation(self):
        v = response_views({"initial_response": "First", "retry_response": "Last", "status": "ok"})
        self.assertEqual(v["raw"], "Last")
        self.assertEqual(v["response_field"], "retry_response")
        self.assertEqual(v["truncation"], "unknown")

    def test_reference_blind(self):
        a = {"prediction": "Text.\nNotes:\nDetails", "status": "ok", "reference": "Text."}
        b = {**a, "reference": "Different reference"}
        self.assertEqual(response_views(a), response_views(b))


class AuditMetricTests(unittest.TestCase):
    def row(self):
        return dict(index=0, source="source", reference="reference", model_key="qwen3", source_id="natugu",
                    condition="ge", model_id="Qwen/Qwen3-VL-8B-Instruct", status="ok")

    def test_duplicate_and_mismatch_rejected(self):
        examples = [TranslationExample(0, "source", "reference")]
        for rows in ([self.row(), self.row()], [{**self.row(), "reference": "bad"}], [{**self.row(), "index": 2}]):
            self.assertEqual(validate_rows(rows, examples, "qwen3", "natugu", "ge")[0], "invalid")
        self.assertEqual(validate_rows([], examples, "qwen3", "natugu", "ge")[0], "partial")

    def test_identity_and_failures(self):
        refs = ["This is a complete sentence.", "This is another sentence."]
        for preds in (refs, [refs[0], ""], ["", ""]):
            metrics = evaluate(preds, refs)
            self.assertEqual(metrics["num_examples"], 2)
            verify_score_inputs(preds, refs, metrics)
        identity = evaluate(refs, refs)
        self.assertEqual(identity["bleu"]["bleu"], 1)
        self.assertEqual(identity["character"]["cer_score"], 0)
        self.assertEqual(evaluate(["", ""], refs)["character"]["cer_score"], 1)

    def test_no_smoothing_short_sentences_and_punctuation(self):
        self.assertEqual(paper_clean("a =b, c!"), "ab c")
        self.assertEqual(paper_bleu(["short"], ["short"])["bleu"], 0)
        preds, refs = ["He came; she left!", ""], ["He came, she left.", "A long reference here."]
        verify_score_inputs(preds, refs, evaluate(preds, refs))

    def test_holm_family(self):
        self.assertEqual(holm([0.01, 0.03, 0.2]), [0.03, 0.06, 0.2])
        self.assertEqual(len(holm([0.5] * 120)), 120)

    def test_seeded_paired_bootstrap(self):
        left = ["One complete sentence is here", "Another complete sentence is here"]
        right = ["One different sentence is here", ""]
        # Only the test fixture uses 20 resamples; production CLI has no reduced-sample switch.
        with patch.dict(SETTINGS["bootstrap"], samples=20):
            a, b = paired_test(left, right, left), paired_test(left, right, left)
            identical = paired_test(left, left, left)
        self.assertEqual(a, b)
        self.assertTrue(all(v["p_raw"] == 1 and v["zero_effect_guard"] for v in identical))
        json.dumps(a, allow_nan=False)
        self.assertEqual({r["metric"] for r in a}, {"BLEU", "chrF"})

    def test_offline_gpt2_and_chunk_overlap(self):
        enc = offline_encoding()
        tokens = enc.encode("Long text repeated for a chunking fixture. " * 200)
        chunks = [tokens[i:i+512] for i in range(0, len(tokens), 256)]
        self.assertEqual(chunks[0][256:], chunks[1][:256])
        self.assertEqual(len(chunks[0]), 512)

    def test_write_boundary_and_resume_binding(self):
        with self.assertRaises(ValueError):
            save(ROOT / "results" / "forbidden.json", {})
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            path = Path(directory) / "checkpoint.json"
            binding = {"fingerprint": "a", "records_sha256": "b"}
            save(path, {"state": "complete", "binding": binding})
            self.assertTrue(reusable(path, binding))
            self.assertFalse(reusable(path, {**binding, "records_sha256": "changed"}))
            save(path, {"state": "error", "binding": binding})
            self.assertFalse(reusable(path, binding))

    def test_scoring_includes_failures_and_resumes(self):
        from mtob_grammar import closure_audit as ca, closure_evaluation as ce
        with tempfile.TemporaryDirectory(dir=OUT) as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(ca, "OUT", root))
            stack.enter_context(patch.object(ce, "OUT", root))
            stack.enter_context(patch.object(ce, "ROOT", root))
            stack.enter_context(patch.object(ce, "matrix", return_value=[("qwen3", "sample", "ge")]))
            stack.enter_context(patch.object(ce, "sources", return_value={"sample": {"test_n": 2}}))
            stack.enter_context(patch.object(ce, "require_audit", return_value={"fingerprint": "fixture", "inputs": {}}))
            stack.enter_context(patch.object(ce, "input_hashes", return_value={}))
            stack.enter_context(patch.object(ce, "write_report"))
            records = [
                {"index": 0, "source": "a", "reference": "This is a complete translation.", "status": "ok",
                 "raw": "This is a complete translation.", "extracted": "This is a complete translation."},
                {"index": 1, "source": "b", "reference": "Here is the second translation.", "status": "refusal", "raw": "", "extracted": ""},
            ]
            ca.save(ca.artifact("records", "qwen3", "sample", "ge"), {"fingerprint": "fixture", "records": records})
            ca.save(root / "metrics" / "qwen3" / "sample" / "metrics_ge.json", evaluate([records[0]["raw"]], [records[0]["reference"]]))
            ce.score()
            value = ca.load(ca.artifact("metrics", "qwen3", "sample", "ge"))
            for view in ("raw", "extracted"):
                self.assertEqual(value["views"][view]["num_examples"], 2)
                self.assertEqual(value["views"][view]["empty_predictions"], 1)
            self.assertEqual(value["historical_comparison"]["reproduced_status_ok_denominator"], 1)
            with patch.object(ce, "evaluate", side_effect=AssertionError("Should reuse valid checkpoint")):
                ce.score()


if __name__ == "__main__":
    unittest.main()
