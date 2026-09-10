import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mtob_grammar.backends import (
    Generation,
    OpenRouterBackend,
    QwenBackend,
    ReasoningViolation,
    _contains_qwen_reasoning,
)
from mtob_grammar.chunking import CHUNK_OVERLAP, CHUNK_SIZE
from mtob_grammar.config import models, output_path, sources
from mtob_grammar.experiment import run
from mtob_grammar.prompts import appendix_c_prompt, long_context, passage_context
from mtob_grammar.retrieval import lcs_length, retrieve_gs


class FakeProcessor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        self.messages = messages
        return "USER prompt ASSISTANT"


class SequenceBackend:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, prompt, seed):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeEncoding:
    def encode(self, text):
        return text.split()


class SuiteTests(unittest.TestCase):
    def test_matrix_size(self):
        self.assertEqual(len(models()) * len(sources()) * 3, 60)

    def test_api_tsez_matrix_uses_first_99_examples(self):
        matrix = json.loads(
            (Path(__file__).parents[1] / "configs" / "experiment_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        api_rows = [
            row
            for row in matrix
            if row["source_id"] == "tsez"
            and row["model_key"] in {"gemini25flashlite", "gpt56luna"}
        ]
        self.assertEqual(len(api_rows), 6)
        self.assertTrue(all(row["configured_examples"] == 445 for row in api_rows))
        self.assertTrue(all(row["expected_examples"] == 99 for row in api_rows))
        self.assertTrue(all(row["example_selection"] == "first_99" for row in api_rows))

        qwen_rows = [
            row
            for row in matrix
            if row["source_id"] == "tsez" and row["model_key"] in {"qwen3", "qwen35"}
        ]
        self.assertTrue(all(row["expected_examples"] == 445 for row in qwen_rows))

    def test_prompt_snapshot_and_source_twice(self):
        context = passage_context("Testlang", ["first", "second"])
        prompt = appendix_c_prompt("Testlang", "Test Place", "UNIQUE-SOURCE", context)
        expected = (
            "Testlang is a language spoken in Test Place. Translate the following sentence from "
            "Testlang to English: UNIQUE-SOURCE\n\n"
            "To help with the translation, here is a passage retrieved from a Testlang-English grammar book:\nfirst\n\n"
            "To help with the translation, here is a passage retrieved from a Testlang-English grammar book:\nsecond\n\n"
            "Now write the translation.\nTestlang: UNIQUE-SOURCE\nEnglish translation:"
        )
        self.assertEqual(prompt, expected)
        self.assertEqual(prompt.count("UNIQUE-SOURCE"), 2)

    def test_long_context_snapshot(self):
        self.assertEqual(
            long_context("Testlang", "book text"),
            "To help with the translation, here is the full text of a Testlang-English grammar book:\n"
            "—\nbook text\nThis is the end of the Testlang-English grammar book.\n—",
        )

    def test_released_lcs_behavior(self):
        fixtures = json.loads((Path(__file__).parent / "fixtures" / "mtob_lcs_cases.json").read_text())
        for fixture in fixtures:
            self.assertEqual(lcs_length(fixture["query"], fixture["candidate"]), fixture["expected"])
        chunks = [{"text": "xxabc", "chunk_index": 0}, {"text": "zabcz", "chunk_index": 1}]
        ranked = retrieve_gs("abc", chunks, k=2)
        self.assertEqual({item["chunk_index"] for item in ranked}, {0, 1})

    def test_identity_metrics_match_paper_ranges(self):
        from mtob_grammar.evaluation import evaluate

        metrics = evaluate(["This is a test"], ["This is a test"])
        self.assertEqual(metrics["chrf"]["score"], 100.0)
        self.assertEqual(metrics["bleu"]["bleu"], 1.0)
        self.assertEqual(metrics["rouge"]["rougeL"], 1.0)
        self.assertEqual(metrics["character"]["cer_score"], 0.0)

    def test_qwen_template_gets_disable_flag(self):
        backend = QwenBackend.__new__(QwenBackend)
        backend.processor = FakeProcessor()
        rendered, metadata = backend._render("prompt")
        self.assertEqual(rendered, "USER prompt ASSISTANT")
        self.assertFalse(backend.processor.kwargs["enable_thinking"])
        self.assertFalse(metadata["enable_thinking"])
        self.assertEqual(backend.processor.messages, [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}])

    def test_qwen_reasoning_detection_uses_protocol_tags(self):
        self.assertTrue(_contains_qwen_reasoning("<think>hidden reasoning</think>Final"))
        self.assertTrue(_contains_qwen_reasoning("unfinished <THINK> block"))
        self.assertTrue(_contains_qwen_reasoning("orphan </think> block"))
        self.assertTrue(_contains_qwen_reasoning("Analysis: hidden reasoning\nFinal answer"))
        self.assertTrue(_contains_qwen_reasoning("Translation\n**Analysis:** hidden reasoning"))
        self.assertTrue(_contains_qwen_reasoning("Thinking process: hidden\nFinal answer"))
        self.assertFalse(_contains_qwen_reasoning("<think>\n\n</think>\nFinal"))
        self.assertFalse(_contains_qwen_reasoning("The expression means 'way of thinking: carefully'."))
        self.assertFalse(_contains_qwen_reasoning("This translation contains the word analysis: safely."))

    def test_reasoning_violation_preserves_audit_data(self):
        violation = ReasoningViolation(
            "excluded",
            text="<think>hidden</think>",
            metadata={"thinking": {"enable_thinking": False}},
        )
        self.assertEqual(violation.text, "<think>hidden</think>")
        self.assertFalse(violation.metadata["thinking"]["enable_thinking"])

    def _run_with_backend(self, backend, *, refusal_detector=None):
        examples = [
            SimpleNamespace(index=0, source="source zero", reference="reference zero"),
            SimpleNamespace(index=1, source="source one", reference="reference one"),
        ]
        source = {
            "language": "Testlang",
            "location": "Test Place",
            "test_file": "unused.txt",
            "test_n": 2,
        }
        model = {
            "backend": "qwen",
            "model_id": "test/model",
            "temperature": 0.05,
            "max_new_tokens": 32,
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            directory = Path(directory)
            result_path = directory / "results.jsonl"
            metrics_path = directory / "metrics.json"
            grammar_path = directory / "grammar.txt"
            grammar_path.write_text("grammar", encoding="utf-8")
            patches = [
                patch("mtob_grammar.experiment.source_config", return_value=source),
                patch("mtob_grammar.experiment.model_config", return_value=model),
                patch("mtob_grammar.experiment.input_path", return_value=Path("unused")),
                patch("mtob_grammar.experiment.parse_igt", return_value=examples),
                patch("mtob_grammar.experiment.output_paths", return_value=(result_path, metrics_path)),
                patch("mtob_grammar.experiment.output_path", return_value=grammar_path),
                patch("mtob_grammar.experiment.ensure_output_parent"),
                patch("mtob_grammar.experiment.create_backend", return_value=backend),
                patch("mtob_grammar.experiment.gpt2_encoding", return_value=FakeEncoding()),
                patch("mtob_grammar.experiment.ROOT", directory),
            ]
            if refusal_detector is not None:
                patches.append(patch("mtob_grammar.experiment.is_refusal", side_effect=refusal_detector))
            for active_patch in patches:
                active_patch.start()
            try:
                metrics = run("qwen35", "test", "gl")
            finally:
                for active_patch in reversed(patches):
                    active_patch.stop()
            records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
        return metrics, records

    def test_initial_reasoning_violation_is_retried(self):
        backend = SequenceBackend(
            [
                ReasoningViolation("excluded", text="<think>hidden</think>"),
                Generation("first translation", {}),
                Generation("second translation", {}),
            ]
        )
        metrics, records = self._run_with_backend(backend)
        self.assertEqual([record["status"] for record in records], ["ok", "ok"])
        self.assertEqual(len(records[0]["initial_reasoning_retries"]), 1)
        self.assertEqual(metrics["valid_examples"], 2)
        self.assertEqual(metrics["status_counts"], {"ok": 2})

    def test_retry_reasoning_violation_does_not_abort_experiment(self):
        backend = SequenceBackend(
            [
                Generation("REFUSAL", {"attempt": "initial"}),
                ReasoningViolation("excluded", text="Analysis: hidden", metadata={"attempt": "retry"}),
                Generation("first translation", {}),
                Generation("second translation", {}),
            ]
        )
        metrics, records = self._run_with_backend(
            backend,
            refusal_detector=lambda text: text == "REFUSAL",
        )
        self.assertEqual([record["status"] for record in records], ["ok", "ok"])
        self.assertEqual(records[0]["retry_response"], "first translation")
        self.assertEqual(records[0]["retry_reasoning_retries"][0]["response"], "Analysis: hidden")
        self.assertEqual(metrics["valid_examples"], 2)

    def test_openrouter_has_one_user_message_and_no_reasoning(self):
        backend = OpenRouterBackend.__new__(OpenRouterBackend)
        backend.model_id = "provider/model"
        backend.temperature = 0.05
        backend.max_new_tokens = 32
        backend.api_key = "test-key"
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "id": "request",
            "model": "provider/model",
            "choices": [{"message": {"content": "translation"}}],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
        with patch("mtob_grammar.backends.requests.post", return_value=response) as post:
            generation = backend.generate("PROMPT", seed=1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "PROMPT"}])
        self.assertNotIn("system", payload)
        self.assertEqual(payload["reasoning"], {"effort": "none", "exclude": True})
        self.assertEqual(generation.metadata["reasoning_tokens"], 0)

    def test_chunk_constants(self):
        self.assertEqual(CHUNK_SIZE, 512)
        self.assertEqual(CHUNK_OVERLAP, 256)

    def test_output_boundary(self):
        with self.assertRaises(ValueError):
            output_path("..", "outside.txt")


if __name__ == "__main__":
    unittest.main()
