import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtob_grammar.backends import OpenRouterBackend, QwenBackend
from mtob_grammar.chunking import CHUNK_OVERLAP, CHUNK_SIZE
from mtob_grammar.config import models, output_path, sources
from mtob_grammar.prompts import appendix_c_prompt, long_context, passage_context
from mtob_grammar.retrieval import lcs_length, retrieve_gs


class FakeProcessor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        self.messages = messages
        return "USER prompt ASSISTANT"


class SuiteTests(unittest.TestCase):
    def test_matrix_size(self):
        self.assertEqual(len(models()) * len(sources()) * 3, 60)

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
