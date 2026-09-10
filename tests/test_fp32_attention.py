"""Run in both Qwen environments; GPU checks run inside the preflight job."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
from fp32_attention import expanded_sdpa, expanded_sdpa_context


class AttentionTests(unittest.TestCase):
    def test_gqa_matches_native_math(self):
        generator = torch.Generator().manual_seed(34)
        for heads, kv_heads, dim in ((32, 8, 128), (16, 4, 256), (16, 16, 72)):
            for mode in ("causal", "decode", "masked", "vision"):
                with self.subTest(heads=heads, dim=dim, mode=mode):
                    nq, nk = (1, 17) if mode == "decode" else (17, 17)
                    q = torch.randn(1, heads, nq, dim, generator=generator)
                    k = torch.randn(1, kv_heads, nk, dim, generator=generator)
                    v = torch.randn(k.shape, generator=generator)
                    mask = None
                    if mode == "masked":
                        mask = torch.zeros(1, 1, nq, nk).masked_fill(~torch.ones(nq, nk, dtype=torch.bool).tril(), float("-inf"))
                    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=True)
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=mask, scale=0.13,
                            is_causal=mode == "causal", enable_gqa=True,
                        ).transpose(1, 2).contiguous()
                        options = {"is_causal":False} if mode == "vision" else {}
                        with patch("torch.nn.functional.scaled_dot_product_attention", wraps=torch.nn.functional.scaled_dot_product_attention) as call:
                            actual, weights = expanded_sdpa(module, q, k, v, mask, scaling=0.13, **options)
                            self.assertNotIn("enable_gqa", call.call_args.kwargs)
                            self.assertEqual(call.call_args.args[1].shape[1], heads)
                        self.assertIsNone(weights)
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_registry_is_restored_even_after_failure(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with expanded_sdpa_context():
                self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], expanded_sdpa)
                raise RuntimeError("probe")
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], original)

    def test_rejects_lower_precision_and_invalid_heads(self):
        module = SimpleNamespace(num_key_value_groups=4)
        q = torch.zeros(1, 4, 3, 8)
        kv = torch.zeros(1, 1, 3, 8)
        with self.assertRaisesRegex(RuntimeError, "FP32"):
            expanded_sdpa(module, q.half(), kv.half(), kv.half(), None)
        with self.assertRaisesRegex(ValueError, "head"):
            expanded_sdpa(module, q, kv.repeat(1, 3, 1, 1), kv, None)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
