"""Adapt HF SDPA to torch 2.7 efficient attention without native GQA dispatch."""
from contextlib import contextmanager

ATTENTION_TAG = "efficient_gqa_v1"


class _ExpandedHeads:
    """Delegate module settings, hiding GQA after K/V have already been expanded."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        if name == "num_key_value_groups":
            raise AttributeError(name)
        return getattr(self.module, name)


def expanded_sdpa(module, query, key, value, attention_mask, **kwargs):
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    if any(x.dtype != torch.float32 for x in (query, key, value)):
        raise RuntimeError("FP32 attention requires FP32 query/key/value")
    heads, kv_heads = query.shape[1], key.shape[1]
    if kv_heads != value.shape[1] or heads % kv_heads:
        raise ValueError("Invalid grouped-query head counts")
    groups = heads // kv_heads
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    # Keep the installed HF mask, causal/cache, scaling and output-layout semantics.
    # Torch 2.7 efficient SDPA cannot dispatch native GQA with unequal heads.
    return sdpa_attention_forward(_ExpandedHeads(module), query, key, value, attention_mask, **kwargs)


@contextmanager
def expanded_sdpa_context():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = expanded_sdpa
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original


def verify_efficient_kernel():
    """Fail before loading weights if FP32 efficient GQA/cache/vision probes fail."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from types import SimpleNamespace

    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Kernel verification requires exactly one H100")
    generator = torch.Generator(device="cuda").manual_seed(2026)
    with torch.inference_mode():
        for heads, kv_heads, dim in ((32, 8, 128), (16, 4, 256), (16, 16, 72)):
            for mode in ("causal", "decode", "masked", "vision"):
                nq, nk = (1, 17) if mode == "decode" else (17, 17)
                q = torch.randn((1, heads, nq, dim), device="cuda", dtype=torch.float32, generator=generator)
                k = torch.randn((1, kv_heads, nk, dim), device="cuda", dtype=torch.float32, generator=generator)
                v = torch.randn(k.shape, device="cuda", dtype=torch.float32, generator=generator)
                mask = torch.ones((nq, nk), dtype=torch.bool, device="cuda").tril()[None, None] if mode == "masked" else None
                module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=mode != "vision")
                # The math reference uses native GQA, independently of our expansion.
                with sdpa_kernel(SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, is_causal=mode == "causal", enable_gqa=True,
                    ).transpose(1, 2).contiguous()
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    actual, _ = expanded_sdpa(module, q, k, v, mask)
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    torch.cuda.synchronize()
    print("[PASS] 12 FP32 efficient kernel probes: GQA, cache, masks, vision", flush=True)
