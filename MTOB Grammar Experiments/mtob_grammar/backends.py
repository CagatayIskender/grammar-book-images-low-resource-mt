from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class Generation:
    text: str
    metadata: dict[str, Any]


class ReasoningViolation(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        text: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.text = text
        self.metadata = metadata or {}


def _reasoning_tokens(usage: dict[str, Any]) -> int:
    candidates = [
        usage.get("reasoning_tokens"),
        usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
        usage.get("output_tokens_details", {}).get("reasoning_tokens"),
    ]
    return max((int(value) for value in candidates if value is not None), default=0)


def _reasoning_present(message: dict[str, Any], usage: dict[str, Any]) -> bool:
    return bool(message.get("reasoning") or message.get("reasoning_details") or _reasoning_tokens(usage) > 0)


def validate_openrouter_model(api_key: str, model_id: str, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    models = {item["id"]: item for item in response.json().get("data", [])}
    if model_id not in models:
        raise RuntimeError(f"OpenRouter model is not currently available: {model_id}")
    model = models[model_id]
    reasoning = model.get("reasoning") or {}
    if reasoning.get("mandatory") is True:
        raise RuntimeError(f"Reasoning is mandatory for {model_id}; refusing to run")
    supported = set(model.get("supported_parameters") or [])
    if supported and "reasoning" not in supported:
        raise RuntimeError(f"{model_id} does not advertise the reasoning control; refusing to run")
    return {
        "model_id": model_id,
        "reasoning": reasoning,
        "supported_parameters": sorted(supported),
        "reasoning_disable_allowed": True,
    }


class OpenRouterBackend:
    def __init__(self, model_id: str, temperature: float | None, max_new_tokens: int):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.validation = validate_openrouter_model(self.api_key, model_id)

    def generate(self, prompt: str, seed: int) -> Generation:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "seed": seed,
            "reasoning": {"effort": "none", "exclude": True},
            "provider": {"require_parameters": True, "data_collection": "deny"},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        retryable = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(6):
            started = time.time()
            try:
                response = requests.post(
                    OPENROUTER_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "MTOB Grammar Experiments",
                    },
                    json=payload,
                    timeout=300,
                )
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(min(60.0, 2**attempt + random.random()))
                continue
            if response.ok:
                data = response.json()
                message = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                if _reasoning_present(message, usage):
                    raise ReasoningViolation(
                        f"Reasoning violation from {self.model_id}: response is excluded from scoring"
                    )
                content = message.get("content") or ""
                if isinstance(content, list):
                    content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
                return Generation(
                    str(content).strip(),
                    {
                        "backend": "openrouter",
                        "requested_model": self.model_id,
                        "returned_model": data.get("model"),
                        "provider": data.get("provider"),
                        "request_id": data.get("id"),
                        "usage": usage,
                        "reasoning_tokens": _reasoning_tokens(usage),
                        "reasoning_request": payload["reasoning"],
                        "latency_seconds": time.time() - started,
                    },
                )
            if response.status_code not in retryable or attempt == 5:
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:1000]}")
            time.sleep(min(60.0, 2**attempt + random.random()))
        raise AssertionError("unreachable")


def _contains_qwen_reasoning(text: str) -> bool:
    # An empty paired block is how Qwen 3.5 represents disabled thinking.
    # Anything inside the block, an unmatched tag, or an explicit reasoning
    # preamble is output that must not be scored as a translation.
    without_empty_blocks = re.sub(
        r"<think>\s*</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if re.search(r"</?think>", without_empty_blocks, flags=re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"^\s*(?:[#>*_`-]+\s*)*(?:analysis|reasoning|thinking(?:\s+process)?)\s*:",
            without_empty_blocks,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


class QwenBackend:
    def __init__(self, model_id: str, temperature: float, max_new_tokens: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def _device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _render(self, prompt: str) -> tuple[str, dict[str, Any]]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        errors = []
        for mode, kwargs in (
            ("enable_thinking", {"enable_thinking": False}),
            ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
        ):
            try:
                rendered = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except TypeError as exc:
                errors.append(f"{mode}: {exc}")
                continue
            folded = rendered.casefold()
            think_start = folded.find("<think>")
            if think_start >= 0:
                think_end = folded.find("</think>", think_start)
                if think_end < 0 or folded[think_start + 7 : think_end].strip():
                    errors.append(f"{mode}: rendered template contains an enabled/nonempty think block")
                    continue
            return rendered, {"enable_thinking": False, "template_argument": mode}
        raise RuntimeError("Qwen thinking could not be disabled: " + " | ".join(errors))

    def generate(self, prompt: str, seed: int) -> Generation:
        rendered, thinking = self._render(prompt)
        inputs = self.processor(text=[rendered], return_tensors="pt").to(self._device())
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        started = time.time()
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
            )
        generated = [output[index][len(inputs.input_ids[index]) :] for index in range(len(output))]
        protocol_text = self.processor.batch_decode(
            generated, skip_special_tokens=False
        )[0].strip()
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        metadata = {
            "backend": "qwen",
            "model": self.model_id,
            "thinking": thinking,
            "temperature": self.temperature,
            "reasoning_protocol_checked": True,
            "latency_seconds": time.time() - started,
        }
        if _contains_qwen_reasoning(protocol_text):
            metadata["decoded_text"] = text
            raise ReasoningViolation(
                "Qwen output contains reasoning content; response is excluded from scoring",
                text=protocol_text,
                metadata=metadata,
            )
        return Generation(text, metadata)


def create_backend(config: dict[str, Any]):
    if config["backend"] == "openrouter":
        return OpenRouterBackend(config["model_id"], config["temperature"], config["max_new_tokens"])
    if config["backend"] == "qwen":
        return QwenBackend(config["model_id"], config["temperature"], config["max_new_tokens"])
    raise ValueError(f"Unknown backend: {config['backend']}")
