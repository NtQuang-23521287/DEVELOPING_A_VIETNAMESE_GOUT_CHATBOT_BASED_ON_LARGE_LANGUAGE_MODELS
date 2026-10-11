from __future__ import annotations

from gout_llmops.models.hf_backend import HuggingFaceBackend


def build_backend(config: dict):
    provider = config.get("provider", "huggingface")
    if provider == "huggingface":
        return HuggingFaceBackend(config)
    raise ValueError(f"Unsupported provider: {provider}")
