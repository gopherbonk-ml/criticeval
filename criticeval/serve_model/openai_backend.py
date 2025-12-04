import os
from typing import List, Optional

from .vllm_backend import LLMBackendConfig, LLMSamplingConfig


class OpenAIChatLLM:
    """Minimal adapter for OpenAI-compatible /chat/completions API."""

    def __init__(self, engine_cfg: LLMBackendConfig):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is required for backend_module='openai'") from e

        api_key = engine_cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key is required (engine.api_key or OPENAI_API_KEY)")

        client_kwargs = {"api_key": api_key}
        if engine_cfg.base_url:
            client_kwargs["base_url"] = engine_cfg.base_url

        self.client = OpenAI(**client_kwargs)
        self.model = engine_cfg.model

    def start():
        pass

    def stop():
        pass

    def generate(self, prompts: List[str], sampling_params: SamplingConfig, images: Optional[list[str]] = None):
        results = []
        for prompt in prompts:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens if sampling_params.max_tokens > 0 else None,
                )
            except Exception as e:
                raise RuntimeError(f"OpenAI completion failed: {e}") from e
            results.append(completion.choices[0].message.content)
        return results
