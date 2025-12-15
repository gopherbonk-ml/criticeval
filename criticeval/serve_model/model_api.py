from dataclasses import dataclass, field
from typing import Optional, Protocol, Literal, Union, Mapping, Any


@dataclass
class LLMSamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 2048


@dataclass
class OpenAIConfig:
    model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class VLLMConfig:
    model: str = ""
    tokenizer: Optional[str] = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    num_devices: int = 1
    max_model_length: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    enforce_eager: bool = False
    device: str = "auto"


@dataclass
class LLMBackendConfig:
    backend_module: Literal["openai", "vllm", "vllm_npu"] = "vllm"
    device: str = "cpu"
    num_devices: int = 1
    num_devices_per_worker: float = 1.0
    
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)

    def __post_init__(self):
        from collections.abc import Mapping

        if isinstance(self.vllm, Mapping):
            self.vllm = VLLMConfig(**self.vllm)

        if isinstance(self.openai, Mapping):
            self.openai = OpenAIConfig(**self.openai)


class Backend(Protocol):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def generate(
            self,
            prompt: str,
            images: Optional[list[str]] = None
        ) -> str:
        pass

    def generate_many(
        self,
        prompts: list[str],
        images: Optional[list[Optional[list[str]]]] = None
    ) -> list[str]:
        pass


def build_backend(backend_cfg: LLMBackendConfig, sampling_cfg: LLMSamplingConfig):
    if backend_cfg.backend_module in ("vllm", "vllm_npu"):
        from .vllm_backend import VLLMBackend
        print(backend_cfg.vllm)
        return VLLMBackend(backend_cfg.vllm, sampling_cfg)

    if backend_cfg.backend_module == "openai":
        from .openai_backend import OpenAIChatLLM

        return OpenAIChatLLM(backend_cfg.openai, sampling_cfg)

    raise ValueError(f"Unknown backend module: {backend_cfg.backend_module}")


class LLMHost:
    def __init__(self, cfg: LLMBackendConfig, sampling_params: LLMSamplingConfig):
        self.cfg = cfg
        self.sampling_params = sampling_params
        self.backend: Optional[Backend] = None

    def start(self):
        self.backend = build_backend(self.cfg, self.sampling_params)
        self.backend.start()

    def stop(self):
        if self.backend:
            self.backend.stop()
        self.backend = None

    def generate(self, prompt: str, images: Optional[list[str]] = None) -> str:
        if self.backend is None:
            raise RuntimeError("LLM host not started. Call start() before generate().")
        return self.backend.generate(prompt, images=images)

    def generate_many(
        self,
        prompts: list[str],
        images: Optional[list[Optional[list[str]]]] = None
    ) -> list[str]:
        results: list[str] = []

        if self.backend is None:
            raise RuntimeError("LLM host not started. Call start() before many_generate().")

        generate_many = getattr(self.backend, "generate_many", None)
        if callable(generate_many):
            results = generate_many(prompts, images=images)
            return results

        for prompt, imgs in zip(prompts, imgs):
            results.append(
                self.backend.generate(
                    prompt, images=imgs
                )
            )
        return results

        
