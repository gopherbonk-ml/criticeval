from dataclasses import dataclass
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
    api_key: str = Optional[str]
    base_url: str = Optional[str]


@dataclass
class VLLMConfig:
    model: str = ""
    tokenizer: Optional[str] = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    enforce_eager: bool = False
    device: str = "auto"


@dataclass
class LLMBackendConfig:
    backend_module: Literal["openai", "vllm", "vllm_npu"] = "vllm"

    vllm: VLLMConfig = VLLMConfig()
    openai: OpenAIConfig = OpenAIConfig()


class Backend(Protocol):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def generate(self, prompt: str, images: Optional[list[str]] = None):
        pass


def build_backend(backend_cfg: LLMBackendConfig, sampling_cfg: LLMSamplingConfig):
    if backend_cfg.backend_module in ("vllm", "vllm_npu"):
        from .vllm_backend import VLLMBackend

        return LLMBackendConfig(backend_cfg, sampling_cfg)

    if backend_cfg.backend_module == "openai":
        from .openai_backend import OpenAIChatBackend

        return LLMBackendConfig(backend_cfg, sampling_cfg)

    raise ValueError(f"Unknown backend module: {backend_cfg.backend_module}")


class LLMHost:
    def __init__(self, cfg: VLLMConfig, sampling_params: LLMSamplingConfig):
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
