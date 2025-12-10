from typing import Optional

from .model_api import VLLMConfig, LLMSamplingConfig
from .utils import import_backend_module, detect_device, empty_cashe

import vllm
import os

from PIL import Image

class VLLMBackend:
    def __init__(self, engine_cfg: VLLMConfig, sampling_cfg: LLMSamplingConfig):
        self.engine_cfg = engine_cfg
        self.sampling_cfg = sampling_cfg
        self.llm = None
        self.sampling_params = None
        print(os.getenv("ASCEND_RT_VISIBLE_DEVICES"))

    def start(self):
        LLM = vllm.LLM
        SamplingParamsCls = vllm.SamplingParams

        engine_args = {
            "model": self.engine_cfg.model,
            "tokenizer": self.engine_cfg.tokenizer,
            "dtype": self.engine_cfg.dtype,
            "tensor_parallel_size": self.engine_cfg.tensor_parallel_size,
            "max_model_len": self.engine_cfg.max_model_length,
            "gpu_memory_utilization": self.engine_cfg.gpu_memory_utilization,
            "trust_remote_code": self.engine_cfg.trust_remote_code,
            "enforce_eager": self.engine_cfg.enforce_eager
        }
        engine_args = {k: v for k, v in engine_args.items() if v is not None}

        sampling_args = {
            "temperature": self.sampling_cfg.temperature,
            "top_p": self.sampling_cfg.top_p,
            "top_k": self.sampling_cfg.top_k,
            "max_tokens": self.sampling_cfg.max_tokens,
        }
        sampling_args = {k: v for k, v in sampling_args.items() if v is not None}

        self.llm = LLM(**engine_args)
        self.sampling_params = SamplingParamsCls(**sampling_args)

    def stop(self):
        self.llm, self.sampling_params = None, None
        empty_cashe()

    def generate(self, prompt: str, images: Optional[list[str]] = None) -> str:
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("vLLM backend not started. Call start() before generate().")
        if images:
            images = [Image.open(image) for image in images]
            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images}
                },
                self.sampling_params
            )
        else:
            outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text
