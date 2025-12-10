import os
from typing import List, Optional
from openai import OpenAI

import json
import requests
import time
from .model_api import OpenAIConfig, LLMSamplingConfig
from dataclasses import asdict


class OpenAIChatLLM:
    def __init__(self, engine_cfg: OpenAIConfig, sampling_cfg):
        self.model = engine_cfg.model
        self.url = engine_cfg.base_url
        self.api_key = engine_cfg.api_key
        self.max_num_retries = 10
        self.sampling_cfg = sampling_cfg

        self.client = None

    def start(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.url
        )

    def stop(self):
        self.client = None

    @staticmethod
    def extract_response(response):
        return response.choices[0].message.content

    def generate(self, prompt: str, images: Optional[str] = None) -> str:
        generation_kwargs = {
            "model": self.model,
            "temperature": self.sampling_cfg.temperature,
            "top_p": self.sampling_cfg.top_p,
            "max_tokens": self.sampling_cfg.max_tokens,
        }

        user_content = [
            {"type": "text", "text": prompt},
        ]

        if images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": images},
                }
            )

        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_content},
            ],
            **generation_kwargs,
        )

        # см. ниже про extract_response
        output = self.extract_response(response)
        return output
