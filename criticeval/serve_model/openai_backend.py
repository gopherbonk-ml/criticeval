import time
from typing import Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


MAX_RETRIES = 10
MAX_WORKERS = 5


class OpenAIChatLLM:
    def __init__(self, engine_cfg, sampling_cfg):
        self.model = engine_cfg.model
        self.url = engine_cfg.base_url
        self.api_key = engine_cfg.api_key
        self.max_retries = MAX_RETRIES
        self.sampling_cfg = sampling_cfg
        self.client = None

    def start(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)

    def stop(self):
        self.client = None

    @staticmethod
    def extract_response(response):
        return response.choices[0].message.content

    def generate(self, prompt: str, images: Optional[list[str]] = None) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized. Call .start() before .generate().")

        generation_kwargs = {
            "model": self.model,
            "temperature": self.sampling_cfg.temperature,
            "top_p": self.sampling_cfg.top_p,
            "max_tokens": self.sampling_cfg.max_tokens,
        }

        user_content = [{"type": "text", "text": prompt}]
        if images:
            user_content.append({"type": "image_url", "image_url": {"url": images}})

        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": user_content}],
                    **generation_kwargs,
                )
                return self.extract_response(response)

            except Exception as e:
                last_err = e

                if attempt < self.max_retries:
                    sleep_s = min(10.0, 0.5 * (2 ** (attempt - 1)))
                    time.sleep(sleep_s)
                    continue

                raise RuntimeError(
                    f"OpenAI request failed after {self.max_retries} retries. Last error: {e!r}"
                ) from e

        raise RuntimeError(f"OpenAI request failed. Last error: {last_err!r}")

    def generate_many(
        self,
        prompts: list[str],
        images: Optional[list[Optional[list[str]]]] = None
    ) -> list[str]:
        results: list[Optional[str]] = [None] * len(prompts)

        def _one(i: int):
            return i, self.generate(prompts[i], images=images[i] if images else None)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_one, i) for i in range(len(prompts))]
            for fut in as_completed(futures):
                i, r = fut.result()
                results[i] = r

        return [r for r in results]