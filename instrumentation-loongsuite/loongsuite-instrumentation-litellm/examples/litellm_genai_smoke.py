# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real LiteLLM smoke traffic for LoongSuite GenAI telemetry.

Run this under ``loongsuite-instrument`` with OTLP configured. The script
exercises non-streaming, streaming, and concurrent async completion calls.
"""

from __future__ import annotations

import asyncio
import os

import litellm

MODEL = os.getenv("LITELLM_MODEL", "qwen-turbo")
API_BASE = os.getenv(
    "LITELLM_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def _configure_provider() -> None:
    if os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    os.environ.setdefault("OPENAI_API_BASE", API_BASE)
    os.environ.setdefault("DASHSCOPE_API_BASE", API_BASE)
    litellm.telemetry = False


def run_non_streaming() -> None:
    response = litellm.completion(
        model=MODEL,
        custom_llm_provider="openai",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly one short sentence.",
            }
        ],
        temperature=0.1,
        max_tokens=64,
    )
    print("non_streaming:", response.choices[0].message.content[:80])


def run_streaming() -> None:
    stream = litellm.completion(
        model=MODEL,
        custom_llm_provider="openai",
        messages=[
            {
                "role": "user",
                "content": "Count from one to five, separated by commas.",
            }
        ],
        stream=True,
        temperature=0.1,
        max_tokens=64,
    )

    chunks = []
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                chunks.append(delta.content)
    print("streaming:", "".join(chunks)[:80])


async def run_concurrent() -> None:
    prompts = [
        "Give one word for sky color.",
        "Give one word for ocean color.",
        "Give one word for grass color.",
    ]

    async def call(prompt: str):
        return await litellm.acompletion(
            model=MODEL,
            custom_llm_provider="openai",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=32,
        )

    responses = await asyncio.gather(*(call(prompt) for prompt in prompts))
    print(
        "concurrent:",
        ", ".join(response.choices[0].message.content[:24] for response in responses),
    )


def main() -> None:
    _configure_provider()
    mode = os.getenv("LITELLM_SMOKE_MODE", "all").lower()

    if mode in ("all", "non_streaming"):
        run_non_streaming()
    if mode in ("all", "streaming"):
        run_streaming()
    if mode in ("all", "concurrent"):
        asyncio.run(run_concurrent())


if __name__ == "__main__":
    main()
