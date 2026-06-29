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

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.autogen import patch
from opentelemetry.instrumentation.autogen.semantic_conventions import (
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GenAIOperation,
    GenAISpanKind,
)
from opentelemetry.sdk.trace import TracerProvider


class Handler:
    def __init__(self):
        self.calls = []

    def start_invoke_agent(self, invocation):
        self.calls.append(("start_agent", invocation.agent_name))

    def stop_invoke_agent(self, invocation):
        self.calls.append(("stop_agent", invocation.agent_name))

    def fail_invoke_agent(self, invocation, error):
        self.calls.append(("fail_agent", invocation.agent_name, error.type.__name__))

    def start_llm(self, invocation):
        self.calls.append(("start_llm", invocation.request_model))

    def stop_llm(self, invocation):
        self.calls.append(("stop_llm", invocation.request_model))

    def fail_llm(self, invocation, error):
        self.calls.append(("fail_llm", invocation.request_model, error.type.__name__))


class Agent:
    name = "assistant"
    description = "answers"


class ModelClient:
    _create_args = {"model": "qwen-plus"}


class Usage:
    prompt_tokens = 1
    completion_tokens = 2


class CreateResult:
    content = "done"
    finish_reason = "stop"
    usage = Usage()


class ModelContext:
    async def get_messages(self):
        return []


def _set_handler(handler: Handler):
    patch._get_handler.handler = handler  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_agent_wrapper_starts_and_stops_invocation():
    handler = Handler()
    _set_handler(handler)

    async def wrapped():
        yield "item"

    items = [
        item
        async for item in patch._on_messages_stream_wrapper(wrapped, Agent(), (), {})
    ]

    assert items == ["item"]
    assert handler.calls == [("start_agent", "assistant"), ("stop_agent", "assistant")]


@pytest.mark.asyncio
async def test_agent_wrapper_fails_invocation_on_exception():
    handler = Handler()
    _set_handler(handler)

    async def wrapped():
        yield "item"
        raise ValueError("boom")

    with pytest.raises(ValueError):
        async for _ in patch._on_messages_stream_wrapper(wrapped, Agent(), (), {}):
            pass

    assert handler.calls == [
        ("start_agent", "assistant"),
        ("fail_agent", "assistant", "ValueError"),
    ]


@pytest.mark.asyncio
async def test_agent_wrapper_skips_when_native_autogen_agent_span_is_active():
    handler = Handler()
    _set_handler(handler)
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    async def wrapped():
        yield "native"

    with tracer.start_as_current_span(
        "invoke_agent assistant",
        attributes={
            GEN_AI_OPERATION_NAME: GenAIOperation.INVOKE_AGENT,
            GEN_AI_PROVIDER_NAME: AUTOGEN_PROVIDER_NAME,
            GEN_AI_SPAN_KIND: GenAISpanKind.AGENT,
        },
    ):
        items = [
            item
            async for item in patch._on_messages_stream_wrapper(
                wrapped, Agent(), (), {}
            )
        ]

    assert items == ["native"]
    assert handler.calls == []


@pytest.mark.asyncio
async def test_llm_wrapper_starts_and_stops_invocation():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        yield CreateResult()

    items = [
        item
        async for item in patch._call_llm_wrapper(
            wrapped,
            None,
            (),
            {
                "model_client": ModelClient(),
                "system_messages": [],
                "model_context": ModelContext(),
                "workbench": [],
                "handoff_tools": [],
                "agent_name": "assistant",
            },
        )
    ]

    assert len(items) == 1
    assert handler.calls == [("start_llm", "qwen-plus"), ("stop_llm", "qwen-plus")]


@pytest.mark.asyncio
async def test_llm_wrapper_fails_invocation_when_closed_early():
    handler = Handler()
    _set_handler(handler)

    async def wrapped(*args, **kwargs):
        yield "chunk"
        yield CreateResult()

    generator = patch._call_llm_wrapper(
        wrapped,
        None,
        (),
        {
            "model_client": ModelClient(),
            "system_messages": [],
            "model_context": ModelContext(),
            "workbench": [],
            "handoff_tools": [],
            "agent_name": "assistant",
        },
    )

    assert await generator.__anext__() == "chunk"
    await generator.aclose()

    assert handler.calls == [
        ("start_llm", "qwen-plus"),
        ("fail_llm", "qwen-plus", "GeneratorExit"),
    ]
