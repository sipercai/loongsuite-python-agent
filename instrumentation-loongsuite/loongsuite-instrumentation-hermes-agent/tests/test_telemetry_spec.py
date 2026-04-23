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

import json
from types import SimpleNamespace

from opentelemetry import trace as trace_api
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler


def _spans_by_kind(span_exporter, span_kind: str):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.span.kind") == span_kind
    ]


def _assert_parent(child, parent):
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


def _messages(attr: str):
    return json.loads(attr)


class _FakeAgent:
    def __init__(
        self,
        *,
        session_id: str,
        model: str = "qwen-turbo",
        provider: str = "dashscope",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        tools=None,
    ):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.tools = tools or []
        self._user_id = None


def _tool_call(
    *,
    call_id: str = "call-1",
    name: str = "read_file",
    arguments: str = '{"path": "/tmp/demo.txt"}',
):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    *,
    content=None,
    finish_reason: str = "stop",
    tool_calls=None,
    model: str = "qwen-turbo",
    response_id: str = "resp-1",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        model=model,
        id=response_id,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _response_with_cached_prompt_tokens(
    *,
    content: str = "最终答案",
    finish_reason: str = "stop",
    model: str = "qwen-turbo",
    response_id: str = "resp-cache-1",
    prompt_tokens: int = 14,
    completion_tokens: int = 5,
    cached_tokens: int = 4,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        model=model,
        id=response_id,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def _codex_response(
    *,
    content: str = "最终答案",
    model: str = "gpt-5.4",
    response_id: str = "resp-codex-1",
    status: str = "completed",
    incomplete_reason: str | None = None,
    input_tokens: int = 12,
    output_tokens: int = 7,
    cached_tokens: int = 3,
    tool_calls=None,
):
    output = []
    if tool_calls:
        for tool_call in tool_calls:
            output.append(
                SimpleNamespace(
                    type="function_call",
                    id=tool_call.id,
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )
    elif content is not None:
        output.append(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=content)],
            )
        )

    return SimpleNamespace(
        output=output,
        output_text=content or "",
        model=model,
        id=response_id,
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason)
            if incomplete_reason is not None
            else None
        ),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def _codex_input_items(user_message: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": user_message,
        }
    ]


def _runtime(instrumentation_module, tracer_provider, meter_provider):
    tracer = trace_api.get_tracer(
        "hermes-agent-spec-tests",
        tracer_provider=tracer_provider,
    )
    handler = ExtendedTelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    metrics = instrumentation_module._HermesMetrics(
        meter_provider=meter_provider,
    )
    return SimpleNamespace(
        tracer=tracer,
        handler=handler,
        run_wrapper=instrumentation_module._RunConversationWrapper(handler),
        llm_wrapper=instrumentation_module._LLMCallWrapper(
            handler,
            metrics,
            streaming=False,
        ),
        streaming_llm_wrapper=instrumentation_module._LLMCallWrapper(
            handler,
            metrics,
            streaming=True,
        ),
        tool_wrapper=instrumentation_module._ToolCallWrapper(handler),
        tool_batch_wrapper=instrumentation_module._ToolBatchWrapper(handler),
    )


def test_agent_layer_reuses_existing_entry_instead_of_creating_a_new_one(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-parent")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="完成", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "完成"}

    with runtime.tracer.start_as_current_span(
        "gateway request",
        attributes={
            "gen_ai.operation.name": "enter",
            "gen_ai.span.kind": "ENTRY",
        },
    ) as ingress_span:
        runtime.run_wrapper(wrapped_run, agent, ("请回复：完成",), {})

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")

    assert len(entry_spans) == 1
    assert len(agent_spans) == 1
    assert agent_spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
    _assert_parent(agent_spans[0], ingress_span)


def test_agent_span_does_not_backfill_agent_id_from_session_id(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-agent-id")

    runtime.run_wrapper(
        lambda user_message: {"final_response": "完成"},
        agent,
        ("请回复：完成",),
        {},
    )

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    assert agent_span.attributes["gen_ai.conversation.id"] == "session-agent-id"
    assert "gen_ai.agent.id" not in agent_span.attributes


def test_agent_provider_and_system_are_normalized_for_hermes(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="session-provider",
        provider="alibaba",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="完成", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "完成"}

    runtime.run_wrapper(wrapped_run, agent, ("请回复：完成",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]

    assert agent_span.attributes["gen_ai.provider.name"] == "hermes"
    assert agent_span.attributes["gen_ai.agent.system"] == "hermes"
    assert llm_span.attributes["gen_ai.provider.name"] == "dashscope"


def test_final_text_response_uses_stop_as_step_finish_reason(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-stop")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="最终答案", finish_reason="stop"),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请回复最终答案",), {})

    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    assert step_span.name == "react step"
    assert step_span.attributes["gen_ai.operation.name"] == "react"
    assert step_span.attributes["gen_ai.react.finish_reason"] == "stop"


def test_react_steps_restore_agent_context_between_rounds(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-react")
    first_tool_call = _tool_call()

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
                response_id="resp-tool",
                prompt_tokens=11,
                completion_tokens=3,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: "tool_ok",
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="tool_ok",
                finish_reason="stop",
                response_id="resp-final",
                prompt_tokens=13,
                completion_tokens=2,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "tool", "content": "tool_ok"},
                    ],
                },
            ),
            {},
        )
        return {"final_response": "tool_ok"}

    runtime.run_wrapper(wrapped_run, agent, ("读取文件后回复内容",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    step_spans = sorted(
        _spans_by_kind(span_exporter, "STEP"),
        key=lambda span: span.attributes["gen_ai.react.round"],
    )
    llm_spans = _spans_by_kind(span_exporter, "LLM")
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    assert len(step_spans) == 2
    _assert_parent(step_spans[0], agent_span)
    _assert_parent(step_spans[1], agent_span)
    _assert_parent(llm_spans[0], step_spans[0])
    _assert_parent(tool_spans[0], step_spans[0])
    _assert_parent(llm_spans[1], step_spans[1])
    assert step_spans[1].parent.span_id != step_spans[0].context.span_id


def test_tool_span_captures_call_id_arguments_and_result(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-tool")
    first_tool_call = _tool_call(call_id="call-read-file")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: "tool_ok",
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        return {"final_response": "tool_ok"}

    runtime.run_wrapper(wrapped_run, agent, ("请调用 read_file",), {})

    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    tool_span = _spans_by_kind(span_exporter, "TOOL")[0]

    _assert_parent(tool_span, step_span)
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "read_file"
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call-read-file"
    assert "/tmp/demo.txt" in tool_span.attributes["gen_ai.tool.call.arguments"]
    assert tool_span.attributes["gen_ai.tool.call.result"] == "tool_ok"


def test_nested_tool_dispatch_reuses_outer_tool_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    dispatch_wrapper = instrumentation_module._ToolDispatchWrapper(runtime.handler)
    agent = _FakeAgent(session_id="session-tool-nested")
    first_tool_call = _tool_call(call_id="call-read-file")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: dispatch_wrapper(
                    lambda *inner_args, **inner_kwargs: "tool_ok",
                    None,
                    ("read_file", {"path": "/tmp/demo.txt"}),
                    {},
                ),
                agent,
                (
                    "read_file",
                    {"path": "/tmp/demo.txt"},
                    "task-1",
                    first_tool_call.id,
                ),
                {},
            ),
            agent,
            (),
            {},
        )
        return {"final_response": "tool_ok"}

    runtime.run_wrapper(wrapped_run, agent, ("请调用 read_file",), {})

    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    tool_spans = _spans_by_kind(span_exporter, "TOOL")

    assert len(tool_spans) == 1
    _assert_parent(tool_spans[0], step_span)
    assert tool_spans[0].attributes["gen_ai.tool.call.id"] == "call-read-file"


def test_agent_rolls_up_last_response_metadata_and_usage(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-rollup")
    first_tool_call = _tool_call()

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[first_tool_call],
                response_id="resp-1",
                prompt_tokens=11,
                completion_tokens=3,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(lambda: "tool_ok", agent, (), {})
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content="最终答案",
                finish_reason="stop",
                response_id="resp-2",
                prompt_tokens=13,
                completion_tokens=5,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "tool", "content": "tool_ok"},
                    ],
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请在调用工具后给出最终答案",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    assert agent_span.attributes["gen_ai.response.model"] == "qwen-turbo"
    assert agent_span.attributes["gen_ai.response.id"] == "resp-2"
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 24
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 8
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 32


def test_streaming_ttft_rolls_up_to_agent_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-stream")

    def wrapped_run(user_message):
        runtime.streaming_llm_wrapper(
            _streaming_api_response,
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "1 2 3"}

    runtime.run_wrapper(wrapped_run, agent, ("请数到 3",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]

    assert llm_span.attributes["gen_ai.operation.name"] == "chat"
    assert llm_span.attributes["gen_ai.response.time_to_first_token"] > 0
    assert agent_span.attributes["gen_ai.response.time_to_first_token"] > 0


def test_codex_responses_usage_and_finish_reason_are_normalized(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(
        session_id="session-codex",
        model="gpt-5.4",
        provider="openai-codex",
    )
    agent.api_mode = "codex_responses"

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _codex_response(
                content="最终答案",
                input_tokens=12,
                output_tokens=7,
                cached_tokens=3,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "instructions": "你是一个严谨的助手。",
                    "input": _codex_input_items(user_message),
                    "max_output_tokens": 256,
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请直接回答",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]
    step_span = _spans_by_kind(span_exporter, "STEP")[0]
    input_messages = _messages(llm_span.attributes["gen_ai.input.messages"])
    output_messages = _messages(llm_span.attributes["gen_ai.output.messages"])

    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 12
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 7
    assert llm_span.attributes["gen_ai.usage.total_tokens"] == 19
    assert llm_span.attributes["gen_ai.response.finish_reason"] == "stop"
    assert llm_span.attributes["gen_ai.response.finish_reasons"] == "[\"stop\"]"
    assert input_messages == [
        {
            "role": "system",
            "parts": [{"type": "text", "content": "你是一个严谨的助手。"}],
        },
        {
            "role": "user",
            "parts": [{"type": "text", "content": "请直接回答"}],
        },
    ]
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "最终答案"}],
            "finish_reason": "stop",
        }
    ]
    assert step_span.attributes["gen_ai.react.finish_reason"] == "stop"
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 12
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 7
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 19


def test_chat_completions_total_tokens_match_hermes_prompt_total(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-cache-chat")

    def wrapped_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response_with_cached_prompt_tokens(
                prompt_tokens=14,
                completion_tokens=5,
                cached_tokens=4,
            ),
            agent,
            (
                {
                    "model": agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "最终答案"}

    runtime.run_wrapper(wrapped_run, agent, ("请直接回答",), {})

    agent_span = _spans_by_kind(span_exporter, "AGENT")[0]
    llm_span = _spans_by_kind(span_exporter, "LLM")[0]

    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 14
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 5
    assert llm_span.attributes["gen_ai.usage.total_tokens"] == 19
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 14
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 5
    assert agent_span.attributes["gen_ai.usage.total_tokens"] == 19


def test_streaming_wrapper_suppresses_nested_llm_span(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    agent = _FakeAgent(session_id="session-nested-stream")

    runtime.streaming_llm_wrapper(
        lambda api_kwargs, **_: runtime.llm_wrapper(
            lambda inner_api_kwargs: _response(content="完成", finish_reason="stop"),
            agent,
            (api_kwargs,),
            {},
        ),
        agent,
        (
            {
                "model": agent.model,
                "messages": [{"role": "user", "content": "请回复：完成"}],
            },
        ),
        {},
    )

    llm_spans = _spans_by_kind(span_exporter, "LLM")
    step_spans = _spans_by_kind(span_exporter, "STEP")

    assert len(llm_spans) == 1
    assert len(step_spans) == 1


def test_child_agent_invocation_reuses_ingress_without_creating_child_entry(
    instrumentation_module,
    tracer_provider,
    meter_provider,
    span_exporter,
):
    runtime = _runtime(instrumentation_module, tracer_provider, meter_provider)
    parent_agent = _FakeAgent(session_id="session-parent")
    child_agent = _FakeAgent(session_id="session-child")
    delegate_call = _tool_call(call_id="call-delegate", name="delegate_task")

    def child_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="child_result", finish_reason="stop"),
            child_agent,
            (
                {
                    "model": child_agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        return {"final_response": "child_result"}

    def parent_run(user_message):
        runtime.llm_wrapper(
            lambda api_kwargs: _response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[delegate_call],
            ),
            parent_agent,
            (
                {
                    "model": parent_agent.model,
                    "messages": [{"role": "user", "content": user_message}],
                },
            ),
            {},
        )
        runtime.tool_batch_wrapper(
            lambda: runtime.tool_wrapper(
                lambda *args, **kwargs: runtime.run_wrapper(
                    child_run,
                    child_agent,
                    ("请读取文件并返回结果",),
                    {},
                ),
                parent_agent,
                (
                    "delegate_task",
                    {"goal": "请读取文件并返回结果"},
                    "task-1",
                    delegate_call.id,
                ),
                {},
            ),
            parent_agent,
            (),
            {},
        )
        runtime.llm_wrapper(
            lambda api_kwargs: _response(content="child_result", finish_reason="stop"),
            parent_agent,
            (
                {
                    "model": parent_agent.model,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "tool", "content": "child_result"},
                    ],
                },
            ),
            {},
        )
        return {"final_response": "child_result"}

    with runtime.tracer.start_as_current_span(
        "gateway request",
        attributes={
            "gen_ai.operation.name": "enter",
            "gen_ai.span.kind": "ENTRY",
        },
    ) as ingress_span:
        runtime.run_wrapper(parent_run, parent_agent, ("委派子 agent 完成任务",), {})

    entry_spans = _spans_by_kind(span_exporter, "ENTRY")
    agent_spans = _spans_by_kind(span_exporter, "AGENT")
    tool_spans = [
        span
        for span in _spans_by_kind(span_exporter, "TOOL")
        if span.attributes["gen_ai.tool.name"] == "delegate_task"
    ]
    parent_agent_span = next(
        span
        for span in agent_spans
        if span.attributes["gen_ai.conversation.id"] == "session-parent"
    )
    child_agent_span = next(
        span
        for span in agent_spans
        if span.attributes["gen_ai.conversation.id"] == "session-child"
    )

    assert len(entry_spans) == 1
    assert len(agent_spans) == 2
    _assert_parent(parent_agent_span, ingress_span)
    _assert_parent(child_agent_span, tool_spans[0])


def _streaming_api_response(api_kwargs, *, on_first_delta=None):
    if on_first_delta is not None:
        on_first_delta()
    return _response(content="1 2 3", finish_reason="stop")
