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

"""OpenTelemetry Hermes Agent instrumentation."""

from __future__ import annotations

import json
import time
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import metrics as metrics_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode

_instruments = ("openai >= 1.0.0",)

_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
_GEN_AI_SESSION_ID = "gen_ai.session.id"
_GEN_AI_SPAN_KIND = "gen_ai.span.kind"

_GEN_AI_KIND_AGENT = "AGENT"
_GEN_AI_KIND_ENTRY = "ENTRY"
_GEN_AI_KIND_LLM = "LLM"
_GEN_AI_KIND_STEP = "STEP"
_GEN_AI_KIND_TOOL = "TOOL"

_GEN_AI_OP_CHAT = "chat"
_GEN_AI_OP_ENTER = "enter"
_GEN_AI_OP_EXECUTE_TOOL = "execute_tool"
_GEN_AI_OP_INVOKE_AGENT = "invoke_agent"
_GEN_AI_OP_REACT = "react"

_HERMES_PROVIDER = "hermes-agent"


def _provider_name(instance: Any) -> str:
    provider = str(getattr(instance, "provider", "") or "").strip().lower()
    if provider:
        return provider

    base_url = str(getattr(instance, "base_url", "") or "").lower()
    if "dashscope" in base_url:
        return "dashscope"
    if "openai" in base_url:
        return "openai"
    return "custom"


def _safe_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=lambda obj: getattr(obj, "__dict__", str(obj)),
    )


def _serialize_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None

    serialized = []
    for message in messages:
        if isinstance(message, dict):
            serialized.append(
                {
                    "role": message.get("role", ""),
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                    "tool_call_id": message.get("tool_call_id"),
                }
            )
        else:
            serialized.append({"role": "", "content": str(message)})
    return _safe_json(serialized)


def _response_message(response: Any) -> dict[str, Any]:
    choice = None
    try:
        choice = response.choices[0]
    except Exception:
        return {"role": "assistant", "content": None}

    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def _usage_value(response: Any, field: str) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    value = getattr(usage, field, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(field)
    return value


def _tool_call_list(response: Any) -> list[Any]:
    message = _response_message(response)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _step_finish_reason(response: Any) -> str:
    if _tool_call_list(response):
        return "tool_calls"

    finish_reason = None
    try:
        finish_reason = response.choices[0].finish_reason
    except Exception:
        finish_reason = None

    if finish_reason == "stop":
        return "stop"

    if finish_reason == "interrupt":
        return "interrupt"

    if finish_reason == "length":
        return "length"

    content = _response_message(response).get("content")
    if content:
        return "stop"
    return "invalid_response"


def _state(instance: Any) -> dict[str, Any]:
    current = getattr(instance, "_otel_hermes_state", None)
    if current is None:
        current = {
            "entry_span": None,
            "agent_cm": None,
            "agent_span": None,
            "current_step_cm": None,
            "current_step_span": None,
            "current_step_round": 0,
            "pending_step_finish_reason": None,
            "last_response_model": None,
            "last_response_id": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "first_token_ns": None,
        }
        setattr(instance, "_otel_hermes_state", current)
    return current


def _clear_state(instance: Any) -> None:
    setattr(instance, "_otel_hermes_state", None)


def _start_step(tracer, instance: Any) -> None:
    state = _state(instance)
    if state["current_step_cm"] is not None:
        _finish_step(instance, state.get("pending_step_finish_reason") or "invalid_response")

    round_number = state["current_step_round"] + 1
    attrs = {
        _GEN_AI_OPERATION_NAME: _GEN_AI_OP_REACT,
        _GEN_AI_SPAN_KIND: _GEN_AI_KIND_STEP,
        "gen_ai.react.round": round_number,
    }
    cm = tracer.start_as_current_span(
        "react step",
        kind=SpanKind.INTERNAL,
        attributes=attrs,
    )
    span = cm.__enter__()
    state["current_step_cm"] = cm
    state["current_step_span"] = span
    state["current_step_round"] = round_number
    state["pending_step_finish_reason"] = None


def _finish_step(
    instance: Any,
    finish_reason: str,
    *,
    exc: Exception | None = None,
) -> None:
    state = _state(instance)
    cm = state.get("current_step_cm")
    span = state.get("current_step_span")
    if cm is None or span is None:
        return

    span.set_attribute("gen_ai.react.finish_reason", finish_reason)
    if exc is None:
        span.set_status(Status(StatusCode.OK))
        cm.__exit__(None, None, None)
    else:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        cm.__exit__(type(exc), exc, exc.__traceback__)

    state["current_step_cm"] = None
    state["current_step_span"] = None
    state["pending_step_finish_reason"] = None


class _HermesMetrics:
    def __init__(self, meter_provider=None):
        meter = metrics_api.get_meter(
            __name__,
            "0.1.0",
            meter_provider=meter_provider,
        )
        self._calls_count = meter.create_counter(
            name="genai_calls_count",
            description="GenAI call count",
            unit="1",
        )
        self._calls_error_count = meter.create_counter(
            name="genai_calls_error_count",
            description="GenAI call error count",
            unit="1",
        )
        self._calls_duration_seconds = meter.create_histogram(
            name="genai_calls_duration_seconds",
            description="GenAI call duration",
            unit="s",
        )
        self._llm_usage_tokens = meter.create_counter(
            name="genai_llm_usage_tokens",
            description="LLM token usage",
            unit="1",
        )

    @staticmethod
    def _attrs(provider: str, model: str, operation: str = "chat") -> dict[str, Any]:
        return {
            "callType": "gen_ai",
            "callKind": "internal",
            "rpcType": 2100,
            "modelName": model,
            "provider": provider,
            "spanKind": "LLM",
            "rpc": f"{operation} {model}",
        }

    def record_llm_call(self, provider: str, model: str, operation: str = "chat"):
        self._calls_count.add(
            1,
            self._attrs(provider, model, operation),
        )

    def record_llm_error(self, provider: str, model: str, operation: str = "chat"):
        self._calls_error_count.add(
            1,
            self._attrs(provider, model, operation),
        )

    def record_llm_duration(
        self,
        provider: str,
        model: str,
        duration_seconds: float,
        operation: str = "chat",
    ):
        self._calls_duration_seconds.record(
            duration_seconds,
            self._attrs(provider, model, operation),
        )

    def record_llm_tokens(
        self,
        provider: str,
        model: str,
        token_type: str,
        value: int,
        operation: str = "chat",
    ):
        if value <= 0:
            return
        attrs = self._attrs(provider, model, operation)
        attrs["tokenType"] = token_type
        self._llm_usage_tokens.add(
            value,
            attrs,
        )


class HermesAgentInstrumentor(BaseInstrumentor):
    """Instrumentation for Hermes Agent."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")

        tracer = trace_api.get_tracer(
            __name__,
            "0.1.0",
            tracer_provider=tracer_provider,
        )
        metrics = _HermesMetrics(meter_provider=meter_provider)

        wrap_function_wrapper(
            "run_agent",
            "AIAgent.run_conversation",
            _RunConversationWrapper(tracer),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_api_call",
            _LLMCallWrapper(tracer, metrics, streaming=False),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_streaming_api_call",
            _LLMCallWrapper(tracer, metrics, streaming=True),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._invoke_tool",
            _ToolCallWrapper(tracer),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._execute_tool_calls",
            _ToolBatchWrapper(),
        )
        wrap_function_wrapper(
            "model_tools",
            "handle_function_call",
            _ToolDispatchWrapper(tracer),
        )
        wrap_function_wrapper(
            "run_agent",
            "handle_function_call",
            _ToolDispatchWrapper(tracer),
        )
        wrap_function_wrapper(
            "tools.memory_tool",
            "memory_tool",
            _ToolExecutionWrapper(tracer, "memory"),
        )
        wrap_function_wrapper(
            "tools.todo_tool",
            "todo_tool",
            _ToolExecutionWrapper(tracer, "todo"),
        )
        wrap_function_wrapper(
            "tools.session_search_tool",
            "session_search",
            _ToolExecutionWrapper(tracer, "session_search"),
        )
        wrap_function_wrapper(
            "tools.delegate_tool",
            "delegate_task",
            _ToolExecutionWrapper(tracer, "delegate_task"),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import run_agent

        unwrap(run_agent.AIAgent, "run_conversation")
        unwrap(run_agent.AIAgent, "_interruptible_api_call")
        unwrap(run_agent.AIAgent, "_interruptible_streaming_api_call")
        unwrap(run_agent.AIAgent, "_invoke_tool")
        unwrap(run_agent.AIAgent, "_execute_tool_calls")
        import model_tools

        unwrap(model_tools, "handle_function_call")
        unwrap(run_agent, "handle_function_call")
        import tools.delegate_tool
        import tools.memory_tool
        import tools.session_search_tool
        import tools.todo_tool

        unwrap(tools.memory_tool, "memory_tool")
        unwrap(tools.todo_tool, "todo_tool")
        unwrap(tools.session_search_tool, "session_search")
        unwrap(tools.delegate_tool, "delegate_task")


class _RunConversationWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        user_message = args[0] if args else kwargs.get("user_message", "")
        session_id = getattr(instance, "session_id", None)

        agent_attrs = {
            _GEN_AI_OPERATION_NAME: _GEN_AI_OP_INVOKE_AGENT,
            _GEN_AI_PROVIDER_NAME: _HERMES_PROVIDER,
            _GEN_AI_SPAN_KIND: _GEN_AI_KIND_AGENT,
            "gen_ai.agent.name": "Hermes",
            "gen_ai.conversation.id": session_id or "",
            "gen_ai.request.model": getattr(instance, "model", ""),
            "gen_ai.input.messages": _safe_json(
                [{"role": "user", "content": user_message}]
            ),
        }
        tools = getattr(instance, "tools", None)
        if tools:
            agent_attrs["gen_ai.tool.definitions"] = _safe_json(tools)

        state = _state(instance)
        current_span = trace_api.get_current_span()
        current_attrs = getattr(current_span, "attributes", None)
        if current_attrs and current_attrs.get(_GEN_AI_SPAN_KIND) == _GEN_AI_KIND_ENTRY:
            state["entry_span"] = current_span

        agent_cm = self._tracer.start_as_current_span(
            "invoke_agent Hermes",
            kind=SpanKind.INTERNAL,
            attributes=agent_attrs,
        )
        agent_span = agent_cm.__enter__()
        state["agent_cm"] = agent_cm
        state["agent_span"] = agent_span

        try:
            result = wrapped(*args, **kwargs)
            if state["current_step_cm"] is not None:
                _finish_step(
                    instance,
                    state.get("pending_step_finish_reason") or "stop",
                )

            output_messages = _safe_json(
                [
                    {
                        "role": "assistant",
                        "content": result.get("final_response"),
                    }
                ]
            )
            agent_span.set_attribute("gen_ai.output.messages", output_messages)
            if state["entry_span"] is not None:
                state["entry_span"].set_attribute(
                    "gen_ai.output.messages",
                    output_messages,
                )

            if getattr(instance, "session_id", None):
                agent_span.set_attribute("gen_ai.conversation.id", instance.session_id)
            if getattr(instance, "_user_id", None):
                if state["entry_span"] is not None:
                    state["entry_span"].set_attribute(
                        "gen_ai.user.id",
                        instance._user_id,
                    )

            if state["last_response_model"]:
                agent_span.set_attribute(
                    "gen_ai.response.model",
                    state["last_response_model"],
                )
            if state["last_response_id"]:
                agent_span.set_attribute(
                    "gen_ai.response.id",
                    state["last_response_id"],
                )
            if state["input_tokens"] > 0:
                agent_span.set_attribute(
                    "gen_ai.usage.input_tokens",
                    state["input_tokens"],
                )
            if state["output_tokens"] > 0:
                agent_span.set_attribute(
                    "gen_ai.usage.output_tokens",
                    state["output_tokens"],
                )
            if state["total_tokens"] > 0:
                agent_span.set_attribute(
                    "gen_ai.usage.total_tokens",
                    state["total_tokens"],
                )
            if state["first_token_ns"] is not None:
                agent_span.set_attribute(
                    "gen_ai.response.time_to_first_token",
                    state["first_token_ns"],
                )
                if state["entry_span"] is not None:
                    state["entry_span"].set_attribute(
                        "gen_ai.response.time_to_first_token",
                        state["first_token_ns"],
                    )

            agent_span.set_status(Status(StatusCode.OK))
            return result
        except Exception as exc:
            _finish_step(instance, "error", exc=exc)
            agent_span.record_exception(exc)
            agent_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            agent_cm.__exit__(None, None, None)
            _clear_state(instance)


class _LLMCallWrapper:
    def __init__(self, tracer, metrics, streaming: bool):
        self._tracer = tracer
        self._metrics = metrics
        self._streaming = streaming

    def __call__(self, wrapped, instance, args, kwargs):
        _start_step(self._tracer, instance)
        state = _state(instance)
        api_kwargs = args[0] if args else kwargs.get("api_kwargs", {})
        provider = _provider_name(instance)
        model = str(api_kwargs.get("model") or getattr(instance, "model", ""))
        attrs = {
            _GEN_AI_OPERATION_NAME: _GEN_AI_OP_CHAT,
            _GEN_AI_PROVIDER_NAME: provider,
            _GEN_AI_SPAN_KIND: _GEN_AI_KIND_LLM,
            "gen_ai.request.model": model,
        }

        input_messages = _serialize_messages(api_kwargs.get("messages"))
        if input_messages is not None:
            attrs["gen_ai.input.messages"] = input_messages
        if api_kwargs.get("max_tokens") is not None:
            attrs["gen_ai.request.max_tokens"] = api_kwargs["max_tokens"]
        if api_kwargs.get("temperature") is not None:
            attrs["gen_ai.request.temperature"] = api_kwargs["temperature"]

        with self._tracer.start_as_current_span(
            f"chat {model}",
            kind=SpanKind.CLIENT,
            attributes=attrs,
        ) as span:
            started_at = time.perf_counter_ns()

            if self._streaming:
                original_first_delta = kwargs.get("on_first_delta")

                def _wrapped_first_delta():
                    ttft_ns = time.perf_counter_ns() - started_at
                    span.set_attribute(
                        "gen_ai.response.time_to_first_token",
                        ttft_ns,
                    )
                    if state["first_token_ns"] is None:
                        state["first_token_ns"] = ttft_ns
                    if original_first_delta:
                        original_first_delta()

                kwargs["on_first_delta"] = _wrapped_first_delta

            try:
                response = wrapped(*args, **kwargs)
                response_model = getattr(response, "model", None)
                if response_model:
                    span.set_attribute("gen_ai.response.model", response_model)
                response_id = getattr(response, "id", None)
                if response_id:
                    span.set_attribute("gen_ai.response.id", response_id)

                output_message = _response_message(response)
                span.set_attribute(
                    "gen_ai.output.messages",
                    _safe_json([output_message]),
                )

                prompt_tokens = _usage_value(response, "prompt_tokens")
                completion_tokens = _usage_value(response, "completion_tokens")
                total_tokens = _usage_value(response, "total_tokens")
                if prompt_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.input_tokens", prompt_tokens
                    )
                if completion_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.output_tokens", completion_tokens
                    )
                if total_tokens is not None:
                    span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

                finish_reason = None
                try:
                    finish_reason = response.choices[0].finish_reason
                except Exception:
                    pass
                if finish_reason:
                    span.set_attribute(
                        "gen_ai.response.finish_reason", finish_reason
                    )
                    span.set_attribute(
                        "gen_ai.response.finish_reasons",
                        _safe_json([finish_reason]),
                    )

                state["last_response_model"] = response_model or model
                state["last_response_id"] = response_id
                state["input_tokens"] += prompt_tokens or 0
                state["output_tokens"] += completion_tokens or 0
                state["total_tokens"] += total_tokens or 0

                step_reason = _step_finish_reason(response)
                state["pending_step_finish_reason"] = step_reason

                self._metrics.record_llm_call(provider=provider, model=model)
                self._metrics.record_llm_duration(
                    provider=provider,
                    model=model,
                    duration_seconds=(time.perf_counter_ns() - started_at) / 1_000_000_000,
                )
                self._metrics.record_llm_tokens(
                    provider=provider,
                    model=model,
                    token_type="input",
                    value=prompt_tokens or 0,
                )
                self._metrics.record_llm_tokens(
                    provider=provider,
                    model=model,
                    token_type="output",
                    value=completion_tokens or 0,
                )
                self._metrics.record_llm_tokens(
                    provider=provider,
                    model=model,
                    token_type="total",
                    value=total_tokens or 0,
                )
                span.set_status(Status(StatusCode.OK))
                if step_reason != "tool_calls":
                    _finish_step(instance, step_reason)
                return response
            except Exception as exc:
                self._metrics.record_llm_error(provider=provider, model=model)
                self._metrics.record_llm_duration(
                    provider=provider,
                    model=model,
                    duration_seconds=(time.perf_counter_ns() - started_at) / 1_000_000_000,
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                _finish_step(instance, "error", exc=exc)
                raise


class _ToolCallWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        function_name = args[0] if args else kwargs.get("function_name", "")
        function_args = args[1] if len(args) > 1 else kwargs.get("function_args")
        tool_call_id = None
        if len(args) > 3:
            tool_call_id = args[3]
        if tool_call_id is None:
            tool_call_id = kwargs.get("tool_call_id")
        attrs = {
            _GEN_AI_OPERATION_NAME: _GEN_AI_OP_EXECUTE_TOOL,
            _GEN_AI_PROVIDER_NAME: _HERMES_PROVIDER,
            _GEN_AI_SPAN_KIND: _GEN_AI_KIND_TOOL,
            "gen_ai.tool.name": function_name,
        }
        if function_args is not None:
            attrs["gen_ai.tool.call.arguments"] = _safe_json(function_args)
        if tool_call_id:
            attrs["gen_ai.tool.call.id"] = tool_call_id

        with self._tracer.start_as_current_span(
            f"execute_tool {function_name}",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute("gen_ai.tool.call.result", str(result))
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise


class _ToolDispatchWrapper(_ToolCallWrapper):
    def __call__(self, wrapped, instance, args, kwargs):
        function_name = args[0] if args else kwargs.get("function_name", "")
        function_args = args[1] if len(args) > 1 else kwargs.get("function_args")
        return super().__call__(
            wrapped,
            instance,
            (function_name, function_args),
            kwargs,
        )


class _ToolBatchWrapper:
    def __call__(self, wrapped, instance, args, kwargs):
        try:
            result = wrapped(*args, **kwargs)
            _finish_step(instance, "tool_calls")
            return result
        except Exception as exc:
            _finish_step(instance, "error", exc=exc)
            raise


class _ToolExecutionWrapper:
    def __init__(self, tracer, tool_name: str):
        self._tracer = tracer
        self._tool_name = tool_name

    def __call__(self, wrapped, instance, args, kwargs):
        current_span = trace_api.get_current_span()
        current_attrs = getattr(current_span, "attributes", None)
        if (
            current_attrs
            and current_attrs.get(_GEN_AI_SPAN_KIND) == _GEN_AI_KIND_TOOL
            and current_attrs.get("gen_ai.tool.name") == self._tool_name
        ):
            return wrapped(*args, **kwargs)

        attrs = {
            _GEN_AI_OPERATION_NAME: _GEN_AI_OP_EXECUTE_TOOL,
            _GEN_AI_PROVIDER_NAME: _HERMES_PROVIDER,
            _GEN_AI_SPAN_KIND: _GEN_AI_KIND_TOOL,
            "gen_ai.tool.name": self._tool_name,
        }
        if self._tool_name == "memory":
            attrs["gen_ai.tool.call.arguments"] = _safe_json(
                {
                    "action": kwargs.get("action") if "action" in kwargs else (args[0] if len(args) > 0 else None),
                    "target": kwargs.get("target") if "target" in kwargs else (args[1] if len(args) > 1 else None),
                    "content": kwargs.get("content") if "content" in kwargs else (args[2] if len(args) > 2 else None),
                    "old_text": kwargs.get("old_text") if "old_text" in kwargs else (args[3] if len(args) > 3 else None),
                }
            )
        elif self._tool_name == "todo":
            attrs["gen_ai.tool.call.arguments"] = _safe_json(
                {
                    "todos": kwargs.get("todos") if "todos" in kwargs else (args[0] if len(args) > 0 else None),
                    "merge": kwargs.get("merge") if "merge" in kwargs else (args[1] if len(args) > 1 else None),
                }
            )
        elif self._tool_name == "session_search":
            attrs["gen_ai.tool.call.arguments"] = _safe_json(
                {
                    "query": kwargs.get("query") if "query" in kwargs else (args[0] if len(args) > 0 else None),
                    "role_filter": kwargs.get("role_filter") if "role_filter" in kwargs else (args[1] if len(args) > 1 else None),
                    "limit": kwargs.get("limit") if "limit" in kwargs else (args[2] if len(args) > 2 else None),
                }
            )
        elif self._tool_name == "delegate_task":
            attrs["gen_ai.tool.call.arguments"] = _safe_json(
                {
                    "goal": kwargs.get("goal"),
                    "context": kwargs.get("context"),
                    "toolsets": kwargs.get("toolsets"),
                    "tasks": kwargs.get("tasks"),
                    "max_iterations": kwargs.get("max_iterations"),
                }
            )

        with self._tracer.start_as_current_span(
            f"execute_tool {self._tool_name}",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute("gen_ai.tool.call.result", str(result))
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
