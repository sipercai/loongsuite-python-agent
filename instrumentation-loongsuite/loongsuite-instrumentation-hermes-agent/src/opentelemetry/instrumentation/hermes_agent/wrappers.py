"""Wrapt wrappers used by Hermes instrumentation."""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.trace import SpanKind, Status, StatusCode

from .constants import (
    GEN_AI_KIND_ENTRY,
    GEN_AI_KIND_LLM,
    GEN_AI_KIND_TOOL,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OP_CHAT,
    GEN_AI_OP_EXECUTE_TOOL,
    GEN_AI_OP_INVOKE_AGENT,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    HERMES_PROVIDER,
)
from .helpers import (
    canonical_usage,
    clear_state,
    provider_name,
    response_finish_reason,
    safe_json,
    serialize_request_messages,
    start_step,
    state,
    step_finish_reason,
    structured_response_message,
)


def finish_step(
    instance: Any,
    finish_reason: str,
    *,
    exc: Exception | None = None,
) -> None:
    current_state = state(instance)
    cm = current_state.get("current_step_cm")
    span = current_state.get("current_step_span")
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

    current_state["current_step_cm"] = None
    current_state["current_step_span"] = None
    current_state["pending_step_finish_reason"] = None


def _is_current_tool_span(tool_name: str) -> bool:
    current_span = trace_api.get_current_span()
    current_attrs = getattr(current_span, "attributes", None)
    return bool(
        current_attrs
        and current_attrs.get(GEN_AI_SPAN_KIND) == GEN_AI_KIND_TOOL
        and current_attrs.get("gen_ai.tool.name") == tool_name
    )


class RunConversationWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        user_message = args[0] if args else kwargs.get("user_message", "")
        session_id = getattr(instance, "session_id", None)

        agent_attrs = {
            GEN_AI_OPERATION_NAME: GEN_AI_OP_INVOKE_AGENT,
            GEN_AI_PROVIDER_NAME: HERMES_PROVIDER,
            GEN_AI_SPAN_KIND: "AGENT",
            "gen_ai.agent.name": "Hermes",
            "gen_ai.conversation.id": session_id or "",
            "gen_ai.request.model": getattr(instance, "model", ""),
            "gen_ai.input.messages": safe_json(
                [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "content": user_message}],
                    }
                ]
            ),
        }
        tools = getattr(instance, "tools", None)
        if tools:
            agent_attrs["gen_ai.tool.definitions"] = safe_json(tools)

        current_state = state(instance)
        current_span = trace_api.get_current_span()
        current_attrs = getattr(current_span, "attributes", None)
        parent_context = trace_api.set_span_in_context(trace_api.INVALID_SPAN)
        if current_attrs:
            current_kind = current_attrs.get(GEN_AI_SPAN_KIND)
            if current_kind == GEN_AI_KIND_ENTRY:
                current_state["entry_span"] = current_span
                parent_context = trace_api.set_span_in_context(current_span)
            elif current_kind == GEN_AI_KIND_TOOL:
                parent_context = trace_api.set_span_in_context(current_span)

        agent_cm = self._tracer.start_as_current_span(
            "invoke_agent Hermes",
            kind=SpanKind.INTERNAL,
            attributes=agent_attrs,
            context=parent_context,
        )
        agent_span = agent_cm.__enter__()
        current_state["agent_cm"] = agent_cm
        current_state["agent_span"] = agent_span

        try:
            result = wrapped(*args, **kwargs)
            if current_state["current_step_cm"] is not None:
                finish_step(
                    instance,
                    current_state.get("pending_step_finish_reason") or "stop",
                )

            output_messages = safe_json(
                [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "text",
                                "content": result.get("final_response"),
                            }
                        ],
                        "finish_reason": "stop",
                    }
                ]
            )
            agent_span.set_attribute("gen_ai.output.messages", output_messages)
            if current_state["entry_span"] is not None:
                current_state["entry_span"].set_attribute(
                    "gen_ai.output.messages",
                    output_messages,
                )

            if getattr(instance, "session_id", None):
                agent_span.set_attribute("gen_ai.conversation.id", instance.session_id)
            if getattr(instance, "_user_id", None) and current_state["entry_span"] is not None:
                current_state["entry_span"].set_attribute(
                    "gen_ai.user.id",
                    instance._user_id,
                )

            if current_state["last_response_model"]:
                agent_span.set_attribute(
                    "gen_ai.response.model",
                    current_state["last_response_model"],
                )
            if current_state["last_response_id"]:
                agent_span.set_attribute(
                    "gen_ai.response.id",
                    current_state["last_response_id"],
                )
            if current_state["input_tokens"] > 0:
                agent_span.set_attribute(
                    "gen_ai.usage.input_tokens",
                    current_state["input_tokens"],
                )
            if current_state["output_tokens"] > 0:
                agent_span.set_attribute(
                    "gen_ai.usage.output_tokens",
                    current_state["output_tokens"],
                )
            if current_state["total_tokens"] > 0:
                agent_span.set_attribute(
                    "gen_ai.usage.total_tokens",
                    current_state["total_tokens"],
                )
            if current_state["first_token_ns"] is not None:
                agent_span.set_attribute(
                    "gen_ai.response.time_to_first_token",
                    current_state["first_token_ns"],
                )
                if current_state["entry_span"] is not None:
                    current_state["entry_span"].set_attribute(
                        "gen_ai.response.time_to_first_token",
                        current_state["first_token_ns"],
                    )

            agent_span.set_status(Status(StatusCode.OK))
            return result
        except Exception as exc:
            finish_step(instance, "error", exc=exc)
            agent_span.record_exception(exc)
            agent_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            agent_cm.__exit__(None, None, None)
            clear_state(instance)


class LLMCallWrapper:
    def __init__(self, tracer, metrics, streaming: bool):
        self._tracer = tracer
        self._metrics = metrics
        self._streaming = streaming

    def __call__(self, wrapped, instance, args, kwargs):
        current_state = state(instance)
        if current_state["active_llm_depth"] > 0:
            return wrapped(*args, **kwargs)

        start_step(self._tracer, instance, finish_step)
        current_state["active_llm_depth"] += 1
        api_kwargs = args[0] if args else kwargs.get("api_kwargs", {})
        provider = provider_name(instance)
        model = str(api_kwargs.get("model") or getattr(instance, "model", ""))
        attrs = {
            GEN_AI_OPERATION_NAME: GEN_AI_OP_CHAT,
            GEN_AI_PROVIDER_NAME: provider,
            GEN_AI_SPAN_KIND: GEN_AI_KIND_LLM,
            "gen_ai.request.model": model,
        }

        input_messages = serialize_request_messages(api_kwargs)
        if input_messages is not None:
            attrs["gen_ai.input.messages"] = input_messages
        if api_kwargs.get("max_tokens") is not None:
            attrs["gen_ai.request.max_tokens"] = api_kwargs["max_tokens"]
        if api_kwargs.get("max_output_tokens") is not None:
            attrs["gen_ai.request.max_tokens"] = api_kwargs["max_output_tokens"]
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
                    span.set_attribute("gen_ai.response.time_to_first_token", ttft_ns)
                    if current_state["first_token_ns"] is None:
                        current_state["first_token_ns"] = ttft_ns
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

                output_message = structured_response_message(instance, response)
                span.set_attribute(
                    "gen_ai.output.messages",
                    safe_json([output_message]),
                )

                input_tokens, output_tokens, total_tokens = canonical_usage(
                    instance, response
                )
                if input_tokens > 0:
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                if output_tokens > 0:
                    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                if total_tokens > 0:
                    span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

                finish_reason = response_finish_reason(instance, response)
                if finish_reason:
                    span.set_attribute("gen_ai.response.finish_reason", finish_reason)
                    span.set_attribute(
                        "gen_ai.response.finish_reasons",
                        safe_json([finish_reason]),
                    )

                current_state["last_response_model"] = response_model or model
                current_state["last_response_id"] = response_id
                current_state["input_tokens"] += input_tokens
                current_state["output_tokens"] += output_tokens
                current_state["total_tokens"] += total_tokens

                normalized_step_reason = step_finish_reason(instance, response)
                current_state["pending_step_finish_reason"] = normalized_step_reason

                self._metrics.record_llm_call(provider=provider, model=model)
                self._metrics.record_llm_duration(
                    provider=provider,
                    model=model,
                    duration_seconds=(
                        time.perf_counter_ns() - started_at
                    )
                    / 1_000_000_000,
                )
                self._metrics.record_llm_tokens(
                    provider=provider,
                    model=model,
                    token_type="input",
                    value=input_tokens,
                )
                self._metrics.record_llm_tokens(
                    provider=provider,
                    model=model,
                    token_type="output",
                    value=output_tokens,
                )
                self._metrics.record_llm_tokens(
                    provider=provider,
                    model=model,
                    token_type="total",
                    value=total_tokens,
                )
                span.set_status(Status(StatusCode.OK))
                if normalized_step_reason != "tool_calls":
                    finish_step(instance, normalized_step_reason)
                return response
            except Exception as exc:
                self._metrics.record_llm_error(provider=provider, model=model)
                self._metrics.record_llm_duration(
                    provider=provider,
                    model=model,
                    duration_seconds=(
                        time.perf_counter_ns() - started_at
                    )
                    / 1_000_000_000,
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                finish_step(instance, "error", exc=exc)
                raise
            finally:
                current_state["active_llm_depth"] = max(
                    0, current_state["active_llm_depth"] - 1
                )


class ToolCallWrapper:
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(self, wrapped, instance, args, kwargs):
        function_name = args[0] if args else kwargs.get("function_name", "")
        function_args = args[1] if len(args) > 1 else kwargs.get("function_args")
        tool_call_id = args[3] if len(args) > 3 else kwargs.get("tool_call_id")
        if _is_current_tool_span(function_name):
            return wrapped(*args, **kwargs)
        attrs = {
            GEN_AI_OPERATION_NAME: GEN_AI_OP_EXECUTE_TOOL,
            GEN_AI_PROVIDER_NAME: HERMES_PROVIDER,
            GEN_AI_SPAN_KIND: GEN_AI_KIND_TOOL,
            "gen_ai.tool.name": function_name,
        }
        if function_args is not None:
            attrs["gen_ai.tool.call.arguments"] = safe_json(function_args)
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


class ToolDispatchWrapper(ToolCallWrapper):
    def __call__(self, wrapped, instance, args, kwargs):
        function_name = args[0] if args else kwargs.get("function_name", "")
        function_args = args[1] if len(args) > 1 else kwargs.get("function_args")
        return super().__call__(
            wrapped,
            instance,
            (function_name, function_args),
            kwargs,
        )


class ToolBatchWrapper:
    def __call__(self, wrapped, instance, args, kwargs):
        try:
            result = wrapped(*args, **kwargs)
            finish_step(instance, "tool_calls")
            return result
        except Exception as exc:
            finish_step(instance, "error", exc=exc)
            raise


class ToolExecutionWrapper:
    def __init__(self, tracer, tool_name: str):
        self._tracer = tracer
        self._tool_name = tool_name

    def __call__(self, wrapped, instance, args, kwargs):
        current_span = trace_api.get_current_span()
        current_attrs = getattr(current_span, "attributes", None)
        if (
            current_attrs
            and current_attrs.get(GEN_AI_SPAN_KIND) == GEN_AI_KIND_TOOL
            and current_attrs.get("gen_ai.tool.name") == self._tool_name
        ):
            return wrapped(*args, **kwargs)

        attrs = {
            GEN_AI_OPERATION_NAME: GEN_AI_OP_EXECUTE_TOOL,
            GEN_AI_PROVIDER_NAME: HERMES_PROVIDER,
            GEN_AI_SPAN_KIND: GEN_AI_KIND_TOOL,
            "gen_ai.tool.name": self._tool_name,
        }
        if self._tool_name == "memory":
            attrs["gen_ai.tool.call.arguments"] = safe_json(
                {
                    "action": kwargs.get("action") if "action" in kwargs else (args[0] if len(args) > 0 else None),
                    "target": kwargs.get("target") if "target" in kwargs else (args[1] if len(args) > 1 else None),
                    "content": kwargs.get("content") if "content" in kwargs else (args[2] if len(args) > 2 else None),
                    "old_text": kwargs.get("old_text") if "old_text" in kwargs else (args[3] if len(args) > 3 else None),
                }
            )
        elif self._tool_name == "todo":
            attrs["gen_ai.tool.call.arguments"] = safe_json(
                {
                    "todos": kwargs.get("todos") if "todos" in kwargs else (args[0] if len(args) > 0 else None),
                    "merge": kwargs.get("merge") if "merge" in kwargs else (args[1] if len(args) > 1 else None),
                }
            )
        elif self._tool_name == "session_search":
            attrs["gen_ai.tool.call.arguments"] = safe_json(
                {
                    "query": kwargs.get("query") if "query" in kwargs else (args[0] if len(args) > 0 else None),
                    "role_filter": kwargs.get("role_filter") if "role_filter" in kwargs else (args[1] if len(args) > 1 else None),
                    "limit": kwargs.get("limit") if "limit" in kwargs else (args[2] if len(args) > 2 else None),
                }
            )
        elif self._tool_name == "delegate_task":
            attrs["gen_ai.tool.call.arguments"] = safe_json(
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
