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

import asyncio
import logging
import timeit
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import Error

from .config import is_agent_span_enabled, is_llm_span_enabled
from .semantic_conventions import (
    AUTOGEN_PROVIDER_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_SPAN_KIND,
    GEN_AI_SYSTEM,
    GenAIOperation,
    GenAISpanKind,
)
from .utils import (
    apply_create_result,
    make_agent_invocation,
    make_llm_invocation,
)

logger = logging.getLogger(__name__)

_ASSISTANT_AGENT_MODULE = "autogen_agentchat.agents._assistant_agent"
_applied = False

_CALL_LLM_PARAM_NAMES = (
    "model_client",
    "model_client_stream",
    "system_messages",
    "model_context",
    "workbench",
    "handoff_tools",
    "agent_name",
    "cancellation_token",
    "output_content_type",
    "message_id",
)


def _span_attr(span: Any, key: str) -> Any:
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        try:
            return attrs.get(key)
        except Exception:
            pass
    attrs = getattr(span, "attributes", None)
    if attrs is not None:
        try:
            return attrs.get(key)
        except Exception:
            pass
    return None


def _current_autogen_agent_span_active() -> bool:
    span = trace.get_current_span()
    if span is None:
        return False
    operation = _span_attr(span, GEN_AI_OPERATION_NAME)
    provider = _span_attr(span, GEN_AI_PROVIDER_NAME)
    system = _span_attr(span, GEN_AI_SYSTEM)
    span_kind = _span_attr(span, GEN_AI_SPAN_KIND)
    return (
        operation == GenAIOperation.INVOKE_AGENT
        and (provider == AUTOGEN_PROVIDER_NAME or system == AUTOGEN_PROVIDER_NAME)
        and (span_kind in (None, GenAISpanKind.AGENT))
    )


def _arg_value(args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    idx = _CALL_LLM_PARAM_NAMES.index(name)
    # wrapt may include the class object as args[0] for classmethod wrappers on
    # some Python/wrapt combinations. Handle both shapes.
    if args and isinstance(args[0], type):
        idx += 1
    if idx < len(args):
        return args[idx]
    return None


def _error_from_exception(exc: BaseException) -> Error:
    message = str(exc) or type(exc).__name__
    return Error(message=message, type=type(exc))


async def _collect_llm_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    system_messages = _arg_value(args, kwargs, "system_messages") or []
    model_context = _arg_value(args, kwargs, "model_context")
    context_messages: list[Any] = []
    get_messages = getattr(model_context, "get_messages", None)
    if callable(get_messages):
        try:
            context_messages = list(await get_messages())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("AutoGen model_context.get_messages failed: %s", exc)
    return list(system_messages) + context_messages


async def _collect_tools(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    tools: list[Any] = []
    workbench = _arg_value(args, kwargs, "workbench") or []
    for wb in workbench:
        list_tools = getattr(wb, "list_tools", None)
        if callable(list_tools):
            try:
                tools.extend(await list_tools())
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("AutoGen workbench.list_tools failed: %s", exc)
    tools.extend(_arg_value(args, kwargs, "handoff_tools") or [])
    return tools


def _on_messages_stream_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if (
        not is_agent_span_enabled(default=True)
        or _current_autogen_agent_span_active()
    ):
        return wrapped(*args, **kwargs)

    async def _generator():  # type: ignore[no-untyped-def]
        handler = _get_handler()
        invocation = make_agent_invocation(instance)
        handler.start_invoke_agent(invocation)
        try:
            async for item in wrapped(*args, **kwargs):
                yield item
        except (GeneratorExit, asyncio.CancelledError) as exc:
            handler.fail_invoke_agent(invocation, _error_from_exception(exc))
            raise
        except Exception as exc:
            handler.fail_invoke_agent(
                invocation, _error_from_exception(exc)
            )
            raise
        handler.stop_invoke_agent(invocation)

    return _generator()


def _call_llm_wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
    if not is_llm_span_enabled(default=True):
        return wrapped(*args, **kwargs)

    async def _generator():  # type: ignore[no-untyped-def]
        handler = _get_handler()
        model_client = _arg_value(args, kwargs, "model_client")
        agent_name = _arg_value(args, kwargs, "agent_name")
        output_content_type = _arg_value(args, kwargs, "output_content_type")
        output_type = "json" if output_content_type is not None else None
        invocation = make_llm_invocation(
            model_client,
            await _collect_llm_messages(args, kwargs),
            await _collect_tools(args, kwargs),
            agent_name=agent_name,
            output_type=output_type,
        )
        handler.start_llm(invocation)
        try:
            async for item in wrapped(*args, **kwargs):
                if (
                    type(item).__name__ == "ModelClientStreamingChunkEvent"
                    and invocation.monotonic_first_token_s is None
                ):
                    invocation.monotonic_first_token_s = timeit.default_timer()
                elif type(item).__name__ == "CreateResult":
                    apply_create_result(invocation, item)
                yield item
        except (GeneratorExit, asyncio.CancelledError) as exc:
            handler.fail_llm(invocation, _error_from_exception(exc))
            raise
        except Exception as exc:
            handler.fail_llm(invocation, _error_from_exception(exc))
            raise
        handler.stop_llm(invocation)

    return _generator()


def _get_handler() -> ExtendedTelemetryHandler:
    if _get_handler.handler is None:
        _get_handler.handler = ExtendedTelemetryHandler()  # type: ignore[attr-defined]
    return _get_handler.handler


_get_handler.handler = None  # type: ignore[attr-defined]


def apply_agentchat_patch(handler: ExtendedTelemetryHandler) -> None:
    global _applied
    if _applied:
        return
    _get_handler.handler = handler  # type: ignore[attr-defined]
    try:
        wrap_function_wrapper(
            _ASSISTANT_AGENT_MODULE,
            "AssistantAgent.on_messages_stream",
            _on_messages_stream_wrapper,
        )
        wrap_function_wrapper(
            _ASSISTANT_AGENT_MODULE,
            "AssistantAgent._call_llm",
            _call_llm_wrapper,
        )
    except (ImportError, AttributeError) as exc:
        for name in (
            "AssistantAgent.on_messages_stream",
            "AssistantAgent._call_llm",
        ):
            try:
                unwrap(_ASSISTANT_AGENT_MODULE, name)
            except Exception:
                pass
        logger.warning("AutoGen AgentChat patch skipped: %s", exc)
        return
    _applied = True


def revert_agentchat_patch() -> None:
    global _applied
    if not _applied:
        return
    try:
        unwrap(_ASSISTANT_AGENT_MODULE, "AssistantAgent.on_messages_stream")
        unwrap(_ASSISTANT_AGENT_MODULE, "AssistantAgent._call_llm")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("AutoGen AgentChat unwrap failed: %s", exc)
    _applied = False
