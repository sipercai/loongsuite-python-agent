# -*- coding: utf-8 -*-
"""Patching functions for AgentScope v1.x instrumentation."""

import asyncio
from logging import getLogger
from timeit import default_timer
from typing import Any, Optional, AsyncGenerator, Generator

from opentelemetry._events import Event, EventLogger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer, StatusCode
from opentelemetry.trace.status import Status
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from agentscope.message import ToolUseBlock
from agentscope.tool import Toolkit, ToolResponse
from agentscope.formatter import FormatterBase

from .utils import (
    _serialize_to_str,
    _trace_async_generator_wrapper
)

logger = getLogger(__name__)

def toolkit_call_tool_function(
    tracer: Tracer,
    event_logger: EventLogger,
):
    """Wrap the Toolkit.call_tool_function method to trace it."""

    async def traced_method(wrapped, instance, args, kwargs) -> ToolResponse:
        tool_call = args[0] if args else kwargs.get("tool_call")

        # 使用 AgentScope 兼容的属性结构
        span_attributes = {
            "span.kind": "TOOL",
            "project.run_id": getattr(instance, "run_id", "unknown"),
            "input": _serialize_to_str({
                "tool_call": tool_call,
            }),
            "metadata": _serialize_to_str(
                {**tool_call}
            ),
        }
        
        with tracer.start_as_current_span(
            name=f"{wrapped.__name__}",
            attributes=span_attributes,
            end_on_exit=False,
        ) as span:
            try:
                res = await wrapped(*args, **kwargs)

                return _trace_async_generator_wrapper(res, span)
                
            except Exception as error:
                span.set_status(
                    StatusCode.ERROR,
                    str(error),
                )
                span.record_exception(error)
                span.end()
                raise error from None

    return traced_method


def formatter_format(
    tracer: Tracer,
    event_logger: EventLogger,
):
    """Wrap the FormatterBase.format method to trace it."""

    async def traced_method(wrapped, instance, args, kwargs) -> list[dict]:
        if not isinstance(instance, FormatterBase):
            logger.warning(
                "Skipping tracing for %s as the first argument"
                "is not an instance of FormatterBase, but %s",
                wrapped.__name__,
                type(instance),
            )
            return await wrapped(*args, **kwargs)
        # 使用 AgentScope 兼容的属性结构
        span_attributes = {
            "span.kind": "FORMATTER",
            "project.run_id": getattr(instance, "run_id", "unknown"),
            "input": _serialize_to_str({
                "args": args,
                "kwargs": kwargs,
            }),
            "metadata": _serialize_to_str({}),
        }

        with tracer.start_as_current_span(
            name=f"{instance.__class__.__name__}.{wrapped.__name__}",
            attributes=span_attributes,
            end_on_exit=False,
        ) as span:
            try:
                # Call the formatter function
                res = await wrapped(*args, **kwargs)

                # Set the output attribute
                span.set_attributes(
                    {"output": _serialize_to_str(res)},
                )
                span.set_status(StatusCode.OK)
                span.end()
                return res

            except Exception as e:
                span.set_status(
                    StatusCode.ERROR,
                    str(e),
                )
                span.record_exception(e)
                span.end()
                raise e from None

    return traced_method