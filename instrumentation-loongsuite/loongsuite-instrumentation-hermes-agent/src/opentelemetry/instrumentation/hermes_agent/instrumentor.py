"""Instrumentor entrypoint for Hermes telemetry."""

from __future__ import annotations

from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from .metrics import HermesMetrics
from .version import __version__
from .wrappers import (
    LLMCallWrapper,
    RunConversationWrapper,
    ToolBatchWrapper,
    ToolCallWrapper,
    ToolDispatchWrapper,
    ToolExecutionWrapper,
)

_INSTRUMENTATION_DEPENDENCIES = ("openai >= 1.0.0",)


class HermesAgentInstrumentor(BaseInstrumentor):
    """Instrumentation for Hermes Agent."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _INSTRUMENTATION_DEPENDENCIES

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        metrics = HermesMetrics(meter_provider=meter_provider)

        wrap_function_wrapper(
            "run_agent",
            "AIAgent.run_conversation",
            RunConversationWrapper(handler),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_api_call",
            LLMCallWrapper(handler, metrics, streaming=False),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_streaming_api_call",
            LLMCallWrapper(handler, metrics, streaming=True),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._invoke_tool",
            ToolCallWrapper(handler),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._execute_tool_calls",
            ToolBatchWrapper(handler),
        )
        wrap_function_wrapper(
            "model_tools",
            "handle_function_call",
            ToolDispatchWrapper(handler),
        )
        wrap_function_wrapper(
            "run_agent",
            "handle_function_call",
            ToolDispatchWrapper(handler),
        )
        wrap_function_wrapper(
            "tools.memory_tool",
            "memory_tool",
            ToolExecutionWrapper(handler, "memory"),
        )
        wrap_function_wrapper(
            "tools.todo_tool",
            "todo_tool",
            ToolExecutionWrapper(handler, "todo"),
        )
        wrap_function_wrapper(
            "tools.session_search_tool",
            "session_search",
            ToolExecutionWrapper(handler, "session_search"),
        )
        wrap_function_wrapper(
            "tools.delegate_tool",
            "delegate_task",
            ToolExecutionWrapper(handler, "delegate_task"),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import model_tools
        import run_agent
        import tools.delegate_tool
        import tools.memory_tool
        import tools.session_search_tool
        import tools.todo_tool

        unwrap(run_agent.AIAgent, "run_conversation")
        unwrap(run_agent.AIAgent, "_interruptible_api_call")
        unwrap(run_agent.AIAgent, "_interruptible_streaming_api_call")
        unwrap(run_agent.AIAgent, "_invoke_tool")
        unwrap(run_agent.AIAgent, "_execute_tool_calls")
        unwrap(model_tools, "handle_function_call")
        unwrap(run_agent, "handle_function_call")
        unwrap(tools.memory_tool, "memory_tool")
        unwrap(tools.todo_tool, "todo_tool")
        unwrap(tools.session_search_tool, "session_search")
        unwrap(tools.delegate_tool, "delegate_task")
