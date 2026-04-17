"""Instrumentor entrypoint for Hermes telemetry."""

from __future__ import annotations

from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from .constants import INSTRUMENTATION_DEPENDENCIES, INSTRUMENTATION_VERSION
from .metrics import HermesMetrics
from .wrappers import (
    LLMCallWrapper,
    RunConversationWrapper,
    ToolBatchWrapper,
    ToolCallWrapper,
    ToolDispatchWrapper,
    ToolExecutionWrapper,
)


class HermesAgentInstrumentor(BaseInstrumentor):
    """Instrumentation for Hermes Agent."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return INSTRUMENTATION_DEPENDENCIES

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")

        tracer = trace_api.get_tracer(
            __name__,
            INSTRUMENTATION_VERSION,
            tracer_provider=tracer_provider,
        )
        metrics = HermesMetrics(meter_provider=meter_provider)

        wrap_function_wrapper(
            "run_agent",
            "AIAgent.run_conversation",
            RunConversationWrapper(tracer),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_api_call",
            LLMCallWrapper(tracer, metrics, streaming=False),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._interruptible_streaming_api_call",
            LLMCallWrapper(tracer, metrics, streaming=True),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._invoke_tool",
            ToolCallWrapper(tracer),
        )
        wrap_function_wrapper(
            "run_agent",
            "AIAgent._execute_tool_calls",
            ToolBatchWrapper(),
        )
        wrap_function_wrapper(
            "model_tools",
            "handle_function_call",
            ToolDispatchWrapper(tracer),
        )
        wrap_function_wrapper(
            "run_agent",
            "handle_function_call",
            ToolDispatchWrapper(tracer),
        )
        wrap_function_wrapper(
            "tools.memory_tool",
            "memory_tool",
            ToolExecutionWrapper(tracer, "memory"),
        )
        wrap_function_wrapper(
            "tools.todo_tool",
            "todo_tool",
            ToolExecutionWrapper(tracer, "todo"),
        )
        wrap_function_wrapper(
            "tools.session_search_tool",
            "session_search",
            ToolExecutionWrapper(tracer, "session_search"),
        )
        wrap_function_wrapper(
            "tools.delegate_tool",
            "delegate_task",
            ToolExecutionWrapper(tracer, "delegate_task"),
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
