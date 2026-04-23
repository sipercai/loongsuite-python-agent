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

"""Instrumentor entrypoint for Hermes telemetry."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from .metrics import HermesMetrics
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
        model_tools = import_module("model_tools")
        run_agent = import_module("run_agent")
        delegate_tool = import_module("tools.delegate_tool")
        memory_tool = import_module("tools.memory_tool")
        session_search_tool = import_module("tools.session_search_tool")
        todo_tool = import_module("tools.todo_tool")

        unwrap(run_agent.AIAgent, "run_conversation")
        unwrap(run_agent.AIAgent, "_interruptible_api_call")
        unwrap(run_agent.AIAgent, "_interruptible_streaming_api_call")
        unwrap(run_agent.AIAgent, "_invoke_tool")
        unwrap(run_agent.AIAgent, "_execute_tool_calls")
        unwrap(model_tools, "handle_function_call")
        unwrap(run_agent, "handle_function_call")
        unwrap(memory_tool, "memory_tool")
        unwrap(todo_tool, "todo_tool")
        unwrap(session_search_tool, "session_search")
        unwrap(delegate_tool, "delegate_task")
