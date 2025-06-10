from typing import Any, Collection
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.agno.package import _instruments
from opentelemetry.instrumentation.agno._wrapper import AgnoAgentWrapper, AgnoFunctionCallWrapper
# from opentelemetry.instrumentation.agno._wrapper import AgnoAgentWrapper, AgnoModelWrapper, AgnoToolWrapper
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.version import (
    __version__,
)

"""OpenTelemetry exporters for BlackSheep instrumentation"""
_AGENT = "agno.agent"
_MODULE = "agno.model.base"
_TOOLKIT = "agno.tools.function"
__all__ = ["AgnoInstrumentor"]

class AgnoInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for agno.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        agent_warpper = AgnoAgentWrapper(tracer)
        function_call_wrapper = AgnoFunctionCallWrapper(tracer)

        # Wrap the agent run
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._run",
            wrapper=agent_warpper.run,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._arun",
            wrapper=agent_warpper.arun,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._run_stream",
            wrapper=agent_warpper.run_stream,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._arun_stream",
            wrapper=agent_warpper.arun_stream,
        )

        # Wrap the function
        wrap_function_wrapper (
            module=_TOOLKIT,
            name="FunctionCall.execute",
            wrapper=function_call_wrapper.execute,
        )
        wrap_function_wrapper (
            module=_TOOLKIT,
            name="FunctionCall.aexecute",
            wrapper=function_call_wrapper.aexecute,
        )
    def _uninstrument(self, **kwargs: Any) -> None:

        # Unwrap the agent call function
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._run",
            wrapper=None,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._arun",
            wrapper=None,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._run_stream",
            wrapper=None,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent._arun_stream",
            wrapper=None,
        )

        # Unwrap the function call
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="FunctionCall.execute",
            wrapper=None,
        )
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="FunctionCall.aexecute",
            wrapper=None,
        )
