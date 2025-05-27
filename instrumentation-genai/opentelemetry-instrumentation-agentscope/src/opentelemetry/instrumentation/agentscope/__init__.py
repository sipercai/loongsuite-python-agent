from typing import Any, Collection
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.agentscope.package import _instruments
from opentelemetry.instrumentation.agentscope._wrapper import AgentscopeRequestWrapper, AgentscopeToolcallWrapper
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.version import (
    __version__,
)

"""OpenTelemetry exporters for BlackSheep instrumentation"""
_MODULE = "agentscope.models.model"
_TOOLKIT = "agentscope.service.service_toolkit"
__all__ = ["AgentScopeInstrumentor"]

class AgentScopeInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for agentscope.
    """

    __slots__ = (
        "_original_call",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        wrap_function_wrapper(
            module=_MODULE,
            name="ModelWrapperBase.__init__",
            wrapper=AgentscopeRequestWrapper(tracer=tracer),
        )
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="ServiceToolkit._execute_func",
            wrapper=AgentscopeToolcallWrapper(tracer=tracer),
        )
    def _uninstrument(self, **kwargs: Any) -> None:
        wrap_function_wrapper(
            module=_MODULE,
            name="ModelWrapperBase.__init__",
            wrapper=None,
        )
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="ServiceToolkit._execute_func",
            wrapper=None,
        )
