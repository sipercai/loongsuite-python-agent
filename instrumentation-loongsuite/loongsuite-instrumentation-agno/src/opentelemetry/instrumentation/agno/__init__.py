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

from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.agno._wrapper import (
    AgnoAgentWrapper,
    AgnoFunctionCallWrapper,
    AgnoModelWrapper,
)
from opentelemetry.instrumentation.agno.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

"""OpenTelemetry exporters for Agno https://github.com/agno-agi/agno"""

_AGENT = "agno.agent"
_MODULE = "agno.models.base"
_TOOLKIT = "agno.tools.function"
__all__ = ["AgnoInstrumentor"]


class AgnoInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for agno.
    """

    def __init__(self):
        super().__init__()
        self._handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
                get_extended_telemetry_handler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "loongsuite-instrumentation-agno requires "
                "opentelemetry-util-genai with ExtendedTelemetryHandler support"
            ) from exc

        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        self._handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

        agent_wrapper = AgnoAgentWrapper(self._handler)
        function_call_wrapper = AgnoFunctionCallWrapper(self._handler)
        model_wrapper = AgnoModelWrapper(self._handler)

        # Wrap the agent run
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent.run",
            wrapper=agent_wrapper.run,
        )
        wrap_function_wrapper(
            module=_AGENT,
            name="Agent.arun",
            wrapper=agent_wrapper.arun,
        )

        # Wrap the function
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="FunctionCall.execute",
            wrapper=function_call_wrapper.execute,
        )
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="FunctionCall.aexecute",
            wrapper=function_call_wrapper.aexecute,
        )

        # Wrap the model
        wrap_function_wrapper(
            module=_MODULE,
            name="Model.response",
            wrapper=model_wrapper.response,
        )
        wrap_function_wrapper(
            module=_MODULE,
            name="Model.aresponse",
            wrapper=model_wrapper.aresponse,
        )
        wrap_function_wrapper(
            module=_MODULE,
            name="Model.response_stream",
            wrapper=model_wrapper.response_stream,
        )
        wrap_function_wrapper(
            module=_MODULE,
            name="Model.aresponse_stream",
            wrapper=model_wrapper.aresponse_stream,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        # Unwrap the agent call function
        import agno.agent  # noqa: PLC0415

        unwrap(agno.agent.Agent, "run")
        unwrap(agno.agent.Agent, "arun")

        # Unwrap the function call
        import agno.tools.function  # noqa: PLC0415

        unwrap(agno.tools.function.FunctionCall, "execute")
        unwrap(agno.tools.function.FunctionCall, "aexecute")

        # Unwrap the model
        import agno.models.base  # noqa: PLC0415

        unwrap(agno.models.base.Model, "response")
        unwrap(agno.models.base.Model, "aresponse")
        unwrap(agno.models.base.Model, "response_stream")
        unwrap(agno.models.base.Model, "aresponse_stream")
        self._handler = None
