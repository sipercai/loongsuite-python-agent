import logging
from typing import Any, Collection

from opentelemetry.instrumentation.dify.package import _instruments
from opentelemetry.instrumentation.dify.wrapper import set_wrappers
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

from opentelemetry.instrumentation.dify.config import is_version_supported, MIN_SUPPORTED_VERSION, MAX_SUPPORTED_VERSION

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DifyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for Dify
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not is_version_supported():
            logger.warning(
                f"Dify version is not supported. Current version must be between {MIN_SUPPORTED_VERSION} and {MAX_SUPPORTED_VERSION}."
            )
            return
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, None, tracer_provider=tracer_provider)

        set_wrappers(tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
