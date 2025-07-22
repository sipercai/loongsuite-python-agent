import time
from abc import ABC, abstractmethod
from typing import Any, Tuple, Mapping

from opentelemetry.metrics import get_meter

from aliyun.instrumentation.dify.dify_utils import get_message_data
from aliyun.instrumentation.dify.version import __version__
from aliyun.sdk.extension.arms.semconv.metrics import ArmsCommonServiceMetrics
from aliyun.sdk.extension.arms.utils.capture_content import set_dict_value
from aliyun.semconv.trace import SpanAttributes


class ProcessStrategy(ABC):
    """Base abstract class for all process strategies in Dify instrumentation.

    This class provides the foundation for tracking and monitoring different aspects of Dify's operations.
    It handles metrics collection, tracing, and event management for various Dify components.

    Attributes:
        _handler: The handler instance that manages the overall instrumentation process
        _tracer: OpenTelemetry tracer for creating and managing spans
        _lock: Thread lock for thread-safe operations
        _event_data: Dictionary storing event-related data
        _logger: Logger instance for recording events and errors
        _meter: OpenTelemetry meter for collecting metrics
    """

    def __init__(self, handler: Any):
        self._handler = handler
        self._tracer = handler._tracer
        self._lock = handler._lock
        self._event_data = handler._event_data
        self._logger = handler._logger
        self._meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        self._init_metrics()

    def _init_metrics(self):
        meter = self._meter
        self.calls_count = ArmsCommonServiceMetrics(meter).calls_count
        self.calls_duration_seconds = ArmsCommonServiceMetrics(meter).calls_duration_seconds
        self.llm_context_size = ArmsCommonServiceMetrics(meter).llm_context_size
        self.llm_prompt_size = ArmsCommonServiceMetrics(meter).llm_prompt_size
        self.llm_output_token_seconds = ArmsCommonServiceMetrics(meter).llm_output_token_seconds
        self.llm_usage_tokens = ArmsCommonServiceMetrics(meter).llm_usage_tokens
        self.llm_first_token_seconds = ArmsCommonServiceMetrics(meter).llm_first_token_seconds
        self.calls_error_count = ArmsCommonServiceMetrics(meter).call_error_count

    def _record_metrics(self, event_data, metrics_attributes, error=None):
        self.calls_count.add(1, attributes=metrics_attributes)
        if error is not None:
            self.calls_error_count.add(1, attributes=metrics_attributes)
        duration = (time.time_ns() - event_data.start_time) / 1_000_000_000
        self.calls_duration_seconds.record(duration, attributes=metrics_attributes)

    def _extract_inputs(self, inputs):
        if inputs is None:
            return {}
        input_attributes = {}
        input_value = ""
        input_key = SpanAttributes.INPUT_VALUE
        if inputs is None:
            input_attributes[input_key] = "{}"
            return input_attributes
        if "sys.query" in inputs:
            input_value = inputs["sys.query"]
        else:
            input_value = f"{inputs}"
        if input_value is None:
            return input_attributes
        set_dict_value(input_attributes, input_key, input_value)
        return input_attributes


    @abstractmethod
    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        pass

    def before_process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], ):
        pass

    def _get_data(self, src, key, default=None):
        if key in src:
            return src[key]
        else:
            return default

    def _get_message_data(self, message_id: str):
        return get_message_data(message_id)
