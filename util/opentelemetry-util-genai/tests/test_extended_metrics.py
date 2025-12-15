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

"""Tests for ExtendedInvocationMetricsRecorder."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest import TestCase
from unittest.mock import patch

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import Error, LLMInvocation


class ExtendedMetricsRecorderTest(TestCase):
    """Test ExtendedInvocationMetricsRecorder functionality."""

    def setUp(self) -> None:
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(
            metric_readers=[self.metric_reader]
        )
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

    def _harvest_metrics(self) -> Dict[str, List[Any]]:
        """Collect metrics from the reader."""
        metrics_data = self.metric_reader.get_metrics_data()
        result: Dict[str, List[Any]] = {}
        
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name not in result:
                        result[metric.name] = []
                    
                    for point in metric.data.data_points:
                        result[metric.name].append(point)
        
        return result

    def test_llm_metrics_with_tokens(self) -> None:
        """Test that LLM invocation records all metrics including tokens."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = LLMInvocation(
            request_model="gpt-4",
            provider="openai"
        )
        invocation.input_tokens = 10
        invocation.output_tokens = 20
        
        # Start and stop with mocked time
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_llm(invocation)
        
        with patch("timeit.default_timer", return_value=1002.5):
            handler.stop_llm(invocation)
        
        # Harvest metrics
        metrics = self._harvest_metrics()
        
        # Check call count
        self.assertIn("genai_calls_count", metrics)
        call_count_points = metrics["genai_calls_count"]
        self.assertEqual(len(call_count_points), 1)
        self.assertEqual(call_count_points[0].value, 1)
        self.assertEqual(call_count_points[0].attributes["modelName"], "gpt-4")
        self.assertEqual(call_count_points[0].attributes["spanKind"], "LLM")
        
        # Check duration
        self.assertIn("genai_calls_duration_seconds", metrics)
        duration_points = metrics["genai_calls_duration_seconds"]
        self.assertEqual(len(duration_points), 1)
        self.assertAlmostEqual(duration_points[0].sum, 2.5, places=1)
        
        # Check token usage - should have 2 points (input and output)
        self.assertIn("genai_llm_usage_tokens", metrics)
        token_points = metrics["genai_llm_usage_tokens"]
        self.assertEqual(len(token_points), 2)
        
        # Find input and output token points
        input_token_point = next(
            p for p in token_points 
            if p.attributes["usageType"] == GenAI.GenAiTokenTypeValues.INPUT.value
        )
        output_token_point = next(
            p for p in token_points 
            if p.attributes["usageType"] == GenAI.GenAiTokenTypeValues.OUTPUT.value
        )
        
        self.assertEqual(input_token_point.value, 10)
        self.assertEqual(output_token_point.value, 20)

    def test_embedding_metrics_with_model_name(self) -> None:
        """Test that Embedding invocation has modelName."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = EmbeddingInvocation(
            request_model="text-embedding-ada-002",
            provider="openai"
        )
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_embedding(invocation)
        
        with patch("timeit.default_timer", return_value=1001.0):
            handler.stop_embedding(invocation)
        
        metrics = self._harvest_metrics()
        
        # Check that modelName is set for embedding
        self.assertIn("genai_calls_count", metrics)
        call_count_points = metrics["genai_calls_count"]
        self.assertEqual(len(call_count_points), 1)
        self.assertEqual(
            call_count_points[0].attributes["modelName"], 
            "text-embedding-ada-002"
        )
        self.assertEqual(call_count_points[0].attributes["spanKind"], "EMBEDDING")

    def test_agent_metrics_without_model_name(self) -> None:
        """Test that Agent invocation does NOT have modelName when not set."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = InvokeAgentInvocation(
            provider="test",
            agent_name="TestAgent"
        )
        # Note: no request_model set
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_invoke_agent(invocation)
        
        with patch("timeit.default_timer", return_value=1001.5):
            handler.stop_invoke_agent(invocation)
        
        metrics = self._harvest_metrics()
        
        # Check that modelName is NOT set for agent without model
        self.assertIn("genai_calls_count", metrics)
        call_count_points = metrics["genai_calls_count"]
        self.assertEqual(len(call_count_points), 1)
        self.assertNotIn("modelName", call_count_points[0].attributes)
        self.assertEqual(call_count_points[0].attributes["spanKind"], "AGENT")

    def test_agent_metrics_with_model_name(self) -> None:
        """Test that Agent invocation CAN have modelName when set."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = InvokeAgentInvocation(
            provider="test",
            agent_name="TestAgent",
            request_model="gpt-4"  # Agent has an associated model
        )
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_invoke_agent(invocation)
        
        with patch("timeit.default_timer", return_value=1001.5):
            handler.stop_invoke_agent(invocation)
        
        metrics = self._harvest_metrics()
        
        # Check that modelName IS set for agent with model
        self.assertIn("genai_calls_count", metrics)
        call_count_points = metrics["genai_calls_count"]
        self.assertEqual(len(call_count_points), 1)
        self.assertEqual(call_count_points[0].attributes["modelName"], "gpt-4")
        self.assertEqual(call_count_points[0].attributes["spanKind"], "AGENT")

    def test_tool_metrics_without_model_name(self) -> None:
        """Test that Tool invocation does NOT have modelName."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = ExecuteToolInvocation(
            tool_name="calculate"
        )
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_execute_tool(invocation)
        
        with patch("timeit.default_timer", return_value=1000.8):
            handler.stop_execute_tool(invocation)
        
        metrics = self._harvest_metrics()
        
        # Check that modelName is NOT set for tool
        self.assertIn("genai_calls_count", metrics)
        call_count_points = metrics["genai_calls_count"]
        self.assertEqual(len(call_count_points), 1)
        self.assertNotIn("modelName", call_count_points[0].attributes)
        self.assertEqual(call_count_points[0].attributes["spanKind"], "TOOL")
        # Tool should have rpc attribute
        self.assertEqual(call_count_points[0].attributes["rpc"], "calculate")

    def test_error_metrics(self) -> None:
        """Test that error metrics are recorded correctly."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = LLMInvocation(
            request_model="gpt-4",
            provider="openai"
        )
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_llm(invocation)
        
        error = Error(message="API Error", type=ValueError)
        with patch("timeit.default_timer", return_value=1001.0):
            handler.fail_llm(invocation, error)
        
        metrics = self._harvest_metrics()
        
        # Check error count
        self.assertIn("genai_calls_error_count", metrics)
        error_count_points = metrics["genai_calls_error_count"]
        self.assertEqual(len(error_count_points), 1)
        self.assertEqual(error_count_points[0].value, 1)

    def test_slow_call_metrics(self) -> None:
        """Test that slow calls are tracked (>3 seconds)."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = LLMInvocation(
            request_model="gpt-4",
            provider="openai"
        )
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_llm(invocation)
        
        # Simulate 4 seconds duration (> 3 second threshold)
        with patch("timeit.default_timer", return_value=1004.0):
            handler.stop_llm(invocation)
        
        metrics = self._harvest_metrics()
        
        # Check slow call count
        self.assertIn("genai_calls_slow_count", metrics)
        slow_count_points = metrics["genai_calls_slow_count"]
        self.assertEqual(len(slow_count_points), 1)
        self.assertEqual(slow_count_points[0].value, 1)

    def test_span_kind_from_span_attributes(self) -> None:
        """Test that spanKind is read from span attributes."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        invocation = LLMInvocation(
            request_model="gpt-4",
            provider="openai"
        )
        
        with patch("timeit.default_timer", return_value=1000.0):
            handler.start_llm(invocation)
        
        with patch("timeit.default_timer", return_value=1001.0):
            handler.stop_llm(invocation)
        
        # Get the span
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        
        # Verify span has GEN_AI_SPAN_KIND attribute
        self.assertIn(GEN_AI_SPAN_KIND, span.attributes)
        self.assertEqual(span.attributes[GEN_AI_SPAN_KIND], "LLM")
        
        # Verify metrics use the same spanKind
        metrics = self._harvest_metrics()
        call_count_points = metrics["genai_calls_count"]
        self.assertEqual(call_count_points[0].attributes["spanKind"], "LLM")

    def test_multiple_invocations(self) -> None:
        """Test metrics aggregation for multiple invocations."""
        handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        
        # Record 3 LLM calls
        for i in range(3):
            invocation = LLMInvocation(
                request_model="gpt-4",
                provider="openai"
            )
            invocation.input_tokens = 10
            invocation.output_tokens = 20
            
            with patch("timeit.default_timer", return_value=1000.0 + i):
                handler.start_llm(invocation)
            
            with patch("timeit.default_timer", return_value=1001.0 + i):
                handler.stop_llm(invocation)
        
        metrics = self._harvest_metrics()
        
        # Check call count is aggregated
        self.assertIn("genai_calls_count", metrics)
        call_count_points = metrics["genai_calls_count"]
        # Should have 1 point with sum of 3
        self.assertEqual(len(call_count_points), 1)
        self.assertEqual(call_count_points[0].value, 3)
        
        # Check token counts are aggregated
        self.assertIn("genai_llm_usage_tokens", metrics)
        token_points = metrics["genai_llm_usage_tokens"]
        # Should have 2 points (input and output)
        self.assertEqual(len(token_points), 2)
        
        input_token_point = next(
            p for p in token_points 
            if p.attributes["usageType"] == GenAI.GenAiTokenTypeValues.INPUT.value
        )
        output_token_point = next(
            p for p in token_points 
            if p.attributes["usageType"] == GenAI.GenAiTokenTypeValues.OUTPUT.value
        )
        
        # 3 calls * 10 tokens = 30
        self.assertEqual(input_token_point.value, 30)
        # 3 calls * 20 tokens = 60
        self.assertEqual(output_token_point.value, 60)

    def test_singleton_handler_has_extended_metrics_recorder(self) -> None:
        """Test that get_extended_telemetry_handler returns handler with ExtendedMetricsRecorder."""
        # Clear singleton if exists
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")
        
        # Get handler using factory function
        handler = get_extended_telemetry_handler(
            tracer_provider=self.tracer_provider,
        )
        
        # Verify it's an ExtendedTelemetryHandler
        self.assertIsInstance(handler, ExtendedTelemetryHandler)
        
        # Verify it has the extended metrics recorder
        from opentelemetry.util.genai.extended_metrics import ExtendedInvocationMetricsRecorder
        self.assertIsInstance(handler._metrics_recorder, ExtendedInvocationMetricsRecorder)
        
        # Verify singleton behavior
        handler2 = get_extended_telemetry_handler()
        self.assertIs(handler, handler2, "Should return the same singleton instance")

