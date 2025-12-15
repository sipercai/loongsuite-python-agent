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

"""
Extended metrics recorder for GenAI invocations.

This module provides LoongSuite GenAI metrics recording following ARMS semantic conventions.
It supports multiple GenAI invocation types: LLM, embedding, execute_tool, invoke_agent,
create_agent, retrieve, and rerank.

All metrics use LoongSuite naming conventions and attributes.
"""

from __future__ import annotations

import logging
import timeit
from numbers import Number
from typing import Optional, Union

from opentelemetry.metrics import Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import Span, set_span_in_context
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    RerankInvocation,
    RetrieveInvocation,
)
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import LLMInvocation

_logger = logging.getLogger(__name__)


class ExtendedInvocationMetricsRecorder(InvocationMetricsRecorder):
    """
    Extended metrics recorder that supports multiple GenAI invocation types.
    
    This class provides LoongSuite GenAI metrics recording following ARMS semantic conventions.
    It supports:
    - LLM/Chat operations
    - Embedding operations
    - Execute tool operations
    - Invoke agent operations
    - Create agent operations
    - Retrieve documents operations
    - Rerank documents operations
    
    All metrics use LoongSuite naming conventions (genai_calls_count, genai_calls_duration_seconds, etc.)
    and attributes (modelName, spanKind, usageType, etc.).
    """

    def __init__(self, meter: Meter):
        """Initialize extended metrics recorder with LoongSuite GenAI metrics."""
        
        # LoongSuite Metrics - Counter
        self._calls_count = meter.create_counter(
            name="genai_calls_count",
            description="Total number of GenAI-related calls",
            unit="1"
        )
        
        self._calls_error_count = meter.create_counter(
            name="genai_calls_error_count",
            description="Total number of GenAI-related call errors",
            unit="1"
        )
        
        self._calls_slow_count = meter.create_counter(
            name="genai_calls_slow_count",
            description="Total number of slow GenAI-related calls",
            unit="1"
        )
        
        self._llm_usage_tokens = meter.create_counter(
            name="genai_llm_usage_tokens",
            description="Token usage statistics",
            unit="1"
        )
        
        # LoongSuite Metrics - Histogram
        self._calls_duration = meter.create_histogram(
            name="genai_calls_duration_seconds",
            description="Response duration of GenAI-related calls",
            unit="s"
        )
        
        self._llm_first_token = meter.create_histogram(
            name="genai_llm_first_token_seconds",
            description="Time to first token for LLM calls",
            unit="s"
        )
        
        # Slow call threshold (3 seconds)
        self._slow_threshold = 3.0
        
        _logger.debug("Initialized ExtendedInvocationMetricsRecorder with LoongSuite metrics")

    def record(
        self,
        span: Optional[Span],
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            ExecuteToolInvocation,
            InvokeAgentInvocation,
            CreateAgentInvocation,
            RetrieveInvocation,
            RerankInvocation,
        ],
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record duration and token metrics for any GenAI invocation type.
        
        This method automatically routes to the appropriate handler based on
        the invocation type. All metrics use LoongSuite naming conventions
        following ARMS semantic conventions.
        
        Args:
            span: The span associated with this invocation
            invocation: The invocation object (any supported type)
            error_type: Optional error type if the invocation failed
        """
        if span is None:
            return

        try:
            # Route to appropriate handler based on invocation type
            if isinstance(invocation, EmbeddingInvocation):
                self._record_embedding(span, invocation, error_type=error_type)
            elif isinstance(invocation, ExecuteToolInvocation):
                self._record_tool(span, invocation, error_type=error_type)
            elif isinstance(invocation, InvokeAgentInvocation):
                self._record_invoke_agent(span, invocation, error_type=error_type)
            elif isinstance(invocation, CreateAgentInvocation):
                self._record_create_agent(span, invocation, error_type=error_type)
            elif isinstance(invocation, RetrieveInvocation):
                self._record_retrieve(span, invocation, error_type=error_type)
            elif isinstance(invocation, RerankInvocation):
                self._record_rerank(span, invocation, error_type=error_type)
            elif isinstance(invocation, LLMInvocation):
                self._record_llm(span, invocation, error_type=error_type)
            else:
                _logger.warning(
                    f"Unknown invocation type: {type(invocation).__name__}"
                )
        except Exception as e:
            _logger.exception(f"Error recording metrics: {e}")

    def _record_llm(
        self,
        span: Span,
        invocation: LLMInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for LLM invocations."""
        try:
            # Build LoongSuite attributes
            attributes = self._build_base_attributes(span, invocation)

            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = None
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)

            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)
            
            # 5. Record token usage (LLM supports tokens)
            if invocation.input_tokens is not None and invocation.input_tokens > 0:
                token_attrs = {**attributes, "usageType": GenAI.GenAiTokenTypeValues.INPUT.value}
                self._llm_usage_tokens.add(invocation.input_tokens, attributes=token_attrs)
            
            if invocation.output_tokens is not None and invocation.output_tokens > 0:
                token_attrs = {**attributes, "usageType": GenAI.GenAiTokenTypeValues.OUTPUT.value}
                self._llm_usage_tokens.add(invocation.output_tokens, attributes=token_attrs)
            
            # 6. Record first token latency (LLM supports first_token)
            if hasattr(invocation, 'first_token_time') and invocation.first_token_time is not None:
                if span_context is None:
                    span_context = set_span_in_context(span)
                self._llm_first_token.record(
                    invocation.first_token_time,
                    attributes=attributes,
                    context=span_context,
                )

            _logger.debug(f"Recorded LLM LoongSuite metrics for {invocation.request_model}")

        except Exception as e:
            _logger.exception(f"Error recording LLM metrics: {e}")

    def _record_embedding(
        self,
        span: Span,
        invocation: EmbeddingInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for embedding invocations."""
        try:
            # Build LoongSuite attributes
            attributes = self._build_base_attributes(span, invocation)
            
            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)
            
            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)
            
            # 5. Record token usage (Embedding supports tokens)
            if invocation.input_tokens is not None and invocation.input_tokens > 0:
                token_attrs = {**attributes, "usageType": GenAI.GenAiTokenTypeValues.INPUT.value}
                self._llm_usage_tokens.add(invocation.input_tokens, attributes=token_attrs)
            
            if invocation.output_tokens is not None and invocation.output_tokens > 0:
                token_attrs = {**attributes, "usageType": GenAI.GenAiTokenTypeValues.OUTPUT.value}
                self._llm_usage_tokens.add(invocation.output_tokens, attributes=token_attrs)

            _logger.debug(
                f"Recorded embedding LoongSuite metrics for model {invocation.request_model}"
            )

        except Exception as e:
            _logger.exception(f"Error recording embedding metrics: {e}")

    def _record_tool(
        self,
        span: Span,
        invocation: ExecuteToolInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for tool execution invocations."""
        try:
            # Build LoongSuite attributes (Tool special: add rpc)
            attributes = self._build_base_attributes(span, invocation)
            attributes["rpc"] = invocation.tool_name  # Tool special: rpc = tool_name

            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)

            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)

            _logger.debug(f"Recorded tool LoongSuite metrics for {invocation.tool_name}")

        except Exception as e:
            _logger.exception(f"Error recording tool metrics: {e}")

    def _record_invoke_agent(
        self,
        span: Span,
        invocation: InvokeAgentInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for agent invocation."""
        try:
            # Build LoongSuite attributes
            attributes = self._build_base_attributes(span, invocation)

            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = None
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)

            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)
            
            # 5. Record token usage (Agent supports tokens)
            if invocation.input_tokens is not None and invocation.input_tokens > 0:
                token_attrs = {**attributes, "usageType": GenAI.GenAiTokenTypeValues.INPUT.value}
                self._llm_usage_tokens.add(invocation.input_tokens, attributes=token_attrs)
            
            if invocation.output_tokens is not None and invocation.output_tokens > 0:
                token_attrs = {**attributes, "usageType": GenAI.GenAiTokenTypeValues.OUTPUT.value}
                self._llm_usage_tokens.add(invocation.output_tokens, attributes=token_attrs)
            
            # 6. Record first token latency (Agent supports first_token)
            if hasattr(invocation, 'first_token_time') and invocation.first_token_time is not None:
                if span_context is None:
                    span_context = set_span_in_context(span)
                self._llm_first_token.record(
                    invocation.first_token_time,
                    attributes=attributes,
                    context=span_context,
                )

            _logger.debug(f"Recorded agent LoongSuite metrics for {invocation.agent_name}")

        except Exception as e:
            _logger.exception(f"Error recording agent metrics: {e}")

    def _record_create_agent(
        self,
        span: Span,
        invocation: CreateAgentInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for agent creation."""
        try:
            # Build LoongSuite attributes
            attributes = self._build_base_attributes(span, invocation)

            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)

            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)

            _logger.debug(f"Recorded create agent LoongSuite metrics for {invocation.agent_name}")

        except Exception as e:
            _logger.exception(f"Error recording create agent metrics: {e}")

    def _record_retrieve(
        self,
        span: Span,
        invocation: RetrieveInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for document retrieval."""
        try:
            # Build LoongSuite attributes
            attributes = self._build_base_attributes(span, invocation)

            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)

            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)

            _logger.debug("Recorded retrieve documents ARMS metrics")

        except Exception as e:
            _logger.exception(f"Error recording retrieve metrics: {e}")

    def _record_rerank(
        self,
        span: Span,
        invocation: RerankInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record LoongSuite metrics for document reranking."""
        try:
            # Build LoongSuite attributes
            attributes = self._build_base_attributes(span, invocation)

            # 1. Record call count
            self._calls_count.add(1, attributes=attributes)
            
            # 2. Record duration
            duration_seconds = self._calculate_duration(invocation)
            if duration_seconds is not None:
                span_context = set_span_in_context(span)
                self._calls_duration.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )
                
                # 3. Check and record slow calls
                if duration_seconds > self._slow_threshold:
                    self._calls_slow_count.add(1, attributes=attributes)

            # 4. Record error count if error occurred
            if error_type:
                self._calls_error_count.add(1, attributes=attributes)

            _logger.debug("Recorded rerank documents LoongSuite metrics")

        except Exception as e:
            _logger.exception(f"Error recording rerank metrics: {e}")

    def _build_base_attributes(
        self,
        span: Span,
        invocation: any,
    ) -> dict:
        """
        Build base LoongSuite attributes for metrics.
        
        Returns:
            Dictionary with spanKind and optionally modelName
        """
        attributes = {}
        
        span_kind = self._get_span_kind_from_span(span)
        if span_kind:
            attributes["spanKind"] = span_kind
        
        model_name = self._get_model_name(invocation)
        if model_name:
            attributes["modelName"] = model_name
        
        return attributes

    def _get_span_kind_from_span(self, span: Span) -> Optional[str]:
        """
        Extract spanKind from span attributes.
        
        Returns:
            spanKind string or None if not found
        """
        if span is None:
            return None
        try:
            if hasattr(span, "attributes") and span.attributes is not None:
                return span.attributes.get(GEN_AI_SPAN_KIND)
        except Exception as e:
            _logger.debug(f"Error extracting span kind from span: {e}")
        return None

    def _get_model_name(self, invocation: any) -> Optional[str]:
        """
        Extract model name from invocation for LoongSuite metrics.
        
        Returns:
            Model name string, or None if not applicable
        """

        if hasattr(invocation, "request_model") and invocation.request_model:
            return invocation.request_model
        # Response model as fallback
        if hasattr(invocation, "response_model_name") and invocation.response_model_name:
            return invocation.response_model_name

        return None

    def _calculate_duration(self, invocation: any) -> Optional[float]:
        """
        Calculate duration from invocation's monotonic_start_s attribute.
        
        Returns duration in seconds, or None if not available.
        """
        # Check if invocation has monotonic_start_s attribute
        if hasattr(invocation, "monotonic_start_s") and invocation.monotonic_start_s is not None:
            duration_seconds = max(
                timeit.default_timer() - invocation.monotonic_start_s, 0.0
            )
            if isinstance(duration_seconds, Number) and duration_seconds >= 0:
                return duration_seconds
        return None



__all__ = ["ExtendedInvocationMetricsRecorder"]

