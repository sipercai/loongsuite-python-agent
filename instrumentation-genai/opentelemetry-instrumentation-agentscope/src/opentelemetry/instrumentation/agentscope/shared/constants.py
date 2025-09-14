# -*- coding: utf-8 -*-
"""
Constants for OpenTelemetry GenAI Semantic Conventions
"""

from enum import Enum

# Environment Variables
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH"
OTEL_INSTRUMENTATION_GENAI_MESSAGE_STRATEGY = "OTEL_INSTRUMENTATION_GENAI_MESSAGE_STRATEGY"


class GenAiOutputType(str, Enum):
    """GenAI output types"""
    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class GenAiSpanKind(str, Enum):
    """GenAI span kinds"""
    LLM = "llm"
    EMBEDDING = "embedding"
    AGENT = "agent"
    TOOL = "tool"
    FORMATTER = "formatter"


class CommonAttributes:
    """Common GenAI attributes shared across all span types"""
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_SPAN_KIND = "gen_ai.span.kind"
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"


class LLMAttributes:
    """LLM-specific GenAI attributes"""
    # Request attributes
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    
    # Response attributes
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN = "gen_ai.response.time_to_first_token"
    GEN_AI_RESPONSE_TIME_PER_OUTPUT_TOKEN = "gen_ai.response.time_per_output_token"
    
    # Legacy compatibility
    GEN_AI_PROVIDER_NAME = CommonAttributes.GEN_AI_PROVIDER_NAME


class EmbeddingAttributes:
    """Embedding-specific GenAI attributes"""
    # Request attributes
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension_count"
    
    # Response attributes  
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"


class AgentAttributes:
    """Agent-specific GenAI attributes"""
    GEN_AI_AGENT_ID = "gen_ai.agent.id"
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"
    GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
    GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
    
    # AgentScope specific
    AGENTSCOPE_AGENT_ID = "agentscope.agent.id"
    AGENTSCOPE_AGENT_NAME = "agentscope.agent.name"
    AGENTSCOPE_OPERATION_NAME = "agentscope.operation.name"


class ToolAttributes:
    """Tool-specific GenAI attributes"""
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_TYPE = "gen_ai.tool.type"