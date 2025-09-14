# -*- coding: utf-8 -*-
"""
GenAI Attributes Classes for AgentScope Instrumentation

This module provides dataclasses for handling GenAI semantic convention attributes
for different types of operations (LLM, Embedding, Agent, Tool).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .constants import (
    GenAiOperationName,
    GenAiOutputType,
    GenAiSpanKind,
    CommonAttributes,
    LLMAttributes,
    EmbeddingAttributes,
    AgentAttributes,
    ToolAttributes,
)


@dataclass
class LLMRequestAttributes:
    """LLM 请求属性类
    
    用于存储和管理 LLM 请求相关的 GenAI 语义约定属性。
    """
    provider_name: Optional[str] = None
    operation_name: Optional[str] = None
    request_model: Optional[str] = None
    request_choice_count: Optional[int] = None
    request_seed: Optional[int] = None
    request_frequency_penalty: Optional[float] = None
    request_presence_penalty: Optional[float] = None
    request_max_tokens: Optional[int] = None
    request_top_p: Optional[float] = None
    request_top_k: Optional[float] = None
    request_temperature: Optional[float] = None
    request_stop_sequences: Optional[List[str]] = None
    system_instructions: Optional[Any] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        # 设置 span 类型
        attributes[CommonAttributes.GEN_AI_SPAN_KIND] = GenAiSpanKind.LLM.value
        
        if self.provider_name is not None:
            attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] = self.provider_name
        if self.operation_name is not None:
            attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] = self.operation_name
        if self.request_model is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = self.request_model
        if self.request_choice_count is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT] = self.request_choice_count
        if self.request_seed is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] = self.request_seed
        if self.request_frequency_penalty is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] = self.request_frequency_penalty
        if self.request_presence_penalty is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY] = self.request_presence_penalty
        if self.request_max_tokens is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] = self.request_max_tokens
        if self.request_top_p is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] = self.request_top_p
        if self.request_top_k is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] = self.request_top_k
        if self.request_temperature is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] = self.request_temperature
        if self.request_stop_sequences is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] = self.request_stop_sequences
        if self.system_instructions is not None:
            attributes[CommonAttributes.GEN_AI_SYSTEM_INSTRUCTIONS] = self.system_instructions
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性（与span属性相同，但用于事件记录）"""
        return self.get_span_attributes()


@dataclass
class LLMResponseAttributes:
    """LLM 响应属性类
    
    用于存储和管理 LLM 响应相关的 GenAI 语义约定属性。
    """
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    output_type: Optional[Union[GenAiOutputType, str]] = None
    response_finish_reasons: Optional[List[str]] = None
    usage_input_tokens: Optional[int] = None
    usage_output_tokens: Optional[int] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        if self.response_id is not None:
            attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] = self.response_id
        if self.response_model is not None:
            attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = self.response_model
        if self.output_type is not None:
            attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
                self.output_type.value if isinstance(self.output_type, GenAiOutputType)
                else self.output_type
            )
        if self.response_finish_reasons is not None:
            attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = self.response_finish_reasons
        if self.usage_input_tokens is not None:
            attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = self.usage_input_tokens
        if self.usage_output_tokens is not None:
            attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = self.usage_output_tokens

        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性（与span属性相同，但用于事件记录）"""
        return self.get_span_attributes()


@dataclass
class EmbeddingRequestAttributes:
    """Embedding 请求属性类
    
    用于存储和管理 Embedding 请求相关的 GenAI 语义约定属性。
    """
    provider_name: Optional[str] = None
    operation_name: Optional[Union[GenAiOperationName, str]] = None
    request_model: Optional[str] = None
    request_encoding_formats: Optional[List[str]] = None
    request_dimension_count: Optional[int] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        # 设置 span 类型
        attributes[CommonAttributes.GEN_AI_SPAN_KIND] = GenAiSpanKind.EMBEDDING.value
        
        if self.provider_name is not None:
            attributes[CommonAttributes.GEN_AI_PROVIDER_NAME] = self.provider_name
        if self.operation_name is not None:
            attributes[CommonAttributes.GEN_AI_OPERATION_NAME] = (
                self.operation_name.value if isinstance(self.operation_name, GenAiOperationName)
                else self.operation_name
            )
        if self.request_model is not None:
            attributes[EmbeddingAttributes.GEN_AI_REQUEST_MODEL] = self.request_model
        if self.request_encoding_formats is not None:
            attributes[EmbeddingAttributes.GEN_AI_REQUEST_ENCODING_FORMATS] = self.request_encoding_formats
        if self.request_dimension_count is not None:
            attributes[EmbeddingAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = self.request_dimension_count
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性"""
        return self.get_span_attributes()


@dataclass
class EmbeddingResponseAttributes:
    """Embedding 响应属性类
    
    用于存储和管理 Embedding 响应相关的 GenAI 语义约定属性。
    """
    usage_input_tokens: Optional[int] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        if self.usage_input_tokens is not None:
            attributes[EmbeddingAttributes.GEN_AI_USAGE_INPUT_TOKENS] = self.usage_input_tokens
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性"""
        return self.get_span_attributes()


@dataclass
class AgentRequestAttributes:
    """Agent 请求属性类
    
    用于存储和管理 Agent 请求相关的 GenAI 语义约定属性。
    """
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_description: Optional[str] = None
    conversation_id: Optional[str] = None
    operation_name: Optional[Union[GenAiOperationName, str]] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        # 设置 span 类型
        attributes[CommonAttributes.GEN_AI_SPAN_KIND] = GenAiSpanKind.AGENT.value
        
        if self.agent_id is not None:
            attributes[AgentAttributes.GEN_AI_AGENT_ID] = self.agent_id
            attributes[AgentAttributes.AGENTSCOPE_AGENT_ID] = self.agent_id
        if self.agent_name is not None:
            attributes[AgentAttributes.GEN_AI_AGENT_NAME] = self.agent_name
            attributes[AgentAttributes.AGENTSCOPE_AGENT_NAME] = self.agent_name
        if self.agent_description is not None:
            attributes[AgentAttributes.GEN_AI_AGENT_DESCRIPTION] = self.agent_description
        if self.conversation_id is not None:
            attributes[AgentAttributes.GEN_AI_CONVERSATION_ID] = self.conversation_id
        if self.operation_name is not None:
            operation_value = (
                self.operation_name.value if isinstance(self.operation_name, GenAiOperationName)
                else self.operation_name
            )
            attributes[CommonAttributes.GEN_AI_OPERATION_NAME] = operation_value
            attributes[AgentAttributes.AGENTSCOPE_OPERATION_NAME] = operation_value
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性"""
        return self.get_span_attributes()


@dataclass
class AgentResponseAttributes:
    """Agent 响应属性类
    
    用于存储和管理 Agent 响应相关的 GenAI 语义约定属性。
    """
    # 当前 Agent 响应主要通过消息内容体现，可能的扩展属性
    response_type: Optional[str] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        if self.response_type is not None:
            attributes["agentscope.response.type"] = self.response_type
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性"""
        return self.get_span_attributes()


@dataclass
class ToolRequestAttributes:
    """Tool 请求属性类
    
    用于存储和管理 Tool 请求相关的 GenAI 语义约定属性。
    """
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    tool_type: Optional[str] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        # 设置 span 类型
        attributes[CommonAttributes.GEN_AI_SPAN_KIND] = GenAiSpanKind.TOOL.value
        attributes[CommonAttributes.GEN_AI_OPERATION_NAME] = GenAiOperationName.TOOL_CALL.value
        
        if self.tool_call_id is not None:
            attributes[ToolAttributes.GEN_AI_TOOL_CALL_ID] = self.tool_call_id
        if self.tool_name is not None:
            attributes[ToolAttributes.GEN_AI_TOOL_NAME] = self.tool_name
        if self.tool_description is not None:
            attributes[ToolAttributes.GEN_AI_TOOL_DESCRIPTION] = self.tool_description
        if self.tool_type is not None:
            attributes[ToolAttributes.GEN_AI_TOOL_TYPE] = self.tool_type
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性"""
        return self.get_span_attributes()


@dataclass
class ToolResponseAttributes:
    """Tool 响应属性类
    
    用于存储和管理 Tool 响应相关的 GenAI 语义约定属性。
    """
    tool_call_id: Optional[str] = None
    
    def get_span_attributes(self) -> Dict[str, Any]:
        """获取用于 span 的属性"""
        attributes = {}
        
        if self.tool_call_id is not None:
            attributes[ToolAttributes.GEN_AI_TOOL_CALL_ID] = self.tool_call_id
            
        return attributes
    
    def get_event_attributes(self) -> Dict[str, Any]:
        """获取用于事件的属性"""
        return self.get_span_attributes()