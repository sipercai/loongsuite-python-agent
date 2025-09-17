# -*- coding: utf-8 -*-
"""Request attributes extractor for AgentScope instrumentation.

This module provides utilities to extract and normalize message formats
from different AI model providers into a unified format for frontend display.
"""

import json
from typing import Any, Dict, List, Optional, Callable, Tuple
import inspect

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agentscope.message import Msg
from agentscope.model import (
    ChatModelBase, 
    OpenAIChatModel,
    GeminiChatModel,
    OllamaChatModel,
    AnthropicChatModel,
    DashScopeChatModel,
)
from agentscope.embedding import EmbeddingModelBase
from agentscope.agent import AgentBase

from ._response_attributes_extractor import _get_chatmodel_output_messages
from ..shared import (
    AgentScopeGenAiProviderName,
    LLMRequestAttributes,
    EmbeddingRequestAttributes,
    AgentRequestAttributes,
)

from .utils import (
    _format_msg_to_parts,
    _serialize_to_str,
)

import logging
logger = logging.getLogger(__name__)


class RequestAttributesExtractor:
    """请求参数提取器类，支持有状态的环境变量管理和参数解析。
    
    该类负责从不同类型的函数调用参数中提取输入消息，并转换为统一格式。
    支持LLM、Embedding和Agent三种类型的参数解析。
    """
    
    def __init__(self):
        """初始化提取器，获取遥测配置选项。"""
        # self.telemetry_options = get_telemetry_options()
        
    def extract_llm_input_messages(
        self,
        call_instance: ChatModelBase ,
        call_args: Tuple[Any, ...],
        call_kwargs: Dict[str, Any]
    ) -> LLMRequestAttributes:
        """提取LLM调用的输入消息。
        
        Args:
            call_instance: llm 模型调用实例
            call_args: 位置参数
            call_kwargs: 关键字参数
            
        Returns:
            转换后的统一格式消息列表
        """
        # 创建LLM请求属性
        request_attrs = LLMRequestAttributes(
            operation_name = GenAIAttributes.GenAiOperationNameValues.CHAT.value, 
            provider_name = self._parse_provider_name(call_instance),
            request_model = getattr(call_instance, "model_name", "unknown_model"),
            request_max_tokens = call_kwargs.get("max_tokens"),
            request_temperature = call_kwargs.get("temperature"),
            request_top_p = call_kwargs.get("top_p"),
            request_top_k = call_kwargs.get("top_k"),
            request_stop_sequences = call_kwargs.get("stop_sequences"),
            request_tool_definitions = self._get_tool_definitions(tools=call_kwargs.get("tools"), tool_choice=call_kwargs.get("tool_choice"), structured_model=call_kwargs.get("structured_model")),
        )
        
        # input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model

        messages = None
        if call_args and len(call_args) > 0:
            messages = call_args[0]
        elif "messages" in call_kwargs:
            messages = call_kwargs["messages"]
        if messages:
            input_messages = get_message_converter(request_attrs.provider_name)(messages)
        else:
            logger.warning(" ChatModelWrapper No messages provided. Skipping input message conversion.")
            input_messages = {
                "args": call_args,
                "kwargs": call_kwargs,
            }

        request_attrs.input_messages = _serialize_to_str(input_messages)
        
        return request_attrs
    
    def extract_embedding_input_messages(
        self,
        call_instance: EmbeddingModelBase,
        call_args: Tuple[Any, ...],
        call_kwargs: Dict[str, Any]
    ) -> EmbeddingRequestAttributes:
        """提取Embedding调用的输入消息。
        
        Args:
            call_args: 位置参数
            call_kwargs: 关键字参数
            
        Returns:
            转换后的统一格式消息列表
        """
        # 创建embedding请求属性
        request_attrs = EmbeddingRequestAttributes(
            operation_name = GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value, 
            request_model = getattr(call_instance, "model_name", "unknown_model"),
            request_encoding_formats = call_kwargs.get("encoding_formats"),
        )

        # input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model

        text_for_embedding = None
        if call_args and len(call_args) > 0:
            text_for_embedding = call_args[0]
        elif "text" in call_kwargs:
            text_for_embedding = call_kwargs["text"]
        if text_for_embedding:
            input_messages = self._get_embedding_message(text_for_embedding)
        else:
            logger.warning(" EmbeddingModelWrapper No text provided. Skipping input message conversion.")
            input_messages = {
                "args": call_args,
                "kwargs": call_kwargs,
            }

        request_attrs.input_messages = _serialize_to_str(input_messages)

        return request_attrs
    
    def extract_agent_input_messages(
        self,
        reply_instance: AgentBase,
        reply_args: Tuple[Any, ...],
        reply_kwargs: Dict[str, Any]
    ) -> AgentRequestAttributes:
        """提取Agent reply调用的输入消息。
        
        Args:
            reply_args: 位置参数
            reply_kwargs: 关键字参数
            
        Returns:
            转换后的统一格式消息列表
        """
        # 创建agent reply请求属性
        request_attrs = AgentRequestAttributes(
            operation_name = GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value, 
            agent_id = getattr(reply_instance, "id", "unknown"),
            agent_name = getattr(reply_instance, "name", "unknown_agent"),
            agent_description = inspect.getdoc(reply_instance.__class__) or "No description available",
            system_instructions = reply_instance.sys_prompt if hasattr(reply_instance, "sys_prompt") else None,
        )
        if hasattr(reply_instance, "model") and reply_instance.model:
            request_attrs.request_model = getattr(reply_instance.model, "model_name", "unknown_model")

        # input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model
        
        msg = None
        if reply_args and len(reply_args) > 0:
            msg = reply_args[0]
        elif "msg" in reply_kwargs:
            msg = reply_kwargs["msg"]
        if msg:
            input_messages = self._get_agent_message(msg)
        else:
            logger.warning(" AgentWrapper No msg provided. Skipping input message conversion.")
            input_messages = {
                "args": reply_args,
                "kwargs": reply_kwargs,
            }
        request_attrs.input_messages = _serialize_to_str(input_messages)

        return request_attrs

    def _get_tool_definitions(self, tools: Optional[list[dict]], tool_choice: Optional[str], structured_model: Optional[bool] = False) -> Optional[str]:
        if structured_model is True or tools is None or tool_choice is None or tool_choice == "none":
            return None
        else:
            return _serialize_to_str(tools)

    def _parse_provider_name(self, chat_model: ChatModelBase) -> str:
        if isinstance(chat_model, OpenAIChatModel):
            return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
        elif isinstance(chat_model, GeminiChatModel):
            return GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
        elif isinstance(chat_model, AnthropicChatModel):
            return GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
        elif isinstance(chat_model, DashScopeChatModel):
            if hasattr(chat_model, "base_http_api_url") and chat_model.base_http_api_url:
                base_url = chat_model.base_http_api_url
                if "openai.com" in base_url:
                    return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
                elif "api.deepseek.com" in base_url:
                    return GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
                elif "dashscope.aliyuncs.com" in base_url:
                    return AgentScopeGenAiProviderName.DASHSCOPE.value
            return AgentScopeGenAiProviderName.DASHSCOPE.value
        elif isinstance(chat_model, OllamaChatModel):
            return AgentScopeGenAiProviderName.OLLAMA.value
        else:
            return "unknown"
    
    def _get_embedding_message(self, text: list[str]) -> list[dict[str, Any]]:
        input_message = []
        for text_item in text:
            input_message.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_item
                    }
                ]
            })
        return input_message

    def _get_agent_message(self, msg: Msg | list[Msg]) -> list[dict[str, Any]]:
        try:
            if isinstance(msg, Msg):
                return [_format_msg_to_parts(msg)]
            elif isinstance(msg, list):
                return [_format_msg_to_parts(msg_item) for msg_item in msg]
        except Exception as e:
            logger.warning(f"Error formatting messages: {e}")
            return msg

def get_message_converter(
    provide_name: Optional[str],
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    根据模型类型返回相应的消息格式转换函数。
    
    Args:
        provide_name: AI模型提供商s
    
    Returns:
        对应的消息转换函数，函数签名为:
        (messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]
        
        转换后的统一格式为:
        [
            {
                "role": "user|assistant|tool|system",
                "parts": [
                    {
                        "type": "text|tool_call|tool_call_response|image|audio",
                        "content": "...",  # for text
                        "id": "...",       # for tool_call
                        "name": "...",     # for tool_call
                        "arguments": {...}, # for tool_call
                        "result": "..."    # for tool_call_response
                    }
                ]
            }
        ]
    """
    # 根据 provide_name 选择对应的转换函数
    if provide_name == GenAIAttributes.GenAiProviderNameValues.OPENAI.value:
        return _convert_openai_format
    elif provide_name == GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value:
        return _convert_anthropic_format
    elif provide_name == GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value:
        return _convert_gemini_format
    elif provide_name == GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value:
        return _convert_deepseek_format
    elif provide_name == AgentScopeGenAiProviderName.DASHSCOPE.value:
        return _convert_dashscope_format
    elif provide_name == AgentScopeGenAiProviderName.OLLAMA.value:
        return _convert_ollama_format
    else:
        raise _convert_default_format



def _convert_openai_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换OpenAI格式消息。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append({
                'type': 'tool_call_response',
                'id': msg.get('tool_call_id', ''),
                'result': msg.get('content', '')
            })
        # 处理普通消息
        else:
            content = msg.get('content')
            
            # 如果有tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = function_info.get('arguments', '{}')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    
                    parts.append({
                        'type': 'tool_call',
                        'id': tool_call.get('id', 'unknown_id'),
                        'name': function_info.get('name', 'unknown_tool'),
                        'arguments': arguments
                    })
            
            # 处理content
            if isinstance(content, str):
                parts.append({
                    'type': 'text',
                    'content': content
                })
            elif isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        content_type = content_item.get('type', 'text')
                        if content_type == 'text':
                            parts.append({
                                'type': 'text',
                                'content': content_item.get('text', '')
                            })
                        elif content_type == 'image_url':
                            if (content_item.get('image_url') is not None
                                and isinstance(content_item.get('image_url'), dict)):
                                image_url = content_item.get('image_url', {}).get('url', '')
                                parts.append({
                                    'type': 'image',
                                    'content': image_url
                                })
                        elif content_type == 'input_audio':
                            audio_data = content_item.get('input_audio', {}).get('data', '')
                            parts.append({
                                'type': 'audio', 
                                'content': audio_data
                            })
                        else:
                            # 支持其他多模态内容类型
                            parts.append({
                                'type': content_type,
                                'content': str(content_item),
                                'hint': 'unsupported content type'
                            })
                    elif isinstance(content_item, str):
                        parts.append({
                            'type': 'text',
                            'content': content_item
                        })
                
                # 如果没有处理任何内容，添加空文本部分
                if not parts and not msg.get('tool_calls'):
                    parts.append({
                        'type': 'text',
                        'content': ''
                    })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages


def _convert_anthropic_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换Anthropic格式消息。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        parts = []
        
        content = msg.get('content', [])
        if isinstance(content, str):
            parts.append({
                'type': 'text',
                'content': content
            })
        elif isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict):
                    content_type = content_item.get('type', 'text')
                    if content_type == 'text':
                        parts.append({
                            'type': 'text',
                            'content': content_item.get('text', '')
                        })
                    elif content_type == 'tool_use':
                        parts.append({
                            'type': 'tool_call',
                            'id': content_item.get('id', ''),
                            'name': content_item.get('name', ''),
                            'arguments': content_item.get('input', {})
                        })
                    elif content_type == 'tool_result':
                        content_blocks = content_item.get('content', [])
                        result_text = ""
                        if isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    result_text += block.get('text', '')
                        else:
                            result_text = str(content_blocks)
                        
                        parts.append({
                            'type': 'tool_call_response',
                            'id': content_item.get('tool_use_id', ''),
                            'result': result_text
                        })
                    elif content_type == 'image':
                        source = content_item.get('source', {})
                        if source.get('type') == 'url':
                            parts.append({
                                'type': 'image',
                                'content': source.get('url', '')
                            })
                        elif source.get('type') == 'base64':
                            parts.append({
                                'type': 'image',
                                'content': f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                            })
                    else:
                        # 支持其他多模态内容类型
                        parts.append({
                            'type': content_type,
                            'content': str(content_item),
                            'hint': 'unsupported content type'
                        })
                elif isinstance(content_item, str):
                    parts.append({
                        'type': 'text',
                        'content': content_item
                    })
            
            # 如果没有处理任何内容，添加空文本部分
            if not parts:
                parts.append({
                    'type': 'text',
                    'content': ''
                })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages


def _convert_gemini_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换Gemini格式消息。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        # 将Gemini的'model'角色转换为'assistant'
        if role == 'model':
            role = 'assistant'
        
        parts = []
        
        # 处理parts列表
        parts_list = msg.get('parts', [])
        if isinstance(parts_list, list):
            for part_item in parts_list:
                if isinstance(part_item, dict):
                    # 处理文本
                    if 'text' in part_item:
                        text_content = part_item.get('text', '')
                        if text_content:  # 只添加非空文本
                            parts.append({
                                'type': 'text',
                                'content': text_content
                            })
                    # 处理function_call
                    elif 'function_call' in part_item:
                        func_call = part_item['function_call']
                        parts.append({
                            'type': 'tool_call',
                            'id': func_call.get('id', ''),
                            'name': func_call.get('name', ''),
                            'arguments': func_call.get('args', {})
                        })
                    # 处理function_response
                    elif 'function_response' in part_item:
                        func_resp = part_item['function_response']
                        response_content = func_resp.get('response', {})
                        result_text = response_content.get('output', '') if isinstance(response_content, dict) else str(response_content)
                        
                        parts.append({
                            'type': 'tool_call_response', 
                            'id': func_resp.get('id', ''),
                            'result': result_text
                        })
                    # 处理inline_data (图片/音频)
                    elif 'inline_data' in part_item:
                        inline_data = part_item['inline_data']
                        mime_type = inline_data.get('mime_type', '')
                        data = inline_data.get('data', '')
                        
                        if mime_type.startswith('image/'):
                            parts.append({
                                'type': 'image',
                                'content': f"data:{mime_type};base64,{data}"
                            })
                        elif mime_type.startswith('audio/'):
                            parts.append({
                                'type': 'audio',
                                'content': data
                            })
                        else:
                            # 其他类型的inline_data
                            parts.append({
                                'type': 'unknown',
                                'content': str(part_item),
                                'hint': 'unsupported inline_data type'
                            })
                    else:
                        # 其他未知类型
                        parts.append({
                            'type': 'unknown',
                            'content': str(part_item),
                            'hint': 'unsupported part type'
                        })
                elif isinstance(part_item, str):
                    # 直接字符串的part
                    parts.append({
                        'type': 'text',
                        'content': part_item
                    })
        
        # 如果没有处理任何内容，添加空文本部分
        if not parts:
            parts.append({
                'type': 'text',
                'content': ''
            })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages


def _convert_dashscope_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换DashScope格式消息。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append({
                'type': 'tool_call_response',
                'id': msg.get('tool_call_id', ''),
                'result': msg.get('content', '')
            })
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = function_info.get('arguments', '{}')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    
                    parts.append({
                        'type': 'tool_call',
                        'id': tool_call.get('id', ''),
                        'name': function_info.get('name', ''),
                        'arguments': arguments
                    })
            
            # 处理content
            content = msg.get('content')
            if isinstance(content, str):
                if content.strip():
                    parts.append({
                        'type': 'text',
                        'content': content
                    })
            elif isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        # DashScope特有的格式：{'text': '...'}, {'image': '...'}, {'audio': '...'}
                        if 'text' in content_item and content_item['text']:
                            parts.append({
                                'type': 'text',
                                'content': content_item['text']
                            })
                        elif 'image' in content_item:
                            parts.append({
                                'type': 'image',
                                'content': content_item['image']
                            })
                        elif 'audio' in content_item:
                            parts.append({
                                'type': 'audio',
                                'content': content_item['audio']
                            })
                        else:
                            # 其他未知类型
                            parts.append({
                                'type': 'unknown',
                                'content': str(content_item),
                                'hint': 'unsupported content type'
                            })
                    elif isinstance(content_item, str):
                        parts.append({
                            'type': 'text',
                            'content': content_item
                        })
                
                # 如果没有处理任何内容且没有tool_calls，添加空文本部分
                if not parts and not msg.get('tool_calls'):
                    parts.append({
                        'type': 'text',
                        'content': ''
                    })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages


def _convert_ollama_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换Ollama格式消息。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append({
                'type': 'tool_call_response',
                'id': msg.get('tool_call_id', ''),
                'result': msg.get('content', '')
            })
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = function_info.get('arguments', {})
                    
                    parts.append({
                        'type': 'tool_call',
                        'id': tool_call.get('id', ''),
                        'name': function_info.get('name', ''),
                        'arguments': arguments
                    })
            
            # 处理文本内容
            content = msg.get('content')
            if isinstance(content, str) and content.strip():
                parts.append({
                    'type': 'text',
                    'content': content
                })
            elif content is None and not msg.get('tool_calls'):
                # 处理content为None的情况（但没有tool_calls）
                parts.append({
                    'type': 'text',
                    'content': ''
                })
            
            # 处理图片（Ollama特有的images字段）
            if 'images' in msg and msg['images']:
                for image_data in msg['images']:
                    if isinstance(image_data, str):
                        parts.append({
                            'type': 'image',
                            'content': f"data:image/png;base64,{image_data}"
                        })
                    else:
                        parts.append({
                            'type': 'image',
                            'content': str(image_data)
                        })
        
        # 如果没有处理任何内容且没有tool_calls，添加空文本部分
        if not parts and not msg.get('tool_calls'):
            parts.append({
                'type': 'text',
                'content': ''
            })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages


def _convert_deepseek_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换DeepSeek格式消息。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append({
                'type': 'tool_call_response',
                'id': msg.get('tool_call_id', ''),
                'result': msg.get('content', '')
            })
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = function_info.get('arguments', '{}')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    
                    parts.append({
                        'type': 'tool_call',
                        'id': tool_call.get('id', ''),
                        'name': function_info.get('name', ''),
                        'arguments': arguments
                    })
            
            # 处理content
            content = msg.get('content')
            if isinstance(content, str):
                if content and content.strip():
                    parts.append({
                        'type': 'text',
                        'content': content
                    })
            elif content is None and msg.get('tool_calls'):
                # DeepSeek格式中，当有tool_calls时content可能为None
                pass  # 不需要添加空文本
            elif isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        content_type = content_item.get('type', 'text')
                        if content_type == 'text' and content_item.get('text'):
                            parts.append({
                                'type': 'text',
                                'content': content_item['text']
                            })
                        else:
                            # 其他未知类型
                            parts.append({
                                'type': content_type,
                                'content': str(content_item),
                                'hint': 'unsupported content type'
                            })
                    elif isinstance(content_item, str):
                        parts.append({
                            'type': 'text',
                            'content': content_item
                        })
            
            # 如果没有处理任何内容且没有tool_calls，添加空文本部分
            if not parts and not msg.get('tool_calls'):
                parts.append({
                    'type': 'text',
                    'content': ''
                })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages


def _convert_default_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """默认格式转换器，处理无法识别的格式。"""
    unified_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        parts = []
        
        # 尝试提取基本的文本内容
        content = msg.get('content', '')
        if isinstance(content, str) and content.strip():
            parts.append({
                'type': 'text',
                'content': content
            })
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # 尝试提取文本
                    text_content = item.get('text') or item.get('content') or str(item)
                    if text_content:
                        parts.append({
                            'type': 'text',
                            'content': text_content
                        })
                else:
                    parts.append({
                        'type': 'text',
                        'content': str(item)
                    })
        
        # 如果没有找到内容，用整个消息作为文本
        if not parts and msg:
            # 排除role字段，将其他内容转为文本
            filtered_msg = {k: v for k, v in msg.items() if k != 'role'}
            if filtered_msg:
                parts.append({
                    'type': 'text',
                    'content': str(filtered_msg)
                })
        
        if parts:
            unified_messages.append({
                'role': role,
                'parts': parts
            })
    
    return unified_messages
