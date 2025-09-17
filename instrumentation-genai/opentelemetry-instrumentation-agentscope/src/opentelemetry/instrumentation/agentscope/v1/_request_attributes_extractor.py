# -*- coding: utf-8 -*-
"""Request attributes extractor for AgentScope instrumentation.

This module provides utilities to extract and normalize message formats
from different AI model providers into a unified format for frontend display.
"""

import json
from typing import Any, Dict, List, Optional, Union, Callable
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiProviderNameValues,
) 
from ..shared.constants import AgentScopeGenAiProviderName

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
    if provide_name == GenAiProviderNameValues.OPENAI.value:
        return _convert_openai_format
    elif provide_name == GenAiProviderNameValues.ANTHROPIC.value:
        return _convert_anthropic_format
    elif provide_name == GenAiProviderNameValues.GCP_GEMINI.value:
        return _convert_gemini_format
    elif provide_name == GenAiProviderNameValues.DEEPSEEK.value:
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


# 导出主要函数
__all__ = ['extract_unified_messages']