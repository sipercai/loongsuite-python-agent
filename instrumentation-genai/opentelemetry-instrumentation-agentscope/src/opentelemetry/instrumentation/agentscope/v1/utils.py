# -*- coding: utf-8 -*-
"""Attributes processor for span attributes."""
import datetime
import enum
import inspect
import json
from dataclasses import is_dataclass
from math import e
from typing import Any, AsyncGenerator, Optional
from agentscope.tool import ToolResponse, Toolkit
import aioitertools
from pydantic import BaseModel
from opentelemetry.trace import Span, StatusCode
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

from typing import TypeVar

from ..shared import (
    GenAITelemetryOptions, 
    CommonAttributes, 
    GenAiSpanKind, 
    AgentScopeGenAiProviderName,
)

T = TypeVar("T")

import logging
logger = logging.getLogger(__name__)


def _to_serializable(
    obj: Any,
) -> Any:
    """Convert an object to a JSON serializable type.

    Args:
        obj (`Any`):
            The object to be converted to JSON serializable.

    Returns:
        `Any`:
            The converted JSON serializable object
    """

    # Handle primitive types first
    if isinstance(obj, (str, int, bool, float, type(None))):
        res = obj

    elif isinstance(obj, (list, tuple, set, frozenset)):
        res = [_to_serializable(x) for x in obj]

    elif isinstance(obj, dict):
        res = {str(key): _to_serializable(val) for (key, val) in obj.items()}

    elif isinstance(obj, (Msg, BaseModel)) or is_dataclass(obj):
        res = repr(obj)

    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        res = repr(obj)

    elif isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        res = obj.isoformat()

    elif isinstance(obj, datetime.timedelta):
        res = obj.total_seconds()

    elif isinstance(obj, enum.Enum):
        res = _to_serializable(obj.value)

    else:
        res = str(obj)

    return res


def _serialize_to_str(value: Any) -> str:
    """Get input attributes

    Args:
        value (`Any`):
            The input value

    Returns:
        `str`:
            JSON serialized string of the input value
    """
    try:
        return json.dumps(value, ensure_ascii=False)

    except TypeError:
        return json.dumps(
            _to_serializable(value),
            ensure_ascii=False,
        )

async def _trace_async_generator_wrapper(
    res: AsyncGenerator[T, None],
    span: Span,
) -> AsyncGenerator[T, None]:
    """Trace the async generator output with OpenTelemetry.

    Args:
        res (`AsyncGenerator[T, None]`):
            The generator or async generator to be traced.
        span (`Span`):
            The OpenTelemetry span to be used for tracing.

    Yields:
        `T`:
            The output of the async generator.
    """
    import opentelemetry

    has_error = False

    try:
        last_chunk = None
        async for chunk in aioitertools.iter(res):
            last_chunk = chunk
            yield chunk

    except Exception as e:
        has_error = True
        span.set_status(StatusCode.ERROR, str(e))
        span.record_exception(e)
        raise e from None

    finally:
        if not has_error:
            # Set the last chunk as output
            if getattr(span, "attributes", {}).get(CommonAttributes.GEN_AI_SPAN_KIND) is GenAiSpanKind.TOOL.value:
                span.set_attributes(
                    {
                        CommonAttributes.GEN_AI_TOOL_CALL_RESULT: _get_tool_result(last_chunk),
                    },
                )
            else:
                span.set_attributes(
                    {
                        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(last_chunk),
                    },
                )
            span.set_status(StatusCode.OK)
        span.end()

def _parse_provider_name(chat_model: ChatModelBase) -> str:
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

def _get_embedding_message(text: list[str]) -> list[dict[str, Any]]:
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

def _get_agent_message(msg: Msg | list[Msg]) -> list[dict[str, Any]]:
    try:
        if isinstance(msg, Msg):
            return [_format_msg_to_parts(msg)]
        elif isinstance(msg, list):
            return [_format_msg_to_parts(msg_item) for msg_item in msg]
    except Exception as e:
        logger.warning(f"Error formatting messages: {e}")
        return msg

def _format_msg_to_parts(msg: Msg) -> dict[str, Any]:
    """将 Msg 转换为标准规范格式（parts结构）
    
    Args:
        msg (Msg): AgentScope 消息对象
        
    Returns:
        dict[str, Any]: 标准规范格式的消息
    """
    try:
        parts = []
        
        # 遍历所有内容块
        for block in msg.get_content_blocks():
            typ = block.get("type")
            
            if typ == "text":
                # 文本块转换
                parts.append({
                    "type": "text",
                    "content": block.get("text", "")
                })
                
            elif typ == "tool_use":
                # 工具调用块转换
                parts.append({
                    "type": "tool_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {})
                })
                
            elif typ == "tool_result":
                # 工具结果块转换
                output = block.get("output", "")
                if isinstance(output, (list, dict)):
                    result = _serialize_to_str(output)
                else:
                    result = str(output)
                    
                parts.append({
                    "type": "tool_call_response",
                    "id": block.get("id", ""),
                    "result": result
                })
                
            elif typ == "image":
                # 图片块转换
                source = block.get("source", {})
                source_type = source.get("type")
                
                if source_type == "url":
                    url = source.get("url", "")
                elif source_type == "base64":
                    data = source.get("data", "")
                    media_type = source.get("media_type", "image/jpeg")
                    url = f"data:{media_type};base64,{data}"
                else:
                    logger.warning(
                        "Unsupported image source type %s, skipped.",
                        source_type,
                    )
                    continue
                    
                parts.append({
                    "type": "image",
                    "url": url
                })
                
            elif typ == "audio":
                # 音频块转换
                source = block.get("source", {})
                parts.append({
                    "type": "audio",
                    "source": source
                })
                
            elif typ == "video":
                # 视频块转换
                source = block.get("source", {})
                parts.append({
                    "type": "video",
                    "source": source
                })
                
            else:
                logger.warning(
                    "Unsupported block type %s in the message, skipped.",
                    typ,
                )
        
        # 构建最终消息格式
        formatted_msg = {
            "role": msg.role,
            "parts": parts
        }
        
        # 如果有name字段且不为空，添加到消息中
        if msg.name:
            formatted_msg["name"] = msg.name
            
        return formatted_msg
        
    except Exception as e:
        logger.warning(f"Error formatting message: {e}")
        # 返回基本格式
        return {
            "role": msg.role,
            "parts": [{
                "type": "text",
                "content": str(msg.content) if msg.content else ""
            }]
        }


def _process_content(content: str, options: GenAITelemetryOptions) -> str:
    """处理消息内容，根据配置决定返回内容或长度信息
    
    Args:
        content (`str`): 原始内容
        options (`GenAITelemetryOptions`): 遥测配置
        
    Returns:
        `str`: 处理后的内容
    """
    try:
        if options.should_capture_content():
            # 捕获内容，应用长度限制
            if content is not None:
                return options.truncate_content(content)
            return ""
        else:
            # 不捕获内容，返回长度信息
            if content is None:
                return "<0size>"
            else:
                size = len(content)
                return f"<{size}size>"
    except Exception as e:
        # 异常处理，返回空字符串
        return ""


def _ot_output_messages(response_data: Any, options: GenAITelemetryOptions) -> list[dict[str, Any]]:
    """处理LLM响应数据，转换为符合OpenTelemetry GenAI规范的格式
    
    Args:
        response_data (`Any`): LLM响应数据
        options (`GenAITelemetryOptions`): 遥测配置
        
    Returns:
        `list[dict[str, Any]]`: 格式化后的输出消息
    """
    output_messages = []
    
    try:
        # 处理响应数据，转换为消息格式
        if response_data is not None:
            # 将响应数据转换为字符串
            response_content = _serialize_to_str(response_data)
            
            # 应用隐私控制
            processed_content = _process_content(response_content, options)
            
            # 构建输出消息，使用parts结构
            output_message = {
                "role": "assistant",
                "parts": [
                    {
                        "type": "text",
                        "content": processed_content
                    }
                ],
                "finish_reason": "stop"
            }
            
            output_messages.append(output_message)
            
    except Exception as e:
        # 异常处理，记录警告并返回空数组
        logger.warning(f"Error processing output messages: {e}")
        return []
        
    return output_messages

def _get_tool_definitions(tools: Optional[list[dict]], tool_choice: Optional[str], structured_model: Optional[bool] = False) -> Optional[list[dict]]:
    if structured_model is True or tools is None or tool_choice is None or tool_choice == "none":
        return None
    else:
        return tools

def _get_tool_description(instance: Toolkit, tool_name: Optional[str]) -> Optional[str]:
    if tool_name is None:
        return None
    try:
        if registered_tool_function := getattr(instance, 'tools', {}).get(tool_name) is None:
            return None
        else:
            if isinstance(func_dict := getattr(registered_tool_function, "json_schema", {}).get("function"), dict):
                return func_dict.get("description")
            return None
    except:
        logger.warning(f"Error getting tool description for tool {tool_name}")
        return None

def _get_tool_result(tool_result: ToolResponse):
    return _serialize_to_str(tool_result.content)