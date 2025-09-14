# -*- coding: utf-8 -*-
"""Attributes processor for span attributes."""
import datetime
import enum
import inspect
import json
from dataclasses import is_dataclass
from typing import Any, AsyncGenerator
import aioitertools
from pydantic import BaseModel
from opentelemetry.trace import Span, StatusCode
from agentscope.message import Msg
from typing import TypeVar
from ..shared import GenAITelemetryOptions

T = TypeVar("T")



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
            span.set_attributes(
                {
                     "gen_ai.output.messages": _serialize_to_str(last_chunk),
                },
            )
            span.set_status(StatusCode.OK)
        span.end()

def _ot_input_messages(call_kwargs, options: GenAITelemetryOptions) -> list[dict[str, str]]:
    """Get input messages with privacy controls applied

    Args:
        call_kwargs (`dict`):
            The input arguments
        options (`GenAITelemetryOptions`):
            The telemetry options for privacy controls

    Returns:
        `List[dict[str, str]]`:
            The input messages for opentelemetry tracing
    """
    input_messages = []

    try:
        if call_kwargs is not None and "messages" in call_kwargs:
            # 转换消息格式为带有parts结构的格式
            for msg in call_kwargs["messages"]:
                converted_msg = {
                    "role": msg.get("role", "user")
                }
                
                parts = []
                
                # 处理OpenAI格式的content
                if "content" in msg and msg["content"]:
                    content = msg["content"]
                    if isinstance(content, str):
                        # 纯字符串内容处理
                        processed_content = _process_content(content, options)
                        parts.append({
                            "type": "text",
                            "content": processed_content
                        })
                    elif isinstance(content, list):
                        # 数组内容（OpenAI多模态格式）
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                    processed_text = _process_content(text_content, options)
                                    parts.append({
                                        "type": "text",
                                        "content": processed_text
                                    })
                                elif item.get("type") == "image_url":
                                    # 图片URL处理
                                    image_url = item.get("image_url", {}).get("url", "")
                                    processed_url = _process_content(image_url, options)
                                    parts.append({
                                        "type": "image",
                                        "content": processed_url
                                    })
                                else:
                                    # 其他类型保持不变
                                    parts.append(item)
                            else:
                                # 非字典类型，可能是文本
                                text_content = str(item)
                                processed_text = _process_content(text_content, options)
                                parts.append({
                                    "type": "text",
                                    "content": processed_text
                                })
                
                # 处理Gemini格式的parts（如果存在）
                if "parts" in msg and msg["parts"]:
                    for part in msg["parts"]:
                        if isinstance(part, dict):
                            if "text" in part:
                                text_content = part["text"]
                                processed_text = _process_content(text_content, options)
                                parts.append({
                                    "type": "text",
                                    "content": processed_text
                                })
                            elif "function_call" in part:
                                func_call = part["function_call"]
                                parts.append({
                                    "type": "tool_call",
                                    "id": func_call.get("id", ""),
                                    "name": func_call.get("name", ""),
                                    "arguments": func_call.get("args", {})
                                })
                            elif "function_response" in part:
                                func_resp = part["function_response"]
                                result_content = str(func_resp.get("response", {}).get("output", ""))
                                processed_result = _process_content(result_content, options)
                                parts.append({
                                    "type": "tool_call_response",
                                    "id": func_resp.get("id", ""),
                                    "result": processed_result
                                })
                            else:
                                # 其他类型保持原样
                                parts.append(part)
                
                # 处理OpenAI格式的工具调用
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            function_info = tool_call.get("function", {})
                            arguments = function_info.get("arguments", "{}")
                            # 如果arguments是字符串，尝试解析为JSON
                            if isinstance(arguments, str):
                                try:
                                    import json
                                    arguments = json.loads(arguments)
                                except:
                                    arguments = {}
                            parts.append({
                                "type": "tool_call",
                                "id": tool_call.get("id", ""),
                                "name": function_info.get("name", ""),
                                "arguments": arguments
                            })
                
                # 处理tool角色的结果
                if msg.get("role") == "tool":
                    tool_content = str(msg.get("content", ""))
                    processed_content = _process_content(tool_content, options)
                    parts.append({
                        "type": "tool_call_response",
                        "id": msg.get("tool_call_id", ""),
                        "result": processed_content
                    })
                
                # 如果没有parts但有其他内容，创建默认文本part
                if not parts and msg.get("content"):
                    text_content = str(msg["content"])
                    processed_content = _process_content(text_content, options)
                    parts.append({
                        "type": "text",
                        "content": processed_content
                    })
                
                converted_msg["parts"] = parts
                input_messages.append(converted_msg)
    
    except Exception as e:
        # 发生异常时返回空数组，避免影响主流程
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error processing input messages: {e}")
        return []
            
    return input_messages


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
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error processing output messages: {e}")
        return []
        
    return output_messages