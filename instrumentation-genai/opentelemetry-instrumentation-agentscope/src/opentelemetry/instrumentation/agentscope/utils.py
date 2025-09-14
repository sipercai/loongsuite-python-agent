import os
import json
import datetime
import enum
import inspect
from importlib.metadata import version
from packaging import version as pkg_version
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Iterable

from opentelemetry.util.types import AttributeValue

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

# 获取 AgentScope 版本
try:
    import agentscope
    _AGENTSCOPE_VERSION = version("agentscope")
except Exception:
    _AGENTSCOPE_VERSION = "0.0.0"

# 环境变量配置
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


def is_agentscope_v1():
    """检查是否为 AgentScope v1.0.0 及以上版本"""
    return pkg_version.parse(_AGENTSCOPE_VERSION) >= pkg_version.parse("1.0.0")

def is_content_enabled() -> bool:
    """检查是否应该捕获消息内容"""
    return (
        os.getenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false").lower() == "true"
    )

def _enable_genai_capture() -> bool:
    """检查是否启用GenAI内容捕获
    
    根据环境变量 OTEL_INSTRUMENTATION_AGENTSCOPE_CAPTURE_MESSAGE_CONTENT 控制功能开关
    """
    return (
        os.getenv(OTEL_INSTRUMENTATION_AGENTSCOPE_CAPTURE_MESSAGE_CONTENT, "false").lower() == "true"
    )

def _to_serializable(obj: Any) -> Any:
    """将对象转换为可JSON序列化的类型

    Args:
        obj: 要转换的对象

    Returns:
        转换后的可JSON序列化对象
    """
    # 处理基本类型
    if isinstance(obj, (str, int, bool, float, type(None))):
        return obj

    elif isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_serializable(x) for x in obj]

    elif isinstance(obj, dict):
        return {str(key): _to_serializable(val) for (key, val) in obj.items()}

    elif BaseModel and isinstance(obj, BaseModel):
        return repr(obj)
    
    elif is_dataclass(obj):
        return repr(obj)

    elif inspect.isclass(obj) and BaseModel and issubclass(obj, BaseModel):
        return repr(obj)

    elif isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return obj.isoformat()

    elif isinstance(obj, datetime.timedelta):
        return obj.total_seconds()

    elif isinstance(obj, enum.Enum):
        return _to_serializable(obj.value)

    else:
        return str(obj)

def _serialize_to_str(value: Any) -> str:
    """将输入值序列化为字符串

    Args:
        value: 输入值

    Returns:
        JSON序列化的字符串
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(
            _to_serializable(value),
            ensure_ascii=False,
        )

def extract_agentscope_attributes(
    instance: Any, 
    operation_name: str = None
) -> Iterable[Tuple[str, AttributeValue]]:
    """从AgentScope实例中提取通用属性
    
    Args:
        instance: AgentScope实例 (Agent, Model, Tool等)
        operation_name: 操作名称
        
    Yields:
        属性键值对
    """
    if hasattr(instance, "name"):
        yield "agentscope.name", str(instance.name)
    if hasattr(instance, "model_name"):
        yield "gen_ai.request.model", str(instance.model_name)
    if hasattr(instance, "model_type"):
        yield "gen_ai.provider.name", str(instance.model_type)
    if hasattr(instance, "id"):
        yield "agentscope.id", str(instance.id)
    if operation_name:
        yield "gen_ai.operation.name", operation_name

def extract_message_content(
    messages: Any,
    content_key: str = "content",
    role_key: str = "role"
) -> Iterable[Tuple[str, AttributeValue]]:
    """从消息中提取内容属性
    
    Args:
        messages: 消息列表或单个消息
        content_key: 内容字段名
        role_key: 角色字段名
        
    Yields:
        消息属性键值对
    """
    if not _enable_genai_capture():
        return
        
    if isinstance(messages, list):
        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                if role_key in message:
                    yield f"gen_ai.input.messages.{idx}.role", str(message[role_key])
                if content_key in message:
                    yield f"gen_ai.input.messages.{idx}.content", _serialize_to_str(message[content_key])
            else:
                # 处理非字典类型的消息
                yield f"gen_ai.input.messages.{idx}", _serialize_to_str(message)
    elif messages is not None:
        # 单个消息
        yield "gen_ai.input.messages", _serialize_to_str(messages)
