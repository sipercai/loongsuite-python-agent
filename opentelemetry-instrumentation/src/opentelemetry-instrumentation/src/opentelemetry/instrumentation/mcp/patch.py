import asyncio
import logging
import time
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from .utils import sanitize_tool_name, is_content_enabled, is_input_capture_enabled

logger = logging.getLogger(__name__)


# MCP 特定的语义约定 - 根据OpenTelemetry官方规范
class MCPAttributes:
    # 核心MCP属性
    MCP_METHOD_NAME = "mcp.method.name"
    MCP_TOOL_NAME = "mcp.tool.name"
    MCP_RESOURCE_URI = "mcp.resource.uri"
    MCP_RESOURCE_SIZE = "mcp.resource.size"
    
    # 消息大小相关属性
    MCP_MESSAGE_SIZE = "mcp.message.size"
    MCP_REQUEST_SIZE = "mcp.request.size"
    MCP_RESPONSE_SIZE = "mcp.response.size"
    
    # 协议和连接信息
    MCP_PROTOCOL_VERSION = "mcp.protocol.version"
    MCP_SERVER_ADDRESS = "server.address"
    MCP_SERVER_PORT = "server.port"
    
    # 工具调用相关属性
    MCP_TOOL_ARGUMENTS = "mcp.tool.arguments"
    MCP_TOOL_RESULT = "mcp.tool.result"
    
    # 错误相关属性
    MCP_ERROR_CODE = "mcp.error.code"
    MCP_ERROR_MESSAGE = "mcp.error.message"
    MCP_ERROR_TYPE = "mcp.error.type"
    
    # 内容相关属性
    MCP_CONTENT_COUNT = "mcp.content.count"
    MCP_CONTENT_TYPES = "mcp.content.types"
    MCP_CONTENTS_COUNT = "mcp.contents.count"
    MCP_CONTENTS_TYPES = "mcp.contents.types"


def _calculate_message_size(obj) -> int:
    """计算消息大小（字节）- 暂时简化实现，后续优化"""
    try:
        if obj is None:
            return 0
        if isinstance(obj, (str, bytes)):
            return len(obj)
        # 暂时不记录复杂对象的大小，避免json dumps的hack方式
        # 后续可以通过更合适的方式来实现
        return 0
    except Exception:
        return 0


def _extract_tool_arguments(args, kwargs) -> Dict[str, Any]:
    """提取工具调用参数"""
    try:
        if len(args) > 1:
            return {"arguments": args[1]}
        elif 'arguments' in kwargs:
            return {"arguments": kwargs['arguments']}
        return {}
    except Exception:
        return {}


def _extract_response_details(result) -> Dict[str, Any]:
    """提取响应详情"""
    try:
        if result is None:
            return {"response_type": "null", "response_size": 0}
        
        response_size = _calculate_message_size(result)
        response_type = type(result).__name__
        
        details = {
            "response_type": response_type,
            "response_size": response_size
        }
        
        # 如果是工具调用结果，提取更多信息
        if hasattr(result, 'content'):
            details["content_count"] = len(result.content) if result.content else 0
            if result.content:
                details["content_types"] = [type(item).__name__ for item in result.content]
        
        return details
    except Exception:
        return {"response_type": "unknown", "response_size": 0}


def async_mcp_client_initialize(tracer, event_logger, instruments):
    """异步MCP客户端初始化包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            # 在wrapper内部动态检查是否启用内容捕获
            capture_content = is_content_enabled()
            
            # 根据OpenTelemetry规范：mcp.client.{method}
            with tracer.start_as_current_span(
                "mcp.client.initialize",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "initialize")
                    
                    # 记录协议版本信息
                    if hasattr(instance, '_protocol_version'):
                        span.set_attribute(MCPAttributes.MCP_PROTOCOL_VERSION, str(instance._protocol_version))
                    
                    # 记录连接信息
                    if hasattr(instance, '_server_params'):
                        server_params = instance._server_params
                        if hasattr(server_params, 'command'):
                            span.set_attribute(MCPAttributes.MCP_SERVER_ADDRESS, str(server_params.command))
                
                # 分离instrumentation异常处理和业务逻辑异常处理
                try:
                    start_time = time.time()
                    result = await wrapped(*a, **kw)
                    duration = time.time() - start_time
                    
                    # 记录操作指标
                    instruments.operation_duration.record(duration, {"mcp.method.name": "initialize"})
                    instruments.operation_count.add(1, {"mcp.method.name": "initialize", "status": "success"})
                    
                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        
                        # 记录响应详情
                        response_details = _extract_response_details(result)
                        span.set_attribute(MCPAttributes.MCP_RESPONSE_SIZE, response_details["response_size"])
                        span.set_attribute("mcp.response.type", response_details["response_type"])
                        
                        # 如果启用内容捕获，记录更多详情
                        if capture_content and result:
                            span.set_attribute("mcp.initialize.result", str(result)[:1000])
                    
                    return result
                except Exception as e:
                    # 记录错误指标
                    instruments.operation_count.add(1, {"mcp.method.name": "initialize", "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        
                        # 记录详细错误信息
                        span.set_attribute(MCPAttributes.MCP_ERROR_MESSAGE, str(e))
                        span.set_attribute(MCPAttributes.MCP_ERROR_TYPE, type(e).__name__)
                        if hasattr(e, 'code'):
                            span.set_attribute(MCPAttributes.MCP_ERROR_CODE, str(e.code))
                    
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper


def async_mcp_client_list_tools(tracer, event_logger, instruments):
    """异步MCP客户端列出工具包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            # 在wrapper内部动态检查是否启用内容捕获
            capture_content = is_content_enabled()
            
            # 根据OpenTelemetry规范：mcp.client.{method}
            with tracer.start_as_current_span(
                "mcp.client.list_tools",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "list_tools")
                
                try:
                    start_time = time.time()
                    result = await wrapped(*a, **kw)
                    duration = time.time() - start_time
                    
                    # 记录操作指标
                    instruments.operation_duration.record(duration, {"mcp.method.name": "list_tools"})
                    instruments.operation_count.add(1, {"mcp.method.name": "list_tools", "status": "success"})
                    
                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        
                        # 记录响应详情
                        response_details = _extract_response_details(result)
                        span.set_attribute(MCPAttributes.MCP_RESPONSE_SIZE, response_details["response_size"])
                        span.set_attribute("mcp.response.type", response_details["response_type"])
                        
                        # 记录工具列表信息
                        if result and hasattr(result, 'tools'):
                            span.set_attribute("mcp.tools.count", len(result.tools))
                            if capture_content:
                                tool_names = [tool.name for tool in result.tools] if result.tools else []
                                span.set_attribute("mcp.tools.list", str(tool_names)[:500])
                    
                    return result
                except Exception as e:
                    instruments.operation_count.add(1, {"mcp.method.name": "list_tools", "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        span.set_attribute(MCPAttributes.MCP_ERROR_MESSAGE, str(e))
                        span.set_attribute(MCPAttributes.MCP_ERROR_TYPE, type(e).__name__)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper


def async_mcp_client_call_tool(tracer, event_logger, instruments):
    """异步MCP客户端调用工具包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            tool_name = args[0] if args else kwargs.get('name', 'unknown')
            sanitized_tool_name = sanitize_tool_name(tool_name)
            
            # 在wrapper内部动态检查是否启用内容捕获和输入参数捕获
            capture_content = is_content_enabled()
            capture_input = is_input_capture_enabled()
            
            # 根据OpenTelemetry规范：mcp.client.{method}
            with tracer.start_as_current_span(
                "mcp.client.call_tool",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "call_tool")
                    span.set_attribute(MCPAttributes.MCP_TOOL_NAME, tool_name)
                    
                    # 根据环境变量开关决定是否记录请求参数详情
                    if capture_input:
                        tool_args = _extract_tool_arguments(args, kwargs)
                        if tool_args:
                            span.set_attribute(MCPAttributes.MCP_TOOL_ARGUMENTS, str(tool_args)[:1000])
                            span.set_attribute(MCPAttributes.MCP_REQUEST_SIZE, _calculate_message_size(tool_args))
                    
                    # 如果启用内容捕获，记录工具参数
                    if capture_content and len(args) > 1:
                        span.set_attribute("mcp.tool.arguments.detailed", str(args[1])[:1000])
                
                try:
                    start_time = time.time()
                    result = await wrapped(*a, **kw)
                    duration = time.time() - start_time
                    
                    # 记录操作指标
                    instruments.operation_duration.record(duration, {
                        "mcp.method.name": "call_tool",
                        "mcp.tool.name": tool_name
                    })
                    instruments.operation_count.add(1, {
                        "mcp.method.name": "call_tool",
                        "mcp.tool.name": tool_name,
                        "status": "success"
                    })
                    
                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        
                        # 记录响应详情
                        response_details = _extract_response_details(result)
                        span.set_attribute(MCPAttributes.MCP_RESPONSE_SIZE, response_details["response_size"])
                        span.set_attribute("mcp.response.type", response_details["response_type"])
                        
                        if capture_content and result:
                            span.set_attribute(MCPAttributes.MCP_TOOL_RESULT, str(result)[:1000])
                            
                            # 记录内容详情
                            if hasattr(result, 'content') and result.content:
                                span.set_attribute(MCPAttributes.MCP_CONTENT_COUNT, len(result.content))
                                content_types = [type(item).__name__ for item in result.content]
                                span.set_attribute(MCPAttributes.MCP_CONTENT_TYPES, str(content_types)[:500])
                    
                    return result
                except Exception as e:
                    instruments.operation_count.add(1, {
                        "mcp.method.name": "call_tool",
                        "mcp.tool.name": tool_name,
                        "status": "error"
                    })
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        span.set_attribute(MCPAttributes.MCP_ERROR_MESSAGE, str(e))
                        span.set_attribute(MCPAttributes.MCP_ERROR_TYPE, type(e).__name__)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper
