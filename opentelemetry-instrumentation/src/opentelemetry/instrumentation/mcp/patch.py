import asyncio
import logging
import time
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from .utils import sanitize_tool_name

logger = logging.getLogger(__name__)


# MCP 特定的语义约定 - 根据社区标准调整
class MCPAttributes:
    MCP_METHOD_NAME = "mcp.method.name"  # 替换 MCP_OPERATION_TYPE
    MCP_TOOL_NAME = "mcp.tool.name"
    MCP_RESOURCE_URI = "mcp.resource.uri"
    MCP_RESOURCE_SIZE = "mcp.resource.size"
    MCP_ERROR_CODE = "mcp.error.code"


def async_mcp_client_initialize(tracer, event_logger, instruments, capture_content):
    """异步MCP客户端初始化包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            # 使用标准命名格式: (mcp.method.name)
            with tracer.start_as_current_span(
                "mcp.client.initialize",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "initialize")
                
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
                    return result
                except Exception as e:
                    # 记录错误指标
                    instruments.operation_count.add(1, {"mcp.method.name": "initialize", "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper


def async_mcp_client_list_tools(tracer, event_logger, instruments, capture_content):
    """异步MCP客户端列出工具包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
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
                    return result
                except Exception as e:
                    instruments.operation_count.add(1, {"mcp.method.name": "list_tools", "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper


def async_mcp_client_call_tool(tracer, event_logger, instruments, capture_content):
    """异步MCP客户端调用工具包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            tool_name = args[0] if args else kwargs.get('name', 'unknown')
            sanitized_tool_name = sanitize_tool_name(tool_name)
            
            # 使用标准命名格式: (mcp.method.name) {target}
            with tracer.start_as_current_span(
                f"tools/call {tool_name}",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "tools/call")
                    span.set_attribute(MCPAttributes.MCP_TOOL_NAME, tool_name)
                    
                    # 如果启用内容捕获，记录工具参数
                    if capture_content and len(args) > 1:
                        span.set_attribute("mcp.tool.arguments", str(args[1])[:1000])
                
                try:
                    start_time = time.time()
                    result = await wrapped(*a, **kw)
                    duration = time.time() - start_time
                    
                    # 记录操作指标
                    instruments.operation_duration.record(duration, {
                        "mcp.method.name": "tools/call",
                        "mcp.tool.name": tool_name
                    })
                    instruments.operation_count.add(1, {
                        "mcp.method.name": "tools/call",
                        "mcp.tool.name": tool_name,
                        "status": "success"
                    })
                    
                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        if capture_content and result:
                            span.set_attribute("mcp.tool.result", str(result)[:1000])
                    
                    return result
                except Exception as e:
                    instruments.operation_count.add(1, {
                        "mcp.method.name": "tools/call",
                        "mcp.tool.name": tool_name,
                        "status": "error"
                    })
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper


def async_mcp_client_read_resource(tracer, event_logger, instruments, capture_content):
    """异步MCP客户端读取资源包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            resource_uri = args[0] if args else kwargs.get('uri', 'unknown')
            
            # 使用标准命名格式: (mcp.method.name) {target}
            with tracer.start_as_current_span(
                f"resources/read {resource_uri}",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "resources/read")
                    span.set_attribute(MCPAttributes.MCP_RESOURCE_URI, resource_uri)
                
                try:
                    start_time = time.time()
                    result = await wrapped(*a, **kw)
                    duration = time.time() - start_time
                    
                    # 记录操作指标
                    instruments.operation_duration.record(duration, {
                        "mcp.method.name": "resources/read",
                        "mcp.resource.uri": resource_uri
                    })
                    instruments.operation_count.add(1, {
                        "mcp.method.name": "resources/read",
                        "mcp.resource.uri": resource_uri,
                        "status": "success"
                    })
                    
                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        if result and hasattr(result, 'contents'):
                            resource_size = len(result.contents) if result.contents else 0
                            span.set_attribute(MCPAttributes.MCP_RESOURCE_SIZE, resource_size)
                    
                    return result
                except Exception as e:
                    instruments.operation_count.add(1, {
                        "mcp.method.name": "resources/read",
                        "mcp.resource.uri": resource_uri,
                        "status": "error"
                    })
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper


def async_mcp_client_send_ping(tracer, event_logger, instruments, capture_content):
    """异步MCP客户端发送ping包装器"""
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper(*a, **kw):
            with tracer.start_as_current_span(
                "mcp.client.send_ping",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_METHOD_NAME, "send_ping")
                
                try:
                    start_time = time.time()
                    result = await wrapped(*a, **kw)
                    duration = time.time() - start_time
                    
                    # 记录操作指标
                    instruments.operation_duration.record(duration, {"mcp.method.name": "send_ping"})
                    instruments.operation_count.add(1, {"mcp.method.name": "send_ping", "status": "success"})
                    
                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    instruments.operation_count.add(1, {"mcp.method.name": "send_ping", "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        return async_wrapper(*args, **kwargs)
    return wrapper