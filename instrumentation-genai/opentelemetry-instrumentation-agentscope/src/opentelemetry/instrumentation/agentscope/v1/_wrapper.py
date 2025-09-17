# -*- coding: utf-8 -*-
"""Wrapper classes for AgentScope v1.x instrumentation with init hijacking."""

from functools import wraps
from typing import Any, Callable, Tuple, Mapping
from abc import ABC
from logging import getLogger
from typing import (
    Generator,
    AsyncGenerator,
)
import inspect

from agentscope.message import Msg
from agentscope.agent import AgentBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse

from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry._events import EventLogger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ..shared import (
    LLMRequestAttributes,
    LLMResponseAttributes,
    EmbeddingRequestAttributes,
    AgentRequestAttributes,
    get_telemetry_options,
)
from ._request_attributes_extractor import get_message_converter
from ._response_attributes_extractor import _get_chatmodel_output_messages

from .utils import (
    _serialize_to_str,
    _trace_async_generator_wrapper,
    _get_tool_definitions,
    _parse_provider_name,
    _get_embedding_message,
    _get_agent_message,
    _format_msg_to_parts
)


logger = getLogger(__name__)


class _WithTracer(ABC):
    
    def __init__(self, tracer: trace.Tracer, event_logger: EventLogger, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._event_logger = event_logger


class AgentScopeV1ChatModelWrapper(_WithTracer):
    """ instrument AgentScope v1.x ChatModel __call__ """
    
    # 类级别的原始方法存储
    _original_methods = {}

    def __init__(self, tracer: trace.Tracer, event_logger: EventLogger):
        super().__init__(tracer, event_logger)
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """恢复所有被替换的原始方法。"""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(
        self,
        original_init: Callable[..., Any],
        instance: ChatModelBase,
        init_args: Tuple[type, Any],
        init_kwargs: Mapping[str, Any],
    ) -> None:

        # run ChatModelBase.__init__
        original_init(*init_args, **init_kwargs)
        
        model_class = instance.__class__
        if model_class in self._instrumented_classes or model_class.__dict__.get("__call__") is None:
            # had already replaced or not callable
            return  

        # 保存原始方法
        original_call = model_class.__call__
        if model_class not in self._original_methods:
            self._original_methods[model_class] = {}
        self._original_methods[model_class]["__call__"] = original_call
        
        @wraps(original_call)
        async def async_wrapped_call(
            call_self: ChatModelBase,
            *call_args: Any,
            **call_kwargs: Any,
        ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
            """The wrapper function for tracing the LLM call."""
            
            # 创建LLM请求属性
            request_attrs = LLMRequestAttributes(
                operation_name = GenAIAttributes.GenAiOperationNameValues.CHAT.value, 
                provider_name = _parse_provider_name(call_self),
                request_model = getattr(call_self, "model_name", "unknown_model"),
                request_max_tokens = call_kwargs.get("max_tokens"),
                request_temperature = call_kwargs.get("temperature"),
                request_top_p = call_kwargs.get("top_p"),
                request_top_k = call_kwargs.get("top_k"),
                request_stop_sequences = call_kwargs.get("stop_sequences"),
                request_tool_definitions = _get_tool_definitions(tools=call_kwargs.get("tools"), tool_choice=call_kwargs.get("tool_choice"), structured_model=call_kwargs.get("structured_model")),
            )
            
            # 获取基础span属性
            input_attributes = request_attrs.get_span_attributes()
            input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model

            # 添加输入消息（符合GenAI规范）
            telemetry_options = get_telemetry_options()
            messages = None
            if call_args and len(call_args) > 0:
                messages = call_args[0]
            elif "messages" in call_kwargs:
                messages = call_kwargs["messages"]
            if messages:
                input_messages = get_message_converter(request_attrs.provider_name)(messages)
            else:
                logger.warning(" ChatModelWrapper No messages provided. Skipping input message conversion.")
                input_messages = []

            input_attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = _serialize_to_str(input_messages)
        
            # Begin the llm call span
            with self._tracer.start_as_current_span(
                f"{request_attrs.operation_name or "unknown_operation"} {request_attrs.request_model or "unknown_model"}",
                attributes=input_attributes,
                end_on_exit=False,
            ) as span:
                try:
                    # Must be an async calling
                    res = await original_call(call_self, *call_args, **call_kwargs)

                    # If the result is a AsyncGenerator
                    if isinstance(res, AsyncGenerator):
                        return _trace_async_generator_wrapper(res, span)

                    # 创建 LLM 响应属性
                    response_attrs = LLMResponseAttributes(
                        output_type = GenAIAttributes.GenAiOutputTypeValues.TEXT.value,
                        response_id = getattr(res, "id", "unknown_id"),
                        response_finish_reasons = '["stop"]',
                    )
                    if hasattr(res, "usage") and res.usage:
                        response_attrs.usage_input_tokens = res.usage.input_tokens,
                        response_attrs.usage_output_tokens = res.usage.output_tokens,
                    # 设置响应属性
                    span.set_attributes(response_attrs.get_span_attributes())

                    output_messages = _get_chatmodel_output_messages(res)
                    if output_messages:
                        span.set_attributes({
                            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(output_messages)
                        })
                    
                    span.set_status(StatusCode.OK)
                    span.end()
                    return res

                except Exception as e:
                    span.set_status(
                        StatusCode.ERROR,
                        str(e),
                    )
                    span.record_exception(e)
                    span.end()
                    raise e from None

        # 替换类的__call__方法
        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(model_class)


class AgentScopeV1EmbeddingModelWrapper(_WithTracer):
    """AgentScope v1.x EmbeddingModel的包装器，通过init挟持实现。"""
    
    # 类级别的原始方法存储
    _original_methods = {}

    def __init__(self, tracer: trace.Tracer, event_logger: EventLogger):
        super().__init__(tracer, event_logger)
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """恢复所有被替换的原始方法。"""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(
        self,
        original_init: Callable[..., Any],
        instance: Any,
        init_args: Tuple[type, Any],
        init_kwargs: Mapping[str, Any],
    ) -> None:
        """挟持__init__方法，然后替换__call__方法。"""
        
        # 先执行原始的__init__
        original_init(*init_args, **init_kwargs)
        
        embedding_class = instance.__class__
        if embedding_class in self._instrumented_classes or embedding_class.__dict__.get("__call__") is None:
            # had already replaced or not callable
            return

        # 保存原始的__call__方法
        original_call = embedding_class.__call__
        if embedding_class not in self._original_methods:
            self._original_methods[embedding_class] = {}
        self._original_methods[embedding_class]["__call__"] = original_call
        
        @wraps(original_call)
        async def async_wrapped_call(
            call_self: EmbeddingModelBase,
            *call_args: Any,
            **call_kwargs: Any,
        ) -> EmbeddingResponse:
            """包装后的__call__方法，添加追踪逻辑。"""

            # Prepare the attributes for the span
            # 创建embedding请求属性
            request_attrs = EmbeddingRequestAttributes(
                operation_name = GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value, 
                request_model = getattr(call_self, "model_name", "unknown_model"),
                request_encoding_formats = call_kwargs.get("encoding_formats"),
            )
            
            # 获取基础span属性
            input_attributes = request_attrs.get_span_attributes()

            text_for_embedding = None
            if call_args and len(call_args) > 0:
                text_for_embedding = call_args[0]
            elif "text" in call_kwargs:
                text_for_embedding = call_kwargs["text"]
            if text_for_embedding:
                input_messages = _get_embedding_message(text_for_embedding)
            else:
                logger.warning(" EmbeddingModelWrapper No text provided. Skipping input message conversion.")
                input_messages = {
                    "args": call_args,
                    "kwargs": call_kwargs,
                }

            input_attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = _serialize_to_str(input_messages)

            # Begin the embedding call span
            with self._tracer.start_as_current_span(
                f"{request_attrs.operation_name or "unknown_operation"} {request_attrs.request_model or "unknown_model"}",
                attributes=input_attributes,
                end_on_exit=False,
            ) as span:
                try:
                    # Must be an async calling
                    res = await original_call(call_self, *call_args, **call_kwargs)

                    # non-generator result
                    span.set_attributes({
                        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(res),
                        GenAIAttributes.GEN_AI_RESPONSE_MODEL: request_attrs.request_model,
                        GenAIAttributes.GEN_AI_DATA_SOURCE_ID: getattr(res, "source", "unknown_source"),
                        GenAIAttributes.GEN_AI_RESPONSE_ID: getattr(res, "id", "unknown_id"),
                    })
                    if hasattr(res, "embeddings") and res.embeddings:
                        span.set_attribute(key="gen_ai.embeddings.dimension.count", value=len(res.embeddings[0]))
                    if hasattr(res, "usage") and res.usage:
                        span.set_attribute(key=GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, value=getattr(res.usage, "tokens", 0))

                    span.set_status(StatusCode.OK)
                    span.end()
                    return res

                except Exception as e:
                    span.set_status(
                        StatusCode.ERROR,
                        str(e),
                    )
                    span.record_exception(e)
                    span.end()
                    raise e from None

        # 替换类的__call__方法
        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(embedding_class)


class AgentScopeV1AgentWrapper(_WithTracer):
    """AgentScope v1.x Agent reply"""
    
    # 类级别的原始方法存储
    _original_methods = {}

    def __init__(self, tracer: trace.Tracer, event_logger: EventLogger):
        super().__init__(tracer, event_logger)
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """恢复所有被替换的原始方法。"""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(
        self,
        original_init: Callable[..., Any],
        instance: Any,
        init_args: Tuple[type, Any],
        init_kwargs: Mapping[str, Any],
    ) -> None:
        """patch __init__ warp reply"""
        
        # 先执行原始的__init__
        original_init(*init_args, **init_kwargs)
        
        agent_class = instance.__class__
        if agent_class in self._instrumented_classes or agent_class.__dict__.get("reply") is None:
            # had already replaced or not reply
            return

        original_reply = agent_class.reply
        
        @wraps(original_reply)
        async def async_wrapped_reply(
            reply_self: AgentBase,
            *reply_args: Any,
            **reply_kwargs: Any,
        ) -> Msg:
            """包装后的reply方法，添加追踪逻辑。"""

            # Prepare the attributes for the span
            # 创建agent reply请求属性
            request_attrs = AgentRequestAttributes(
                operation_name = GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value, 
                agent_id = getattr(reply_self, "id", "unknown"),
                agent_name = getattr(reply_self, "name", "unknown_agent"),
                agent_description = inspect.getdoc(reply_self.__class__) or "No description available",
                system_instructions = reply_self.sys_prompt if hasattr(reply_self, "sys_prompt") else None,
            )
            if hasattr(reply_self, "model") and reply_self.model:
                request_attrs.request_model = getattr(reply_self.model, "model_name", "unknown_model")
             # 获取基础span属性
            input_attributes = request_attrs.get_span_attributes()

            msg = None
            if reply_args and len(reply_args) > 0:
                msg = reply_args[0]
            elif "msg" in reply_kwargs:
                msg = reply_kwargs["msg"]
            if msg:
                input_attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = _serialize_to_str(_get_agent_message(msg))
            else:
                logger.warning(" AgentWrapper No msg provided. Skipping input message conversion.")
                input_attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = _serialize_to_str({
                    "args": reply_args,
                    "kwargs": reply_kwargs,
                })

            # Begin the agent reply span
            with self._tracer.start_as_current_span(
                f"{request_attrs.operation_name or "unknown_operation"} {request_attrs.agent_name or "unknown_agent"}",
                attributes=input_attributes,
                end_on_exit=False,
            ) as span:
                try:
                    # Must be an async calling
                    res = await original_reply(reply_self, *reply_args, **reply_kwargs)

                    # non-generator result
                    span.set_attributes(
                        {
                            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(_format_msg_to_parts(res)),
                            GenAIAttributes.GEN_AI_RESPONSE_MODEL: request_attrs.request_model,
                        },
                    )
                    span.set_status(StatusCode.OK)
                    span.end()
                    return res

                except Exception as e:
                    span.set_status(
                        StatusCode.ERROR,
                        str(e),
                    )
                    span.record_exception(e)
                    span.end()
                    raise e from None

        # 替换类的reply方法
        instance.__class__.reply = async_wrapped_reply        
        self._instrumented_classes.add(agent_class)