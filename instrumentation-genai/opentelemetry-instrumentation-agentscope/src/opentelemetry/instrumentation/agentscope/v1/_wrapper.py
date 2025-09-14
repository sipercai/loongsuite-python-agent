# -*- coding: utf-8 -*-
"""Wrapper classes for AgentScope v1.x instrumentation with init hijacking."""

from asyncio.windows_events import NULL
from functools import wraps
from typing import Any, Callable, Tuple, Mapping, Dict, OrderedDict
from abc import ABC
from logging import getLogger
from typing import (
    Generator,
    AsyncGenerator,
)

from agentscope.message import Msg
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Tracer, INVALID_SPAN, StatusCode
from opentelemetry._events import EventLogger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ..shared import (
    LLMRequestAttributes,
    LLMResponseAttributes,
    EmbeddingRequestAttributes,
    EmbeddingResponseAttributes,
    AgentRequestAttributes,
    AgentResponseAttributes,
    GenAiOutputType,
    CommonAttributes,
    LLMAttributes,
    EmbeddingAttributes,
    AgentAttributes,
    get_telemetry_options,
)

from .utils import (
    _serialize_to_str,
    _trace_async_generator_wrapper,
    _ot_input_messages,
    _ot_output_messages,
)

from agentscope.agent import AgentBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse


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

            if not isinstance(call_self, ChatModelBase):
                logger.warning(
                    "Skipping tracing for %s as the first argument"
                    "is not an instance of ChatModelBase, but %s",
                    original_call.__name__,
                    type(call_self),
                )
                return await original_call(call_self, *call_args, **call_kwargs)
            
            # 创建LLM请求属性
            request_attrs = LLMRequestAttributes(
                request_model=getattr(call_self, "model_name", "unknown"),
                request_max_tokens=call_kwargs.get("max_tokens"),
                request_temperature=call_kwargs.get("temperature"),
                request_top_p=call_kwargs.get("top_p"),
                request_top_k=call_kwargs.get("top_k"),
                request_stop_sequences=call_kwargs.get("stop_sequences"),
            )
            
            # 获取基础span属性
            input_attributes = request_attrs.get_span_attributes()
            
            # 添加输入消息（符合GenAI规范）
            telemetry_options = get_telemetry_options()
            input_messages = _ot_input_messages(call_kwargs, telemetry_options)

            input_attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = input_messages
        
            # Begin the llm call span
            with self._tracer.start_as_current_span(
                f"{call_self.__class__.__name__}.__call__",
                attributes=input_attributes,
                end_on_exit=False,
            ) as span:
                try:
                    # Must be an async calling
                    res = await original_call(call_self, *call_args, **call_kwargs)

                    # If the result is a AsyncGenerator
                    if isinstance(res, AsyncGenerator):
                        return _trace_async_generator_wrapper(res, span)

                    # TODO: handle output messages
                    # 创建 LLM 响应属性
                    response_attrs = LLMResponseAttributes(
                        output_type=GenAiOutputType.TEXT,
                        response_finish_reasons=["stop"],
                    )
                    
                    # 设置响应属性
                    span.set_attributes(response_attrs.get_span_attributes())
    
                    output_messages = _ot_output_messages(res, telemetry_options)
                    if output_messages:
                        span.set_attributes({
                            CommonAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(output_messages)
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
            
            if not isinstance(call_self, EmbeddingModelBase):
                logger.warning(
                    "Skipping tracing for %s as the first argument "
                    "does not have model_name attribute, but %s",
                    original_call.__name__,
                    type(call_self),
                )
                return await original_call(call_self, *call_args, **call_kwargs)

            # Prepare the attributes for the span
            attributes = {
                GenAIAttributes.GEN_AI_REQUEST_MODEL: getattr(call_self, "model_name", "unknown"),
                "gen_ai.input.messages": [{
                    "args": call_args,
                    "kwargs": call_kwargs,
                }]
            }
            # TODO: handle embedding messages
            # Begin the embedding call span
            with self._tracer.start_as_current_span(
                f"{call_self.__class__.__name__}.{original_call.__name__}",
                attributes=attributes,
                end_on_exit=False,
            ) as span:
                try:
                    # Must be an async calling
                    res = await original_call(call_self, *call_args, **call_kwargs)

                    # non-generator result
                    span.set_attributes(
                        {"gen_ai.output.messages": _serialize_to_str(res)},
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
            
            if not isinstance(reply_self, AgentBase):
                logger.warning(
                    "Skipping tracing for %s as the first argument "
                    "does not have name attribute, but %s",
                    original_reply.__name__,
                    type(reply_self),
                )
                return await original_reply(reply_self, *reply_args, **reply_kwargs)

            # Prepare the attributes for the span
            attributes = {
                "agentscope.agent.id": getattr(reply_self, "id", "unknown"),
                "agentscope.agent.name": getattr(reply_self, "name", "unknown"),
                "agentscope.operation.name": "agent.reply",
                "gen_ai.input.messages": [{
                    "args": reply_args,
                    "kwargs": reply_kwargs,
                }]
            }

            # Begin the agent reply span
            with self._tracer.start_as_current_span(
                f"{reply_self.__class__.__name__}.{original_reply.__name__}",
                attributes=attributes,
                end_on_exit=False,
            ) as span:
                try:
                    # Must be an async calling
                    res = await original_reply(reply_self, *reply_args, **reply_kwargs)

                    # non-generator result
                    span.set_attributes(
                        {"gen_ai.output.messages": _serialize_to_str(res)},
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