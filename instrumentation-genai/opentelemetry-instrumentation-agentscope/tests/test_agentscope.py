import pytest
from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider, Resource
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from typing import (
    Generator
)
from agentscope.agents import DialogAgent
import agentscope


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()

@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    tracer_provider.add_span_processor(span_processor=SimpleSpanProcessor(ConsoleSpanExporter()))
    # trace_api.set_tracer_provider(tracer_provider)
    return tracer_provider

@pytest.fixture(autouse=True)
def instrument(
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter
) -> Generator:
    AgentScopeInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgentScopeInstrumentor().uninstrument()

def test_chat_model_call(request, in_memory_span_exporter: InMemorySpanExporter):
    agentscope.init(
        model_configs={
            "config_name": "my-qwen-max",
            "model_name": "qwen-max",
            "model_type": "dashscope_chat",
            "api_key": request.config.option.api_key,
        },
    )
    agent = DialogAgent(
        name="Agent",
        sys_prompt="You're a helpful assistant.",
        model_config_name="my-qwen-max",
    )
    msg = None
    agent(msg)
    spans = in_memory_span_exporter.get_finished_spans()
    attributes = spans[0].attributes
    assert attributes is not None
    for attribute in attributes:
        if GenAIAttributes.GEN_AI_PROMPT in attribute:
            assert True
            return
    assert False, "GEN_AI_PROMPT attribute not found in span attributes"
