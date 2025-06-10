import pytest
from opentelemetry.instrumentation.agno import AgnoInstrumentor
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
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

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
    return tracer_provider

@pytest.fixture(autouse=True,scope="module")
def instrument(
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter
) -> Generator:
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgnoInstrumentor().uninstrument()

def test_agno(request, in_memory_span_exporter: InMemorySpanExporter):
    agent = Agent(
        model=DeepSeek(id="deepseek-chat"),
        tools=[
            ReasoningTools(add_instructions=True),
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                company_info=True,
                company_news=True,
            ),
        ],
        instructions=[
            "Use tables to display data",
            "Only output the report, no other text",
        ],
        markdown=True,
    )
    agent.print_response(
        "Write a report on NVDA",
    )
    check_agent, check_model, check_tool = False, False, False
    spans = in_memory_span_exporter.get_finished_spans()
    for span in spans:
        if span.name == "Agent.run":
            check_agent = True
        if span.name == "Model.response":
            check_model = True
        if span.name == "ToolCall":
            check_tool = True
    assert check_agent and check_model and check_tool, "Agent, Model or ToolCall span not found"