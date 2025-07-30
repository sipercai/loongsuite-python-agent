import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.mcp import MCPClientInstrumentor

@pytest.mark.asyncio
class TestMCPInstrumentation:
    def setup_method(self):
        self.span_exporter = InMemorySpanExporter()
        self.metric_reader = InMemoryMetricReader()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        self.meter_provider = MeterProvider(metric_readers=[self.metric_reader])
        trace.set_tracer_provider(self.tracer_provider)
        metrics.set_meter_provider(self.meter_provider)
        self.instrumentor = MCPClientInstrumentor()

    def teardown_method(self):
        self.instrumentor.uninstrument()
        self.span_exporter.clear()
        self.metric_reader = InMemoryMetricReader()

    @patch('mcp.client.session.ClientSession.initialize', new_callable=AsyncMock)
    async def test_initialize_instrumentation(self, mock_initialize):
        mock_initialize.return_value = AsyncMock()
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
        from mcp.client.session import ClientSession
        client = ClientSession.__new__(ClientSession)  # 避免构造参数
        await client.initialize()
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mcp.client.initialize"
        assert spans[0].attributes.get("mcp.method.name") == "initialize"

    @patch('mcp.client.session.ClientSession.read_resource', new_callable=AsyncMock)
    async def test_read_resource_instrumentation(self, mock_read_resource):
        mock_read_resource.return_value = AsyncMock(contents=b"hello")
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
        from mcp.client.session import ClientSession
        client = ClientSession.__new__(ClientSession)
        await client.read_resource("test://resource")
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "resources/read test://resource"
        assert spans[0].attributes.get("mcp.method.name") == "resources/read"
        assert spans[0].attributes.get("mcp.resource.uri") == "test://resource"

    @patch('mcp.client.session.ClientSession.call_tool', new_callable=AsyncMock)
    async def test_call_tool_instrumentation(self, mock_call_tool):
        mock_call_tool.return_value = AsyncMock()
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
        from mcp.client.session import ClientSession
        client = ClientSession.__new__(ClientSession)
        await client.call_tool("test_tool", {"param": "value"})
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "tools/call test_tool"
        assert spans[0].attributes.get("mcp.method.name") == "tools/call"
        assert spans[0].attributes.get("mcp.tool.name") == "test_tool"

    @patch('mcp.client.session.ClientSession.list_tools', new_callable=AsyncMock)
    async def test_list_tools_instrumentation(self, mock_list_tools):
        mock_list_tools.return_value = AsyncMock()
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
        from mcp.client.session import ClientSession
        client = ClientSession.__new__(ClientSession)
        await client.list_tools()
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mcp.client.list_tools"
        assert spans[0].attributes.get("mcp.method.name") == "list_tools"

    @patch('mcp.client.session.ClientSession.send_ping', new_callable=AsyncMock)
    async def test_send_ping_instrumentation(self, mock_send_ping):
        mock_send_ping.return_value = AsyncMock()
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
        from mcp.client.session import ClientSession
        client = ClientSession.__new__(ClientSession)
        await client.send_ping()
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mcp.client.send_ping"
        assert spans[0].attributes.get("mcp.method.name") == "send_ping"

    async def test_uninstrument(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
        assert hasattr(self.instrumentor, '_meter')
        self.instrumentor.uninstrument()
        # 取消后再次调用不会生成span
        from mcp.client.session import ClientSession
        with patch('mcp.client.session.ClientSession.initialize', new_callable=AsyncMock) as mock_initialize:
            mock_initialize.return_value = AsyncMock()
            client = ClientSession.__new__(ClientSession)
            await client.initialize()
            spans = self.span_exporter.get_finished_spans()
            # 由于已uninstrument，不应有新span
            assert len(spans) == 0