#!/usr/bin/env python3
"""
Simple MCP Client Example

This client connects to an MCP server, lists available tools, and calls the addition tool.
"""
import asyncio
import sys
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from opentelemetry import trace

async def main():
    """Main function demonstrating basic MCP client usage"""
    print("Starting MCP Client...")
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    server_params = StdioServerParameters(
        command="opentelemetry-instrument", # use opentelemetry-instrument to instrument the server
        args=["--service_name", "demo-mcp-server", sys.executable, os.path.join(current_dir, "simple_server.py")],
        env={
            "PYTHONPATH": os.getcwd(),
            "PATH": os.environ.get("PATH", "")
        }
    )

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("mcp-demo"): 
        async with stdio_client(server_params) as (read_stream, write_stream):
            # Create client session
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize connection
                await session.initialize()

                print("Connection successful!")
                print("=" * 40)

                # List available tools
                print("Getting available tools...")
                tools_result = await session.list_tools()
                tools = tools_result.tools

                print(f"Found {len(tools)} tools:")
                for tool in tools:
                    print(f"   - {tool.name}: {tool.description}")

                print("=" * 40)

                # Call addition tool
                print("Calling add_numbers tool...")

                # Call addition tool with parameters a=15, b=27
                result = await session.call_tool(
                    name="add_numbers",
                    arguments={"a": 15, "b": 27}
                )

                print("Tool call result:")
                for content in result.content:
                    print(f"   {content.text}")

                print("=" * 40)
                print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
