#!/usr/bin/env python3
"""
Simple MCP Server Example

This server provides a simple addition tool to demonstrate MCP server capabilities.
"""

from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Simple Math Server")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of the two numbers
    """
    result = a + b
    print(f"Calculation: {a} + {b} = {result}")
    return result


if __name__ == "__main__":
    print("Starting Simple MCP Server...")
    print("Server provides 'add_numbers' tool")
    print("Press Ctrl+C to stop the server")
    mcp.run()
