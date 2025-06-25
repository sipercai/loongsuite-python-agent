import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_client_example():
    # 创建服务器参数配置
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None
    )

    # 使用stdio_client连接到服务器
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 初始化连接
            result = await session.initialize()
            print(f"Connected to server: {result.serverInfo.name}")

            # 列出可用工具
            tools_result = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools_result.tools]}")

            # 调用add工具
            add_result = await session.call_tool("add", {"a": 5, "b": 3})
            print(f"Add result: {add_result.content}")

            # 调用echo工具
            echo_result = await session.call_tool("echo", {"message": "Hello MCP!"})
            print(f"Echo result: {echo_result.content}")

            # 读取资源
            try:
                resource_content, mime_type = await session.read_resource("greeting://World")
                print(f"Resource content: {resource_content}")
                print(f"MIME type: {mime_type}")
            except Exception as e:
                print(f"Resource read failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_client_example())