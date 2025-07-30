 # OpenTelemetry MCP Instrumentation

## 概述

这是一个为MCP (Message Control Protocol) 客户端提供OpenTelemetry可观测性的instrumentation库。它能够自动追踪MCP客户端的操作，包括连接、工具调用、资源读取等，并生成相应的spans和metrics。

**✅ 完全符合OpenTelemetry MCP语义约定规范**

## 功能特性

### ✅ 已实现功能
- **异步MCP操作追踪**: 支持所有异步MCP客户端操作
- **OpenTelemetry规范兼容**: 完全遵循官方MCP语义约定
- **整合的Metrics**: 使用标签方式减少metrics数量
- **异常处理优化**: 分离instrumentation和业务逻辑异常处理
- **完整的测试覆盖**: 包含单元测试和集成测试
- **详细的Trace信息**: 包含消息大小、请求参数、响应内容等详细信息

### 🔧 支持的MCP操作
- `initialize` - 客户端初始化
- `list_tools` - 列出可用工具
- `call_tool` - 调用工具
- `read_resource` - 读取资源
- `send_ping` - 发送ping

## 安装

```bash
pip install opentelemetry-instrumentation-mcp
```

## 使用方法

### 基本使用

```python
from opentelemetry.instrumentation.mcp import MCPClientInstrumentor
from opentelemetry import trace, metrics

# 设置OpenTelemetry providers
tracer_provider = TracerProvider()
meter_provider = MeterProvider()

# 启用MCP instrumentation
MCPClientInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider
)

# 现在所有MCP客户端操作都会被自动追踪
```

### 示例代码

查看 `example/mcp/` 目录下的完整示例：
- `demo.py` - 完整的演示脚本
- `client.py` - MCP客户端包装器
- `server.py` - 示例MCP服务器

## 配置选项

### 环境变量

- `OTEL_MCP_CAPTURE_CONTENT`: 是否捕获请求和响应内容（默认: false）

### 自定义配置

```python
# 启用内容捕获
MCPClientInstrumentor().instrument(
    capture_content=True
)
```

## 观测数据

### Spans

每个MCP操作都会生成相应的span：

#### 标准命名格式（符合OpenTelemetry MCP语义约定）
- `mcp.client.initialize` - 客户端初始化
- `mcp.client.list_tools` - 列出工具
- `mcp.client.call_tool` - 工具调用
- `mcp.client.read_resource` - 资源读取
- `mcp.client.send_ping` - 发送ping

#### 核心属性
- `mcp.method.name` - 操作类型
- `mcp.tool.name` - 工具名称（仅工具调用）
- `mcp.resource.uri` - 资源URI（仅资源读取）
- `mcp.resource.size` - 资源大小（仅资源读取）

#### 详细属性
- `mcp.request.size` - 请求大小（字节）
- `mcp.response.size` - 响应大小（字节）
- `mcp.response.type` - 响应类型
- `mcp.tool.arguments` - 工具调用参数
- `mcp.tool.result` - 工具调用结果
- `mcp.content.count` - 内容数量
- `mcp.content.types` - 内容类型
- `mcp.contents.count` - 资源内容数量
- `mcp.contents.types` - 资源内容类型
- `mcp.tools.count` - 工具数量
- `mcp.tools.list` - 工具列表

#### 错误属性
- `mcp.error.message` - 错误消息
- `mcp.error.type` - 错误类型
- `mcp.error.code` - 错误代码

### Metrics

#### 整合的Metrics
- `mcp.client.operation.duration` - 操作持续时间
- `mcp.client.operation.count` - 操作计数
- `mcp.client.connection.duration` - 连接持续时间
- `mcp.client.connection.count` - 连接计数

#### 标签
- `mcp.method.name` - 操作类型
- `mcp.tool.name` - 工具名称（低基数属性）
- `mcp.resource.uri` - 资源URI
- `status` - 操作状态（success/error）

## 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行MCP相关测试
pytest tests/test_mcp_instrumentation.py -v
```

### 运行示例

```bash
cd example/mcp
python demo.py
```

## 更新日志

### 最新版本 (符合OpenTelemetry MCP语义约定规范)

#### 🎯 主要改进
1. **OpenTelemetry规范兼容**: 完全遵循官方MCP语义约定
2. **标准化命名**: 使用 `mcp.client.{method}` 格式
3. **详细Trace信息**: 添加消息大小、请求参数、响应内容等详细信息
4. **Metrics整合**: 将8个独立metrics整合为4个，使用标签区分操作类型
5. **属性名称优化**: 使用标准化的MCP属性名称
6. **代码清理**: 删除所有无用的同步函数
7. **异常处理优化**: 分离instrumentation和业务逻辑异常处理

#### 📊 Metrics变化
**之前**: 8个独立metrics
- `message_count`, `message_duration`, `message_size`
- `tool_call_duration`, `tool_call_count`
- `resource_read_duration`, `resource_read_count`, `resource_size`

**现在**: 4个整合metrics
- `mcp.client.operation.duration` (带标签)
- `mcp.client.operation.count` (带标签)
- `mcp.client.connection.duration`
- `mcp.client.connection.count`

#### 🔍 Span命名变化
**之前**:
- `tools/call {tool_name}`
- `resources/read {resource_uri}`

**现在** (符合OpenTelemetry规范):
- `mcp.client.call_tool`
- `mcp.client.read_resource`

#### 🏷️ 新增详细属性
- `mcp.request.size` - 请求大小
- `mcp.response.size` - 响应大小
- `mcp.response.type` - 响应类型
- `mcp.tool.arguments` - 工具参数
- `mcp.content.count` - 内容数量
- `mcp.content.types` - 内容类型
- `mcp.contents.count` - 资源内容数量
- `mcp.contents.types` - 资源内容类型
- `mcp.tools.count` - 工具数量
- `mcp.tools.list` - 工具列表

#### 🚨 错误处理增强
- `mcp.error.message` - 详细错误消息
- `mcp.error.type` - 错误类型
- `mcp.error.code` - 错误代码

## 规范参考

本实现完全遵循以下OpenTelemetry规范：
- [OpenTelemetry MCP语义约定](https://github.com/open-telemetry/semantic-conventions/blob/dc77673926c7b236f62440cf70f1dcc79bebc575/docs/gen-ai/mcp.md)
- [OpenTelemetry通用语义约定](https://opentelemetry.io/docs/specs/semconv/)

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

Apache License 2.0