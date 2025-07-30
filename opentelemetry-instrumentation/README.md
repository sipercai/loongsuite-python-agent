 # OpenTelemetry MCP Instrumentation

## 概述

这是一个为MCP (Message Control Protocol) 客户端提供OpenTelemetry可观测性的instrumentation库。它能够自动追踪MCP客户端的操作，包括连接、工具调用、资源读取等，并生成相应的spans和metrics。

## 功能特性

### ✅ 已实现功能
- **异步MCP操作追踪**: 支持所有异步MCP客户端操作
- **标准化命名规范**: 遵循OpenTelemetry语义约定
- **整合的Metrics**: 使用标签方式减少metrics数量
- **异常处理优化**: 分离instrumentation和业务逻辑异常处理
- **完整的测试覆盖**: 包含单元测试和集成测试

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

#### 标准命名格式
- `mcp.client.initialize` - 客户端初始化
- `mcp.client.list_tools` - 列出工具
- `tools/call {tool_name}` - 工具调用
- `resources/read {resource_uri}` - 资源读取
- `mcp.client.send_ping` - 发送ping

#### 属性
- `mcp.method.name` - 操作类型
- `mcp.tool.name` - 工具名称（仅工具调用）
- `mcp.resource.uri` - 资源URI（仅资源读取）
- `mcp.resource.size` - 资源大小（仅资源读取）

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

### 最新版本 (根据文档建议优化)

#### 🎯 主要改进
1. **Metrics整合**: 将8个独立metrics整合为4个，使用标签区分操作类型
2. **命名规范标准化**: 采用社区标准的span命名格式
3. **属性名称优化**: 使用 `mcp.method.name` 替代 `mcp.operation.type`
4. **代码清理**: 删除所有无用的同步函数
5. **异常处理优化**: 分离instrumentation和业务逻辑异常处理

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
- `mcp.client.call_tool.{tool_name}`
- `mcp.client.read_resource`

**现在**:
- `tools/call {tool_name}`
- `resources/read {resource_uri}`

#### 🏷️ 属性变化
**之前**: `mcp.operation.type`
**现在**: `mcp.method.name`

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

Apache License 2.0