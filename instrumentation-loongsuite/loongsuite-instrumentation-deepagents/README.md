# LoongSuite DeepAgents Instrumentation

LoongSuite instrumentation for [DeepAgents](https://github.com/langchain-ai/deepagents).

## Installation

```bash
pip install loongsuite-instrumentation-deepagents
```

## Usage

```python
from opentelemetry.instrumentation.deepagents import DeepAgentsInstrumentor

DeepAgentsInstrumentor().instrument()

from deepagents import create_deep_agent
```

When instrumenting manually, call `DeepAgentsInstrumentor().instrument()` before
importing or binding `deepagents.create_deep_agent`. Auto-instrumentation runs
before application imports and does not need this extra ordering step.

## What it does

This instrumentation patches `deepagents.graph.create_deep_agent` so the final
graph returned by DeepAgents is marked with `_loongsuite_react_agent = True`.
It also marks the graph as a DeepAgents agent and injects the same flags into
call-time `RunnableConfig` metadata for `invoke`, `ainvoke`, `stream`, and
`astream`.

The marker is consumed by `loongsuite-instrumentation-langchain`, which routes
the DeepAgents root graph span as an `AGENT` span instead of a generic `CHAIN`
span.

The LangChain tracer also maps each DeepAgents `model` decision node to a
`react step` span. Multi-round tool flows end the previous step with
`gen_ai.react.finish_reason = "tool_calls"` and the final step with `"stop"`.
This package intentionally does not create separate ENTRY spans or GenAI
metrics.

## Compatibility

- `deepagents >= 0.6.0, < 0.7.0`
- Python 3.10+
