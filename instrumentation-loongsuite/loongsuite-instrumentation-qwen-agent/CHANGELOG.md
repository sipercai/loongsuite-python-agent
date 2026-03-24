# Changelog

## Unreleased

### Added

- Initial release of `loongsuite-instrumentation-qwen-agent`
- Instrumentation for `Agent.run()` (invoke_agent spans; `run_nonstream()` is covered via its internal `run()` call — no duplicate span)
- Instrumentation for `BaseChatModel.chat()` (LLM spans)
- Instrumentation for `Agent._call_tool()` (execute_tool spans)
- Support for streaming and non-streaming LLM responses
- Message conversion from qwen-agent types to GenAI semantic conventions
- Provider detection for DashScope, OpenAI, and Azure backends
