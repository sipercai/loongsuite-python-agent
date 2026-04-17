"""Shared constants for Hermes telemetry."""

from __future__ import annotations

INSTRUMENTATION_VERSION = "0.1.0"

INSTRUMENTATION_DEPENDENCIES = ("openai >= 1.0.0",)

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_SESSION_ID = "gen_ai.session.id"
GEN_AI_SPAN_KIND = "gen_ai.span.kind"

GEN_AI_KIND_AGENT = "AGENT"
GEN_AI_KIND_ENTRY = "ENTRY"
GEN_AI_KIND_LLM = "LLM"
GEN_AI_KIND_STEP = "STEP"
GEN_AI_KIND_TOOL = "TOOL"

GEN_AI_OP_CHAT = "chat"
GEN_AI_OP_ENTER = "enter"
GEN_AI_OP_EXECUTE_TOOL = "execute_tool"
GEN_AI_OP_INVOKE_AGENT = "invoke_agent"
GEN_AI_OP_REACT = "react"

HERMES_PROVIDER = "hermes-agent"
