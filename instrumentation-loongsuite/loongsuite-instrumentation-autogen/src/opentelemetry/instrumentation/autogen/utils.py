# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Reasoning,
    Text,
    ToolCall,
    ToolCallResponse,
)

from .semantic_conventions import AUTOGEN_PROVIDER_NAME, GEN_AI_AGENT_NAME


def field_value(value: Any, *names: str) -> Any:
    if value is None:
        return None
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        try:
            attr_value = getattr(value, name)
        except Exception:
            attr_value = None
        if attr_value is not None:
            return attr_value
        get_method = getattr(value, "get", None)
        if callable(get_method):
            try:
                got_value = get_method(name)
            except Exception:
                got_value = None
            if got_value is not None:
                return got_value
    return None


def _type_name(value: Any) -> str:
    return type(value).__name__


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _content_parts(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content=content)]
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            if isinstance(item, str):
                parts.append(Text(content=item))
                continue
            if all(
                hasattr(item, name) for name in ("id", "arguments", "name")
            ):
                parts.append(
                    ToolCall(
                        arguments=field_value(item, "arguments"),
                        name=_text(field_value(item, "name")),
                        id=field_value(item, "id"),
                    )
                )
                continue
            parts.append(Text(content=_text(item)))
        return parts
    return [Text(content=_text(content))]


def to_input_messages(messages: Sequence[Any] | None) -> list[InputMessage]:
    result: list[InputMessage] = []
    for message in messages or []:
        type_name = _type_name(message)
        if type_name == "SystemMessage":
            result.append(
                InputMessage(
                    role="system",
                    parts=[
                        Text(content=_text(field_value(message, "content")))
                    ],
                )
            )
        elif type_name == "UserMessage":
            result.append(
                InputMessage(
                    role="user",
                    parts=_content_parts(field_value(message, "content")),
                )
            )
        elif type_name == "AssistantMessage":
            parts = _content_parts(field_value(message, "content"))
            thought = field_value(message, "thought")
            if thought:
                parts.append(Reasoning(content=_text(thought)))
            result.append(InputMessage(role="assistant", parts=parts))
        elif type_name == "FunctionExecutionResultMessage":
            parts = []
            for item in field_value(message, "content") or []:
                parts.append(
                    ToolCallResponse(
                        response=field_value(item, "content"),
                        id=field_value(item, "call_id"),
                    )
                )
            result.append(InputMessage(role="tool", parts=parts))
        else:
            result.append(
                InputMessage(role="user", parts=[Text(content=_text(message))])
            )
    return result


def tool_definitions(tools: Iterable[Any] | None) -> list[Any]:
    definitions: list[Any] = []
    for tool in tools or []:
        schema = field_value(tool, "schema")
        if callable(schema):
            schema = schema()
        if isinstance(schema, Mapping):
            definitions.append(
                FunctionToolDefinition(
                    name=_text(schema.get("name")),
                    description=schema.get("description"),
                    parameters=schema.get("parameters"),
                )
            )
            continue
        name = field_value(tool, "name")
        if name is not None:
            definitions.append(
                GenericToolDefinition(name=_text(name), type="function")
            )
    return definitions


def model_name(model_client: Any) -> str:
    create_args = field_value(model_client, "_create_args", "create_args")
    if isinstance(create_args, Mapping):
        value = create_args.get("model")
        if value:
            return _text(value)
    for name in ("model", "_model", "model_name", "_model_name"):
        value = field_value(model_client, name)
        if value:
            return _text(value)
    return "unknown"


def response_model_name(
    model_client: Any, fallback: str | None = None
) -> str | None:
    resolved = field_value(model_client, "_resolved_model", "resolved_model")
    if resolved:
        return _text(resolved)
    return fallback


def provider_name(model_client: Any) -> str:
    cls_name = type(model_client).__name__.lower()
    module = type(model_client).__module__.lower()
    raw = f"{module}.{cls_name}"
    base_url = _text(
        field_value(model_client, "base_url", "_base_url")
        or field_value(
            field_value(model_client, "_client"), "base_url", "_base_url"
        )
    ).lower()
    if "dashscope" in base_url:
        return "dashscope"
    if "azure" in raw and "openai" in raw:
        return "azure_ai_openai"
    if "openai" in raw:
        return "openai"
    if "ollama" in raw:
        return "ollama"
    if "anthropic" in raw:
        return "anthropic"
    if "gemini" in raw or "google" in raw:
        return "gemini"
    return AUTOGEN_PROVIDER_NAME


def apply_create_result(
    invocation: LLMInvocation | InvokeAgentInvocation, result: Any
) -> None:
    finish_reason = field_value(result, "finish_reason")
    if finish_reason is not None:
        invocation.finish_reasons = [_text(finish_reason)]

    usage = field_value(result, "usage")
    prompt_tokens = field_value(usage, "prompt_tokens")
    completion_tokens = field_value(usage, "completion_tokens")
    try:
        if prompt_tokens is not None:
            invocation.input_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        pass
    try:
        if completion_tokens is not None:
            invocation.output_tokens = int(completion_tokens)
    except (TypeError, ValueError):
        pass

    content = field_value(result, "content")
    parts = _content_parts(content)
    thought = field_value(result, "thought")
    if thought:
        parts.append(Reasoning(content=_text(thought)))
    if parts:
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=parts,
                finish_reason=_text(finish_reason or "unknown"),
            )
        ]


def make_llm_invocation(
    model_client: Any,
    messages: Sequence[Any] | None,
    tools: Iterable[Any] | None,
    *,
    agent_name: str | None = None,
    output_type: str | None = None,
) -> LLMInvocation:
    request_model = model_name(model_client)
    invocation = LLMInvocation(
        request_model=request_model,
        response_model_name=response_model_name(model_client, request_model),
        provider=provider_name(model_client),
        input_messages=to_input_messages(messages),
        tool_definitions=tool_definitions(tools),
        output_type=output_type,
    )
    if agent_name:
        invocation.attributes[GEN_AI_AGENT_NAME] = agent_name
    return invocation


def make_agent_invocation(instance: Any) -> InvokeAgentInvocation:
    name = _text(field_value(instance, "name") or type(instance).__name__)
    description = field_value(instance, "description")
    model_client = field_value(instance, "_model_client")
    return InvokeAgentInvocation(
        provider=AUTOGEN_PROVIDER_NAME,
        agent_name=name,
        agent_description=_text(description)
        if description is not None
        else None,
        request_model=model_name(model_client)
        if model_client is not None
        else None,
    )
