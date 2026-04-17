"""Helper utilities for Hermes telemetry wrappers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from .constants import (
    GEN_AI_KIND_STEP,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OP_REACT,
    GEN_AI_SPAN_KIND,
)


def obj_get(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def provider_name(instance: Any) -> str:
    provider = str(getattr(instance, "provider", "") or "").strip().lower()
    if provider:
        return provider

    base_url = str(getattr(instance, "base_url", "") or "").lower()
    if "dashscope" in base_url:
        return "dashscope"
    if "chatgpt.com/backend-api/codex" in base_url:
        return "openai-codex"
    if "openai" in base_url:
        return "openai"
    return "custom"


def safe_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=lambda obj: getattr(obj, "__dict__", str(obj)),
    )


def _flatten_content(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = obj_get(item, "text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            content = obj_get(item, "content")
            if isinstance(content, str) and content:
                parts.append(content)
        flattened = "\n".join(part for part in parts if part).strip()
        return flattened or safe_json(value)
    return str(value)


def _normalize_chat_message(message: Any) -> dict[str, Any]:
    role = obj_get(message, "role", "")
    return {
        "role": role or "",
        "content": _flatten_content(obj_get(message, "content")),
        "tool_calls": obj_get(message, "tool_calls"),
        "tool_call_id": obj_get(message, "tool_call_id"),
    }


def _normalize_codex_input_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    role = item.get("role")

    if role in {"system", "user", "assistant", "tool"}:
        return {
            "role": role,
            "content": _flatten_content(item.get("content")),
            "tool_call_id": item.get("tool_call_id"),
        }

    if item_type == "function_call":
        arguments = item.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = safe_json(arguments)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name") or item.get("function_name"),
                        "arguments": arguments,
                    },
                }
            ],
        }

    if item_type == "function_call_output":
        output = item.get("output")
        return {
            "role": "tool",
            "content": _flatten_content(output),
            "tool_call_id": item.get("call_id"),
        }

    if item_type == "message":
        return {
            "role": item.get("role", "assistant"),
            "content": _flatten_content(item.get("content")),
        }

    return None


def serialize_request_messages(api_kwargs: Any) -> str | None:
    if not isinstance(api_kwargs, dict):
        return None

    serialized: list[dict[str, Any]] = []

    instructions = api_kwargs.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        serialized.append({"role": "system", "content": instructions.strip()})

    messages = api_kwargs.get("messages")
    if isinstance(messages, list):
        serialized.extend(_normalize_chat_message(message) for message in messages)

    input_items = api_kwargs.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            normalized = _normalize_codex_input_item(item)
            if normalized is not None:
                serialized.append(normalized)
    elif isinstance(input_items, str) and input_items.strip():
        serialized.append({"role": "user", "content": input_items.strip()})

    if not serialized:
        return None
    return safe_json(serialized)


def response_message(response: Any) -> dict[str, Any]:
    choice = None
    try:
        choice = response.choices[0]
    except Exception:
        choice = None

    if choice is not None:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")

        return {
            "role": "assistant",
            "content": _flatten_content(obj_get(message, "content")),
            "tool_calls": obj_get(message, "tool_calls"),
        }

    content_parts: list[str] = []
    tool_calls: list[Any] = []
    for item in obj_get(response, "output", []) or []:
        item_type = obj_get(item, "type", "")
        if item_type == "function_call":
            arguments = obj_get(item, "arguments", obj_get(item, "input", "{}"))
            if not isinstance(arguments, str):
                arguments = safe_json(arguments)
            tool_calls.append(
                SimpleNamespace(
                    id=obj_get(item, "call_id") or obj_get(item, "id"),
                    type="function",
                    function=SimpleNamespace(
                        name=obj_get(item, "name") or obj_get(item, "function_name"),
                        arguments=arguments,
                    ),
                )
            )
            continue

        text = _flatten_content(obj_get(item, "content"))
        if isinstance(text, str) and text:
            content_parts.append(text)
            continue

        raw_text = obj_get(item, "text") or obj_get(item, "output_text")
        if isinstance(raw_text, str) and raw_text:
            content_parts.append(raw_text)

    content = "\n".join(part for part in content_parts if part).strip()
    if not content:
        output_text = obj_get(response, "output_text")
        if isinstance(output_text, str) and output_text:
            content = output_text

    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": tool_calls,
    }


def tool_call_list(response: Any) -> list[Any]:
    tool_calls = response_message(response).get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def canonical_usage(instance: Any, response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    try:
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(
            usage,
            provider=provider_name(instance),
            api_mode=str(getattr(instance, "api_mode", "") or "").strip().lower(),
        )
        input_tokens = to_int(getattr(canonical, "input_tokens", 0))
        output_tokens = to_int(getattr(canonical, "output_tokens", 0))
        total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens
    except Exception:
        input_tokens = to_int(obj_get(usage, "input_tokens"))
        output_tokens = to_int(obj_get(usage, "output_tokens"))
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = to_int(obj_get(usage, "prompt_tokens"))
            output_tokens = to_int(obj_get(usage, "completion_tokens"))
        total_tokens = to_int(obj_get(usage, "total_tokens")) or (
            input_tokens + output_tokens
        )
        return input_tokens, output_tokens, total_tokens


def response_finish_reason(instance: Any, response: Any) -> str | None:
    if tool_call_list(response):
        return "tool_calls"

    api_mode = str(getattr(instance, "api_mode", "") or "").strip().lower()
    if api_mode == "codex_responses":
        status = obj_get(response, "status")
        incomplete_details = obj_get(response, "incomplete_details")
        incomplete_reason = obj_get(incomplete_details, "reason")
        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
            return "length"
        if status == "completed" and response_message(response).get("content"):
            return "stop"
        return None

    try:
        finish_reason = response.choices[0].finish_reason
    except Exception:
        finish_reason = None
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason
    return None


def step_finish_reason(instance: Any, response: Any) -> str:
    if tool_call_list(response):
        return "tool_calls"

    finish_reason = response_finish_reason(instance, response)
    if finish_reason in {"stop", "interrupt", "length"}:
        return finish_reason

    content = response_message(response).get("content")
    if content:
        return "stop"
    return "invalid_response"


def state(instance: Any) -> dict[str, Any]:
    current = getattr(instance, "_otel_hermes_state", None)
    if current is None:
        current = {
            "entry_span": None,
            "agent_cm": None,
            "agent_span": None,
            "current_step_cm": None,
            "current_step_span": None,
            "current_step_round": 0,
            "pending_step_finish_reason": None,
            "last_response_model": None,
            "last_response_id": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "first_token_ns": None,
            "active_llm_depth": 0,
        }
        setattr(instance, "_otel_hermes_state", current)
    return current


def clear_state(instance: Any) -> None:
    setattr(instance, "_otel_hermes_state", None)


def start_step(tracer, instance: Any, finish_step) -> None:
    current_state = state(instance)
    if current_state["current_step_cm"] is not None:
        finish_step(
            instance,
            current_state.get("pending_step_finish_reason") or "invalid_response",
        )

    round_number = current_state["current_step_round"] + 1
    attrs = {
        GEN_AI_OPERATION_NAME: GEN_AI_OP_REACT,
        GEN_AI_SPAN_KIND: GEN_AI_KIND_STEP,
        "gen_ai.react.round": round_number,
    }
    cm = tracer.start_as_current_span(
        "react step",
        attributes=attrs,
    )
    span = cm.__enter__()
    current_state["current_step_cm"] = cm
    current_state["current_step_span"] = span
    current_state["current_step_round"] = round_number
    current_state["pending_step_finish_reason"] = None
