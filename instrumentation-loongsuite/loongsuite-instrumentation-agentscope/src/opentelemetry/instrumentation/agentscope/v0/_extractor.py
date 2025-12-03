import json
from typing import (
    Any,
    Dict,
    Iterable,
    Tuple,
)

from agentscope.message.block import ToolResultBlock, ToolUseBlock
from agentscope.models.response import ModelResponse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.types import AttributeValue


class ModelRequestExtractor(object):
    def extract(
        self, agent: Any, arguments: Dict[Any, Any]
    ) -> Iterable[Tuple[str, AttributeValue]]:
        if agent.model_name:
            yield GenAIAttributes.GEN_AI_AGENT_NAME, f"{agent.model_name}"
        if agent.model_type:
            yield (
                GenAIAttributes.GEN_AI_AGENT_DESCRIPTION,
                f"{agent.model_type}",
            )

        for item in arguments.items():
            key, entry_value = item
            if key == "messages":
                messages = entry_value
                for idx in range(len(messages)):
                    message = messages[idx]
                    if "role" in message:
                        yield (
                            f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.role",
                            f"{message['role']}",
                        )
                    if "content" in message:
                        yield (
                            f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.content",
                            f"{message['content']}",
                        )

            if key == "tools":
                tools = entry_value
                for idx in range(len(tools)):
                    tool = tools[idx]
                    yield (
                        f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}",
                        f"{json.dumps(tool, indent=2)}",
                    )

            if key == "texts":
                if isinstance(entry_value, str):
                    yield GenAIAttributes.GEN_AI_PROMPT, f"{entry_value}"
                elif isinstance(entry_value, list):
                    for idx, text in enumerate(entry_value):
                        yield (
                            f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}",
                            f"{text}",
                        )

            if key == "prompt":
                yield GenAIAttributes.GEN_AI_PROMPT, f"{entry_value}"


class ModelResponseExtractor(object):
    def extract(
        self, response: ModelResponse
    ) -> Iterable[Tuple[str, AttributeValue]]:
        if response._text:
            yield (
                f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.text",
                response._text,
            )

        if response.image_urls:
            for idx, image_url in enumerate(response.image_urls):
                yield (
                    f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.image_urls.{idx}",
                    image_url,
                )

        if response.tool_calls:
            for idx, tool_call in enumerate(response.tool_calls):
                yield (
                    f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}",
                    f"{json.dumps(tool_call, indent=2)}",
                )


class ToolRequestExtractor(object):
    def extract(
        self, tool_call: ToolUseBlock
    ) -> Iterable[Tuple[str, AttributeValue]]:
        if tool_call["type"]:
            yield GenAIAttributes.GEN_AI_TOOL_TYPE, f"{tool_call['type']}"
        if tool_call["name"]:
            yield GenAIAttributes.GEN_AI_TOOL_NAME, f"{tool_call['name']}"
        if tool_call["id"]:
            yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{tool_call['id']}"
        if tool_call["input"]:
            yield (
                GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
                f"{json.dumps(tool_call['input'], indent=2)}",
            )


class ToolResponseExtractor(object):
    def extract(
        self, response: ToolResultBlock
    ) -> Iterable[Tuple[str, AttributeValue]]:
        if response["type"]:
            yield GenAIAttributes.GEN_AI_TOOL_TYPE, f"{response['type']}"
        if response["id"]:
            yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{response['id']}"
        if response["output"]:
            yield (
                GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
                f"{json.dumps(response['output'], indent=2)}",
            )
