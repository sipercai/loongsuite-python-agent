from typing import (
    Iterable,
    Tuple,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agentscope.message.block import ToolUseBlock, ToolResultBlock

class RequestKey:
    TYPE = "type"
    NAME = "name"
    ID = "id"
    INPUT = "input"

REQUEST_KEY_LIST = [
    RequestKey.TYPE,
    RequestKey.NAME,
    RequestKey.ID,
    RequestKey.INPUT,
]
class ResponseKey:
    TYPE = "type"
    NAME = "name"
    ID = "id"
    OUTPUT = "output"

RESPONSE_KEY_LIST = [
    ResponseKey.TYPE,
    ResponseKey.ID,
    ResponseKey.OUTPUT,
]

class ToolRequestAttributesExtractor(object):

    def extract(self, tool_call: ToolUseBlock) -> Iterable[Tuple[str, AttributeValue]]:
        for key in REQUEST_KEY_LIST:
            if key == RequestKey.TYPE:
                yield GenAIAttributes.GEN_AI_TOOL_TYPE, f"{tool_call['type']}"
            if key == RequestKey.NAME:
                yield GenAIAttributes.GEN_AI_TOOL_NAME, f"{tool_call['name']}"
            if key == RequestKey.ID:
                yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{tool_call['id']}"
            if key == RequestKey.INPUT:
                yield GenAIAttributes.GEN_AI_TOOL_DESCRIPTION, f"{tool_call['input']}"


class ToolResponseAttributesExtractor(object):

    def extract(self, reponse : ToolResultBlock) -> Iterable[Tuple[str, AttributeValue]]:
        for key in RESPONSE_KEY_LIST:
            if key == ResponseKey.TYPE:
                yield GenAIAttributes.GEN_AI_TOOL_TYPE, f"{reponse['type']}"
            if key == ResponseKey.ID:
                yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{reponse['id']}"
            if key == ResponseKey.OUTPUT:
                yield GenAIAttributes.GEN_AI_TOOL_DESCRIPTION, f"{reponse['output']}"
