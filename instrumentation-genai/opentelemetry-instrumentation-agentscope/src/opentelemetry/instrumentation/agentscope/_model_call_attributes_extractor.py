from typing import (
    Any,
    Iterable,
    Mapping,
    Tuple,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agentscope.models.response import ModelResponse

class RequestKey:
    MESSAGES = "messages"
    STREAM = "stream"
    TOOLS = "tools"

REQUEST_KEY_LIST = [
    RequestKey.MESSAGES,
    RequestKey.STREAM,
    RequestKey.TOOLS,
]

class ResponseKey:
    TEXT = "text"
    EMBEDDING = "embedding"
    IMAGE_URLS = "image_urls"
    RAW = "raw"
    PARSED = "parsed"
    STEAM = "stream"
    TOOL_CALLS = "tool_calls"

RESPONSE_KEY_LIST = [
    ResponseKey.TEXT,
]

class MessageKey:
    ROLE = "role"
    CONTENT = "content"

class RequestAttributesExtractor(object):

    def extract(self, *args: Tuple[type, Any], **kwargs: Mapping[str, Any]) -> Iterable[Tuple[str, AttributeValue]]:
        request = kwargs
        for i in range(len(args)):
            request[REQUEST_KEY_LIST[i]] = args[i]
        for request_key in request:
            if request_key not in REQUEST_KEY_LIST:
                continue
            if request_key == RequestKey.MESSAGES:
                messages = request[request_key]
                for idx in range(len(messages)):
                    message = messages[idx]
                    if MessageKey.ROLE in message:
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.{MessageKey.ROLE}", f"{message[
                            MessageKey.ROLE]}"
                    if MessageKey.CONTENT in message:
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.{MessageKey.CONTENT}", f"{message[MessageKey.CONTENT]}"

            if request_key == RequestKey.STREAM:
                stream_val = request[request_key]
                yield GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT, stream_val
            if request_key == RequestKey.TOOLS:
                tools = request[request_key]
                for idx in range(len(tools)):
                    tool = tools[idx]
                    if 'function' in tool:
                        function = tool['function']
                        if 'name' in function:
                            yield f"{GenAIAttributes.GEN_AI_TOOL_NAME}.{idx}.tool.name", f"{function['name']}"
                        if 'description' in function:
                            yield f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}.tool.description", f"{function['description']}"
                        if 'parameters' in function:
                            yield f"{GenAIAttributes.GEN_AI_TOOL_NAME}.{idx}.tool.parameters", f"{function['parameters']}"


SPEC_MAP = {
    ResponseKey.TEXT: GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
}

class ResponseAttributesExtractor(object):

    def extract(self, reponse : ModelResponse) -> Iterable[Tuple[str, AttributeValue]]:
        for request_key in SPEC_MAP:
            if hasattr(reponse, request_key):
                if request_key == ResponseKey.TEXT:
                    prompt_val = getattr(reponse, request_key)
                    yield SPEC_MAP[request_key], prompt_val
                else:
                    yield SPEC_MAP[request_key], getattr(reponse, request_key)
