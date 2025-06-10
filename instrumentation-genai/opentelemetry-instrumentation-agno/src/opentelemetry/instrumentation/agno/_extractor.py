import json
from typing import (
    Any,
    Iterable,
    Tuple,
    Dict,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

class AgentRunRequestExactor(object):

    def extract(self, agent : Any, arguments : Dict[Any, Any]) -> Iterable[Tuple[str, AttributeValue]]:
        if agent.name:
            yield GenAIAttributes.GEN_AI_AGENT_NAME, f"{agent.name}"

        if agent.session_id:
            yield GenAIAttributes.GEN_AI_AGENT_ID, f"{agent.session_id}"

        if agent.knowledge:
            pass

        if agent.tools:
            tool_names = []
            from agno.tools.toolkit import Toolkit
            from agno.tools.function import Function
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield GenAIAttributes.GEN_AI_TOOL_NAME, ", ".join(tool_names)
        
        for key in arguments.keys():
            if key == "run_response":
                yield GenAIAttributes.GEN_AI_RESPONSE_ID, f"{arguments[key].run_id}"
            elif key == "run_messages":
                messages = arguments[key].messages
                for idx in range(len(messages)):  
                    message = messages[idx]
                    yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message", f"{json.dumps(message.to_dict(), indent=2)}"
            elif key == "session_id":
                pass
            elif key == "user_id":
                pass
            elif key == "response_format":
                yield GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT, f"{arguments[key]}"

class AgentRunResponseExactor(object):

    def extract(self, response : Any) -> Iterable[Tuple[str, AttributeValue]]:
        yield GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, f"{response.to_json()}"

class FunctionCallRequestExactor(object):

    def extract(self, function_call : Any) -> Iterable[Tuple[str, AttributeValue]]:

        if function_call.function.name:
            yield GenAIAttributes.GEN_AI_TOOL_NAME, f"{function_call.function.name}"

        if function_call.function.description:
            yield GenAIAttributes.GEN_AI_TOOL_DESCRIPTION, f"{function_call.function.description}"

        if function_call.call_id:
            yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{function_call.call_id}"

        if function_call.arguments:
            yield "gen_ai.tool.arguments", f"{json.dumps(function_call.arguments, indent=2)}"

class FunctionCallResponseExactor(object):

    def extract(self, response : Any) -> Iterable[Tuple[str, AttributeValue]]:
        yield "gen_ai.tool.response", f"{response.result}"