from enum import Enum

GEN_AI_USER_ID = "gen_ai.user.id"
GEN_AI_SESSION_ID = "gen_ai.session.id"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"

GEN_AI_SPAN_KIND = "gen_ai.span.kind"

GEN_AI_MODEL_NAME = ""
GEN_AI_REQUEST_MODEL_NAME = ""
RETRIEVAL_DOCUMENTS = ""
GEN_AI_INPUT_MESSAGES = ""
GEN_AI_OUTPUT_MESSAGES = ""
GEN_AI_USAGE_TOTAL_TOKENS = ""


class SpanKindValues(Enum):
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    RERANKER = "RERANKER"
    TASK = "TASK"
    UNKNOWN = "UNKNOWN"

class DocumentAttributes:
    """
    Attributes for a document
    """

    DOCUMENT_ID = "document.id"
    """
    The id of the document
    """
    DOCUMENT_SCORE = "document.score"
    """
    The score of the document
    """
    DOCUMENT_CONTENT = "document.content"
    """
    The content of the document
    """
    DOCUMENT_METADATA = "document.metadata"
    """
    The metadata of the document represented as a dictionary
    JSON string, e.g. `"{ 'title': 'foo' }"`
    """