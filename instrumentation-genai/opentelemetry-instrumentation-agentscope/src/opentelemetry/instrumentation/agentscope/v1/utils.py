# -*- coding: utf-8 -*-
"""Attributes processor for span attributes."""
import datetime
import enum
import inspect
import json
from dataclasses import is_dataclass
from typing import Any, AsyncGenerator
import aioitertools
from pydantic import BaseModel
from opentelemetry.trace import Span, StatusCode
from agentscope.message import Msg
from typing import TypeVar

T = TypeVar("T")



def _to_serializable(
    obj: Any,
) -> Any:
    """Convert an object to a JSON serializable type.

    Args:
        obj (`Any`):
            The object to be converted to JSON serializable.

    Returns:
        `Any`:
            The converted JSON serializable object
    """

    # Handle primitive types first
    if isinstance(obj, (str, int, bool, float, type(None))):
        res = obj

    elif isinstance(obj, (list, tuple, set, frozenset)):
        res = [_to_serializable(x) for x in obj]

    elif isinstance(obj, dict):
        res = {str(key): _to_serializable(val) for (key, val) in obj.items()}

    elif isinstance(obj, (Msg, BaseModel)) or is_dataclass(obj):
        res = repr(obj)

    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        res = repr(obj)

    elif isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        res = obj.isoformat()

    elif isinstance(obj, datetime.timedelta):
        res = obj.total_seconds()

    elif isinstance(obj, enum.Enum):
        res = _to_serializable(obj.value)

    else:
        res = str(obj)

    return res


def _serialize_to_str(value: Any) -> str:
    """Get input attributes

    Args:
        value (`Any`):
            The input value

    Returns:
        `str`:
            JSON serialized string of the input value
    """
    try:
        return json.dumps(value, ensure_ascii=False)

    except TypeError:
        return json.dumps(
            _to_serializable(value),
            ensure_ascii=False,
        )

async def _trace_async_generator_wrapper(
    res: AsyncGenerator[T, None],
    span: Span,
) -> AsyncGenerator[T, None]:
    """Trace the async generator output with OpenTelemetry.

    Args:
        res (`AsyncGenerator[T, None]`):
            The generator or async generator to be traced.
        span (`Span`):
            The OpenTelemetry span to be used for tracing.

    Yields:
        `T`:
            The output of the async generator.
    """
    import opentelemetry

    has_error = False

    try:
        last_chunk = None
        async for chunk in aioitertools.iter(res):
            last_chunk = chunk
            yield chunk

    except Exception as e:
        has_error = True
        span.set_status(StatusCode.ERROR, str(e))
        span.record_exception(e)
        raise e from None

    finally:
        if not has_error:
            # Set the last chunk as output
            span.set_attributes(
                {
                     "gen_ai.output.messages": _serialize_to_str(last_chunk),
                },
            )
            span.set_status(StatusCode.OK)
        span.end()