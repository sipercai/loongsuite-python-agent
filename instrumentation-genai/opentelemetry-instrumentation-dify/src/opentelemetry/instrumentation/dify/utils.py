import time
from datetime import datetime


def get_timestamp_from_datetime_attr(obj, attr_name):
    attr_value = getattr(obj, attr_name, None)
    if attr_value is not None and isinstance(attr_value, datetime):
        timestamp = attr_value.timestamp() * 1_000_000_000 + attr_value.microsecond * 1_000
        return int(timestamp)
    return time.time_ns()

def get_llm_common_attributes() -> dict:
    return {"callType": "gen_ai",
            "callKind": "custom_entry"}