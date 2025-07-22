import socket
from os import environ

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

def get_hostname():
    try:
        hostname = socket.gethostname()
        return hostname
    except socket.error as e:
        print(f"Unable to get hostname: {e}")


def get_ip_address():
    try:
        # 获取本地主机名
        hostname = socket.gethostname()
        # 获取本地IP
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.error as e:
        print(f"Unable to get IP Address: {e}")

def is_capture_content_enabled() -> bool:
    capture_content = environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "true"
    )
    return is_true_value(capture_content)

def convert_to_env_var(env_key: str) -> str:
    return env_key.replace(".", "_").upper()

def is_true_value(value) -> bool:
    return value.lower() in {"1", "y", "yes", "true"}