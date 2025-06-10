import os
import pytest

def pytest_configure(config: pytest.Config):
    # 尝试获取环境变量
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if api_key is None:
        pytest.exit("Environment variable 'DEEPSEEK_API_KEY' is not set. Aborting tests.")
    else:
        # 将环境变量保存到全局配置中，以便后续测试使用
        config.option.api_key = api_key