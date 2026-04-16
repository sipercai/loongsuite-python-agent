from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
HERMES_ROOT = REPO_ROOT / "hermes-agent"
INSTRUMENTATION_SRC = Path(__file__).resolve().parents[1] / "src"
EXAMPLES_DIR = Path(__file__).resolve().parent
LOCAL_MCP_SERVER = EXAMPLES_DIR / "local_demo_mcp_server.py"
DEMO_KB = EXAMPLES_DIR / "demo_knowledge_base.json"

for candidate in (str(HERMES_ROOT), str(INSTRUMENTATION_SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from run_agent import AIAgent
from hermes_state import SessionDB

_INSTRUMENTOR_PATH = (
    INSTRUMENTATION_SRC
    / "opentelemetry"
    / "instrumentation"
    / "hermes_agent"
    / "__init__.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "hermes_agent_local_instrumentation",
    _INSTRUMENTOR_PATH,
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
HermesAgentInstrumentor = _MODULE.HermesAgentInstrumentor


def _write_hermes_config(home: Path, *, enable_mcp: bool) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
    }
    if enable_mcp:
        config["mcp_servers"] = {
            "demo": {
                "command": sys.executable,
                "args": [str(LOCAL_MCP_SERVER)],
                "env": {
                    "HERMES_DEMO_KB_PATH": str(DEMO_KB),
                },
            }
        }
    (home / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def _reload_mcp_registry(enabled: bool) -> None:
    try:
        from tools.mcp_tool import discover_mcp_tools, shutdown_mcp_servers
    except ImportError:
        return

    shutdown_mcp_servers()
    if enabled:
        discover_mcp_tools()


def _build_agent(
    *,
    enabled_toolsets: list[str],
    hermes_home: Path,
    max_iterations: int = 4,
    enable_mcp: bool = False,
    session_db: SessionDB | None = None,
) -> AIAgent:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for the otelgui demo")

    os.environ["HERMES_HOME"] = str(hermes_home)
    _reload_mcp_registry(enable_mcp)

    return AIAgent(
        model="qwen-turbo",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        provider="dashscope",
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        enabled_toolsets=enabled_toolsets,
        max_iterations=max_iterations,
        session_db=session_db,
    )


def _build_file_tool_prompt(target_file: str) -> str:
    return (
        f"请务必调用 read_file 工具读取文件 {target_file}。"
        "禁止猜测或编造内容。"
        "读取完成后只回复文件原文，不要解释。"
    )


def _build_demo_prompt(target_file: str) -> str:
    return (
        "请务必调用 delegate_task，把任务委派给子agent。"
        f"子agent需要读取文件 {target_file} 的内容，并返回结果。"
        "你最后只回复子agent返回的文件内容，不要解释。"
    )


def _build_planning_prompt(target_file: str) -> str:
    return (
        "这是一个必须展示规划过程的多步任务。"
        "你必须先调用 todo 工具创建 3 个任务："
        "p1=读取文件，p2=提取 CHECKPOINT 行，p3=输出最终答案。"
        "创建时只能让 p1 处于 in_progress。"
        f"随后必须调用 read_file 读取文件 {target_file}。"
        "拿到内容后，再次调用 todo 把 p1/p2/p3 更新为 completed。"
        "最后只回复文件中以 CHECKPOINT= 开头的整行，不要解释，也不要输出 todo 内容。"
    )


def _build_mcp_time_prompt() -> str:
    return (
        "请务必调用 mcp_demo_get_current_time 工具，timezone 传 Asia/Shanghai。"
        "拿到工具结果后，只回复返回 JSON 里的 iso 字段，不要解释，禁止自己猜时间。"
    )


def _build_rag_prompt() -> str:
    return (
        "请务必调用 mcp_demo_search_briefing 工具，query 传 apollo telemetry。"
        "然后只回复工具结果里的 answer 字段，不要解释，也不要补充别的信息。"
    )


def _build_evaluation_prompt(candidate: str) -> str:
    return (
        "请务必调用 mcp_demo_grade_candidate 工具做评估。"
        "reference 传 ENTRY > AGENT > STEP hierarchy with tools under the active step。"
        f"candidate 传 {candidate}。"
        "rubric 传 exact_keyword_overlap。"
        "拿到结果后，只回复 verdict 字段，不要解释。"
    )


def _build_history_seed_prompt(anchor: str) -> str:
    return f"请记住这句话：{anchor}。你只回复：seeded"


def _build_history_search_prompt(anchor: str) -> str:
    return (
        "请务必调用 session_search 工具。"
        f"query 传 {anchor}。"
        "搜索后只回复你找到的锚点短语本身，不要解释。"
    )


def _trace_api_url_from_otlp(otlp_endpoint: str) -> str | None:
    parsed = urllib_parse.urlparse(otlp_endpoint)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urllib_parse.urlunparse((parsed.scheme, parsed.netloc, "/api/traces", "", "", ""))


def _trace_ui_url_from_otlp(otlp_endpoint: str, trace_id: str) -> str | None:
    parsed = urllib_parse.urlparse(otlp_endpoint)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urllib_parse.urlunparse((parsed.scheme, parsed.netloc, f"/traces/{trace_id}", "", "", ""))


def _fetch_trace_index(api_url: str | None) -> list[dict]:
    if not api_url:
        return []
    try:
        with urllib_request.urlopen(api_url, timeout=3) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        return data if isinstance(data, list) else []
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []


def _wait_for_new_trace(api_url: str | None, before_ids: set[str], *, timeout_s: float = 8.0) -> str | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        traces = _fetch_trace_index(api_url)
        new_items = [item for item in traces if item.get("traceId") not in before_ids]
        if new_items:
            return new_items[0].get("traceId")
        time.sleep(0.5)
    return None


def _run_single_scenario(
    *,
    scenario: str,
    tracer,
    tracer_provider: TracerProvider,
    otlp_endpoint: str,
) -> dict[str, str]:
    hermes_home = Path(tempfile.mkdtemp(prefix=f"hermes-otelgui-{scenario}-"))
    enable_mcp = scenario in {"mcp_time", "rag_mcp", "evaluation_pass", "evaluation_fail"}
    _write_hermes_config(hermes_home, enable_mcp=enable_mcp)

    session_db = SessionDB(hermes_home / "state.db")
    session_id = f"otelgui-{scenario}-{uuid.uuid4().hex[:8]}"
    demo_file = hermes_home / f"{session_id}.txt"
    demo_file.write_text(
        "title=otelgui demo\nCHECKPOINT=planning_trace_ok\npayload=otelgui_e2e_ok\n",
        encoding="utf-8",
    )

    if scenario == "delegate":
        prompt = _build_demo_prompt(str(demo_file))
        agent = _build_agent(
            enabled_toolsets=["delegation", "file"],
            hermes_home=hermes_home,
            max_iterations=4,
            session_db=session_db,
        )
    elif scenario == "file_tool":
        prompt = _build_file_tool_prompt(str(demo_file))
        agent = _build_agent(
            enabled_toolsets=["file"],
            hermes_home=hermes_home,
            max_iterations=4,
            session_db=session_db,
        )
    elif scenario == "planning":
        prompt = _build_planning_prompt(str(demo_file))
        agent = _build_agent(
            enabled_toolsets=["todo", "file"],
            hermes_home=hermes_home,
            max_iterations=6,
            session_db=session_db,
        )
    elif scenario == "mcp_time":
        prompt = _build_mcp_time_prompt()
        agent = _build_agent(
            enabled_toolsets=["demo"],
            hermes_home=hermes_home,
            max_iterations=4,
            enable_mcp=True,
            session_db=session_db,
        )
    elif scenario == "rag_mcp":
        prompt = _build_rag_prompt()
        agent = _build_agent(
            enabled_toolsets=["demo"],
            hermes_home=hermes_home,
            max_iterations=4,
            enable_mcp=True,
            session_db=session_db,
        )
    elif scenario == "evaluation_pass":
        prompt = _build_evaluation_prompt(
            "The trace keeps ENTRY > AGENT > STEP hierarchy with tools under the active step."
        )
        agent = _build_agent(
            enabled_toolsets=["demo"],
            hermes_home=hermes_home,
            max_iterations=4,
            enable_mcp=True,
            session_db=session_db,
        )
    elif scenario == "evaluation_fail":
        prompt = _build_evaluation_prompt(
            "The trace is just a flat list of spans without a step hierarchy."
        )
        agent = _build_agent(
            enabled_toolsets=["demo"],
            hermes_home=hermes_home,
            max_iterations=4,
            enable_mcp=True,
            session_db=session_db,
        )
    elif scenario == "history_search":
        anchor = "history_anchor_otelgui_ok"
        seed_agent = _build_agent(
            enabled_toolsets=[],
            hermes_home=hermes_home,
            max_iterations=1,
            session_db=session_db,
        )
        seed_agent.session_id = f"otelgui-history-seed-{uuid.uuid4().hex[:8]}"
        seed_agent._disable_streaming = True
        seed_agent.run_conversation(_build_history_seed_prompt(anchor))

        prompt = _build_history_search_prompt(anchor)
        agent = _build_agent(
            enabled_toolsets=["session_search"],
            hermes_home=hermes_home,
            max_iterations=3,
            session_db=session_db,
        )
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    agent.session_id = session_id
    agent._disable_streaming = True

    api_url = _trace_api_url_from_otlp(otlp_endpoint)
    before_ids = {item.get("traceId") for item in _fetch_trace_index(api_url) if item.get("traceId")}

    entry_attrs = {
        "gen_ai.operation.name": "enter",
        "gen_ai.span.kind": "ENTRY",
        "gen_ai.session.id": session_id,
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "content": prompt}],
            ensure_ascii=False,
        ),
    }

    try:
        with tracer.start_as_current_span(
            "enter_ai_application_system",
            attributes=entry_attrs,
        ) as entry_span:
            result = agent.run_conversation(prompt)
            entry_span.set_attribute(
                "gen_ai.output.messages",
                json.dumps(
                    [{"role": "assistant", "content": result["final_response"]}],
                    ensure_ascii=False,
                ),
            )

        tracer_provider.force_flush()
        trace_id = _wait_for_new_trace(api_url, before_ids)
        trace_url = _trace_ui_url_from_otlp(otlp_endpoint, trace_id) if trace_id else None
        return {
            "scenario": scenario,
            "session_id": session_id,
            "demo_file": str(demo_file),
            "hermes_home": str(hermes_home),
            "final_response": result["final_response"],
            "trace_id": trace_id or "",
            "trace_url": trace_url or "",
        }
    finally:
        with contextlib.suppress(Exception):
            session_db.close()
        if enable_mcp:
            _reload_mcp_registry(False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real Hermes Agent conversation and export traces to otelgui."
    )
    parser.add_argument(
        "--otlp-endpoint",
        default=os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:5173/v1/traces"),
        help="OTLP HTTP traces endpoint. Defaults to otelgui local endpoint.",
    )
    parser.add_argument(
        "--service-name",
        default="hermes-agent-e2e",
        help="OpenTelemetry service.name resource attribute.",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "delegate",
            "file_tool",
            "planning",
            "mcp_time",
            "rag_mcp",
            "evaluation_pass",
            "evaluation_fail",
            "history_search",
            "all",
        ),
        default="delegate",
        help="Which end-to-end telemetry scenario to run.",
    )
    args = parser.parse_args()

    resource = Resource.create(
        {
            "service.name": args.service_name,
            "deployment.environment": "local-demo",
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=args.otlp_endpoint))
    )
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer("hermes-agent-otelgui-demo")

    instrumentor = HermesAgentInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    scenarios = (
        [
            "file_tool",
            "delegate",
            "planning",
            "mcp_time",
            "rag_mcp",
            "evaluation_pass",
            "evaluation_fail",
            "history_search",
        ]
        if args.scenario == "all"
        else [args.scenario]
    )

    try:
        summaries = [
            _run_single_scenario(
                scenario=scenario,
                tracer=tracer,
                tracer_provider=tracer_provider,
                otlp_endpoint=args.otlp_endpoint,
            )
            for scenario in scenarios
        ]
    finally:
        instrumentor.uninstrument()
        tracer_provider.shutdown()

    print("Hermes otelgui demo completed.")
    print(f"otlp_endpoint={args.otlp_endpoint}")
    for summary in summaries:
        print("---")
        for key in (
            "scenario",
            "session_id",
            "demo_file",
            "hermes_home",
            "final_response",
            "trace_id",
            "trace_url",
        ):
            print(f"{key}={summary.get(key, '')}")
    print("Open otelgui UI and inspect the trace_url values above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
