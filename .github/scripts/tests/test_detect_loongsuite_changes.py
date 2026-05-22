from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "detect_loongsuite_changes.py"
SPEC = importlib.util.spec_from_file_location(
    "detect_loongsuite_changes",
    SCRIPT_PATH,
)
detect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect)


def _run_detector(monkeypatch, tmp_path, changed_files=None, event=None):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    if changed_files is None:
        monkeypatch.delenv("LOONGSUITE_CHANGED_FILES", raising=False)
    else:
        monkeypatch.setenv(
            "LOONGSUITE_CHANGED_FILES",
            "\n".join(changed_files),
        )

    if event is not None:
        event_path = tmp_path / "event.json"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    else:
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)

    assert detect.main() == 0

    return dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )


def test_scopes_single_package_change(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/src/package.py",
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/tests/test_package.py",
        ],
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "|loongsuite-instrumentation-crewai|"
    assert outputs["degraded"] == "false"


def test_util_genai_change_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        ["util/opentelemetry-util-genai/src/opentelemetry/util/genai/foo.py"],
    )

    assert outputs["full"] == "true"
    assert outputs["degraded"] == "false"


def test_release_label_runs_full_suite(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/loongsuite-instrumentation-crewai/foo.py"
        ],
        {"pull_request": {"labels": [{"name": "prepare-release"}]}},
    )

    assert outputs["full"] == "true"
    assert outputs["reason"] == "release or backport pull request"


def test_unknown_package_falls_back_to_full(monkeypatch, tmp_path, capsys):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-newpkg/src/package.py"
        ],
    )

    captured = capsys.readouterr()
    assert outputs["full"] == "true"
    assert "loongsuite-instrumentation-newpkg" in outputs["reason"]
    assert "::error::" in captured.err


def test_empty_changed_files_is_degraded_full_run(monkeypatch, tmp_path):
    outputs = _run_detector(monkeypatch, tmp_path, [])

    assert outputs["full"] == "true"
    assert outputs["degraded"] == "true"
    assert outputs["reason"] == "unable to compute changed files"


def test_package_docs_only_change_skips_package(monkeypatch, tmp_path):
    outputs = _run_detector(
        monkeypatch,
        tmp_path,
        [
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/README.md",
            "instrumentation-loongsuite/"
            "loongsuite-instrumentation-crewai/CHANGELOG.md",
        ],
    )

    assert outputs["full"] == "false"
    assert outputs["packages"] == "||"


def test_unexpected_detector_error_falls_back_to_full(monkeypatch, tmp_path):
    def broken_changed_files(_event):
        raise KeyError("boom")

    monkeypatch.setattr(detect, "_changed_files", broken_changed_files)
    outputs = _run_detector(monkeypatch, tmp_path, None)

    assert outputs["full"] == "true"
    assert outputs["degraded"] == "true"
    assert "unexpected detector failure" in outputs["reason"]
