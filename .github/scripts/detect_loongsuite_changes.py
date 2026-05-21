#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import PurePosixPath

FULL_RUN_LABELS = {"prepare-release", "backport"}
FULL_RUN_PREFIXES = (
    ".github/scripts/detect_loongsuite_changes.py",
    ".github/workflows/generate_workflows_loongsuite.py",
    ".github/workflows/generate_workflows_lib/src/generate_workflows_lib/",
    ".github/workflows/loongsuite_",
    "loongsuite-distro/",
    "loongsuite-site-bootstrap/",
    "scripts/loongsuite/",
)
FULL_RUN_FILES = {
    ".pre-commit-config.yaml",
    ".pylintrc",
    "dev-requirements.txt",
    "eachdist.ini",
    "gen-requirements.txt",
    "pkg-requirements.txt",
    "pyproject.toml",
    "pytest.ini",
    "test-constraints.txt",
    "tox-loongsuite.ini",
    "tox-uv.toml",
    "uv.lock",
}
UTIL_GENAI_PREFIX = "util/opentelemetry-util-genai/"
LOONGSUITE_INSTRUMENTATION_PREFIX = "instrumentation-loongsuite/"


def _load_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return {}

    try:
        with open(event_path, encoding="utf-8") as event_file:
            return json.load(event_file)
    except OSError as exc:
        print(f"Unable to read GitHub event: {exc}", file=sys.stderr)
        return {}


def _run_git_diff(base_ref: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _changed_files(event: dict) -> list[str]:
    changed_files = os.environ.get("LOONGSUITE_CHANGED_FILES")
    if changed_files:
        return [
            line.strip()
            for line in changed_files.splitlines()
            if line.strip()
        ]

    pull_request = event.get("pull_request", {})
    base_sha = pull_request.get("base", {}).get("sha")
    if not base_sha:
        raise RuntimeError("pull request base SHA is unavailable")

    return _run_git_diff(base_sha)


def _has_full_run_label(event: dict) -> bool:
    pull_request = event.get("pull_request", {})
    labels = {
        label.get("name")
        for label in pull_request.get("labels", [])
        if label.get("name")
    }
    return bool(labels & FULL_RUN_LABELS)


def _is_release_pull_request(event: dict) -> bool:
    pull_request = event.get("pull_request", {})
    base_ref = pull_request.get("base", {}).get("ref", "")
    head_ref = pull_request.get("head", {}).get("ref", "")
    return base_ref.startswith("release/") or head_ref.startswith("release/")


def _package_for_path(path: str) -> str | None:
    normalized = path.strip("/")
    if not normalized.startswith(LOONGSUITE_INSTRUMENTATION_PREFIX):
        return None

    package_path = PurePosixPath(normalized)
    if len(package_path.parts) < 2:
        return None

    package = package_path.parts[1]
    if package.startswith("loongsuite-instrumentation-"):
        return package

    return None


def _requires_full_run(path: str) -> bool:
    normalized = path.strip("/")
    return (
        normalized in FULL_RUN_FILES
        or normalized.startswith(FULL_RUN_PREFIXES)
        or normalized.startswith(UTIL_GENAI_PREFIX)
    )


def _write_outputs(outputs: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    with open(output_path, "a", encoding="utf-8") as output_file:
        for name, value in outputs.items():
            output_file.write(f"{name}={value}\n")


def main() -> int:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = _load_event()

    full = event_name != "pull_request"
    packages: set[str] = set()
    reason = "non-pull-request event"

    if not full and (
        _has_full_run_label(event) or _is_release_pull_request(event)
    ):
        full = True
        reason = "release or backport pull request"

    if not full:
        try:
            changed_files = _changed_files(event)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            full = True
            reason = f"could not determine changed files: {exc}"
        else:
            for changed_file in changed_files:
                if _requires_full_run(changed_file):
                    full = True
                    reason = f"shared LoongSuite file changed: {changed_file}"
                    break

                package = _package_for_path(changed_file)
                if package:
                    packages.add(package)

            if not full:
                if packages:
                    reason = "package-scoped LoongSuite change"
                else:
                    reason = "no LoongSuite package changes"

    package_output = "|" + "|".join(sorted(packages)) + "|"
    outputs = {
        "full": str(full).lower(),
        "packages": package_output,
        "reason": reason.replace("\n", " "),
    }
    _write_outputs(outputs)

    print(f"full={outputs['full']}")
    print(f"packages={outputs['packages']}")
    print(f"reason={outputs['reason']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
