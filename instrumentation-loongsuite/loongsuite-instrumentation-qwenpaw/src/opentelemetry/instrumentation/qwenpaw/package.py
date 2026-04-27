# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Collection

from packaging.requirements import Requirement

_instruments = ()
_instruments_copaw = "copaw >= 0.1.0, <= 1.0.2"
_instruments_qwenpaw = "qwenpaw >= 1.1.0"
_instruments_any = (_instruments_qwenpaw, _instruments_copaw)
_runtime_targets = (
    (_instruments_qwenpaw, "qwenpaw", "qwenpaw.app.runner.runner"),
    (_instruments_copaw, "copaw", "copaw.app.runner.runner"),
)


def get_installed_instrumentation_dependencies() -> Collection[str]:
    installed: list[str] = []
    for requirement, distribution_name, _ in _runtime_targets:
        try:
            installed_version = version(distribution_name)
        except PackageNotFoundError:
            continue
        if Requirement(requirement).specifier.contains(installed_version):
            installed.append(requirement)
    return tuple(installed)


def get_installed_runner_modules() -> Collection[str]:
    modules: list[str] = []
    for requirement, distribution_name, module_name in _runtime_targets:
        try:
            installed_version = version(distribution_name)
        except PackageNotFoundError:
            continue
        if Requirement(requirement).specifier.contains(installed_version):
            modules.append(module_name)
    return tuple(modules)
