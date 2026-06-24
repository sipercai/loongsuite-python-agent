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

"""Tests for DeepAgents instrumentor lifecycle."""

from __future__ import annotations

from opentelemetry.instrumentation.deepagents import DeepAgentsInstrumentor
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor


def _cleanup_instrumentors() -> None:
    for instrumentor in (
        DeepAgentsInstrumentor(),
        LangGraphInstrumentor(),
        LangChainInstrumentor(),
    ):
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_uninstrument_cleans_dependencies_it_instrumented():
    _cleanup_instrumentors()
    instrumentor = DeepAgentsInstrumentor()

    try:
        instrumentor.instrument()

        assert LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert LangGraphInstrumentor().is_instrumented_by_opentelemetry

        instrumentor.uninstrument()

        assert not instrumentor.is_instrumented_by_opentelemetry
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert not LangGraphInstrumentor().is_instrumented_by_opentelemetry
    finally:
        _cleanup_instrumentors()


def test_uninstrument_preserves_preinstrumented_dependencies():
    _cleanup_instrumentors()
    langchain_instrumentor = LangChainInstrumentor()
    langgraph_instrumentor = LangGraphInstrumentor()
    instrumentor = DeepAgentsInstrumentor()

    try:
        langchain_instrumentor.instrument()
        langgraph_instrumentor.instrument()

        instrumentor.instrument()
        instrumentor.uninstrument()

        assert langchain_instrumentor.is_instrumented_by_opentelemetry
        assert langgraph_instrumentor.is_instrumented_by_opentelemetry
    finally:
        _cleanup_instrumentors()
