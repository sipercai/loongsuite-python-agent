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

"""
Thread-local storage utilities for Claude Agent SDK tracing.

This module provides thread-local storage for the parent invocation context,
which is used by hooks to maintain trace context when async context
propagation is broken (Claude's async event loop breaks OpenTelemetry context).
"""

import threading
from typing import Any, Optional

# Thread-local store for passing the parent invocation into hooks.
# Claude's async event loop by default breaks OpenTelemetry context propagation.
# The parent invocation is threaded via thread-local as a fallback.
_thread_local = threading.local()


def set_parent_invocation(invocation: Any) -> None:
    """Set the parent agent invocation in thread-local storage.

    Args:
        invocation: InvokeAgentInvocation or ExecuteToolInvocation instance
    """
    _thread_local.parent_invocation = invocation


def clear_parent_invocation() -> None:
    """Clear the parent invocation from thread-local storage."""
    if hasattr(_thread_local, "parent_invocation"):
        delattr(_thread_local, "parent_invocation")


def get_parent_invocation() -> Optional[Any]:
    """Get the parent invocation from thread-local storage.

    Returns:
        Parent invocation or None if not set
    """
    return getattr(_thread_local, "parent_invocation", None)
