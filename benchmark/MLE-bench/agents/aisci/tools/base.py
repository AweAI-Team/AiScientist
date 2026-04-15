"""
Tool base class — schema, execution, and completion signals.

Key design choices:
- ``name()`` returns the function name
- ``execute()`` receives a ``ShellInterface`` instead of ``ComputerInterface``
- ``get_tool_schema()`` returns an OpenAI-format tool dict
  (Chat Completions API, *not* the Responses API)
- ``SubagentCompleteTool`` / ``SubagentCompleteSignal`` are the standard way
  for subagents to signal that they are done
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SubagentCompleteSignal(Exception):
    """Raised by SubagentCompleteTool to break the subagent loop."""

    def __init__(self, content: str, artifacts: dict[str, Any] | None = None):
        self.content = content
        self.artifacts = artifacts or {}
        super().__init__(content)


class Tool(ABC):
    """Abstract base for all MLE-Bench AI Scientist tools."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def execute(self, shell, **kwargs) -> str:
        """Execute the tool. ``shell`` is a ``ShellInterface``."""
        ...

    @abstractmethod
    def get_tool_schema(self) -> dict:
        """Return the OpenAI Chat-Completions tool definition dict."""
        ...


class SubagentCompleteTool(Tool):
    """Standard tool for a subagent to signal completion."""

    def name(self) -> str:
        return "subagent_complete"

    def execute(self, shell, content: str = "", **kwargs) -> str:  # noqa: ARG002
        raise SubagentCompleteSignal(
            content=content,
            artifacts=kwargs,
        )

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "subagent_complete",
                "description": (
                    "Signal that your task is complete. "
                    "Provide a concise summary of what was accomplished."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Summary of what was accomplished and key results",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        }
