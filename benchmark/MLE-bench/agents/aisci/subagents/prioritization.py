"""
Prioritization Subagent — orders tasks for the main agent.

Reads competition description and data analysis to produce
/home/agent/prioritized_tasks.md with P0/P1/P2/P3 task rankings.
"""

from __future__ import annotations

from typing import Any

from subagents.base import Subagent, SubagentConfig
from subagents.configs import (
    DEFAULT_PRIORITIZATION_CONFIG,
    PRIORITIZATION_BASH_DEFAULT_TIMEOUT,
    PRIORITIZATION_BASH_MAX_TIMEOUT,
)
from tools.base import Tool, SubagentCompleteTool
from tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
    FileEditTool,
)
from prompts.templates import PRIORITIZATION_SYSTEM_PROMPT


class PrioritizationSubagent(Subagent):
    """
    Creates a prioritised task list for the competition.

    Input:  /home/data/description.md, /home/agent/analysis/summary.md
    Output: /home/agent/prioritized_tasks.md
    """

    def __init__(self, shell, llm, config=None):
        super().__init__(shell, llm, config or DEFAULT_PRIORITIZATION_CONFIG)

    @property
    def name(self) -> str:
        return "prioritization"

    def system_prompt(self) -> str:
        return PRIORITIZATION_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            BashToolWithTimeout(
                default_timeout=PRIORITIZATION_BASH_DEFAULT_TIMEOUT,
                max_timeout=PRIORITIZATION_BASH_MAX_TIMEOUT,
            ),
            PythonTool(
                default_timeout=PRIORITIZATION_BASH_DEFAULT_TIMEOUT,
                max_timeout=PRIORITIZATION_BASH_MAX_TIMEOUT,
            ),
            ReadFileChunkTool(),
            SearchFileTool(),
            FileEditTool(),
            SubagentCompleteTool(),
        ]

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        artifacts["prioritization_complete"] = True
        artifacts["tasks_path"] = "/home/agent/prioritized_tasks.md"
        return raw_output, artifacts
