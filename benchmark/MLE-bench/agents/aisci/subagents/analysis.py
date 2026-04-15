"""
Data Analysis Subagent.

Analyses competition data and produces /home/agent/analysis/summary.md
which is referenced by all subsequent subagents.
"""

from __future__ import annotations

from typing import Any

from subagents.base import Subagent, SubagentConfig
from subagents.configs import (
    DEFAULT_ANALYSIS_CONFIG,
    ANALYSIS_BASH_DEFAULT_TIMEOUT,
    ANALYSIS_BASH_MAX_TIMEOUT,
)
from tools.base import Tool, SubagentCompleteTool
from tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
    FileEditTool,
)
from prompts.templates import ANALYSIS_SYSTEM_PROMPT


class DataAnalysisSubagent(Subagent):
    """
    Analyses competition data and produces a structured summary.

    Output: /home/agent/analysis/summary.md
    """

    def __init__(self, shell, llm, config=None):
        super().__init__(shell, llm, config or DEFAULT_ANALYSIS_CONFIG)

    @property
    def name(self) -> str:
        return "analysis"

    def system_prompt(self) -> str:
        return ANALYSIS_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            BashToolWithTimeout(
                default_timeout=ANALYSIS_BASH_DEFAULT_TIMEOUT,
                max_timeout=ANALYSIS_BASH_MAX_TIMEOUT,
            ),
            PythonTool(
                default_timeout=ANALYSIS_BASH_DEFAULT_TIMEOUT,
                max_timeout=ANALYSIS_BASH_MAX_TIMEOUT,
            ),
            ReadFileChunkTool(),
            SearchFileTool(),
            FileEditTool(),
            SubagentCompleteTool(),
        ]

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        artifacts["analysis_complete"] = True
        artifacts["summary_path"] = "/home/agent/analysis/summary.md"
        return raw_output, artifacts
