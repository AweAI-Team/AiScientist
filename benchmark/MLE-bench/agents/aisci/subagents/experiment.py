"""
Experiment Subagent — runs and evaluates experiments in the workspace.

Key design preserved:
- End-to-end validation of the solution output (submission.csv instead of reproduce.sh)
- Metric collection and comparison
- Failure diagnosis with actionable root-cause analysis
- Can apply trivial fixes and re-run
- Records results to exp_log.md

MLE-Bench adaptations:
- Validates submission.csv format against sample_submission.csv
- Kaggle-specific error diagnosis (data loading, submission format, etc.)
"""

from __future__ import annotations

from typing import Any

from subagents.base import Subagent, SubagentConfig
from subagents.configs import (
    DEFAULT_EXPERIMENT_CONFIG,
    EXPERIMENT_BASH_DEFAULT_TIMEOUT,
    EXPERIMENT_BASH_MAX_TIMEOUT,
    EXPERIMENT_COMMAND_TIMEOUT,
)
from tools.base import Tool, SubagentCompleteTool
from tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
    FileEditTool,
    GitCommitTool,
    ExecCommandTool,
    AddExpLogTool,
)
from prompts.templates import experiment_system_prompt_for_run


class ExperimentSubagent(Subagent):
    """
    Experiment subagent for running and validating Kaggle solutions.

    Responsibilities:
    - Run training and inference code
    - Validate submission.csv format and completeness
    - Diagnose failures with actionable suggestions
    - Apply trivial fixes and re-run
    - Record results to exp_log.md
    """

    def __init__(self, shell, llm, config=None, *, file_as_bus: bool = True):
        super().__init__(shell, llm, config or DEFAULT_EXPERIMENT_CONFIG)
        self._file_as_bus = file_as_bus

    @property
    def name(self) -> str:
        return "experiment"

    def system_prompt(self) -> str:
        return experiment_system_prompt_for_run(self._file_as_bus)

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = [
            # Information gathering
            ReadFileChunkTool(),
            SearchFileTool(),

            # Execution
            BashToolWithTimeout(
                default_timeout=EXPERIMENT_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),
            PythonTool(
                default_timeout=EXPERIMENT_BASH_DEFAULT_TIMEOUT,
                max_timeout=EXPERIMENT_BASH_MAX_TIMEOUT,
            ),
            ExecCommandTool(
                default_timeout=EXPERIMENT_COMMAND_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),

            # Fixing & committing (for trivial fixes)
            FileEditTool(),
            GitCommitTool(),
        ]
        if self._file_as_bus:
            tools.append(AddExpLogTool())
        tools.append(SubagentCompleteTool())
        return tools

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        artifacts["experiment_complete"] = True
        return raw_output, artifacts
