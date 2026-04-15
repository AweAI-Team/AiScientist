"""
Implementation Subagent — core coding agent for the competition workspace.

Key design:
- Autonomous task selection from prioritized_tasks.md (NOT directed by main agent)
- Two modes: "full" (breadth-first through all tasks) and "fix" (targeted fixes)
- Breadth-first strategy — Phase 1 skeleton → Phase 2 core → Phase 3 improvements
- Git commit early and often
- Dependency consistency self-check
- Time-aware reminders at 60% and 90%

MLE-Bench adaptations:
- submission.csv instead of reproduce.sh as the critical deliverable
- Kaggle-specific coding patterns (pandas, sklearn, torch for ML)
- /home/code/ as the working directory, /home/data/ for competition data
"""

from __future__ import annotations

from typing import Any

from subagents.base import Subagent, SubagentConfig
from subagents.configs import (
    DEFAULT_IMPLEMENTATION_CONFIG,
    IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
    IMPLEMENTATION_BASH_MAX_TIMEOUT,
)
from tools.base import Tool, SubagentCompleteTool
from tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
    FileEditTool,
    GitCommitTool,
    AddImplLogTool,
)
from prompts.templates import implementation_system_prompt_for_run


class ImplementationSubagent(Subagent):
    """
    Core implementation subagent.

    In "full" mode it reads prioritized_tasks.md and works autonomously
    (breadth-first: skeleton → core → improvements).

    In "fix" mode it receives specific directives and applies targeted fixes.
    """

    def __init__(self, shell, llm, config=None, *, file_as_bus: bool = True):
        super().__init__(shell, llm, config or DEFAULT_IMPLEMENTATION_CONFIG)
        self._file_as_bus = file_as_bus

    @property
    def name(self) -> str:
        return "implementation"

    def system_prompt(self) -> str:
        return implementation_system_prompt_for_run(self._file_as_bus)

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = [
            # Information gathering
            ReadFileChunkTool(),
            SearchFileTool(),

            # Code writing
            FileEditTool(),
            BashToolWithTimeout(
                default_timeout=IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
                max_timeout=IMPLEMENTATION_BASH_MAX_TIMEOUT,
            ),
            PythonTool(
                default_timeout=IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
                max_timeout=IMPLEMENTATION_BASH_MAX_TIMEOUT,
            ),

            # Git
            GitCommitTool(),
        ]
        if self._file_as_bus:
            tools.append(AddImplLogTool())
        tools.append(SubagentCompleteTool())
        return tools

    def _build_reminder(self, step: int, elapsed: float) -> str | None:
        """Enhanced reminder — 3-stage implementation reminder."""
        base = super()._build_reminder(step, elapsed)
        if base is None:
            return None

        pct = elapsed / self.config.time_limit * 100

        if pct >= 90:
            extras = (
                "🚨 URGENT: Almost out of time!\n"
                "- `git_commit` ALL uncommitted changes immediately\n"
                "- Ensure /home/submission/submission.csv exists and is valid\n"
                "- Call `subagent_complete` NOW"
            )
        elif pct >= 60:
            extras = (
                "⚠ Over 60% time used. Wrap up current work:\n"
                "- `git_commit` any uncommitted changes\n"
                "- Finish generating submission.csv for the current model\n"
                "- Verify submission.csv matches sample format before moving to next task"
            )
        else:
            extras = (
                "Reminders:\n"
                "- `git_commit` after each working component\n"
                "- Keep submission.csv updated — a valid simple submission beats an incomplete better one\n"
                "- Estimate execution time before long training runs (use small-scale test first)\n"
                "- Ensure requirements.txt includes ALL imports used in your code"
            )

        return base + "\n\n" + extras

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        artifacts["implementation_complete"] = True
        return raw_output, artifacts
