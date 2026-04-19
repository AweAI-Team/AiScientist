"""
State Manager for AI Scientist Solver

This module provides logging tools for subagents to record their work:
- Implementation log: Track code changes
- Experiment log: Track experiment runs (success, failure, partial)

Design Philosophy:
- Logs provide history and traceability
- Main communication is via return values (SubagentOutput)
- Logs are supplementary for debugging and history tracking
- Format is flexible but includes automatic timestamps
"""

from __future__ import annotations

import shlex
from datetime import datetime

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool

# =============================================================================
# Log File Paths
# =============================================================================

IMPL_LOG_PATH = "/home/agent/impl_log.md"
EXP_LOG_PATH = "/home/agent/exp_log.md"


# =============================================================================
# Implementation Log Tool
# =============================================================================

class AddImplLogTool(Tool):
    """Tool for logging implementation changes."""

    def name(self) -> str:
        return "add_impl_log"

    async def execute(
        self,
        computer: ComputerInterface,
        summary: str,
        files_changed: str = "",
        commit_hash: str = "",
        details: str = "",
    ) -> str:
        """
        Add an implementation log entry.

        Args:
            computer: ComputerInterface
            summary: Brief summary of what was done
            files_changed: Files that were changed (optional)
            commit_hash: Git commit hash (optional)
            details: Additional details (optional, can be multi-line)

        Returns:
            Confirmation message
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build log entry
        entry_lines = [
            f"## [{now}] {summary}",
            "",
        ]

        if files_changed:
            entry_lines.append(f"**Files**: {files_changed}")
        if commit_hash:
            entry_lines.append(f"**Commit**: {commit_hash}")
        if details:
            entry_lines.append("")
            entry_lines.append(details)

        entry_lines.append("")
        entry_lines.append("---")
        entry_lines.append("")

        entry = "\n".join(entry_lines)

        # Ensure log file exists and append
        await self._ensure_log_exists(computer, IMPL_LOG_PATH, "Implementation Log")
        await self._append_to_log(computer, IMPL_LOG_PATH, entry)

        return f"Implementation log added: {summary}"

    async def _ensure_log_exists(self, computer: ComputerInterface, path: str, title: str) -> None:
        """Create log file if it doesn't exist."""
        result = await computer.send_shell_command(f"test -f {path} && echo exists || echo missing")
        if "missing" in result.output.decode("utf-8", errors="replace"):
            header = f"# {title}\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
            await computer.upload(header.encode("utf-8"), path)

    async def _append_to_log(self, computer: ComputerInterface, path: str, content: str) -> None:
        """Append content to log file using shell append (avoids full file rewrite)."""
        escaped = shlex.quote(content)
        await computer.send_shell_command(f"printf %s {escaped} >> {path}")

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Log implementation changes to /home/agent/impl_log.md.

Use this after making code changes to track what was done.
Automatically adds timestamp.""",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was implemented/changed",
                    },
                    "files_changed": {
                        "type": "string",
                        "description": "Files that were changed (e.g., 'src/model.py, src/train.py')",
                    },
                    "commit_hash": {
                        "type": "string",
                        "description": "Git commit hash if committed",
                    },
                    "details": {
                        "type": "string",
                        "description": "Additional details (can be multi-line)",
                    },
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# Experiment Log Tool
# =============================================================================

class AddExpLogTool(Tool):
    """Tool for logging experiment runs (success, failure, or partial)."""

    def name(self) -> str:
        return "add_exp_log"

    async def execute(
        self,
        computer: ComputerInterface,
        summary: str,
        status: str = "completed",
        metrics: str = "",
        error: str = "",
        diagnosis: str = "",
        log_path: str = "",
        details: str = "",
    ) -> str:
        """
        Add an experiment log entry.

        Args:
            computer: ComputerInterface
            summary: Brief summary of the experiment
            status: Status - "success", "partial", or "failed"
            metrics: Key metrics (e.g., "acc=0.85, loss=0.12")
            error: Error message if failed
            diagnosis: Root cause analysis if failed/partial
            log_path: Path to detailed experiment log
            details: Additional details (can be multi-line)

        Returns:
            Confirmation message
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Status icons
        status_icons = {
            "success": "✅",
            "partial": "🟡",
            "failed": "❌",
        }
        icon = status_icons.get(status, "❓")

        # Build log entry
        entry_lines = [
            f"## [{now}] {icon} {summary}",
            "",
            f"**Status**: {status}",
        ]

        if metrics:
            entry_lines.append(f"**Metrics**: {metrics}")
        if error:
            entry_lines.append(f"**Error**: {error}")
        if diagnosis:
            entry_lines.append("")
            entry_lines.append("**Diagnosis**:")
            entry_lines.append(diagnosis)
        if log_path:
            entry_lines.append(f"**Log**: {log_path}")
        if details:
            entry_lines.append("")
            entry_lines.append(details)

        entry_lines.append("")
        entry_lines.append("---")
        entry_lines.append("")

        entry = "\n".join(entry_lines)

        # Ensure log file exists and append
        await self._ensure_log_exists(computer, EXP_LOG_PATH, "Experiment Log")
        await self._append_to_log(computer, EXP_LOG_PATH, entry)

        return f"Experiment log added: {icon} {summary}"

    async def _ensure_log_exists(self, computer: ComputerInterface, path: str, title: str) -> None:
        """Create log file if it doesn't exist."""
        result = await computer.send_shell_command(f"test -f {path} && echo exists || echo missing")
        if "missing" in result.output.decode("utf-8", errors="replace"):
            header = f"# {title}\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
            await computer.upload(header.encode("utf-8"), path)

    async def _append_to_log(self, computer: ComputerInterface, path: str, content: str) -> None:
        """Append content to log file using shell append (avoids full file rewrite)."""
        escaped = shlex.quote(content)
        await computer.send_shell_command(f"printf %s {escaped} >> {path}")

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Log experiment execution to /home/agent/exp_log.md.

Use this after running experiments to track results.
Handles success, partial success, and failures.
Automatically adds timestamp.""",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the experiment",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["success", "partial", "failed"],
                        "description": "Experiment status",
                    },
                    "metrics": {
                        "type": "string",
                        "description": "Key metrics (e.g., 'acc=0.85, loss=0.12')",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if failed",
                    },
                    "diagnosis": {
                        "type": "string",
                        "description": "Root cause analysis if failed/partial",
                    },
                    "log_path": {
                        "type": "string",
                        "description": "Path to detailed experiment log file",
                    },
                    "details": {
                        "type": "string",
                        "description": "Additional details",
                    },
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            strict=False,
        )
