"""
Shell-based tools for file, search, edit, and command execution.

Patterns preserved:
- BashToolWithTimeout: two-layer timeout, 50 K char output truncation
- FileEditTool: create / str_replace / insert modes with error messages
- GitCommitTool: auto-.gitignore management, temp-file commit messages
- ExecCommandTool: experiment logging with output truncation, exit-137 detection
- AddImplLogTool / AddExpLogTool: append-only markdown logs with timestamps
"""

from __future__ import annotations

import os
import shlex
import time
from typing import Any

from tools.base import Tool

OUTPUT_LIMIT = 50_000
HEAD_TAIL = 24_000


def _truncate_output(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Truncate long output keeping head + tail."""
    if len(text) <= limit:
        return text
    half = HEAD_TAIL
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - 2 * half} chars] ...\n\n"
        + text[-half:]
    )


# ====================================================================== #
# BashToolWithTimeout
# ====================================================================== #

class BashToolWithTimeout(Tool):
    """
    Execute a bash command with configurable timeout.

    Two-layer timeout pattern:
    - Layer 1: shell ``timeout --signal=KILL``
    - Layer 2: subprocess timeout (handled by ShellInterface)
    """

    def __init__(self, default_timeout: int = 300, max_timeout: int = 3600):
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout

    def name(self) -> str:
        return "bash"

    def execute(self, shell, command: str, timeout: int | None = None, **kwargs) -> str:
        if timeout is None:
            timeout = self.default_timeout
        timeout = min(int(timeout), self.max_timeout)

        result = shell.send_command(command, timeout=timeout)
        output = _truncate_output(result.output)

        if result.exit_code == 137:
            output += (
                f"\n\n⚠ Command was killed (exit 137) — likely exceeded the "
                f"{timeout}s timeout.\nSuggestions:\n"
                "- Break the command into smaller steps\n"
                "- Increase the timeout parameter\n"
                "- Check if the process is hanging"
            )

        if result.exit_code != 0:
            output += f"\n[exit code: {result.exit_code}]"

        return output

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    f"Execute a bash command. Default timeout {self.default_timeout}s, "
                    f"max {self.max_timeout}s.  Output is truncated to ~50 K chars."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": f"Timeout in seconds (default {self.default_timeout}, max {self.max_timeout})",
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# PythonTool
# ====================================================================== #

class PythonTool(Tool):
    """Execute a Python snippet via ``python3 -c``."""

    def __init__(self, default_timeout: int = 300, max_timeout: int = 3600):
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout

    def name(self) -> str:
        return "python"

    def execute(self, shell, code: str, timeout: int | None = None, **kwargs) -> str:
        if timeout is None:
            timeout = self.default_timeout
        timeout = min(int(timeout), self.max_timeout)

        escaped = code.replace("'", "'\\''")
        cmd = f"python3 -c '{escaped}'"
        result = shell.send_command(cmd, timeout=timeout)
        return _truncate_output(result.output)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "python",
                "description": "Execute Python code via ``python3 -c``. Use for quick computations or checks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                        },
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# ReadFileChunkTool
# ====================================================================== #

class ReadFileChunkTool(Tool):
    """Read a range of lines from a file (chunked read).

    Hard-caps output at OUTPUT_LIMIT chars (same 50 K limit as BashToolWithTimeout)
    to prevent a single wide CSV / large file from flooding the context window.
    When the raw output exceeds the limit, it is truncated head+tail with a
    clear marker so the model knows data was cut.
    """

    def name(self) -> str:
        return "read_file_chunk"

    def execute(self, shell, path: str, start_line: int = 1, num_lines: int = 500, **kwargs) -> str:
        # Cap num_lines to a sane maximum so the model cannot accidentally
        # request millions of lines and destroy the context.
        num_lines = min(int(num_lines), 2000)
        cmd = f"sed -n '{start_line},{start_line + num_lines - 1}p' {shlex.quote(path)}"
        result = shell.send_command(cmd, timeout=30)
        if result.exit_code != 0:
            return f"Error reading {path}: {result.output}"
        # Add line numbers
        lines = result.output.split("\n")
        numbered = [f"{start_line + i:6d} | {line}" for i, line in enumerate(lines)]
        output = "\n".join(numbered)
        # Apply the same 50 K char output cap used by BashToolWithTimeout.
        # This is the critical guard: without it a wide CSV (e.g. sample_submission
        # with 120 columns) can return 1 M+ chars and immediately fill the context.
        return _truncate_output(output)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "read_file_chunk",
                "description": (
                    "Read lines from a file with line numbers.  "
                    "Defaults to the first 500 lines (max 2000 lines per call).  "
                    "Output is capped at 50 000 chars; use start_line to page through large files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "start_line": {"type": "integer", "description": "Starting line (1-indexed, default 1)"},
                        "num_lines": {"type": "integer", "description": "Number of lines to read (default 500, max 2000)"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# SearchFileTool
# ====================================================================== #

class SearchFileTool(Tool):
    """Search files with ``grep``."""

    def name(self) -> str:
        return "search_file"

    def execute(self, shell, pattern: str, path: str = ".", include: str = "", **kwargs) -> str:
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)}"
        if include:
            cmd = f"grep -rn --include={shlex.quote(include)} {shlex.quote(pattern)} {shlex.quote(path)}"
        result = shell.send_command(cmd, timeout=60)
        return _truncate_output(result.output) if result.output.strip() else "No matches found."

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_file",
                "description": "Search for a pattern in files using grep.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "Directory or file to search (default '.')"},
                        "include": {"type": "string", "description": "File glob filter, e.g. '*.py'"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# FileEditTool
# ====================================================================== #

class FileEditTool(Tool):
    """
    Create / edit files in the workspace.

    Modes:
    - ``create``:      Write content to a new (or existing) file.
    - ``str_replace``: Replace the first occurrence of *old_str* with *new_str*.
    - ``insert``:      Insert *new_str* after line *insert_line*.
    """

    def name(self) -> str:
        return "edit_file"

    def execute(
        self,
        shell,
        command: str,
        path: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        **kwargs,
    ) -> str:
        if command == "create":
            shell.write_file(path, file_text)
            lines = file_text.count("\n") + 1
            return f"Created {path} ({lines} lines)"

        if command == "str_replace":
            if not old_str:
                return "Error: old_str is required for str_replace"
            if not os.path.exists(path):
                return f"Error: {path} does not exist"

            content = shell.read_file(path)
            count = content.count(old_str)
            if count == 0:
                return (
                    f"Error: old_str not found in {path}. "
                    "Double-check the exact text (including whitespace/indentation). "
                    "Use read_file_chunk to verify."
                )
            if count > 1:
                return (
                    f"Error: old_str appears {count} times in {path}. "
                    "Provide more surrounding context to make the match unique."
                )
            content = content.replace(old_str, new_str, 1)
            shell.write_file(path, content)
            return f"Replaced in {path}"

        if command == "insert":
            if not os.path.exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            lines = content.split("\n")
            idx = max(0, min(insert_line, len(lines)))
            lines.insert(idx, new_str)
            shell.write_file(path, "\n".join(lines))
            return f"Inserted at line {idx} in {path}"

        return f"Error: unknown command '{command}'. Use create / str_replace / insert."

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Create or edit a file.  Three modes:\n"
                    "- create: write file_text to path (creates parent dirs)\n"
                    "- str_replace: replace old_str with new_str (must be unique)\n"
                    "- insert: insert new_str after insert_line"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["create", "str_replace", "insert"],
                            "description": "The edit mode",
                        },
                        "path": {"type": "string", "description": "File path"},
                        "file_text": {"type": "string", "description": "Full file content (for 'create')"},
                        "old_str": {"type": "string", "description": "Text to find (for 'str_replace')"},
                        "new_str": {"type": "string", "description": "Replacement text (for 'str_replace' / 'insert')"},
                        "insert_line": {"type": "integer", "description": "Line number to insert after (for 'insert')"},
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# GitCommitTool
# ====================================================================== #

_DEFAULT_GITIGNORE = """\
# Auto-managed by AI Scientist
venv/
.venv/
__pycache__/
*.pyc
*.egg-info/
models/
data/
*.ckpt
*.pt
*.pth
*.h5
*.bin
*.safetensors
wandb/
.cache/
"""


class GitCommitTool(Tool):
    """Git add + commit with auto-.gitignore management."""

    def name(self) -> str:
        return "git_commit"

    def execute(self, shell, message: str, **kwargs) -> str:
        repo = "/home/code"

        # Ensure git is initialised
        shell.send_command(f"cd {repo} && git init 2>/dev/null", timeout=10)

        # Ensure .gitignore exists
        gitignore_path = f"{repo}/.gitignore"
        if not shell.file_exists(gitignore_path):
            shell.write_file(gitignore_path, _DEFAULT_GITIGNORE)

        # Write commit message to a temp file to avoid shell escaping issues
        msg_path = "/tmp/_commit_msg.txt"
        shell.write_file(msg_path, message)

        # Stage & commit
        result = shell.send_command(
            f"cd {repo} && git add -A && git commit -F {msg_path} 2>&1",
            timeout=60,
        )
        return _truncate_output(result.output)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": (
                    "Stage all changes in /home/code and commit with the given message.  "
                    "Automatically manages .gitignore to exclude large files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Commit message",
                        },
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# ExecCommandTool  (Experiment Subagent)
# ====================================================================== #

class ExecCommandTool(Tool):
    """
    Run a long-running experiment command with logging.

    Run a command in the experiment shell:
    - Output logged to ``/home/agent/experiments/[task_id]/[run_id].log``
    - ``set -o pipefail`` for accurate exit codes
    - Exit-137 detection (SIGKILL / timeout)
    - Output truncation (50 K chars)
    """

    def __init__(self, default_timeout: int = 3600, max_timeout: int = 7200):
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self._run_counter = 0

    def name(self) -> str:
        return "exec_command"

    def execute(
        self,
        shell,
        command: str,
        task_id: str = "default",
        timeout: int | None = None,
        **kwargs,
    ) -> str:
        if timeout is None:
            timeout = self.default_timeout
        timeout = min(int(timeout), self.max_timeout)

        self._run_counter += 1
        run_id = f"run_{self._run_counter:03d}"
        log_dir = f"/home/agent/experiments/{task_id}"
        log_path = f"{log_dir}/{run_id}.log"
        shell.send_command(f"mkdir -p {log_dir}", timeout=10)

        # Run with pipefail and tee to log
        wrapped = (
            f"set -o pipefail; "
            f"{{ {command} ; }} 2>&1 | tee {shlex.quote(log_path)}"
        )
        t0 = time.time()
        result = shell.send_command(wrapped, timeout=timeout)
        duration = time.time() - t0

        output = _truncate_output(result.output)

        status_line = f"Exit code: {result.exit_code} | Duration: {duration:.0f}s | Log: {log_path}"
        if result.exit_code == 137:
            status_line += (
                f"\n⚠ Process killed (exit 137) — likely exceeded {timeout}s timeout. "
                "Consider: reduce dataset size, fewer epochs, or increase timeout."
            )

        return f"{status_line}\n\n{output}"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": (
                    "Run a long-running command (training, evaluation, etc.) "
                    "with output logged to a file.  "
                    f"Default timeout {self.default_timeout}s, max {self.max_timeout}s."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"},
                        "task_id": {"type": "string", "description": "Experiment task identifier (for log directory)"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# AddImplLogTool  (state_manager port)
# ====================================================================== #

class AddImplLogTool(Tool):
    """Append an entry to /home/agent/impl_log.md."""

    LOG_PATH = "/home/agent/impl_log.md"

    def name(self) -> str:
        return "add_impl_log"

    def execute(
        self,
        shell,
        summary: str,
        files_changed: str = "",
        commit_hash: str = "",
        details: str = "",
        **kwargs,
    ) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry_parts = [f"\n### [{ts}] {summary}\n"]
        if files_changed:
            entry_parts.append(f"**Files changed:** {files_changed}\n")
        if commit_hash:
            entry_parts.append(f"**Commit:** `{commit_hash}`\n")
        if details:
            entry_parts.append(f"\n{details}\n")
        entry = "\n".join(entry_parts)

        safe = shlex.quote(entry)
        shell.send_command(
            f"mkdir -p /home/agent && printf '%s' {safe} >> {self.LOG_PATH}",
            timeout=10,
        )
        return f"Logged to {self.LOG_PATH}"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "add_impl_log",
                "description": (
                    "Append an implementation progress entry to /home/agent/impl_log.md. "
                    "Call after each meaningful code change or commit."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "One-line summary of the change"},
                        "files_changed": {"type": "string", "description": "Comma-separated list of changed files"},
                        "commit_hash": {"type": "string", "description": "Git commit hash (if committed)"},
                        "details": {"type": "string", "description": "Additional details or context"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# AddExpLogTool  (state_manager port)
# ====================================================================== #

class AddExpLogTool(Tool):
    """Append an entry to /home/agent/exp_log.md."""

    LOG_PATH = "/home/agent/exp_log.md"
    _STATUS_ICONS = {"success": "✅", "partial": "🟡", "failed": "❌"}

    def name(self) -> str:
        return "add_exp_log"

    def execute(
        self,
        shell,
        summary: str,
        status: str = "failed",
        metrics: str = "",
        error: str = "",
        diagnosis: str = "",
        log_path: str = "",
        details: str = "",
        **kwargs,
    ) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        icon = self._STATUS_ICONS.get(status, "❓")

        entry_parts = [f"\n### {icon} [{ts}] {summary}\n"]
        entry_parts.append(f"**Status:** {status}\n")
        if metrics:
            entry_parts.append(f"**Metrics:** {metrics}\n")
        if error:
            entry_parts.append(f"**Error:** {error}\n")
        if diagnosis:
            entry_parts.append(f"**Diagnosis:** {diagnosis}\n")
        if log_path:
            entry_parts.append(f"**Log:** `{log_path}`\n")
        if details:
            entry_parts.append(f"\n{details}\n")
        entry = "\n".join(entry_parts)

        safe = shlex.quote(entry)
        shell.send_command(
            f"mkdir -p /home/agent && printf '%s' {safe} >> {self.LOG_PATH}",
            timeout=10,
        )
        return f"Logged to {self.LOG_PATH}"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "add_exp_log",
                "description": (
                    "Append an experiment result entry to /home/agent/exp_log.md. "
                    "Call after every experiment run with status and metrics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "One-line summary of the experiment"},
                        "status": {
                            "type": "string",
                            "enum": ["success", "partial", "failed"],
                            "description": "Experiment outcome",
                        },
                        "metrics": {"type": "string", "description": "Key metrics (e.g., accuracy, RMSE)"},
                        "error": {"type": "string", "description": "Error message if failed"},
                        "diagnosis": {"type": "string", "description": "Root-cause analysis"},
                        "log_path": {"type": "string", "description": "Path to the full experiment log"},
                        "details": {"type": "string", "description": "Additional details"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }
