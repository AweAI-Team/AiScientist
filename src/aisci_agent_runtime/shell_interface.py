"""
Shell Interface — replaces PaperBench's ComputerInterface.

Preserves the two-layer timeout pattern from PaperBench:
- Layer 1: shell ``timeout --signal=KILL`` kills the process
- Layer 2: ``subprocess.run(..., timeout=...)`` cancels the call if the
  container is unresponsive

Also provides file I/O methods (read/write/append/upload/download) that mirror
ComputerInterface's upload/download semantics.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import structlog

logger = structlog.stdlib.get_logger(component=__name__)

TIMEOUT_BUFFER = 30  # seconds beyond shell timeout before subprocess kills


@dataclass
class ShellResult:
    output: str
    exit_code: int


@runtime_checkable
class ComputerInterface(Protocol):
    def send_shell_command(self, cmd: str, timeout: int = 300) -> ShellResult:
        ...

    def read_file(self, path: str) -> str:
        ...

    def download(self, path: str) -> bytes:
        ...

    def write_file(self, path: str, content: str) -> None:
        ...

    def upload(self, data: bytes, path: str) -> None:
        ...

    def append_file(self, path: str, content: str) -> None:
        ...

    def file_exists(self, path: str) -> bool:
        ...


class ShellInterface:
    """
    Local shell execution interface — drop-in for ComputerInterface.

    This variant is for runtimes where the agent and the sandboxed workspace
    share the same process environment, so commands can be executed directly via
    subprocess. The two-layer timeout pattern from PaperBench
    ``send_shell_command_with_timeout`` is preserved.
    """

    def __init__(self, working_dir: str = "/home"):
        self.working_dir = working_dir

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #

    def send_shell_command(self, cmd: str, timeout: int = 300) -> ShellResult:
        """
        Execute *cmd* under ``bash -c`` with two-layer timeout.

        Layer 1 – ``timeout --signal=KILL <timeout> bash -c '<cmd>'`` –
        the shell kills the child after *timeout* seconds.

        Layer 2 – ``subprocess.run(..., timeout=timeout + TIMEOUT_BUFFER)`` –
        catches an unresponsive shell.
        """
        refusal = _refuse_broad_python_kill(cmd)
        if refusal is not None:
            logger.warning("refused dangerous shell command", cmd_preview=cmd[:120])
            return ShellResult(output=refusal, exit_code=1)

        wrapped = f"timeout --signal=KILL {timeout} bash -c {_shell_quote(cmd)}"
        try:
            result = subprocess.run(
                ["bash", "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout + TIMEOUT_BUFFER,
                cwd=self.working_dir,
            )
            output = result.stdout + result.stderr
            return ShellResult(output=output, exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            return ShellResult(
                output=(
                    f"ERROR: Command timed out (shell + subprocess) after {timeout}s. "
                    "The container may be unresponsive."
                ),
                exit_code=137,
            )
        except Exception as e:
            return ShellResult(output=f"ERROR: {e}", exit_code=1)

    def send_command(self, cmd: str, timeout: int = 300) -> ShellResult:
        return self.send_shell_command(cmd, timeout=timeout)

    # ------------------------------------------------------------------ #
    # File I/O  (mirrors ComputerInterface upload / download)
    # ------------------------------------------------------------------ #

    def read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def download(self, path: str) -> bytes:
        """Return raw bytes — ComputerInterface compat."""
        with open(path, "rb") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            os.chmod(path, 0o644)  # so host user can read when path is under volume-mounted LOGS_DIR
        except OSError:
            pass

    def upload(self, data: bytes, path: str) -> None:
        """Write raw bytes — ComputerInterface compat."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def append_file(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        try:
            os.chmod(path, 0o644)  # so host user can read when path is under volume-mounted LOGS_DIR
        except OSError:
            pass

    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)


def _shell_quote(s: str) -> str:
    """Single-quote a string for bash, escaping embedded single-quotes."""
    return "'" + s.replace("'", "'\\''") + "'"


# Commands like ``pkill -f python`` match the agent's own Python process and
# cause immediate SIGTERM (exit 143). Refuse broad kills and tell the model to
# use a narrow pattern or PID-based kill.
_REFUSE_BROAD_PYTHON_KILL_MSG = """\
REFUSED: This command would match the agent's own Python process and terminate the run (SIGTERM / exit 143).

The active agent loop and subagents are Python processes in this runtime. \
``pkill -f python`` / ``killall python`` / broad ``pkill python`` will kill them too.

To free GPU memory safely, use a **narrow** match or **PID** only, e.g.:
  • List GPU PIDs: nvidia-smi
  • Kill one training PID: kill <pid>
  • Kill by script name only: pkill -f train_siim.py   (or your script filename)
  • Avoid: pkill -f python, killall python, pkill python, pkill -9 python
"""


def _refuse_broad_python_kill(cmd: str) -> str | None:
    """
    Return a refusal message if *cmd* would broadly kill Python and thus the
    agent; otherwise return None.

    Catches: pkill python, pkill -9 python, pkill -f python, pkill -f "python",
    killall python, etc.
    (pkill -9 python was not caught by the original regex and killed the agent
    in jigsaw impl_006.)
    """
    if not cmd.strip():
        return None
    low = cmd.lower()
    # killall python
    if re.search(r"\bkillall\s+python\b", low):
        return _REFUSE_BROAD_PYTHON_KILL_MSG
    # pkill python, pkill -9 python, pkill -f python, pkill -SIGKILL python, etc.
    if re.search(r"\bpkill\s+(-\S+\s+)*python\b", low):
        return _REFUSE_BROAD_PYTHON_KILL_MSG
    # pkill -f "python..." / pkill -f '...python...' (pattern contains python)
    if re.search(r"\bpkill\s+(-\S+\s+)*-f\s+[^;|&\n]*\bpython\b", low):
        return _REFUSE_BROAD_PYTHON_KILL_MSG
    return None
