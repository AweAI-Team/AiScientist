"""
BashTool with configurable timeout.

A single reusable tool class used by the main agent, experiment subagent,
and env-setup subagent — each with different timeout values loaded from
their respective configs.

Delegates to ``send_shell_command_with_timeout`` which provides two-layer
protection: shell ``timeout --signal=KILL`` + ``asyncio.wait_for`` fallback.
"""

from __future__ import annotations

import asyncio

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.utils import send_shell_command_with_timeout


class BashToolWithTimeout(Tool):
    """
    BashTool with configurable per-command timeout and output truncation.

    The agent can optionally pass a ``timeout`` parameter per call; if omitted,
    ``default_timeout`` is used.  The effective timeout is always clamped to
    ``max_timeout``.

    Long command outputs are truncated to ``max_output_chars`` to prevent
    blowing up the LLM context window.  The truncation preserves the first
    ``keep_front_chars`` and last ``keep_back_chars`` characters, replacing
    the middle with a ``[truncated]`` marker.

    Usage examples::

        # env-setup: short timeouts for pip/apt
        BashToolWithTimeout(default_timeout=300, max_timeout=600)

        # experiment: long timeouts for training
        BashToolWithTimeout(default_timeout=18000, max_timeout=21600)

        # main agent: cap to solver time_limit
        BashToolWithTimeout(default_timeout=18000, max_timeout=86400)
    """

    default_timeout: int = 300
    max_timeout: int = 3600

    # Output truncation settings (generous defaults to keep most output)
    max_output_chars: int = 50_000        # 50K chars total cap
    keep_front_chars: int = 24_000         # keep first 24K
    keep_back_chars: int = 24_000          # keep last 24K

    def name(self) -> str:
        return "bash"

    @staticmethod
    def _truncate_output(
        output: str,
        max_chars: int,
        keep_front: int,
        keep_back: int,
    ) -> str:
        """Truncate long output, preserving head and tail."""
        if len(output) <= max_chars:
            return output
        omitted = len(output) - keep_front - keep_back
        return (
            output[:keep_front]
            + f"\n\n... [truncated {omitted:,} characters out of {len(output):,} total] ...\n\n"
            + output[-keep_back:]
        )

    async def execute(
        self, computer: ComputerInterface, cmd: str, timeout: int | None = None
    ) -> str:
        """
        Execute a bash command with timeout.

        Args:
            cmd: The bash command to execute.
            timeout: Optional per-call timeout in seconds.  Clamped to
                ``max_timeout``.  Defaults to ``default_timeout``.

        Returns:
            Command output as string, or an error message on timeout.
        """
        if timeout is not None and timeout <= 0:
            timeout = None
        effective = self.default_timeout if timeout is None else timeout
        actual_timeout = min(effective, self.max_timeout)

        try:
            result = await send_shell_command_with_timeout(
                computer, cmd, timeout=actual_timeout
            )
        except asyncio.TimeoutError:
            return (
                f"ERROR: Command timed out due to network/system issue "
                f"after {actual_timeout}s. "
                f"The container may be unresponsive."
            )

        output = result.output.decode("utf-8", errors="replace").strip()

        # Truncate very long outputs to protect the LLM context window
        output = self._truncate_output(
            output, self.max_output_chars, self.keep_front_chars, self.keep_back_chars
        )

        # exit_code 137 = 128 + 9 (SIGKILL) means the shell timeout killed the process
        if result.exit_code == 137:
            return (
                f"ERROR: Command was killed after {actual_timeout}s timeout.\n"
                f"Consider breaking the command into smaller parts or "
                f"increasing timeout.\n"
                f"Partial output:\n{output}"
            )

        return output

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Execute a bash command with timeout protection. "
                "For long-running commands, you can specify a custom timeout."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Optional timeout in seconds "
                            f"(default: {self.default_timeout}s, "
                            f"max: {self.max_timeout}s). "
                            f"Use higher values for long-running tasks."
                        ),
                    },
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        )
