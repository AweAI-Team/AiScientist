"""
Implementation Tool

This tool allows the Main Agent to invoke the Implementation Subagent.
It handles code implementation for specific tasks from the prioritized task list.

Usage Flow:
1. Main Agent calls implement(task_id, context)
2. Implementation Subagent reads paper analysis, executes the task
3. Returns summary to Main Agent
4. Detailed logs in impl_log.md (via add_impl_log tool)

Session Directory Structure:
Each implement() call creates a session directory:
  run_dir/subagent_logs/impl_001_20240115_143022/
    implementation_xxx.log
    env_setup_xxx.log
    resource_download_xxx.log
"""

from __future__ import annotations

import os
import shlex
from datetime import datetime

import blobfile as bf
import structlog
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.state_manager import EXP_LOG_PATH, IMPL_LOG_PATH
from paperbench.solvers.aiscientist.subagents.base import (
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_IMPLEMENTATION_CONFIG,
)
from paperbench.solvers.aiscientist.subagents.implementation import (
    ImplementationSubagent,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


class ImplementationTool(Tool):
    """
    Tool for Main Agent to invoke Implementation Subagent.

    The Implementation Subagent:
    - Reads paper analysis and prioritized tasks
    - Reads directives from plan.md
    - Writes code to /home/submission
    - Can spawn env_setup and resource_download subagents
    - Updates impl_log.md with changelog
    - Commits changes to git
    """

    # Set by solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    run_dir: str | None = None
    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "implement"

    async def execute(
        self,
        computer: ComputerInterface,
        task: str = "",
        context: str = "",
        time_budget: int | None = None,
        mode: str = "full",
    ) -> str:
        """
        Execute implementation for a specific task.

        Args:
            computer: ComputerInterface
            task: Task description (used in fix mode for specific directives; ignored in full mode)
            context: Additional context (e.g., previous experiment feedback, issues to fix)
            time_budget: Time budget in seconds (if not specified, a default is used based on mode)
            mode: "full" (work through prioritized_tasks.md) or "fix" (targeted fix)

        Returns:
            Summary of implementation results
        """
        ctx_logger = logger.bind(tool="implement", mode=mode, task=task[:50])
        ctx_logger.info(f"Starting implementation (mode={mode}): {task[:100]}")

        if self.completer_config is None:
            return "Error: ImplementationTool not properly configured."

        if mode not in ("full", "fix"):
            return f"Error: Invalid mode '{mode}'. Use 'full' or 'fix'."

        # Set default time budget based on mode
        if time_budget is None:
            time_budget = DEFAULT_IMPLEMENTATION_CONFIG.time_limit if mode == "full" else 7200

        # Build task description for subagent based on mode
        if mode == "full":
            task_description = """## Implementation Task (Full Scope)

### Your Task
Read `/home/agent/prioritized_tasks.md` and work through the tasks using the **breadth-first** strategy described in your system prompt:

1. **Phase 1 — Skeleton**: Create project structure, reproduce.sh skeleton, and basic scaffolding for ALL P0 tasks. Commit early.
2. **Phase 2 — Core Implementation**: Fill in real logic for P0 tasks in priority order. For each: implement → test → git_commit → next
3. **Phase 3 — Remaining Tasks** (if time permits): Work through P1 → P2 tasks

You have the full session to work autonomously through as many tasks as possible.

"""
        else:  # mode == "fix"
            task_description = f"""## Implementation Task (Fix Mode)

### Fix the specific issues below
{task if task else "See context for details."}

"""

        if context:
            task_description += f"""### Context from Main Agent
{context}

"""

        # Create session directory for this impl call (need count for log separator)
        session_dir, session_count = self._create_session_dir()
        ctx_logger.info(f"Impl session directory: {session_dir}")

        # Write session separator to impl_log
        try:
            separator = f"\n=== Implement Session {session_count} ===\n\n"
            escaped_sep = shlex.quote(separator)
            await computer.send_shell_command(
                f"touch {IMPL_LOG_PATH} && printf %s {escaped_sep} >> {IMPL_LOG_PATH}"
            )
        except Exception:
            pass  # Don't block on separator write failure

        # Inject recent experiment log for context continuity (read from last session separator)
        try:
            exp_log_cmd = (
                f"LAST_SEP=$(grep -n '^=== Experiment Session' {EXP_LOG_PATH} 2>/dev/null | tail -1 | cut -d: -f1); "
                f"if [ -n \"$LAST_SEP\" ]; then sed -n \"$LAST_SEP,\\$p\" {EXP_LOG_PATH}; "
                f"else cat {EXP_LOG_PATH} 2>/dev/null || echo '(no experiment log yet)'; fi"
            )
            exp_log_result = await computer.send_shell_command(exp_log_cmd)
            exp_log_content = exp_log_result.output.decode("utf-8", errors="replace").strip()
            if exp_log_content and exp_log_content != "(no experiment log yet)":
                task_description += f"""### Recent Experiment History (auto-injected, last session)
> Below is the latest experiment session from `exp_log.md`. Earlier sessions may exist — read the full file with `read_file_chunk("{EXP_LOG_PATH}")` if needed.
> **Important**: Cross-reference these logs with the actual code (`git log`, source files) to understand the current state.

{exp_log_content}

"""
            else:
                task_description += """### Experiment History
> No experiment has been run yet — this is the first round. Skip the "Assess current state" step and proceed directly to reading tasks.

"""
        except Exception:
            pass  # Don't block implementation on log read failure

        # Configure subagent with session-specific log_dir
        config = SubagentConfig(
            max_steps=DEFAULT_IMPLEMENTATION_CONFIG.max_steps,
            time_limit=time_budget or DEFAULT_IMPLEMENTATION_CONFIG.time_limit,
            reminder_freq=DEFAULT_IMPLEMENTATION_CONFIG.reminder_freq,
            log_dir=session_dir,  # Logs go to session directory
        )

        subagent = ImplementationSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
            session_dir=session_dir,  # Pass session_dir for child subagents
        )

        try:
            result = await subagent.run(
                computer=computer,
                task_description=task_description,
                constraints=self.constraints,
            )

            return self._format_result(result, task, mode)

        except Exception as e:
            ctx_logger.error(f"Implementation failed: {e}")
            return f"Error during implementation: {str(e)}"

    def _create_session_dir(self) -> tuple[str, int]:
        """
        Create a session directory for this impl call.

        Returns:
            Tuple of (session_dir_path, session_count)
        """
        if self.run_dir:
            base_dir = bf.join(self.run_dir, "subagent_logs")
        else:
            base_dir = "/tmp/subagent_logs"

        os.makedirs(base_dir, exist_ok=True)

        # Count existing impl sessions
        try:
            existing = [d for d in os.listdir(base_dir) if d.startswith("impl_")]
            count = len(existing) + 1
        except Exception:
            count = 1

        # Create session directory with count and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = bf.join(base_dir, f"impl_{count:03d}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        return session_dir, count

    def _format_result(self, result: SubagentOutput, task: str, mode: str) -> str:
        """Format the implementation result for Main Agent."""
        status_map = {
            SubagentStatus.COMPLETED: "✅",
            SubagentStatus.FAILED: "❌",
            SubagentStatus.TIMEOUT: "⏰",
        }
        status_icon = status_map.get(result.status, "❓")

        mode_label = "Full Scope" if mode == "full" else "Fix"
        # Truncate task for header
        task_short = task[:60] + "..." if len(task) > 60 else task
        header = f"[Implementation {status_icon} | {mode_label}] {task_short}"
        header += f" ({result.num_steps} steps, {result.runtime_seconds:.1f}s)"

        output_lines = [
            header,
            "",
            "## Summary",
            result.content,
            "",
        ]

        if result.status == SubagentStatus.COMPLETED:
            output_lines.extend([
                "## What's Next?",
                "- Run `run_experiment()` to validate reproduce.sh end-to-end",
                "- If experiment fails, use `implement(mode='fix', task='...', context='<diagnosis>')` to fix",
            ])
        elif result.status == SubagentStatus.FAILED:
            output_lines.extend([
                "## Error",
                result.error_message or "Unknown error",
                "",
                "## Recommendation",
                "- Review the error and decide how to proceed",
                "- Re-run with `implement(mode='fix', ...)` if targeted fix needed",
            ])
        elif result.status == SubagentStatus.TIMEOUT:
            output_lines.extend([
                "## Note",
                "Implementation timed out. Check if partial progress was made.",
                "- Review git log for any commits",
                "- Run `run_experiment()` to validate what was completed",
            ])

        if result.log_path:
            output_lines.extend([
                "",
                f"**Detailed log**: {result.log_path}",
            ])

        return "\n".join(output_lines)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Invoke the Implementation Subagent to write code.

**Two modes**:

1. `mode="full"` (default): The impl agent reads prioritized_tasks.md and works through P0→P1→P2 tasks autonomously. Use this for the main implementation round.
   ```
   implement(mode="full")
   ```

2. `mode="fix"`: The impl agent receives specific fix directives and applies targeted fixes. Use this after experiment failures.
   ```
   implement(mode="fix", task="Fix import error in model.py", context="Experiment showed: ModuleNotFoundError: No module named 'torchvision'")
   ```

**What it does**:
1. Reads paper analysis for implementation details
2. Sets up environment if needed
3. Downloads resources if needed
4. Writes code to /home/submission, uses quick_test to verify
5. Commits changes to git
6. Returns summary of what was done""",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["full", "fix"],
                        "description": "Implementation mode: 'full' (work through prioritized_tasks.md) or 'fix' (targeted fix)",
                        "default": "full",
                    },
                    "task": {
                        "type": "string",
                        "description": "What to fix (only needed for mode='fix', e.g., 'Fix import error in model.py')",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context (e.g., experiment diagnosis, error tracebacks)",
                    },
                    "time_budget": {
                        "type": "integer",
                        "description": "Time budget in seconds. If not specified, a default is used based on mode.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )
