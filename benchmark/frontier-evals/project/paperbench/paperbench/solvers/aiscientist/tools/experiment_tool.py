"""
Experiment Tool

This tool allows the Main Agent to invoke the Experiment Subagent.
It runs experiments, validates results, and reports back with diagnostics.

Usage Flow:
1. Main Agent calls run_experiment(task, mode)
2. Experiment Subagent runs the experiment, collects results
3. Returns summary with status and diagnostics
4. If failed, records diagnosis in exp_log.md
"""

from __future__ import annotations

import os
import shlex
from datetime import datetime

import blobfile as bf
import structlog
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.state_manager import IMPL_LOG_PATH, EXP_LOG_PATH
from paperbench.solvers.aiscientist.subagents.base import (
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_EXPERIMENT_CONFIG,
    EXPERIMENT_VALIDATE_TIME_LIMIT,
)
from paperbench.solvers.aiscientist.subagents.experiment import (
    ExperimentSubagent,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


class ExperimentTool(Tool):
    """
    Tool for Main Agent to invoke Experiment Subagent.

    The Experiment Subagent:
    - Reads experiment configurations from paper analysis
    - Runs experiments (full or validation mode)
    - Validates results against expectations
    - Diagnoses failures
    - Updates exp_log.md with results
    """

    # Set by solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    run_dir: str | None = None
    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "run_experiment"

    async def execute(
        self,
        computer: ComputerInterface,
        task: str,
        mode: str = "full",
        context: str = "",
        time_budget: int | None = None,
    ) -> str:
        """
        Run experiment for a specific task.

        Args:
            computer: ComputerInterface
            task: What to validate/test (e.g., "P0-1: Validate transformer encoder outputs")
            mode: Experiment mode
                - "full": Run complete experiment end-to-end (training + evaluation)
                - "validate": Quick validation check (e.g., reproduce.sh smoke test)
            context: Additional context from Main Agent (e.g., previous results, specific concerns)
            time_budget: Time budget in seconds (default: ~10h for full, ~5h for validate)

        Returns:
            Summary of experiment results with diagnostics
        """
        ctx_logger = logger.bind(tool="run_experiment", task=task[:50], mode=mode)
        ctx_logger.info(f"Starting experiment: {task[:100]}")

        if self.completer_config is None:
            return "Error: ExperimentTool not properly configured."

        # Build task description for subagent
        # Note: The system prompt already covers workflow, workspace reference,
        # and diagnostic guidance. Here we only pass task-specific info.
        task_description = f"""## Experiment Task

### Your Task
{task}

### Mode: {mode.upper()}
{"- Full experiment run with complete validation" if mode == "full" else ""}
{"- Quick validation run (shorter time budget)" if mode == "validate" else ""}
"""

        if context:
            task_description += f"""
### Context from Main Agent
{context}
"""

        # Create session directory for this experiment call (need count for log separator)
        session_dir, session_count = self._create_session_dir()
        ctx_logger.info(f"Experiment session directory: {session_dir}")

        # Write session separator to exp_log
        try:
            separator = f"\n=== Experiment Session {session_count} ===\n\n"
            escaped_sep = shlex.quote(separator)
            await computer.send_shell_command(
                f"touch {EXP_LOG_PATH} && printf %s {escaped_sep} >> {EXP_LOG_PATH}"
            )
        except Exception:
            pass  # Don't block on separator write failure

        # Inject recent implementation log for context continuity (read from last session separator)
        try:
            impl_log_cmd = (
                f"LAST_SEP=$(grep -n '^=== Implement Session' {IMPL_LOG_PATH} 2>/dev/null | tail -1 | cut -d: -f1); "
                f"if [ -n \"$LAST_SEP\" ]; then sed -n \"$LAST_SEP,\\$p\" {IMPL_LOG_PATH}; "
                f"else cat {IMPL_LOG_PATH} 2>/dev/null || echo '(no implementation log yet)'; fi"
            )
            impl_log_result = await computer.send_shell_command(impl_log_cmd)
            impl_log_content = impl_log_result.output.decode("utf-8", errors="replace").strip()
            if impl_log_content and impl_log_content != "(no implementation log yet)":
                task_description += f"""
### Recent Implementation History (auto-injected, last session)
> Below is the latest implementation session from `impl_log.md`. Earlier sessions may exist — read the full file with `read_file_chunk("{IMPL_LOG_PATH}")` if needed.
> **Important**: Cross-reference with `git log --oneline -20` and actual source files to verify what was actually changed.

{impl_log_content}
"""
            else:
                task_description += """
### Implementation History
> No implementation log yet — this may be the first experiment run. Check `git log` and the code directly to understand the current state.
"""
        except Exception:
            pass  # Don't block experiment on log read failure

        # Configure subagent
        default_time = DEFAULT_EXPERIMENT_CONFIG.time_limit
        if mode == "validate":
            default_time = EXPERIMENT_VALIDATE_TIME_LIMIT

        config = SubagentConfig(
            max_steps=DEFAULT_EXPERIMENT_CONFIG.max_steps,
            time_limit=time_budget or default_time,
            reminder_freq=DEFAULT_EXPERIMENT_CONFIG.reminder_freq,
            log_dir=session_dir,
        )

        subagent = ExperimentSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
        )

        try:
            result = await subagent.run(
                computer=computer,
                task_description=task_description,
                constraints=self.constraints,
            )

            return self._format_result(result, task, mode)

        except Exception as e:
            ctx_logger.error(f"Experiment failed: {e}")
            return f"Error during experiment: {str(e)}"

    def _create_session_dir(self) -> tuple[str, int]:
        """
        Create a session directory for this experiment call.

        Returns:
            Tuple of (session_dir_path, session_count)
        """
        if self.run_dir:
            base_dir = bf.join(self.run_dir, "subagent_logs")
        else:
            base_dir = "/tmp/subagent_logs"

        os.makedirs(base_dir, exist_ok=True)

        try:
            existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
            count = len(existing) + 1
        except Exception:
            count = 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = bf.join(base_dir, f"exp_{count:03d}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        return session_dir, count

    def _format_result(
        self,
        result: SubagentOutput,
        task: str,
        mode: str,
    ) -> str:
        """Format the experiment result for Main Agent."""
        status_map = {
            SubagentStatus.COMPLETED: "✅",
            SubagentStatus.FAILED: "❌",
            SubagentStatus.TIMEOUT: "⏰",
        }
        status_icon = status_map.get(result.status, "❓")

        # Truncate task for header
        task_short = task[:50] + "..." if len(task) > 50 else task
        header = f"[Experiment {status_icon}] {task_short} | Mode: {mode}"
        header += f" ({result.num_steps} steps, {result.runtime_seconds:.1f}s)"

        output_lines = [
            header,
            "",
            "## Results",
            result.content,
            "",
        ]

        # Add recommendations based on SubagentStatus only.
        # The subagent's report (result.content) already contains structured
        # status/metrics/diagnosis — the main agent can interpret it directly.
        if result.status == SubagentStatus.COMPLETED:
            output_lines.extend([
                "## What's Next?",
                "Read the experiment report above and decide:",
                "- If successful: move to the next task",
                "- If failed/partial: use `implement(mode='fix', task='fix ...', context='<diagnosis from above>')` then re-run",
            ])
        elif result.status == SubagentStatus.FAILED:
            output_lines.extend([
                "## Error",
                result.error_message or "Unknown error",
                "",
                "## What's Next?",
                "- Review the error and decide how to proceed",
                "- Use `implement(mode='fix', task='fix ...')` if code changes needed",
            ])
        elif result.status == SubagentStatus.TIMEOUT:
            output_lines.extend([
                "## Note",
                "Experiment timed out. This could be normal for long-running experiments.",
                "- Check /home/agent/experiments/ for partial logs",
                "- Consider increasing time_budget or using mode='validate'",
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
            description="""Invoke the Experiment Subagent to run and validate experiments.

**What it does**:
1. Runs experiments with proper logging (output saved to /home/agent/experiments/)
2. Validates results against paper expectations
3. Diagnoses failures with actionable suggestions
4. Can fix trivial issues encountered during execution (wrong paths, missing imports)
5. Records results to exp_log.md
6. Returns summary with status, metrics, and recommended next steps

**Parameters**:
- task: Required. What to validate (e.g., "P0-1: Validate transformer encoder")
- mode: Optional. "full" (default, complete run) or "validate" (quick check)
- context: Optional. Additional context (e.g., previous experiment feedback, specific concerns)

**Example**:
```
run_experiment(task="P0-1: Validate encoder output shapes")
run_experiment(task="Run full training pipeline", mode="full")
run_experiment(task="Quick check reproduce.sh works", mode="validate")
run_experiment(task="Re-run training after lr fix", context="Previous run diverged with lr=1e-3, now fixed to 1e-4")
```""",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to validate (e.g., 'P0-1: Validate transformer encoder')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["full", "validate"],
                        "description": "Experiment mode: 'full' (complete run) or 'validate' (quick check)",
                        "default": "full",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context from previous experiments, diagnosis, or specific concerns",
                    },
                    "time_budget": {
                        "type": "integer",
                        "description": "Time budget in seconds (default: ~10h for full, ~5h for validate)",
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            strict=False,
        )
