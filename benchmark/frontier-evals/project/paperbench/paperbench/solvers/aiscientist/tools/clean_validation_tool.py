"""
Clean Reproduce Validation Tool

A specialized tool that simulates the grading environment before submission.
It cleans cached state (venv, HF datasets, torch hub) then runs reproduce.sh
via the Experiment Subagent for diagnosis.

Purpose:
- Catch "works on my machine" bugs (e.g., cached datasets masking download failures)
- Ensure reproduce.sh works in a clean environment like grading uses
- Should be called once before final submit()
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
    DEFAULT_EXPERIMENT_CONFIG,
)
from paperbench.solvers.aiscientist.subagents.experiment import (
    ExperimentSubagent,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)

# Cleanup commands that simulate the grading environment.
# Executed in order: protect work first, then clean, then simulate grading retry.
CLEANUP_COMMANDS = [
    # Step 0: Protect — auto-commit all changes so git clean -fd is safe
    "cd /home/submission && git add -A && "
    "git diff --cached --quiet || git commit -m 'auto-commit before clean validation'",
    # Step 1: Simulate git clean -fd (grading runs this; safe because we just committed)
    "cd /home/submission && git clean -fd",
    # Step 2: Grading always deletes venv (reproduce.py line 50-51)
    "rm -rf /home/submission/venv /home/submission/.venv",
    # Step 3: Clear HuggingFace dataset cache (catches dataset processing bugs)
    "rm -rf ~/.cache/huggingface/datasets/",
    # Step 4: Clear torch hub cache (catches model download issues)
    "rm -rf ~/.cache/torch/hub/",
    # Step 5: Clear project-local HF caches (stale metadata without data files causes errors)
    "rm -rf /home/submission/.hf_cache /home/submission/.cache /home/submission/huggingface",
    # Step 6: Check for hardcoded /home/submission paths in code.
    # Grading runs from /submission (NOT /home/submission), so any hardcoded path will break.
    "echo '--- Hardcoded path check ---' && "
    "grep -rn '/home/submission' /home/submission/src/ /home/submission/*.py /home/submission/scripts/*.sh 2>/dev/null "
    "| grep -v '.pyc' | grep -v '__pycache__' "
    "&& echo 'ERROR: Hardcoded /home/submission paths found above — these WILL break during grading (grading runs from /submission).' "
    "|| echo 'OK: No hardcoded /home/submission paths found in source code.'",
    # Step 7: Simulate grading retry with make_venv=True — creates an EMPTY venv.
    # This catches the bug where setup_env.sh skips all pip installs when venv exists.
    "cd /home/submission && python3 -m venv venv",
]


class CleanReproduceValidationTool(Tool):
    """
    Tool for Main Agent to run reproduce.sh in a simulated clean environment.

    Before running the Experiment Subagent, this tool:
    1. Auto-commits all changes (protects work from git clean)
    2. Runs git clean -fd (removes untracked files, matching grading behavior)
    3. Deletes venv/ and .venv/ (matches grading behavior)
    4. Clears HuggingFace dataset cache and Torch hub cache
    5. Checks for hardcoded /home/submission paths (grading runs from /submission)
    6. Creates an EMPTY venv (simulates grading retry with make_venv=True)
    7. Then runs reproduce.sh via the Experiment Subagent for diagnosis

    Should be called once before final submit() to catch environment-dependent bugs.
    """

    # Set by solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    run_dir: str | None = None
    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "clean_reproduce_validation"

    async def execute(
        self,
        computer: ComputerInterface,
        time_budget: int | None = None,
    ) -> str:
        """
        Run reproduce.sh in a simulated clean environment.

        Args:
            computer: ComputerInterface
            time_budget: Time budget in seconds (default: ~10h)

        Returns:
            Summary of validation results with cleanup log
        """
        ctx_logger = logger.bind(tool="clean_reproduce_validation")
        ctx_logger.info("Starting clean reproduce validation")

        if self.completer_config is None:
            return "Error: CleanReproduceValidationTool not properly configured."

        # Step 1: Run cleanup commands
        cleanup_results = []
        for cmd in CLEANUP_COMMANDS:
            try:
                result = await computer.send_shell_command(cmd)
                output = result.output.decode("utf-8", errors="replace").strip()
                cleanup_results.append(f"  ✓ {cmd}")
                if output:
                    cleanup_results.append(f"    {output}")
            except Exception as e:
                cleanup_results.append(f"  ✗ {cmd}: {e}")

        cleanup_summary = "\n".join(cleanup_results)
        ctx_logger.info(f"Cleanup completed:\n{cleanup_summary}")

        # Step 2: Build task description for experiment subagent
        task_description = f"""## Clean Environment Validation (Pre-Submit)

### Environment Cleanup Performed
The following cleanup was automatically performed to simulate the grading environment:
```
{cleanup_summary}
```

**This simulates what happens during grading**: reproduce.sh runs in a fresh container with no cached datasets, no pre-installed venv, and no model caches. Additionally, an EMPTY venv was pre-created to simulate the grading system's retry behavior with `make_venv=True` — this catches bugs where setup_env.sh skips all pip installs when a venv directory already exists.

All uncommitted files were auto-committed before cleanup, and `git clean -fd` was run to remove untracked files (matching grading behavior).

### Your Task
1. Run `cd /home/submission && bash reproduce.sh` from scratch — this is exactly what grading does
2. **FIRST check the cleanup output above for "Hardcoded path check"**. If it says "ERROR: Hardcoded /home/submission paths found", report this as a FAILURE — these paths will break during grading because grading runs from `/submission`, not `/home/submission`. The fix is to remove hardcoded paths and use `PYTHONPATH` in reproduce.sh instead.
3. Monitor the output carefully for:
   - Package installation issues (an empty venv exists — setup_env.sh must still install all dependencies)
   - Dataset download failures (the HF cache was cleared, so datasets must re-download)
   - Missing files (git clean -fd removed untracked files — anything not committed is gone)
   - Any errors that were previously masked by cached state
3. If it fails, diagnose the root cause with actionable detail
4. If it succeeds, verify that output files are generated correctly

**Critical**: This is the FINAL validation before submission. Any failure here means the submission WILL score zero on all execution rubric items.
"""

        # Inject recent impl log for context
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
> Below is the latest implementation session from `impl_log.md`.

{impl_log_content}
"""
        except Exception:
            pass

        # Step 3: Create session directory and write separator to exp_log
        session_dir, session_count = self._create_session_dir()
        ctx_logger.info(f"Clean validation session directory: {session_dir}")

        # Write session separator to exp_log
        # Use "=== Experiment Session" prefix so implementation_tool's grep matches it,
        # with "(Clean Validation N)" suffix to distinguish from regular experiment sessions.
        try:
            separator = f"\n=== Experiment Session (Clean Validation {session_count}) ===\n\n"
            escaped_sep = shlex.quote(separator)
            await computer.send_shell_command(
                f"touch {EXP_LOG_PATH} && printf %s {escaped_sep} >> {EXP_LOG_PATH}"
            )
        except Exception:
            pass  # Don't block on separator write failure

        # Step 4: Run experiment subagent
        config = SubagentConfig(
            max_steps=DEFAULT_EXPERIMENT_CONFIG.max_steps,
            time_limit=time_budget or DEFAULT_EXPERIMENT_CONFIG.time_limit,
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

            return self._format_result(result, cleanup_summary)

        except Exception as e:
            ctx_logger.error(f"Clean validation failed: {e}")
            return f"Error during clean validation: {str(e)}"

    def _create_session_dir(self) -> tuple[str, int]:
        """
        Create a session directory for this clean validation call.

        Returns:
            Tuple of (session_dir_path, session_count)
        """
        if self.run_dir:
            base_dir = bf.join(self.run_dir, "subagent_logs")
        else:
            base_dir = "/tmp/subagent_logs"

        os.makedirs(base_dir, exist_ok=True)

        try:
            existing = [d for d in os.listdir(base_dir) if d.startswith("clean_val_")]
            count = len(existing) + 1
        except Exception:
            count = 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = bf.join(base_dir, f"clean_val_{count:03d}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        return session_dir, count

    def _format_result(self, result: SubagentOutput, cleanup_summary: str) -> str:
        """Format the clean validation result for Main Agent."""
        status_map = {
            SubagentStatus.COMPLETED: "✅",
            SubagentStatus.FAILED: "❌",
            SubagentStatus.TIMEOUT: "⏰",
        }
        status_icon = status_map.get(result.status, "❓")

        header = f"[Clean Reproduce Validation {status_icon}]"
        header += f" ({result.num_steps} steps, {result.runtime_seconds:.1f}s)"

        output_lines = [
            header,
            "",
            "## Environment Cleanup",
            cleanup_summary,
            "",
            "## Validation Results",
            result.content,
            "",
        ]

        if result.status == SubagentStatus.COMPLETED:
            output_lines.extend([
                "## What's Next?",
                "If the validation passed, you can safely call `submit()`. "
                "If it failed, fix the issues with `implement(mode='fix', ...)` and re-run `clean_reproduce_validation()`.",
            ])
        elif result.status == SubagentStatus.FAILED:
            output_lines.extend([
                "## Error",
                result.error_message or "Unknown error",
                "",
                "## What's Next?",
                "Fix the issues identified above with `implement(mode='fix', ...)`, then re-run `clean_reproduce_validation()` before submitting.",
            ])
        elif result.status == SubagentStatus.TIMEOUT:
            output_lines.extend([
                "## Note",
                "Clean validation timed out. Check partial results and consider increasing time_budget.",
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
            description="""Run reproduce.sh in a simulated clean grading environment. MUST be called before submit().

**What it does**:
1. Auto-commits all changes (protects your work)
2. Runs `git clean -fd` (removes untracked files, matching grading behavior)
3. Cleans the environment to simulate grading:
   - Deletes venv/ and .venv/ (grading always does this)
   - Clears HuggingFace dataset cache (catches dataset processing bugs)
   - Clears Torch hub cache
4. Creates an EMPTY venv (simulates grading retry with make_venv=True)
5. Runs reproduce.sh from scratch via the Experiment Subagent
6. Diagnoses any failures that were masked by cached state

**When to use**: Call this ONCE before your final submit(). This catches "works on my machine" bugs where reproduce.sh passes during development but fails during grading because cached data/models/venvs masked underlying issues.

**Note**: This takes extra time because it re-downloads datasets and re-installs packages. Only use it as the final validation step.""",
            parameters={
                "type": "object",
                "properties": {
                    "time_budget": {
                        "type": "integer",
                        "description": "Time budget in seconds (default: ~10h). Usually 1-2 hours is sufficient.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )
