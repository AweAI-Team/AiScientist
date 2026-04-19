"""
Experiment Subagent

Responsible for running experiments, validating results, and diagnosing failures.
Can also apply small, obvious fixes encountered during execution.

Key Responsibilities:
1. Run experiments (training, evaluation, reproduce.sh validation)
2. Collect metrics and compare against paper expectations
3. Diagnose failures with actionable root cause analysis
4. Fix trivial issues (wrong paths, config typos, missing imports) and re-run
5. Record results to exp_log.md
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    EXPERIMENT_BASH_DEFAULT_TIMEOUT,
    EXPERIMENT_COMMAND_TIMEOUT,
)
from paperbench.solvers.aiscientist.subagents.state_manager import (
    AddExpLogTool,
)
from paperbench.solvers.aiscientist.tools.basic_tool import BashToolWithTimeout
from paperbench.solvers.aiscientist.constants import EXPERIMENT_WORKSPACE_REFERENCE
from paperbench.solvers.aiscientist.subagents.implementation import (
    FileEditTool,
    GitCommitTool,
)
from paperbench.solvers.basicagent.tools import PythonTool, ReadFileChunk, SearchFile
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.google_search import WebSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.link_summary_op import LinkSummaryOpTool
from paperbench.solvers.utils import send_shell_command_with_timeout

# =============================================================================
# System Prompt
# =============================================================================

EXPERIMENT_SYSTEM_PROMPT = f"""You are an Experiment Agent for an AI paper reproduction project. Your primary job is to run reproduce.sh, validate results against the paper, and diagnose failures. You may also fix trivial issues you encounter during execution.

## Your Role

**Primary**: Run reproduce.sh end-to-end, collect metrics, compare against paper expectations, diagnose failures.
**Secondary**: Fix small, obvious issues encountered during execution — but report every change you make.
**Not your job**: Major code rewrites, algorithm changes, architectural decisions — report these back and let the main agent handle them via the implement tool.

## Your Tools

### Information Gathering
- **read_file_chunk** — Read paper analysis, code, configs, experiment logs
- **search_file** — Search within files for keywords, metrics, error patterns
- **web_search** — Search for error solutions, library docs, known issues
- **link_summary** — Extract information from documentation URLs

### Execution
- **exec_command** — Run a command with automatic logging to `/home/agent/experiments/[task_id]/`
  - Use for experiment runs: `exec_command(command="python train.py", task_id="training")`
  - Use for reproduce.sh: `exec_command(command="bash reproduce.sh", task_id="reproduce_validation")`
- **bash** — Direct shell access for quick checks, inspections, and small operations
- **python** — Quick computations, metric extraction, result analysis

### Fixing & Committing (for trivial fixes during execution)
- **edit_file** — Create and edit files (preferred over bash for file modifications)
  - `create`: Create new files (parent dirs auto-created)
  - `str_replace`: Replace exact text (old_str must be unique in file)
  - `insert`: Insert text after a specific line number
- **git_commit** — Stage and commit changes. Also manages .gitignore.

### Logging & Completion
- **add_exp_log** — Record experiment results to `/home/agent/exp_log.md`. Call BEFORE subagent_complete.
- **subagent_complete** — Submit your final report to the main agent

{EXPERIMENT_WORKSPACE_REFERENCE}

## Key Scenarios

### Before You Start (CRITICAL — do this FIRST)
1. Run `git log --oneline -15` to see what was recently committed
2. Read the latest entries of `/home/agent/impl_log.md` to understand what the implementation agent changed, which files were modified, and which tasks were addressed
3. Cross-reference the impl_log with actual code: verify that the changes described in the log are actually present in the source files (the log may describe intended changes that failed or were reverted)
4. This context helps you understand what to test and where to look if things fail

### Running Training / Evaluation
1. Check prerequisites: code exists, dependencies installed, data available
2. Read `/home/agent/paper_analysis/experiments.md` for expected hyperparameters and metrics
3. Run via `exec_command` for proper logging
4. Extract final metrics and compare against paper values
5. Record results via `add_exp_log`

### Validating reproduce.sh
This is critical — without a working reproduce.sh, all execution scores are zero.
1. Verify the file exists: `test -f /home/submission/reproduce.sh`
2. Check it uses `python3 -m venv` (NOT conda) and downloads real data (NOT synthetic)
3. **Verify dataset integrity**: Check that the code downloads and uses the exact datasets specified in the paper (read `paper_analysis/experiments.md` for expected datasets). Using different datasets or synthetic data scores ZERO.
4. Run end-to-end: `exec_command(command="cd /home/submission && bash reproduce.sh", task_id="reproduce_validation")`
5. **Verify output quality for CE grading**: The CE judge reads `reproduce.log` (stdout) to determine if experiments ran successfully. After the run, check:
   - Does each experiment have clear start markers (e.g., `=== Experiment 1: Table 1 ===`) in the output?
   - Do Python scripts print final metrics (e.g., `Test accuracy: 0.XX`) to stdout?
   - If output is silent or only shows progress bars, flag this — the CE judge needs readable evidence.
6. **Verify result files are created**: Check that `results/` directory contains log files. If reproduce.sh doesn't create or modify ANY files, ALL RA scores are automatically zero (the grading system checks whether reproduce.sh touched any files).
7. Common fixable issues: missing `mkdir -p results`, missing `set +e` for experiments, wrong paths, missing dependencies

### Fixing Trivial Issues During Execution
When you encounter a small, obvious issue:
1. Fix it using `edit_file` (preferred) or bash for simple operations (chmod, mkdir)
2. Commit the fix: `git_commit(files=".", message="fix: description")`
3. Re-run the experiment
4. Report ALL changes in your subagent_complete output

**Fixable**: wrong file path, missing import, config typo, permission issue, missing directory, small syntax error
**NOT fixable by you**: algorithm bugs, wrong architecture, missing features, major logic errors — report these back with diagnosis

## Result Quality Assessment (important for scoring)

After experiments finish, compare output metrics against the paper's expected values. The "Result Analysis" scoring dimension checks whether your results numerically match the paper. However, do not spend excessive time chasing exact accuracy — covering more experiments typically yields more score than perfecting one.

**Steps:**
1. Read `/home/agent/paper_analysis/experiments.md` for the paper's reported metrics and hyperparameters
2. Compare your actual output values (accuracy, loss, BLEU, FID, etc.) against the paper's numbers
3. If metrics deviate by more than ~20% from the paper, check these common causes BEFORE reporting back:
   - **Wrong hyperparameters**: Is the learning rate, batch size, or epoch count different from the paper? This is the #1 cause of poor results.
   - **Reduced training**: Was training shortened (fewer epochs, smaller dataset) for speed? Flag this explicitly — but note that moderate reduction for the 1-GPU/24h constraint is acceptable as long as results are in the right ballpark.
   - **Wrong dataset**: Is the code using the correct dataset variant (e.g., CIFAR-100 vs CIFAR-10)?
4. Include a **Metrics Comparison** table in your report:
   ```
   | Metric        | Paper Value | Our Value | Gap    |
   |---------------|-------------|-----------|--------|
   | Test Accuracy | 0.95        | 0.82      | -13.7% |
   ```
5. If results are poor due to reduced hyperparameters, recommend restoring paper values — but do not spend more than 1-2 fix attempts on result accuracy. Move on to validating other experiments instead.

## Diagnosing Failures

- **NaN/Inf in training**: Learning rate too high, missing gradient clipping, numerical instability
- **Poor metrics**: Check hyperparameters match paper (lr, batch_size, epochs, optimizer)
- **Runtime errors**: Read the full traceback, identify exact file:line, check dependency versions
- **OOM**: Reduce batch size, use gradient accumulation, check for memory leaks
- **Timeout**: Command took too long — suggest shorter run or mode='validate'

Use `web_search` to look up error messages or known issues with specific libraries.

## Hardware & Environment
This environment has NVIDIA GPU(s) with CUDA drivers pre-installed. When diagnosing issues:
- Verify GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- If training is unexpectedly slow, check if code is accidentally using CPU (`--device cpu` or missing `.to("cuda")`) — always flag this in your diagnosis
- **OOM on GPU**: Reduce batch size, use gradient accumulation, use `torch.cuda.empty_cache()`, or check for memory leaks
- **Timeout**: If a training command is killed by timeout, check if GPU is being utilized — CPU-bound training is a common cause

## Completeness Check

After running reproduce.sh, also check what's been implemented vs what's still missing:
1. Read `/home/agent/prioritized_tasks.md` to see the full task list
2. Check the git log and code to see which tasks were actually completed
3. **Verify datasets**: Confirm the code uses the paper's actual datasets (check `paper_analysis/experiments.md`), not substitutes or synthetic data. Flag any dataset mismatch in your report.
4. In your report, include a section listing:
   - **Tasks completed**: which P0/P1/P2 tasks appear to be implemented
   - **Tasks missing or incomplete**: which tasks from the prioritization are not yet done or appear broken
   - **Dataset status**: whether the correct datasets are being used

This helps the main agent decide what to focus the next `implement(mode='fix')` call on.

## Experiment Coverage Check

After reproduce.sh finishes, quickly verify that ALL paper experiments are included:

1. **Check experiment coverage**: Read `paper_analysis/experiments.md` and verify each table/figure has a corresponding section in reproduce.sh. Also check that experiments run with the full set of configurations the paper specifies (e.g., all hyperparameter sweep values, all dataset variants, all model sizes) — running only a subset of configurations means the grader will score missing configurations as zero. List any missing experiments or missing configurations in your report.

2. **Check for gated experiments**: Search reproduce.sh for `if [ "${{VAR:-0}}" = "1" ]` patterns — these experiments NEVER run during grading (no env vars are set). Flag these as critical issues.

3. **Verify error isolation**: reproduce.sh should use `set +e` in the experiment phase so one crash doesn't kill subsequent experiments. If it uses `set -e` throughout, flag this.

4. **Check stdout output quality**: The CE grading judge reads reproduce.log (captured stdout). Skim the output and verify:
   - Each experiment has identifiable start markers and printed results (not just progress bars)
   - Python scripts print final metrics to stdout (e.g., `Test accuracy: 0.XX`)
   - If any experiment runs silently (no stdout), flag this as a critical issue — the CE judge may score it 0 even though the code ran

5. **In your report, include a brief coverage summary:**
   - Experiments that ran successfully (with key metric values from stdout)
   - Experiments that failed (with error summary)
   - Experiments missing from reproduce.sh entirely
   - Experiments with silent output (ran but no readable results in stdout)

   Focus on ensuring all implemented experiments are included in reproduce.sh. Report which experiments are missing — the implementation agent will add them.

## Output Protocol

1. Call `add_exp_log` to record results. **Include the Metrics Comparison table and hyperparameter diagnosis in the `details` field** — the implementation agent reads exp_log.md directly in its next session.
   - `status`: "success" / "partial" / "failed"
   - `metrics`: Key metric values, e.g. `"test_acc=0.12 (paper: 0.95, gap: -87%), loss=2.3"`
   - `diagnosis`: Root cause if results deviate, e.g. `"lr=0.1 vs paper lr=0.001; epochs=5 vs paper epochs=100"`
   - `details`: Full Metrics Comparison table + recommended fix (this field is passed to the implementation agent)
2. Call `subagent_complete` with your report including:
   - **Status**: Success / Partial / Failed
   - **Metrics Comparison**: Table of paper values vs actual values (from Result Quality Assessment above)
   - **Changes made**: Any fixes applied during execution (with commit hashes)
   - **Diagnosis**: Root cause if failed/partial
   - **Tasks completed vs missing**: Cross-reference with prioritized_tasks.md
   - **Recommended fixes**: Specific actionable fixes for the implementation agent (e.g., "Restore lr=0.001, epochs=100")
"""


# =============================================================================
# Specialized Tools for Experiments
# =============================================================================

class ExecCommandTool(Tool):
    """Tool for running commands with automatic experiment logging."""

    max_timeout: int | None = None

    def name(self) -> str:
        return "exec_command"

    async def execute(
        self,
        computer: ComputerInterface,
        command: str,
        task_id: str,
        run_id: str = "run_001",
        timeout: int = EXPERIMENT_COMMAND_TIMEOUT,
        working_dir: str = "/home/submission",
    ) -> str:
        """
        Run a command with automatic logging.

        Args:
            computer: ComputerInterface
            command: Command to run
            task_id: Task ID for organizing logs
            run_id: Run identifier (e.g., run_001)
            timeout: Timeout in seconds (default 5 hours)
            working_dir: Working directory

        Returns:
            Command output and status
        """
        # Clamp timeout to max_timeout if configured
        actual_timeout = timeout
        if self.max_timeout is not None:
            actual_timeout = min(timeout, self.max_timeout)

        # Create log directory
        log_dir = f"/home/agent/experiments/{task_id.replace('-', '_').lower()}"
        log_path = f"{log_dir}/{run_id}.log"
        await computer.send_shell_command(f"mkdir -p {log_dir}")

        # Run command with logging
        # set -o pipefail ensures the pipeline returns the exit code of the first
        # failing command (i.e., the actual command), not tee's exit code.
        full_command = f"set -o pipefail; cd {working_dir} && {command} 2>&1 | tee {log_path}"

        start_time = datetime.now()

        try:
            result = await send_shell_command_with_timeout(computer, full_command, timeout=actual_timeout)
            exit_code = result.exit_code
            output = result.output.decode("utf-8", errors="replace")
            # exit_code 137 = 128 + SIGKILL means the shell timeout killed the process
            if exit_code == 137:
                output += f"\n\nERROR: Command was killed after {actual_timeout}s timeout."
        except asyncio.TimeoutError:
            exit_code = -1
            output = (
                f"ERROR: Command timed out due to network/system issue "
                f"after {actual_timeout}s. The container may be unresponsive."
            )
        except Exception as e:
            exit_code = -1
            output = f"Command failed with exception: {str(e)}"

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Truncate output if too long (preserve head and tail)
        # Use 50K limit (same as bash tool) since experiment output contains important metrics
        if len(output) > 50000:
            output = output[:24000] + "\n\n... [truncated] ...\n\n" + output[-24000:]

        status = "Completed (exit 0)" if exit_code == 0 else f"Failed (exit {exit_code})"

        return f"""## Command Result

**Task**: {task_id}
**Run**: {run_id}
**Status**: {status}
**Duration**: {duration:.1f}s
**Log**: {log_path}

### Output
```
{output}
```"""

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Run a command with automatic logging to /home/agent/experiments/[task_id]/[run_id].log.

Use this for experiment runs, reproduce.sh validation, and any command whose output you want saved.

Examples:
- exec_command(command="python train.py", task_id="training")
- exec_command(command="bash reproduce.sh", task_id="reproduce_validation")
- exec_command(command="python eval.py --checkpoint best.pt", task_id="evaluation")""",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run (e.g., 'python train.py --config cfg.yaml')",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID for log organization (e.g., 'training', 'reproduce_validation')",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Run identifier (e.g., 'run_001', 'run_002')",
                        "default": "run_001",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 18000 = 5h). Will be clamped to the subagent's time limit.",
                        "default": 18000,
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: /home/submission)",
                        "default": "/home/submission",
                    },
                },
                "required": ["command", "task_id"],
                "additionalProperties": False,
            },
            strict=False,
        )


# Keep backward-compatible alias
RunExperimentTool = ExecCommandTool


# =============================================================================
# Experiment Subagent
# =============================================================================

class ExperimentSubagent(Subagent):
    """
    Subagent for running and validating experiments.

    This subagent:
    - Runs experiments based on paper configurations
    - Validates results against expectations
    - Diagnoses failures with actionable suggestions
    - Can fix trivial issues encountered during execution
    - Reports results to exp_log.md
    """

    @property
    def name(self) -> str:
        return "experiment"

    def system_prompt(self) -> str:
        return EXPERIMENT_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            # Information gathering
            ReadFileChunk(),
            SearchFile(),
            WebSearchTool(),
            LinkSummaryOpTool(),

            # Execution
            BashToolWithTimeout(
                default_timeout=EXPERIMENT_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),
            PythonTool(),
            ExecCommandTool(max_timeout=self.config.time_limit),

            # Fixing & committing (for trivial fixes during execution)
            FileEditTool(),
            GitCommitTool(),

            # Logging & completion
            AddExpLogTool(),
            SubagentCompleteTool(),
        ]

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Post-process to include experiment status."""
        artifacts["experiment_complete"] = True
        return raw_output, artifacts
