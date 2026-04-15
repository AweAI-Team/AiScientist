"""
AI Scientist Orchestrator for MLE-Bench.

Key design:
- Main agent loop: LLM chat → tool dispatch → message management
- Subagent adapter tools: analyze_data, prioritize_tasks, implement, run_experiment
- Submit pre-checks: hard blocks + warnings
- Implement→experiment balance monitoring
- Periodic time/step reminders with escalation
- Context injection between subagents (impl_log → experiment, exp_log → implement)
- Session management with timestamped directories
- Message pruning on context length errors
- Conversation logging (JSONL)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Ensure agent directory is on sys.path when running in container (e.g. exec_run
# may not have cwd set to AGENT_DIR). Support both layouts: (1) correct build with
# context agents/aisci/ -> /home/agent/subagents; (2) wrong build with context .
# -> /home/agent/agents/aisci/subagents.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _script_dir,
    os.path.join(_script_dir, "agents", "aisci"),
]
_agent_dir = None
for _d in _candidates:
    _d = os.path.abspath(_d)
    if os.path.isdir(os.path.join(_d, "subagents")):
        _agent_dir = _d
        break
if _agent_dir is None:
    _msg = (
        "ModuleNotFoundError: 'subagents' not found under any candidate.\n"
        "Tried: %s\n"
        "Rebuild from mle-bench root with: docker build --no-cache ... -t aisci agents/aisci/\n"
        "Verify: docker run --rm --entrypoint \"\" aisci ls -la /home/agent/subagents/"
    ) % (_candidates,)
    print(_msg, file=sys.stderr)
    sys.exit(1)
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)
# Diagnostic: log which agent root was chosen (visible in run.log if import still fails later)
try:
    _listing = os.listdir(_agent_dir)
except OSError:
    _listing = []
print(f"[orchestrator] agent root for imports: {_agent_dir!r}, entries: {sorted(_listing)[:20]}", file=sys.stderr)

import structlog

from constants import is_file_as_bus_enabled
from openai import BadRequestError
from llm_client import LLMClient, LLMConfig, ContextLengthError, ContentPolicyError, create_llm_client
from shell_interface import ShellInterface
from subagents.base import SubagentConfig, SubagentOutput, SubagentStatus, prune_messages, prune_messages_individual, fix_message_consistency, _fmt
from subagents.configs import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_PRIORITIZATION_CONFIG,
    DEFAULT_IMPLEMENTATION_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    EXPERIMENT_VALIDATE_TIME_LIMIT,
    MAIN_BASH_DEFAULT_TIMEOUT,
    MAIN_BASH_MAX_TIMEOUT,
)
from subagents.analysis import DataAnalysisSubagent
from subagents.prioritization import PrioritizationSubagent
from subagents.implementation import ImplementationSubagent
from subagents.experiment import ExperimentSubagent
from tools.base import Tool
from tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
)
from tools.spawn_subagent_tool import SpawnSubagentTool
from prompts.templates import (
    SUMMARY_FIRST_TIME_PROMPT,
    SUMMARY_INCREMENTAL_PROMPT,
    main_agent_system_prompt_for_run,
)
from log_utils import log_messages_to_file, log_model_response_event, log_tool_result_event, _box, _short
from summary_utils import (
    parse_rest_into_turns,
    serialize_segment_messages,
    SUMMARY_USER_INTRO,
)

logger = structlog.stdlib.get_logger(component=__name__)

# LOGS_DIR is extracted by MLE-Bench after the container stops.
# All logs written here are preserved in the per-task run directory on the host.
# If nonroot cannot create subdirs under /home/logs (base image entrypoint only did chmod a+rw),
# we fall back to /home/agent/logs so the run can proceed.
LOGS_DIR = os.environ.get("LOGS_DIR", "/home/logs")

# ====================================================================== #
# Subagent Adapter Tools (main agent dispatches to subagents via these)
# ====================================================================== #


class AnalyzeDataTool(Tool):
    """Main-agent tool to dispatch the Data Analysis Subagent."""

    def __init__(self, shell: ShellInterface, llm: LLMClient):
        self._shell = shell
        self._llm = llm
        self._session_counter = 0

    def name(self) -> str:
        return "analyze_data"

    def execute(self, shell, **kwargs) -> str:
        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{LOGS_DIR}/subagent_logs/analysis_{self._session_counter:03d}_{ts}"
        os.makedirs(session_dir, mode=0o755, exist_ok=True)

        config = SubagentConfig(
            max_steps=DEFAULT_ANALYSIS_CONFIG.max_steps,
            time_limit=DEFAULT_ANALYSIS_CONFIG.time_limit,
            reminder_freq=DEFAULT_ANALYSIS_CONFIG.reminder_freq,
            log_dir=session_dir,
        )

        context = (
            "Analyze the competition data in /home/data/.\n"
            "Read description.md, examine all data files, and produce /home/agent/analysis/summary.md.\n"
            "Include: competition overview, dataset shapes, column types, missing values, key features, "
            "evaluation metric, and strategy recommendations."
        )

        logger.info("Data analysis subagent started", session_dir=session_dir)
        subagent = DataAnalysisSubagent(self._shell, self._llm, config)
        result = subagent.run(context=context)
        logger.info("Data analysis subagent finished", steps=result.num_steps, runtime_s=result.runtime_seconds)
        return _format_subagent_result("DataAnalysis", result)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "analyze_data",
                "description": (
                    "Dispatch a Data Analysis Subagent to examine competition data. "
                    "Produces /home/agent/analysis/summary.md with dataset overview, "
                    "column types, distributions, and strategy recommendations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }


class PrioritizeTasksTool(Tool):
    """Main-agent tool to dispatch the Prioritization Subagent."""

    def __init__(self, shell: ShellInterface, llm: LLMClient):
        self._shell = shell
        self._llm = llm
        self._session_counter = 0

    def name(self) -> str:
        return "prioritize_tasks"

    def execute(self, shell, **kwargs) -> str:
        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{LOGS_DIR}/subagent_logs/prio_{self._session_counter:03d}_{ts}"
        os.makedirs(session_dir, mode=0o755, exist_ok=True)

        config = SubagentConfig(
            max_steps=DEFAULT_PRIORITIZATION_CONFIG.max_steps,
            time_limit=DEFAULT_PRIORITIZATION_CONFIG.time_limit,
            reminder_freq=DEFAULT_PRIORITIZATION_CONFIG.reminder_freq,
            log_dir=session_dir,
        )

        # Inject data analysis summary if available
        context_parts = [
            "Create a prioritized task list for this Kaggle competition.\n"
            "Read /home/data/description.md and /home/data/sample_submission.csv.\n"
        ]
        analysis_path = "/home/agent/analysis/summary.md"
        if os.path.exists(analysis_path):
            context_parts.append(f"Data analysis is available at {analysis_path} — read it for insights.\n")
        context = "\n".join(context_parts)

        subagent = PrioritizationSubagent(self._shell, self._llm, config)
        result = subagent.run(context=context)
        return _format_subagent_result("Prioritization", result)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "prioritize_tasks",
                "description": (
                    "Dispatch a Prioritization Subagent to create a ranked task list. "
                    "Produces /home/agent/prioritized_tasks.md with P0-P3 priority rankings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }


class ImplementTool(Tool):
    """
    Main-agent tool to dispatch the Implementation Subagent.

    Implementation tool adapter:
    - Session management (timestamped directories)
    - Context injection from recent exp_log (when ``file_as_bus``)
    - Mode support (full / fix / explore / refine / ensemble)
    """

    def __init__(self, shell: ShellInterface, llm: LLMClient, file_as_bus: bool = True):
        self._shell = shell
        self._llm = llm
        self._file_as_bus = file_as_bus
        self._session_counter = 0

    def name(self) -> str:
        return "implement"

    def execute(
        self,
        shell,
        task: str = "",
        mode: str = "full",
        context: str = "",
        time_budget: int | None = None,
        **kwargs,
    ) -> str:
        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{LOGS_DIR}/subagent_logs/impl_{self._session_counter:03d}_{ts}"
        os.makedirs(session_dir, mode=0o755, exist_ok=True)

        if self._file_as_bus:
            # Write session separator to impl_log — title must match grep pattern '^=== Implement Session'
            _write_session_separator(
                "/home/agent/impl_log.md",
                f"Implement Session {self._session_counter}",
                f"Mode: {mode} | Task: {task or '(full prioritization)'}",
            )

        mode = (mode or "full").strip().lower()
        if mode not in {"full", "fix", "explore", "refine", "ensemble"}:
            mode = "full"

        # Determine time budget — implementation subagent:
        #   mode="full"  → DEFAULT_IMPLEMENTATION_CONFIG.time_limit (28800s = 8h)
        #   mode="fix"   → 7200s (2h)
        # Model may override by passing an explicit value; None means use the default.
        if time_budget is not None:
            tl = time_budget
        else:
            tl = DEFAULT_IMPLEMENTATION_CONFIG.time_limit if mode == "full" else 7200

        config = SubagentConfig(
            max_steps=DEFAULT_IMPLEMENTATION_CONFIG.max_steps,
            time_limit=tl,
            reminder_freq=DEFAULT_IMPLEMENTATION_CONFIG.reminder_freq,
            log_dir=session_dir,
        )

        # Build context for the subagent
        context_parts = []
        if mode == "full":
            context_parts.append(
                "## Mode: Full Implementation\n"
                "Read /home/agent/prioritized_tasks.md and work through tasks autonomously "
                "(P0 first, breadth-first strategy).\n"
            )
        elif mode == "explore":
            context_parts.append(
                "## Mode: Explore\n"
                "Goal: quickly test a distinct hypothesis and measure potential.\n"
                "- Keep scope narrow and fast.\n"
                "- Prefer one model family or one major change per run.\n"
                "- Produce a valid candidate submission and report metrics clearly.\n"
            )
        elif mode == "refine":
            context_parts.append(
                "## Mode: Refine\n"
                "Goal: improve a promising existing pipeline.\n"
                "- Focus on hyperparameters, training stability, data augmentation, and validation quality.\n"
                "- Avoid broad rewrites; iterate on what already works.\n"
            )
        elif mode == "ensemble":
            context_parts.append(
                "## Mode: Ensemble\n"
                "Goal: combine strong diverse candidates to improve medal probability.\n"
                "- Prioritize weighted average/stacking with validation-backed weights.\n"
                "- Keep intermediate candidate submissions as separate files before final publish.\n"
            )
        else:
            if task and task.strip():
                context_parts.append(
                    f"## Mode: Fix\n"
                    f"Apply targeted fixes for: {task}\n"
                )
            else:
                if self._file_as_bus:
                    context_parts.append(
                        "## Mode: Fix\n"
                        "Apply targeted fixes. Read the context and experiment log (exp_log.md) to determine what to fix.\n"
                    )
                else:
                    context_parts.append(
                        "## Mode: Fix\n"
                        "Apply targeted fixes using the context in this message, git history, and the codebase.\n"
                    )
        if task and mode == "full":
            context_parts.append(f"## Focus\n{task}\n")
        if context:
            context_parts.append(f"## Context from previous rounds\n{context}\n")

        # Inject last experiment session for context continuity (extract last session)
        if self._file_as_bus:
            exp_log_path = "/home/agent/exp_log.md"
            if os.path.exists(exp_log_path):
                try:
                    exp_log_cmd = (
                        f"LAST_SEP=$(grep -n '^=== Experiment Session' {exp_log_path} 2>/dev/null "
                        f"| tail -1 | cut -d: -f1); "
                        f"if [ -n \"$LAST_SEP\" ]; then sed -n \"${{LAST_SEP}},\\$p\" {exp_log_path}; "
                        f"else cat {exp_log_path} 2>/dev/null || echo '(no experiment log yet)'; fi"
                    )
                    exp_log_content = self._shell.send_command(exp_log_cmd, timeout=10).strip()
                    if exp_log_content and exp_log_content != "(no experiment log yet)":
                        context_parts.append(
                            "### Recent Experiment History (auto-injected, last session)\n"
                            f"> Below is the latest experiment session from `exp_log.md`. "
                            "Earlier sessions may exist — read the full file with `read_file_chunk` if needed.\n"
                            "> **Important**: Cross-reference with git log and actual code to verify current state.\n\n"
                            f"{exp_log_content}\n"
                        )
                    else:
                        context_parts.append(
                            "### Experiment History\n"
                            "> No experiment has been run yet — this is the first round. "
                            "Skip the 'Assess current state' step and proceed directly to reading tasks.\n"
                        )
                except Exception:
                    pass

        full_context = "\n".join(context_parts)

        logger.info("Implementation subagent started", mode=mode, session_dir=session_dir)
        subagent = ImplementationSubagent(self._shell, self._llm, config, file_as_bus=self._file_as_bus)
        result = subagent.run(context=full_context)
        logger.info("Implementation subagent finished", steps=result.num_steps, runtime_s=result.runtime_seconds)

        mode_label = {
            "full": "Full",
            "fix": "Fix",
            "explore": "Explore",
            "refine": "Refine",
            "ensemble": "Ensemble",
        }.get(mode, "Full")
        return _format_subagent_result(f"Implementation ({mode_label})", result)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "implement",
                "description": (
                    "Dispatch an Implementation Subagent for substantial coding work.\n"
                    "- mode='full': autonomous breadth-first implementation from prioritized_tasks.md\n"
                    "- mode='fix': targeted fixes with specific task description and context\n"
                    "- mode='explore': fast bounded hypothesis test\n"
                    "- mode='refine': improve a promising pipeline\n"
                    "- mode='ensemble': build blending/stacking candidates\n"
                    "The subagent reads prioritized_tasks.md autonomously (not directed task-by-task)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What to build or fix — be specific",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["full", "fix", "explore", "refine", "ensemble"],
                            "description": (
                                "full = autonomous implementation, "
                                "fix = targeted fixes, explore = quick hypothesis test, "
                                "refine = focused optimization, ensemble = model combination"
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": "Feedback from previous attempts (experiment diagnosis, error logs)",
                        },
                        "time_budget": {
                            "type": "integer",
                            "description": "Time budget in seconds. If not specified, a default is used based on mode.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }


class RunExperimentTool(Tool):
    """
    Main-agent tool to dispatch the Experiment Subagent.

    Experiment tool adapter:
    - Session management (timestamped directories)
    - Context injection from recent impl_log (when ``file_as_bus``)
    - Mode support (full / validate)
    """

    def __init__(self, shell: ShellInterface, llm: LLMClient, file_as_bus: bool = True):
        self._shell = shell
        self._llm = llm
        self._file_as_bus = file_as_bus
        self._session_counter = 0

    def name(self) -> str:
        return "run_experiment"

    def execute(
        self,
        shell,
        task: str = "Run the solution and validate submission.csv",
        mode: str = "full",
        time_budget: int | None = None,
        **kwargs,
    ) -> str:
        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{LOGS_DIR}/subagent_logs/exp_{self._session_counter:03d}_{ts}"
        os.makedirs(session_dir, mode=0o755, exist_ok=True)

        if self._file_as_bus:
            # Write session separator to exp_log — title must match grep pattern '^=== Experiment Session'
            _write_session_separator(
                "/home/agent/exp_log.md",
                f"Experiment Session {self._session_counter}",
                f"Mode: {mode} | Task: {task}",
            )

        # Determine time budget — experiment subagent:
        #   mode="full"      → DEFAULT_EXPERIMENT_CONFIG.time_limit (36000s = 10h)
        #   mode="validate"  → EXPERIMENT_VALIDATE_TIME_LIMIT (18000s = 5h)
        # Model may override by passing an explicit value; None means use the default.
        if time_budget is not None:
            tl = time_budget
        elif mode == "validate":
            tl = EXPERIMENT_VALIDATE_TIME_LIMIT
        else:
            tl = DEFAULT_EXPERIMENT_CONFIG.time_limit

        config = SubagentConfig(
            max_steps=DEFAULT_EXPERIMENT_CONFIG.max_steps,
            time_limit=tl,
            reminder_freq=DEFAULT_EXPERIMENT_CONFIG.reminder_freq,
            log_dir=session_dir,
        )

        # Build context
        context_parts = [f"## Task\n{task}\n", f"## Mode: {mode}\n"]

        # Inject last implementation session for context continuity (extract last session)
        if self._file_as_bus:
            impl_log_path = "/home/agent/impl_log.md"
            if os.path.exists(impl_log_path):
                try:
                    impl_log_cmd = (
                        f"LAST_SEP=$(grep -n '^=== Implement Session' {impl_log_path} 2>/dev/null "
                        f"| tail -1 | cut -d: -f1); "
                        f"if [ -n \"$LAST_SEP\" ]; then sed -n \"${{LAST_SEP}},\\$p\" {impl_log_path}; "
                        f"else cat {impl_log_path} 2>/dev/null || echo '(no implementation log yet)'; fi"
                    )
                    impl_log_content = self._shell.send_command(impl_log_cmd, timeout=10).strip()
                    if impl_log_content and impl_log_content != "(no implementation log yet)":
                        context_parts.append(
                            "### Recent Implementation History (auto-injected, last session)\n"
                            f"> Below is the latest implementation session from `impl_log.md`. "
                            "Earlier sessions may exist — read the full file with `read_file_chunk` if needed.\n"
                            "> **Important**: Cross-reference with `git log --oneline -20` and actual source files.\n\n"
                            f"{impl_log_content}\n"
                        )
                    else:
                        context_parts.append(
                            "### Implementation History\n"
                            "> No implementation log yet. Check `git log` and the code directly.\n"
                        )
                except Exception:
                    pass

        full_context = "\n".join(context_parts)

        logger.info("Experiment subagent started", mode=mode, session_dir=session_dir)
        subagent = ExperimentSubagent(self._shell, self._llm, config, file_as_bus=self._file_as_bus)
        result = subagent.run(context=full_context)
        logger.info("Experiment subagent finished", steps=result.num_steps, runtime_s=result.runtime_seconds)

        mode_label = "Full" if mode == "full" else "Validate"
        return _format_subagent_result(f"Experiment ({mode_label})", result)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "run_experiment",
                "description": (
                    "Dispatch an Experiment Subagent to run and validate the solution.\n"
                    "- mode='full': complete training + inference + submission validation\n"
                    "- mode='validate': quick format check of submission.csv"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What to validate (e.g., 'Run training and validate submission')",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["full", "validate"],
                            "description": "full = complete run, validate = quick check (default: full)",
                        },
                        "time_budget": {
                            "type": "integer",
                            "description": "Time budget in seconds (default: ~10h for full, ~5h for validate)",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }


class SubmitTool(Tool):
    """
    Final submission tool with pre-checks.

    Hard blocks (non-bypassable — attempt not counted):
    - submission.csv does not exist
    - column names don't match sample_submission.csv
    - row count doesn't match sample_submission.csv

    Soft warnings (shown once on first attempt, bypassed by calling submit() again):
    - NaN values in submission
    - Early submission (> 50% time remaining, first attempt only)
    """

    def __init__(self, time_limit: int, start_time: float, llm: "LLMClient | None" = None):
        self._time_limit = time_limit
        self._start_time = start_time
        self._llm = llm  # used to exclude retry-wait from elapsed (use_real_time_limit)
        self._submit_attempts = 0  # counts real (non-blocked) attempts

    def _real_elapsed(self) -> float:
        """Wall-clock elapsed minus time spent in API retries (use_real_time_limit)."""
        elapsed = time.time() - self._start_time
        if self._llm is not None:
            elapsed -= self._llm.total_retry_time
        return max(elapsed, 0.0)

    def name(self) -> str:
        return "submit"

    def execute(self, shell, confirm: str = "yes", **kwargs) -> str:
        warnings = []

        # Hard block 1: submission.csv must exist
        if not os.path.exists("/home/submission/submission.csv"):
            # Don't count this as a real attempt
            return (
                "SUBMISSION BLOCKED:\n\n"
                "❌ HARD BLOCK: /home/submission/submission.csv does not exist.\n"
                "You MUST generate a valid submission.csv before submitting.\n"
                "Use implement() or bash to create it.\n\n"
                "Fix this issue first. submit() will remain blocked until resolved."
            )

        # Format validation against sample_submission
        if os.path.exists("/home/data/sample_submission.csv"):
            try:
                import pandas as pd
                sub = pd.read_csv("/home/submission/submission.csv")
                sample = pd.read_csv("/home/data/sample_submission.csv")

                # Column mismatch — HARD BLOCK (wrong columns = score zero)
                if list(sub.columns) != list(sample.columns):
                    return (
                        "SUBMISSION BLOCKED:\n\n"
                        f"❌ HARD BLOCK: Column mismatch — submission has {list(sub.columns)}, "
                        f"sample expects {list(sample.columns)}.\n"
                        "Fix column names before submitting. "
                        "Use implement(mode='fix', task='Fix submission.csv column names') to resolve."
                        "\n\nFix this issue first. submit() will remain blocked until resolved."
                    )

                # Row count mismatch — HARD BLOCK (missing predictions = score zero)
                if len(sub) != len(sample):
                    return (
                        "SUBMISSION BLOCKED:\n\n"
                        f"❌ HARD BLOCK: Row count mismatch — submission has {len(sub)} rows, "
                        f"sample expects {len(sample)} rows.\n"
                        "Fix test set coverage before submitting. "
                        "Ensure your inference script generates predictions for ALL test examples."
                        "\n\nFix this issue first. submit() will remain blocked until resolved."
                    )

                # NaN values — soft warning
                nan_count = sub.isnull().sum().sum()
                if nan_count > 0:
                    warnings.append(
                        f"⚠ Submission contains {nan_count} NaN values — may reduce score. "
                        "Consider filling NaNs with a fallback value (e.g., 0 or mean)."
                    )
            except Exception as e:
                warnings.append(f"⚠ Could not validate submission format: {e}")

        # Count this as a real attempt now (after all hard blocks passed)
        self._submit_attempts += 1

        # Early submission warning — soft, only on FIRST real attempt
        if self._submit_attempts == 1:
            elapsed = self._real_elapsed()
            time_ratio = elapsed / self._time_limit if self._time_limit > 0 else 1.0
            remaining_hours = (self._time_limit - elapsed) / 3600.0
            if time_ratio < 0.5:
                warnings.append(
                    f"⚠ EARLY SUBMISSION: Only {time_ratio*100:.0f}% of time used "
                    f"({remaining_hours:.1f} hours remaining). "
                    "You have significant time to improve your score. "
                    "Review /home/agent/prioritized_tasks.md for P1/P2 tasks you can still tackle."
                )

        if warnings:
            return (
                "SUBMIT PRE-CHECK WARNINGS:\n\n"
                + "\n".join(f"- {w}" for w in warnings)
                + "\n\nIf you still want to submit, call submit() again."
            )

        return "✅ Submission accepted. The agent will now terminate."

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "submit",
                "description": (
                    "Signal that your work is complete and submit. "
                    "This will run pre-checks on submission.csv before accepting. "
                    "Call only when you are confident."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "string",
                            "description": "Type 'yes' to confirm submission",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }


# ====================================================================== #
# Helper functions
# ====================================================================== #

def _format_subagent_result(label: str, result: SubagentOutput) -> str:
    icon = {
        SubagentStatus.COMPLETED: "✅",
        SubagentStatus.FAILED: "❌",
        SubagentStatus.TIMEOUT: "⏰",
    }.get(result.status, "❓")

    header = (
        f"[{label} {icon}] "
        f"({result.num_steps} steps, {result.runtime_seconds:.0f}s)"
    )
    log_info = f"\n\n**Log**: {result.log_path}" if result.log_path else ""
    return f"{header}\n\n{result.content}{log_info}"


def _write_session_separator(path: str, title: str, details: str) -> None:
    """Write a session separator for stable log extraction.

    The separator line must start with '=== <Title>' so that the last-session extraction
    in ImplementTool and RunExperimentTool (using grep '^=== Implement Session' /
    grep '^=== Experiment Session') can locate it correctly.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    separator = f"\n=== {title} ===\n**Time**: {ts} | {details}\n\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(separator)


def _build_reminder(
    step: int, elapsed: float, time_limit: int,
    impl_count: int, exp_count: int,
    submission_exists: bool = False,
) -> str:
    """Build a periodic reminder — 4-stage schedule + implement/experiment balance."""
    remaining = time_limit - elapsed
    pct = elapsed / time_limit * 100 if time_limit > 0 else 0

    parts = [
        f"⏱ Step {step} | Time: {_fmt(elapsed)}/{_fmt(time_limit)} ({pct:.0f}%) | "
        f"Remaining: {_fmt(remaining)} | "
        f"Impl calls: {impl_count} | Exp calls: {exp_count}"
    ]

    # === Stage-based reminders (4-stage schedule) ===
    if pct >= 85:
        if submission_exists:
            parts.append(
                "⚠️ Less than 15% time remaining.\n"
                "- Make sure all changes are git committed\n"
                "- Focus on finishing current work rather than starting new tasks\n"
                "- Verify /home/submission/submission.csv is valid and up to date"
            )
        else:
            parts.append(
                "🚨 Less than 15% time remaining AND submission.csv DOES NOT EXIST!\n"
                "Without submission.csv, your score is automatically ZERO.\n"
                "Creating a valid submission.csv should be your only priority right now."
            )
    elif pct >= 70:
        if submission_exists:
            parts.append(
                "⚠️ 70%+ time used. Consider wrapping up current work.\n"
                "- If submission.csv hasn't been validated recently, validate it now via run_experiment()\n"
                "- Git commit all changes regularly\n"
                "- Avoid starting large new tasks — focus on finishing in-progress work"
            )
        else:
            parts.append(
                "🚨 70%+ time used AND submission.csv DOES NOT EXIST!\n"
                "Without submission.csv, your score is automatically ZERO.\n"
                "Use implement() immediately to generate a valid submission.csv."
            )
    elif pct >= 50:
        parts.append(
            "50%+ time used. Keep improving — do NOT submit yet.\n"
            "- Check /home/agent/prioritized_tasks.md for remaining P1/P2 tasks\n"
            "- Each additional model improvement earns more points\n"
            "- Git commit regularly!"
        )
        if not submission_exists:
            parts.append(
                "🚨 submission.csv DOES NOT EXIST yet! Generate one NOW via implement().\n"
                "Without it, your score is automatically zero."
            )
    else:
        # Early phase (<50%)
        parts.append("Reminders:")
        parts.append("- Focus on P0-Critical tasks first! Check /home/agent/prioritized_tasks.md")
        parts.append("- Don't forget to git commit regularly!")
        parts.append("- DO NOT submit early. After P0 tasks, keep improving with P1/P2 items.")
        if not submission_exists:
            parts.append(
                "- IMPORTANT: submission.csv doesn't exist yet — "
                "your FIRST implement() task should generate a baseline submission."
            )

    # === Implement/Experiment balance warnings ===
    if impl_count > 0 or exp_count > 0:
        exp_impl_gap = exp_count - impl_count
        impl_exp_gap = impl_count - exp_count

        if exp_impl_gap >= 4:
            parts.append(
                f"\n🚨 IMPLEMENT/EXPERIMENT IMBALANCE: run_experiment called {exp_count} times "
                f"but implement only {impl_count} times (gap: {exp_impl_gap}).\n"
                "Running experiments without code changes is WASTED TIME — experiments are deterministic.\n"
                "You MUST either:\n"
                "1. Call `implement(mode='fix', context='<last experiment diagnosis>')` to fix code, OR\n"
                "2. Move on to the next priority task if stuck after 2-3 fix attempts.\n"
                "Do NOT call run_experiment() again until you have made code changes."
            )
        elif exp_impl_gap >= 2:
            parts.append(
                f"\n⚠️ Note: experiment calls ({exp_count}) are outpacing implement calls ({impl_count}). "
                "If experiments are failing, call implement(mode='fix') to fix the code before running more experiments."
            )

        if impl_exp_gap >= 3:
            parts.append(
                f"\n⚠️ VALIDATION GAP: implement called {impl_count} times "
                f"but run_experiment only {exp_count} times (gap: {impl_exp_gap}).\n"
                "You are writing code without validating it — bugs accumulate.\n"
                "Call `run_experiment()` to check that your code actually works."
            )

    return "\n".join(parts)


# ====================================================================== #
# Main orchestrator
# ====================================================================== #

def main():
    """Main entry point for the AI Scientist orchestrator."""

    # ---- Configuration from environment ----
    time_limit = int(os.environ.get("TIME_LIMIT_SECS", 14400))
    max_steps = int(os.environ.get("AISCI_MAX_STEPS", 500))
    reminder_freq = int(os.environ.get("AISCI_REMINDER_FREQ", 5))
    model = os.environ.get("AISCI_MODEL", "gpt-5.2-2025-12-11")
    hardware = os.environ.get("HARDWARE", "unknown")
    # Context reduction: "prune" (drop oldest ~30%) or "summary" (summarize then replace)
    context_reduce_strategy = (os.environ.get("AISCI_CONTEXT_REDUCE_STRATEGY", "summary") or "summary").strip().lower()
    summary_segment_ratio = float(os.environ.get("AISCI_SUMMARY_SEGMENT_RATIO", "0.3"))
    summary_min_turns = int(os.environ.get("AISCI_SUMMARY_MIN_TURNS_TO_SUMMARIZE", "4"))
    summary_segment_max_chars = int(os.environ.get("AISCI_SUMMARY_SEGMENT_MAX_CHARS", "25000"))
    summary_incremental = (os.environ.get("AISCI_SUMMARY_INCREMENTAL", "true") or "true").strip().lower() in ("true", "1", "yes")
    file_as_bus = is_file_as_bus_enabled()

    logger.info("Starting AI Scientist orchestrator",
                time_limit=time_limit, max_steps=max_steps, model=model,
                api_mode=os.environ.get("AISCI_API_MODE", "completions"), hardware=hardware,
                context_reduce_strategy=context_reduce_strategy, file_as_bus=file_as_bus)

    # ---- Initialise infrastructure ----
    shell = ShellInterface(working_dir="/home")
    api_mode = os.environ.get("AISCI_API_MODE", "completions")
    llm = create_llm_client()  # reads model, api_mode, web_search, reasoning from env vars

    # Create directories; if nonroot cannot write to LOGS_DIR (e.g. base image only did chmod a+rw on /home/logs),
    # fall back to /home/agent/logs so the run can proceed.
    global LOGS_DIR
    try:
        os.makedirs(LOGS_DIR, mode=0o755, exist_ok=True)
        os.makedirs(f"{LOGS_DIR}/subagent_logs", mode=0o755, exist_ok=True)
    except PermissionError:
        LOGS_DIR = "/home/agent/logs"
        os.makedirs(LOGS_DIR, mode=0o755, exist_ok=True)
        os.makedirs(f"{LOGS_DIR}/subagent_logs", mode=0o755, exist_ok=True)
        logger.warning("Using LOGS_DIR fallback", logs_dir=LOGS_DIR, reason="Permission denied on default LOGS_DIR")
    for d in ["/home/agent", LOGS_DIR, f"{LOGS_DIR}/subagent_logs", "/home/code", "/home/submission"]:
        os.makedirs(d, mode=0o755, exist_ok=True)

    # Initialise git in /home/code
    shell.send_command("cd /home/code && git init && git add -A && git commit -m 'init' --allow-empty", timeout=30)

    # Dump environment info (to both /home/agent and LOGS_DIR)
    env_info = {
        "model": model,
        "api_mode": api_mode,
        "web_search": os.environ.get("AISCI_WEB_SEARCH", "false"),
        "reasoning_effort": os.environ.get("AISCI_REASONING_EFFORT", ""),
        "time_limit": time_limit,
        "max_steps": max_steps,
        "hardware": hardware,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "competition_id": os.environ.get("COMPETITION_ID", "unknown"),
        "file_as_bus": file_as_bus,
    }
    env_json = json.dumps(env_info, indent=2)
    shell.write_file("/home/agent/env.json", env_json)
    shell.write_file(f"{LOGS_DIR}/env.json", env_json)

    # ---- Build tools ----
    start_time = time.time()

    tools: list[Tool] = [
        # Direct tools (information gathering)
        BashToolWithTimeout(default_timeout=MAIN_BASH_DEFAULT_TIMEOUT, max_timeout=MAIN_BASH_MAX_TIMEOUT),
        PythonTool(default_timeout=MAIN_BASH_DEFAULT_TIMEOUT, max_timeout=MAIN_BASH_MAX_TIMEOUT),
        ReadFileChunkTool(),
        SearchFileTool(),

        # Subagent adapter tools
        AnalyzeDataTool(shell, llm),
        PrioritizeTasksTool(shell, llm),
        ImplementTool(shell, llm, file_as_bus=file_as_bus),
        RunExperimentTool(shell, llm, file_as_bus=file_as_bus),

        # Generic subagent (explore/plan/general — spawn_subagent)
        SpawnSubagentTool(shell, llm),

        # Completion
        SubmitTool(time_limit=time_limit, start_time=start_time, llm=llm),
    ]

    tool_schemas = [t.get_tool_schema() for t in tools]
    tool_map = {t.name(): t for t in tools}

    # ---- Build initial messages ----
    comp_id = os.environ.get("COMPETITION_ID", "").strip()
    gpu_line = f"**Hardware**: {hardware}" if hardware != "unknown" else "**Hardware**: Check with `nvidia-smi`"
    task_prompt = (
        f"You have **{_fmt(time_limit)}** to solve this Kaggle competition.\n\n"
        f"{gpu_line}\n\n"
        "Competition data is in `/home/data/`. Read `/home/data/description.md` first.\n"
        "Your code goes in `/home/code/` (git repo). Final submission goes to `/home/submission/submission.csv`.\n\n"
        "Start by analysing the data and creating a prioritized task list, then implement and validate."
    )
    messages: list[dict] = [
        {"role": "system", "content": main_agent_system_prompt_for_run(file_as_bus)},
        {"role": "user", "content": task_prompt},
    ]

    # Log paths — written to LOGS_DIR so MLE-Bench extracts them per-task
    convo_jsonl_path = f"{LOGS_DIR}/conversation.jsonl"
    agent_log_path = f"{LOGS_DIR}/agent.log"
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # Write initial agent.log
    log_messages_to_file(messages, agent_log_path)

    # ---- Main loop ----
    impl_count = 0
    exp_count = 0
    submitted = False
    last_summary = None  # for incremental context summarization (strategy=summary)
    for step in range(1, max_steps + 1):
        # Exclude API retry-wait time from the agent's effective budget,
        # use_real_time_limit=True: exclude API retry wait from agent budget.
        elapsed = max(time.time() - start_time - llm.total_retry_time, 0.0)

        if elapsed >= time_limit:
            logger.info("Time limit reached", step=step, elapsed=elapsed)
            # Emergency: copy any submission.csv if it exists in code directory
            _emergency_finalize(shell)
            break

        # Periodic reminder
        if step > 1 and step % reminder_freq == 0:
            try:
                sub_check = shell.send_command(
                    "test -f /home/submission/submission.csv && echo EXISTS || echo MISSING",
                    timeout=5
                )
                submission_exists = "EXISTS" in (sub_check or "")
            except Exception:
                submission_exists = False
            reminder = _build_reminder(step, elapsed, time_limit, impl_count, exp_count, submission_exists)
            messages.append({"role": "user", "content": reminder})

        # ---- LLM call ----
        try:
            resp = llm.chat(messages, tools=tool_schemas)
        except ContentPolicyError as e:
            # o-series safety filter — fail immediately to preserve Azure quota.
            # The triggering messages have already been dumped by the LLM client.
            logger.error(
                "Content policy violation — stopping orchestrator",
                step=step, dump=e.dump_path,
            )
            break
        except ContextLengthError as _ctx_err:
            summary_succeeded_in_loop = False  # set True only when adaptive summary + retry succeeds
            # strategy=summary: 优先 summary，失败时回退到 prune（保证程序不崩）
            if context_reduce_strategy == "summary":
                system_msgs = [m for m in messages if m.get("role") == "system"]
                non_system = [m for m in messages if m.get("role") != "system"]
                first_user = non_system[0] if (non_system and non_system[0].get("role") == "user") else None
                rest = non_system[1:] if first_user else non_system
                turns = parse_rest_into_turns(rest)
                num_turns = len(turns)
                if num_turns < summary_min_turns:
                    logger.info(
                        "Context length exceeded — too few turns for summary, falling back to prune",
                        step=step, num_turns=num_turns, min_turns=summary_min_turns,
                    )
                    if _ctx_err.prune_individual:
                        messages = prune_messages_individual(
                            messages,
                            max_tokens_per_message=llm.config.context_window,
                        )
                    messages = prune_messages(messages)
                else:
                    _summary_max_ratio = 0.95
                    _summary_ratio_step = 0.1
                    task_content = (first_user or {}).get("content") or ""
                    if isinstance(task_content, list):
                        task_content = " ".join(
                            item.get("text", "") for item in task_content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                    task_for_prompt = (task_content[:2000] + "\n(task description truncated.)") if len(task_content) > 2000 else task_content
                    original_messages = messages
                    ratio = summary_segment_ratio
                    while ratio <= _summary_max_ratio:
                        target_drop_turns = max(1, min(int(num_turns * ratio), num_turns - 1))
                        segment_messages = [m for t in turns[:target_drop_turns] for m in t]
                        kept_tail_messages = [m for t in turns[target_drop_turns:] for m in t]
                        segment_text = serialize_segment_messages(
                            segment_messages, segment_max_chars=summary_segment_max_chars
                        )
                        if summary_incremental and last_summary:
                            prompt = SUMMARY_INCREMENTAL_PROMPT.format(
                                task=task_for_prompt, last_summary=last_summary, segment=segment_text
                            )
                        else:
                            prompt = SUMMARY_FIRST_TIME_PROMPT.format(task=task_for_prompt, segment=segment_text)
                        summary_req_messages = [{"role": "user", "content": prompt}]
                        summary_raw = ""
                        summary_text = ""
                        try:
                            summary_resp = llm.chat(summary_req_messages, tools=None)
                            summary_raw = (summary_resp.text_content or "").strip()
                            if summary_raw and len(summary_raw) >= 50:
                                if "Essential Information:" in summary_raw:
                                    summary_text = summary_raw.split("Essential Information:", 1)[-1].strip()
                                else:
                                    summary_text = summary_raw
                                if len(summary_text) > 4000:
                                    summary_text = summary_text[:3000] + "\n(summary truncated.)"
                                summary_user_content = SUMMARY_USER_INTRO + "\n\nSummary:\n" + summary_text
                                summary_user_msg = {"role": "user", "content": summary_user_content}
                                messages = system_msgs + ([first_user] if first_user else []) + [summary_user_msg] + kept_tail_messages
                                try:
                                    sub_check = shell.send_command(
                                        "test -f /home/submission/submission.csv && echo EXISTS || echo MISSING",
                                        timeout=5,
                                    )
                                    submission_exists = "EXISTS" in (sub_check or "")
                                except Exception:
                                    submission_exists = False
                                elapsed_after = max(time.time() - start_time - llm.total_retry_time, 0.0)
                                reminder = _build_reminder(
                                    step, elapsed_after, time_limit, impl_count, exp_count, submission_exists
                                )
                                messages.append({"role": "user", "content": reminder})
                                resp = llm.chat(messages, tools=tool_schemas)
                                last_summary = summary_text
                                summary_succeeded_in_loop = True
                                logger.info(
                                    "Context summarization succeeded; replaced N turns with summary",
                                    step=step, N=target_drop_turns, ratio_pct=int(ratio * 100),
                                )
                                try:
                                    summary_log_path = os.path.join(LOGS_DIR, "context_summary_requests.jsonl")
                                    record = {
                                        "step": step,
                                        "N": target_drop_turns,
                                        "ratio_pct": int(ratio * 100),
                                        "num_turns": num_turns,
                                        "segment_chars": len(segment_text),
                                        "summary_chars": len(summary_text),
                                        "prompt_preview": prompt[:500] + ("..." if len(prompt) > 500 else ""),
                                        "summary_preview": summary_text[:1000] + ("..." if len(summary_text) > 1000 else ""),
                                    }
                                    with open(summary_log_path, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                except Exception as _e:
                                    logger.debug("Failed to write context_summary_requests.jsonl", err=str(_e))
                                break
                            else:
                                raise ValueError("Summary response empty or too short")
                        except ContextLengthError:
                            logger.info(
                                "Context still over limit after summary at ratio %d%% — retrying with higher ratio (step=%s)",
                                int(ratio * 100), step,
                            )
                            ratio += _summary_ratio_step
                            continue
                        except Exception as e:
                            logger.warning(
                                "Context summarization failed (reason: %s); falling back to prune (step=%s).",
                                str(e)[:200], step,
                            )
                            break
                    if not summary_succeeded_in_loop:
                        logger.info(
                            "Context length exceeded — summary failed at all ratios, falling back to prune",
                            step=step,
                        )
                        if _ctx_err.prune_individual:
                            messages = prune_messages_individual(
                                original_messages,
                                max_tokens_per_message=llm.config.context_window,
                            )
                            messages = prune_messages(messages)
                        else:
                            messages = prune_messages(original_messages)
            else:
                # context_reduce_strategy == "prune": 走 prune 路径
                if _ctx_err.prune_individual:
                    logger.warning(
                        "Context length exceeded — truncating individual messages",
                        step=step,
                        context_window=llm.config.context_window,
                    )
                    messages = prune_messages_individual(
                        messages,
                        max_tokens_per_message=llm.config.context_window,
                    )
                messages = prune_messages(messages)
            # after context reduction (prune or summary), inject reminder and retry unless we already succeeded in summary loop
            if not summary_succeeded_in_loop:
                try:
                    sub_check = shell.send_command(
                        "test -f /home/submission/submission.csv && echo EXISTS || echo MISSING",
                        timeout=5,
                    )
                    submission_exists = "EXISTS" in (sub_check or "")
                except Exception:
                    submission_exists = False
                elapsed_after = max(time.time() - start_time - llm.total_retry_time, 0.0)
                reminder = _build_reminder(
                    step, elapsed_after, time_limit, impl_count, exp_count, submission_exists
                )
                messages.append({"role": "user", "content": reminder})
                try:
                    resp = llm.chat(messages, tools=tool_schemas)
                except ContextLengthError:
                    logger.error("Context length exceeded even after pruning — aborting", step=step)
                    break
                except ContentPolicyError as e:
                    logger.error("Content policy violation after pruning", step=step, dump=e.dump_path)
                    break
                except Exception as e:
                    logger.error("LLM call failed after pruning", step=step, err=str(e))
                    break
        except BadRequestError as e:
            # Non-safety BadRequestErrors (e.g., -4003 "No tool output found"):
            # fix message consistency first, then retry.
            error_code = str(getattr(e, "code", "") or "")
            logger.warning(
                "BadRequestError — fixing message consistency",
                step=step, error_code=error_code, err=str(e)[:200],
            )
            messages = fix_message_consistency(messages)
            try:
                resp = llm.chat(messages, tools=tool_schemas)
            except Exception as e2:
                logger.error("LLM call failed after consistency fix", step=step, err=str(e2))
                break
        except Exception as e:
            logger.error("LLM call failed", step=step, err=str(e))
            time.sleep(5)
            try:
                resp = llm.chat(messages, tools=tool_schemas)
            except Exception as e2:
                logger.error("LLM call failed on retry", step=step, err=str(e2))
                break

        # Log model response (structured JSONL)
        log_model_response_event(
            convo_path=convo_jsonl_path,
            run_id=run_id,
            step=step,
            n_input_messages=len(messages),
            text_content=resp.text_content,
            tool_calls=[{"id": tc.call_id, "name": tc.name, "args": tc.arguments} for tc in resp.tool_calls],
            usage=resp.usage,
            reasoning_content=resp.reasoning_content,
        )

        # Build assistant message. reasoning_content is added only when the
        # response has it: Completions API models that return it (GLM-5, Kimi
        # use the same field name) need it written back; others (e.g. MiniMax
        # uses reasoning_details, not yet handled here) do not get this key.
        asst_msg: dict[str, Any] = {"role": "assistant", "content": resp.text_content}
        if resp.reasoning_content:
            asst_msg["reasoning_content"] = resp.reasoning_content
        if resp.tool_calls:
            asst_msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    **({"extra_content": tc.extra_content} if tc.extra_content else {}),
                }
                for tc in resp.tool_calls
            ]
        # Responses API: store raw response.output so next request can inject
        # it per official guide (platform.openai.com/docs/guides/reasoning):
        # "pass in all output items from a previous response into the input
        # of a new one."
        raw = getattr(resp, "raw_message", None)
        if isinstance(raw, list):
            asst_msg["_response_output"] = raw
        messages.append(asst_msg)

        # Update human-readable agent.log
        log_messages_to_file(messages, agent_log_path)

        # GLM-5 / DeepSeek-R1: append reasoning chain as a separate box so it
        # appears in agent.log immediately after the assistant reply.
        if resp.reasoning_content:
            reasoning_lines = _box(
                "reasoning_content",
                _short(resp.reasoning_content, 600),
            )
            try:
                with open(agent_log_path, "a", encoding="utf-8") as _f:
                    _f.write("\n".join(reasoning_lines) + "\n")
            except Exception:
                pass

        # No tool calls — prompt continuation
        if not resp.tool_calls:
            if not resp.text_content:
                messages.append({"role": "user", "content": "Please continue. Use your tools to make progress."})
            continue

        # ---- Execute tool calls ----
        for tc in resp.tool_calls:
            tool = tool_map.get(tc.name)
            if not tool:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": f"Error: unknown tool '{tc.name}'. Available: {list(tool_map.keys())}",
                })
                continue

            # Track impl/exp counts
            if tc.name == "implement":
                impl_count += 1
            elif tc.name == "run_experiment":
                exp_count += 1

            try:
                result = tool.execute(shell, **tc.arguments)

                # Check for submit — only accepted when result contains the acceptance marker
                if tc.name == "submit" and "✅ Submission accepted" in result:
                    submitted = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": result,
                    })
                    logger.info("Submission accepted", step=step)
                    break

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": str(result),
                })
                if tc.name in {"implement", "run_experiment"}:
                    _snapshot_submission(shell, reason=tc.name)
            except Exception as e:
                logger.error("Tool execution error", tool=tc.name, err=str(e))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": f"Error executing {tc.name}: {e}",
                })

            log_tool_result_event(
                convo_path=convo_jsonl_path,
                run_id=run_id,
                step=step,
                tool_name=tc.name,
                tool_call_id=tc.call_id,
                result_preview=messages[-1].get("content", ""),
            )
            # Update human-readable agent.log after each tool result
            log_messages_to_file(messages, agent_log_path)

        if submitted:
            break

    # ---- Finalization ----
    _finalize(shell, llm, start_time, impl_count, exp_count)
    wall_elapsed = time.time() - start_time
    real_elapsed = max(wall_elapsed - llm.total_retry_time, 0.0)
    logger.info("Orchestrator finished",
                steps=step,
                wall_elapsed=wall_elapsed,
                real_elapsed=real_elapsed,
                total_retry_time=llm.total_retry_time,
                tokens=llm.total_tokens, submitted=submitted)


def _emergency_finalize(shell: ShellInterface) -> None:
    """Last-resort: copy any submission.csv found in code dir to submission dir."""
    candidates = [
        "/home/code/submission.csv",
        "/home/code/output/submission.csv",
        "/home/code/submissions/submission.csv",
    ]
    if os.path.exists("/home/submission/submission.csv"):
        return
    for path in candidates:
        if os.path.exists(path):
            shell.send_command(f"cp {path} /home/submission/submission.csv", timeout=10)
            logger.info("Emergency: copied submission.csv", src=path)
            return


def _snapshot_submission(shell: ShellInterface, reason: str) -> None:
    """
    Keep immutable submission snapshots to prevent accidental regression/overwrite.
    """
    src = "/home/submission/submission.csv"
    if not os.path.exists(src):
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_reason = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (reason or "snapshot"))
    dst_dir = "/home/submission/candidates"
    dst = f"{dst_dir}/submission_{ts}_{safe_reason}.csv"
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shell.send_command(f"cp {src} {dst}", timeout=20)
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "src": src,
            "dst": dst,
        }
        reg = "/home/submission/submission_registry.jsonl"
        with open(reg, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to snapshot submission", reason=reason, err=str(e))


def _finalize(shell: ShellInterface, llm: LLMClient, start_time: float,
              impl_count: int, exp_count: int) -> None:
    """Write final summary and copy important state files to LOGS_DIR."""
    _emergency_finalize(shell)

    summary = {
        "runtime_seconds": time.time() - start_time,
        "total_tokens": llm.total_tokens,
        "total_retry_time": llm.total_retry_time,
        "impl_calls": impl_count,
        "exp_calls": exp_count,
        "submission_exists": os.path.exists("/home/submission/submission.csv"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_json = json.dumps(summary, indent=2)
    try:
        shell.write_file("/home/agent/summary.json", summary_json)
        shell.write_file(f"{LOGS_DIR}/summary.json", summary_json)
    except Exception:
        pass

    # Copy important state files to LOGS_DIR for post-run inspection
    for src in [
        "/home/agent/impl_log.md",
        "/home/agent/exp_log.md",
        "/home/agent/prioritized_tasks.md",
        "/home/agent/analysis/summary.md",
        "/home/submission/submission_registry.jsonl",
    ]:
        if os.path.exists(src):
            try:
                dst = f"{LOGS_DIR}/{os.path.basename(src)}"
                shell.send_command(f"cp {src} {dst}", timeout=10)
            except Exception:
                pass


if __name__ == "__main__":
    main()
