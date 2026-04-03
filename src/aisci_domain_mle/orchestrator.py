"""
AI Scientist Orchestrator for MLE-Bench — mirrors PaperBench's AiScientistSolver.

Key patterns preserved from PaperBench:
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

from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from aisci_domain_mle.local_runtime_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import structlog

from openai import BadRequestError
from aisci_agent_runtime.llm_client import LLMClient, ContextLengthError, ContentPolicyError
from aisci_agent_runtime.shell_interface import ShellInterface
from aisci_agent_runtime.subagents.base import SubagentConfig, SubagentOutput, SubagentStatus, prune_messages, prune_messages_individual, fix_message_consistency, _fmt
from aisci_domain_mle.constants import is_file_as_bus_enabled
from aisci_domain_mle.orchestrator_runtime import (
    OrchestratorPaths,
    OrchestratorRuntimeConfig,
    build_task_prompt,
    create_orchestrator_llm,
    load_runtime_config_from_env,
)
from aisci_domain_mle.subagents.configs import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_PRIORITIZATION_CONFIG,
    DEFAULT_IMPLEMENTATION_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    EXPERIMENT_VALIDATE_TIME_LIMIT,
    MAIN_BASH_DEFAULT_TIMEOUT,
    MAIN_BASH_MAX_TIMEOUT,
)
from aisci_domain_mle.subagents.analysis import DataAnalysisSubagent
from aisci_domain_mle.subagents.prioritization import PrioritizationSubagent
from aisci_domain_mle.subagents.implementation import ImplementationSubagent
from aisci_domain_mle.subagents.experiment import ExperimentSubagent
from aisci_agent_runtime.tools.base import Tool
from aisci_agent_runtime.tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
)
from aisci_domain_mle.tools.spawn_subagent_tool import SpawnSubagentTool
from aisci_domain_mle.prompts.templates import (
    SUMMARY_FIRST_TIME_PROMPT,
    SUMMARY_INCREMENTAL_PROMPT,
    main_agent_system_prompt_for_run,
)
from aisci_agent_runtime.log_utils import log_messages_to_file, log_model_response_event, log_tool_result_event, _box, _short
from aisci_agent_runtime.summary_utils import (
    parse_rest_into_turns,
    serialize_segment_messages,
    SUMMARY_USER_INTRO,
)

logger = structlog.stdlib.get_logger(component=__name__)

LOGS_DIR = os.environ.get("LOGS_DIR", "/home/logs")


def _mapped_path(shell: ShellInterface | None, path: str) -> Path:
    mapper = getattr(shell, "mapped", None)
    if callable(mapper):
        try:
            mapped = mapper(path)
        except Exception:
            mapped = None
        if mapped is not None:
            return Path(mapped)
    return Path(path)


def _command_output(result: Any) -> str:
    if hasattr(result, "output"):
        value = getattr(result, "output")
        if isinstance(value, str):
            return value
        return str(value or "")
    if result is None:
        return ""
    return str(result)


def _path_exists(shell: ShellInterface | None, path: str) -> bool:
    if shell is not None and hasattr(shell, "file_exists"):
        try:
            return bool(shell.file_exists(path))
        except Exception:
            pass
    return _mapped_path(shell, path).exists()


def _ensure_local_dir(shell: ShellInterface | None, path: str) -> Path:
    resolved = _mapped_path(shell, path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _append_text(shell: ShellInterface | None, path: str, content: str) -> None:
    if shell is not None and hasattr(shell, "append_file"):
        try:
            shell.append_file(path, content)
            return
        except Exception:
            pass
    resolved = _mapped_path(shell, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(content)


def _prepare_runtime_dirs(
    shell: ShellInterface | None,
    paths: OrchestratorPaths,
) -> OrchestratorPaths:
    """Create runtime directories and preserve upstream LOGS_DIR fallback."""
    global LOGS_DIR

    try:
        _ensure_local_dir(shell, paths.logs_dir)
        _ensure_local_dir(shell, f"{paths.logs_dir}/subagent_logs")
    except PermissionError:
        fallback_logs_dir = "/home/agent/logs"
        _ensure_local_dir(shell, fallback_logs_dir)
        _ensure_local_dir(shell, f"{fallback_logs_dir}/subagent_logs")
        paths = replace(paths, logs_dir=fallback_logs_dir)
        logger.warning(
            "Using LOGS_DIR fallback",
            logs_dir=paths.logs_dir,
            reason="Permission denied on default LOGS_DIR",
        )

    LOGS_DIR = paths.logs_dir
    os.environ["LOGS_DIR"] = paths.logs_dir
    for directory in (
        paths.home_root,
        paths.data_dir,
        paths.agent_dir,
        paths.code_dir,
        paths.submission_dir,
        paths.logs_dir,
        f"{paths.logs_dir}/subagent_logs",
    ):
        _ensure_local_dir(shell, directory)
    return paths

# ====================================================================== #
# Subagent Adapter Tools (main agent dispatches to subagents via these)
# ====================================================================== #


class AnalyzeDataTool(Tool):
    """Main-agent tool to dispatch the Data Analysis Subagent."""

    def __init__(self, shell: ShellInterface, llm: LLMClient, paths: OrchestratorPaths):
        self._shell = shell
        self._llm = llm
        self._paths = paths
        self._session_counter = 0

    def name(self) -> str:
        return "analyze_data"

    def execute(self, shell, **kwargs) -> str:
        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{self._paths.logs_dir}/subagent_logs/analysis_{self._session_counter:03d}_{ts}"
        local_session_dir = _ensure_local_dir(self._shell, session_dir)

        config = SubagentConfig(
            max_steps=DEFAULT_ANALYSIS_CONFIG.max_steps,
            time_limit=DEFAULT_ANALYSIS_CONFIG.time_limit,
            reminder_freq=DEFAULT_ANALYSIS_CONFIG.reminder_freq,
            log_dir=str(local_session_dir),
        )

        context = (
            f"Analyze the competition data in {self._paths.data_dir}/.\n"
            f"Read {self._paths.description_path}, examine all data files, "
            f"and produce {self._paths.analysis_summary_path}.\n"
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

    def __init__(self, shell: ShellInterface, llm: LLMClient, paths: OrchestratorPaths):
        self._shell = shell
        self._llm = llm
        self._paths = paths
        self._session_counter = 0

    def name(self) -> str:
        return "prioritize_tasks"

    def execute(self, shell, **kwargs) -> str:
        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{self._paths.logs_dir}/subagent_logs/prio_{self._session_counter:03d}_{ts}"
        local_session_dir = _ensure_local_dir(self._shell, session_dir)

        config = SubagentConfig(
            max_steps=DEFAULT_PRIORITIZATION_CONFIG.max_steps,
            time_limit=DEFAULT_PRIORITIZATION_CONFIG.time_limit,
            reminder_freq=DEFAULT_PRIORITIZATION_CONFIG.reminder_freq,
            log_dir=str(local_session_dir),
        )

        # Inject data analysis summary if available
        context_parts = [
            "Create a prioritized task list for this Kaggle competition.\n"
            f"Read {self._paths.description_path} and {self._paths.sample_submission_path}.\n"
        ]
        analysis_path = self._paths.analysis_summary_path
        if _path_exists(self._shell, analysis_path):
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

    Mirrors PaperBench's ImplementationTool with:
    - Session management (timestamped directories)
    - Context injection from recent exp_log
    - Mode support (full / fix / explore / refine / ensemble)
    """

    def __init__(self, shell: ShellInterface, llm: LLMClient, paths: OrchestratorPaths, file_as_bus: bool = True):
        self._shell = shell
        self._llm = llm
        self._paths = paths
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
        session_dir = f"{self._paths.logs_dir}/subagent_logs/impl_{self._session_counter:03d}_{ts}"
        local_session_dir = _ensure_local_dir(self._shell, session_dir)

        if self._file_as_bus:
            _write_session_separator(
                self._shell,
                self._paths.impl_log_path,
                f"Implement Session {self._session_counter}",
                f"Mode: {mode} | Task: {task or '(full prioritization)'}",
            )

        mode = (mode or "full").strip().lower()
        if mode not in {"full", "fix", "explore", "refine", "ensemble"}:
            mode = "full"

        # Determine time budget: mirrors legacy behavior.
        if time_budget is not None:
            tl = time_budget
        else:
            tl = DEFAULT_IMPLEMENTATION_CONFIG.time_limit if mode == "full" else 7200

        config = SubagentConfig(
            max_steps=DEFAULT_IMPLEMENTATION_CONFIG.max_steps,
            time_limit=tl,
            reminder_freq=DEFAULT_IMPLEMENTATION_CONFIG.reminder_freq,
            log_dir=str(local_session_dir),
        )

        # Build context for the subagent
        context_parts = []
        if mode == "full":
            context_parts.append(
                "## Mode: Full Implementation\n"
                f"Read {self._paths.prioritized_tasks_path} and work through tasks autonomously "
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
                        f"Apply targeted fixes. Read the context and experiment log ({self._paths.exp_log_path}) to determine what to fix.\n"
                    )
                else:
                    context_parts.append(
                        "## Mode: Fix\n"
                        "Apply targeted fixes using the context in this message, git history, and the current codebase.\n"
                    )
        if task and mode == "full":
            context_parts.append(f"## Focus\n{task}\n")
        if context:
            context_parts.append(f"## Context from previous rounds\n{context}\n")

        # Inject last experiment session for context continuity (PaperBench pattern: extract last session)
        exp_log_path = self._paths.exp_log_path
        if self._file_as_bus and _path_exists(self._shell, exp_log_path):
            try:
                exp_log_cmd = (
                    f"LAST_SEP=$(grep -n '^=== Experiment Session' {shlex.quote(exp_log_path)} 2>/dev/null "
                    f"| tail -1 | cut -d: -f1); "
                    f"if [ -n \"$LAST_SEP\" ]; then sed -n \"${{LAST_SEP}},\\$p\" {shlex.quote(exp_log_path)}; "
                    f"else cat {shlex.quote(exp_log_path)} 2>/dev/null || echo '(no experiment log yet)'; fi"
                )
                exp_log_content = _command_output(self._shell.send_command(exp_log_cmd, timeout=10)).strip()
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

    Mirrors PaperBench's ExperimentTool with:
    - Session management (timestamped directories)
    - Context injection from recent impl_log
    - Mode support (full / validate)
    """

    def __init__(self, shell: ShellInterface, llm: LLMClient, paths: OrchestratorPaths, file_as_bus: bool = True):
        self._shell = shell
        self._llm = llm
        self._paths = paths
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
        session_dir = f"{self._paths.logs_dir}/subagent_logs/exp_{self._session_counter:03d}_{ts}"
        local_session_dir = _ensure_local_dir(self._shell, session_dir)

        if self._file_as_bus:
            _write_session_separator(
                self._shell,
                self._paths.exp_log_path,
                f"Experiment Session {self._session_counter}",
                f"Mode: {mode} | Task: {task}",
            )

        # Determine time budget: mirrors legacy behavior.
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
            log_dir=str(local_session_dir),
        )

        # Build context
        context_parts = [f"## Task\n{task}\n", f"## Mode: {mode}\n"]

        # Inject last implementation session for context continuity (PaperBench pattern: extract last session)
        impl_log_path = self._paths.impl_log_path
        if self._file_as_bus and _path_exists(self._shell, impl_log_path):
            try:
                impl_log_cmd = (
                    f"LAST_SEP=$(grep -n '^=== Implement Session' {shlex.quote(impl_log_path)} 2>/dev/null "
                    f"| tail -1 | cut -d: -f1); "
                    f"if [ -n \"$LAST_SEP\" ]; then sed -n \"${{LAST_SEP}},\\$p\" {shlex.quote(impl_log_path)}; "
                    f"else cat {shlex.quote(impl_log_path)} 2>/dev/null || echo '(no implementation log yet)'; fi"
                )
                impl_log_content = _command_output(self._shell.send_command(impl_log_cmd, timeout=10)).strip()
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
    Final submission tool with pre-checks (mirrors PaperBench's submit pre-checks).

    Hard blocks (non-bypassable — attempt not counted):
    - submission.csv does not exist
    - column names don't match sample_submission.csv
    - row count doesn't match sample_submission.csv

    Soft warnings (shown once on first attempt, bypassed by calling submit() again):
    - NaN values in submission
    - Early submission (> 50% time remaining, first attempt only — PaperBench pattern)
    """

    def __init__(self, time_limit: int, start_time: float, paths: OrchestratorPaths, llm: "LLMClient | None" = None):
        self._time_limit = time_limit
        self._start_time = start_time
        self._paths = paths
        self._llm = llm  # used to exclude retry-wait from elapsed (use_real_time_limit)
        self._submit_attempts = 0  # counts real (non-blocked) attempts

    def _real_elapsed(self) -> float:
        """Wall-clock elapsed minus time spent in API retries (PaperBench use_real_time_limit)."""
        elapsed = time.time() - self._start_time
        if self._llm is not None:
            elapsed -= self._llm.total_retry_time
        return max(elapsed, 0.0)

    def name(self) -> str:
        return "submit"

    def execute(self, shell, confirm: str = "yes", **kwargs) -> str:
        warnings = []

        # Hard block 1: submission.csv must exist
        if not _path_exists(shell, self._paths.submission_csv_path):
            # Don't count this as a real attempt (PaperBench pattern)
            return (
                "SUBMISSION BLOCKED:\n\n"
                f"❌ HARD BLOCK: {self._paths.submission_csv_path} does not exist.\n"
                "You MUST generate a valid submission.csv before submitting.\n"
                "Use implement() or bash to create it.\n\n"
                "Fix this issue first. submit() will remain blocked until resolved."
            )

        # Format validation against sample_submission
        if _path_exists(shell, self._paths.sample_submission_path):
            try:
                import pandas as pd
                sub = pd.read_csv(_mapped_path(shell, self._paths.submission_csv_path))
                sample = pd.read_csv(_mapped_path(shell, self._paths.sample_submission_path))

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

        # Early submission warning — soft, only on FIRST real attempt (PaperBench pattern)
        if self._submit_attempts == 1:
            elapsed = self._real_elapsed()
            time_ratio = elapsed / self._time_limit if self._time_limit > 0 else 1.0
            remaining_hours = (self._time_limit - elapsed) / 3600.0
            if time_ratio < 0.5:
                warnings.append(
                    f"⚠ EARLY SUBMISSION: Only {time_ratio*100:.0f}% of time used "
                    f"({remaining_hours:.1f} hours remaining). "
                    "You have significant time to improve your score. "
                    f"Review {self._paths.prioritized_tasks_path} for P1/P2 tasks you can still tackle."
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


def _write_session_separator(
    shell: ShellInterface | None,
    path: str,
    title: str,
    details: str,
) -> None:
    """Write a session separator matching PaperBench format so grep-based extraction works.

    The separator line must start with '=== <Title>' so that the last-session extraction
    in ImplementTool and RunExperimentTool (using grep '^=== Implement Session' /
    grep '^=== Experiment Session') can locate it correctly.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    separator = f"\n=== {title} ===\n**Time**: {ts} | {details}\n\n"
    _append_text(shell, path, separator)


def _build_reminder(
    step: int, elapsed: float, time_limit: int,
    impl_count: int, exp_count: int,
    submission_exists: bool = False,
) -> str:
    """Build a periodic reminder — mirrors PaperBench's 4-stage + balance logic."""
    remaining = time_limit - elapsed
    pct = elapsed / time_limit * 100 if time_limit > 0 else 0

    parts = [
        f"⏱ Step {step} | Time: {_fmt(elapsed)}/{_fmt(time_limit)} ({pct:.0f}%) | "
        f"Remaining: {_fmt(remaining)} | "
        f"Impl calls: {impl_count} | Exp calls: {exp_count}"
    ]

    # === Stage-based reminders (PaperBench 4-stage pattern) ===
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

    # === Implement/Experiment balance warnings (PaperBench pattern) ===
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

class EmbeddedMLEEngine:
    def __init__(
        self,
        *,
        config: OrchestratorRuntimeConfig,
        shell: ShellInterface,
        llm: LLMClient,
    ) -> None:
        self.config = config
        self.shell = shell
        self.llm = llm

    def run(self) -> str:
        runtime = self.config
        paths = _prepare_runtime_dirs(self.shell, runtime.paths)

        logger.info(
            "Starting AI Scientist orchestrator",
            time_limit=runtime.time_limit,
            max_steps=runtime.max_steps,
            model=runtime.model,
            api_mode=runtime.api_mode,
            hardware=runtime.hardware,
            context_reduce_strategy=runtime.context_reduce_strategy,
            file_as_bus=runtime.file_as_bus,
            home_root=paths.home_root,
            logs_dir=paths.logs_dir,
        )

        self.shell.send_command(
            f"cd {shlex.quote(paths.code_dir)} && git init && git add -A && git commit -m 'init' --allow-empty",
            timeout=30,
        )

        env_info = {
            "model": runtime.model,
            "api_mode": runtime.api_mode,
            "web_search": os.environ.get("AISCI_WEB_SEARCH", "false"),
            "reasoning_effort": os.environ.get("AISCI_REASONING_EFFORT", ""),
            "time_limit": runtime.time_limit,
            "max_steps": runtime.max_steps,
            "hardware": runtime.hardware,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "competition_id": os.environ.get("COMPETITION_ID", "unknown"),
            "file_as_bus": runtime.file_as_bus,
            "paths": {
                "data_dir": paths.data_dir,
                "code_dir": paths.code_dir,
                "submission_dir": paths.submission_dir,
                "agent_dir": paths.agent_dir,
                "logs_dir": paths.logs_dir,
            },
        }
        env_json = json.dumps(env_info, indent=2)
        self.shell.write_file(paths.agent_env_path, env_json)
        self.shell.write_file(f"{paths.logs_dir}/env.json", env_json)

        start_time = time.time()
        tools: list[Tool] = [
            BashToolWithTimeout(default_timeout=MAIN_BASH_DEFAULT_TIMEOUT, max_timeout=MAIN_BASH_MAX_TIMEOUT),
            PythonTool(default_timeout=MAIN_BASH_DEFAULT_TIMEOUT, max_timeout=MAIN_BASH_MAX_TIMEOUT),
            ReadFileChunkTool(),
            SearchFileTool(),
            AnalyzeDataTool(self.shell, self.llm, paths),
            PrioritizeTasksTool(self.shell, self.llm, paths),
            ImplementTool(self.shell, self.llm, paths, file_as_bus=runtime.file_as_bus),
            RunExperimentTool(self.shell, self.llm, paths, file_as_bus=runtime.file_as_bus),
            SpawnSubagentTool(self.shell, self.llm),
            SubmitTool(time_limit=runtime.time_limit, start_time=start_time, paths=paths, llm=self.llm),
        ]

        tool_schemas = [tool.get_tool_schema() for tool in tools]
        tool_map = {tool.name(): tool for tool in tools}
        messages: list[dict] = [
            {"role": "system", "content": main_agent_system_prompt_for_run(runtime.file_as_bus)},
            {"role": "user", "content": build_task_prompt(runtime)},
        ]

        convo_jsonl_path = str(_mapped_path(self.shell, f"{paths.logs_dir}/conversation.jsonl"))
        agent_log_path = str(_mapped_path(self.shell, f"{paths.logs_dir}/agent.log"))
        run_id = time.strftime("%Y%m%d_%H%M%S")

        log_messages_to_file(messages, agent_log_path)

        impl_count = 0
        exp_count = 0
        submitted = False
        last_summary = None
        step = 0
        for step in range(1, runtime.max_steps + 1):
            # Exclude API retry-wait time from the agent's effective budget,
            # mirroring PaperBench's use_real_time_limit=True behaviour.
            elapsed = max(time.time() - start_time - self.llm.total_retry_time, 0.0)

            if elapsed >= runtime.time_limit:
                logger.info("Time limit reached", step=step, elapsed=elapsed)
                _emergency_finalize(self.shell, paths)
                break

            if step > 1 and step % runtime.reminder_freq == 0:
                try:
                    sub_check = self.shell.send_command(
                        f"test -f {shlex.quote(paths.submission_csv_path)} && echo EXISTS || echo MISSING",
                        timeout=5,
                    )
                    submission_exists = "EXISTS" in _command_output(sub_check)
                except Exception:
                    submission_exists = False
                reminder = _build_reminder(step, elapsed, runtime.time_limit, impl_count, exp_count, submission_exists)
                messages.append({"role": "user", "content": reminder})

            try:
                resp = self.llm.chat(messages, tools=tool_schemas)
            except ContentPolicyError as e:
                # o-series safety filter — fail immediately to preserve Azure quota.
                # The triggering messages have already been dumped by the LLM client.
                logger.error(
                    "Content policy violation — stopping orchestrator",
                    step=step, dump=e.dump_path,
                )
                break
            except ContextLengthError as _ctx_err:
                summary_succeeded_in_loop = False
                if runtime.context_reduce_strategy == "summary":
                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    non_system = [m for m in messages if m.get("role") != "system"]
                    first_user = non_system[0] if (non_system and non_system[0].get("role") == "user") else None
                    rest = non_system[1:] if first_user else non_system
                    turns = parse_rest_into_turns(rest)
                    num_turns = len(turns)
                    if num_turns < runtime.summary_min_turns:
                        logger.info(
                            "Context length exceeded — too few turns for summary, falling back to prune",
                            step=step, num_turns=num_turns, min_turns=runtime.summary_min_turns,
                        )
                        if _ctx_err.prune_individual:
                            messages = prune_messages_individual(
                                messages,
                                max_tokens_per_message=self.llm.config.prune_context_window,
                            )
                        messages = prune_messages(messages)
                    else:
                        summary_max_ratio = 0.95
                        summary_ratio_step = 0.1
                        task_content = (first_user or {}).get("content") or ""
                        if isinstance(task_content, list):
                            task_content = " ".join(
                                item.get("text", "") for item in task_content
                                if isinstance(item, dict) and item.get("type") == "text"
                            )
                        task_for_prompt = (
                            task_content[:2000] + "\n(task description truncated.)"
                            if len(task_content) > 2000
                            else task_content
                        )
                        original_messages = messages
                        ratio = runtime.summary_segment_ratio
                        while ratio <= summary_max_ratio:
                            target_drop_turns = max(1, min(int(num_turns * ratio), num_turns - 1))
                            segment_messages = [m for turn in turns[:target_drop_turns] for m in turn]
                            kept_tail_messages = [m for turn in turns[target_drop_turns:] for m in turn]
                            segment_text = serialize_segment_messages(
                                segment_messages,
                                segment_max_chars=runtime.summary_segment_max_chars,
                            )
                            if runtime.summary_incremental and last_summary:
                                prompt = SUMMARY_INCREMENTAL_PROMPT.format(
                                    task=task_for_prompt,
                                    last_summary=last_summary,
                                    segment=segment_text,
                                )
                            else:
                                prompt = SUMMARY_FIRST_TIME_PROMPT.format(
                                    task=task_for_prompt,
                                    segment=segment_text,
                                )
                            summary_req_messages = [{"role": "user", "content": prompt}]
                            try:
                                summary_resp = self.llm.chat(summary_req_messages, tools=None)
                                summary_raw = (summary_resp.text_content or "").strip()
                                if not summary_raw or len(summary_raw) < 50:
                                    raise ValueError("Summary response empty or too short")
                                if "Essential Information:" in summary_raw:
                                    summary_text = summary_raw.split("Essential Information:", 1)[-1].strip()
                                else:
                                    summary_text = summary_raw
                                if len(summary_text) > 4000:
                                    summary_text = summary_text[:3000] + "\n(summary truncated.)"
                                summary_user_content = SUMMARY_USER_INTRO + "\n\nSummary:\n" + summary_text
                                summary_user_msg = {"role": "user", "content": summary_user_content}
                                messages = (
                                    system_msgs
                                    + ([first_user] if first_user else [])
                                    + [summary_user_msg]
                                    + kept_tail_messages
                                )
                                try:
                                    sub_check = self.shell.send_command(
                                        f"test -f {shlex.quote(paths.submission_csv_path)} && echo EXISTS || echo MISSING",
                                        timeout=5,
                                    )
                                    submission_exists = "EXISTS" in _command_output(sub_check)
                                except Exception:
                                    submission_exists = False
                                elapsed_after = max(time.time() - start_time - self.llm.total_retry_time, 0.0)
                                reminder = _build_reminder(
                                    step,
                                    elapsed_after,
                                    runtime.time_limit,
                                    impl_count,
                                    exp_count,
                                    submission_exists,
                                )
                                messages.append({"role": "user", "content": reminder})
                                resp = self.llm.chat(messages, tools=tool_schemas)
                                last_summary = summary_text
                                summary_succeeded_in_loop = True
                                logger.info(
                                    "Context summarization succeeded; replaced N turns with summary",
                                    step=step, N=target_drop_turns, ratio_pct=int(ratio * 100),
                                )
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
                                _append_text(
                                    self.shell,
                                    f"{paths.logs_dir}/context_summary_requests.jsonl",
                                    json.dumps(record, ensure_ascii=False) + "\n",
                                )
                                break
                            except ContextLengthError:
                                logger.info(
                                    "Context still over limit after summary at ratio %d%% — retrying with higher ratio (step=%s)",
                                    int(ratio * 100), step,
                                )
                                ratio += summary_ratio_step
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
                                    max_tokens_per_message=self.llm.config.prune_context_window,
                                )
                            else:
                                messages = original_messages
                            messages = prune_messages(messages)
                else:
                    if _ctx_err.prune_individual:
                        logger.warning(
                            "Context length exceeded — truncating individual messages",
                            step=step,
                            context_window=self.llm.config.context_window,
                            prune_context_window=self.llm.config.prune_context_window,
                        )
                        messages = prune_messages_individual(
                            messages,
                            max_tokens_per_message=self.llm.config.prune_context_window,
                        )
                    messages = prune_messages(messages)

                if not summary_succeeded_in_loop:
                    try:
                        sub_check = self.shell.send_command(
                            f"test -f {shlex.quote(paths.submission_csv_path)} && echo EXISTS || echo MISSING",
                            timeout=5,
                        )
                        submission_exists = "EXISTS" in _command_output(sub_check)
                    except Exception:
                        submission_exists = False
                    elapsed_after = max(time.time() - start_time - self.llm.total_retry_time, 0.0)
                    reminder = _build_reminder(
                        step,
                        elapsed_after,
                        runtime.time_limit,
                        impl_count,
                        exp_count,
                        submission_exists,
                    )
                    messages.append({"role": "user", "content": reminder})
                    try:
                        resp = self.llm.chat(messages, tools=tool_schemas)
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
                error_code = str(getattr(e, "code", "") or "")
                logger.warning(
                    "BadRequestError — fixing message consistency",
                    step=step, error_code=error_code, err=str(e)[:200],
                )
                messages = fix_message_consistency(messages)
                try:
                    resp = self.llm.chat(messages, tools=tool_schemas)
                except Exception as e2:
                    logger.error("LLM call failed after consistency fix", step=step, err=str(e2))
                    break
            except Exception as e:
                logger.error("LLM call failed", step=step, err=str(e))
                time.sleep(5)
                try:
                    resp = self.llm.chat(messages, tools=tool_schemas)
                except Exception as e2:
                    logger.error("LLM call failed on retry", step=step, err=str(e2))
                    break

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
            raw = getattr(resp, "raw_message", None)
            if isinstance(raw, list):
                asst_msg["_response_output"] = raw
            messages.append(asst_msg)

            log_messages_to_file(messages, agent_log_path)

        # GLM-5 / DeepSeek-R1: append reasoning chain as a separate box so it
        # appears in agent.log immediately after the assistant reply.
            if resp.reasoning_content:
                reasoning_lines = _box(
                    "reasoning_content",
                    _short(resp.reasoning_content, 600),
                )
                try:
                    _append_text(self.shell, f"{paths.logs_dir}/agent.log", "\n".join(reasoning_lines) + "\n")
                except Exception:
                    pass

            if not resp.tool_calls:
                if not resp.text_content:
                    messages.append({"role": "user", "content": "Please continue. Use your tools to make progress."})
                continue

            for tc in resp.tool_calls:
                tool = tool_map.get(tc.name)
                if not tool:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": f"Error: unknown tool '{tc.name}'. Available: {list(tool_map.keys())}",
                    })
                    continue

                if tc.name == "implement":
                    impl_count += 1
                elif tc.name == "run_experiment":
                    exp_count += 1

                try:
                    result = tool.execute(self.shell, **tc.arguments)

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
                        _snapshot_submission(self.shell, paths, reason=tc.name)
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
                log_messages_to_file(messages, agent_log_path)

            if submitted:
                break

        _finalize(self.shell, self.llm, paths, start_time, impl_count, exp_count)
        wall_elapsed = time.time() - start_time
        real_elapsed = max(wall_elapsed - self.llm.total_retry_time, 0.0)
        logger.info(
            "Orchestrator finished",
            steps=step,
            wall_elapsed=wall_elapsed,
            real_elapsed=real_elapsed,
            total_retry_time=self.llm.total_retry_time,
            tokens=self.llm.total_tokens,
            submitted=submitted,
        )
        return (
            f"MLE host-side engine finished after {step} steps; "
            f"submitted={submitted}; real_elapsed={real_elapsed:.1f}s."
        )


def run(runtime: OrchestratorRuntimeConfig | None = None) -> None:
    """Run the AI Scientist orchestrator with an explicit runtime config."""

    runtime = runtime or load_runtime_config_from_env()
    engine = EmbeddedMLEEngine(
        config=runtime,
        shell=ShellInterface(working_dir=runtime.paths.home_root),
        llm=create_orchestrator_llm(runtime),
    )
    engine.run()

def main() -> None:
    run(load_runtime_config_from_env())


def _emergency_finalize(shell: ShellInterface, paths: OrchestratorPaths) -> None:
    """Last-resort: copy any submission.csv found in code dir to submission dir."""
    candidates = [
        f"{paths.code_dir}/submission.csv",
        f"{paths.code_dir}/output/submission.csv",
        f"{paths.code_dir}/submissions/submission.csv",
    ]
    if _path_exists(shell, paths.submission_csv_path):
        return
    for path in candidates:
        if _path_exists(shell, path):
            shell.send_command(
                f"cp {shlex.quote(path)} {shlex.quote(paths.submission_csv_path)}",
                timeout=10,
            )
            logger.info("Emergency: copied submission.csv", src=path)
            return


def _snapshot_submission(shell: ShellInterface, paths: OrchestratorPaths, reason: str) -> None:
    """
    Keep immutable submission snapshots to prevent accidental regression/overwrite.
    """
    src = paths.submission_csv_path
    if not _path_exists(shell, src):
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_reason = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (reason or "snapshot"))
    dst_dir = paths.submission_candidates_dir
    dst = f"{dst_dir}/submission_{ts}_{safe_reason}.csv"
    try:
        _ensure_local_dir(shell, dst_dir)
        shell.send_command(
            f"cp {shlex.quote(src)} {shlex.quote(dst)}",
            timeout=20,
        )
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "src": src,
            "dst": dst,
        }
        reg = paths.submission_registry_path
        _append_text(shell, reg, json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to snapshot submission", reason=reason, err=str(e))


def _finalize(shell: ShellInterface, llm: LLMClient, paths: OrchestratorPaths, start_time: float,
              impl_count: int, exp_count: int) -> None:
    """Write final summary and copy important state files to LOGS_DIR."""
    _emergency_finalize(shell, paths)

    summary = {
        "runtime_seconds": time.time() - start_time,
        "total_tokens": llm.total_tokens,
        "total_retry_time": llm.total_retry_time,
        "impl_calls": impl_count,
        "exp_calls": exp_count,
        "submission_exists": _path_exists(shell, paths.submission_csv_path),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_json = json.dumps(summary, indent=2)
    try:
        shell.write_file(paths.agent_summary_path, summary_json)
        shell.write_file(f"{paths.logs_dir}/summary.json", summary_json)
    except Exception:
        pass

    # Copy important state files to LOGS_DIR for post-run inspection
    for src in [
        paths.impl_log_path,
        paths.exp_log_path,
        paths.prioritized_tasks_path,
        paths.analysis_summary_path,
        paths.submission_registry_path,
    ]:
        if _path_exists(shell, src):
            try:
                dst = f"{paths.logs_dir}/{os.path.basename(src)}"
                shell.send_command(
                    f"cp {shlex.quote(src)} {shlex.quote(dst)}",
                    timeout=10,
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
