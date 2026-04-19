"""
AI Scientist Solver

An advanced solver for PaperBench that uses specialized subagents for
different phases of paper reproduction. Key improvements over BasicAgent:

1. **Systematic Paper Reading**: Dedicated subagents for thorough paper analysis
2. **Parallel Processing**: Multiple subagents can work concurrently
3. **Structured Extraction**: Algorithms, experiments, and baselines extracted systematically
4. **Phase-Based Workflow**: Clear progression through reading -> planning -> implementation

Based on case study findings:
- High-scoring cases read papers 1.7x more thoroughly
- Structured approach prevents missing critical details
- Time management through clear phases
"""

from __future__ import annotations

import json
import hashlib
import time

import blobfile as bf
import structlog
from openai import BadRequestError, LengthFinishReasonError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.responses.web_search_tool_param import WebSearchToolParam
from typing_extensions import override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import AGENT_DIR_CONFIG
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    OpenAIResponsesTurnCompleterConfig,
)
from paperbench.solvers.basicagent.logging import LoggableMessages
from paperbench.solvers.basicagent.log_utils import (
    log_model_response_event,
    log_tool_result_event,
)
from paperbench.solvers.basicagent.prompts.templates import DEFAULT_CONTINUE_MESSAGE
from paperbench.solvers.basicagent.tools import (
    PythonTool,
    ReadFileChunk,
    SearchFile,
    SubmitTool,
)
from paperbench.solvers.aiscientist.tools.basic_tool import BashToolWithTimeout
from paperbench.solvers.aiscientist.subagents.configs import (
    MAIN_AGENT_BASH_DEFAULT_TIMEOUT,
    MAIN_AGENT_BASH_MAX_TIMEOUT,
)
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.basicagent.utils import (
    cancel_async_task,
    fix_message_consistency,
    format_progress_time,
    get_instructions,
    handle_sonnet_limits,
    handle_tool_call,
    optionally_upload_heavy_logs,
    prune_messages,
)
from paperbench.solvers.upload import (
    start_periodic_light_log_upload,
    upload_heavy_logs,
)
from paperbench.solvers.utils import (
    check_for_existing_run,
    log_messages_to_file,
    sanity_check_docker,
)
from paperbench.paper_registry import paper_registry
from paperbench.solvers.cus_tools.aweai_mcp.utils import build_blocked_patterns_from_blacklist

# AI Scientist specific imports
from paperbench.solvers.aiscientist.prompts.templates import get_ai_scientist_system_message
from paperbench.solvers.aiscientist.tools.paper_reader_tool import ReadPaperTool
from paperbench.solvers.aiscientist.tools.prioritization_tool import PrioritizeTasksTool
from paperbench.solvers.aiscientist.tools.spawn_subagent_tool import SpawnSubagentTool
from paperbench.solvers.aiscientist.tools.implementation_tool import ImplementationTool
from paperbench.solvers.aiscientist.tools.experiment_tool import ExperimentTool
from paperbench.solvers.aiscientist.tools.clean_validation_tool import CleanReproduceValidationTool
from paperbench.solvers.cus_tools.aweai_mcp.google_search import WebSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.link_summary_op import LinkSummaryOpTool
from paperbench.solvers.aiscientist.summary_utils import SummaryConfig, summarize_messages

import asyncio

logger = structlog.stdlib.get_logger(component=__name__)


def build_safety_identifier(scope: str, paper_id: str, run_id: str) -> str:
    raw = f"paperbench:{scope}:{paper_id}:{run_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"paperbench-{scope}-{digest}"


@chz.chz
class AiScientistSolver(BasePBSolver):
    """
    AI Scientist Solver - An enhanced solver with subagent support.

    Key differences from BasicAgentSolver:
    1. Uses specialized subagents for paper reading
    2. Has a structured workflow with phases
    3. Provides additional tools for subagent coordination

    Configuration:
        - completer_config: LLM configuration (supports OpenAI/Azure/etc.)
        - max_steps: Maximum agent steps (None = unlimited)
        - time_limit: Time limit in seconds
        - use_subagents: Whether to enable specialized subagents (paper reading, implementation, etc.)
    """

    completer_config: BasicAgentTurnCompleterConfig = chz.field(
        default_factory=lambda: OpenAIResponsesTurnCompleterConfig(
            model="gpt-4.1-mini",
            tools=[WebSearchToolParam(type="web_search_preview")],
        )
    )
    max_steps: int | None = chz.field(default=None)
    time_limit: int | None = chz.field(default=5 * 60, doc="Time limit in seconds (5min default)")
    use_submit_tool: bool = chz.field(default=True)
    use_real_time_limit: bool = chz.field(
        default=True,
        doc="If True, don't count API retrying time towards time limit",
    )

    # AI Scientist specific options
    use_subagents: bool = chz.field(
        default=True,
        doc="Enable specialized subagents (paper reading, implementation, experiment, etc.)",
    )

    upload_interval_messages: int | None = chz.field(default=None)
    upload_interval_seconds: int | None = chz.field(default=1800)

    reminder_freq: int = chz.field(default=5, doc="Every how many steps to remind agent of state")

    summary_config: SummaryConfig = chz.field(
        default_factory=SummaryConfig,
        doc="Summary-based context reduction config (set enabled=False for prune-only)",
    )

    @override
    def shortname(self) -> str:
        return "aiscientist"

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        """
        Solver-specific setup, run after task setup and before agent starts.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )
        ctx_logger.info("Running AI Scientist pre-commands...", destinations=["run"])

        # Create agent output directory
        await computer.send_shell_command("mkdir -p /home/agent")

        # Dump container environment variables to run_dir/env.json for debugging
        try:
            env_result = await computer.send_shell_command(
                'python3 -c "import json, os; print(json.dumps(dict(os.environ), indent=2))"'
            )
            env_json_str = env_result.output.decode("utf-8", errors="replace").strip()
            env_data = json.loads(env_json_str)

            # Mask sensitive values (API keys, tokens)
            sensitive_keywords = ("KEY", "TOKEN", "SECRET", "PASSWORD")
            masked_env = {}
            for k, v in env_data.items():
                if any(kw in k.upper() for kw in sensitive_keywords) and v:
                    masked_env[k] = v[:8] + "..." + v[-4:] if len(v) > 12 else "***"
                else:
                    masked_env[k] = v

            env_path = bf.join(task.run_dir, "env.json")
            with bf.BlobFile(env_path, "w") as f:
                f.write(json.dumps(masked_env, indent=2, sort_keys=True))
            ctx_logger.info(f"Container env saved to {env_path}", destinations=["run"])
        except Exception as e:
            ctx_logger.warning(
                f"Failed to dump container environment: {e}", destinations=["run"]
            )

        ctx_logger.info("AI Scientist setup completed.", destinations=["run"])

    def _get_tools(self) -> list[Tool]:
        """Get the tools available to the main agent."""
        tools: list[Tool] = [
            BashToolWithTimeout(
                default_timeout=MAIN_AGENT_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.time_limit or MAIN_AGENT_BASH_MAX_TIMEOUT,
            ),
            PythonTool(),
            SearchFile(),
            ReadFileChunk(),
            WebSearchTool(),
            LinkSummaryOpTool(),
        ]

        # Add subagent tools if enabled
        if self.use_subagents:
            # 1. read_paper: Fixed workflow for comprehensive paper analysis
            read_paper_tool = ReadPaperTool()
            read_paper_tool.completer_config = self.completer_config
            tools.append(read_paper_tool)

            # 2. prioritize_tasks: Create prioritized TODO list from rubric + paper
            prioritize_tool = PrioritizeTasksTool()
            prioritize_tool.completer_config = self.completer_config
            tools.append(prioritize_tool)

            # 3. spawn_subagent: Claude Code Task-style generic subagent
            spawn_subagent_tool = SpawnSubagentTool()
            spawn_subagent_tool.completer_config = self.completer_config
            tools.append(spawn_subagent_tool)

            # 5. implement: Invoke Implementation Subagent for code writing
            impl_tool = ImplementationTool()
            impl_tool.completer_config = self.completer_config
            tools.append(impl_tool)

            # 6. run_experiment: Invoke Experiment Subagent for validation
            exp_tool = ExperimentTool()
            exp_tool.completer_config = self.completer_config
            tools.append(exp_tool)

            # 7. clean_reproduce_validation: Clean environment validation before submit
            clean_val_tool = CleanReproduceValidationTool()
            clean_val_tool.completer_config = self.completer_config
            tools.append(clean_val_tool)

        if self.use_submit_tool:
            tools.append(SubmitTool())

        return tools

    def _build_tool_constraints(self, paper_id: str) -> dict | None:
        """
        Build tool constraints from the paper's blacklist.
        """
        try:
            paper = paper_registry.get_paper(paper_id)
            if not paper.blacklist.exists():
                return None

            blacklist_content = paper.blacklist.read_text().splitlines()
            blacklist = [
                line.strip() for line in blacklist_content
                if line.strip() and not line.startswith("#")
            ]
            if blacklist == ["none"] or not blacklist:
                return None

            blocked_patterns = build_blocked_patterns_from_blacklist(blacklist)
            if not blocked_patterns:
                return None

            return {"blocked_search_patterns": blocked_patterns}
        except Exception as e:
            logger.warning(
                f"Failed to build tool constraints from blacklist: {e}",
                paper_id=paper_id,
            )
            return None

    async def _execute_agent(
        self,
        computer: ComputerInterface,
        task: PBTask,
        prompt: list[ChatCompletionMessageParam],
        start_time: int,
    ) -> int:
        """
        Executes the AI Scientist agent.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        tools = self._get_tools()

        # Configure subagent-related tools with constraints and run_dir for logging
        constraints = self._build_tool_constraints(task.paper_id)
        for tool in tools:
            if isinstance(tool, (ReadPaperTool, PrioritizeTasksTool, SpawnSubagentTool,
                                 ImplementationTool, ExperimentTool, CleanReproduceValidationTool)):
                tool.constraints = constraints
                tool.run_dir = task.run_dir  # Set run_dir for subagent logging
            # WebSearchTool and LinkSummaryOpTool use execute_with_constraints via handle_tool_call,
            # so they don't need tool.constraints — constraints are passed at call time.

        completer_config = self.completer_config.model_copy(deep=True)
        completer_config.basicagent_tools = tools
        if hasattr(completer_config, "safety_identifier"):
            completer_config.safety_identifier = build_safety_identifier(
                "solver", task.paper_id, task.run_id
            )
        completer = completer_config.build()

        messages = LoggableMessages(
            init=prompt, log_path=bf.join(task.run_dir, "agent.log"), logger=log_messages_to_file
        )

        ctx_logger.info(
            "Built tool constraints from blacklist",
            constraints=constraints,
            destinations=["run"],
        )

        num_steps = 0
        total_retry_time = 0.0
        last_time_uploaded = time.time()
        submit_attempts = 0  # Track submit attempts for pre-check warnings
        # Track implement/experiment call sequence for balance monitoring
        impl_exp_sequence: list[str] = []  # e.g., ["impl", "exp", "exp", "impl", "exp"]
        # Track whether clean_reproduce_validation was called before submit
        clean_validation_called = False
        # Content policy violation retry tracking (per-request)
        content_policy_retries = 0
        # Summary-based context reduction state
        last_summary: str | None = None
        convo_path = bf.join(task.run_dir, "conversation.jsonl")

        upload_task = None
        try:
            while not self._has_agent_finished(
                num_steps=num_steps, start_time=start_time, total_retry_time=total_retry_time
            ):
                num_steps += 1
                optional_upload_outcome = await optionally_upload_heavy_logs(
                    computer=computer,
                    task=task,
                    run_dir=task.run_dir,
                    num_steps=num_steps,
                    start_time=start_time,
                    last_time_uploaded=last_time_uploaded,
                    upload_interval_messages=self.upload_interval_messages,
                    upload_interval_seconds=self.upload_interval_seconds,
                )
                last_time_uploaded, upload_task = (
                    optional_upload_outcome.last_time_uploaded,
                    optional_upload_outcome.upload_task,
                )

                # Handle Sonnet message limits
                messages = handle_sonnet_limits(self.completer_config, messages)

                # Send reminder every self.reminder_freq steps
                if num_steps % self.reminder_freq == 0:
                    # Lightweight reproduce.sh existence check for reminder
                    reproduce_sh_exists = False
                    try:
                        check = await computer.send_shell_command(
                            "test -f /home/submission/reproduce.sh && echo EXISTS || echo MISSING"
                        )
                        reproduce_sh_exists = "EXISTS" in check.output.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    messages.append(self._construct_periodic_reminder(
                        start_time, total_retry_time, impl_exp_sequence, reproduce_sh_exists
                    ))

                # Generate response
                try:
                    n_input_messages = len(messages.data)
                    res = await make_completer_request(completer=completer, messages=messages.data)
                    (response_messages, tool_calls, response_retry_time) = (
                        res.response_messages,
                        res.tool_calls,
                        res.time_spent_retrying,
                    )
                    total_retry_time += response_retry_time

                    # Token usage logging
                    usage_obj = res.usage
                    usage_dict: dict[str, object] | None = None
                    in_tokens = out_tokens = total_tokens = None
                    if usage_obj is not None:
                        try:
                            usage_dict = usage_obj.model_dump(exclude_none=True)
                        except Exception:
                            try:
                                usage_dict = dict(usage_obj)
                            except Exception:
                                usage_dict = {"repr": repr(usage_obj)}

                        usage_dict_for_norm: dict[str, object] = usage_dict or {}
                        in_tokens = usage_dict_for_norm.get("prompt_tokens")
                        if in_tokens is None:
                            in_tokens = usage_dict_for_norm.get("input_tokens")
                        out_tokens = usage_dict_for_norm.get("completion_tokens")
                        if out_tokens is None:
                            out_tokens = usage_dict_for_norm.get("output_tokens")
                        total_tokens = usage_dict_for_norm.get("total_tokens")
                        if total_tokens is None and isinstance(in_tokens, int) and isinstance(
                            out_tokens, int
                        ):
                            total_tokens = in_tokens + out_tokens

                    # Conversation logging
                    try:
                        log_model_response_event(
                            convo_path=convo_path,
                            run_id=task.run_id,
                            step=num_steps,
                            n_input_messages=n_input_messages,
                            response_messages=response_messages,
                            tool_calls=tool_calls,
                            usage=usage_dict,
                            normalized={
                                "in_tokens": in_tokens,
                                "out_tokens": out_tokens,
                                "total_tokens": total_tokens,
                            },
                        )
                    except Exception as e:
                        ctx_logger.warning(
                            "Failed to write conversation.jsonl (model_response)",
                            convo_path=convo_path,
                            error=str(e),
                            destinations=["run"],
                        )

                except (LengthFinishReasonError, IndexError) as e:
                    prune_individual = "PRUNE_INDIVIDUAL_MESSAGES" in str(e)
                    if self.summary_config.enabled:
                        ctx_logger.info(
                            "Context length exceeded. Attempting summary...",
                            destinations=["run"],
                        )
                        messages, last_summary, succeeded = await summarize_messages(
                            completer_config=self.completer_config,
                            messages=messages,
                            last_summary=last_summary,
                            config=self.summary_config,
                            prune_individual=prune_individual,
                            log_dir=task.run_dir,
                            step=num_steps,
                        )
                        ctx_logger.info(
                            f"Context reduction: {'summary succeeded' if succeeded else 'fell back to prune'}",
                            destinations=["run"],
                        )
                    else:
                        ctx_logger.info(
                            "Context length exceeded. Pruning messages...",
                            destinations=["run"],
                        )
                        messages = prune_messages(
                            messages,
                            prune_individual=prune_individual,
                        )
                    continue
                except BadRequestError as e:
                    err_str = str(e).lower()
                    if "context length" in err_str or "maximum context" in err_str or "exceeds max length" in err_str or "input token count exceeds" in err_str:
                        if self.summary_config.enabled:
                            ctx_logger.info(
                                "Input exceeds context length. Attempting summary...",
                                destinations=["run"],
                            )
                            messages, last_summary, succeeded = await summarize_messages(
                                completer_config=self.completer_config,
                                messages=messages,
                                last_summary=last_summary,
                                config=self.summary_config,
                                prune_individual=True,
                                log_dir=task.run_dir,
                                step=num_steps,
                            )
                        else:
                            ctx_logger.info(
                                "Input exceeds context length. Pruning messages...",
                                destinations=["run"],
                            )
                            messages = prune_messages(messages, prune_individual=True)
                        continue

                    error_code = str(getattr(e, 'code', '') or '')
                    error_msg = str(getattr(e, 'message', '') or '')

                    if '-4321' in error_code or 'invalid_prompt' in error_msg:
                        # Content policy violation (-4321 / invalid_prompt).
                        # Graduated retry: 2 direct retries → 1 prune retry → stop.
                        # Total max 4 violations, safe for Azure (lockout at 5).
                        content_policy_retries += 1
                        req_logid = e.request.headers.get("X-TT-LOGID", "unknown")
                        x_request_id = getattr(e, "request_id", None) or "unknown"

                        ctx_logger.warning(
                            f"Content policy violation [paper_id={task.paper_id}] "
                            f"[tt_logid={req_logid}] "
                            f"[x-request-id={x_request_id}] "
                            f"(attempt {content_policy_retries}/4)",
                            paper_id=task.paper_id,
                            error_code=error_code,
                            tt_logid=req_logid,
                            x_request_id=x_request_id,
                            status_code=e.status_code,
                            response_body=e.body,
                            response_headers=dict(e.response.headers) if e.response else None,
                            destinations=["run"],
                        )

                        if content_policy_retries <= 2:
                            # Direct retry (handles transient/non-deterministic violations)
                            ctx_logger.info(
                                f"Direct retry {content_policy_retries}/2 "
                                f"[paper_id={task.paper_id}]...",
                                destinations=["run"],
                            )
                            continue
                        elif content_policy_retries == 3:
                            # Prune then retry (handles content-triggered violations)
                            ctx_logger.info(
                                f"Pruning messages then retrying "
                                f"[paper_id={task.paper_id}]...",
                                destinations=["run"],
                            )
                            messages = prune_messages(
                                messages,
                                prune_individual=True,
                            )
                            continue
                        else:
                            # Exhausted all retries, stop task
                            ctx_logger.error(
                                f"Content policy violation retry exhausted "
                                f"[paper_id={task.paper_id}], stopping task.",
                                destinations=["run"],
                            )
                            raise
                    else:
                        # Other BadRequestErrors (-4003 "No tool output found",
                        # malformed content, etc.) — try fixing message consistency
                        # first (zero information loss).  If the same error recurs
                        # after the fix, the outer retry/exception handling will
                        # surface it rather than looping forever.
                        ctx_logger.warning(
                            f"BadRequestError in main loop, fixing message consistency: "
                            f"{e.message}",
                            error_code=error_code,
                            destinations=["run"],
                        )
                        messages = fix_message_consistency(messages)
                        continue

                # Reset content policy retry counter on successful request
                content_policy_retries = 0

                messages += response_messages

                if tool_calls:
                    for tool_call in tool_calls:
                        # Pre-check for submit tool (runs every attempt)
                        if tool_call.name == "submit":
                            submit_attempts += 1
                            hard_errors = []
                            warnings = []

                            # Check 1: Early submission warning (only on first attempt)
                            if submit_attempts == 1:
                                if self.use_real_time_limit and self.time_limit:
                                    elapsed = time.time() - start_time - total_retry_time
                                elif self.time_limit:
                                    elapsed = time.time() - start_time
                                else:
                                    elapsed = 0
                                time_ratio = elapsed / self.time_limit if self.time_limit else 1.0
                                remaining_hours = (self.time_limit - elapsed) / 3600.0 if self.time_limit else 0.0

                                if time_ratio < 0.5:
                                    warnings.append(
                                        f"EARLY SUBMISSION: Only {time_ratio*100:.0f}% of time used "
                                        f"({remaining_hours:.1f} hours remaining). "
                                        "You have significant time remaining to improve your score. "
                                        "The rubric typically has 100+ leaf dimensions — each additional "
                                        "dataset, baseline, or model you cover earns more points. "
                                        "Review /home/agent/prioritized_tasks.md for other tasks you "
                                        "can still tackle."
                                    )

                            # Check 2a: reproduce.sh existence (HARD BLOCK — non-bypassable)
                            try:
                                check_result = await computer.send_shell_command(
                                    "test -f /home/submission/reproduce.sh && echo 'EXISTS' || echo 'MISSING'"
                                )
                                reproduce_exists = "EXISTS" in check_result.output.decode("utf-8", errors="replace")
                            except Exception:
                                reproduce_exists = False

                            if not reproduce_exists:
                                hard_errors.append(
                                    "reproduce.sh DOES NOT EXIST: /home/submission/reproduce.sh is missing! "
                                    "Without it, ALL Code Execution and Result Analysis rubric items will score 0 "
                                    "You MUST create reproduce.sh before submitting. "
                                    "Use implement(task='Create reproduce.sh ...')."
                                )

                            # Check 2b: reproduce.sh committed to git (HARD BLOCK — non-bypassable)
                            if reproduce_exists:
                                try:
                                    git_check = await computer.send_shell_command(
                                        "cd /home/submission && git ls-files reproduce.sh | grep -q reproduce.sh && echo 'TRACKED' || echo 'UNTRACKED'"
                                    )
                                    git_tracked = "TRACKED" in git_check.output.decode("utf-8", errors="replace")
                                except Exception:
                                    git_tracked = False

                                if not git_tracked:
                                    hard_errors.append(
                                        "reproduce.sh NOT COMMITTED TO GIT: The file exists but is not tracked by git. "
                                        "During grading, `git clean -fd` removes all untracked files, so reproduce.sh will be LOST. "
                                        "Use implement() or bash to run: cd /home/submission && git add reproduce.sh && git commit -m 'Add reproduce.sh'"
                                    )

                            # If hard errors exist, block submission (don't count as submit attempt)
                            if hard_errors:
                                submit_attempts -= 1  # Don't count this as a real attempt
                                error_msg = (
                                    "SUBMISSION BLOCKED:\n\n"
                                    + "\n\n".join(f"BLOCKED -- {e}" for e in hard_errors)
                                    + "\n\nFix these issues first. submit() will remain blocked until they are resolved."
                                )
                                handled_block: ChatCompletionToolMessageParam = {
                                    "role": "tool",
                                    "content": error_msg,
                                    "tool_call_id": tool_call.call_id,
                                }
                                messages.append(handled_block)
                                ctx_logger.info(
                                    f"Submit HARD BLOCKED (attempt {submit_attempts})",
                                    hard_errors=hard_errors,
                                    destinations=["run"],
                                )
                                continue  # Block this submit, force agent to fix

                            # Check 3: reproduce.sh quick syntax/dependency sanity check (soft warning)
                            if reproduce_exists:
                                try:
                                    dep_check = await computer.send_shell_command(
                                        'cd /home/submission && '
                                        'bash -n reproduce.sh 2>&1 && echo "SYNTAX_OK" || echo "SYNTAX_ERR"; '
                                        'if [ -f requirements.txt ]; then echo "REQS_FOUND"; '
                                        'else echo "REQS_MISSING"; fi'
                                    )
                                    dep_output = dep_check.output.decode("utf-8", errors="replace")
                                    if "SYNTAX_ERR" in dep_output:
                                        warnings.append(
                                            "reproduce.sh has SYNTAX ERRORS. "
                                            "Use implement() to fix it before submitting."
                                        )
                                    if "REQS_MISSING" in dep_output:
                                        warnings.append(
                                            "requirements.txt is MISSING. reproduce.sh likely calls "
                                            "'pip install -r requirements.txt' — this will fail during grading. "
                                            "Use implement() to create requirements.txt."
                                        )
                                except Exception:
                                    pass

                            # Check 4: Clean validation not performed (soft warning, first attempt only)
                            if not clean_validation_called and submit_attempts == 1:
                                warnings.append(
                                    "CLEAN ENVIRONMENT VALIDATION NOT PERFORMED: You have not run "
                                    "`clean_reproduce_validation()` before submitting. This tool simulates "
                                    "the grading environment by clearing venv, dataset caches, and model caches, "
                                    "then running reproduce.sh from scratch. Without it, bugs masked by cached "
                                    "state (e.g., dataset download failures, missing dependencies) will cause "
                                    "reproduce.sh to FAIL during grading, scoring ZERO on all execution items. "
                                    "STRONGLY RECOMMENDED: Call `clean_reproduce_validation()` before submitting."
                                )

                            if warnings:
                                warning_msg = (
                                    "SUBMIT PRE-CHECK WARNINGS:\n\n"
                                    + "\n\n".join(f"- {w}" for w in warnings)
                                    + "\n\nIf you still want to submit, call submit() again."
                                )
                                handled_warning: ChatCompletionToolMessageParam = {
                                    "role": "tool",
                                    "content": warning_msg,
                                    "tool_call_id": tool_call.call_id,
                                }
                                messages.append(handled_warning)
                                ctx_logger.info(
                                    f"Submit pre-check warnings issued (attempt {submit_attempts})",
                                    warnings=warnings,
                                    destinations=["run"],
                                )
                                continue  # Skip this submit, let agent react to warnings

                        handled = await handle_tool_call(
                            tool_call, tools, task, computer, constraints=constraints
                        )
                        if handled is None:  # Submit tool called
                            return num_steps
                        messages.append(handled)

                        # Track implement/experiment calls for balance monitoring
                        if tool_call.name == "implement":
                            impl_exp_sequence.append("impl")
                        elif tool_call.name == "run_experiment":
                            impl_exp_sequence.append("exp")
                        elif tool_call.name == "clean_reproduce_validation":
                            clean_validation_called = True

                        # Tool result logging
                        try:
                            log_tool_result_event(
                                convo_path=convo_path,
                                run_id=task.run_id,
                                step=num_steps,
                                tool_call=tool_call,
                                tool_message=dict(handled),
                            )
                        except Exception as e:
                            ctx_logger.warning(
                                "Failed to write conversation.jsonl (tool_result)",
                                convo_path=convo_path,
                                error=str(e),
                                destinations=["run"],
                            )
                else:
                    messages.append({"role": "user", "content": DEFAULT_CONTINUE_MESSAGE})

        except asyncio.TimeoutError as e:
            if upload_task:
                await asyncio.shield(upload_task)
            raise e

        return num_steps

    def _has_agent_finished(
        self, num_steps: int, start_time: float, total_retry_time: float
    ) -> bool:
        """Check if the agent should stop."""
        over_step_limit = self.max_steps is not None and num_steps > self.max_steps

        if self.use_real_time_limit and self.time_limit:
            over_time_limit = time.time() - start_time - total_retry_time > self.time_limit
        elif self.time_limit:
            over_time_limit = time.time() - start_time > self.time_limit
        else:
            over_time_limit = False

        return over_step_limit or over_time_limit

    def _construct_periodic_reminder(
        self, start_time: float, total_retry_time: float,
        impl_exp_sequence: list[str] | None = None,
        reproduce_sh_exists: bool = False,
    ) -> ChatCompletionUserMessageParam:
        """Constructs a periodic reminder message with AI Scientist specific tips."""
        if self.use_real_time_limit and self.time_limit:
            elapsed_time = time.time() - start_time - total_retry_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(self.time_limit)}."
        elif self.time_limit:
            elapsed_time = time.time() - start_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(self.time_limit)}."
        else:
            elapsed_time = time.time() - start_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed"

        # === Implement/Experiment balance warning ===
        if impl_exp_sequence:
            total_impl = impl_exp_sequence.count("impl")
            total_exp = impl_exp_sequence.count("exp")
            # The gap between experiment and implement counts signals imbalance.
            # Healthy runs have roughly equal counts (diff 0-2).
            # Large gaps mean the agent is re-running experiments without fixing code.
            exp_impl_gap = total_exp - total_impl

            if exp_impl_gap >= 4:
                periodic_msg += (
                    f"\n\n🚨 IMPLEMENT/EXPERIMENT IMBALANCE: You have called run_experiment {total_exp} times "
                    f"but implement only {total_impl} times (gap: {exp_impl_gap}).\n"
                    "This means you are running experiments repeatedly without fixing code — "
                    "this is WASTED TIME since experiments are deterministic.\n"
                    "You MUST either:\n"
                    "1. Call `implement(mode='fix', context='<diagnosis from last experiment>')` to fix the issues, OR\n"
                    "2. Move on to the next priority task if this one is stuck after 2-3 fix attempts.\n"
                    "Do NOT call run_experiment() again until you have made code changes via implement()."
                )
            elif exp_impl_gap >= 2:
                periodic_msg += (
                    f"\n\n⚠️ Note: experiment calls ({total_exp}) are outpacing implement calls ({total_impl}). "
                    "If experiments are failing, call implement(mode='fix') to fix the code before running more experiments."
                )

            # Symmetric check: too many implements without validation
            impl_exp_gap = total_impl - total_exp
            if impl_exp_gap >= 3:
                periodic_msg += (
                    f"\n\n⚠️ VALIDATION GAP: You have called implement {total_impl} times "
                    f"but run_experiment only {total_exp} times (gap: {impl_exp_gap}).\n"
                    "You are writing code without validating it — bugs accumulate and become harder to fix.\n"
                    "Call `run_experiment(task='Validate reproduce.sh end-to-end', mode='validate')` "
                    "to check that your code actually works before implementing more."
                )

        # Calculate time ratio for escalating reminders
        time_ratio = elapsed_time / self.time_limit if self.time_limit else 0.0

        if time_ratio >= 0.85:
            # Final phase: inform about remaining time, don't force early submission
            if reproduce_sh_exists:
                periodic_msg += (
                    "\n\n⚠️ Less than 15% time remaining.\n"
                    "- Make sure all changes are git committed\n"
                    "- Focus on finishing current work rather than starting new tasks\n"
                    "- Ensure reproduce.sh is up to date with your latest code"
                )
            else:
                periodic_msg += (
                    "\n\n🚨 Less than 15% time remaining AND reproduce.sh DOES NOT EXIST!\n"
                    "Without reproduce.sh, ALL execution scores are ZERO.\n"
                    "Creating reproduce.sh should be your top priority right now."
                )
        elif time_ratio >= 0.70:
            # Late phase: advise wrapping up, but don't force submission
            if reproduce_sh_exists:
                periodic_msg += (
                    "\n\n⚠️ 70%+ time used. Consider wrapping up current work.\n"
                    "- If reproduce.sh hasn't been validated yet, run a validation now\n"
                    "- Git commit all changes regularly\n"
                    "- Avoid starting large new tasks — focus on finishing in-progress work"
                )
            else:
                periodic_msg += (
                    "\n\n🚨 70%+ time used AND reproduce.sh DOES NOT EXIST!\n"
                    "Without reproduce.sh, ALL Code Execution and Result Analysis scores will be 0.\n"
                    "Creating reproduce.sh should be your highest priority now."
                )
        elif time_ratio >= 0.50:
            # Mid-game: remind to maximize coverage before submitting
            periodic_msg += (
                "\n\n50%+ time used. Keep implementing — do NOT submit yet unless all rubric items are covered."
                "\n- Check /home/agent/prioritized_tasks.md for remaining tasks"
                "\n- Each additional dataset/baseline/model you cover earns points"
                "\n- Git commit regularly!"
            )
            if not reproduce_sh_exists:
                periodic_msg += (
                    "\n- 🚨 reproduce.sh DOES NOT EXIST yet! Create it NOW via implement(). "
                    "It's your #1 deliverable — without it, all execution scores are zero."
                )
        else:
            # Normal reminders (< 50% time used)
            periodic_msg += "\n\nReminders:"
            periodic_msg += "\n- Focus on P0-Critical tasks first! Check /home/agent/prioritized_tasks.md"
            periodic_msg += "\n- Don't forget to git commit regularly!"
            periodic_msg += "\n- Reference /home/agent/paper_analysis/ for paper details (summary.md, algorithm.md, experiments.md, etc.)"
            periodic_msg += "\n- DO NOT submit early. The rubric has many dimensions — maximize coverage."
            if not reproduce_sh_exists:
                periodic_msg += "\n- IMPORTANT: reproduce.sh doesn't exist yet — create it NOW via implement(). It's your #1 deliverable."

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": periodic_msg,
        }
        return message

    async def _execute_agent_and_periodically_upload_logs(
        self,
        computer: ComputerInterface,
        task: PBTask,
        prompt: list[ChatCompletionMessageParam],
    ) -> None:
        """
        Starts agent execution and periodically uploads logs.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
            destinations=["run"],
        )

        num_steps = -1
        start_time = int(time.time())
        agent_task: asyncio.Task[int] | None = None
        light_periodic_upload_task: asyncio.Task[None] | None = None

        try:
            async with asyncio.timeout(self.time_limit):
                light_periodic_upload_task = await start_periodic_light_log_upload(
                    agent_start_time=start_time,
                    run_dir=task.run_dir,
                    run_group_id=task.run_group_id,
                    runs_dir=task.runs_dir,
                    run_id=task.run_id,
                )

                agent_task = asyncio.create_task(
                    self._execute_agent(
                        computer=computer,
                        task=task,
                        prompt=prompt,
                        start_time=start_time,
                    )
                )

                while not agent_task.done():
                    ctx_logger.info("Waiting for AI Scientist agent to finish...")
                    if light_periodic_upload_task.done():
                        exc = light_periodic_upload_task.exception()
                        if exc:
                            raise exc
                    await asyncio.sleep(60)
                num_steps = await agent_task

        except asyncio.TimeoutError as e:
            ctx_logger.info(
                f"AI Scientist run timed out after {time.time() - start_time} seconds (timeout: {self.time_limit}): {e}",
            )
        except asyncio.CancelledError as e:
            ctx_logger.info(
                f"AI Scientist run cancelled after {time.time() - start_time} seconds: {e}",
            )
        finally:
            await cancel_async_task(async_task=agent_task, pb_task=task)
            await cancel_async_task(async_task=light_periodic_upload_task, pb_task=task)

        # Final upload
        await upload_heavy_logs(
            computer=computer,
            agent_start_time=int(start_time),
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=task.run_dir,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
            num_messages=num_steps,
        )

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """
        Prepares and runs the AI Scientist agent.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        # Don't run if we already have logs
        agent_output = await check_for_existing_run(task)
        if agent_output:
            return agent_output

        await sanity_check_docker(computer)

        start_time = time.time()
        instructions = await get_instructions(computer, task, iterative_agent=False, time_limit=self.time_limit)

        # Use AI Scientist system message
        system_message = get_ai_scientist_system_message(code_only=task.judge.code_only)

        await self._execute_agent_and_periodically_upload_logs(
            computer=computer,
            task=task,
            prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": instructions},
            ],
        )

        ctx_logger.info(
            f"AI Scientist `{self.shortname()}` finished running for `{task.question_id}.{task.attempt_id}`!",
            destinations=["group", "run"],
            _print=True,
        )

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=time.time(),
            error_msg=None,
            runtime_in_seconds=time.time() - start_time,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
        )
