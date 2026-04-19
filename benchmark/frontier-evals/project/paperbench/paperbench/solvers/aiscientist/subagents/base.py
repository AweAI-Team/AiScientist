"""
Subagent Base Classes

This module defines the core abstraction for subagents in the AI Scientist solver.
Subagents are specialized agents that focus on specific tasks (e.g., reading papers,
writing code, debugging). They share the same completer infrastructure as the main
agent but have their own prompts, tools, and execution context.

Key Design Decisions:
1. Subagents reuse the existing completer architecture (supports OpenAI/Azure/etc.)
2. Subagents share the same ComputerInterface (sandbox) with the main agent
3. Each subagent has its own system prompt and tool set
4. Subagent outputs are structured and returned to the main agent
5. Subagent messages are logged to files for debugging and analysis
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import blobfile as bf
import structlog
from openai import NOT_GIVEN, BadRequestError, LengthFinishReasonError
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel
from pydantic import Field as PydanticField

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.log_utils import (
    log_model_response_event,
    log_tool_result_event,
)
from paperbench.solvers.basicagent.logging import LoggableMessages
from paperbench.solvers.basicagent.tools.base import Tool, ToolCall
from paperbench.solvers.basicagent.utils import (
    fix_message_consistency,
    prune_messages,
)
from paperbench.solvers.aiscientist.summary_utils import SummaryConfig, summarize_messages
from paperbench.solvers.utils import log_messages_to_file

logger = structlog.stdlib.get_logger(component=__name__)


# =============================================================================
# Completion Signal
# =============================================================================

class SubagentCompleteSignal(Exception):
    """
    Signal raised by SubagentCompleteTool to indicate task completion.

    This provides a clean way for the subagent to signal completion without
    hardcoding tool name checks in the execution loop.
    """
    def __init__(self, output: str, artifacts: dict[str, Any] | None = None):
        self.output = output
        self.artifacts = artifacts or {}
        super().__init__(f"Subagent completed with output length: {len(output)}")


# =============================================================================
# Data Classes
# =============================================================================

class SubagentStatus(str, Enum):
    """Status of a subagent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SubagentOutput:
    """
    Structured output from a subagent execution.

    This is designed to be easily consumable by the main agent or other subagents.
    The `content` field contains the main output, while `artifacts` can contain
    structured data like extracted configurations, file paths, etc.
    """
    subagent_name: str
    status: SubagentStatus
    content: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    num_steps: int = 0
    runtime_seconds: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    log_path: str | None = None  # Path to the conversation log file

    def to_tool_result(self) -> str:
        """Format output as a tool result string for the main agent."""
        if self.status == SubagentStatus.COMPLETED:
            result = f"[Subagent: {self.subagent_name}] Completed successfully.\n\n"
            result += self.content
            if self.artifacts:
                result += f"\n\n[Artifacts: {list(self.artifacts.keys())}]"
            if self.log_path:
                result += f"\n\n[Log: {self.log_path}]"
            return result
        elif self.status == SubagentStatus.FAILED:
            return f"[Subagent: {self.subagent_name}] Failed: {self.error_message}\n\nPartial output:\n{self.content}"
        elif self.status == SubagentStatus.TIMEOUT:
            return f"[Subagent: {self.subagent_name}] Timed out after {self.runtime_seconds:.1f}s.\n\nPartial output:\n{self.content}"
        else:
            return f"[Subagent: {self.subagent_name}] Status: {self.status}"


class SubagentConfig(BaseModel):
    """
    Configuration for a subagent.

    This allows customization of subagent behavior without modifying the code.
    """
    max_steps: int = PydanticField(default=50, description="Maximum steps before stopping")
    time_limit: int = PydanticField(default=300, description="Time limit in seconds (5min default)")
    reminder_freq: int = PydanticField(default=10, description="Steps between state reminders")
    log_dir: str | None = PydanticField(default=None, description="Directory for conversation logs (None = use run_dir)")
    output_dir: str = PydanticField(default="/home/agent/subagent_outputs", description="Directory for output files")
    summary_config: SummaryConfig | None = PydanticField(
        default=None,
        description="Summary config (None = prune-only for backward compat)",
    )


# =============================================================================
# Subagent Complete Tool
# =============================================================================

class SubagentCompleteTool(Tool):
    """
    Tool for subagents to signal completion and submit their findings.

    When executed, this tool raises SubagentCompleteSignal which is caught
    by the subagent execution loop to cleanly handle completion.
    """

    def name(self) -> str:
        return "subagent_complete"

    async def execute(
        self,
        computer: ComputerInterface,
        output: str,
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        """
        Signal completion by raising SubagentCompleteSignal.

        This method never returns normally - it always raises the signal.
        """
        raise SubagentCompleteSignal(output=output, artifacts=artifacts)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Submit your findings and complete the subagent task. Use this when you have finished analyzing the assigned content.",
            parameters={
                "type": "object",
                "properties": {
                    "output": {
                        "type": "string",
                        "description": "Your complete findings in markdown format.",
                    },
                    "artifacts": {
                        "type": "object",
                        "description": "Optional structured data extracted from the paper (e.g., hyperparameters, section boundaries).",
                        "additionalProperties": True,
                    },
                },
                "required": ["output"],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# Subagent Base Class
# =============================================================================

class Subagent(ABC):
    """
    Abstract base class for all subagents.

    Subagents are specialized agents that perform specific tasks. They:
    1. Have their own system prompt and task description
    2. Use a subset of tools appropriate for their task
    3. Execute within a shared sandbox environment
    4. Return structured output to the calling agent
    5. Log their conversations to files for debugging

    To create a new subagent:
    1. Inherit from this class
    2. Implement the abstract methods: name(), system_prompt(), get_tools()
    3. Optionally override _post_process_output() for custom output formatting
    """

    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,
        config: SubagentConfig | None = None,
        run_dir: str | None = None,
    ):
        """
        Initialize a subagent.

        Args:
            completer_config: Configuration for the LLM completer (supports OpenAI/Azure/etc.)
            config: Optional subagent-specific configuration
            run_dir: Directory for logs (will use config.log_dir or create temp if None)
        """
        self.completer_config = completer_config
        self.config = config or SubagentConfig()
        self.run_dir = run_dir
        self._messages: LoggableMessages | None = None
        self._output_buffer: list[str] = []
        self._total_tokens = {"input": 0, "output": 0}
        self._last_summary: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this subagent."""
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this subagent."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return the tools available to this subagent."""
        ...

    @staticmethod
    def _truncate_buffer_output(text: str, max_chars: int = 20000) -> str:
        """Truncate raw output_buffer text for failed/timeout results.

        Keeps the first 30% and last 70% of the budget so the main agent
        sees initial context *and* the most recent state before failure.
        """
        # Marker overhead is ~60 chars; skip truncation unless it actually saves space
        if len(text) <= max_chars + 100:
            return text
        head_budget = int(max_chars * 0.3)  # 6k
        tail_budget = max_chars - head_budget  # 14k
        omitted = len(text) - head_budget - tail_budget
        return (
            text[:head_budget]
            + f"\n\n... [omitted {omitted} chars of intermediate output] ...\n\n"
            + text[-tail_budget:]
        )

    def _post_process_output(self, raw_output: str, artifacts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Post-process the subagent output.

        Override this method to implement custom output formatting or artifact extraction.

        Args:
            raw_output: The raw text output from the subagent
            artifacts: Artifacts collected during execution

        Returns:
            Tuple of (processed_output, updated_artifacts)
        """
        return raw_output, artifacts

    def _build_initial_messages(self, task_description: str) -> list[ChatCompletionMessageParam]:
        """Build the initial message list for the subagent."""
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task_description},
        ]

    def _construct_reminder(self, elapsed_time: float) -> ChatCompletionMessageParam:
        """Construct a periodic reminder message."""
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        return {
            "role": "user",
            "content": f"[Subagent Reminder] Time elapsed: {time_str} / {self.config.time_limit}s. "
                       f"Focus on your task and output your findings clearly.",
        }

    def _get_log_path(self, run_id: str) -> str:
        """Get the path for the conversation log file."""
        if self.config.log_dir:
            log_dir = self.config.log_dir
        elif self.run_dir:
            log_dir = bf.join(self.run_dir, "subagent_logs")
        else:
            log_dir = "/tmp/subagent_logs"

        # Use os.makedirs for local paths (bf.makedirs may have issues with local paths)
        os.makedirs(log_dir, exist_ok=True)

        return bf.join(log_dir, f"{self.name}_{run_id}.log")

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        tools: list[Tool],
        computer: ComputerInterface,
        constraints: dict | None = None,
    ) -> ChatCompletionMessageParam | None:
        """
        Execute a single tool call.

        This method handles the SubagentCompleteSignal specially by re-raising it.
        Other tool results are returned as message dicts.

        Args:
            tool_call: The tool call to execute
            tools: Available tools
            computer: ComputerInterface for execution
            constraints: Optional constraints

        Returns:
            Tool result message, or None if tool not found

        Raises:
            SubagentCompleteSignal: If the subagent_complete tool is called
        """
        # Find the tool
        tool = next((t for t in tools if t.name() == tool_call.name), None)
        if tool is None:
            available_tools = [t.name() for t in tools]
            return {
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": f"Error: Tool '{tool_call.name}' not found. Available tools: {available_tools}",
            }

        try:
            # Execute the tool - this may raise SubagentCompleteSignal
            if tool.supports_constraints() and constraints:
                result = await tool.execute_with_constraints(
                    computer, constraints=constraints, **tool_call.arguments
                )
            else:
                result = await tool.execute(computer, **tool_call.arguments)

            return {
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": result if result else "Tool executed successfully.",
            }
        except SubagentCompleteSignal:
            # Re-raise completion signal to be handled by the run loop
            raise
        except Exception as e:
            logger.warning(f"Tool {tool_call.name} failed: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": f"Error executing tool: {str(e)}",
            }

    async def run(
        self,
        computer: ComputerInterface,
        task_description: str,
        constraints: dict | None = None,
        context: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> SubagentOutput:
        """
        Execute the subagent with the given task.

        Args:
            computer: ComputerInterface for executing commands
            task_description: Description of the task to perform
            constraints: Optional constraints (e.g., blacklist patterns)
            context: Optional context from previous subagents or main agent
            run_id: Optional run ID for logging (defaults to timestamp)

        Returns:
            SubagentOutput with the execution results
        """
        # end debug
        ctx_logger = logger.bind(subagent=self.name)
        ctx_logger.info(f"Starting subagent: {self.name}")

        start_time = time.time()
        artifacts: dict[str, Any] = {}
        run_id = run_id or f"{int(start_time)}"

        # Add context to task description if provided
        if context:
            context_str = "\n\n## Context from previous analysis:\n"
            for key, value in context.items():
                if isinstance(value, str):
                    context_str += f"### {key}\n{value}\n\n"
            task_description = task_description + context_str

        # Setup logging
        log_path = self._get_log_path(run_id)
        convo_path = log_path.rsplit(".log", 1)[0] + "_conversation.jsonl"
        ctx_logger.info(f"Subagent {self.name} logging to: {log_path}")

        # Initialize messages with logging
        initial_messages = self._build_initial_messages(task_description)
        self._messages = LoggableMessages(
            init=initial_messages,
            log_path=log_path,
            logger=log_messages_to_file,
        )
        self._output_buffer = []
        self._total_tokens = {"input": 0, "output": 0}

        # Setup tools and completer
        tools = self.get_tools()
        # Clone the completer config to avoid modifying the shared one
        completer_config = self.completer_config.model_copy()
        completer_config.basicagent_tools = tools
        # Clear existing tools to ensure only subagent's tools are used
        # (parent's build() may have already set self.tools, and build() appends rather than replaces)
        completer_config.tools = NOT_GIVEN
        completer = completer_config.build()

        num_steps = 0
        consecutive_empty_responses = 0
        MAX_CONSECUTIVE_EMPTY_RESPONSES = 10
        # Content policy violation retry tracking (per-request)
        content_policy_retries = 0

        try:
            while num_steps < self.config.max_steps:
                elapsed = time.time() - start_time
                if elapsed > self.config.time_limit:
                    ctx_logger.info(f"Subagent {self.name} reached time limit")
                    output, artifacts = self._post_process_output(
                        self._truncate_buffer_output("\n".join(self._output_buffer)), artifacts
                    )
                    return SubagentOutput(
                        subagent_name=self.name,
                        status=SubagentStatus.TIMEOUT,
                        content=output,
                        artifacts=artifacts,
                        num_steps=num_steps,
                        runtime_seconds=elapsed,
                        token_usage=self._total_tokens,
                        log_path=log_path,
                    )

                num_steps += 1

                # Send reminder periodically
                if num_steps % self.config.reminder_freq == 0:
                    self._messages.append(self._construct_reminder(elapsed))

                # Get LLM response
                try:
                    n_input_messages = len(self._messages.data)
                    res = await make_completer_request(
                        completer=completer, messages=self._messages.data
                    )
                    response_messages = res.response_messages
                    tool_calls = res.tool_calls

                    # Track token usage
                    usage_dict: dict[str, object] | None = None
                    in_tokens = out_tokens = total_tokens = None
                    if res.usage:
                        try:
                            usage_dict = res.usage.model_dump(exclude_none=True)
                            _in = usage_dict.get("prompt_tokens")
                            if _in is None:
                                _in = usage_dict.get("input_tokens", 0)
                            _out = usage_dict.get("completion_tokens")
                            if _out is None:
                                _out = usage_dict.get("output_tokens", 0)
                            self._total_tokens["input"] += _in or 0
                            self._total_tokens["output"] += _out or 0
                            in_tokens = _in
                            out_tokens = _out
                            total_tokens = usage_dict.get("total_tokens")
                            if total_tokens is None and isinstance(in_tokens, int) and isinstance(out_tokens, int):
                                total_tokens = in_tokens + out_tokens
                        except Exception:
                            pass

                    # Conversation logging (structured JSONL)
                    try:
                        log_model_response_event(
                            convo_path=convo_path,
                            run_id=run_id,
                            step=num_steps,
                            n_input_messages=n_input_messages,
                            response_messages=response_messages,
                            tool_calls=tool_calls or [],
                            usage=usage_dict,
                            normalized={
                                "in_tokens": in_tokens,
                                "out_tokens": out_tokens,
                                "total_tokens": total_tokens,
                            },
                        )
                    except Exception:
                        pass

                except LengthFinishReasonError:
                    if self.config.summary_config and self.config.summary_config.enabled:
                        ctx_logger.info("Context length exceeded, attempting summary")
                        log_dir = self.config.log_dir or (
                            bf.join(self.run_dir, "subagent_logs") if self.run_dir else None
                        )
                        self._messages, self._last_summary, _ok = await summarize_messages(
                            completer_config=self.completer_config,
                            messages=self._messages,
                            last_summary=self._last_summary,
                            config=self.config.summary_config,
                            prune_individual=False,
                            log_dir=log_dir,
                            step=num_steps,
                        )
                        ctx_logger.info(
                            "Context reduction: %s",
                            "summary succeeded" if _ok else "fell back to prune",
                        )
                    else:
                        ctx_logger.info("Context length exceeded, pruning messages")
                        self._messages = prune_messages(
                            self._messages,
                            prune_individual=False,
                        )
                    continue
                except BadRequestError as e:
                    error_code = str(getattr(e, 'code', '') or '')
                    error_msg = str(getattr(e, 'message', '') or '')

                    if "context length" in error_msg.lower() or "maximum context" in error_msg.lower() or "exceeds max length" in error_msg.lower() or "input token count exceeds" in error_msg.lower():
                        # Context length exceeded — treat same as LengthFinishReasonError
                        if self.config.summary_config and self.config.summary_config.enabled:
                            ctx_logger.info("Context length exceeded (BadRequestError), attempting summary")
                            log_dir = self.config.log_dir or (
                                bf.join(self.run_dir, "subagent_logs") if self.run_dir else None
                            )
                            self._messages, self._last_summary, _ok = await summarize_messages(
                                completer_config=self.completer_config,
                                messages=self._messages,
                                last_summary=self._last_summary,
                                config=self.config.summary_config,
                                prune_individual=False,
                                log_dir=log_dir,
                                step=num_steps,
                            )
                            ctx_logger.info(
                                "Context reduction: %s",
                                "summary succeeded" if _ok else "fell back to prune",
                            )
                        else:
                            ctx_logger.info("Context length exceeded (BadRequestError), pruning messages")
                            self._messages = prune_messages(
                                self._messages,
                                prune_individual=False,
                            )
                        continue
                    elif '-4321' in error_code or 'invalid_prompt' in error_msg:
                        # Content policy violation (-4321 / invalid_prompt).
                        # Graduated retry: 2 direct retries → 1 prune retry → return FAILED.
                        content_policy_retries += 1
                        req_logid = e.request.headers.get("X-TT-LOGID", "unknown")
                        x_request_id = getattr(e, "request_id", None) or "unknown"

                        _paper_id = "unknown"
                        if self.run_dir:
                            _dirname = os.path.basename(self.run_dir.rstrip("/"))
                            _m = re.match(
                                r'^(.+)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                                _dirname,
                            )
                            _paper_id = _m.group(1) if _m else _dirname

                        ctx_logger.warning(
                            f"Content policy violation [paper_id={_paper_id}] "
                            f"[tt_logid={req_logid}] "
                            f"[x-request-id={x_request_id}] "
                            f"(attempt {content_policy_retries}/4)",
                            paper_id=_paper_id,
                            error_code=error_code,
                            tt_logid=req_logid,
                            x_request_id=x_request_id,
                            status_code=e.status_code,
                            response_body=e.body,
                            response_headers=dict(e.response.headers) if e.response else None,
                        )

                        if content_policy_retries <= 2:
                            # Direct retry (handles transient/non-deterministic violations)
                            ctx_logger.info(
                                f"Direct retry {content_policy_retries}/2 for content policy violation"
                            )
                            continue
                        elif content_policy_retries == 3:
                            # Prune then retry (handles content-triggered violations)
                            ctx_logger.info("Pruning messages then retrying for content policy violation")
                            self._messages = prune_messages(
                                self._messages,
                                prune_individual=True,
                            )
                            continue
                        else:
                            # Exhausted all retries, return FAILED
                            ctx_logger.error(
                                f"Content policy violation retry exhausted "
                                f"({content_policy_retries} attempts) — stopping subagent"
                            )
                            return SubagentOutput(
                                subagent_name=self.name,
                                status=SubagentStatus.FAILED,
                                content=self._truncate_buffer_output("\n".join(self._output_buffer)),
                                error_message=(
                                    f"Content policy violation (invalid_prompt -4321). "
                                    f"Exhausted all retries ({content_policy_retries} attempts: "
                                    f"2 direct + 1 prune)."
                                ),
                                num_steps=num_steps,
                                runtime_seconds=time.time() - start_time,
                                token_usage=self._total_tokens,
                                log_path=log_path,
                            )
                    else:
                        # Other BadRequestErrors (-4003 "No tool output found",
                        # malformed content, etc.) — try fixing message consistency
                        # first (zero information loss).
                        ctx_logger.warning(
                            f"BadRequestError in subagent loop, fixing message "
                            f"consistency: {e.message}",
                            error_code=error_code,
                        )
                        self._messages = fix_message_consistency(self._messages)
                        continue

                # Reset content policy retry counter on successful request
                content_policy_retries = 0

                # Add response to messages (LoggableMessages handles logging)
                self._messages += response_messages

                # Extract text content for output buffer
                for msg in response_messages:
                    if isinstance(msg, dict) and msg.get("content"):
                        self._output_buffer.append(str(msg["content"]))

                # Handle tool calls
                if tool_calls:
                    consecutive_empty_responses = 0  # Reset on valid tool call
                    for tool_call in tool_calls:
                        try:
                            handled = await self._execute_tool_call(
                                tool_call, tools, computer, constraints
                            )
                            if handled is not None:
                                self._messages.append(handled)
                                # Log tool result (structured JSONL)
                                try:
                                    log_tool_result_event(
                                        convo_path=convo_path,
                                        run_id=run_id,
                                        step=num_steps,
                                        tool_call=tool_call,
                                        tool_message=dict(handled),
                                    )
                                except Exception:
                                    pass
                        except SubagentCompleteSignal as signal:
                            # Clean completion via the subagent_complete tool
                            ctx_logger.info(f"Subagent {self.name} signaled completion")
                            output, artifacts = self._post_process_output(
                                signal.output, {**artifacts, **signal.artifacts}
                            )
                            return SubagentOutput(
                                subagent_name=self.name,
                                status=SubagentStatus.COMPLETED,
                                content=output,
                                artifacts=artifacts,
                                num_steps=num_steps,
                                runtime_seconds=time.time() - start_time,
                                token_usage=self._total_tokens,
                                log_path=log_path,
                            )
                else:
                    # No tool calls, check for consecutive empty responses
                    consecutive_empty_responses += 1
                    if consecutive_empty_responses >= MAX_CONSECUTIVE_EMPTY_RESPONSES:
                        ctx_logger.warning(
                            f"Subagent {self.name} produced {MAX_CONSECUTIVE_EMPTY_RESPONSES} "
                            "consecutive empty responses, force completing."
                        )
                        output, artifacts = self._post_process_output(
                            self._truncate_buffer_output("\n".join(self._output_buffer)), artifacts
                        )
                        error_msg = (
                            f"Subagent was force-terminated after {MAX_CONSECUTIVE_EMPTY_RESPONSES} "
                            "consecutive empty responses (no tool calls). "
                            "The task may be incomplete. Consider retrying with clearer instructions."
                        )
                        return SubagentOutput(
                            subagent_name=self.name,
                            status=SubagentStatus.FAILED,
                            content=output,
                            error_message=error_msg,
                            artifacts=artifacts,
                            num_steps=num_steps,
                            runtime_seconds=time.time() - start_time,
                            token_usage=self._total_tokens,
                            log_path=log_path,
                        )
                    self._messages.append({
                        "role": "user",
                        "content": "Continue with your analysis. When done, use the subagent_complete tool to submit your findings.",
                    })

            # Reached max steps without calling subagent_complete — treat as timeout
            ctx_logger.info(f"Subagent {self.name} reached max steps ({self.config.max_steps})")
            output, artifacts = self._post_process_output(
                self._truncate_buffer_output("\n".join(self._output_buffer)), artifacts
            )

            return SubagentOutput(
                subagent_name=self.name,
                status=SubagentStatus.TIMEOUT,
                content=output,
                artifacts=artifacts,
                error_message=f"Reached max steps ({self.config.max_steps}) without completion",
                num_steps=num_steps,
                runtime_seconds=time.time() - start_time,
                token_usage=self._total_tokens,
                log_path=log_path,
            )

        except Exception as e:
            ctx_logger.error(f"Subagent {self.name} failed: {e}")
            return SubagentOutput(
                subagent_name=self.name,
                status=SubagentStatus.FAILED,
                content=self._truncate_buffer_output("\n".join(self._output_buffer)),
                error_message=str(e),
                num_steps=num_steps,
                runtime_seconds=time.time() - start_time,
                token_usage=self._total_tokens,
                log_path=log_path,
            )
