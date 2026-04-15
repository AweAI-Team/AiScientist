"""
Subagent base class.

Preserves:
- ``SubagentConfig``: max_steps, time_limit, reminder_freq, log_dir
- ``SubagentOutput``: status, content, artifacts, token_usage, runtime
- ``Subagent.run()``: the main execution loop with:
    - LLM call → tool execution → message management
    - ``ContextLengthError`` → prune messages and retry
    - ``SubagentCompleteSignal`` → clean exit
    - Periodic time/step reminders
    - Force-termination after 10 consecutive empty responses
    - JSONL conversation logging
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from openai import BadRequestError
from llm_client import LLMClient, ContextLengthError, ContentPolicyError
from shell_interface import ShellInterface
from tools.base import Tool, SubagentCompleteSignal, SubagentCompleteTool

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

logger = structlog.stdlib.get_logger(component=__name__)


# ====================================================================== #
# Data structures
# ====================================================================== #

class SubagentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SubagentConfig:
    max_steps: int = 50
    time_limit: int = 300
    reminder_freq: int = 10
    log_dir: str = "/home/agent/subagent_logs"
    output_dir: str = "/home/agent"


@dataclass
class SubagentOutput:
    status: SubagentStatus
    content: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    num_steps: int = 0
    runtime_seconds: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    log_path: str | None = None


# ====================================================================== #
# Message pruning
# ====================================================================== #

def fix_message_consistency(messages: list[dict]) -> list[dict]:
    """Fix tool_call / tool_result pairing issues WITHOUT removing valid messages.

    The OAI API requires:
    - Every tool result must have a preceding assistant message with a matching tool_call
    - Every assistant tool_call must have a following tool result

    Use this instead of ``prune_messages`` when the problem is message
    consistency (e.g. -4003 "No tool output found") rather than context length.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Pass 1: drop orphaned tool results (no parent assistant)
    pass1: list[dict] = []
    active_tool_ids: set[str] = set()
    for msg in non_system:
        role = msg.get("role")
        if role == "assistant":
            active_tool_ids = {tc["id"] for tc in (msg.get("tool_calls") or [])}
            pass1.append(msg)
        elif role == "tool":
            if msg.get("tool_call_id") in active_tool_ids:
                pass1.append(msg)
        else:
            active_tool_ids = set()
            pass1.append(msg)

    # Pass 2: fix orphaned tool_calls (no matching tool result)
    cleaned = _fix_orphaned_tool_calls(pass1)

    return system_msgs + cleaned


def prune_messages(messages: list[dict]) -> list[dict]:
    """Remove the oldest ~30 % of conversation messages.

    Message order is preserved (only the prefix is dropped), so GLM-5 Preserved
    Thinking's requirement that reasoning_content blocks stay in original order
    is satisfied.

    Invariants preserved:
    - All ``system`` messages are kept
    - The first ``user`` message (task prompt) is kept
    - Messages whose content contains "prompt is too long" are filtered out
      (prevents feeding API error text back as conversation history)
    - Tool messages are only kept if their tool_call_id was registered by a
      surviving assistant message (exact id-set tracking)
    - Assistant tool_calls are only kept if their tool results survive
      (Pass 2 — also handles partial-result and no-result cases)
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= 2:
        return messages

    first_user = non_system[0] if non_system and non_system[0]["role"] == "user" else None
    rest = non_system[1:] if first_user else non_system

    drop_count = max(1, len(rest) * 3 // 10)
    kept = rest[drop_count:]

    # Pass 1: discard orphaned tool results and messages that contain the
    # "prompt is too long" API error text.
    # Uses active_tool_ids for exact id-set tracking:
    #   active_tool_ids updated on assistant messages
    #   tool messages kept only when tool_call_id in active_tool_ids
    #   user messages reset active_tool_ids
    pass1: list[dict] = []
    active_tool_ids: set[str] = set()
    for m in kept:
        role = m.get("role", "")

        # Filter: drop any message whose text content contains the literal
        # API error string "prompt is too long" — drop so it is not replayed.
        content_str = m.get("content") or ""
        if isinstance(content_str, list):
            content_str = " ".join(
                item.get("text", "") for item in content_str
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if "prompt is too long" in content_str:
            continue

        if role == "assistant":
            active_tool_ids = {tc["id"] for tc in (m.get("tool_calls") or [])}
            pass1.append(m)
        elif role == "tool":
            if m.get("tool_call_id") in active_tool_ids:
                pass1.append(m)
            # else: orphaned tool result — drop silently
        else:
            # user or other role: reset active set
            active_tool_ids = set()
            pass1.append(m)

    # Pass 2: fix orphaned tool_calls (assistant tool_calls whose results
    # were dropped during Pass 1 — OAI rejects these with -4003).
    clean = _fix_orphaned_tool_calls(pass1)

    result = system_msgs[:]
    if first_user:
        result.append(first_user)
    result.extend(clean)
    return result


def _fix_orphaned_tool_calls(messages: list[dict]) -> list[dict]:
    """Strip or patch assistant tool_calls that lost their tool results.

    For each assistant message with tool_calls, look ahead for matching
    tool results:
    - All results present → keep as-is
    - Partial results → keep only the tool_calls that have results
    - No results at all → keep text content only, strip tool_calls
    """
    cleaned: list[dict] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}
            found_ids: set[str] = set()
            for j in range(i + 1, len(messages)):
                nxt = messages[j]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") in tool_call_ids:
                    found_ids.add(nxt["tool_call_id"])
                elif nxt.get("role") != "tool":
                    break
            if found_ids == tool_call_ids:
                cleaned.append(msg)
            elif found_ids:
                msg_copy = dict(msg)
                msg_copy["tool_calls"] = [
                    tc for tc in msg["tool_calls"] if tc["id"] in found_ids
                ]
                cleaned.append(msg_copy)
            else:
                if msg.get("content"):
                    msg_copy = dict(msg)
                    msg_copy.pop("tool_calls", None)
                    cleaned.append(msg_copy)
        else:
            cleaned.append(msg)
    return cleaned


# ====================================================================== #
# Per-message truncation — equivalent to per-message length handling in utils
# ====================================================================== #

# Default maximum tiktoken tokens per single message when prune_individual=True.
# Default 190_000 is a buffer under ~200k context (for GPT / tiktoken-aligned
# models).  For GLM models where tiktoken underestimates by ~1.26×, the correct
# tiktoken-safe per-message limit is:
#   GLM-5:  182272 / 1.26 - 5000 ≈ 139660  (= AISCI_CONTEXT_WINDOW)
#   GLM-4.7:167232 / 1.26 - 5000 ≈ 127700  (= AISCI_CONTEXT_WINDOW)
# callers should pass max_tokens_per_message = context_window (already tiktoken-corrected)
# rather than using this global default whenever the model context_window is known.
_MAX_TOKENS_PER_MESSAGE_DEFAULT = 190_000


def _truncate_string_by_tokens(text: str, tokenizer, max_tok: int) -> str:
    """Truncate *text* to *max_tok* tokens, keeping head and tail halves.

    Head + tail keep with middle ellipsis (same idea as classic truncate_string):
      first_half = decode(tokens[:keep])
      second_half = decode(tokens[-keep:])
      return first_half + '\\n...[content truncated]...\\n' + second_half
    """
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= max_tok:
        return text
    keep = max_tok // 2
    first_half = tokenizer.decode(tokens[:keep])
    second_half = tokenizer.decode(tokens[-keep:])
    return first_half + "\n...[content truncated due to length]...\n" + second_half


def _truncate_message_content(msg: dict, tokenizer, max_tok: int) -> dict:
    """Truncate a single message's content to fit within *max_tok* tokens.

    Per-message handling:
    - str content   → truncate the string directly
    - list content  → distribute budget proportionally across text parts
    - non-text parts (images, etc.) are kept as-is

    GLM-5 Preserved Thinking: Only ``content`` is truncated. The ``reasoning_content``
    field (when present on assistant messages) is left unchanged so that consecutive
    reasoning_content blocks match the model's original order and are not edited.
    """
    content = msg.get("content")
    if isinstance(content, str):
        new_content = _truncate_string_by_tokens(content, tokenizer, max_tok)
        if new_content is content:
            return msg
        msg = dict(msg)
        msg["content"] = new_content
    elif isinstance(content, list):
        # Collect token counts for text items only
        token_lists: list[list[int]] = []
        token_counts: list[int] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                toks = tokenizer.encode(item.get("text", ""), disallowed_special=())
                token_lists.append(toks)
                token_counts.append(len(toks))
            else:
                token_lists.append([])
                token_counts.append(0)

        total = sum(token_counts)
        if total == 0:
            return msg

        # Distribute max_tok proportionally across text parts
        per_item = [
            max(1, int((c / total) * max_tok)) if c > 0 else 0
            for c in token_counts
        ]

        new_content = []
        changed = False
        for item, toks, budget in zip(content, token_lists, per_item):
            if isinstance(item, dict) and item.get("type") == "text" and toks:
                new_text = _truncate_string_by_tokens(item["text"], tokenizer, budget)
                if new_text != item["text"]:
                    item = dict(item)
                    item["text"] = new_text
                    changed = True
            new_content.append(item)

        if changed:
            msg = dict(msg)
            msg["content"] = new_content
    return msg


def prune_messages_individual(
    messages: list[dict],
    max_tokens_per_message: int | None = None,
) -> list[dict]:
    """Truncate oversized individual messages using tiktoken.

    Called when ``ContextLengthError.prune_individual=True`` — meaning a
    single message is so large that even after dropping old turns the context
    is still exceeded.  This follows ``prune_messages(prune_individual=True)``
    which applies ``_handle_message_len``
    to every message.

    ``max_tokens_per_message`` controls the tiktoken-based per-message cap.
    It should equal ``LLMConfig.context_window`` (already tokenizer-ratio-
    corrected for GLM) so that a single tool response cannot exceed the
    model's real input limit even when tiktoken underestimates token counts.

    If not provided, falls back to the GPT-oriented default (190_000) which
    is only accurate for models whose tokenizer ≈ tiktoken.

    Falls back to ordinary ``prune_messages`` when tiktoken is unavailable.
    """
    if not _TIKTOKEN_AVAILABLE:
        logger.warning(
            "tiktoken not available — falling back to bulk prune_messages "
            "(install tiktoken to enable per-message truncation)"
        )
        return prune_messages(messages)

    try:
        tokenizer = _tiktoken.get_encoding("o200k_base")
    except Exception as exc:
        logger.warning("tiktoken get_encoding failed — bulk prune fallback", err=str(exc))
        return prune_messages(messages)

    limit = max_tokens_per_message if max_tokens_per_message is not None else _MAX_TOKENS_PER_MESSAGE_DEFAULT
    logger.debug(
        "prune_messages_individual",
        max_tokens_per_message=limit,
        num_messages=len(messages),
    )
    return [_truncate_message_content(m, tokenizer, limit) for m in messages]


# ====================================================================== #
# Subagent ABC
# ====================================================================== #

class Subagent(ABC):
    """Abstract base class for all subagents."""

    def __init__(
        self,
        shell: ShellInterface,
        llm: LLMClient,
        config: SubagentConfig | None = None,
    ):
        self.shell = shell
        self.llm = llm
        self.config = config or SubagentConfig()

    # ---- abstract interface ----

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        ...

    # ---- optional hooks ----

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        return raw_output, artifacts

    def _build_reminder(self, step: int, elapsed: float) -> str | None:
        """Return a reminder string or None."""
        remaining = self.config.time_limit - elapsed
        pct = elapsed / self.config.time_limit * 100
        steps_left = self.config.max_steps - step
        parts = [
            f"⏱ Step {step}/{self.config.max_steps} | "
            f"Time: {_fmt(elapsed)}/{_fmt(self.config.time_limit)} ({pct:.0f}%) | "
            f"Remaining: {_fmt(remaining)} | Steps left: {steps_left}"
        ]
        if pct >= 90:
            parts.append("🚨 CRITICAL: <10 % time left — wrap up NOW and call subagent_complete.")
        elif pct >= 60:
            parts.append("⚠ Over 60 % time used — prioritise finishing your current task.")
        return "\n".join(parts)

    # ---- main loop ----

    def run(self, context: str = "") -> SubagentOutput:
        """
        Run the subagent loop.

        ``context`` is injected as the first user message and typically
        contains the task description and any prior logs.
        """
        tools = self.get_tools()
        tool_schemas = [t.get_tool_schema() for t in tools]
        tool_map = {t.name(): t for t in tools}

        run_id = time.strftime("%Y%m%d_%H%M%S")
        log_dir = self.config.log_dir
        os.makedirs(log_dir, mode=0o755, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.name}_{run_id}.jsonl")

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt()},
        ]
        if context:
            messages.append({"role": "user", "content": context})

        start = time.time()
        total_tokens: dict[str, int] = {"input": 0, "output": 0}
        empty_streak = 0

        for step in range(1, self.config.max_steps + 1):
            elapsed = time.time() - start
            if elapsed >= self.config.time_limit:
                logger.info("subagent time limit reached", name=self.name, step=step)
                return self._make_output(
                    SubagentStatus.TIMEOUT,
                    f"Time limit reached ({_fmt(self.config.time_limit)}). Last step: {step}.",
                    {},
                    step,
                    time.time() - start,
                    total_tokens,
                    log_path,
                )

            # Periodic reminder
            if step > 1 and step % self.config.reminder_freq == 0:
                reminder = self._build_reminder(step, elapsed)
                if reminder:
                    messages.append({"role": "user", "content": reminder})

            # --- LLM call ---
            try:
                resp = self.llm.chat(messages, tools=tool_schemas)
            except ContentPolicyError as e:
                # o-series safety filter — fail immediately to preserve Azure
                # account quota (5 cumulative violations → account lockout).
                logger.error(
                    "Content policy violation — failing subagent immediately",
                    name=self.name, step=step, dump=e.dump_path,
                )
                return self._make_output(
                    SubagentStatus.FAILED,
                    (
                        f"Content policy violation (o-series safety filter). "
                        f"Stopped immediately to avoid Azure account lockout. "
                        f"Dump: {e.dump_path}"
                    ),
                    {},
                    step,
                    time.time() - start,
                    total_tokens,
                    log_path,
                )
            except ContextLengthError as _ctx_err:
                if _ctx_err.prune_individual:
                    # A single message is too large — truncate per-message content
                    # first (tiktoken head+tail), then also drop old turns.
                    # prune_individual=True path.
                    # Pass context_window (already tokenizer-corrected for GLM) as
                    # the per-message limit so GLM's 1.26× token inflation doesn't
                    # overflow the model's hard limit after truncation.
                    logger.warning(
                        "context length exceeded — truncating individual messages",
                        name=self.name, step=step,
                        context_window=self.llm.config.context_window,
                    )
                    messages = prune_messages_individual(
                        messages,
                        max_tokens_per_message=self.llm.config.context_window,
                    )
                    messages = prune_messages(messages)
                else:
                    logger.warning(
                        "context length exceeded — pruning", name=self.name, step=step,
                    )
                    messages = prune_messages(messages)
                try:
                    resp = self.llm.chat(messages, tools=tool_schemas)
                except ContextLengthError:
                    logger.error(
                        "context length exceeded even after pruning — failing subagent",
                        name=self.name, step=step,
                    )
                    return self._make_output(
                        SubagentStatus.FAILED,
                        "Context length exceeded even after pruning.",
                        {},
                        step,
                        time.time() - start,
                        total_tokens,
                        log_path,
                    )
            except BadRequestError as e:
                # Non-safety BadRequestErrors (e.g., -4003): fix consistency.
                error_code = str(getattr(e, "code", "") or "")
                logger.warning(
                    "BadRequestError — fixing message consistency",
                    name=self.name, step=step, error_code=error_code,
                )
                messages = fix_message_consistency(messages)
                continue
            except Exception as e:
                logger.error("LLM call failed", name=self.name, step=step, err=str(e))
                return self._make_output(
                    SubagentStatus.FAILED,
                    f"LLM call failed: {e}",
                    {},
                    step,
                    time.time() - start,
                    total_tokens,
                    log_path,
                )

            for k in ("input", "output"):
                total_tokens[k] += resp.usage.get(k, 0)

            # Log model response
            _resp_event: dict = {
                "event": "model_response",
                "step": step,
                "text": resp.text_content,
                "tool_calls": [
                    {"id": tc.call_id, "name": tc.name, "args": tc.arguments}
                    for tc in resp.tool_calls
                ],
                "usage": resp.usage,
                "ts": time.time(),
            }
            # GLM-5 / DeepSeek-R1: store reasoning chain when present
            if resp.reasoning_content:
                _resp_event["reasoning_content"] = resp.reasoning_content
            self._log_event(log_path, _resp_event)

            # Build assistant message
            asst_msg: dict[str, Any] = {"role": "assistant", "content": resp.text_content}
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
            messages.append(asst_msg)

            # Empty response detection (10 consecutive empties → force-terminate)
            if not resp.tool_calls and not resp.text_content:
                empty_streak += 1
                if empty_streak >= 10:
                    return self._make_output(
                        SubagentStatus.FAILED,
                        "Force-terminated after 10 consecutive empty responses.",
                        {},
                        step,
                        time.time() - start,
                        total_tokens,
                        log_path,
                    )
                messages.append({"role": "user", "content": "Please continue with your task, or call subagent_complete if done."})
                continue
            empty_streak = 0

            # --- Execute tool calls ---
            if not resp.tool_calls:
                messages.append({"role": "user", "content": "Continue. Use tools to make progress, or call subagent_complete when done."})
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

                try:
                    result = tool.execute(self.shell, **tc.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": str(result),
                    })
                except SubagentCompleteSignal as sig:
                    # Clean completion
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": f"Subagent completed: {sig.content}",
                    })
                    self._log_event(log_path, {
                        "event": "subagent_complete",
                        "step": step,
                        "content": sig.content,
                        "ts": time.time(),
                    })
                    return self._make_output(
                        SubagentStatus.COMPLETED,
                        sig.content,
                        sig.artifacts,
                        step,
                        time.time() - start,
                        total_tokens,
                        log_path,
                    )
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": f"Error executing {tc.name}: {e}",
                    })

                self._log_event(log_path, {
                    "event": "tool_result",
                    "step": step,
                    "tool": tc.name,
                    "result_len": len(messages[-1].get("content", "")),
                    "ts": time.time(),
                })

        # Exhausted steps
        return self._make_output(
            SubagentStatus.TIMEOUT,
            f"Max steps reached ({self.config.max_steps}).",
            {},
            self.config.max_steps,
            time.time() - start,
            total_tokens,
            log_path,
        )

    # ---- helpers ----

    def _make_output(
        self,
        status: SubagentStatus,
        content: str,
        artifacts: dict,
        steps: int,
        runtime: float,
        tokens: dict,
        log_path: str,
    ) -> SubagentOutput:
        content, artifacts = self._post_process_output(content, artifacts)
        return SubagentOutput(
            status=status,
            content=content,
            artifacts=artifacts,
            num_steps=steps,
            runtime_seconds=runtime,
            token_usage=tokens,
            log_path=log_path,
        )

    @staticmethod
    def _log_event(path: str, event: dict) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass


def _fmt(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
