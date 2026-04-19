"""
Context summarization for the AI Scientist solver and subagents.

When context length is exceeded, the oldest 30% of turns (by count) can be
summarized with the same LLM and replaced by a single user message containing
the summary.  Supports incremental summarization (merge with previous summary).

Provides:
- ``SummaryConfig``: configuration dataclass for summary-based context reduction
- ``summarize_messages()``: async main entry point — partition → serialize →
  LLM call → reassemble, with adaptive ratio and prune fallback.

Ported from mle-bench ``agents/aisci/summary_utils.py`` with adaptations for
paperbench's async architecture, ``LoggableMessages`` wrapper, and OpenAI's
``LengthFinishReasonError``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from openai import NOT_GIVEN, BadRequestError, LengthFinishReasonError
from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
    from paperbench.solvers.basicagent.logging import LoggableMessages

logger = structlog.stdlib.get_logger(component=__name__)


# ====================================================================== #
# Prompt constants (ported from mle-bench prompts/templates.py)
# ====================================================================== #

SUMMARY_FIRST_TIME_PROMPT = """You are summarizing an earlier part of a long conversation so the agent can continue the task with a condensed context.

**Task (what the agent is working on):**
{task}

**Conversation history to summarize:**
{segment}

Produce a concise summary that preserves:
- Key decisions and conclusions
- Important file paths, metrics, and outcomes
- What has been tried and what remains to do

Output your summary under the heading "Essential Information:" (nothing else). Be factual and compact so the agent can resume work without re-reading the full history."""

SUMMARY_INCREMENTAL_PROMPT = """You are merging a previous summary with new conversation content to keep context condensed.

**Task (what the agent is working on):**
{task}

**Previous summary (already condensed):**
{last_summary}

**New conversation segment to merge in:**
{segment}

Produce an updated single summary that merges the previous summary with the new segment. Preserve:
- Key decisions and conclusions
- Important file paths, metrics, and outcomes
- What has been tried and what remains to do

Output your updated summary under the heading "Essential Information:" (nothing else). Be factual and compact."""

SUMMARY_USER_INTRO = (
    "Below is a summary of the earlier part of the conversation. "
    "This summary condenses key information from earlier steps; "
    "please consider it carefully and use it as the basis for further "
    "reasoning and optimization to improve your score."
)


# ====================================================================== #
# Turn parsing & serialization (ported verbatim from mle-bench)
# ====================================================================== #

def parse_rest_into_turns(rest: list[dict]) -> list[list[dict]]:
    """Parse messages after the first user (rest) into complete turns.

    Turn = (i) one user message, or (ii) one assistant message plus the
    maximal contiguous following tool messages whose tool_call_id is in that
    assistant's tool_calls.

    Returns:
        List of turns; each turn is a list of message dicts (in order).
    """
    turns: list[list[dict]] = []
    i = 0
    while i < len(rest):
        msg = rest[i]
        role = msg.get("role", "")
        if role == "user":
            turns.append([msg])
            i += 1
            continue
        if role == "assistant":
            turn = [msg]
            tool_ids = {tc["id"] for tc in (msg.get("tool_calls") or [])}
            j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool":
                if rest[j].get("tool_call_id") in tool_ids:
                    turn.append(rest[j])
                j += 1
            turns.append(turn)
            i = j
            continue
        if role == "tool":
            turns.append([msg])
            i += 1
            continue
        turns.append([msg])
        i += 1
    return turns


def serialize_segment_messages(
    segment_messages: list[dict],
    tool_result_max_chars: int = 500,
    segment_max_chars: int = 25000,
) -> str:
    """Serialize messages to text for the summary prompt."""
    parts: list[str] = []
    for msg in segment_messages:
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            parts.append("[User]\n" + (content or "(empty)"))
        elif role == "assistant":
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            line = "[Assistant]\n" + (content or "(empty)")
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                short_calls = []
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "?")
                    args = (tc.get("function") or {}).get("arguments", "") or ""
                    if len(args) > 80:
                        args = args[:77] + "..."
                    short_calls.append(f"{name}({args})")
                line += "\n[Tool calls: " + ", ".join(short_calls) + "]"
            parts.append(line)
        elif role == "tool":
            call_id = msg.get("tool_call_id", "?")
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            if len(content) > tool_result_max_chars:
                content = content[:tool_result_max_chars] + "... (truncated)"
            parts.append(f"[Tool result: {call_id}]\n{content}")
        else:
            parts.append(f"[{role}]\n{msg.get('content', '')}")
    segment_str = "\n\n".join(parts)
    if len(segment_str) > segment_max_chars:
        segment_str = (
            "(Earlier part of this segment was truncated due to length.)\n\n"
            + segment_str[-segment_max_chars:]
        )
    return segment_str


# ====================================================================== #
# SummaryConfig
# ====================================================================== #

@dataclass
class SummaryConfig:
    """Configuration for summary-based context reduction.

    Used by both the main solver and subagents.  Set ``enabled=False``
    to fall back to prune-only behaviour.
    """
    enabled: bool = True
    segment_ratio: float = 0.3        # starting ratio of turns to summarize
    min_turns: int = 4                 # skip summary when fewer turns than this
    segment_max_chars: int = 25000     # max chars for serialized segment text
    tool_result_max_chars: int = 500   # per-tool-result truncation in serialization
    incremental: bool = True           # merge with previous summary when available
    max_summary_chars: int = 4000      # hard cap on summary length
    summary_truncate_chars: int = 3000 # truncate to this when exceeding hard cap
    task_desc_max_chars: int = 2000    # truncate task description in prompt
    max_ratio: float = 0.95           # upper bound on adaptive ratio
    ratio_step: float = 0.1           # increment per retry
    min_summary_len: int = 50         # quality gate: reject summaries shorter than this


# ====================================================================== #
# Core async LLM call for summary generation
# ====================================================================== #

async def _make_summary_llm_call(
    completer_config: "BasicAgentTurnCompleterConfig",
    prompt: str,
) -> str:
    """Tool-free async LLM call to generate a summary.

    Clones *completer_config*, strips tools, builds a fresh completer,
    and calls ``async_completion`` with a single user message.
    """
    cfg = completer_config.model_copy(deep=True)
    cfg.basicagent_tools = None
    cfg.tools = NOT_GIVEN
    completer = cfg.build()
    completion = await completer.async_completion(
        conversation=[{"role": "user", "content": prompt}]
    )
    msg = completion.output_messages[0]
    return (getattr(msg, "content", None) or "").strip()


# ====================================================================== #
# summarize_messages — main async entry point
# ====================================================================== #

async def summarize_messages(
    completer_config: "BasicAgentTurnCompleterConfig",
    messages: "LoggableMessages[ChatCompletionMessageParam]",
    last_summary: str | None,
    config: SummaryConfig,
    prune_individual: bool = False,
    log_dir: str | None = None,
    step: int = 0,
) -> tuple["LoggableMessages[ChatCompletionMessageParam]", str | None, bool]:
    """Attempt to summarize older turns; fall back to prune on failure.

    Args:
        completer_config: LLM completer configuration (tools will be stripped).
        messages: current conversation (``LoggableMessages`` wrapper).
        last_summary: previous summary text for incremental mode, or None.
        config: ``SummaryConfig`` controlling ratios, thresholds, etc.
        prune_individual: if True, also run per-message truncation on prune fallback.
        log_dir: directory for ``context_summary_requests.jsonl`` (None = skip).
        step: current step number (for logging only).

    Returns:
        ``(new_messages, updated_last_summary, succeeded)``
        - ``succeeded=True``: summary produced, messages replaced.
        - ``succeeded=False``: fell back to prune.
    """
    from paperbench.solvers.basicagent.logging import LoggableMessages
    from paperbench.solvers.basicagent.utils import prune_messages

    # ---- Summary disabled → prune directly ----
    if not config.enabled:
        pruned = prune_messages(messages, prune_individual=prune_individual)
        return pruned, last_summary, False

    # ---- Partition ----
    data = messages.data  # underlying list
    system_msgs: list[ChatCompletionMessageParam] = [
        m for m in data if m.get("role") == "system"
    ]
    non_system = [m for m in data if m.get("role") != "system"]
    first_user = non_system[0] if (non_system and non_system[0].get("role") == "user") else None
    rest = non_system[1:] if first_user else non_system

    # ---- Parse turns ----
    turns = parse_rest_into_turns(rest)
    num_turns = len(turns)

    # ---- Too few turns → prune ----
    if num_turns < config.min_turns:
        logger.info(
            "Too few turns for summary, falling back to prune",
            step=step, num_turns=num_turns, min_turns=config.min_turns,
        )
        pruned = prune_messages(messages, prune_individual=prune_individual)
        return pruned, last_summary, False

    # ---- Build task description for prompt ----
    task_content = (first_user or {}).get("content") or ""
    if isinstance(task_content, list):
        task_content = " ".join(
            item.get("text", "") for item in task_content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    task_for_prompt = (
        (task_content[:config.task_desc_max_chars] + "\n(task description truncated.)")
        if len(task_content) > config.task_desc_max_chars
        else task_content
    )

    # ---- Adaptive ratio loop ----
    ratio = config.segment_ratio
    while ratio <= config.max_ratio:
        target_drop_turns = max(1, min(int(num_turns * ratio), num_turns - 1))
        segment_messages = [m for t in turns[:target_drop_turns] for m in t]
        kept_tail_messages = [m for t in turns[target_drop_turns:] for m in t]

        segment_text = serialize_segment_messages(
            segment_messages,
            tool_result_max_chars=config.tool_result_max_chars,
            segment_max_chars=config.segment_max_chars,
        )

        # Build prompt (first-time vs incremental)
        if config.incremental and last_summary:
            prompt = SUMMARY_INCREMENTAL_PROMPT.format(
                task=task_for_prompt, last_summary=last_summary, segment=segment_text,
            )
        else:
            prompt = SUMMARY_FIRST_TIME_PROMPT.format(
                task=task_for_prompt, segment=segment_text,
            )

        try:
            summary_raw = await _make_summary_llm_call(completer_config, prompt)

            if not summary_raw or len(summary_raw) < config.min_summary_len:
                raise ValueError("Summary response empty or too short")

            # Extract content after "Essential Information:" heading
            if "Essential Information:" in summary_raw:
                summary_text = summary_raw.split("Essential Information:", 1)[-1].strip()
            else:
                summary_text = summary_raw

            # Truncate if too long
            if len(summary_text) > config.max_summary_chars:
                summary_text = summary_text[:config.summary_truncate_chars] + "\n(summary truncated.)"

            # Reassemble messages
            summary_user_content = SUMMARY_USER_INTRO + "\n\nSummary:\n" + summary_text
            summary_user_msg: ChatCompletionMessageParam = {
                "role": "user",
                "content": summary_user_content,
            }
            new_list: list[ChatCompletionMessageParam] = (
                system_msgs
                + ([first_user] if first_user else [])
                + [summary_user_msg]
                + kept_tail_messages
            )
            new_messages = LoggableMessages(
                new_list, log_path=messages._log_path, logger=messages._logger
            )

            logger.info(
                "Context summarization succeeded",
                step=step, N=target_drop_turns, ratio_pct=int(ratio * 100),
            )

            _log_summary_request(
                log_dir, step, target_drop_turns, ratio, num_turns,
                segment_text, summary_text,
            )

            return new_messages, summary_text, True

        except (LengthFinishReasonError, BadRequestError) as exc:
            # LengthFinishReasonError: context still too long after summary
            # BadRequestError with "context length": same, just a different surface
            err_str = str(exc).lower()
            if isinstance(exc, LengthFinishReasonError) or "context length" in err_str or "maximum context" in err_str or "exceeds max length" in err_str:
                logger.info(
                    "Context still over limit after summary at ratio %d%% — retrying with higher ratio",
                    int(ratio * 100), step=step,
                )
                ratio += config.ratio_step
                continue
            # Other BadRequestError — break and fall back to prune
            logger.warning(
                "Summary LLM call failed with BadRequestError: %s; falling back to prune",
                str(exc)[:200], step=step,
            )
            break
        except Exception as e:
            logger.warning(
                "Context summarization failed (reason: %s); falling back to prune",
                str(e)[:200], step=step,
            )
            break

    # ---- All ratios exhausted or error → prune fallback ----
    logger.info(
        "Summary failed at all ratios, falling back to prune", step=step,
    )
    pruned = prune_messages(messages, prune_individual=prune_individual)
    return pruned, last_summary, False


# ====================================================================== #
# Helpers
# ====================================================================== #

def _log_summary_request(
    log_dir: str | None,
    step: int,
    target_drop_turns: int,
    ratio: float,
    num_turns: int,
    segment_text: str,
    summary_text: str,
) -> None:
    """Append a record to context_summary_requests.jsonl."""
    if not log_dir:
        return
    try:
        summary_log_path = os.path.join(log_dir, "context_summary_requests.jsonl")
        record = {
            "step": step,
            "N": target_drop_turns,
            "ratio_pct": int(ratio * 100),
            "num_turns": num_turns,
            "segment_chars": len(segment_text),
            "summary_chars": len(summary_text),
            "summary_preview": summary_text[:1000] + ("..." if len(summary_text) > 1000 else ""),
        }
        with open(summary_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as _e:
        logger.debug("Failed to write context_summary_requests.jsonl", err=str(_e))
