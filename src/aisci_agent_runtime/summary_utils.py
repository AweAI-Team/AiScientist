"""
Context summarization utilities shared by paper and mle agent loops.

This keeps the message-reduction behavior close to PaperBench:
- summarize the oldest complete turns instead of dropping them immediately
- support incremental summaries
- fall back to prune when summarization fails or is not worthwhile
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aisci_agent_runtime.llm_client import LLMClient


SUMMARY_FIRST_TIME_PROMPT = """You are summarizing an earlier part of a long conversation so the agent can continue with condensed context.

Task:
{task}

Conversation history to summarize:
{segment}

Produce a concise summary that preserves:
- important decisions and conclusions
- file paths, commands, metrics, and outcomes
- what was already tried
- what still needs to be done

Output under the heading "Essential Information:" and nothing else.
"""


SUMMARY_INCREMENTAL_PROMPT = """You are merging a previous summary with new conversation content.

Task:
{task}

Previous summary:
{last_summary}

New conversation segment:
{segment}

Produce a single updated summary that preserves:
- important decisions and conclusions
- file paths, commands, metrics, and outcomes
- what was already tried
- what still needs to be done

Output under the heading "Essential Information:" and nothing else.
"""


SUMMARY_USER_INTRO = (
    "Below is a summary of the earlier part of the conversation. "
    "This summary condenses key information from earlier steps; "
    "please consider it carefully and use it as the basis for further reasoning."
)


@dataclass(frozen=True)
class SummaryConfig:
    enabled: bool = True
    segment_ratio: float = 0.3
    min_turns: int = 4
    segment_max_chars: int = 25_000
    tool_result_max_chars: int = 500
    incremental: bool = True
    max_summary_chars: int = 4_000
    summary_truncate_chars: int = 3_000
    task_desc_max_chars: int = 2_000
    max_ratio: float = 0.95
    ratio_step: float = 0.1
    min_summary_len: int = 50


def parse_rest_into_turns(rest: list[dict]) -> list[list[dict]]:
    """Parse messages after the first user into complete turns."""
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
        turns.append([msg])
        i += 1
    return turns


def serialize_segment_messages(
    segment_messages: list[dict],
    tool_result_max_chars: int = 500,
    segment_max_chars: int = 25_000,
) -> str:
    """Serialize a message segment into compact plain text for summarization."""
    parts: list[str] = []
    for msg in segment_messages:
        role = msg.get("role", "")
        if role == "user":
            content = _flatten_content(msg.get("content") or "")
            parts.append("[User]\n" + (content or "(empty)"))
        elif role == "assistant":
            content = _flatten_content(msg.get("content") or "")
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
            content = _flatten_content(msg.get("content") or "")
            if len(content) > tool_result_max_chars:
                content = content[:tool_result_max_chars] + "... (truncated)"
            parts.append(f"[Tool result: {call_id}]\n{content}")
        else:
            parts.append(f"[{role}]\n{_flatten_content(msg.get('content') or '')}")
    segment = "\n\n".join(parts)
    if len(segment) > segment_max_chars:
        segment = "(Earlier part of this segment was truncated due to length.)\n\n" + segment[-segment_max_chars:]
    return segment


def summarize_messages(
    *,
    llm: "LLMClient",
    messages: list[dict],
    config: SummaryConfig,
    task_description: str = "",
    last_summary: str | None = None,
    log_path: str | None = None,
    step: int | None = None,
    actor: str | None = None,
) -> tuple[list[dict], str | None, bool]:
    """Summarize older complete turns.

    Returns ``(new_messages, updated_summary, succeeded)``.
    """
    if not config.enabled:
        return messages, last_summary, False

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) <= 2:
        return messages, last_summary, False

    first_user = non_system[0] if non_system[0].get("role") == "user" else None
    rest = non_system[1:] if first_user else non_system
    turns = parse_rest_into_turns(rest)
    if len(turns) < config.min_turns:
        return messages, last_summary, False

    ratio = config.segment_ratio
    task = _truncate(task_description.strip(), config.task_desc_max_chars)

    while ratio <= config.max_ratio + 1e-9:
        num_turns = max(1, int(len(turns) * ratio))
        if num_turns >= len(turns):
            num_turns = len(turns) - 1
        if num_turns <= 0:
            return messages, last_summary, False

        segment_turns = turns[:num_turns]
        remaining_turns = turns[num_turns:]
        segment_messages = [msg for turn in segment_turns for msg in turn]
        remainder_messages = [msg for turn in remaining_turns for msg in turn]

        prompt = _summary_prompt(
            task=task,
            segment=serialize_segment_messages(
                segment_messages,
                tool_result_max_chars=config.tool_result_max_chars,
                segment_max_chars=config.segment_max_chars,
            ),
            last_summary=last_summary,
            use_incremental=config.incremental and bool(last_summary),
        )
        try:
            response = llm.chat([{"role": "user", "content": prompt}], tools=None)
        except Exception:
            _log_summary_request(
                log_path,
                step=step,
                actor=actor,
                target_drop_turns=num_turns,
                ratio=ratio,
                num_turns=len(turns),
                segment_text=prompt,
                summary_text="",
                status="llm_error",
            )
            ratio += config.ratio_step
            continue

        raw_summary = (response.text_content or "").strip()
        summary = _extract_summary(raw_summary)
        if len(summary) < config.min_summary_len:
            _log_summary_request(
                log_path,
                step=step,
                actor=actor,
                target_drop_turns=num_turns,
                ratio=ratio,
                num_turns=len(turns),
                segment_text=prompt,
                summary_text=summary,
                status="too_short",
            )
            ratio += config.ratio_step
            continue
        if len(summary) > config.max_summary_chars:
            summary = _truncate(summary, config.summary_truncate_chars)

        _log_summary_request(
            log_path,
            step=step,
            actor=actor,
            target_drop_turns=num_turns,
            ratio=ratio,
            num_turns=len(turns),
            segment_text=prompt,
            summary_text=summary,
            status="accepted",
        )

        summary_message = {
            "role": "user",
            "content": f"{SUMMARY_USER_INTRO}\n\n{summary}",
        }
        rebuilt = [*system_msgs]
        if first_user:
            rebuilt.append(first_user)
        rebuilt.append(summary_message)
        rebuilt.extend(remainder_messages)
        return rebuilt, summary, True

    return messages, last_summary, False


def _summary_prompt(*, task: str, segment: str, last_summary: str | None, use_incremental: bool) -> str:
    if use_incremental and last_summary:
        return SUMMARY_INCREMENTAL_PROMPT.format(
            task=task or "(task description unavailable)",
            last_summary=last_summary,
            segment=segment,
        )
    return SUMMARY_FIRST_TIME_PROMPT.format(
        task=task or "(task description unavailable)",
        segment=segment,
    )


def _flatten_content(content: str | list | object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _extract_summary(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Essential Information:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _log_summary_request(
    log_path: str | None,
    *,
    step: int | None,
    actor: str | None,
    target_drop_turns: int,
    ratio: float,
    num_turns: int,
    segment_text: str,
    summary_text: str,
    status: str,
) -> None:
    if not log_path:
        return
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "step": step,
            "actor": actor,
            "status": status,
            "N": target_drop_turns,
            "ratio_pct": int(ratio * 100),
            "num_turns": num_turns,
            "segment_chars": len(segment_text),
            "summary_chars": len(summary_text),
            "segment_preview": segment_text[:1000] + ("..." if len(segment_text) > 1000 else ""),
            "summary_preview": summary_text[:1000] + ("..." if len(summary_text) > 1000 else ""),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(1, limit // 2)
    return text[:keep] + "\n...[truncated]...\n" + text[-keep:]
