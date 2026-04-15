"""
Logging utilities for human-readable and JSONL conversation logs.

Generates:
- ``agent.log``:  human-readable conversation log with box-drawing formatting
- ``conversation.jsonl``:  structured JSONL event log (model_response / tool_result)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


# ====================================================================== #
# Human-readable agent.log
# ====================================================================== #

BOX_WIDTH = 100
MAX_LINES = 600  # Truncate messages longer than this


def log_messages_to_file(messages: list[dict], path: str) -> None:
    """
    Write messages to a human-readable log file using box-drawing chars.

    This is called after every message append so it rewrites the full file.
    For long conversations the rewrite is cheap compared to LLM latency.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        # Tool calls inside assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            lines.extend(_box(f"assistant", _short(content, MAX_LINES)))
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.dumps(json.loads(args), indent=2)
                    except Exception:
                        pass
                lines.extend(_box(f"Tool Call: {name}, ID: {tc.get('id', '?')}", _short(args, MAX_LINES)))
        elif role == "tool":
            call_id = msg.get("tool_call_id", "?")
            lines.extend(_box(f"Tool Output: {call_id}", _short(content, MAX_LINES)))
        else:
            lines.extend(_box(role, _short(content, MAX_LINES)))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o644)  # so host user can read when LOGS_DIR is volume-mounted
    except OSError:
        pass


def _box(header: str, body: str) -> list[str]:
    """Render a message in a unicode box."""
    w = BOX_WIDTH
    lines = [
        f"╭─ {header} " + "─" * max(0, w - len(header) - 4) + "╮",
    ]
    for line in body.split("\n"):
        # Truncate overly long lines
        if len(line) > w - 4:
            line = line[: w - 7] + "..."
        lines.append(f"│ {line:<{w - 4}} │")
    lines.append("╰" + "─" * (w - 2) + "╯")
    lines.append("")
    return lines


def _short(text: str, max_lines: int) -> str:
    """Truncate to *max_lines* lines, keeping head and tail."""
    parts = text.split("\n")
    if len(parts) <= max_lines:
        return text
    half = max_lines // 2
    return "\n".join(
        parts[:half]
        + [f"... [{len(parts) - max_lines} lines truncated] ..."]
        + parts[-half:]
    )


# ====================================================================== #
# Structured conversation.jsonl
# ====================================================================== #

def log_model_response_event(
    convo_path: str,
    run_id: str,
    step: int,
    n_input_messages: int,
    text_content: str | None,
    tool_calls: list[dict],
    usage: dict,
    reasoning_content: str | None = None,
) -> None:
    """Log a model_response event to the conversation JSONL."""
    entry: dict = {
        "ts": time.time(),
        "run_id": run_id,
        "step": step,
        "event": "model_response",
        "n_input_messages": n_input_messages,
        "text": text_content,
        "tool_calls": tool_calls,
        "usage": usage,
    }
    # GLM-5 / DeepSeek-R1 expose a separate reasoning chain.  Store it when
    # present so it can be reviewed alongside the final answer.
    if reasoning_content:
        entry["reasoning_content"] = reasoning_content
    _append_jsonl(convo_path, entry)


def log_tool_result_event(
    convo_path: str,
    run_id: str,
    step: int,
    tool_name: str,
    tool_call_id: str,
    result_preview: str,
) -> None:
    """Log a tool_result event to the conversation JSONL."""
    # Keep result preview short for the JSONL
    preview = result_preview[:2000] if len(result_preview) > 2000 else result_preview
    _append_jsonl(convo_path, {
        "ts": time.time(),
        "run_id": run_id,
        "step": step,
        "event": "tool_result",
        "tool": tool_name,
        "tool_call_id": tool_call_id,
        "result_len": len(result_preview),
        "result_preview": preview,
    })


def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj, default=str) + "\n")
        os.chmod(path, 0o644)  # so host user can read when LOGS_DIR is volume-mounted
    except Exception:
        pass
