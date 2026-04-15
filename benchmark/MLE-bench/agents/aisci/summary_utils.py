"""
Context summarization for the AI Scientist orchestrator.

When context length is exceeded, the oldest 30% of turns (by count) can be
summarized with the same LLM and replaced by a single user message containing
the summary. Supports incremental summarization (merge with previous summary).

Used only when AISCI_CONTEXT_REDUCE_STRATEGY=summary. See implementation plan:
  mle-bench_scripts_chenjie/doc/summary_instead_of_prune_implementation_plan.md
"""

from __future__ import annotations


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
            # Orphan tool message: treat as its own "turn" so we don't skip it
            turns.append([msg])
            i += 1
            continue
        # Unknown role: single-message turn
        turns.append([msg])
        i += 1
    return turns


def serialize_segment_messages(
    segment_messages: list[dict],
    tool_result_max_chars: int = 500,
    segment_max_chars: int = 25000,
) -> str:
    """Serialize a list of messages (user/assistant/tool) to text for the summary prompt.

    - user: "[User]\\n" + content (flatten list content to text)
    - assistant: "[Assistant]\\n" + content; if tool_calls, append "\\n[Tool calls: name(args), ...]"
    - tool: "[Tool result: <id>]\\n" + content truncated to tool_result_max_chars

    If total length > segment_max_chars, truncate from the left (keep tail) and
    optionally prepend a truncation notice.
    """
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
                content = " ".join(
                    str(item) for item in content
                )
            if len(content) > tool_result_max_chars:
                content = content[: tool_result_max_chars] + "... (truncated)"
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


# Fixed intro for the summary_user message (plan §7)
SUMMARY_USER_INTRO = (
    "Below is a summary of the earlier part of the conversation. "
    "This summary condenses key information from earlier steps; "
    "please consider it carefully and use it as the basis for further reasoning and optimization to improve your score."
)
