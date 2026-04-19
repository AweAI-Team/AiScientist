from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Mapping, Sequence

import blobfile as bf

from paperbench.solvers.basicagent.tools.base import ToolCall


def _json_default(obj: Any) -> Any:
    """
    Best-effort JSON serializer for odd objects.
    We prefer to preserve information rather than fail logging.
    """
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    # Fall back to string representation
    return repr(obj)


def append_jsonl(path: str, row: Mapping[str, Any]) -> None:
    with bf.BlobFile(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    # ToolCall.arguments may have non-string keys; normalize for JSON.
    args: Any = tc.arguments
    if isinstance(args, dict):
        args = {str(k): v for k, v in args.items()}
    return {"id": tc.call_id, "name": tc.name, "args": args}


def log_model_response_event(
    *,
    convo_path: str,
    run_id: str,
    step: int,
    n_input_messages: int,
    response_messages: Sequence[Mapping[str, Any]],
    tool_calls: list[ToolCall],
    usage: dict[str, Any] | None,
    normalized: dict[str, Any],
    ts: float | None = None,
) -> None:
    """
    Append a full, non-truncated model response event to conversation.jsonl.

    NOTE: This intentionally records full payloads (messages / tool args).
    """
    append_jsonl(
        convo_path,
        {
            "ts": time.time() if ts is None else ts,
            "run_id": run_id,
            "step": step,
            "event": "model_response",
            "n_input_messages": n_input_messages,
            "response_messages": list(response_messages),
            "tool_calls": [tool_call_to_dict(tc) for tc in tool_calls],
            "usage": usage,
            "normalized": normalized,
        },
    )


def log_tool_result_event(
    *,
    convo_path: str,
    run_id: str,
    step: int,
    tool_call: ToolCall,
    tool_message: dict[str, Any],
    ts: float | None = None,
) -> None:
    """
    Append a full, non-truncated tool result event to conversation.jsonl.
    """
    append_jsonl(
        convo_path,
        {
            "ts": time.time() if ts is None else ts,
            "run_id": run_id,
            "step": step,
            "event": "tool_result",
            "tool_call": tool_call_to_dict(tool_call),
            "tool_message": tool_message,
        },
    )

