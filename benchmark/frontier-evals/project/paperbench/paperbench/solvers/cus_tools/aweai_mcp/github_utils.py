import json
import os
import random
from typing import Any

from paperbench.solvers.cus_tools.aweai_mcp.utils import iter_url_like_values, is_url_blocked

ENV_GITHUB_TOKEN = "GITHUB_TOKEN"
GITHUB_TOKEN_POOL = [
    "ADD_TOKEN_HERE"
]


def resolve_github_token(token: str | None) -> str:
    """
    Resolve GitHub token from input, pool, or environment.
    """
    if token:
        return token
    if GITHUB_TOKEN_POOL:
        return random.choice(GITHUB_TOKEN_POOL)
    env_token = os.environ.get(ENV_GITHUB_TOKEN)
    if not env_token:
        raise ValueError(
            f"GitHub token is required. Provide `token` or set {ENV_GITHUB_TOKEN}."
        )
    return env_token


def parse_json_result(result: str) -> Any:
    """
    Best-effort JSON parsing for tool results.
    """
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return result


def is_item_blocked(item: Any, blocked_patterns: dict[str, list[str]]) -> bool:
    """
    Check if any URL-like field in item matches blocked patterns.
    """
    for url in iter_url_like_values(item):
        if is_url_blocked(url, blocked_patterns):
            return True
    return False


def filter_blocked_items(
    items: list[Any], blocked_patterns: dict[str, list[str]]
) -> tuple[list[Any], int]:
    """
    Filter out items referencing blocked URLs.
    """
    if not blocked_patterns:
        return items, 0

    filtered: list[Any] = []
    blocked_count = 0
    for item in items:
        if isinstance(item, dict) and is_item_blocked(item, blocked_patterns):
            blocked_count += 1
        else:
            filtered.append(item)
    return filtered, blocked_count
