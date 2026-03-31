from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

from aisci_agent_runtime.tools.base import Tool
from aisci_agent_runtime.tools.constraints import (
    filter_blocked_result_items,
    is_url_blocked,
)

BLOCKED_CONTENT_MARKER = "ACCESS DENIED:"

def _blocked_by_constraints(candidate: str, constraints: dict[str, Any] | None) -> str | None:
    if not constraints:
        return None
    blacklist = constraints.get("blacklist") or constraints.get("blocked_resources") or []
    for blocked in blacklist:
        if not blocked:
            continue
        if str(blocked).lower() in candidate.lower():
            return f"Blocked by blacklist constraint: {blocked}"
    return None


def _fetch_raw(url: str, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AiScientist/1.0",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def _html_to_text(text: str, max_chars: int = 20_000) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:max_chars]


def _fetch_text(url: str, headers: dict[str, str] | None = None, max_chars: int = 20_000) -> str:
    return _html_to_text(_fetch_raw(url, headers=headers), max_chars=max_chars)


def _decode_duckduckgo_href(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params and params["uddg"]:
            return urllib.parse.unquote(params["uddg"][0])
    return href


def _extract_duckduckgo_results(html: str, limit: int) -> list[dict[str, Any]]:
    anchor_pattern = re.compile(
        r'(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    )
    snippet_pattern = re.compile(
        r'(?is)<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>'
    )

    anchors = list(anchor_pattern.finditer(html))
    results: list[dict[str, Any]] = []
    for idx, match in enumerate(anchors[:limit], start=1):
        href = _decode_duckduckgo_href(unescape(match.group(1)))
        title = _html_to_text(match.group(2), max_chars=500)
        chunk_end = anchors[idx].start() if idx < len(anchors) else len(html)
        chunk = html[match.end() : chunk_end]
        snippet_match = snippet_pattern.search(chunk)
        description = _html_to_text(snippet_match.group(1), max_chars=800) if snippet_match else ""
        results.append(
            {
                "position": idx,
                "title": title or "(untitled)",
                "description": description or "",
                "url": href,
            }
        )
    return results


class WebSearchTool(Tool):
    def name(self) -> str:
        return "web_search"

    def supports_constraints(self) -> bool:
        return True

    def execute_with_constraints(self, shell, constraints: dict[str, Any] | None = None, **kwargs) -> str:
        return self.execute(shell, constraints=constraints, **kwargs)

    def execute(
        self,
        shell,  # noqa: ARG002
        query: str | list[str],
        num: int = 10,
        start: int = 0,
        constraints: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        queries = query if isinstance(query, list) else [query]
        outputs: list[str] = []
        per_query = max(1, min(num, 10))
        for single_query in queries:
            blocked = _blocked_by_constraints(single_query, constraints)
            if blocked:
                outputs.append(f"### Search Query: {single_query}\n{blocked}")
                continue
            encoded = urllib.parse.quote_plus(single_query)
            offset = max(start, 0)
            url = f"https://duckduckgo.com/html/?q={encoded}&s={offset}"
            try:
                html = _fetch_raw(url)
            except Exception as exc:  # noqa: BLE001
                outputs.append(f"### Search Query: {single_query}\nError: {exc}")
                continue
            results = _extract_duckduckgo_results(html, per_query)
            blocked_patterns = (constraints or {}).get("blocked_search_patterns", {})
            filtered_results, filtered_count = filter_blocked_result_items(results, blocked_patterns)
            warning = (
                "WARNING: search results referencing blocked resources were filtered out.\n\n"
                if filtered_count > 0
                else ""
            )
            if not filtered_results:
                message = "No search results remain after blacklist filtering." if filtered_count > 0 else "No search results extracted."
                outputs.append(f"### Search Query: {single_query}\n{warning}{message}")
            else:
                outputs.append(
                    f"### Search Query: {single_query}\n"
                    f"{warning}"
                    f"{json.dumps(filtered_results, ensure_ascii=False, indent=2)}"
                )
        return "\n\n".join(outputs)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for documentation, installation guidance, dataset pages, or debugging hints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "description": "The search keyword(s) to look up.",
                            "anyOf": [
                                {"type": "string", "description": "A single search query."},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "A list of search queries.",
                                },
                            ],
                        },
                        "num": {
                            "type": "integer",
                            "description": "Number of results to return. Defaults to 10.",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }


class LinkSummaryTool(Tool):
    def name(self) -> str:
        return "link_summary"

    def supports_constraints(self) -> bool:
        return True

    def execute_with_constraints(self, shell, constraints: dict[str, Any] | None = None, **kwargs) -> str:
        return self.execute(shell, constraints=constraints, **kwargs)

    def execute(
        self,
        shell,  # noqa: ARG002
        url: str,
        goal: str = "",
        constraints: dict[str, Any] | None = None,
        focus: str = "",
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        blocked = _blocked_by_constraints(url, constraints)
        if blocked:
            return f"{BLOCKED_CONTENT_MARKER} {blocked}"
        blocked_patterns = (constraints or {}).get("blocked_search_patterns", {})
        if is_url_blocked(url, blocked_patterns):
            return f"{BLOCKED_CONTENT_MARKER} URL matches blocked_search_patterns."
        try:
            text = _fetch_text(url)
        except Exception as exc:  # noqa: BLE001
            return f"link_summary failed: {exc}"
        instruction = goal or focus
        summary_lines = [f"URL: {url}"]
        if instruction:
            summary_lines.append(f"Goal: {instruction}")
        summary_lines.extend(
            [
                "",
                "Content preview:",
                text[:4_000] or "(empty response)",
            ]
        )
        return "\n".join(summary_lines)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "link_summary",
                "description": "Fetch a URL and return a concise text preview for targeted documentation lookup.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "goal": {"type": "string"},
                    },
                    "required": ["url", "goal"],
                    "additionalProperties": False,
                },
            },
        }


class LinterTool(Tool):
    def name(self) -> str:
        return "linter"

    def execute(
        self,
        shell,
        path: str = "/home/submission",
        command: str = "",
        timeout: int = 120,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        lint_command = command.strip() or f"python -m compileall {path}"
        result = shell.send_shell_command(lint_command, timeout=timeout)
        return result.output.strip() or f"exit_code={result.exit_code}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "linter",
                "description": "Run a lightweight lint or syntax-validation command over the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            },
        }
