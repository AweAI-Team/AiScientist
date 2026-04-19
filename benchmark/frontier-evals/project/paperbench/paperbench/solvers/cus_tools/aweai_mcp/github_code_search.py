"""
GitHub code search tool with blacklist constraint support.
This tool wraps ByteDance internal SearchGithubCode (bandai_mcp_host).

Environment Variables:
    GITHUB_TOKEN: Default GitHub token if not provided in call.
"""

import json
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.github_utils import (
    filter_blocked_items,
    parse_json_result,
    resolve_github_token,
)


class GithubCodeSearchTool(Tool):
    """
    Search GitHub code using bandai_mcp_host.

    Supports blacklist constraints to filter out blocked URLs from results.
    """

    max_attempts: int = 1
    _execute_fn: Any = None

    def name(self) -> str:
        return "github_code_search"

    def _get_execute_fn(self) -> Any:
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools

                self._execute_fn = map_tools("SearchGithubCode")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for GithubCodeSearchTool."
                ) from e
        return self._execute_fn

    async def execute(
        self,
        computer: ComputerInterface,
        keywords: str,
        in_qualifier: str | None = None,
        language: str | None = None,
        repo: str | None = None,
        path: str | None = None,
        filename: str | None = None,
        extension: str | None = None,
        size_limit: str | None = None,
        token: str | None = None,
        max_age: int = -1,
    ) -> str:
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse

        execute_fn = self._get_execute_fn()
        resolved_token = resolve_github_token(token)

        params = {
            "keywords": keywords,
            "in_qualifier": in_qualifier,
            "language": language,
            "repo": repo,
            "path": path,
            "filename": filename,
            "extension": extension,
            "size_limit": size_limit,
            "token": resolved_token,
            "max_age": max_age,
        }
        params = {k: v for k, v in params.items() if v is not None}

        response: BandaiToolResponse | None = None
        for _ in range(self.max_attempts):
            response = await execute_fn(**params)
            if response is not None and response.status.is_succeeded():
                break

        if response is not None and response.status.is_succeeded():
            return response.result
        error_msg = response.result if response is not None else "Unknown error"
        return f"Error searching GitHub code: {error_msg}"

    async def execute_with_constraints(
        self,
        computer: ComputerInterface,
        constraints: dict | None = None,
        **kwargs: Any,
    ) -> str:
        constraints = constraints or {}
        blocked_patterns = constraints.get("blocked_search_patterns", {})
        if not blocked_patterns:
            return await self.execute(computer, **kwargs)

        raw_result = await self.execute(computer, **kwargs)
        parsed = parse_json_result(raw_result)
        if not isinstance(parsed, list):
            return raw_result

        filtered, blocked_count = filter_blocked_items(parsed, blocked_patterns)
        warning = ""
        if blocked_count > 0:
            warning = (
                f"WARNING: {blocked_count} result(s) were filtered out because they "
                f"reference blocked resources.\n\n"
            )
        return warning + json.dumps(filtered, ensure_ascii=False, indent=2)

    def supports_constraints(self) -> bool:
        return True

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Search for code on GitHub by keywords and qualifiers.",
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords (required)."},
                    "in_qualifier": {
                        "type": "string",
                        "description": "Optional search scope (e.g., file).",
                    },
                    "language": {"type": "string", "description": "Optional programming language."},
                    "repo": {"type": "string", "description": "Optional repo filter."},
                    "path": {"type": "string", "description": "Optional path filter."},
                    "filename": {"type": "string", "description": "Optional filename filter."},
                    "extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., py).",
                    },
                    "size_limit": {
                        "type": "string",
                        "description": "Optional file size filter (e.g., >100, <=5000).",
                    },
                    "max_age": {
                        "type": "integer",
                        "description": "Cache max age; -1 disables cache.",
                        "default": -1,
                    },
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
            strict=False,
        )
