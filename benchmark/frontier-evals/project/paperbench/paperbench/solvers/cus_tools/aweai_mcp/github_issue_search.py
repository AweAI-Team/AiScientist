"""
GitHub issue search tool with blacklist constraint support.
This tool wraps ByteDance internal SearchGithubIssue (bandai_mcp_host).

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


class GithubIssueSearchTool(Tool):
    """
    Search GitHub issues and pull requests using bandai_mcp_host.

    Supports blacklist constraints to filter out blocked URLs from results.
    """

    max_attempts: int = 1
    _execute_fn: Any = None

    def name(self) -> str:
        return "github_issue_search"

    def _get_execute_fn(self) -> Any:
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools

                self._execute_fn = map_tools("SearchGithubIssue")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for GithubIssueSearchTool."
                ) from e
        return self._execute_fn

    async def execute(
        self,
        computer: ComputerInterface,
        keywords: str,
        type_qualifier: str | None = "issue",
        state: str | None = None,
        in_qualifier: str | None = None,
        author: str | None = None,
        assignee: str | None = None,
        mentions: str | None = None,
        label: str | None = None,
        repo: str | None = None,
        language: str | None = None,
        created_after: str | None = None,
        updated_after: str | None = None,
        comments_count: str | None = None,
        sort_by: str | None = None,
        is_ascending: bool = False,
        is_draft: bool | None = None,
        review_status: str | None = None,
        is_merged: bool | None = None,
        no_assignee: bool = False,
        no_label: bool = False,
        is_archived: bool | None = None,
        is_locked: bool | None = None,
        token: str | None = None,
        max_age: int = -1,
    ) -> str:
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse

        execute_fn = self._get_execute_fn()
        resolved_token = resolve_github_token(token)

        params = {
            "keywords": keywords,
            "type_qualifier": type_qualifier,
            "state": state,
            "in_qualifier": in_qualifier,
            "author": author,
            "assignee": assignee,
            "mentions": mentions,
            "label": label,
            "repo": repo,
            "language": language,
            "created_after": created_after,
            "updated_after": updated_after,
            "comments_count": comments_count,
            "sort_by": sort_by,
            "is_ascending": is_ascending,
            "is_draft": is_draft,
            "review_status": review_status,
            "is_merged": is_merged,
            "no_assignee": no_assignee,
            "no_label": no_label,
            "is_archived": is_archived,
            "is_locked": is_locked,
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
        return f"Error searching GitHub issues: {error_msg}"

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
            description=(
                "Search GitHub issues and pull requests by keywords and qualifiers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords (required)."},
                    "type_qualifier": {
                        "type": "string",
                        "description": "Optional issue or pr qualifier (issue|pr). Defaults to issue.",
                        "default": "issue",
                    },
                    "state": {
                        "type": "string",
                        "description": "Optional state filter (open|closed).",
                    },
                    "in_qualifier": {
                        "type": "string",
                        "description": "Optional search scope qualifiers (e.g., title,body).",
                    },
                    "author": {"type": "string", "description": "Optional author filter."},
                    "assignee": {"type": "string", "description": "Optional assignee filter."},
                    "mentions": {"type": "string", "description": "Optional mentions filter."},
                    "label": {"type": "string", "description": "Optional label filter."},
                    "repo": {"type": "string", "description": "Optional repo filter."},
                    "language": {"type": "string", "description": "Optional language filter."},
                    "created_after": {
                        "type": "string",
                        "description": "Optional created-after date (YYYY-MM-DD, e.g., 2023-01-01).",
                    },
                    "updated_after": {
                        "type": "string",
                        "description": "Optional updated-after date (YYYY-MM-DD, e.g., 2023-01-01).",
                    },
                    "comments_count": {
                        "type": "string",
                        "description": "Optional comments count filter (e.g., >0, >=10).",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Optional sort field (e.g., created).",
                    },
                    "is_ascending": {
                        "type": "boolean",
                        "description": "Optional ascending order.",
                        "default": False,
                    },
                    "is_draft": {"type": "boolean", "description": "Optional draft PR filter."},
                    "review_status": {"type": "string", "description": "Optional review status."},
                    "is_merged": {"type": "boolean", "description": "Optional merged PR filter."},
                    "no_assignee": {
                        "type": "boolean",
                        "description": "Optional no-assignee filter.",
                        "default": False,
                    },
                    "no_label": {
                        "type": "boolean",
                        "description": "Optional no-label filter.",
                        "default": False,
                    },
                    "is_archived": {"type": "boolean", "description": "Optional archived repo filter."},
                    "is_locked": {"type": "boolean", "description": "Optional locked issue filter."},
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
