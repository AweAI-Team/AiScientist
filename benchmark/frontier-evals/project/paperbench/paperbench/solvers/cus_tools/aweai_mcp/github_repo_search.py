"""
GitHub repository search tool with blacklist constraint support.
This tool wraps ByteDance internal SearchGithubRepo (bandai_mcp_host).

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


class GithubRepoSearchTool(Tool):
    """
    Search GitHub repositories using ByteDance internal bandai_mcp_host service.

    Supports blacklist constraints to filter out blocked URLs from results.
    """

    max_attempts: int = 1
    _execute_fn: Any = None

    def name(self) -> str:
        return "github_repo_search"

    def _get_execute_fn(self) -> Any:
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools

                self._execute_fn = map_tools("SearchGithubRepo")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for GithubRepoSearchTool."
                ) from e
        return self._execute_fn

    async def execute(
        self,
        computer: ComputerInterface,
        keywords: str,
        in_qualifier: str | None = None,
        user: str | None = None,
        org: str | None = None,
        repo: str | None = None,
        size_limit: str | None = None,
        stars: str | None = None,
        forks: str | None = None,
        followers: str | None = None,
        created: str | None = None,
        pushed: str | None = None,
        language: str | None = None,
        topic: str | None = None,
        topics: str | None = None,
        license: str | None = None,
        is_public: bool = False,
        is_private: bool = False,
        archived: bool = False,
        mirror: bool = False,
        template: bool = False,
        good_first_issues: str | None = None,
        help_wanted_issues: str | None = None,
        sort_by: str | None = None,
        is_ascending: bool = False,
        token: str | None = None,
        max_age: int = -1,
    ) -> str:
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse

        execute_fn = self._get_execute_fn()
        resolved_token = resolve_github_token(token)

        params = {
            "keywords": keywords,
            "in_qualifier": in_qualifier,
            "user": user,
            "org": org,
            "repo": repo,
            "size_limit": size_limit,
            "stars": stars,
            "forks": forks,
            "followers": followers,
            "created": created,
            "pushed": pushed,
            "language": language,
            "topic": topic,
            "topics": topics,
            "license": license,
            "is_public": is_public,
            "is_private": is_private,
            "archived": archived,
            "mirror": mirror,
            "template": template,
            "good_first_issues": good_first_issues,
            "help_wanted_issues": help_wanted_issues,
            "sort_by": sort_by,
            "is_ascending": is_ascending,
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
        return f"Error searching GitHub repos: {error_msg}"

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
                "Search GitHub repositories by keywords and qualifiers. "
                "Supports filtering by stars, language, topics, and more."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords (required)."},
                    "in_qualifier": {
                        "type": "string",
                        "description": "Optional search scope qualifiers (GitHub search syntax).",
                    },
                    "user": {"type": "string", "description": "Optional owner user filter."},
                    "org": {"type": "string", "description": "Optional organization filter."},
                    "repo": {"type": "string", "description": "Optional repo name filter."},
                    "size_limit": {
                        "type": "string",
                        "description": "Optional size filter (e.g., >10, <1000).",
                    },
                    "stars": {
                        "type": "string",
                        "description": "Optional stars filter (e.g., >10, <=500).",
                    },
                    "forks": {
                        "type": "string",
                        "description": "Optional forks filter (e.g., >0, >=50).",
                    },
                    "followers": {
                        "type": "string",
                        "description": "Optional followers filter (e.g., >10).",
                    },
                    "created": {
                        "type": "string",
                        "description": "Optional creation date filter (YYYY-MM-DD or with operator, e.g., >2019-01-01).",
                    },
                    "pushed": {
                        "type": "string",
                        "description": "Optional last push date filter (YYYY-MM-DD or with operator, e.g., >2022-01-01).",
                    },
                    "language": {"type": "string", "description": "Optional programming language."},
                    "topic": {
                        "type": "string",
                        "description": "Optional single topic filter (e.g., machine-learning).",
                    },
                    "topics": {
                        "type": "string",
                        "description": "Optional multiple topics filter (comma-separated).",
                    },
                    "license": {"type": "string", "description": "Optional license filter."},
                    "is_public": {
                        "type": "boolean",
                        "description": "Optional public repos only.",
                        "default": False,
                    },
                    "is_private": {
                        "type": "boolean",
                        "description": "Optional private repos only.",
                        "default": False,
                    },
                    "archived": {"type": "boolean", "description": "Optional archived filter."},
                    "mirror": {"type": "boolean", "description": "Optional mirror filter."},
                    "template": {"type": "boolean", "description": "Optional template filter."},
                    "good_first_issues": {
                        "type": "string",
                        "description": "Optional good first issues filter (e.g., >0).",
                    },
                    "help_wanted_issues": {
                        "type": "string",
                        "description": "Optional help wanted issues filter (e.g., >0).",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Optional sort field (e.g., stars).",
                    },
                    "is_ascending": {
                        "type": "boolean",
                        "description": "Optional ascending sort order.",
                        "default": False,
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
