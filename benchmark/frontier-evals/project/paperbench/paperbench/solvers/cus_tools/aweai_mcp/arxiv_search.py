"""
ArXiv search tool with blacklist constraint support.
This tool wraps ByteDance internal SearchArxiv (bandai_mcp_host).
"""

import json
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.github_utils import filter_blocked_items, parse_json_result


class ArxivSearchTool(Tool):
    """
    Search arXiv papers using bandai_mcp_host.

    Supports blacklist constraints to filter out blocked URLs from results.
    """

    max_attempts: int = 1
    _execute_fn: Any = None

    def name(self) -> str:
        return "arxiv_search"

    def _get_execute_fn(self) -> Any:
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools

                self._execute_fn = map_tools("SearchArxiv")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for ArxivSearchTool."
                ) from e
        return self._execute_fn

    async def execute(
        self,
        computer: ComputerInterface,
        query: str,
        max_results: int | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        submitted_from: str | None = None,
        submitted_to: str | None = None,
        categories: str | None = None,
        max_age: int = -1,
    ) -> str:
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse

        execute_fn = self._get_execute_fn()

        params = {
            "query": query,
            "max_results": max_results,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "submitted_from": submitted_from or "1900-01-01",
            "submitted_to": submitted_to or "2100-12-31",
            "categories": categories,
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
        return f"Error searching arXiv: {error_msg}"

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
            description="Search arXiv papers by query and filters.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (required)."},
                    "max_results": {
                        "type": "integer",
                        "description": "Optional maximum number of results.",
                        "default": 10,
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Optional sort by field (e.g., relevance).",
                        "default": "relevance",
                    },
                    "sort_order": {
                        "type": "string",
                        "description": "Optional sort order (e.g., descending).",
                        "default": "descending",
                    },
                    "submitted_from": {
                        "type": "string",
                        "description": "Optional start date (YYYY-MM-DD). Defaults to 1900-01-01.",
                        "default": "1900-01-01",
                    },
                    "submitted_to": {
                        "type": "string",
                        "description": "Optional end date (YYYY-MM-DD). Defaults to 2100-12-31.",
                        "default": "2100-12-31",
                    },
                    "categories": {
                        "type": "string",
                        "description": "Optional comma-separated categories (e.g., cs.AI,cs.LG,cs.CV).",
                    },
                    "max_age": {
                        "type": "integer",
                        "description": "Cache max age; -1 disables cache.",
                        "default": -1,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
        )
