"""
GitHub file reader tool with blacklist constraint support.
This tool wraps ByteDance internal ReadGithubFile (bandai_mcp_host).

Environment Variables:
    GITHUB_TOKEN: Default GitHub token if not provided in call.
"""

from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.github_utils import resolve_github_token
from paperbench.solvers.cus_tools.aweai_mcp.utils import is_url_blocked


class GithubFileReaderTool(Tool):
    """
    Read GitHub file contents using bandai_mcp_host.

    Supports blacklist constraints to block access to specific URLs.
    """

    max_attempts: int = 1
    _execute_fn: Any = None

    def name(self) -> str:
        return "github_file_reader"

    def _get_execute_fn(self) -> Any:
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools

                self._execute_fn = map_tools("ReadGithubFile")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for GithubFileReaderTool."
                ) from e
        return self._execute_fn

    async def execute(
        self,
        computer: ComputerInterface,
        url: str,
        token: str | None = None,
        max_age: int = -1,
    ) -> str:
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse

        execute_fn = self._get_execute_fn()
        resolved_token = resolve_github_token(token)

        params = {
            "url": url,
            "token": resolved_token,
            "max_age": max_age,
        }

        response: BandaiToolResponse | None = None
        for _ in range(self.max_attempts):
            response = await execute_fn(**params)
            if response is not None and response.status.is_succeeded():
                break

        if response is not None and response.status.is_succeeded():
            return response.result
        error_msg = response.result if response is not None else "Unknown error"
        return f"Error reading GitHub file: {error_msg}"

    async def execute_with_constraints(
        self,
        computer: ComputerInterface,
        constraints: dict | None = None,
        url: str = "",
        **kwargs: Any,
    ) -> str:
        constraints = constraints or {}
        blocked_patterns = constraints.get("blocked_search_patterns", {})
        if blocked_patterns and is_url_blocked(url, blocked_patterns):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked because it references a "
                f"blacklisted resource."
            )
        return await self.execute(computer, url=url, **kwargs)

    def supports_constraints(self) -> bool:
        return True

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Read raw file content from a GitHub file URL.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "GitHub file URL (required, e.g., https://github.com/org/repo/blob/branch/path).",
                    },
                    "max_age": {
                        "type": "integer",
                        "description": "Cache max age; -1 disables cache.",
                        "default": -1,
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            strict=False,
        )
