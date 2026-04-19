"""
Unified GitHub Tool

Wraps three GitHub tools (repo search, code search, file reader) into a single
tool with a ``mode`` parameter.  This reduces the tool count seen by the LLM
while keeping the same functionality.

Modes:
    - ``repo``  (default): Search GitHub repositories.
    - ``code``:  Search GitHub code snippets.
    - ``file``:  Read a specific file from a GitHub URL.
"""

from __future__ import annotations

import structlog
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.github_repo_search import GithubRepoSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.github_code_search import GithubCodeSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.github_file_reader import GithubFileReaderTool

logger = structlog.stdlib.get_logger(component=__name__)


class GithubTool(Tool):
    """
    Unified GitHub tool that delegates to repo search, code search, or file
    reader based on the ``mode`` parameter.
    """

    _repo_tool: GithubRepoSearchTool | None = None
    _code_tool: GithubCodeSearchTool | None = None
    _file_tool: GithubFileReaderTool | None = None

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "github"

    def _get_repo_tool(self) -> GithubRepoSearchTool:
        if self._repo_tool is None:
            self._repo_tool = GithubRepoSearchTool()
        return self._repo_tool

    def _get_code_tool(self) -> GithubCodeSearchTool:
        if self._code_tool is None:
            self._code_tool = GithubCodeSearchTool()
        return self._code_tool

    def _get_file_tool(self) -> GithubFileReaderTool:
        if self._file_tool is None:
            self._file_tool = GithubFileReaderTool()
        return self._file_tool

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        computer: ComputerInterface,
        mode: str = "repo",
        keywords: str = "",
        url: str = "",
        language: str | None = None,
        repo: str | None = None,
        path: str | None = None,
        stars: str | None = None,
        extension: str | None = None,
    ) -> str:
        """
        Execute a GitHub operation.

        Args:
            mode: One of "repo", "code", "file".
            keywords: Search keywords (required for repo / code mode).
            url: GitHub file URL (required for file mode).
            language: Programming language filter (repo / code mode).
            repo: Repository filter, e.g. "owner/repo" (code mode).
            path: Path filter inside a repo (code mode).
            stars: Stars filter, e.g. ">100" (repo mode).
            extension: File extension filter, e.g. "py" (code mode).
        """
        mode = mode.strip().lower()

        if mode == "repo":
            if not keywords:
                return "Error: 'keywords' is required for repo search mode."
            return await self._get_repo_tool().execute(
                computer,
                keywords=keywords,
                language=language,
                stars=stars,
            )

        elif mode == "code":
            if not keywords:
                return "Error: 'keywords' is required for code search mode."
            return await self._get_code_tool().execute(
                computer,
                keywords=keywords,
                language=language,
                repo=repo,
                path=path,
                extension=extension,
            )

        elif mode == "file":
            if not url:
                return "Error: 'url' is required for file read mode."
            return await self._get_file_tool().execute(
                computer,
                url=url,
            )

        else:
            return f"Error: Unknown mode '{mode}'. Use 'repo', 'code', or 'file'."

    # ------------------------------------------------------------------
    # constraint-aware execution
    # ------------------------------------------------------------------

    async def execute_with_constraints(
        self,
        computer: ComputerInterface,
        constraints: dict | None = None,
        **kwargs: Any,
    ) -> str:
        mode = kwargs.get("mode", "repo").strip().lower()
        constraints = constraints or {}

        if mode == "repo":
            kw = {
                "keywords": kwargs.get("keywords", ""),
                "language": kwargs.get("language"),
                "stars": kwargs.get("stars"),
            }
            return await self._get_repo_tool().execute_with_constraints(
                computer, constraints=constraints, **kw
            )

        elif mode == "code":
            kw = {
                "keywords": kwargs.get("keywords", ""),
                "language": kwargs.get("language"),
                "repo": kwargs.get("repo"),
                "path": kwargs.get("path"),
                "extension": kwargs.get("extension"),
            }
            return await self._get_code_tool().execute_with_constraints(
                computer, constraints=constraints, **kw
            )

        elif mode == "file":
            kw = {"url": kwargs.get("url", "")}
            return await self._get_file_tool().execute_with_constraints(
                computer, constraints=constraints, **kw
            )

        else:
            return f"Error: Unknown mode '{mode}'. Use 'repo', 'code', or 'file'."

    def supports_constraints(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # tool schema
    # ------------------------------------------------------------------

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Search GitHub for repositories, code, or read file contents.

Modes:
- **repo** (default): Search repositories. Example: github(keywords="adaptive pruning transformers", stars=">10")
- **code**: Search code snippets. Example: github(mode="code", keywords="class PruningScheduler", language="python")
- **file**: Read a file from a GitHub URL. Example: github(mode="file", url="https://github.com/owner/repo/blob/main/model.py")

Use this to find reference implementations, libraries, or code patterns when implementing papers.""",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["repo", "code", "file"],
                        "description": "Search mode: 'repo' (search repositories), 'code' (search code), 'file' (read a file by URL).",
                        "default": "repo",
                    },
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords (required for 'repo' and 'code' modes).",
                    },
                    "url": {
                        "type": "string",
                        "description": "GitHub file URL (required for 'file' mode, e.g. https://github.com/org/repo/blob/main/path/to/file.py).",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language filter (for 'repo' and 'code' modes).",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository filter as 'owner/repo' (for 'code' mode).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path filter inside a repo (for 'code' mode).",
                    },
                    "stars": {
                        "type": "string",
                        "description": "Stars filter, e.g. '>100' (for 'repo' mode).",
                    },
                    "extension": {
                        "type": "string",
                        "description": "File extension filter, e.g. 'py' (for 'code' mode).",
                    },
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
            strict=False,
        )
