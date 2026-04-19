"""
Search Paper Tool

An intelligent search coordinator that autonomously decides:
1. How many parallel search subagents to spawn (1-3)
2. Which files/sections to search

This complements the fixed read_paper workflow by allowing dynamic,
targeted searches during implementation.

Design Philosophy:
- Main agent provides a search task
- This tool analyzes the task and creates an optimal search strategy
- Multiple subagents can search in parallel for efficiency
- Results are synthesized and returned to main agent

Note: Subagent implementations are in subagents/search.py
"""

from __future__ import annotations

import asyncio

import structlog
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents import (
    SubagentOutput,
    SubagentStatus,
    SearchStrategistSubagent,
    SearchExecutorSubagent,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_SEARCH_EXECUTOR_CONFIG,
    DEFAULT_SEARCH_SIMPLE_CONFIG,
    DEFAULT_SEARCH_STRATEGY_CONFIG,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


class SearchPaperTool(Tool):
    """
    Intelligent search coordinator for paper-related queries.

    This tool:
    1. Analyzes the search request
    2. Creates an optimal search strategy
    3. Spawns parallel search subagents (1-3)
    4. Synthesizes results

    Use this for dynamic, targeted searches during implementation.

    Note: Subagent implementations are defined in subagents/search.py
    """

    # Set by solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    run_dir: str | None = None
    max_parallel_searches: int = 3

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "search_paper"

    async def execute(
        self,
        computer: ComputerInterface,
        query: str,
        search_strategy: str = "auto",
    ) -> str:
        """
        Execute an intelligent paper search.

        Args:
            computer: ComputerInterface for execution
            query: What to search for (natural language)
            search_strategy: "auto" (let tool decide) or "simple" (single search)

        Returns:
            Synthesized search results
        """
        ctx_logger = logger.bind(tool="search_paper")

        if self.completer_config is None:
            return "Error: SearchPaperTool not properly configured."

        ctx_logger.info(f"Starting paper search: {query[:50]}...")

        try:
            if search_strategy == "simple":
                # Simple mode: just one search
                return await self._simple_search(computer, query)
            else:
                # Auto mode: create strategy and execute
                return await self._intelligent_search(computer, query)

        except Exception as e:
            ctx_logger.error(f"Search failed: {e}")
            return f"Error during search: {str(e)}"

    async def _simple_search(
        self,
        computer: ComputerInterface,
        query: str,
    ) -> str:
        """Execute a simple single-subagent search."""
        config = DEFAULT_SEARCH_SIMPLE_CONFIG

        executor = SearchExecutorSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
            search_id=0,
        )

        task = f"""Search for: {query}

Files to search:
- /home/paper/paper.md (main paper)
- /home/paper/addendum.md (additional info)

Find the relevant information and provide clear findings with citations."""

        result = await executor.run(
            computer=computer,
            task_description=task,
            constraints=self.constraints,
        )

        return self._format_single_result(result)

    async def _intelligent_search(
        self,
        computer: ComputerInterface,
        query: str,
    ) -> str:
        """Execute intelligent search with strategy planning."""
        ctx_logger = logger.bind(tool="search_paper", mode="intelligent")

        # Step 1: Create search strategy
        strategy = await self._create_search_strategy(computer, query)

        if strategy is None:
            # Fallback to simple search
            ctx_logger.warning("Strategy creation failed, falling back to simple search")
            return await self._simple_search(computer, query)

        # Step 2: Parse strategy and create search tasks
        search_tasks = self._parse_strategy(strategy, query)

        if not search_tasks:
            ctx_logger.warning("No search tasks created, falling back to simple search")
            return await self._simple_search(computer, query)

        # Limit to max parallel searches
        search_tasks = search_tasks[:self.max_parallel_searches]
        ctx_logger.info(f"Executing {len(search_tasks)} parallel searches")

        # Step 3: Execute searches in parallel
        results = await self._execute_parallel_searches(computer, search_tasks)

        # Step 4: Synthesize results
        return self._synthesize_results(query, results, len(search_tasks))

    async def _create_search_strategy(
        self,
        computer: ComputerInterface,
        query: str,
    ) -> str | None:
        """Create a search strategy using the strategist subagent."""
        config = DEFAULT_SEARCH_STRATEGY_CONFIG

        strategist = SearchStrategistSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
        )

        task = f"""Create a search strategy for: {query}

Decide how many parallel searches (1-3) would be most effective and what each should focus on."""

        result = await strategist.run(
            computer=computer,
            task_description=task,
            constraints=self.constraints,
        )

        if result.status == SubagentStatus.COMPLETED:
            return result.content
        return None

    def _parse_strategy(self, strategy: str, original_query: str) -> list[dict]:
        """Parse the strategy output into search tasks."""
        import json
        import re

        # Try to extract JSON from the strategy
        json_match = re.search(r'\{[\s\S]*"searches"[\s\S]*\}', strategy)

        if json_match:
            try:
                data = json.loads(json_match.group())
                searches = data.get("searches", [])

                tasks = []
                for i, search in enumerate(searches):
                    task = {
                        "id": i,
                        "focus": search.get("focus", f"Search {i+1}"),
                        "files": search.get("files", ["/home/paper/paper.md"]),
                        "keywords": search.get("keywords", []),
                        "sections": search.get("sections", ""),
                    }
                    tasks.append(task)

                return tasks if tasks else [{"id": 0, "focus": original_query, "files": ["/home/paper/paper.md"], "keywords": [], "sections": ""}]

            except json.JSONDecodeError:
                pass

        # Fallback: single search with original query
        return [{"id": 0, "focus": original_query, "files": ["/home/paper/paper.md"], "keywords": [], "sections": ""}]

    async def _execute_parallel_searches(
        self,
        computer: ComputerInterface,
        search_tasks: list[dict],
    ) -> list[SubagentOutput]:
        """Execute multiple searches in parallel."""
        config = DEFAULT_SEARCH_EXECUTOR_CONFIG

        async def run_single_search(task: dict) -> SubagentOutput:
            executor = SearchExecutorSubagent(
                completer_config=self.completer_config,
                config=config,
                run_dir=self.run_dir,
                search_id=task["id"],
            )

            files_str = ", ".join(task["files"])
            keywords_str = ", ".join(task["keywords"]) if task["keywords"] else "relevant terms"

            task_desc = f"""Search Focus: {task["focus"]}

Files to search: {files_str}
Keywords to look for: {keywords_str}
{f"Focus on sections: {task['sections']}" if task['sections'] else ""}

Find the relevant information and provide clear findings with citations."""

            return await executor.run(
                computer=computer,
                task_description=task_desc,
                constraints=self.constraints,
            )

        # Run all searches in parallel
        results = await asyncio.gather(
            *[run_single_search(task) for task in search_tasks],
            return_exceptions=True,
        )

        # Convert exceptions to failed outputs
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SubagentOutput(
                    subagent_name=f"search_executor_{i}",
                    status=SubagentStatus.FAILED,
                    content="",
                    error_message=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    def _format_single_result(self, result: SubagentOutput) -> str:
        """Format a single search result."""
        status_icon = "✓" if result.status == SubagentStatus.COMPLETED else "✗"

        header = f"[Search {status_icon}] ({result.num_steps} steps, {result.runtime_seconds:.1f}s)"

        if result.status == SubagentStatus.COMPLETED:
            return f"{header}\n\n{result.content}"
        else:
            return f"{header}\n\nSearch failed: {result.error_message or 'Unknown error'}"

    def _synthesize_results(
        self,
        query: str,
        results: list[SubagentOutput],
        num_searches: int,
    ) -> str:
        """Synthesize results from multiple parallel searches."""
        lines = [
            f"# Search Results for: {query}",
            "",
            f"**Parallel Searches**: {num_searches}",
            "",
        ]

        successful = sum(1 for r in results if r.status == SubagentStatus.COMPLETED)
        lines.append(f"**Successful**: {successful}/{num_searches}")
        lines.append("")
        lines.append("---")

        for i, result in enumerate(results):
            status_icon = "✓" if result.status == SubagentStatus.COMPLETED else "✗"
            lines.append(f"\n## Search {i+1} [{status_icon}]")
            lines.append(f"*({result.num_steps} steps, {result.runtime_seconds:.1f}s)*\n")

            if result.status == SubagentStatus.COMPLETED:
                lines.append(result.content)
            else:
                lines.append(f"Failed: {result.error_message or 'Unknown error'}")

            lines.append("")

        return "\n".join(lines)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Intelligent paper search that autonomously decides the search strategy.

This tool:
1. Analyzes your query
2. Decides how many parallel searches to run (1-3)
3. Determines what each search should focus on
4. Synthesizes results from all searches

Use this for dynamic, targeted searches during implementation when you need
specific information not covered by the initial paper analysis.

## Examples
- search_paper(query="What are all the batch sizes used in experiments?")
- search_paper(query="How is numerical stability handled in the algorithm?")
- search_paper(query="What datasets are used and where can they be found?")

## Modes
- search_strategy="auto" (default): Intelligent multi-search
- search_strategy="simple": Quick single search for simple queries
""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (natural language description)",
                    },
                    "search_strategy": {
                        "type": "string",
                        "enum": ["auto", "simple"],
                        "description": "Search mode: 'auto' for intelligent multi-search, 'simple' for quick single search",
                        "default": "auto",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
        )
