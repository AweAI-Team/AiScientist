"""
Search Subagents

This module contains subagents specialized for intelligent paper searching:
- SearchStrategistSubagent: Creates optimal search strategies
- SearchExecutorSubagent: Executes individual search tasks

Design Philosophy:
- Strategist analyzes queries and creates parallel search plans
- Executors perform focused searches based on the strategy
- Results are synthesized by the SearchPaperTool
"""

from __future__ import annotations

from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
    SubagentConfig,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools import ReadFileChunk, SearchFile
from paperbench.solvers.basicagent.tools.base import Tool

# =============================================================================
# System Prompts
# =============================================================================

SEARCH_STRATEGIST_PROMPT = """You are a Search Strategist. Your job is to analyze a search request and create an optimal search plan.

## Your Task
Given a search query, determine:
1. How many parallel searches to run (1-3)
2. What each search should focus on
3. Which files/sections each search should target

## Available Files
- /home/paper/paper.md - Main paper content
- /home/paper/addendum.md - Additional instructions
- /home/paper/blacklist.txt - Forbidden resources

## Output Format
You MUST output a JSON search plan using subagent_complete:

```json
{
  "searches": [
    {
      "focus": "Brief description of what this search looks for",
      "files": ["/home/paper/paper.md"],
      "keywords": ["keyword1", "keyword2"],
      "sections": "Optional: specific sections to focus on"
    }
  ],
  "reasoning": "Why this search strategy"
}
```

## Guidelines
- Use 1 search for simple queries
- Use 2-3 searches for complex queries that benefit from parallel exploration
- Each search should have a distinct focus
- Consider both main paper and appendix for technical details
"""

SEARCH_EXECUTOR_PROMPT = """You are a Search Executor. Your job is to find specific information in documents.

## Your Task
Search for the requested information in the specified files.

## Guidelines
1. Use search_file to find relevant sections
2. Use read_file_chunk to read context around matches
3. Extract the specific information requested
4. Cite sources with file paths and line numbers

## Output
Provide your findings in clear markdown format with:
- The information found
- Source citations (file:line)
- Relevant quotes if applicable
- Note if information was not found
"""


# =============================================================================
# Subagent Classes
# =============================================================================

class SearchStrategistSubagent(Subagent):
    """Subagent that creates search strategies."""

    @property
    def name(self) -> str:
        return "search_strategist"

    def system_prompt(self) -> str:
        return SEARCH_STRATEGIST_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),  # To peek at paper structure
            SubagentCompleteTool(),
        ]


class SearchExecutorSubagent(Subagent):
    """Subagent that executes a specific search task."""

    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,
        config: SubagentConfig | None = None,
        run_dir: str | None = None,
        search_id: int = 0,
    ):
        super().__init__(completer_config, config, run_dir)
        self._search_id = search_id

    @property
    def name(self) -> str:
        return f"search_executor_{self._search_id}"

    def system_prompt(self) -> str:
        return SEARCH_EXECUTOR_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            SearchFile(),
            ReadFileChunk(),
            SubagentCompleteTool(),
        ]
