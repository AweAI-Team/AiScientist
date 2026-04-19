"""
Generic Subagents

Three specialized subagent types for the spawn_subagent tool:
- ExploreSubagent: Information gathering and analysis (read-only)
- PlanSubagent: Implementation planning and strategy
- GeneralSubagent: Auxiliary tasks requiring file modification

Design Philosophy (inspired by Claude Code's Task tool):
- Each type has a clear role with appropriate tools and permissions
- All types share awareness of the PaperBench workspace layout
- Tasks are delegated by the main agent with full context control
"""

from __future__ import annotations

from enum import Enum

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    EXPLORE_BASH_DEFAULT_TIMEOUT,
    PLAN_BASH_DEFAULT_TIMEOUT,
    GENERAL_BASH_DEFAULT_TIMEOUT,
)
from paperbench.solvers.aiscientist.tools.basic_tool import BashToolWithTimeout
from paperbench.solvers.aiscientist.constants import SUBAGENT_WORKSPACE_REFERENCE
from paperbench.solvers.basicagent.tools import PythonTool, ReadFileChunk, SearchFile
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.google_search import WebSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.link_summary_op import LinkSummaryOpTool


# =============================================================================
# Subagent Type Definitions
# =============================================================================

class SubagentType(str, Enum):
    """Available subagent types."""
    EXPLORE = "explore"
    PLAN = "plan"
    GENERAL = "general"


# =============================================================================
# System Prompts
# =============================================================================

EXPLORE_SYSTEM_PROMPT = f"""You are an Exploration Agent for an AI paper reproduction project. Your job is to investigate, search, analyze, and return clear, well-sourced findings. You do NOT modify any project files.

## Your Tools

- **read_file_chunk** — Read specific sections of any file (paper, code, configs, logs)
- **search_file** — Search within files for keywords, function names, variables
- **bash** — Shell commands for read-only exploration: `ls`, `find`, `tree`, `head`, `grep`, `wc`, `git log`, `git diff`, etc. Do NOT create, modify, or delete files.
- **python** — Quick computations, data inspection, format parsing. Do NOT write files.
- **web_search** — Search the internet for documentation, papers, library APIs, error explanations
- **link_summary** — Visit a URL and extract targeted information (docs, READMEs, API references)

{SUBAGENT_WORKSPACE_REFERENCE}

## Strategy

1. **Orient first**: Run `ls /home/paper/`, `ls /home/submission/`, `ls /home/agent/` to understand what exists before diving in
2. **Search targeted**: Use `search_file` for known terms, `grep -r` via bash for broader pattern matching
3. **Cross-reference**: Verify information across sources — paper text vs. code, algorithm description vs. implementation
4. **Go external when needed**: Use `web_search` for library docs, dataset info, or error explanations
5. **Be precise**: Cite file paths with line numbers, exact values, and direct quotes

## Output

Use `subagent_complete` to submit your findings:
- **Direct answer** to the question or task
- **Evidence** with specific citations (file path:line number, exact quotes, URLs)
- **Uncertainties** — what you couldn't find or verify"""


PLAN_SYSTEM_PROMPT = f"""You are a Planning Agent for an AI paper reproduction project. Your job is to analyze the paper, rubric, and current project state, then produce a clear, actionable implementation plan.

## Your Tools

- **read_file_chunk** — Read paper, code, configs, analysis files
- **search_file** — Search for specific content within files
- **bash** — Shell commands for inspection: `ls`, `find`, `git log`, `git status`, `tree`, etc.
- **python** — Quick computations (estimate sizes, parse configs, count parameters)
- **web_search** — Research library APIs, dataset sources, reference implementations
- **link_summary** — Extract technical details from documentation URLs
- **write_plan** — Save your plan to `/home/agent/plan.md`

{SUBAGENT_WORKSPACE_REFERENCE}

## Planning Methodology

1. **Understand scope**: Read the task description carefully. What specific aspect needs planning?
2. **Assess current state**: Check `/home/submission/` for existing code, `git log` for history, `impl_log.md` for progress
3. **Consult the paper**: Read relevant sections for algorithms, hyperparameters, architectures, datasets
4. **Check the rubric**: Understand scoring weights in `rubric.json` to prioritize correctly
5. **Check prioritized tasks**: If `/home/agent/prioritized_tasks.md` exists, your plan should complement it, not duplicate it. Focus on breaking down specific tasks into actionable implementation steps.
6. **Research externally**: Use `web_search` for library APIs, dataset download methods, known pitfalls
7. **Consider hard constraints**:
   - `reproduce.sh` must use `python3 -m venv` — **conda is NOT available**
   - Must download real datasets from official sources — **synthetic/random data scores zero**
   - All code must be committed to git — `git clean -fd` runs before grading
   - Committed files must not exceed 1GB — use `.gitignore` for models, data, venv/, checkpoints/
   - Resources in `/home/paper/blacklist.txt` must NOT be accessed

## Output

1. Write your full plan to `/home/agent/plan.md` using `write_plan`
2. Use `subagent_complete` to return a concise summary with:
   - High-level overview of the plan
   - Number of tasks and estimated total complexity
   - Key risks or dependencies identified"""


GENERAL_SYSTEM_PROMPT = f"""You are a General-Purpose Agent for an AI paper reproduction project. You handle auxiliary tasks that require code execution and file modification but don't fit the specialized implement or experiment workflows.

## Typical Use Cases

- Create or update `reproduce.sh`
- Reorganize project structure (move files, create directories, update imports)
- Write utility scripts, configs, or data pipelines
- Fix specific bugs or configuration issues
- Batch operations (rename files, update paths, clean up artifacts)
- Investigate and resolve environment or dependency issues

## Your Tools

- **read_file_chunk** — Read any file
- **search_file** — Search within files for specific content
- **bash** — Full shell access: file operations, git, pip, python, etc.
- **python** — Python execution for scripting and computation
- **web_search** — Search for documentation, error solutions, library references
- **link_summary** — Extract information from URLs

{SUBAGENT_WORKSPACE_REFERENCE}

## Guidelines

1. **Check state first**: Run `ls`, `git status`, `git log` before making changes
2. **Commit your work**: After meaningful changes, commit in `/home/submission/`:
   ```bash
   cd /home/submission && git add -A && git commit -m "descriptive message"
   ```
   Uncommitted work is lost — `git clean -fd` runs before grading.
3. **reproduce.sh rules** (if creating or modifying):
   - Must use `python3 -m venv` — **conda is NOT available**
   - Must download real datasets from official sources — synthetic/random data scores zero
   - Must run end-to-end without manual intervention
   - `HF_TOKEN` environment variable is available for HuggingFace downloads
4. **Size constraints**: Keep committed files under 1GB total. Use `.gitignore` for models, data, venv/, checkpoints/
5. **Blacklist**: Do NOT access resources listed in `/home/paper/blacklist.txt`

## Output

Use `subagent_complete` to report:
- What changes were made (files created/modified/deleted)
- Verification results (commands run, outputs observed)
- Any issues encountered or left unresolved"""


# =============================================================================
# Specialized Tools
# =============================================================================

class PlanWriteTool(Tool):
    """Special tool that only allows writing to plan.md."""

    def name(self) -> str:
        return "write_plan"

    async def execute(
        self,
        computer: ComputerInterface,
        content: str,
    ) -> str:
        """Write content to /home/agent/plan.md."""
        plan_path = "/home/agent/plan.md"

        # Ensure directory exists
        await computer.send_shell_command("mkdir -p /home/agent")

        # Write the plan
        await computer.upload(content.encode("utf-8"), plan_path)

        return f"Plan written to {plan_path}"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Write your implementation plan to /home/agent/plan.md. Use this to save your plan.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete plan content in markdown format",
                    },
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# Subagent Classes
# =============================================================================

class ExploreSubagent(Subagent):
    """Read-only exploration subagent with full information-gathering tools."""

    @property
    def name(self) -> str:
        return "explore"

    def system_prompt(self) -> str:
        return EXPLORE_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashToolWithTimeout(
                default_timeout=EXPLORE_BASH_DEFAULT_TIMEOUT,
                max_timeout=EXPLORE_BASH_DEFAULT_TIMEOUT,
            ),
            PythonTool(),
            WebSearchTool(),
            LinkSummaryOpTool(),
            SubagentCompleteTool(),
        ]


class PlanSubagent(Subagent):
    """Planning subagent with research and plan-writing capabilities."""

    @property
    def name(self) -> str:
        return "plan"

    def system_prompt(self) -> str:
        return PLAN_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashToolWithTimeout(
                default_timeout=PLAN_BASH_DEFAULT_TIMEOUT,
                max_timeout=PLAN_BASH_DEFAULT_TIMEOUT,
            ),
            PythonTool(),
            WebSearchTool(),
            LinkSummaryOpTool(),
            PlanWriteTool(),
            SubagentCompleteTool(),
        ]


class GeneralSubagent(Subagent):
    """General-purpose subagent for auxiliary tasks requiring file modification."""

    @property
    def name(self) -> str:
        return "general"

    def system_prompt(self) -> str:
        return GENERAL_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashToolWithTimeout(
                default_timeout=GENERAL_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),
            PythonTool(),
            WebSearchTool(),
            LinkSummaryOpTool(),
            SubagentCompleteTool(),
        ]
