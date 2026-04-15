"""
Generic Subagents — exploratory and planning helpers.

Three specialized subagent types for the spawn_subagent tool:
- ExploreSubagent: Information gathering and analysis (read-only)
- PlanSubagent: Implementation planning and strategy
- GeneralSubagent: Auxiliary tasks requiring file modification

Adapted from paper-reproduction style tasks to Kaggle competitions.
"""

from __future__ import annotations

from enum import Enum

from subagents.base import Subagent, SubagentConfig
from subagents.configs import (
    EXPLORE_BASH_DEFAULT_TIMEOUT,
    PLAN_BASH_DEFAULT_TIMEOUT,
    GENERAL_BASH_DEFAULT_TIMEOUT,
)
from tools.base import Tool, SubagentCompleteTool
from tools.shell_tools import (
    BashToolWithTimeout,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
)
from constants import SUBAGENT_WORKSPACE_REFERENCE


class SubagentType(str, Enum):
    EXPLORE = "explore"
    PLAN = "plan"
    GENERAL = "general"


# ====================================================================== #
# System Prompts (adapted for Kaggle)
# ====================================================================== #

EXPLORE_SYSTEM_PROMPT = f"""You are an Exploration Agent for a Kaggle competition. Your job is to investigate, search, analyze, and return clear, well-sourced findings. You do NOT modify any project files.

## Your Tools

- **read_file_chunk** — Read specific sections of any file (data, code, configs, logs)
- **search_file** — Search within files for keywords, patterns
- **bash** — Shell commands for read-only exploration: `ls`, `find`, `tree`, `head`, `grep`, `wc`, `git log`, `git diff`, etc. Do NOT create, modify, or delete files.
- **python** — Quick computations, data inspection, format parsing. Do NOT write files.

{SUBAGENT_WORKSPACE_REFERENCE}

## Strategy

1. **Orient first**: Run `ls /home/data/`, `ls /home/code/`, `ls /home/agent/` to understand what exists
2. **Search targeted**: Use `search_file` for known terms, `grep -r` via bash for broader patterns
3. **Inspect data**: Use python + pandas to check data shapes, dtypes, missing values
4. **Be precise**: Cite file paths with line numbers, exact values, and direct quotes

## Output

Use `subagent_complete` to submit your findings:
- **Direct answer** to the question or task
- **Evidence** with specific citations (file path, line number, exact values)
- **Uncertainties** — what you couldn't find or verify"""


PLAN_SYSTEM_PROMPT = f"""You are a Planning Agent for a Kaggle competition. Your job is to analyze the competition, data, and current project state, then produce a clear, actionable implementation plan.

## Your Tools

- **read_file_chunk** — Read competition data, code, analysis files
- **search_file** — Search for specific content within files
- **bash** — Shell commands for inspection: `ls`, `find`, `git log`, `git status`, `tree`, etc.
- **python** — Quick computations (estimate data sizes, parse configs, check formats)
- **write_plan** — Save your plan to `/home/agent/plan.md`

{SUBAGENT_WORKSPACE_REFERENCE}

## Planning Methodology

1. **Understand scope**: Read the task description carefully
2. **Assess current state**: Check `/home/code/` for existing code, `git log` for history, `impl_log.md` for progress
3. **Consult the competition**: Read `/home/data/description.md` for goals, metrics, data format
4. **Check prioritized tasks**: If `/home/agent/prioritized_tasks.md` exists, complement it
5. **Consider constraints**:
   - `submission.csv` must match `sample_submission.csv` format exactly
   - All code must be committed to git
   - Use `.gitignore` for models, data, venv/, checkpoints/

## Output

1. Write your full plan to `/home/agent/plan.md` using `write_plan`
2. Use `subagent_complete` to return a concise summary"""


GENERAL_SYSTEM_PROMPT = f"""You are a General-Purpose Agent for a Kaggle competition. You handle auxiliary tasks that require code execution and file modification but don't fit the specialized implement or experiment workflows.

## Typical Use Cases

- Create or update submission generation scripts
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

{SUBAGENT_WORKSPACE_REFERENCE}

## Guidelines

1. **Check state first**: Run `ls`, `git status`, `git log` before making changes
2. **Commit your work**: After meaningful changes, commit in `/home/code/`:
   ```bash
   cd /home/code && git add -A && git commit -m "descriptive message"
   ```
3. **submission.csv rules**: Must match sample_submission.csv format exactly
4. **Size constraints**: Keep committed files under 1GB total. Use `.gitignore`

## Output

Use `subagent_complete` to report:
- What changes were made (files created/modified/deleted)
- Verification results (commands run, outputs observed)
- Any issues encountered or left unresolved"""


# ====================================================================== #
# Specialized Tools
# ====================================================================== #

class PlanWriteTool(Tool):
    """Special tool that only allows writing to plan.md."""

    def name(self) -> str:
        return "write_plan"

    def description(self) -> str:
        return "Write your implementation plan to /home/agent/plan.md."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The complete plan content in markdown format",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        }

    def execute(self, shell, **kwargs) -> str:
        content = kwargs.get("content", "")
        import os
        os.makedirs("/home/agent", exist_ok=True)
        plan_path = "/home/agent/plan.md"
        with open(plan_path, "w") as f:
            f.write(content)
        return f"Plan written to {plan_path}"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "write_plan",
                "description": "Write your implementation plan to /home/agent/plan.md.",
                "parameters": {
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
            },
        }


# ====================================================================== #
# Subagent Classes
# ====================================================================== #

class ExploreSubagent(Subagent):
    """Read-only exploration subagent."""

    @property
    def name(self) -> str:
        return "explore"

    def system_prompt(self) -> str:
        return EXPLORE_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunkTool(),
            SearchFileTool(),
            BashToolWithTimeout(
                default_timeout=EXPLORE_BASH_DEFAULT_TIMEOUT,
                max_timeout=EXPLORE_BASH_DEFAULT_TIMEOUT,
            ),
            PythonTool(
                default_timeout=EXPLORE_BASH_DEFAULT_TIMEOUT,
                max_timeout=EXPLORE_BASH_DEFAULT_TIMEOUT,
            ),
            SubagentCompleteTool(),
        ]


class PlanSubagent(Subagent):
    """Planning subagent with plan-writing capability."""

    @property
    def name(self) -> str:
        return "plan"

    def system_prompt(self) -> str:
        return PLAN_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunkTool(),
            SearchFileTool(),
            BashToolWithTimeout(
                default_timeout=PLAN_BASH_DEFAULT_TIMEOUT,
                max_timeout=PLAN_BASH_DEFAULT_TIMEOUT,
            ),
            PythonTool(
                default_timeout=PLAN_BASH_DEFAULT_TIMEOUT,
                max_timeout=PLAN_BASH_DEFAULT_TIMEOUT,
            ),
            PlanWriteTool(),
            SubagentCompleteTool(),
        ]


class GeneralSubagent(Subagent):
    """General-purpose subagent with full capabilities."""

    @property
    def name(self) -> str:
        return "general"

    def system_prompt(self) -> str:
        return GENERAL_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunkTool(),
            SearchFileTool(),
            BashToolWithTimeout(
                default_timeout=GENERAL_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),
            PythonTool(
                default_timeout=GENERAL_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),
            SubagentCompleteTool(),
        ]
