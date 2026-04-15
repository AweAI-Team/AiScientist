"""
Spawn Subagent Tool — delegates to a short-lived subagent session.

Allows the main agent to spawn generic subagents for various tasks:
- explore: read-only information gathering and analysis
- plan: implementation planning, can write plan.md
- general: auxiliary tasks with full capabilities
"""

from __future__ import annotations

import os
import time

import structlog

from llm_client import LLMClient
from shell_interface import ShellInterface
from subagents.base import SubagentConfig, SubagentOutput, SubagentStatus
from subagents.configs import (
    DEFAULT_EXPLORE_SUBAGENT_CONFIG,
    DEFAULT_PLAN_SUBAGENT_CONFIG,
    DEFAULT_GENERAL_SUBAGENT_CONFIG,
)
from subagents.generic import ExploreSubagent, PlanSubagent, GeneralSubagent
from tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)

LOGS_DIR = os.environ.get("LOGS_DIR", "/home/logs")

_TYPE_CONFIGS = {
    "explore": DEFAULT_EXPLORE_SUBAGENT_CONFIG,
    "plan": DEFAULT_PLAN_SUBAGENT_CONFIG,
    "general": DEFAULT_GENERAL_SUBAGENT_CONFIG,
}

_TYPE_CLASSES = {
    "explore": ExploreSubagent,
    "plan": PlanSubagent,
    "general": GeneralSubagent,
}


class SpawnSubagentTool(Tool):
    """
    Tool for spawning generic subagents.

    The main agent uses this for tasks that benefit from context isolation
    or don't fit the implement/experiment tools.
    """

    def __init__(self, shell: ShellInterface, llm: LLMClient):
        self._shell = shell
        self._llm = llm
        self._session_counter = 0

    def name(self) -> str:
        return "spawn_subagent"

    def description(self) -> str:
        return (
            "Spawn a specialized subagent to perform a task. Use this for tasks that "
            "benefit from context isolation or don't fit the implement/experiment tools.\n\n"
            "## Subagent Types\n\n"
            "**explore** - Information gathering and analysis (read-only)\n"
            "- Use for: understanding data/code, finding patterns, investigating issues\n\n"
            "**plan** - Implementation planning and strategy\n"
            "- Use for: creating implementation plans, analyzing priorities, task breakdown\n\n"
            "**general** - Auxiliary tasks with full capabilities\n"
            "- Use for: creating scripts, reorganizing files, fixing configs, batch operations"
        )

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "enum": ["explore", "plan", "general"],
                    "description": "Type of subagent to spawn",
                },
                "task": {
                    "type": "string",
                    "description": "Specific task description - be detailed about what to investigate, plan, or build",
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context - error messages, partial results, or relevant findings",
                },
                "time_budget": {
                    "type": "integer",
                    "description": "Optional time budget in seconds (overrides default per-type config)",
                },
            },
            "required": ["subagent_type", "task"],
            "additionalProperties": False,
        }

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.parameters(),
            },
        }

    def execute(self, shell, **kwargs) -> str:
        subagent_type = kwargs.get("subagent_type", "")
        task = kwargs.get("task", "")
        context = kwargs.get("context", "")
        time_budget = kwargs.get("time_budget")

        valid_types = list(_TYPE_CONFIGS.keys())
        if subagent_type not in valid_types:
            return f"Error: Unknown subagent type '{subagent_type}'. Available: {valid_types}"

        config = _TYPE_CONFIGS[subagent_type]
        if time_budget is not None and int(time_budget) > 0:
            config = SubagentConfig(
                max_steps=config.max_steps,
                time_limit=int(time_budget),
                reminder_freq=config.reminder_freq,
            )

        self._session_counter += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{LOGS_DIR}/subagent_logs/generic_{subagent_type}_{self._session_counter:03d}_{ts}"
        os.makedirs(session_dir, exist_ok=True)

        config_with_log = SubagentConfig(
            max_steps=config.max_steps,
            time_limit=config.time_limit,
            reminder_freq=config.reminder_freq,
            log_dir=session_dir,
        )

        subagent_class = _TYPE_CLASSES[subagent_type]
        subagent = subagent_class(self._shell, self._llm, config_with_log)

        logger.info(
            f"Spawning {subagent_type} subagent",
            time_budget=config_with_log.time_limit,
            session_dir=session_dir,
        )

        full_task = task
        if context:
            full_task += f"\n\n## Additional Context\n{context}"

        try:
            result = subagent.run(context=full_task)
            logger.info(
                f"{subagent_type} subagent finished",
                steps=result.num_steps,
                runtime_s=result.runtime_seconds,
                session_dir=session_dir,
            )
            return self._format_result(subagent_type, result)
        except Exception as e:
            logger.error(f"Subagent failed: {e}")
            return f"Error: Subagent '{subagent_type}' failed: {str(e)}"

    def _format_result(self, subagent_type: str, result: SubagentOutput) -> str:
        status_icon = "+" if result.status == SubagentStatus.COMPLETED else "x"
        header = (
            f"[{subagent_type.upper()} Subagent {status_icon}] "
            f"({result.num_steps} steps, {result.runtime_seconds:.1f}s)"
        )

        if result.status == SubagentStatus.COMPLETED:
            output = f"{header}\n\n{result.content}"
            if subagent_type == "plan":
                output += "\n\nFull plan saved to: /home/agent/plan.md"
            return output
        elif result.status == SubagentStatus.FAILED:
            return f"{header}\n\nFailed: {result.error_message}\n\nPartial output:\n{result.content}"
        elif result.status == SubagentStatus.TIMEOUT:
            return f"{header}\n\nTimed out. Partial output:\n{result.content}"
        else:
            return f"{header}\n\nStatus: {result.status}"
