"""
Spawn Subagent Tool

A generic tool that allows the main agent to spawn subagents for various tasks.

Subagent Types:
- explore: Read-only information gathering and analysis
- plan: Implementation planning, can write plan.md
- general: Auxiliary tasks with full capabilities

Subagent implementations are in subagents/generic.py
"""

from __future__ import annotations

import structlog
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents import (
    Subagent,
    SubagentOutput,
    SubagentStatus,
    ExploreSubagent,
    PlanSubagent,
    GeneralSubagent,
)
from paperbench.solvers.aiscientist.subagents.base import SubagentConfig
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_EXPLORE_SUBAGENT_CONFIG,
    DEFAULT_PLAN_SUBAGENT_CONFIG,
    DEFAULT_GENERAL_SUBAGENT_CONFIG,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


# Per-type default configs
_TYPE_CONFIGS = {
    "explore": DEFAULT_EXPLORE_SUBAGENT_CONFIG,
    "plan": DEFAULT_PLAN_SUBAGENT_CONFIG,
    "general": DEFAULT_GENERAL_SUBAGENT_CONFIG,
}

# Per-type subagent classes
_TYPE_CLASSES = {
    "explore": ExploreSubagent,
    "plan": PlanSubagent,
    "general": GeneralSubagent,
}


class SpawnSubagentTool(Tool):
    """
    Tool for spawning subagents to perform delegated tasks.

    The main agent uses this for tasks that benefit from context isolation
    or don't fit the implement/experiment tools.
    """

    # Set by solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    run_dir: str | None = None
    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "spawn_subagent"

    def _create_subagent(
        self,
        subagent_type: str,
        time_budget: int | None = None,
    ) -> Subagent | None:
        """Create a subagent instance based on type, with optional time budget override."""
        if self.completer_config is None:
            return None

        if subagent_type not in _TYPE_CONFIGS:
            return None

        config = _TYPE_CONFIGS[subagent_type]

        # Override time_limit if time_budget is provided
        if time_budget is not None and time_budget > 0:
            config = SubagentConfig(
                max_steps=config.max_steps,
                time_limit=time_budget,
                reminder_freq=config.reminder_freq,
            )

        subagent_class = _TYPE_CLASSES[subagent_type]
        return subagent_class(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
        )

    async def execute(
        self,
        computer: ComputerInterface,
        subagent_type: str,
        task: str,
        context: str = "",
        time_budget: int | None = None,
    ) -> str:
        """
        Spawn a subagent to perform a task.

        Args:
            computer: ComputerInterface for execution
            subagent_type: Type of subagent (explore, plan, general)
            task: Description of the task to perform
            context: Optional context — error messages, partial results, or relevant
                     findings from previous work to pass to the subagent
            time_budget: Optional time budget in seconds (overrides default per-type config)

        Returns:
            Subagent output as formatted string
        """
        ctx_logger = logger.bind(tool="spawn_subagent", subagent_type=subagent_type)

        if self.completer_config is None:
            return "Error: SpawnSubagentTool not properly configured."

        # Validate subagent type
        valid_types = list(_TYPE_CONFIGS.keys())
        if subagent_type not in valid_types:
            return f"Error: Unknown subagent type '{subagent_type}'. Available: {valid_types}"

        # Create subagent
        subagent = self._create_subagent(subagent_type, time_budget)
        if subagent is None:
            return f"Error: Failed to create subagent of type '{subagent_type}'"

        ctx_logger.info(
            f"Spawning {subagent_type} subagent",
            time_budget=time_budget or _TYPE_CONFIGS[subagent_type].time_limit,
        )

        try:
            # Build complete task description with optional context
            full_task = task
            if context:
                full_task += f"\n\n## Additional Context\n{context}"

            # Run subagent
            result = await subagent.run(
                computer=computer,
                task_description=full_task,
                constraints=self.constraints,
            )

            # Format result based on type
            return self._format_result(subagent_type, result)

        except Exception as e:
            ctx_logger.error(f"Subagent failed: {e}")
            return f"Error: Subagent '{subagent_type}' failed: {str(e)}"

    def _format_result(self, subagent_type: str, result: SubagentOutput) -> str:
        """Format the subagent result for the main agent."""
        status_icon = "✓" if result.status == SubagentStatus.COMPLETED else "✗"

        header = f"[{subagent_type.upper()} Subagent {status_icon}] "
        header += f"({result.num_steps} steps, {result.runtime_seconds:.1f}s)"

        if result.status == SubagentStatus.COMPLETED:
            output = f"{header}\n\n{result.content}"

            # For plan type, mention where the plan was saved
            if subagent_type == "plan":
                output += "\n\nFull plan saved to: /home/agent/plan.md"

            return output

        elif result.status == SubagentStatus.FAILED:
            return f"{header}\n\nFailed: {result.error_message}\n\nPartial output:\n{result.content}"

        elif result.status == SubagentStatus.TIMEOUT:
            return f"{header}\n\nTimed out. Partial output:\n{result.content}"

        else:
            return f"{header}\n\nStatus: {result.status}"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Spawn a specialized subagent to perform a task. Use this for tasks that benefit from context isolation or don't fit the implement/experiment tools.

## Subagent Types

**explore** — Information gathering and analysis (read-only)
- Tools: read files, search, bash (read-only), python, web_search, link_summary
- Use for: Understanding code/papers, finding hyperparameters, researching libraries, investigating issues

**plan** — Implementation planning and strategy
- Tools: read, search, bash (read-only), python, web_search, link_summary, write_plan
- Use for: Creating implementation plans, analyzing rubric priorities, task breakdown

**general** — Auxiliary tasks with full capabilities
- Tools: read, search, bash (full), python, web_search, link_summary
- Use for: Creating reproduce.sh, reorganizing files, fixing configs, writing utilities, batch operations""",
            parameters={
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "enum": ["explore", "plan", "general"],
                        "description": "Type of subagent to spawn",
                    },
                    "task": {
                        "type": "string",
                        "description": "Specific task description — be detailed about what to investigate, plan, or build",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context — error messages, partial results, or relevant findings from previous work",
                    },
                    "time_budget": {
                        "type": "integer",
                        "description": "Optional time budget in seconds. Defaults: explore=1800, plan=900, general=3600",
                    },
                },
                "required": ["subagent_type", "task"],
                "additionalProperties": False,
            },
            strict=False,
        )
