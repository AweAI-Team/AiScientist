"""
Prioritization Tool

This tool allows the main agent to invoke the prioritization workflow.
It analyzes the paper and rubric to create a prioritized implementation plan.

Usage in Main Agent:
    After calling read_paper, call prioritize_tasks to:
    1. Analyze rubric weights and structure
    2. Cross-reference with paper analysis
    3. Create prioritized TODO list
    4. Save to /home/agent/prioritized_tasks.md

Design Philosophy:
    - This is a "fixed workflow" tool (like read_paper), not a flexible spawn
    - It runs a specialized subagent with focused prompts
    - Output is structured for consumption by implementation phase
"""

from __future__ import annotations

import structlog
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_PRIORITIZATION_CONFIG,
)
from paperbench.solvers.aiscientist.subagents.prioritization import (
    PrioritizationSubagent,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


class PrioritizeTasksTool(Tool):
    """
    Tool for creating a prioritized implementation plan.

    This tool bridges paper reading and implementation by:
    1. Analyzing rubric structure and weights
    2. Cross-referencing with paper analysis
    3. Creating a prioritized task list

    Should be called after read_paper and before starting implementation.
    """

    # Set by solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    run_dir: str | None = None
    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "prioritize_tasks"

    async def execute(
        self,
        computer: ComputerInterface,
        paper_analysis_dir: str = "/home/agent/paper_analysis",
        rubric_path: str = "/home/paper/rubric.json",
        focus_areas: str | None = None,
    ) -> str:
        """
        Execute the prioritization workflow.

        Args:
            computer: ComputerInterface for file access
            paper_analysis_dir: Directory containing paper analysis files from read_paper
            rubric_path: Path to rubric.json
            focus_areas: Optional comma-separated areas to focus on

        Returns:
            Summary of prioritization results
        """
        ctx_logger = logger.bind(tool="prioritize_tasks")

        if self.completer_config is None:
            return "Error: PrioritizeTasksTool not properly configured."

        ctx_logger.info("Starting prioritization workflow")

        # Check that paper analysis exists and read summary as context
        summary_path = f"{paper_analysis_dir}/summary.md"
        try:
            summary_content = await computer.download(summary_path)
            summary_text = summary_content.decode("utf-8", errors="replace")
        except Exception:
            return (
                f"Error: Paper analysis not found at {paper_analysis_dir}/. "
                "Please run read_paper first to generate the paper analysis files."
            )

        # Build task description with summary as context
        # This passes the file structure dynamically from read_paper's output
        task_description = f"""Analyze the paper and rubric to create a prioritized implementation plan.

## Paper Analysis Context

The paper has been analyzed by specialized subagents. The analysis is saved in `/home/agent/paper_analysis/`:
- **summary.md** - Executive summary (included below)
- **structure.md** - Paper structure, section index, abstract, constraints
- **algorithm.md** - Core algorithms, pseudo-code, architecture, hyperparameters
- **experiments.md** - Experiment configurations, datasets, training settings, expected outputs
- **baseline.md** - Baseline methods categorized by implementation effort

Here is the executive summary:

---
{summary_text}
---

## Your Task

1. **Parse the rubric** using `parse_rubric` tool to analyze `/home/paper/rubric.json`
   - Note: Some tasks may not have a rubric file. If rubric is not found, infer priorities from paper structure and contributions.
2. **Review detailed analysis** - use `read_file_chunk` to access files in `/home/agent/paper_analysis/` when you need more details
3. **Cross-reference** rubric items (if available) with paper sections
4. **Assign priorities** (P0/P1/P2/P3) based on evidence:
   - Rubric weights (if available)
   - Paper structure (main text vs appendix)
   - Dependencies between tasks
5. **Identify dependencies** and optimal execution order
6. **Write output** using `write_priorities` tool

## Other Files to Check
- `/home/paper/addendum.md` - Scope clarifications and constraints
- `/home/paper/blacklist.txt` - Blocked resources

## Key Considerations

- P0 tasks should represent the core contribution
- Consider that partial credit is awarded - better to have something for each major component
- Account for time constraints - recommend time allocation
- Flag any risks or unclear requirements

When done, call `subagent_complete` with a brief summary of the prioritization.
"""

        if focus_areas:
            task_description += f"\n\n## Focus Areas\nPay special attention to: {focus_areas}"

        # Create and run the prioritization subagent
        config = DEFAULT_PRIORITIZATION_CONFIG

        subagent = PrioritizationSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
        )

        try:
            result = await subagent.run(
                computer=computer,
                task_description=task_description,
                constraints=self.constraints,
            )

            return self._format_result(result)

        except Exception as e:
            ctx_logger.error(f"Prioritization failed: {e}")
            return f"Error during prioritization: {str(e)}"

    def _format_result(self, result: SubagentOutput) -> str:
        """Format the prioritization result for the main agent."""
        status_icon = "✓" if result.status == SubagentStatus.COMPLETED else "✗"

        header = f"[Prioritization {status_icon}] "
        header += f"({result.num_steps} steps, {result.runtime_seconds:.1f}s)"

        if result.status == SubagentStatus.COMPLETED:
            output_lines = [
                header,
                "",
                "**Prioritized tasks saved to**: `/home/agent/prioritized_tasks.md`",
                "",
                "## Summary",
                result.content,
                "",
                "---",
                "",
                "**Next Steps**:",
                "1. Review the prioritized tasks in `/home/agent/prioritized_tasks.md`",
                "2. Start with P0-Critical tasks",
                "3. Use `spawn_subagent(type='plan')` if you need detailed planning for a specific task",
            ]
            return "\n".join(output_lines)

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
            description="""Create a prioritized implementation plan based on paper analysis and rubric.

**When to use**: After read_paper, before starting implementation.

**What it does**:
1. Analyzes rubric.json to understand task weights and structure
2. Cross-references with paper analysis files:
   - summary.md - Executive summary
   - structure.md - Paper structure and constraints
   - algorithm.md - Algorithms and architecture
   - experiments.md - Experiment configurations
   - baseline.md - Baseline methods
3. Creates prioritized task list with P0 (critical) to P3 (optional) ratings
4. Identifies dependencies and recommended execution order
5. Provides time allocation strategy

**Output**: Saves prioritized plan to /home/agent/prioritized_tasks.md

**Priority Levels**:
- P0-Critical: Core algorithm, main experiments (must complete)
- P1-Important: Baselines, key validations (should complete)
- P2-Valuable: Ablations, additional tests (if time permits)
- P3-Optional: Appendix-only, edge cases (low priority)""",
            parameters={
                "type": "object",
                "properties": {
                    "paper_analysis_dir": {
                        "type": "string",
                        "description": "Directory containing paper analysis files",
                        "default": "/home/agent/paper_analysis",
                    },
                    "rubric_path": {
                        "type": "string",
                        "description": "Path to rubric.json",
                        "default": "/home/paper/rubric.json",
                    },
                    "focus_areas": {
                        "type": "string",
                        "description": "Optional: comma-separated areas to focus on (e.g., 'algorithm,main experiments')",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )


