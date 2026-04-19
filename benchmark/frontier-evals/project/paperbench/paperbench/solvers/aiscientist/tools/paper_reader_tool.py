"""
Paper Reader Tool

This tool allows the main agent to invoke the paper reading subagent workflow.
It implements a two-level reading strategy:

Level 1 (Returned to Agent):
- Executive summary with quick reference tables
- Navigation table pointing to detailed files

Level 2 (Saved to Files):
- /home/agent/paper_analysis/summary.md    (Executive summary)
- /home/agent/paper_analysis/structure.md  (Paper structure with line numbers)
- /home/agent/paper_analysis/algorithm.md  (Algorithms, architecture, initialization)
- /home/agent/paper_analysis/experiments.md (Experiment configs, seeds, outputs)
- /home/agent/paper_analysis/baseline.md   (Baseline methods and implementation needs)

The main agent receives a concise summary immediately, with pointers to
detailed files for on-demand access using read_file_chunk or search_file.
"""

from __future__ import annotations

import structlog
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
)
from paperbench.solvers.aiscientist.subagents.paper_reader import PaperReaderCoordinator
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)

# Output directory for paper analysis files
PAPER_ANALYSIS_DIR = "/home/agent/paper_analysis"


class ReadPaperTool(Tool):
    """
    Tool for the main agent to invoke paper reading subagents.

    This tool implements a two-level reading strategy:

    **What the agent receives (Level 1):**
    - Executive summary with key takeaways
    - Navigation table pointing to detailed files
    - Concise information for immediate planning

    **What gets saved to files (Level 2):**
    - /home/agent/paper_analysis/summary.md    (Executive summary)
    - /home/agent/paper_analysis/structure.md  (Paper structure)
    - /home/agent/paper_analysis/algorithm.md  (Algorithms & architecture)
    - /home/agent/paper_analysis/experiments.md (Experiment configs)
    - /home/agent/paper_analysis/baseline.md   (Baseline methods)

    The main agent can access detailed files on-demand using read_file_chunk
    or search_file when more information is needed.

    Workflow:
    1. Phase 1: Structure extraction
    2. Phase 2: Parallel deep reading (Algorithm, Experiments, Baseline)
    3. Phase 3: Synthesis (creates executive summary)
    """

    # These are set by the solver before use
    completer_config: BasicAgentTurnCompleterConfig | None = None
    constraints: dict | None = None
    output_dir: str = PAPER_ANALYSIS_DIR
    run_dir: str | None = None  # Directory for subagent logs
    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "read_paper"

    async def execute(
        self,
        computer: ComputerInterface,
        paper_path: str = "/home/paper/paper.md",
    ) -> str:
        """
        Execute the paper reading workflow with two-level output.

        Args:
            computer: ComputerInterface for file access
            paper_path: Path to the paper markdown file

        Returns:
            Executive summary with navigation table (Level 1)
            Detailed files are saved to /home/agent/paper_analysis/ (Level 2)
        """
        ctx_logger = logger.bind(tool="read_paper")

        if self.completer_config is None:
            return "Error: ReadPaperTool not properly configured. Completer config is missing."

        ctx_logger.info(f"Starting paper reading workflow for {paper_path}")

        try:
            # Create coordinator with appropriate configs
            coordinator = PaperReaderCoordinator(
                completer_config=self.completer_config,
                structure_config=DEFAULT_PAPER_STRUCTURE_CONFIG,
                reader_config=DEFAULT_PAPER_READER_CONFIG,
                synthesis_config=DEFAULT_PAPER_SYNTHESIS_CONFIG,
                run_dir=self.run_dir,
            )

            # Execute paper reading with structured output
            result = await coordinator.read_paper_structured(
                computer=computer,
                paper_path=paper_path,
                constraints=self.constraints,
            )

            # Save detailed files (Level 2)
            await self._save_analysis_files(computer, result)

            # Build and return summary for the main agent (Level 1)
            return self._build_agent_response(result)

        except Exception as e:
            ctx_logger.error(f"Paper reading failed: {e}")
            import traceback
            return f"Error reading paper: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

    async def _save_analysis_files(
        self,
        computer: ComputerInterface,
        result,
    ) -> None:
        """
        Save detailed analysis files to the paper_analysis directory.

        Creates the directory structure:
        /home/agent/paper_analysis/
            summary.md
            structure.md
            algorithm.md
            experiments.md
            baseline.md
        """
        ctx_logger = logger.bind(tool="read_paper", action="save_files")

        # Create output directory
        await computer.send_shell_command(f"mkdir -p {self.output_dir}")

        # Save each section to its own file
        for section_name, section in result.sections.items():
            filepath = f"{self.output_dir}/{section.filename}"
            try:
                await computer.upload(
                    section.content.encode("utf-8"),
                    filepath,
                )
                ctx_logger.info(f"Saved {section_name} to {filepath}")
            except Exception as e:
                ctx_logger.warning(f"Failed to save {filepath}: {e}")

        # Save the executive summary WITH navigation table as summary.md
        # This includes the file navigation table so other tools know about detailed files
        summary_path = f"{self.output_dir}/summary.md"
        try:
            await computer.upload(
                result.summary_with_navigation.encode("utf-8"),
                summary_path,
            )
            ctx_logger.info(f"Saved executive summary to {summary_path}")
        except Exception as e:
            ctx_logger.warning(f"Failed to save summary: {e}")

    def _build_agent_response(self, result) -> str:
        """
        Build the response returned to the main agent.

        This contains:
        1. Status summary
        2. Executive summary with key takeaways
        3. Navigation table for accessing detailed files
        4. Instructions for accessing details
        """
        lines = [
            "# Paper Reading Complete",
            "",
            f"**Total Runtime**: {result.total_runtime_seconds:.1f}s",
            f"**All Success**: {result.all_success}",
            "",
        ]

        if result.failed_subagents:
            lines.append(f"⚠️ **Failed Subagents**: {', '.join(result.failed_subagents)}")
            lines.append("")

        # Include the summary with navigation
        lines.append(result.summary_with_navigation)

        return "\n".join(lines)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Invoke the paper reading subagent workflow to comprehensively analyze the paper.

**Output:**
- Returns executive summary with key takeaways and navigation table
- Saves detailed analysis to `/home/agent/paper_analysis/` directory
- The returned summary includes file paths for accessing details on-demand

**What Gets Extracted:**
1. Paper structure with line numbers for navigation
2. Core algorithms with pseudo-code and hyperparameters
3. Experiment configurations with exact parameters and random seeds
4. Baseline methods categorized by implementation effort
5. Constraints from addendum.md and blacklist.txt

**When to Call:**
- Call this tool FIRST before starting implementation
- Call ONCE - the analysis is comprehensive and saved to files

**Accessing Details:**
Use `read_file_chunk` or `search_file` with paths from the navigation table.""",
            parameters={
                "type": "object",
                "properties": {
                    "paper_path": {
                        "type": "string",
                        "description": "Path to the paper markdown file",
                        "default": "/home/paper/paper.md",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )
