# AI Scientist Tools
#
# This module contains Tool classes that the main agent can call.
# Subagent implementations are in the subagents/ directory.
#
# Tool vs Subagent:
# - Tool: Interface that main agent calls (has get_oai_tool_call())
# - Subagent: Execution agent that performs tasks (has system_prompt(), get_tools())
#
# Workflow:
# 1. read_paper: Comprehensive paper analysis
# 2. prioritize_tasks: Create prioritized TODO list
# 3. spawn_subagent: Flexible task execution (explore/plan/general)
# 4. implement / run_experiment: Core execution

from paperbench.solvers.aiscientist.tools.paper_reader_tool import (
    ReadPaperTool,
)
from paperbench.solvers.aiscientist.tools.prioritization_tool import (
    PrioritizeTasksTool,
)
from paperbench.solvers.aiscientist.tools.spawn_subagent_tool import (
    SpawnSubagentTool,
)
from paperbench.solvers.aiscientist.tools.implementation_tool import (
    ImplementationTool,
)
from paperbench.solvers.aiscientist.tools.experiment_tool import (
    ExperimentTool,
)
from paperbench.solvers.aiscientist.tools.basic_tool import (
    BashToolWithTimeout,
)

__all__ = [
    # Paper reading tools
    "ReadPaperTool",
    # Prioritization tools (bridges reading and implementation)
    "PrioritizeTasksTool",
    # Spawn subagent tool (Claude Code style)
    "SpawnSubagentTool",
    # Implementation tool
    "ImplementationTool",
    # Experiment tool
    "ExperimentTool",
    # Bash tool with timeout
    "BashToolWithTimeout",
]
