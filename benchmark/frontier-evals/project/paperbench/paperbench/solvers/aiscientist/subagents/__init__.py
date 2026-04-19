# Subagent implementations for AI Scientist Solver
#
# This module contains all subagent definitions. Tools that use these subagents
# are defined in the tools/ directory.
#
# Directory Structure:
# - base.py: Core abstractions (Subagent, SubagentConfig, SubagentOutput, etc.)
# - configs.py: Centralized subagent configurations (all DEFAULT_*_CONFIG constants)
# - coordinator.py: Parallel execution coordination
# - paper_reader.py: Paper reading subagents (Structure, Algorithm, Experiments, Baseline)
# - prioritization.py: Prioritization subagent (bridges reading and implementation)
# - search.py: Search subagents (Strategist, Executor)
# - generic.py: Generic subagents (Explore, Plan, General)

from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteSignal,
    SubagentCompleteTool,
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_DOWNLOAD_CONFIG,
    DEFAULT_ENV_SETUP_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_IMPLEMENTATION_CONFIG,
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
    DEFAULT_PRIORITIZATION_CONFIG,
    DEFAULT_SEARCH_EXECUTOR_CONFIG,
    DEFAULT_SEARCH_SIMPLE_CONFIG,
    DEFAULT_SEARCH_STRATEGY_CONFIG,
    DEFAULT_SPAWN_SUBAGENT_CONFIG,
    DEFAULT_EXPLORE_SUBAGENT_CONFIG,
    DEFAULT_PLAN_SUBAGENT_CONFIG,
    DEFAULT_GENERAL_SUBAGENT_CONFIG,
    EXPLORE_BASH_DEFAULT_TIMEOUT,
    PLAN_BASH_DEFAULT_TIMEOUT,
    GENERAL_BASH_DEFAULT_TIMEOUT,
    ENV_SETUP_BASH_DEFAULT_TIMEOUT,
    ENV_SETUP_BASH_MAX_TIMEOUT,
    EXPERIMENT_COMMAND_TIMEOUT,
    EXPERIMENT_VALIDATE_TIME_LIMIT,
)
from paperbench.solvers.aiscientist.subagents.coordinator import (
    CoordinatorResult,
    SequentialCoordinator,
    SubagentCoordinator,
    SubagentTask,
)
from paperbench.solvers.aiscientist.subagents.env_setup import (
    CheckEnvStatusTool,
    EnvSetupSubagent,
    RecordEnvSetupTool,
)
from paperbench.solvers.aiscientist.subagents.experiment import (
    ExecCommandTool,
    ExperimentSubagent,
    RunExperimentTool,
)
from paperbench.solvers.aiscientist.subagents.generic import (
    ExploreSubagent,
    GeneralSubagent,
    PlanSubagent,
    PlanWriteTool,
    SubagentType,
)
from paperbench.solvers.aiscientist.subagents.implementation import (
    FileEditTool,
    GitCommitTool,
    ImplementationSubagent,
    SpawnEnvSetupTool,
    SpawnResourceDownloadTool,
)
from paperbench.solvers.aiscientist.subagents.paper_reader import (
    AlgorithmSubagent,
    BaselineSubagent,
    ExperimentsSubagent,
    PaperAnalysisResult,
    PaperAnalysisSection,
    PaperReaderCoordinator,
    StructureSubagent,
    SynthesisSubagent,
)
from paperbench.solvers.aiscientist.subagents.prioritization import (
    DEFAULT_PRIORITIZATION_TASK,
    ParseRubricTool,
    PrioritizationSubagent,
    Priority,
    PriorityWriteTool,
)
from paperbench.solvers.aiscientist.subagents.resource_download import (
    CheckDownloadStatusTool,
    RecordDownloadTool,
    ResourceDownloadSubagent,
)
from paperbench.solvers.aiscientist.subagents.search import (
    SearchExecutorSubagent,
    SearchStrategistSubagent,
)
from paperbench.solvers.aiscientist.subagents.state_manager import (
    AddExpLogTool,
    AddImplLogTool,
)

__all__ = [
    # Base classes
    "Subagent",
    "SubagentConfig",
    "SubagentOutput",
    "SubagentStatus",
    "SubagentCompleteTool",
    "SubagentCompleteSignal",
    # Centralized configs
    "DEFAULT_IMPLEMENTATION_CONFIG",
    "DEFAULT_EXPERIMENT_CONFIG",
    "EXPERIMENT_VALIDATE_TIME_LIMIT",
    "EXPERIMENT_COMMAND_TIMEOUT",
    "DEFAULT_ENV_SETUP_CONFIG",
    "ENV_SETUP_BASH_DEFAULT_TIMEOUT",
    "ENV_SETUP_BASH_MAX_TIMEOUT",
    "DEFAULT_DOWNLOAD_CONFIG",
    "DEFAULT_PAPER_STRUCTURE_CONFIG",
    "DEFAULT_PAPER_READER_CONFIG",
    "DEFAULT_PAPER_SYNTHESIS_CONFIG",
    "DEFAULT_PRIORITIZATION_CONFIG",
    "DEFAULT_SEARCH_SIMPLE_CONFIG",
    "DEFAULT_SEARCH_STRATEGY_CONFIG",
    "DEFAULT_SEARCH_EXECUTOR_CONFIG",
    "DEFAULT_SPAWN_SUBAGENT_CONFIG",
    "DEFAULT_EXPLORE_SUBAGENT_CONFIG",
    "DEFAULT_PLAN_SUBAGENT_CONFIG",
    "DEFAULT_GENERAL_SUBAGENT_CONFIG",
    "EXPLORE_BASH_DEFAULT_TIMEOUT",
    "PLAN_BASH_DEFAULT_TIMEOUT",
    "GENERAL_BASH_DEFAULT_TIMEOUT",
    # Coordinators
    "SubagentCoordinator",
    "SequentialCoordinator",
    "SubagentTask",
    "CoordinatorResult",
    # Paper reader subagents
    "StructureSubagent",
    "AlgorithmSubagent",
    "ExperimentsSubagent",
    "BaselineSubagent",
    "SynthesisSubagent",
    "PaperReaderCoordinator",
    "PaperAnalysisResult",
    "PaperAnalysisSection",
    # Prioritization subagent
    "PrioritizationSubagent",
    "PriorityWriteTool",
    "ParseRubricTool",
    "Priority",
    "DEFAULT_PRIORITIZATION_TASK",
    # Search subagents
    "SearchStrategistSubagent",
    "SearchExecutorSubagent",
    # Generic subagents
    "SubagentType",
    "ExploreSubagent",
    "PlanSubagent",
    "GeneralSubagent",
    "PlanWriteTool",
    # Logging tools
    "AddImplLogTool",
    "AddExpLogTool",
    # Environment setup
    "EnvSetupSubagent",
    "CheckEnvStatusTool",
    "RecordEnvSetupTool",
    # Resource download
    "ResourceDownloadSubagent",
    "CheckDownloadStatusTool",
    "RecordDownloadTool",
    # Implementation
    "ImplementationSubagent",
    "SpawnEnvSetupTool",
    "SpawnResourceDownloadTool",
    "GitCommitTool",
    "FileEditTool",
    # Experiment
    "ExperimentSubagent",
    "ExecCommandTool",
    "RunExperimentTool",
]
