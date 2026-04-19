"""
Default (production) subagent configurations.

Used when SUBAGENT_CONFIG_PROFILE is unset or set to "default".
Total solver time budget: 24 hours (86400s).

Time allocation strategy (generous — avoid premature cutoff):
- All per-invocation limits set high (~10h) as safe maximums.
- The real constraint is the solver-level time_limit (24h).
- Main Agent manages the overall budget; subagent limits are safety nets only.
"""

from paperbench.solvers.aiscientist.subagents.base import SubagentConfig
from paperbench.solvers.aiscientist.summary_utils import SummaryConfig

# --- Implementation ---
# Single invocation budget; impl now works autonomously for the full session.
# env_setup and resource_download run nested inside impl, sharing this budget.
DEFAULT_IMPLEMENTATION_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=28800,    # 8 hours per invocation (impl works autonomously with full prioritization)
    reminder_freq=20,
    summary_config=SummaryConfig(),  # summary-based context reduction for long sessions
)
IMPLEMENTATION_BASH_DEFAULT_TIMEOUT = 36000  # 10 hours default per bash command

# --- Experiment ---
# Single invocation budget; full mode for real training, validate for quick check.
DEFAULT_EXPERIMENT_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36000,    # 10 hours per invocation (training can be long)
    reminder_freq=30,
    summary_config=SummaryConfig(),  # summary-based context reduction for long sessions
)
EXPERIMENT_VALIDATE_TIME_LIMIT = 18000          # 5 hours for validate (safety net; shorter than full but won't cut off legitimate runs)
EXPERIMENT_COMMAND_TIMEOUT = 36000            # 10 hours per exec_command (training runs can be long)
EXPERIMENT_BASH_DEFAULT_TIMEOUT = 36000       # 10 hours per bash command


# --- Env Setup (nested inside Implementation) ---
DEFAULT_ENV_SETUP_CONFIG = SubagentConfig(
    max_steps=300,
    time_limit=7200,      # 2 hours (pip install large packages can be slow)
)
ENV_SETUP_BASH_DEFAULT_TIMEOUT = 36000        # 10 hours default per bash command
ENV_SETUP_BASH_MAX_TIMEOUT = 36000            # 10 hours max per bash command

# --- Resource Download (nested inside Implementation) ---
DEFAULT_DOWNLOAD_CONFIG = SubagentConfig(
    max_steps=300,
    time_limit=7200,     # 2 hours (large model/dataset downloads)
)

# --- Paper Reader (3 phases, runs once at the start) ---
# These subagents NEVER hang; set to safe maximum so they always finish naturally.
DEFAULT_PAPER_STRUCTURE_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36000,    # 10 hours (safe max; won't hang, let it finish naturally)
)
DEFAULT_PAPER_READER_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36000,    # 10 hours each (safe max; algorithm/experiments/baseline readers)
)
DEFAULT_PAPER_SYNTHESIS_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36000,    # 10 hours (safe max; cross-referencing all reader outputs)
)

# --- Prioritization (runs once after paper reading) ---
DEFAULT_PRIORITIZATION_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36000,    # 10 hours (safe max; won't hang)
)

# --- Search (3 modes) ---
DEFAULT_SEARCH_SIMPLE_CONFIG = SubagentConfig(
    max_steps=100,
    time_limit=1800,     # 30 minutes
)
DEFAULT_SEARCH_STRATEGY_CONFIG = SubagentConfig(
    max_steps=50,
    time_limit=900,      # 15 minutes
)
DEFAULT_SEARCH_EXECUTOR_CONFIG = SubagentConfig(
    max_steps=100,
    time_limit=1800,     # 30 minutes
)

# --- Spawn Subagent (per-type configs) ---
# These replace the old single DEFAULT_SPAWN_SUBAGENT_CONFIG.

# Explore: Read-only analysis and information gathering.
# Generous time since it may need to search through many files and web sources.
DEFAULT_EXPLORE_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=300,
    time_limit=14400,     # 4 hours
    reminder_freq=15,
)
EXPLORE_BASH_DEFAULT_TIMEOUT = 36000 # 10 hours

# Plan: Research + planning, produces plan.md.
DEFAULT_PLAN_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=200,
    time_limit=7200,      # 2 hours
    reminder_freq=15,
)
PLAN_BASH_DEFAULT_TIMEOUT = 36000    # 10 hours

# General: Auxiliary tasks with full capabilities (create reproduce.sh, reorganize files, etc.).
# Longer budget since it may involve real file operations and testing.
DEFAULT_GENERAL_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=300,
    time_limit=14400,     # 4 hours
    reminder_freq=20,
)
GENERAL_BASH_DEFAULT_TIMEOUT = 36000 # 10 hours

# Backward-compatible alias (prefer per-type configs above)
DEFAULT_SPAWN_SUBAGENT_CONFIG = DEFAULT_EXPLORE_SUBAGENT_CONFIG

# --- Main Agent Bash Timeout ---
# The main agent may run bash commands directly (debugging, quick tests, etc.).
# These need a timeout to prevent indefinite blocking.
MAIN_AGENT_BASH_DEFAULT_TIMEOUT = 36000   # 10 hours default per bash command
MAIN_AGENT_BASH_MAX_TIMEOUT = 86400       # 24 hours absolute max (capped to time_limit at runtime)
