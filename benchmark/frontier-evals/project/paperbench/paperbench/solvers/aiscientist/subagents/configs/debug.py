"""
Debug subagent configurations.

Used when SUBAGENT_CONFIG_PROFILE=debug.
Shorter timeouts and fewer steps for fast iteration.
"""

from paperbench.solvers.aiscientist.subagents.base import SubagentConfig

# --- Implementation ---
DEFAULT_IMPLEMENTATION_CONFIG = SubagentConfig(
    max_steps=50,
    time_limit=1200,     # 20 minutes
    reminder_freq=5,
)
IMPLEMENTATION_BASH_DEFAULT_TIMEOUT = 60     # 1 min default per bash command (debug, quick ops)

# --- Experiment ---
DEFAULT_EXPERIMENT_CONFIG = SubagentConfig(
    max_steps=15,
    time_limit=600,      # 10 minutes
    reminder_freq=5,
)
EXPERIMENT_VALIDATE_TIME_LIMIT = 120          # 2 min for validate mode
EXPERIMENT_COMMAND_TIMEOUT = 600              # 10 min per exec_command (debug)
EXPERIMENT_BASH_DEFAULT_TIMEOUT = 120         # 2 min per bash command (debug, quick ops)

# --- Env Setup ---
DEFAULT_ENV_SETUP_CONFIG = SubagentConfig(
    max_steps=15,
    time_limit=120,      # 2 minutes
)
ENV_SETUP_BASH_DEFAULT_TIMEOUT = 120          # 2 minutes default
ENV_SETUP_BASH_MAX_TIMEOUT = 300              # 5 minutes max

# --- Resource Download ---
DEFAULT_DOWNLOAD_CONFIG = SubagentConfig(
    max_steps=10,
    time_limit=180,      # 3 minutes
)

# --- Paper Reader (3 phases) ---
DEFAULT_PAPER_STRUCTURE_CONFIG = SubagentConfig(
    max_steps=10,
    time_limit=120,      # 2 minutes
)
DEFAULT_PAPER_READER_CONFIG = SubagentConfig(
    max_steps=15,
    time_limit=120,      # 2 minutes
)
DEFAULT_PAPER_SYNTHESIS_CONFIG = SubagentConfig(
    max_steps=10,
    time_limit=60,       # 1 minute
)

# --- Prioritization ---
DEFAULT_PRIORITIZATION_CONFIG = SubagentConfig(
    max_steps=15,
    time_limit=120,      # 2 minutes
)

# --- Search (3 modes) ---
DEFAULT_SEARCH_SIMPLE_CONFIG = SubagentConfig(
    max_steps=8,
    time_limit=60,       # 1 minute
)
DEFAULT_SEARCH_STRATEGY_CONFIG = SubagentConfig(
    max_steps=5,
    time_limit=30,       # 30 seconds
)
DEFAULT_SEARCH_EXECUTOR_CONFIG = SubagentConfig(
    max_steps=6,
    time_limit=45,       # 45 seconds
)

# --- Spawn Subagent (per-type configs) ---
DEFAULT_EXPLORE_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=15,
    time_limit=120,       # 2 minutes
    reminder_freq=5,
)
EXPLORE_BASH_DEFAULT_TIMEOUT = 60    # 1 min

DEFAULT_PLAN_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=15,
    time_limit=120,       # 2 minutes
    reminder_freq=5,
)
PLAN_BASH_DEFAULT_TIMEOUT = 60       # 1 min

DEFAULT_GENERAL_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=20,
    time_limit=300,       # 5 minutes
    reminder_freq=5,
)
GENERAL_BASH_DEFAULT_TIMEOUT = 180   # 3 min

# Backward-compatible alias
DEFAULT_SPAWN_SUBAGENT_CONFIG = DEFAULT_EXPLORE_SUBAGENT_CONFIG

# --- Main Agent Bash Timeout ---
MAIN_AGENT_BASH_DEFAULT_TIMEOUT = 600     # 10 minutes default (debug mode)
MAIN_AGENT_BASH_MAX_TIMEOUT = 1800        # 30 minutes max (debug mode)
