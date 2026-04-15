"""
Subagent configurations — profile-based time limits and step caps.

Two profiles:
- **default** (production): generous time budgets (24h overall, 8h impl, 10h exp defaults).
- **debug**: short budgets for fast iteration

The active profile is selected via the ``AISCI_CONFIG_PROFILE`` env var
(default: ``"default"``).

Individual values can be overridden via env vars:
- ``AISCI_MAX_STEPS``: shared max_steps for implementation/experiment
- ``AISCI_IMPL_TIME``: implementation time budget (seconds)
- ``AISCI_EXP_TIME``:  experiment time budget (seconds)
"""

from __future__ import annotations

import os

from subagents.base import SubagentConfig

# ====================================================================== #
# Profile: default (production)
# ====================================================================== #

_DEFAULT = {
    "analysis":       SubagentConfig(max_steps=200, time_limit=3600,  reminder_freq=15),
    "prioritization": SubagentConfig(max_steps=200, time_limit=3600,  reminder_freq=15),
    "implementation": SubagentConfig(max_steps=500, time_limit=28800, reminder_freq=20),
    "experiment":     SubagentConfig(max_steps=500, time_limit=36000, reminder_freq=30),  # 10h experiment budget

    "explore": SubagentConfig(max_steps=300, time_limit=14400, reminder_freq=15),  # 2h→4h
    "plan":    SubagentConfig(max_steps=200, time_limit=3600,  reminder_freq=15),
    "general": SubagentConfig(max_steps=300, time_limit=14400, reminder_freq=20),  # 2h→4h

    "ANALYSIS_BASH_DEFAULT_TIMEOUT": 3600,
    "ANALYSIS_BASH_MAX_TIMEOUT": 7200,
    "PRIORITIZATION_BASH_DEFAULT_TIMEOUT": 1800,
    "PRIORITIZATION_BASH_MAX_TIMEOUT": 3600,
    "IMPLEMENTATION_BASH_DEFAULT_TIMEOUT": 36000,
    "IMPLEMENTATION_BASH_MAX_TIMEOUT": 36000,
    "EXPERIMENT_BASH_DEFAULT_TIMEOUT": 36000,
    "EXPERIMENT_BASH_MAX_TIMEOUT": 36000,
    "EXPERIMENT_COMMAND_TIMEOUT": 36000,  # 4h→10h: single exec_command may run long training
    "EXPERIMENT_VALIDATE_TIME_LIMIT": 18000,  # 5h for validate mode
    "EXPLORE_BASH_DEFAULT_TIMEOUT": 36000,
    "PLAN_BASH_DEFAULT_TIMEOUT": 3600,
    "GENERAL_BASH_DEFAULT_TIMEOUT": 36000,
    "MAIN_BASH_DEFAULT_TIMEOUT": 14400,
    "MAIN_BASH_MAX_TIMEOUT": 86400,
}

# ====================================================================== #
# Profile: debug — same subagent time/step limits as default (production).
# Only difference: run one task and optionally shorter overall TIME_LIMIT_SECS in config.
# ====================================================================== #

_DEBUG = {
    "analysis":       SubagentConfig(max_steps=200, time_limit=3600,  reminder_freq=15),
    "prioritization": SubagentConfig(max_steps=200, time_limit=3600,  reminder_freq=15),
    "implementation": SubagentConfig(max_steps=500, time_limit=28800, reminder_freq=20),
    "experiment":     SubagentConfig(max_steps=500, time_limit=36000, reminder_freq=30),

    "explore": SubagentConfig(max_steps=300, time_limit=14400, reminder_freq=15),
    "plan":    SubagentConfig(max_steps=200, time_limit=3600,  reminder_freq=15),
    "general": SubagentConfig(max_steps=300, time_limit=14400, reminder_freq=20),

    "ANALYSIS_BASH_DEFAULT_TIMEOUT": 3600,
    "ANALYSIS_BASH_MAX_TIMEOUT": 7200,
    "PRIORITIZATION_BASH_DEFAULT_TIMEOUT": 1800,
    "PRIORITIZATION_BASH_MAX_TIMEOUT": 3600,
    "IMPLEMENTATION_BASH_DEFAULT_TIMEOUT": 36000,
    "IMPLEMENTATION_BASH_MAX_TIMEOUT": 36000,
    "EXPERIMENT_BASH_DEFAULT_TIMEOUT": 36000,
    "EXPERIMENT_BASH_MAX_TIMEOUT": 36000,
    "EXPERIMENT_COMMAND_TIMEOUT": 36000,
    "EXPERIMENT_VALIDATE_TIME_LIMIT": 18000,
    "EXPLORE_BASH_DEFAULT_TIMEOUT": 36000,
    "PLAN_BASH_DEFAULT_TIMEOUT": 3600,
    "GENERAL_BASH_DEFAULT_TIMEOUT": 36000,
    "MAIN_BASH_DEFAULT_TIMEOUT": 14400,
    "MAIN_BASH_MAX_TIMEOUT": 86400,
}

# ====================================================================== #
# Profile selection + env-var overrides
# ====================================================================== #

_profile_name = os.environ.get("AISCI_CONFIG_PROFILE", "default")
_profile = _DEBUG if _profile_name == "debug" else _DEFAULT

# Env-var overrides
_max_steps_override = int(os.environ.get("AISCI_MAX_STEPS", 0))
_impl_time_override = int(os.environ.get("AISCI_IMPL_TIME", 0))
_exp_time_override = int(os.environ.get("AISCI_EXP_TIME", 0))

DEFAULT_ANALYSIS_CONFIG: SubagentConfig = _profile["analysis"]
DEFAULT_PRIORITIZATION_CONFIG: SubagentConfig = _profile["prioritization"]

DEFAULT_IMPLEMENTATION_CONFIG: SubagentConfig = _profile["implementation"]
if _max_steps_override:
    DEFAULT_IMPLEMENTATION_CONFIG.max_steps = _max_steps_override
if _impl_time_override:
    DEFAULT_IMPLEMENTATION_CONFIG.time_limit = _impl_time_override

DEFAULT_EXPERIMENT_CONFIG: SubagentConfig = _profile["experiment"]
if _max_steps_override:
    DEFAULT_EXPERIMENT_CONFIG.max_steps = _max_steps_override
if _exp_time_override:
    DEFAULT_EXPERIMENT_CONFIG.time_limit = _exp_time_override

# Generic subagent configs
DEFAULT_EXPLORE_SUBAGENT_CONFIG: SubagentConfig = _profile["explore"]
DEFAULT_PLAN_SUBAGENT_CONFIG: SubagentConfig = _profile["plan"]
DEFAULT_GENERAL_SUBAGENT_CONFIG: SubagentConfig = _profile["general"]

# Timeout constants exported for tool construction
ANALYSIS_BASH_DEFAULT_TIMEOUT: int = _profile["ANALYSIS_BASH_DEFAULT_TIMEOUT"]
ANALYSIS_BASH_MAX_TIMEOUT: int = _profile["ANALYSIS_BASH_MAX_TIMEOUT"]

PRIORITIZATION_BASH_DEFAULT_TIMEOUT: int = _profile["PRIORITIZATION_BASH_DEFAULT_TIMEOUT"]
PRIORITIZATION_BASH_MAX_TIMEOUT: int = _profile["PRIORITIZATION_BASH_MAX_TIMEOUT"]

IMPLEMENTATION_BASH_DEFAULT_TIMEOUT: int = _profile["IMPLEMENTATION_BASH_DEFAULT_TIMEOUT"]
IMPLEMENTATION_BASH_MAX_TIMEOUT: int = _profile["IMPLEMENTATION_BASH_MAX_TIMEOUT"]

EXPERIMENT_BASH_DEFAULT_TIMEOUT: int = _profile["EXPERIMENT_BASH_DEFAULT_TIMEOUT"]
EXPERIMENT_BASH_MAX_TIMEOUT: int = _profile["EXPERIMENT_BASH_MAX_TIMEOUT"]
EXPERIMENT_COMMAND_TIMEOUT: int = _profile["EXPERIMENT_COMMAND_TIMEOUT"]
EXPERIMENT_VALIDATE_TIME_LIMIT: int = _profile["EXPERIMENT_VALIDATE_TIME_LIMIT"]

EXPLORE_BASH_DEFAULT_TIMEOUT: int = _profile["EXPLORE_BASH_DEFAULT_TIMEOUT"]
PLAN_BASH_DEFAULT_TIMEOUT: int = _profile["PLAN_BASH_DEFAULT_TIMEOUT"]
GENERAL_BASH_DEFAULT_TIMEOUT: int = _profile["GENERAL_BASH_DEFAULT_TIMEOUT"]

MAIN_BASH_DEFAULT_TIMEOUT: int = _profile["MAIN_BASH_DEFAULT_TIMEOUT"]
MAIN_BASH_MAX_TIMEOUT: int = _profile["MAIN_BASH_MAX_TIMEOUT"]
