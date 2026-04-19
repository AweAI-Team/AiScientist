"""
Centralized Subagent Configurations (dynamic profile loader)

Reads SUBAGENT_CONFIG_PROFILE env var to decide which config module to load.
Profile name must match a .py file under this package, e.g.:
    SUBAGENT_CONFIG_PROFILE=debug  ->  configs/debug.py
    SUBAGENT_CONFIG_PROFILE=default (or unset)  ->  configs/default.py

All downstream imports remain unchanged:
    from paperbench.solvers.aiscientist.subagents.configs import DEFAULT_IMPLEMENTATION_CONFIG
"""

import os
import importlib

_profile = os.environ.get("SUBAGENT_CONFIG_PROFILE", "default")
_module = importlib.import_module(
    f"paperbench.solvers.aiscientist.subagents.configs.{_profile}"
)

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
