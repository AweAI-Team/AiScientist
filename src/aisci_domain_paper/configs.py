from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aisci_agent_runtime.subagents.base import SubagentConfig
from aisci_agent_runtime.summary_utils import SummaryConfig
from aisci_core.paths import repo_root

DEFAULT_PAPER_SUBAGENT_CONFIG_PATH = Path("config") / "paper_subagents.yaml"

_DEFAULT_SUBAGENT_PAYLOAD: dict[str, Any] = {
    "subagents": {
        "implementation": {
            "max_steps": 500,
            "time_limit": 28_800,
            "reminder_freq": 20,
            "summary": {"enabled": True},
        },
        "experiment": {
            "max_steps": 500,
            "time_limit": 36_000,
            "reminder_freq": 30,
            "summary": {"enabled": True},
        },
        "env_setup": {
            "max_steps": 300,
            "time_limit": 7_200,
            "reminder_freq": 15,
        },
        "resource_download": {
            "max_steps": 300,
            "time_limit": 7_200,
            "reminder_freq": 15,
        },
        "paper_structure": {
            "max_steps": 500,
            "time_limit": 36_000,
            "reminder_freq": 15,
        },
        "paper_reader": {
            "max_steps": 500,
            "time_limit": 36_000,
            "reminder_freq": 15,
        },
        "paper_synthesis": {
            "max_steps": 500,
            "time_limit": 36_000,
            "reminder_freq": 15,
        },
        "prioritization": {
            "max_steps": 500,
            "time_limit": 36_000,
            "reminder_freq": 15,
        },
        "explore": {
            "max_steps": 300,
            "time_limit": 14_400,
            "reminder_freq": 15,
        },
        "plan": {
            "max_steps": 200,
            "time_limit": 7_200,
            "reminder_freq": 15,
        },
        "general": {
            "max_steps": 300,
            "time_limit": 14_400,
            "reminder_freq": 20,
        },
    },
    "timeouts": {
        "main_agent_bash_default": 36_000,
        "main_agent_bash_max": 86_400,
        "implementation_bash_default": 36_000,
        "experiment_bash_default": 36_000,
        "experiment_command_timeout": 36_000,
        "experiment_validate_time_limit": 18_000,
        "env_setup_bash_default": 36_000,
        "env_setup_bash_max": 36_000,
        "resource_download_bash_default": 600,
        "resource_download_bash_max": 7_200,
        "explore_bash_default": 36_000,
        "plan_bash_default": 36_000,
        "general_bash_default": 36_000,
    },
}


@dataclass(frozen=True)
class PaperSubagentRegistry:
    subagents: dict[str, SubagentConfig]
    timeouts: dict[str, int]


def _resolve_config_path(config_file: str | None = None) -> Path:
    if config_file:
        path = Path(config_file).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    return (repo_root() / DEFAULT_PAPER_SUBAGENT_CONFIG_PATH).resolve()


def resolved_paper_subagent_config_path(config_file: str | None = None) -> Path:
    return _resolve_config_path(config_file)


def _read_config_source(config_file: str | None = None) -> tuple[str, str]:
    path = _resolve_config_path(config_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Paper subagent config file not found: {path}. "
            f"Create {DEFAULT_PAPER_SUBAGENT_CONFIG_PATH} in the repo root."
        )
    return path.read_text(encoding="utf-8"), str(path)


def _require_mapping(value: Any, *, label: str, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label} in {source}: expected a mapping.")
    return dict(value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _parse_summary_config(raw: Any) -> SummaryConfig | None:
    if raw in (None, False):
        return None
    if raw is True:
        return SummaryConfig()
    if not isinstance(raw, dict):
        raise ValueError("summary must be a mapping, true, false, or null.")
    return SummaryConfig(**raw)


def _build_subagent_config(name: str, raw: dict[str, Any], *, source: str) -> SubagentConfig:
    try:
        max_steps = int(raw["max_steps"])
        time_limit = int(raw["time_limit"])
    except KeyError as exc:
        raise ValueError(f"Subagent {name!r} in {source} is missing {exc.args[0]!r}.") from exc
    reminder_freq = int(raw.get("reminder_freq", 10))
    summary_config = _parse_summary_config(raw.get("summary"))
    return SubagentConfig(
        max_steps=max_steps,
        time_limit=time_limit,
        reminder_freq=reminder_freq,
        summary_config=summary_config,
    )


def load_paper_subagent_registry(config_file: str | None = None) -> PaperSubagentRegistry:
    text, source = _read_config_source(config_file)
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid paper subagent config in {source}: root must be a mapping.")
    merged = _deep_merge(_DEFAULT_SUBAGENT_PAYLOAD, payload)

    subagents_raw = _require_mapping(merged.get("subagents"), label="subagents", source=source)
    timeouts_raw = _require_mapping(merged.get("timeouts"), label="timeouts", source=source)

    subagents = {
        str(name): _build_subagent_config(str(name), raw, source=source)
        for name, raw in subagents_raw.items()
    }
    timeouts = {str(name): int(value) for name, value in timeouts_raw.items()}
    return PaperSubagentRegistry(subagents=subagents, timeouts=timeouts)


_REGISTRY = load_paper_subagent_registry()

DEFAULT_IMPLEMENTATION_CONFIG = _REGISTRY.subagents["implementation"]
DEFAULT_EXPERIMENT_CONFIG = _REGISTRY.subagents["experiment"]
DEFAULT_ENV_SETUP_CONFIG = _REGISTRY.subagents["env_setup"]
DEFAULT_DOWNLOAD_CONFIG = _REGISTRY.subagents["resource_download"]
DEFAULT_PAPER_STRUCTURE_CONFIG = _REGISTRY.subagents["paper_structure"]
DEFAULT_PAPER_READER_CONFIG = _REGISTRY.subagents["paper_reader"]
DEFAULT_PAPER_SYNTHESIS_CONFIG = _REGISTRY.subagents["paper_synthesis"]
DEFAULT_PRIORITIZATION_CONFIG = _REGISTRY.subagents["prioritization"]
DEFAULT_EXPLORE_SUBAGENT_CONFIG = _REGISTRY.subagents["explore"]
DEFAULT_PLAN_SUBAGENT_CONFIG = _REGISTRY.subagents["plan"]
DEFAULT_GENERAL_SUBAGENT_CONFIG = _REGISTRY.subagents["general"]

MAIN_AGENT_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["main_agent_bash_default"]
MAIN_AGENT_BASH_MAX_TIMEOUT = _REGISTRY.timeouts["main_agent_bash_max"]
IMPLEMENTATION_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["implementation_bash_default"]
EXPERIMENT_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["experiment_bash_default"]
EXPERIMENT_COMMAND_TIMEOUT = _REGISTRY.timeouts["experiment_command_timeout"]
EXPERIMENT_VALIDATE_TIME_LIMIT = _REGISTRY.timeouts["experiment_validate_time_limit"]
ENV_SETUP_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["env_setup_bash_default"]
ENV_SETUP_BASH_MAX_TIMEOUT = _REGISTRY.timeouts["env_setup_bash_max"]
RESOURCE_DOWNLOAD_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["resource_download_bash_default"]
RESOURCE_DOWNLOAD_BASH_MAX_TIMEOUT = _REGISTRY.timeouts["resource_download_bash_max"]
EXPLORE_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["explore_bash_default"]
PLAN_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["plan_bash_default"]
GENERAL_BASH_DEFAULT_TIMEOUT = _REGISTRY.timeouts["general_bash_default"]
