from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aisci_core.paths import repo_root

DEFAULT_LLM_PROFILE_FILE_ENV = "AISCI_LLM_PROFILE_FILE"
DEFAULT_LLM_PROFILE_PATH = Path("config") / "llm_profiles.yaml"
SUPPORTED_BACKEND_TYPES = {"openai", "azure-openai"}


@dataclass(frozen=True)
class BackendEnvVar:
    name: str
    env_var: str
    required: bool = False


@dataclass(frozen=True)
class BackendConfig:
    name: str
    kind: str
    env: dict[str, BackendEnvVar]


@dataclass(frozen=True)
class LLMProfile:
    name: str
    backend_name: str
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    web_search: bool = False
    max_tokens: int = 32768
    # Maximum model context window from the YAML profile. The runtime derives
    # the actual prune budget locally from this value and max_tokens.
    context_window: int | None = None
    use_phase: bool = False
    temperature: float | None = None
    clear_thinking: bool | None = None
    backend_env: dict[str, BackendEnvVar] | None = None


@dataclass(frozen=True)
class LLMRegistry:
    defaults: dict[str, str]
    backends: dict[str, BackendConfig]
    profiles: dict[str, dict[str, Any]]


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_profile_path(profile_file: str | None = None) -> Path:
    configured = (profile_file or os.environ.get(DEFAULT_LLM_PROFILE_FILE_ENV, "")).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    return (repo_root() / DEFAULT_LLM_PROFILE_PATH).resolve()


def resolved_profile_path(profile_file: str | None = None) -> Path:
    return _resolve_profile_path(profile_file)


def _read_profile_source(profile_file: str | None = None) -> tuple[str, str]:
    path = _resolve_profile_path(profile_file)
    if not path.exists():
        raise FileNotFoundError(
            f"LLM profile file not found: {path}. "
            f"Create {DEFAULT_LLM_PROFILE_PATH} in the repo root or set {DEFAULT_LLM_PROFILE_FILE_ENV}."
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


def load_llm_registry(profile_file: str | None = None) -> LLMRegistry:
    text, source = _read_profile_source(profile_file)
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid LLM profile registry in {source}: root must be a mapping.")

    defaults = _require_mapping(payload.get("defaults"), label="defaults", source=source)
    backends_raw = _require_mapping(payload.get("backends"), label="backends", source=source)
    profiles_raw = _require_mapping(payload.get("profiles"), label="profiles", source=source)

    backends: dict[str, BackendConfig] = {}
    for name, raw in backends_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid backend {name!r} in {source}: backend config must be a mapping.")
        kind = str(raw.get("type") or "").strip().lower()
        if kind not in SUPPORTED_BACKEND_TYPES:
            supported = ", ".join(sorted(SUPPORTED_BACKEND_TYPES))
            raise ValueError(
                f"Invalid backend {name!r} in {source}: type must be one of {supported}."
            )
        env_raw = _require_mapping(raw.get("env"), label=f"backend {name!r}.env", source=source)
        env: dict[str, BackendEnvVar] = {}
        for env_name, env_spec in env_raw.items():
            if not isinstance(env_spec, dict):
                raise ValueError(
                    f"Invalid backend {name!r} env {env_name!r} in {source}: env entry must be a mapping."
                )
            env_var = str(env_spec.get("var") or "").strip()
            if not env_var:
                raise ValueError(
                    f"Invalid backend {name!r} env {env_name!r} in {source}: missing var."
                )
            env[str(env_name)] = BackendEnvVar(
                name=str(env_name),
                env_var=env_var,
                required=_coerce_bool(env_spec.get("required"), default=False),
            )
        backends[str(name)] = BackendConfig(name=str(name), kind=kind, env=env)

    normalized_profiles: dict[str, dict[str, Any]] = {}
    for name, raw in profiles_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid profile {name!r} in {source}: profile config must be a mapping.")
        normalized_profiles[str(name)] = dict(raw)

    return LLMRegistry(
        defaults={str(key): str(value) for key, value in defaults.items() if value is not None},
        backends=backends,
        profiles=normalized_profiles,
    )


def default_llm_profile_name(kind: str = "default", profile_file: str | None = None) -> str:
    registry = load_llm_registry(profile_file)
    if kind in registry.defaults:
        return registry.defaults[kind]
    if "default" in registry.defaults:
        return registry.defaults["default"]
    if len(registry.profiles) == 1:
        return next(iter(registry.profiles))
    raise KeyError(f"No default LLM profile configured for {kind!r}.")


def _merged_profile_map(
    registry: LLMRegistry,
    profile_name: str,
    *,
    stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if profile_name in stack:
        chain = " -> ".join((*stack, profile_name))
        raise ValueError(f"Cyclic LLM profile extends chain: {chain}")
    try:
        current = registry.profiles[profile_name]
    except KeyError as exc:
        raise KeyError(f"Unknown LLM profile: {profile_name}") from exc

    base_name = current.get("extends")
    current_without_extends = {key: value for key, value in current.items() if key != "extends"}
    if base_name is None:
        return dict(current_without_extends)

    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError(f"Invalid extends value for LLM profile {profile_name!r}.")
    merged = _merged_profile_map(registry, base_name.strip(), stack=(*stack, profile_name))
    return _deep_merge(merged, current_without_extends)


def resolve_llm_profile(
    profile_name: str | None,
    *,
    default_for: str | None = None,
    profile_file: str | None = None,
) -> LLMProfile:
    registry = load_llm_registry(profile_file)
    selected = (profile_name or "").strip() or default_llm_profile_name(default_for or "default", profile_file)
    raw = _merged_profile_map(registry, selected)

    backend_name = str(raw.get("backend") or "").strip()
    if not backend_name:
        raise ValueError(f"LLM profile {selected!r} is missing backend.")
    if backend_name not in registry.backends:
        raise ValueError(f"LLM profile {selected!r} references unknown backend {backend_name!r}.")
    backend = registry.backends[backend_name]

    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError(f"LLM profile {selected!r} is missing model.")

    api_mode = str(raw.get("api") or "").strip().lower()
    if api_mode not in {"responses", "completions"}:
        raise ValueError(
            f"LLM profile {selected!r} must set api to 'responses' or 'completions'."
        )

    limits = _require_mapping(raw.get("limits"), label=f"profile {selected!r}.limits", source=selected)
    reasoning = _require_mapping(raw.get("reasoning"), label=f"profile {selected!r}.reasoning", source=selected)
    features = _require_mapping(raw.get("features"), label=f"profile {selected!r}.features", source=selected)
    sampling = _require_mapping(raw.get("sampling"), label=f"profile {selected!r}.sampling", source=selected)

    max_tokens_raw = limits.get("max_completion_tokens", 32768)
    context_window_raw = limits.get("context_window")
    temperature_raw = sampling.get("temperature")

    max_tokens = int(max_tokens_raw)
    context_window = int(context_window_raw) if context_window_raw is not None else None
    if context_window is not None and context_window <= max_tokens:
        raise ValueError(
            f"LLM profile {selected!r} has context_window={context_window}, "
            f"which must be greater than max_completion_tokens={max_tokens}."
        )

    return LLMProfile(
        name=selected,
        backend_name=backend_name,
        provider=backend.kind,
        model=model,
        api_mode=api_mode,
        reasoning_effort=(
            str(reasoning["effort"]) if reasoning.get("effort") is not None else None
        ),
        reasoning_summary=(
            str(reasoning["summary"]) if reasoning.get("summary") is not None else None
        ),
        web_search=_coerce_bool(features.get("web_search"), default=False),
        max_tokens=max_tokens,
        context_window=context_window,
        use_phase=_coerce_bool(features.get("use_phase"), default=False),
        temperature=float(temperature_raw) if temperature_raw is not None else None,
        clear_thinking=(
            _coerce_bool(features.get("clear_thinking"))
            if features.get("clear_thinking") is not None
            else None
        ),
        backend_env=backend.env,
    )


def required_backend_env_vars(profile: LLMProfile) -> list[str]:
    required: list[str] = []
    for spec in (profile.backend_env or {}).values():
        if spec.required:
            required.append(spec.env_var)
    return required


def missing_backend_env_vars(profile: LLMProfile) -> list[str]:
    return [env_var for env_var in required_backend_env_vars(profile) if not os.environ.get(env_var)]


def backend_env_values(profile: LLMProfile) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, spec in (profile.backend_env or {}).items():
        value = os.environ.get(spec.env_var)
        if value:
            values[name] = value
    return values


def llm_env(profile_name: str | None, *, default_for: str | None = None, profile_file: str | None = None) -> dict[str, str]:
    profile = resolve_llm_profile(profile_name, default_for=default_for, profile_file=profile_file)
    env = {
        "AISCI_LLM_PROFILE": profile.name,
        "AISCI_PROVIDER": profile.provider,
        "AISCI_MODEL": profile.model,
        "AISCI_API_MODE": profile.api_mode,
        "AISCI_WEB_SEARCH": "true" if profile.web_search else "false",
        "AISCI_MAX_TOKENS": str(profile.max_tokens),
    }
    if profile.reasoning_effort:
        env["AISCI_REASONING_EFFORT"] = profile.reasoning_effort
    if profile.reasoning_summary:
        env["AISCI_REASONING_SUMMARY"] = profile.reasoning_summary
    if profile.context_window is not None:
        env["AISCI_CONTEXT_WINDOW"] = str(profile.context_window)
    if profile.use_phase:
        env["AISCI_USE_PHASE"] = "true"
    if profile.temperature is not None:
        env["AISCI_TEMPERATURE"] = str(profile.temperature)
    if profile.clear_thinking is not None:
        env["AISCI_CLEAR_THINKING"] = "true" if profile.clear_thinking else "false"
    for spec in (profile.backend_env or {}).values():
        value = os.environ.get(spec.env_var)
        if value:
            env[spec.env_var] = value
    return env
