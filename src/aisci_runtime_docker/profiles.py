from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aisci_core.models import PullPolicy
from aisci_core.paths import repo_root
from aisci_runtime_docker.models import DockerProfile

DEFAULT_IMAGE_PROFILE_FILE_ENV = "AISCI_IMAGE_PROFILE_FILE"
DEFAULT_IMAGE_PROFILE_PATH = Path("config") / "image_profiles.yaml"


@dataclass(frozen=True)
class ImageRegistry:
    defaults: dict[str, str]
    profiles: dict[str, DockerProfile]


def _resolve_profile_path(profile_file: str | None = None) -> Path:
    configured = (profile_file or os.environ.get(DEFAULT_IMAGE_PROFILE_FILE_ENV, "")).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    return (repo_root() / DEFAULT_IMAGE_PROFILE_PATH).resolve()


def resolved_image_profile_path(profile_file: str | None = None) -> Path:
    return _resolve_profile_path(profile_file)


def _read_profile_source(profile_file: str | None = None) -> tuple[str, str]:
    path = _resolve_profile_path(profile_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Image profile file not found: {path}. "
            f"Create {DEFAULT_IMAGE_PROFILE_PATH} in the repo root or set {DEFAULT_IMAGE_PROFILE_FILE_ENV}."
        )
    return path.read_text(encoding="utf-8"), str(path)


def _require_mapping(value: Any, *, label: str, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label} in {source}: expected a mapping.")
    return dict(value)


def _parse_pull_policy(value: Any, *, profile_name: str, source: str) -> PullPolicy:
    raw = str(value or PullPolicy.IF_MISSING.value).strip().lower().replace("_", "-")
    try:
        return PullPolicy(raw)
    except ValueError as exc:
        supported = ", ".join(policy.value for policy in PullPolicy)
        raise ValueError(
            f"Invalid pull_policy for image profile {profile_name!r} in {source}: "
            f"expected one of {supported}."
        ) from exc


def load_image_registry(profile_file: str | None = None) -> ImageRegistry:
    text, source = _read_profile_source(profile_file)
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid image profile registry in {source}: root must be a mapping.")

    defaults = _require_mapping(payload.get("defaults"), label="defaults", source=source)
    profiles_raw = _require_mapping(payload.get("profiles"), label="profiles", source=source)

    profiles: dict[str, DockerProfile] = {}
    for name, raw in profiles_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid image profile {name!r} in {source}: profile config must be a mapping."
            )
        image = str(raw.get("image") or "").strip()
        if not image:
            raise ValueError(f"Image profile {name!r} in {source} is missing image.")
        profiles[str(name)] = DockerProfile(
            name=str(name),
            image=image,
            pull_policy=_parse_pull_policy(raw.get("pull_policy"), profile_name=str(name), source=source),
        )

    return ImageRegistry(
        defaults={str(key): str(value) for key, value in defaults.items() if value is not None},
        profiles=profiles,
    )


def default_image_profile_name(kind: str = "default", profile_file: str | None = None) -> str:
    registry = load_image_registry(profile_file)
    if kind in registry.defaults:
        return registry.defaults[kind]
    if "default" in registry.defaults:
        return registry.defaults["default"]
    if len(registry.profiles) == 1:
        return next(iter(registry.profiles))
    raise KeyError(f"No default image profile configured for {kind!r}.")


def resolve_image_profile(
    profile_name: str | None,
    *,
    default_for: str | None = None,
    profile_file: str | None = None,
) -> DockerProfile:
    registry = load_image_registry(profile_file)
    selected = (profile_name or "").strip() or default_image_profile_name(default_for or "default", profile_file)
    try:
        return registry.profiles[selected]
    except KeyError as exc:
        raise KeyError(f"Unknown image profile: {selected}") from exc


def default_paper_profile(profile_name: str | None = None) -> DockerProfile:
    return resolve_image_profile(profile_name, default_for="paper")


def default_mle_profile(profile_name: str | None = None) -> DockerProfile:
    return resolve_image_profile(profile_name, default_for="mle")
