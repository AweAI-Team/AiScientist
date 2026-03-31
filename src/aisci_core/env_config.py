from __future__ import annotations

import os
from pathlib import Path

from aisci_core.paths import repo_root

DEFAULT_ENV_FILENAMES = (".env", ".env.aisci", ".env.local")


def _candidate_env_files(explicit_path: str | None = None) -> list[Path]:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return [candidate]

    roots: list[Path] = []
    for root in (repo_root(), Path.cwd().resolve()):
        if root not in roots:
            roots.append(root)

    candidates: list[Path] = []
    for root in roots:
        for filename in DEFAULT_ENV_FILENAMES:
            path = (root / filename).resolve()
            if path not in candidates:
                candidates.append(path)
    return candidates


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].lstrip()

    key, sep, value = text.partition("=")
    if not sep:
        return None

    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = bytes(value, "utf-8").decode("unicode_escape")
    else:
        comment_idx = value.find(" #")
        if comment_idx >= 0:
            value = value[:comment_idx].rstrip()

    return key, value


def load_runtime_env(explicit_path: str | None = None, *, override: bool = False) -> list[Path]:
    loaded: list[Path] = []
    existing_keys = set(os.environ)
    for path in _candidate_env_files(explicit_path):
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _parse_env_assignment(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in existing_keys:
                os.environ[key] = value
        loaded.append(path)
    return loaded
