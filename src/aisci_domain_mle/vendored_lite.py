from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable


VENDORED_MLEBENCH_LITE_ROOT = (
    Path(__file__).resolve().parent / "vendored_mlebench_lite"
).resolve()

_ACTIVE_MLEBENCH_IMPORT_ROOT: Path | None = None


def vendored_lite_repo_root() -> Path:
    return VENDORED_MLEBENCH_LITE_ROOT


def vendored_lite_competitions_dir() -> Path:
    return vendored_lite_repo_root() / "mlebench" / "competitions"


def vendored_lite_splits_dir() -> Path:
    return vendored_lite_repo_root() / "experiments" / "splits"


def vendored_lite_competition_ids() -> tuple[str, ...]:
    split_path = vendored_lite_splits_dir() / "low.txt"
    return tuple(
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def is_vendored_lite_competition(competition_name: str) -> bool:
    return competition_name in set(vendored_lite_competition_ids())


def vendored_lite_competition_dir(competition_name: str) -> Path:
    return (vendored_lite_competitions_dir() / competition_name).resolve()


def _purge_mlebench_modules() -> None:
    for name in list(sys.modules):
        if name == "mlebench" or name.startswith("mlebench."):
            del sys.modules[name]


def ensure_mlebench_import_root(repo_root: Path) -> Path:
    global _ACTIVE_MLEBENCH_IMPORT_ROOT

    resolved = Path(repo_root).expanduser().resolve()
    if _ACTIVE_MLEBENCH_IMPORT_ROOT != resolved:
        _purge_mlebench_modules()
        _ACTIVE_MLEBENCH_IMPORT_ROOT = resolved

    root_text = str(resolved)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    return resolved


def import_mlebench_callable(import_string: str, *, repo_root: Path) -> Callable:
    module_name, fn_name = import_string.split(":")
    ensure_mlebench_import_root(repo_root)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(f"import target is not callable: {import_string}")
    return fn


def import_mlebench_module(module_name: str, *, repo_root: Path):
    ensure_mlebench_import_root(repo_root)
    return importlib.import_module(module_name)
