from __future__ import annotations

import importlib
import importlib.util
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


def _module_path_from_repo_root(module_name: str, repo_root: Path) -> Path | None:
    parts = module_name.split(".")
    file_candidate = repo_root.joinpath(*parts).with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    package_candidate = repo_root.joinpath(*parts) / "__init__.py"
    if package_candidate.is_file():
        return package_candidate
    return None


def _import_module_from_repo_path(module_name: str, *, repo_root: Path):
    module_path = _module_path_from_repo_root(module_name, repo_root)
    if module_path is None:
        raise ModuleNotFoundError(module_name)
    synthetic_name = f"aisci_dynamic_{module_name.replace('-', '_').replace('.', '_')}"
    existing = sys.modules.get(synthetic_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(synthetic_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[synthetic_name] = module
    spec.loader.exec_module(module)
    return module


def import_mlebench_callable(import_string: str, *, repo_root: Path) -> Callable:
    module_name, fn_name = import_string.split(":")
    ensure_mlebench_import_root(repo_root)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = _import_module_from_repo_path(module_name, repo_root=repo_root)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(f"import target is not callable: {import_string}")
    return fn


def import_mlebench_module(module_name: str, *, repo_root: Path):
    ensure_mlebench_import_root(repo_root)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return _import_module_from_repo_path(module_name, repo_root=repo_root)
