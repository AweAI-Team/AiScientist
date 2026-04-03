from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

import yaml

from aisci_domain_mle.contracts import InputSourceKind, LegacyPreparePlan, ResolvedInputState
from aisci_domain_mle.vendored_lite import (
    import_mlebench_callable,
    is_vendored_lite_competition,
    vendored_lite_competition_ids,
    vendored_lite_repo_root,
)


AISCI_REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_MLEBENCH_REPO_ENV = "AISCI_MLEBENCH_REPO"
HISTORICAL_LEGACY_MLEBENCH_REPO = Path("/home/chenjie.survivi/aisci-0331/mle-bench")
DEFAULT_LEGACY_MLEBENCH_REPO = vendored_lite_repo_root()


def _normalize_repo_candidate(raw_path: Path | str | None) -> Path | None:
    if raw_path is None:
        return None
    return Path(raw_path).expanduser().resolve()


def _legacy_mlebench_repo_candidates(repo_root: Path | None = None) -> list[Path]:
    if repo_root is not None:
        return [_normalize_repo_candidate(repo_root)]

    candidates: list[Path] = []
    configured = os.environ.get(LEGACY_MLEBENCH_REPO_ENV)
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            vendored_lite_repo_root(),
            (AISCI_REPO_ROOT.parent / "mle-bench"),
            HISTORICAL_LEGACY_MLEBENCH_REPO,
        )
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = _normalize_repo_candidate(candidate)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _is_legacy_mlebench_repo_root(candidate: Path) -> bool:
    return (
        (candidate / "mlebench" / "cli.py").is_file()
        or (candidate / "mlebench" / "competitions").is_dir()
    )


def find_legacy_mlebench_repo_root(repo_root: Path | None = None) -> Path | None:
    for candidate in _legacy_mlebench_repo_candidates(repo_root):
        if _is_legacy_mlebench_repo_root(candidate):
            return candidate
    return None


def resolve_legacy_mlebench_repo_root(repo_root: Path | None = None) -> Path:
    resolved = find_legacy_mlebench_repo_root(repo_root)
    if resolved is not None:
        return resolved

    attempted = ", ".join(str(path) for path in _legacy_mlebench_repo_candidates(repo_root))
    raise ValueError(
        "No usable MLE competition registry was found. AiScientist ships a built-in "
        "MLE-bench Lite registry, and you can optionally point to a broader external "
        f"registry with {LEGACY_MLEBENCH_REPO_ENV}=/path/to/mle-bench. "
        f"Looked for: {attempted}"
    )


def _uses_vendored_lite_registry(repo_root: Path) -> bool:
    return repo_root == vendored_lite_repo_root()


def _unsupported_vendored_competition_message(competition_name: str) -> str:
    supported = ", ".join(vendored_lite_competition_ids())
    return (
        f"competition {competition_name!r} is not included in AiScientist's built-in "
        "MLE-bench Lite registry. Supported competitions are: "
        f"{supported}. Configure {LEGACY_MLEBENCH_REPO_ENV} to use an external full "
        "mle-bench checkout if you need broader coverage."
    )


def _ensure_optional_legacy_import_stubs() -> None:
    if "py7zr" not in sys.modules:
        sys.modules["py7zr"] = types.ModuleType("py7zr")


class LegacyMlebenchDownloader:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        python_executable: str | None = None,
        runner: Callable[[LegacyPreparePlan], None] | None = None,
    ) -> None:
        self._repo_root_override = repo_root
        self.python_executable = python_executable or sys.executable
        self._runner = runner or self._default_runner

    @property
    def repo_root(self) -> Path:
        return resolve_legacy_mlebench_repo_root(self._repo_root_override)

    def build_prepare_plan(self, competition_name: str, data_dir: Path) -> LegacyPreparePlan:
        if (
            _uses_vendored_lite_registry(self.repo_root)
            and not is_vendored_lite_competition(competition_name)
        ):
            raise ValueError(_unsupported_vendored_competition_message(competition_name))

        cli_path = self.repo_root / "mlebench" / "cli.py"
        if not cli_path.is_file():
            raise ValueError(f"legacy mlebench cli is missing at {cli_path}")

        if _uses_vendored_lite_registry(self.repo_root):
            command = [
                self.python_executable,
                "-m",
                "aisci_domain_mle.vendored_lite_cli",
                "prepare",
                "--competition-id",
                competition_name,
                "--data-dir",
                str(data_dir),
                "--keep-raw",
            ]
            env: dict[str, str] = {}
            module = "aisci_domain_mle.vendored_lite_cli"
            cwd = str(AISCI_REPO_ROOT)
        else:
            command = [
                self.python_executable,
                "-m",
                "mlebench.cli",
                "prepare",
                "--competition-id",
                competition_name,
                "--data-dir",
                str(data_dir),
                "--keep-raw",
            ]
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{self.repo_root}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else str(self.repo_root)
            )
            module = "mlebench.cli"
            cwd = str(self.repo_root)
        return LegacyPreparePlan(
            competition_name=competition_name,
            data_dir=str(data_dir),
            python_executable=self.python_executable,
            module=module,
            cwd=cwd,
            command=command,
            env={key: value for key, value in env.items() if key == "PYTHONPATH"},
        )

    def prepare(self, competition_name: str, data_dir: Path) -> LegacyPreparePlan:
        plan = self.build_prepare_plan(competition_name, data_dir)
        self._runner(plan)
        return plan

    def _default_runner(self, plan: LegacyPreparePlan) -> None:
        env = dict(os.environ)
        env.update(plan.env)
        subprocess.run(
            plan.command,
            cwd=plan.cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )


class LegacyCompetitionPreparer:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self._repo_root_override = repo_root

    @property
    def repo_root(self) -> Path:
        return resolve_legacy_mlebench_repo_root(self._repo_root_override)

    def competition_dir(self, competition_name: str) -> Path:
        competition_dir = self.repo_root / "mlebench" / "competitions" / competition_name
        if not competition_dir.is_dir():
            if _uses_vendored_lite_registry(self.repo_root):
                raise ValueError(_unsupported_vendored_competition_message(competition_name))
            raise ValueError(f"legacy competition is missing at {competition_dir}")
        return competition_dir

    def _competition_config(self, competition_name: str) -> dict[str, Any]:
        competition_dir = self.competition_dir(competition_name)
        config_path = competition_dir / "config.yaml"
        if not config_path.is_file():
            raise ValueError(f"legacy competition config is missing at {config_path}")
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    def prepare_local_dataset(
        self,
        competition_name: str,
        *,
        raw_dir: Path,
        public_dir: Path,
        private_dir: Path,
    ) -> None:
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        _ensure_optional_legacy_import_stubs()
        payload = self._competition_config(competition_name)
        import_string = str(payload.get("preparer") or "").strip()
        if not import_string:
            raise ValueError(f"legacy competition preparer is missing for {competition_name}")
        prepare_fn = import_mlebench_callable(import_string, repo_root=self.repo_root)
        prepare_fn(
            raw=raw_dir.resolve(),
            public=public_dir.resolve(),
            private=private_dir.resolve(),
        )

    def resolve_public_metadata_paths(
        self,
        competition_name: str,
        *,
        prepared_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        payload = self._competition_config(competition_name)
        description_raw = str(payload.get("description") or "").strip()
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
        sample_raw = str(dataset.get("sample_submission") or "").strip()

        description_path = (
            (self.repo_root / description_raw).resolve()
            if description_raw
            else None
        )
        sample_submission_path = self._resolve_prepared_dataset_path(
            sample_raw,
            prepared_dir=prepared_dir.resolve(),
        )
        if sample_submission_path is not None and "public" not in sample_submission_path.parts:
            sample_submission_path = None
        return (
            description_path if description_path and description_path.is_file() else None,
            sample_submission_path if sample_submission_path and sample_submission_path.is_file() else None,
        )

    def _resolve_prepared_dataset_path(
        self,
        raw_path: str,
        *,
        prepared_dir: Path,
    ) -> Path | None:
        if not raw_path:
            return None
        relative_path = Path(raw_path)
        parts = relative_path.parts
        if "prepared" not in parts:
            return None
        prepared_index = parts.index("prepared")
        if prepared_index + 1 >= len(parts):
            return None
        relative_under_prepared = Path(*parts[prepared_index + 1 :])
        return (prepared_dir / relative_under_prepared).resolve()


@dataclass(frozen=True)
class LegacyCompetitionGradeSpec:
    competition_name: str
    answers_path: Path
    sample_submission_path: Path
    leaderboard_path: Path
    grade_fn_import: str


class LegacyCompetitionGrader:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self._repo_root_override = repo_root

    @property
    def repo_root(self) -> Path:
        return resolve_legacy_mlebench_repo_root(self._repo_root_override)

    def competition_dir(self, competition_name: str) -> Path:
        competition_dir = self.repo_root / "mlebench" / "competitions" / competition_name
        if not competition_dir.is_dir():
            if _uses_vendored_lite_registry(self.repo_root):
                raise ValueError(_unsupported_vendored_competition_message(competition_name))
            raise ValueError(f"legacy competition is missing at {competition_dir}")
        return competition_dir

    def load_grade_spec(
        self,
        competition_name: str,
        *,
        cache_root: Path,
    ) -> LegacyCompetitionGradeSpec:
        competition_dir = self.competition_dir(competition_name)
        config_path = competition_dir / "config.yaml"
        if not config_path.is_file():
            raise ValueError(f"legacy competition config is missing at {config_path}")

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
        grader = payload.get("grader") if isinstance(payload.get("grader"), dict) else {}
        answers_raw = str(dataset.get("answers") or "").strip()
        sample_raw = str(dataset.get("sample_submission") or "").strip()
        grade_fn_import = str(grader.get("grade_fn") or "").strip()
        if not answers_raw or not sample_raw or not grade_fn_import:
            raise ValueError(f"legacy grading metadata is incomplete for {competition_name}")

        return LegacyCompetitionGradeSpec(
            competition_name=competition_name,
            answers_path=(cache_root / answers_raw).resolve(),
            sample_submission_path=(cache_root / sample_raw).resolve(),
            leaderboard_path=(competition_dir / "leaderboard.csv").resolve(),
            grade_fn_import=grade_fn_import,
        )

    def grade_submission(
        self,
        submission_path: Path,
        *,
        competition_name: str,
        cache_root: Path,
    ) -> dict[str, Any]:
        spec = self.load_grade_spec(competition_name, cache_root=cache_root.resolve())
        leaderboard = self._read_csv(spec.leaderboard_path)
        thresholds = self._rank_thresholds(leaderboard)

        if not submission_path.is_file():
            return {
                **thresholds,
                "competition_id": competition_name,
                "score": None,
                "submission_exists": False,
                "valid_submission": False,
                "error": f"submission file does not exist: {submission_path}",
                "created_at": datetime.now().isoformat(),
            }
        if submission_path.suffix.lower() != ".csv":
            return {
                **thresholds,
                "competition_id": competition_name,
                "score": None,
                "submission_exists": False,
                "valid_submission": False,
                "error": f"submission file must be a CSV: {submission_path}",
                "created_at": datetime.now().isoformat(),
            }
        if not spec.answers_path.is_file():
            raise ValueError(
                f"legacy grading answers are missing for {competition_name}: {spec.answers_path}"
            )
        if not spec.sample_submission_path.is_file():
            raise ValueError(
                "legacy grading sample submission is missing for "
                f"{competition_name}: {spec.sample_submission_path}"
            )

        submission_df = self._read_csv(submission_path)
        answers = self._load_answers(spec.answers_path)
        grade_fn = self._load_grade_function(spec.grade_fn_import)
        try:
            score = round(float(grade_fn(submission_df, answers)), 5)
            rank_info = self._rank_score(score, leaderboard)
            return {
                **rank_info,
                "competition_id": competition_name,
                "score": score,
                "submission_exists": True,
                "valid_submission": True,
                "error": None,
                "created_at": datetime.now().isoformat(),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                **thresholds,
                "competition_id": competition_name,
                "score": None,
                "submission_exists": True,
                "valid_submission": False,
                "error": str(exc),
                "created_at": datetime.now().isoformat(),
            }

    def _read_csv(self, path: Path):
        import pandas as pd

        return pd.read_csv(path, float_precision="round_trip")

    def _load_answers(self, path: Path) -> Any:
        if path.suffix.lower() == ".csv":
            return self._read_csv(path)
        if path.suffix.lower() == ".jsonl":
            return self._read_jsonl(path)
        raise ValueError(f"unsupported answers format for legacy grading: {path}")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if cleaned:
                    rows.append(json.loads(cleaned))
        return rows

    def _load_grade_function(self, import_string: str) -> Callable[[Any, Any], Any]:
        _ensure_optional_legacy_import_stubs()
        return import_mlebench_callable(import_string, repo_root=self.repo_root)

    def _rank_thresholds(self, leaderboard) -> dict[str, Any]:
        if "score" not in leaderboard.columns:
            raise ValueError("legacy leaderboard is missing required `score` column")
        scores = leaderboard["score"]
        lower_is_better = bool(scores.iloc[0] < scores.iloc[-1])
        num_teams = len(scores)

        def score_at(position: int) -> float:
            if position < 1 or position > len(scores):
                raise IndexError("leaderboard position is out of bounds")
            return float(scores.iloc[position - 1])

        if 1 <= num_teams < 100:
            gold_threshold = score_at(max(1, int(num_teams * 0.1)))
            silver_threshold = score_at(max(1, int(num_teams * 0.2)))
            bronze_threshold = score_at(max(1, int(num_teams * 0.4)))
        elif 100 <= num_teams < 250:
            gold_threshold = score_at(10)
            silver_threshold = score_at(max(1, int(num_teams * 0.2)))
            bronze_threshold = score_at(max(1, int(num_teams * 0.4)))
        elif 250 <= num_teams < 1000:
            gold_threshold = score_at(10 + int(num_teams * 0.002))
            silver_threshold = score_at(50)
            bronze_threshold = score_at(100)
        elif num_teams >= 1000:
            gold_threshold = score_at(10 + int(num_teams * 0.002))
            silver_threshold = score_at(max(1, int(num_teams * 0.05)))
            bronze_threshold = score_at(max(1, int(num_teams * 0.1)))
        else:
            raise ValueError("legacy leaderboard is empty")

        return {
            "gold_threshold": gold_threshold,
            "silver_threshold": silver_threshold,
            "bronze_threshold": bronze_threshold,
            "median_threshold": float(scores.median()),
            "is_lower_better": lower_is_better,
            "gold_medal": False,
            "silver_medal": False,
            "bronze_medal": False,
            "above_median": False,
            "any_medal": False,
        }

    def _rank_score(self, score: float, leaderboard) -> dict[str, Any]:
        thresholds = self._rank_thresholds(leaderboard)
        lower_is_better = bool(thresholds["is_lower_better"])
        gold_medal = score <= thresholds["gold_threshold"] if lower_is_better else score >= thresholds["gold_threshold"]
        silver_medal = not gold_medal and (
            score <= thresholds["silver_threshold"] if lower_is_better else score >= thresholds["silver_threshold"]
        )
        bronze_medal = (
            not gold_medal
            and not silver_medal
            and (score <= thresholds["bronze_threshold"] if lower_is_better else score >= thresholds["bronze_threshold"])
        )
        above_median = score < thresholds["median_threshold"] if lower_is_better else score > thresholds["median_threshold"]
        return {
            **thresholds,
            "gold_medal": gold_medal,
            "silver_medal": silver_medal,
            "bronze_medal": bronze_medal,
            "above_median": above_median,
            "any_medal": bool(gold_medal or silver_medal or bronze_medal),
        }


def resolve_competition_source(
    *,
    competition_name: str | None,
    competition_zip_path: str | None,
    cache_root: Path,
    allow_download: bool = False,
    downloader: LegacyMlebenchDownloader | None = None,
) -> ResolvedInputState:
    cache_root = cache_root.expanduser().resolve()

    if competition_zip_path:
        zip_path = Path(competition_zip_path).expanduser().resolve()
        if not zip_path.exists():
            raise ValueError(f"competition zip path does not exist: {zip_path}")
        return ResolvedInputState(
            source_kind=InputSourceKind.LOCAL_ZIP,
            competition_name=competition_name,
            competition_zip_path=str(zip_path),
            zip_exists=True,
            cache_root=str(cache_root),
        )

    cleaned_name = (competition_name or "").strip()
    if not cleaned_name:
        raise ValueError("competition name is required when no local zip path is provided")

    cache_dir = cache_root / cleaned_name
    prepared_dir = cache_dir / "prepared"
    cache_dir_exists = cache_dir.exists()
    cache_prepared_exists = prepared_dir.exists()

    if cache_prepared_exists:
        return ResolvedInputState(
            source_kind=InputSourceKind.COMPETITION_NAME,
            competition_name=cleaned_name,
            cache_root=str(cache_root),
            cache_dir=str(cache_dir),
            cache_dir_exists=cache_dir_exists,
            cache_prepared_dir=str(prepared_dir),
            cache_prepared_exists=True,
        )

    if downloader is None:
        downloader = LegacyMlebenchDownloader()
    prepare_plan = downloader.build_prepare_plan(cleaned_name, cache_root)

    if allow_download:
        downloader.prepare(cleaned_name, cache_root)
        cache_dir_exists = cache_dir.exists()
        cache_prepared_exists = prepared_dir.exists()
        if not cache_prepared_exists:
            raise ValueError(
                f"legacy mlebench prepare path ran but did not create prepared cache for {cleaned_name}"
            )

    return ResolvedInputState(
        source_kind=InputSourceKind.COMPETITION_NAME,
        competition_name=cleaned_name,
        cache_root=str(cache_root),
        cache_dir=str(cache_dir),
        cache_dir_exists=cache_dir_exists,
        cache_prepared_dir=str(prepared_dir),
        cache_prepared_exists=cache_prepared_exists,
        legacy_prepare_plan=None if cache_prepared_exists else prepare_plan,
    )
