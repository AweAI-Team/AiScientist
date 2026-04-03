from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")

    def Field(default=None, default_factory=None, **kwargs):
        del kwargs
        if default_factory is not None:
            return default_factory()
        return default

    def model_validator(*args, **kwargs):
        del args, kwargs

        def decorator(func):
            func._is_model_validator = True
            return func

        return decorator

    class BaseModel:
        def __init__(self, **data):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in data:
                    value = data[name]
                else:
                    value = copy.deepcopy(getattr(self.__class__, name, None))
                setattr(self, name, value)
            for name, value in data.items():
                if name not in annotations:
                    setattr(self, name, value)
            for name in dir(self):
                attr = getattr(self, name)
                if callable(attr) and getattr(attr, "_is_model_validator", False):
                    attr()

        @classmethod
        def model_validate(cls, data):
            return cls(**data)

        def model_dump(self, mode=None):
            del mode
            return dict(self.__dict__)

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.model_validator = model_validator
    sys.modules["pydantic"] = pydantic_stub

if "structlog" not in sys.modules:
    structlog_stub = types.ModuleType("structlog")

    class _Logger:
        def debug(self, *args, **kwargs):
            del args, kwargs

        info = warning = error = exception = debug

    class _Stdlib:
        @staticmethod
        def get_logger(*args, **kwargs):
            del args, kwargs
            return _Logger()

    structlog_stub.stdlib = _Stdlib()
    sys.modules["structlog"] = structlog_stub

from aisci_domain_mle.adapter import MLEDomainAdapter


class _FakeCompetitionPreparer:
    def __init__(self, description_path: Path | None = None) -> None:
        self.description_path = description_path
        self.prepare_calls: list[str] = []

    def prepare_local_dataset(
        self,
        competition_name: str,
        *,
        raw_dir: Path,
        public_dir: Path,
        private_dir: Path,
    ) -> None:
        self.prepare_calls.append(competition_name)
        assert (raw_dir / "train.csv").is_file()
        assert (raw_dir / "test_with_solutions.csv").is_file()
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        (public_dir / "sample_submission_public.csv").write_text(
            "id,target\n1,0\n",
            encoding="utf-8",
        )
        (private_dir / "answers.csv").write_text("id,target\n1,1\n", encoding="utf-8")
        (private_dir / "gold_submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")

    def resolve_public_metadata_paths(
        self,
        competition_name: str,
        *,
        prepared_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        del competition_name
        return self.description_path, prepared_dir / "public" / "sample_submission_public.csv"


class DataVisibilityAdapterTests(unittest.TestCase):
    def _spec(self, **overrides) -> SimpleNamespace:
        payload = {
            "competition_name": None,
            "competition_zip_path": None,
            "mlebench_data_dir": None,
            "workspace_bundle_zip": None,
            "competition_bundle_zip": None,
            "data_dir": None,
            "code_repo_zip": None,
            "description_path": None,
            "sample_submission_path": None,
            "validation_command": None,
            "grading_config_path": None,
        }
        payload.update(overrides)
        return SimpleNamespace(**payload)

    def _job_paths(self, root: Path) -> SimpleNamespace:
        input_dir = root / "input"
        workspace_dir = root / "workspace"
        logs_dir = root / "logs"
        artifacts_dir = root / "artifacts"
        state_dir = root / "state"
        input_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            input_dir=input_dir,
            workspace_dir=workspace_dir,
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            state_dir=state_dir,
        )

    def test_raw_zip_is_prepared_into_public_only_workspace_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "detecting-insults-in-social-commentary.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
                zf.writestr("test_with_solutions.csv", "id,target\n1,1\n")
                zf.writestr("impermium_verification_labels.csv", "label\n1\n")

            description_path = tmp_path / "description.md"
            description_path.write_text("# Detecting Insults\n", encoding="utf-8")
            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=_FakeCompetitionPreparer(description_path),
            )
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                workspace_bundle_zip=str(zip_path),
                validation_command="echo should-stay-host-side",
            )

            adapter._stage_inputs(spec, job_paths)

            self.assertEqual(
                (job_paths.workspace_dir / "data" / "train.csv").read_text(encoding="utf-8"),
                "x,y\n1,2\n",
            )
            self.assertTrue((job_paths.workspace_dir / "data" / "description.md").is_file())
            self.assertTrue((job_paths.workspace_dir / "data" / "sample_submission.csv").is_file())
            self.assertFalse((job_paths.workspace_dir / "data" / "answers.csv").exists())
            self.assertFalse((job_paths.workspace_dir / "data" / "gold_submission.csv").exists())
            self.assertFalse((job_paths.workspace_dir / "data" / "test_with_solutions.csv").exists())
            self.assertFalse((job_paths.workspace_dir / "data" / "eval_cmd.txt").exists())
            self.assertFalse((job_paths.input_dir / "workspace_bundle.zip").exists())
            self.assertTrue((job_paths.workspace_dir / "code" / "README.md").is_file())

    def test_competition_zip_path_is_prepared_into_public_only_workspace_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "detecting-insults-in-social-commentary.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
                zf.writestr("test_with_solutions.csv", "id,target\n1,1\n")

            description_path = tmp_path / "description.md"
            description_path.write_text("# Detecting Insults\n", encoding="utf-8")
            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=_FakeCompetitionPreparer(description_path),
            )
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_zip_path=str(zip_path),
                mlebench_data_dir=str(tmp_path / "cache"),
            )

            adapter._stage_inputs(spec, job_paths)

            self.assertTrue((job_paths.workspace_dir / "data" / "train.csv").is_file())
            self.assertFalse((job_paths.workspace_dir / "data" / "answers.csv").exists())
            self.assertFalse((job_paths.workspace_dir / "data" / "test_with_solutions.csv").exists())

    def test_explicit_competition_name_overrides_zip_basename_for_local_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "arbitrary-name.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
                zf.writestr("test_with_solutions.csv", "id,target\n1,1\n")

            description_path = tmp_path / "description.md"
            description_path.write_text("# Detecting Insults\n", encoding="utf-8")
            preparer = _FakeCompetitionPreparer()
            preparer.description_path = description_path
            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=preparer,
            )
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=str(zip_path),
            )

            adapter._stage_inputs(spec, job_paths)

            self.assertEqual(preparer.prepare_calls, ["detecting-insults-in-social-commentary"])
            self.assertTrue((job_paths.workspace_dir / "data" / "train.csv").is_file())
            self.assertFalse((job_paths.workspace_dir / "data" / "answers.csv").exists())

            with adapter._resolve_legacy_validation_target(spec) as target:
                assert target is not None
                self.assertEqual(target.competition_name, "detecting-insults-in-social-commentary")

    def test_vendored_lite_local_zip_stages_and_grades_without_external_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "arbitrary-name.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "Insult,Date,Comment\n0,2024-01-01,hello\n1,2024-01-02,go away\n")
                zf.writestr(
                    "test_with_solutions.csv",
                    "Insult,Date,Comment,Usage\n0,2024-01-03,nice,Private\n1,2024-01-04,awful,Private\n",
                )

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=str(zip_path),
            )

            adapter._stage_inputs(spec, job_paths)

            self.assertTrue((job_paths.workspace_dir / "data" / "train.csv").is_file())
            self.assertTrue((job_paths.workspace_dir / "data" / "description.md").is_file())
            self.assertTrue((job_paths.workspace_dir / "data" / "sample_submission.csv").is_file())
            self.assertFalse((job_paths.workspace_dir / "data" / "test_with_solutions.csv").exists())

            submission_path = tmp_path / "submission.csv"
            submission_path.write_text(
                (job_paths.workspace_dir / "data" / "sample_submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with adapter._resolve_legacy_validation_target(spec) as target:
                assert target is not None
                report = adapter._competition_grader.grade_submission(
                    submission_path,
                    competition_name=target.competition_name,
                    cache_root=target.cache_root,
                )

            self.assertTrue(report["submission_exists"])
            self.assertTrue(report["valid_submission"])
            self.assertIn("score", report)

    def test_local_zip_uses_env_discovered_legacy_repo_for_prepare_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_repo = tmp_path / "mle-bench"
            competition_name = "detecting-insults-in-social-commentary"
            competition_dir = legacy_repo / "mlebench" / "competitions" / competition_name
            competition_dir.mkdir(parents=True)
            (competition_dir / "prepare.py").write_text(
                "import shutil\n"
                "from pathlib import Path\n"
                "\n"
                "def prepare(raw: Path, public: Path, private: Path) -> None:\n"
                "    public.mkdir(parents=True, exist_ok=True)\n"
                "    private.mkdir(parents=True, exist_ok=True)\n"
                "    shutil.copy2(raw / 'train.csv', public / 'train.csv')\n"
                "    (public / 'sample_submission.csv').write_text('id,target\\n1,0\\n', encoding='utf-8')\n"
                "    (private / 'answers.csv').write_text('id,target\\n1,1\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            (competition_dir / "config.yaml").write_text(
                "preparer: mlebench.competitions.detecting-insults-in-social-commentary.prepare:prepare\n"
                "description: mlebench/competitions/detecting-insults-in-social-commentary/description.md\n"
                "dataset:\n"
                "  sample_submission: detecting-insults-in-social-commentary/prepared/public/sample_submission.csv\n",
                encoding="utf-8",
            )
            (competition_dir / "description.md").write_text("# Detecting Insults\n", encoding="utf-8")

            zip_path = tmp_path / "arbitrary-name.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
                zf.writestr("test_with_solutions.csv", "id,target\n1,1\n")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_name=competition_name,
                competition_zip_path=str(zip_path),
            )

            with mock.patch.dict("os.environ", {"AISCI_MLEBENCH_REPO": str(legacy_repo)}, clear=False):
                adapter._stage_inputs(spec, job_paths)

                self.assertTrue((job_paths.workspace_dir / "data" / "train.csv").is_file())
                self.assertTrue((job_paths.workspace_dir / "data" / "description.md").is_file())
                self.assertTrue((job_paths.workspace_dir / "data" / "sample_submission.csv").is_file())
                self.assertFalse((job_paths.workspace_dir / "data" / "answers.csv").exists())

                with adapter._resolve_legacy_validation_target(spec) as target:
                    assert target is not None
                    self.assertEqual(target.competition_name, competition_name)
                    self.assertTrue((target.prepared_dir / "private" / "answers.csv").is_file())
                    self.assertFalse(str(target.prepared_dir).startswith(str(job_paths.workspace_dir)))

    def test_local_zip_requires_public_metadata_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "detecting-insults-in-social-commentary.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
                zf.writestr("test_with_solutions.csv", "id,target\n1,1\n")

            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=_FakeCompetitionPreparer(),
            )
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(competition_zip_path=str(zip_path))

            with self.assertRaisesRegex(ValueError, "missing description.md"):
                adapter._stage_inputs(spec, job_paths)

    def test_cache_hit_requires_public_metadata_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cache_root = tmp_path / "cache"
            public_dir = cache_root / "demo-competition" / "prepared" / "public"
            private_dir = cache_root / "demo-competition" / "prepared" / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (private_dir / "answers.csv").write_text("id,target\n1,1\n", encoding="utf-8")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_name="demo-competition",
                mlebench_data_dir=str(cache_root),
            )

            with self.assertRaisesRegex(ValueError, "missing description.md, sample_submission.csv"):
                adapter._stage_inputs(spec, job_paths)

    def test_local_zip_validation_restages_private_data_outside_solver_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "detecting-insults-in-social-commentary.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
                zf.writestr("test_with_solutions.csv", "id,target\n1,1\n")

            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=_FakeCompetitionPreparer(),
            )
            spec = self._spec(competition_zip_path=str(zip_path))

            with adapter._resolve_legacy_validation_target(spec) as target:
                assert target is not None
                self.assertEqual(target.competition_name, "detecting-insults-in-social-commentary")
                self.assertTrue((target.prepared_dir / "private" / "answers.csv").is_file())
                self.assertTrue((target.prepared_dir / "public" / "train.csv").is_file())
                self.assertFalse(str(target.prepared_dir).startswith(str(tmp_path / "job" / "workspace")))

    def test_run_real_loop_restores_submission_and_materializes_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runtime = mock.Mock()
            runtime.prepare_image.return_value = "aisci-mle:test"
            runtime.create_session_spec.return_value = SimpleNamespace(
                profile=SimpleNamespace(name="mle-default", pull_policy=SimpleNamespace(value="if-missing")),
                runtime_profile=SimpleNamespace(pull_policy=None),
                workdir="/home/code",
                env=(),
                mounts=(),
                labels=(),
                run_as_user="1000:1000",
                workspace_layout=SimpleNamespace(value="mle"),
            )
            runtime.start_session.return_value = SimpleNamespace(
                container_name="mle-test-session",
                image_tag="aisci-mle:test",
                profile=runtime.create_session_spec.return_value.profile,
                runtime_profile=runtime.create_session_spec.return_value.runtime_profile,
                workspace_layout=runtime.create_session_spec.return_value.workspace_layout,
                mounts=runtime.create_session_spec.return_value.mounts,
                workdir=runtime.create_session_spec.return_value.workdir,
                labels=runtime.create_session_spec.return_value.labels,
                run_as_user=runtime.create_session_spec.return_value.run_as_user,
                started_at=SimpleNamespace(isoformat=lambda: "2026-04-01T00:00:00+00:00"),
            )
            runtime.inspect_session.return_value = {}

            def engine_run() -> str:
                submission_dir = job_paths.workspace_dir / "submission"
                submission_dir.mkdir(parents=True, exist_ok=True)
                (submission_dir / "submission_registry.jsonl").write_text("", encoding="utf-8")
                fallback_path = job_paths.workspace_dir / "code" / "output"
                fallback_path.mkdir(parents=True, exist_ok=True)
                (fallback_path / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
                return "engine finished"

            adapter = MLEDomainAdapter(runtime=runtime)
            job_paths = self._job_paths(tmp_path / "job")
            job = SimpleNamespace(
                id="mle-timeout-test",
                llm_profile="gpt-5.4",
                objective="mle timeout test",
                runtime_profile=SimpleNamespace(
                    time_limit="1h",
                    keep_container_on_failure=False,
                    pull_policy=None,
                    image=None,
                    gpu_ids=[],
                    gpu_count=0,
                ),
                mode_spec=SimpleNamespace(validation_command=None),
            )
            resolved_profile = SimpleNamespace(name="gpt-5.4")

            with (
                mock.patch("aisci_domain_mle.adapter.default_domain_mle_profile", return_value=mock.sentinel.profile),
                mock.patch.object(adapter, "_session_env", return_value={"LOGS_DIR": "/home/logs"}),
                mock.patch.object(adapter, "_build_llm_client", return_value=mock.sentinel.llm),
                mock.patch.object(adapter, "_orchestrator_runtime", return_value=mock.sentinel.runtime_config),
                mock.patch("aisci_domain_mle.adapter.DockerShellInterface", return_value=mock.sentinel.shell),
                mock.patch("aisci_domain_mle.adapter.EmbeddedMLEEngine") as engine_cls,
            ):
                engine_cls.return_value.run.side_effect = engine_run
                adapter._run_real_loop(job, job_paths, resolved_profile)

            self.assertTrue((job_paths.workspace_dir / "submission" / "submission.csv").is_file())
            registry_text = (job_paths.workspace_dir / "submission" / "submission_registry.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("champion_selected", registry_text)
            runtime.cleanup.assert_called_once_with(runtime.start_session.return_value)

    def test_competition_name_cache_hit_stages_prepared_public_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cache_root = tmp_path / "cache"
            public_dir = cache_root / "demo-competition" / "prepared" / "public"
            private_dir = cache_root / "demo-competition" / "prepared" / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (public_dir / "sample_submission_public.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (private_dir / "answers.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            description_path = tmp_path / "description.md"
            description_path.write_text("# Demo\n", encoding="utf-8")

            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=_FakeCompetitionPreparer(description_path),
            )
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_name="demo-competition",
                mlebench_data_dir=str(cache_root),
            )

            adapter._stage_inputs(spec, job_paths)

            self.assertTrue((job_paths.workspace_dir / "data" / "train.csv").is_file())
            self.assertFalse((job_paths.workspace_dir / "data" / "answers.csv").exists())

    def test_competition_name_cache_miss_requires_proxy_on_prepare_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cache_root = tmp_path / "cache"

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                competition_name="demo-competition",
                mlebench_data_dir=str(cache_root),
            )

            from unittest.mock import patch

            resolved_inputs = SimpleNamespace(
                competition_name="demo-competition",
                competition_zip_path=None,
                cache_prepared_exists=False,
                cache_prepared_dir=None,
                legacy_prepare_plan=SimpleNamespace(
                    command=[
                        "/usr/bin/python3",
                        "-m",
                        "aisci_domain_mle.vendored_lite_cli",
                        "prepare",
                        "--competition-id",
                        "demo-competition",
                    ]
                ),
            )
            with patch("aisci_domain_mle.adapter.resolve_competition_source", return_value=resolved_inputs):
                with self.assertRaisesRegex(ValueError, "proxy-on"):
                    adapter._stage_inputs(spec, job_paths)

    def test_prepared_competition_root_data_dir_copies_only_public_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prepared_root = tmp_path / "demo-competition" / "prepared"
            public_dir = prepared_root / "public"
            private_dir = prepared_root / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (public_dir / "description.md").write_text("# Demo\n", encoding="utf-8")
            (public_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (private_dir / "answers.csv").write_text("id,target\n1,1\n", encoding="utf-8")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(data_dir=str(prepared_root.parent))

            adapter._stage_inputs(spec, job_paths)

            self.assertTrue((job_paths.workspace_dir / "data" / "train.csv").is_file())
            self.assertFalse((job_paths.workspace_dir / "data" / "answers.csv").exists())

    def test_ambiguous_raw_data_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            (raw_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (raw_dir / "test_with_solutions.csv").write_text("id,target\n1,1\n", encoding="utf-8")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(data_dir=str(raw_dir))

            with self.assertRaisesRegex(ValueError, "data_dir must point to a public competition directory"):
                adapter._stage_inputs(spec, job_paths)

    def test_multiple_competition_data_sources_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "demo.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "x,y\n1,2\n")
            public_dir = tmp_path / "prepared" / "public"
            public_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")

            adapter = MLEDomainAdapter(
                runtime=SimpleNamespace(),
                competition_preparer=_FakeCompetitionPreparer(),
            )
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                workspace_bundle_zip=str(zip_path),
                data_dir=str(public_dir),
            )

            with self.assertRaisesRegex(ValueError, "exactly one competition data source"):
                adapter._stage_inputs(spec, job_paths)

    def test_grading_config_path_is_rejected_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            public_dir = tmp_path / "prepared" / "public"
            public_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            grading_config = tmp_path / "grading_config.json"
            grading_config.write_text("{}", encoding="utf-8")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                data_dir=str(public_dir),
                grading_config_path=str(grading_config),
            )

            with self.assertRaisesRegex(ValueError, "grading_config_path is not supported"):
                adapter._stage_inputs(spec, job_paths)

    def test_private_sample_submission_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            competition_root = tmp_path / "demo"
            public_dir = competition_root / "prepared" / "public"
            private_dir = competition_root / "prepared" / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            private_sample = private_dir / "sample_submission.csv"
            private_sample.write_text("id,target\n1,1\n", encoding="utf-8")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(
                data_dir=str(competition_root),
                sample_submission_path=str(private_sample),
            )

            with self.assertRaisesRegex(ValueError, "sample_submission_path must not point to private competition data"):
                adapter._stage_inputs(spec, job_paths)

    def test_public_data_dir_with_eval_command_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            public_dir = tmp_path / "prepared" / "public"
            public_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (public_dir / "eval_cmd.txt").write_text("cat /private/data/secret.txt\n", encoding="utf-8")

            adapter = MLEDomainAdapter(runtime=SimpleNamespace())
            job_paths = self._job_paths(tmp_path / "job")
            spec = self._spec(data_dir=str(public_dir))

            with self.assertRaisesRegex(ValueError, "validation or grading artifacts"):
                adapter._stage_inputs(spec, job_paths)


if __name__ == "__main__":
    unittest.main()
