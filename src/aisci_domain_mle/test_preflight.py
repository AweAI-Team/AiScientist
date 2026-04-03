from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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

        def model_copy(self, *, deep=False):
            if deep:
                return self.__class__(**copy.deepcopy(self.__dict__))
            return self.__class__(**dict(self.__dict__))

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.model_validator = model_validator
    sys.modules["pydantic"] = pydantic_stub

if "structlog" not in sys.modules:
    structlog_stub = types.ModuleType("structlog")
    stdlib_stub = types.ModuleType("structlog.stdlib")

    class _BoundLogger:
        def bind(self, **kwargs):
            del kwargs
            return self

        def info(self, *args, **kwargs):
            del args, kwargs

        def warning(self, *args, **kwargs):
            del args, kwargs

        def error(self, *args, **kwargs):
            del args, kwargs

        def debug(self, *args, **kwargs):
            del args, kwargs

    get_logger = lambda *args, **kwargs: _BoundLogger()
    stdlib_stub.get_logger = get_logger
    structlog_stub.get_logger = get_logger
    structlog_stub.stdlib = stdlib_stub
    sys.modules["structlog"] = structlog_stub
    sys.modules["structlog.stdlib"] = stdlib_stub

from aisci_app.presentation import build_job_spec_clone, build_mle_job_spec
from aisci_core.models import JobRecord, JobStatus, JobType, PullPolicy, RunPhase, RuntimeProfile, WorkspaceLayout
from aisci_domain_mle.preflight import evaluate_mle_launch_preflight


class _FakeRuntime:
    def __init__(self, *, docker_ok: bool = True, image_exists: bool = True) -> None:
        self._docker_ok = docker_ok
        self._image_exists = image_exists

    def can_use_docker(self) -> bool:
        return self._docker_ok

    def image_exists(self, image_ref: str) -> bool:
        del image_ref
        return self._image_exists


class MLEPreflightTests(unittest.TestCase):
    def test_preflight_blocks_cache_miss_before_job_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_spec = build_mle_job_spec(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=None,
                mlebench_data_dir=str(Path(tmp_dir) / "cache"),
                workspace_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus=0,
                gpu_ids=None,
                time_limit="1h",
                image="aisci-mle:test",
                pull_policy=None,
                run_final_validation=False,
            )

            with mock.patch(
                "aisci_domain_mle.preflight.default_domain_mle_profile",
                return_value=types.SimpleNamespace(image="aisci-mle:test", pull_policy=PullPolicy.IF_MISSING),
            ):
                preflight = evaluate_mle_launch_preflight(
                    job_spec,
                    runtime=_FakeRuntime(docker_ok=True, image_exists=True),
                    competition_preparer=types.SimpleNamespace(resolve_public_metadata_paths=lambda *args, **kwargs: (None, None)),
                )

            self.assertFalse(preflight.ready)
            self.assertTrue(any("proxy-on" in error for error in preflight.errors))

    def test_preflight_blocks_network_image_pull_without_proxy(self) -> None:
        job_spec = build_mle_job_spec(
            competition_name=None,
            competition_zip_path="/tmp/demo.zip",
            mlebench_data_dir=None,
            workspace_zip=None,
            competition_bundle_zip=None,
            data_dir=None,
            code_repo_zip=None,
            description_path=None,
            sample_submission_path=None,
            validation_command=None,
            grading_config_path=None,
            metric_direction=None,
            llm_profile="gpt-5.4",
            gpus=0,
            gpu_ids=None,
            time_limit="1h",
            image="aisci-mle:test",
            pull_policy=None,
            run_final_validation=False,
        )

        with mock.patch("aisci_domain_mle.preflight.proxy_enabled", return_value=False):
            with mock.patch(
                "aisci_domain_mle.preflight.default_domain_mle_profile",
                return_value=types.SimpleNamespace(image="aisci-mle:test", pull_policy=PullPolicy.IF_MISSING),
            ):
                preflight = evaluate_mle_launch_preflight(
                    job_spec,
                    runtime=_FakeRuntime(docker_ok=True, image_exists=False),
                    competition_preparer=types.SimpleNamespace(resolve_public_metadata_paths=lambda *args, **kwargs: (None, None)),
                )

        self.assertFalse(preflight.ready)
        self.assertTrue(any("would pull it" in error for error in preflight.errors))

    def test_cloned_mle_validation_job_keeps_preflight_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_spec = build_mle_job_spec(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=None,
                mlebench_data_dir=str(Path(tmp_dir) / "cache"),
                workspace_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus=0,
                gpu_ids=None,
                time_limit="1h",
                image="aisci-mle:test",
                pull_policy=None,
                run_final_validation=False,
            )
            now = datetime.now(timezone.utc)
            cloned_spec = build_job_spec_clone(
                JobRecord(
                    id="demo-job",
                    job_type=JobType.MLE,
                    status=JobStatus.SUCCEEDED,
                    phase=RunPhase.FINALIZE,
                    objective="demo",
                    llm_profile="gpt-5.4",
                    runtime_profile=RuntimeProfile(
                        workspace_layout=WorkspaceLayout.MLE,
                        run_final_validation=False,
                        image="aisci-mle:test",
                        time_limit="1h",
                    ),
                    mode_spec=original_spec.mode_spec,
                    created_at=now,
                    updated_at=now,
                ),
                objective_suffix=" [self-check]",
                run_final_validation=True,
            )

            with mock.patch(
                "aisci_domain_mle.preflight.default_domain_mle_profile",
                return_value=types.SimpleNamespace(image="aisci-mle:test", pull_policy=PullPolicy.IF_MISSING),
            ):
                preflight = evaluate_mle_launch_preflight(
                    cloned_spec,
                    runtime=_FakeRuntime(docker_ok=True, image_exists=True),
                    competition_preparer=types.SimpleNamespace(resolve_public_metadata_paths=lambda *args, **kwargs: (None, None)),
                )

        self.assertTrue(cloned_spec.runtime_profile.run_final_validation)
        self.assertFalse(preflight.ready)
        self.assertTrue(any("proxy-on" in error for error in preflight.errors))


if __name__ == "__main__":
    unittest.main()
