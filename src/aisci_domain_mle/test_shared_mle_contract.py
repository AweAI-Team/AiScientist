from __future__ import annotations

import copy
import sys
import types
import unittest

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

from aisci_app.presentation import build_mle_job_spec, default_mle_llm_profile_name
from aisci_core.models import MLESpec, NetworkPolicy


class SharedMleContractTests(unittest.TestCase):
    def test_legacy_mle_spec_payload_still_validates(self) -> None:
        spec = MLESpec.model_validate({"workspace_bundle_zip": "/tmp/demo.zip"})

        self.assertEqual(spec.workspace_bundle_zip, "/tmp/demo.zip")
        self.assertIsNone(spec.competition_name)
        self.assertIsNone(spec.competition_zip_path)
        self.assertIsNone(spec.mlebench_data_dir)

    def test_shared_mle_spec_rejects_multiple_competition_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one competition data source"):
            MLESpec.model_validate(
                {
                    "competition_name": "demo-competition",
                    "data_dir": "/tmp/demo-competition",
                }
            )

    def test_shared_mle_spec_allows_competition_name_metadata_for_local_zip(self) -> None:
        spec = MLESpec.model_validate(
            {
                "competition_name": "detecting-insults-in-social-commentary",
                "competition_zip_path": "/tmp/arbitrary-name.zip",
            }
        )

        self.assertEqual(spec.competition_name, "detecting-insults-in-social-commentary")
        self.assertEqual(spec.competition_zip_path, "/tmp/arbitrary-name.zip")

    def test_build_mle_job_spec_supports_competition_name_inputs(self) -> None:
        job_spec = build_mle_job_spec(
            competition_name="demo-competition",
            competition_zip_path=None,
            mlebench_data_dir="/tmp/mle-cache",
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
            gpu_ids=["0"],
            time_limit="12h",
            image="aisci-mle:test",
            pull_policy=None,
            run_final_validation=True,
        )

        self.assertEqual(job_spec.mode_spec.competition_name, "demo-competition")
        self.assertEqual(job_spec.mode_spec.mlebench_data_dir, "/tmp/mle-cache")
        self.assertIsNone(job_spec.mode_spec.competition_zip_path)
        self.assertEqual(job_spec.runtime_profile.gpu_ids, ["0"])
        self.assertEqual(job_spec.runtime_profile.gpu_count, 0)
        self.assertEqual(job_spec.runtime_profile.network_policy, NetworkPolicy.BRIDGE)

    def test_build_mle_job_spec_supports_local_zip_inputs(self) -> None:
        job_spec = build_mle_job_spec(
            competition_name=None,
            competition_zip_path="/tmp/demo-competition.zip",
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
            llm_profile="glm-5",
            gpus=1,
            gpu_ids=None,
            time_limit="24h",
            image=None,
            pull_policy=None,
            run_final_validation=False,
        )

        self.assertIsNone(job_spec.mode_spec.competition_name)
        self.assertEqual(job_spec.mode_spec.competition_zip_path, "/tmp/demo-competition.zip")
        self.assertEqual(job_spec.runtime_profile.gpu_count, 1)
        self.assertEqual(job_spec.runtime_profile.gpu_ids, [])
        self.assertEqual(job_spec.runtime_profile.network_policy, NetworkPolicy.BRIDGE)

    def test_default_mle_llm_profile_uses_domain_registry(self) -> None:
        self.assertEqual(default_mle_llm_profile_name(), "gpt-5.4")


if __name__ == "__main__":
    unittest.main()
