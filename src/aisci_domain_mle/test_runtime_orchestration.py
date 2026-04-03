from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aisci_domain_mle.cli import _build_parser
from aisci_domain_mle.contracts import DockerBuildPolicy, InputSourceKind, ValidationKind
from aisci_domain_mle.input_resolver import build_dry_run_report, build_phase1_job_spec
from aisci_domain_mle.runtime_orchestration import build_runtime_plan
class RuntimeOrchestrationTests(unittest.TestCase):
    def test_cli_parser_accepts_runtime_plan_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "runtime-plan",
                "--name",
                "detecting-insults-in-social-commentary",
                "--gpus",
                "0,1",
                "--run-root",
                "/tmp/mle-runtime",
                "--build-policy",
                "force",
                "--docker-binary",
                "docker",
            ]
        )
        self.assertEqual(args.command, "runtime-plan")
        self.assertEqual(args.gpus, "0,1")
        self.assertEqual(args.build_policy, "force")
        self.assertEqual(args.run_root, "/tmp/mle-runtime")

    def test_force_build_plan_previews_shared_runtime_and_legacy_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_repo = tmp_path / "mle-bench"
            competition_dir = (
                legacy_repo
                / "mlebench"
                / "competitions"
                / "detecting-insults-in-social-commentary"
            )
            competition_dir.mkdir(parents=True)
            (legacy_repo / "mlebench" / "cli.py").write_text("print('placeholder')\n", encoding="utf-8")
            (competition_dir / "config.yaml").write_text(
                "description: mlebench/competitions/detecting-insults-in-social-commentary/description.md\n"
                "dataset:\n"
                "  sample_submission: detecting-insults-in-social-commentary/prepared/public/sample_submission_null.csv\n",
                encoding="utf-8",
            )
            (competition_dir / "description.md").write_text("# Demo\n", encoding="utf-8")
            public_dir = (
                tmp_path
                / "cache"
                / "detecting-insults-in-social-commentary"
                / "prepared"
                / "public"
            )
            private_dir = public_dir.parent / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            sample_submission = public_dir / "sample_submission_null.csv"
            sample_submission.write_text("Comment,Insult\nx,0\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"AISCI_MLEBENCH_REPO": str(legacy_repo)}, clear=False):
                job_spec = build_phase1_job_spec(
                    competition_name="detecting-insults-in-social-commentary",
                    competition_zip_path=None,
                    mlebench_data_dir=str(tmp_path / "cache"),
                    workspace_bundle_zip=None,
                    competition_bundle_zip=None,
                    data_dir=None,
                    code_repo_zip=None,
                    description_path=None,
                    sample_submission_path=None,
                    validation_command=None,
                    grading_config_path=None,
                    metric_direction=None,
                    llm_profile="gpt-5.4",
                    gpus="0,1",
                    time_limit="2h",
                    dockerfile=None,
                    run_final_validation=True,
                    dry_run=True,
                    objective="runtime orchestration cache test",
                )
                dry_run_report = build_dry_run_report(job_spec, wait_requested=True)
                plan = build_runtime_plan(
                    job_spec=job_spec,
                    resolved_inputs=dry_run_report.resolved_inputs,
                    run_root=tmp_path / "run",
                    build_policy="force",
                )

            self.assertTrue(plan.ready_to_execute)
            self.assertEqual(plan.build_policy, DockerBuildPolicy.FORCE)
            self.assertIsNone(plan.build_command)
            self.assertEqual(
                plan.image_inspect_command,
                ["docker", "image", "inspect", "hub.byted.org/your-team/aisci-mle:latest"],
            )
            self.assertIn("--gpus", plan.docker_run_command)
            self.assertIn("device=0,1", plan.docker_run_command)
            self.assertEqual(plan.env["AISCI_JOB_ID"], "detecting-insults-in-social-commentary")
            self.assertEqual(plan.env["AISCI_OBJECTIVE"], "runtime orchestration cache test")
            self.assertEqual(plan.env["LOGS_DIR"], "/home/logs")
            self.assertNotIn("TIME_LIMIT_SECS", plan.env)
            self.assertNotIn("AISCI_REPO_ROOT", plan.env)
            self.assertNotIn("PYTHONPATH", plan.env)
            self.assertNotIn("AISCI_REASONING_EFFORT", plan.env)
            self.assertNotIn("AISCI_REASONING_SUMMARY", plan.env)
            self.assertEqual(
                {mount.target for mount in plan.mounts},
                {"/home", "/home/logs", "/workspace/logs"},
            )
            self.assertIn(
                f"{(tmp_path / 'run' / 'workspace').resolve()}:/home",
                plan.docker_run_command,
            )
            self.assertIsNone(plan.docker_exec_command)
            self.assertTrue(
                any("build-policy is kept for compatibility only" in warning for warning in plan.warnings)
            )
            self.assertTrue(
                any("host and uses Docker only for the /home sandbox workspace" in warning for warning in plan.warnings)
            )
            self.assertIn(["rm", "-rf", str((tmp_path / "run" / "workspace" / "data").resolve())], plan.host_setup_commands)
            self.assertIn(
                ["cp", "-a", f"{public_dir.resolve()}/.", str((tmp_path / 'run' / 'workspace' / 'data').resolve())],
                plan.host_setup_commands,
            )
            self.assertIn(
                [
                            "cp",
                            "-f",
                            str(
                                (
                                    legacy_repo
                                    / "mlebench"
                                    / "competitions"
                                    / "detecting-insults-in-social-commentary"
                            / "description.md"
                        ).resolve()
                    ),
                    str((tmp_path / "run" / "workspace" / "data" / "description.md").resolve()),
                ],
                plan.host_setup_commands,
            )
            self.assertIn(
                [
                    "cp",
                    "-f",
                    str(sample_submission.resolve()),
                    str((tmp_path / "run" / "workspace" / "data" / "sample_submission.csv").resolve()),
                ],
                plan.host_setup_commands,
            )
            self.assertEqual(plan.validation.kind, ValidationKind.LEGACY_GRADE)
            assert plan.validation.host_command is not None
            self.assertIn("grade-sample", plan.validation.host_command)
            self.assertEqual(
                plan.validation.docker_cleanup_command,
                plan.docker_cleanup_command,
            )
            self.assertTrue(public_dir.exists())
            self.assertTrue(private_dir.exists())

    def test_local_zip_plan_uses_shared_validation_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "detecting-insults-in-social-commentary.zip"
            zip_path.write_text("placeholder", encoding="utf-8")

            job_spec = build_phase1_job_spec(
                competition_name=None,
                competition_zip_path=str(zip_path),
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="1h",
                dockerfile=None,
                run_final_validation=True,
                dry_run=True,
                objective="runtime orchestration zip test",
            )
            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)
            plan = build_runtime_plan(
                job_spec=job_spec,
                resolved_inputs=dry_run_report.resolved_inputs,
                run_root=tmp_path / "run",
                build_policy="never",
            )

            self.assertTrue(plan.ready_to_execute)
            self.assertEqual(plan.build_policy, DockerBuildPolicy.NEVER)
            self.assertIsNone(plan.build_command)
            self.assertEqual(plan.validation.kind, ValidationKind.LEGACY_GRADE)
            self.assertIn(
                f"{(tmp_path / 'run' / 'workspace').resolve()}:/home",
                plan.docker_run_command,
            )
            self.assertIsNone(plan.docker_exec_command)
            self.assertTrue(
                any("Local zip input is supported in the live adapter" in warning for warning in plan.warnings)
            )

    def test_vendored_lite_cache_hit_plan_uses_internal_grade_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            public_dir = (
                tmp_path
                / "cache"
                / "detecting-insults-in-social-commentary"
                / "prepared"
                / "public"
            )
            private_dir = public_dir.parent / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            (public_dir / "sample_submission_null.csv").write_text(
                "Insult,Date,Comment\n0,2024-01-03,nice\n",
                encoding="utf-8",
            )

            job_spec = build_phase1_job_spec(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=None,
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="1h",
                dockerfile=None,
                run_final_validation=True,
                dry_run=True,
                objective="vendored lite cache hit",
            )
            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)
            plan = build_runtime_plan(
                job_spec=job_spec,
                resolved_inputs=dry_run_report.resolved_inputs,
                run_root=tmp_path / "run",
                build_policy="auto",
            )

            self.assertTrue(plan.ready_to_execute)
            assert plan.validation is not None
            self.assertEqual(plan.validation.kind, ValidationKind.LEGACY_GRADE)
            assert plan.validation.host_command is not None
            self.assertEqual(
                plan.validation.host_command[:3],
                ["python3", "-m", "aisci_domain_mle.vendored_lite_cli"],
            )
            self.assertIn("grade-sample", plan.validation.host_command)

    def test_runtime_plan_warns_when_legacy_docker_flags_are_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            public_dir = tmp_path / "cache" / "demo" / "prepared" / "public"
            public_dir.mkdir(parents=True)
            (public_dir.parent / "private").mkdir(parents=True)

            job_spec = build_phase1_job_spec(
                competition_name="demo",
                competition_zip_path=None,
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="1h",
                dockerfile=None,
                run_final_validation=False,
                dry_run=True,
                objective="invalid dockerfile test",
            )
            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)

            plan = build_runtime_plan(
                job_spec=job_spec,
                resolved_inputs=dry_run_report.resolved_inputs,
                run_root=tmp_path / "run",
                build_policy="force",
                dockerfile_path=str(tmp_path / "missing.Dockerfile"),
                image_tag="custom:image",
                container_name="custom-container",
            )

            self.assertIsNone(plan.build_command)
            self.assertFalse(plan.ready_to_execute)
            self.assertTrue(any("description metadata is missing" in warning for warning in plan.warnings))
            self.assertTrue(any("sample submission metadata is missing" in warning for warning in plan.warnings))
            self.assertTrue(any("--dockerfile is ignored" in warning for warning in plan.warnings))
            self.assertTrue(any("--image-tag is ignored" in warning for warning in plan.warnings))
            self.assertTrue(any("--container-name is ignored" in warning for warning in plan.warnings))

    def test_runtime_plan_prefers_explicit_metadata_overrides_for_cache_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            public_dir = tmp_path / "cache" / "demo" / "prepared" / "public"
            public_dir.mkdir(parents=True)
            (public_dir.parent / "private").mkdir(parents=True)

            description_path = tmp_path / "override_description.md"
            description_path.write_text("# Demo\n", encoding="utf-8")
            sample_submission_path = tmp_path / "override_sample_submission.csv"
            sample_submission_path.write_text("id,target\n1,0\n", encoding="utf-8")

            job_spec = build_phase1_job_spec(
                competition_name="demo",
                competition_zip_path=None,
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=str(description_path),
                sample_submission_path=str(sample_submission_path),
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="1h",
                dockerfile=None,
                run_final_validation=False,
                dry_run=True,
                objective="metadata override test",
            )
            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)

            plan = build_runtime_plan(
                job_spec=job_spec,
                resolved_inputs=dry_run_report.resolved_inputs,
                run_root=tmp_path / "run",
                build_policy="auto",
            )

            self.assertTrue(plan.ready_to_execute)
            self.assertIn(
                [
                    "cp",
                    "-f",
                    str(description_path.resolve()),
                    str((tmp_path / "run" / "workspace" / "data" / "description.md").resolve()),
                ],
                plan.host_setup_commands,
            )
            self.assertIn(
                [
                    "cp",
                    "-f",
                    str(sample_submission_path.resolve()),
                    str((tmp_path / "run" / "workspace" / "data" / "sample_submission.csv").resolve()),
                ],
                plan.host_setup_commands,
            )
            self.assertFalse(any("metadata is missing" in warning for warning in plan.warnings))

    def test_data_dir_input_stays_safe_in_dry_run_and_runtime_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "demo-competition"
            public_dir = data_dir / "prepared" / "public"
            (data_dir / "prepared" / "private").mkdir(parents=True)
            public_dir.mkdir(parents=True)
            (public_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (public_dir / "description.md").write_text("# Demo\n", encoding="utf-8")
            (public_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

            job_spec = build_phase1_job_spec(
                competition_name=None,
                competition_zip_path=None,
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=str(data_dir),
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="30m",
                dockerfile=None,
                run_final_validation=False,
                dry_run=True,
                objective="data dir runtime test",
            )
            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)
            plan = build_runtime_plan(
                job_spec=job_spec,
                resolved_inputs=dry_run_report.resolved_inputs,
                run_root=tmp_path / "run",
                build_policy="auto",
            )

            self.assertEqual(dry_run_report.resolved_inputs.source_kind, InputSourceKind.DATA_DIR)
            self.assertTrue(plan.ready_to_execute)
            self.assertEqual(
                {mount.target for mount in plan.mounts},
                {"/home", "/home/logs", "/workspace/logs"},
            )
            self.assertIn(
                f"{(tmp_path / 'run' / 'workspace').resolve()}:/home",
                plan.docker_run_command,
            )
            self.assertIn(["rm", "-rf", str((tmp_path / "run" / "workspace" / "data").resolve())], plan.host_setup_commands)
            self.assertIn(
                [
                    "cp",
                    "-a",
                    f"{(data_dir / 'prepared' / 'public').resolve()}/.",
                    str((tmp_path / 'run' / 'workspace' / 'data').resolve()),
                ],
                plan.host_setup_commands,
            )

    def test_raw_data_dir_is_not_ready_in_dry_run_or_runtime_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            (raw_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")

            job_spec = build_phase1_job_spec(
                competition_name=None,
                competition_zip_path=None,
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=str(raw_dir),
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="30m",
                dockerfile=None,
                run_final_validation=False,
                dry_run=True,
                objective="raw data dir runtime test",
            )

            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)
            plan = build_runtime_plan(
                job_spec=job_spec,
                resolved_inputs=dry_run_report.resolved_inputs,
                run_root=tmp_path / "run",
                build_policy="auto",
            )

            self.assertEqual(dry_run_report.status, "needs_prepare")
            self.assertTrue(
                any("data_dir must point to a public competition directory" in warning for warning in dry_run_report.warnings)
            )
            self.assertFalse(plan.ready_to_execute)
            self.assertTrue(
                any("data_dir must point to a public competition directory" in warning for warning in plan.warnings)
            )

    def test_runtime_plan_redacts_credentials_and_preserves_proxy_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            public_dir = tmp_path / "cache" / "demo" / "prepared" / "public"
            public_dir.mkdir(parents=True)
            (public_dir.parent / "private").mkdir(parents=True)

            job_spec = build_phase1_job_spec(
                competition_name="demo",
                competition_zip_path=None,
                mlebench_data_dir=str(tmp_path / "cache"),
                workspace_bundle_zip=None,
                competition_bundle_zip=None,
                data_dir=None,
                code_repo_zip=None,
                description_path=None,
                sample_submission_path=None,
                validation_command=None,
                grading_config_path=None,
                metric_direction=None,
                llm_profile="gpt-5.4",
                gpus="0",
                time_limit="1h",
                dockerfile=None,
                run_final_validation=False,
                dry_run=True,
                objective="redaction test",
            )
            dry_run_report = build_dry_run_report(job_spec, wait_requested=True)
            with mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-secret",
                    "HTTP_PROXY": "http://user:pass@proxy.example:8080",
                    "HTTPS_PROXY": "http://user:pass@proxy.example:8080",
                },
                clear=False,
            ):
                plan = build_runtime_plan(
                    job_spec=job_spec,
                    resolved_inputs=dry_run_report.resolved_inputs,
                    run_root=tmp_path / "run",
                    build_policy="auto",
                )

            self.assertEqual(plan.env["HTTP_PROXY"], "<redacted>")
            self.assertEqual(plan.env["HTTPS_PROXY"], "<redacted>")
            self.assertEqual(plan.env["AISCI_JOB_ID"], "demo")
            self.assertNotIn("OPENAI_API_KEY", plan.env)
            self.assertNotIn("AISCI_MAX_TOKENS", plan.env)
            self.assertNotIn("AISCI_REASONING_EFFORT", plan.env)
            self.assertNotIn("AISCI_REASONING_SUMMARY", plan.env)
            self.assertIn("HTTP_PROXY", plan.passthrough_env_keys)
            self.assertIn("HTTPS_PROXY", plan.passthrough_env_keys)
            self.assertIn("HTTP_PROXY=<redacted>", plan.docker_run_command)


if __name__ == "__main__":
    unittest.main()
