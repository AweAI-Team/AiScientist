from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import aisci_domain_mle.adapter as adapter_module
import aisci_domain_mle.orchestrator as orchestrator_module
import aisci_domain_mle.orchestrator_runtime as orchestrator_runtime_module
import aisci_domain_mle.tools.spawn_subagent_tool as spawn_tool_module
from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_core.models import WorkspaceLayout
from aisci_domain_mle.adapter import MLEDomainAdapter
from aisci_domain_mle.cli import _build_parser
from aisci_domain_mle.orchestrator import ImplementTool, RunExperimentTool, SubmitTool, run
from aisci_domain_mle.orchestrator_runtime import (
    StubLLMClient,
    build_task_prompt,
    build_smoke_runtime_config,
    create_orchestrator_llm,
)
from aisci_domain_mle.prompts.templates import (
    experiment_system_prompt_for_run,
    implementation_system_prompt_for_run,
)
from aisci_domain_mle.subagents.configs import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_IMPLEMENTATION_CONFIG,
    EXPERIMENT_VALIDATE_TIME_LIMIT,
)
from aisci_domain_mle.subagents.experiment import ExperimentSubagent
from aisci_domain_mle.subagents.implementation import ImplementationSubagent


class OrchestratorRuntimeTests(unittest.TestCase):
    def test_build_task_prompt_matches_legacy_wording(self) -> None:
        runtime = build_smoke_runtime_config("/tmp/mle-orchestrator-prompt", time_limit=300)
        prompt = build_task_prompt(runtime)
        self.assertIn("You have **00:05:00** to solve this Kaggle competition.", prompt)
        self.assertIn(f"Competition data is in `{runtime.paths.data_dir}/`.", prompt)
        self.assertIn(f"Your code goes in `{runtime.paths.code_dir}/` (git repo).", prompt)

    def test_cli_parser_accepts_orchestrator_smoke_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "orchestrator-smoke",
                "--run-root",
                "/tmp/mle-orchestrator-smoke",
                "--max-steps",
                "4",
                "--no-file-as-bus",
            ]
        )
        self.assertEqual(args.command, "orchestrator-smoke")
        self.assertEqual(args.max_steps, 4)
        self.assertFalse(args.file_as_bus)

    def test_stub_llm_drives_python_then_submit_sequence(self) -> None:
        runtime = build_smoke_runtime_config("/tmp/mle-orchestrator-sequence")
        llm = create_orchestrator_llm(runtime)
        self.assertIsInstance(llm, StubLLMClient)

        tool_schemas = [
            {"type": "function", "function": {"name": "python"}},
            {"type": "function", "function": {"name": "submit"}},
        ]

        first = llm.chat(messages=[], tools=tool_schemas)
        self.assertEqual(first.tool_calls[0].name, "python")
        second = llm.chat(messages=[], tools=tool_schemas)
        self.assertEqual(second.tool_calls[0].name, "submit")

    def test_create_orchestrator_llm_uses_shared_client_factory(self) -> None:
        runtime = replace(
            build_smoke_runtime_config("/tmp/mle-orchestrator-shared-client"),
            api_mode="responses",
            model="gpt-5.4",
        )
        shared_client = object()

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                orchestrator_runtime_module,
                "create_llm_client",
                return_value=shared_client,
            ) as factory:
                llm = create_orchestrator_llm(runtime)

        self.assertIs(llm, shared_client)
        factory.assert_called_once_with()

    def test_file_as_bus_false_uses_no_bus_prompts_and_tools(self) -> None:
        self.assertNotIn("`add_impl_log` —", implementation_system_prompt_for_run(False))
        self.assertNotIn("`add_exp_log` —", experiment_system_prompt_for_run(False))

        impl_tools = [tool.name() for tool in ImplementationSubagent(None, None, file_as_bus=False).get_tools()]
        exp_tools = [tool.name() for tool in ExperimentSubagent(None, None, file_as_bus=False).get_tools()]
        self.assertNotIn("add_impl_log", impl_tools)
        self.assertNotIn("add_exp_log", exp_tools)

    def test_implement_tool_restores_legacy_default_budgets(self) -> None:
        captured: dict[str, object] = {}
        runtime = build_smoke_runtime_config("/tmp/mle-orchestrator-impl-budget", file_as_bus=False)

        class FakeImplementationSubagent:
            def __init__(self, shell, llm, config, file_as_bus=True):
                captured["config"] = config
                captured["file_as_bus"] = file_as_bus

            def run(self, context: str) -> SubagentOutput:
                captured["context"] = context
                return SubagentOutput(
                    status=SubagentStatus.COMPLETED,
                    content="ok",
                    num_steps=1,
                    runtime_seconds=0.1,
                    log_path="/tmp/fake-impl.log",
                )

        with mock.patch.object(orchestrator_module, "ImplementationSubagent", FakeImplementationSubagent):
            tool = ImplementTool(SimpleNamespace(), None, runtime.paths, file_as_bus=False)
            tool.execute(None, mode="explore")
            self.assertEqual(captured["config"].time_limit, 7200)
            tool.execute(None, mode="full")
            self.assertEqual(captured["config"].time_limit, DEFAULT_IMPLEMENTATION_CONFIG.time_limit)
            self.assertFalse(captured["file_as_bus"])

    def test_run_experiment_tool_restores_legacy_default_budgets(self) -> None:
        captured: dict[str, object] = {}
        runtime = build_smoke_runtime_config("/tmp/mle-orchestrator-exp-budget", file_as_bus=False)

        class FakeExperimentSubagent:
            def __init__(self, shell, llm, config, file_as_bus=True):
                captured["config"] = config
                captured["file_as_bus"] = file_as_bus

            def run(self, context: str) -> SubagentOutput:
                captured["context"] = context
                return SubagentOutput(
                    status=SubagentStatus.COMPLETED,
                    content="ok",
                    num_steps=1,
                    runtime_seconds=0.1,
                    log_path="/tmp/fake-exp.log",
                )

        with mock.patch.object(orchestrator_module, "ExperimentSubagent", FakeExperimentSubagent):
            tool = RunExperimentTool(SimpleNamespace(), None, runtime.paths, file_as_bus=False)
            tool.execute(None, mode="validate")
            self.assertEqual(captured["config"].time_limit, EXPERIMENT_VALIDATE_TIME_LIMIT)
            tool.execute(None, mode="full")
            self.assertEqual(captured["config"].time_limit, DEFAULT_EXPERIMENT_CONFIG.time_limit)
            self.assertFalse(captured["file_as_bus"])

    def test_prepare_runtime_dirs_falls_back_to_home_agent_logs(self) -> None:
        runtime = build_smoke_runtime_config("/tmp/mle-orchestrator-fallback")
        attempted_dirs: list[str] = []

        def fake_ensure_local_dir(shell, path: str):
            del shell
            attempted_dirs.append(path)
            if path == runtime.paths.logs_dir:
                raise PermissionError("unwritable logs dir")
            return Path(path)

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(orchestrator_module, "LOGS_DIR", runtime.paths.logs_dir):
                with mock.patch.object(orchestrator_module, "_ensure_local_dir", side_effect=fake_ensure_local_dir):
                    effective_paths = orchestrator_module._prepare_runtime_dirs(None, runtime.paths)
                    self.assertEqual(os.environ["LOGS_DIR"], "/home/agent/logs")

        self.assertEqual(effective_paths.logs_dir, "/home/agent/logs")
        self.assertIn("/home/agent/logs", attempted_dirs)
        self.assertIn("/home/agent/logs/subagent_logs", attempted_dirs)

    def test_session_env_is_minimal_sandbox_only(self) -> None:
        adapter = MLEDomainAdapter(runtime=SimpleNamespace())
        job = SimpleNamespace(
            id="job-123",
            llm_profile="gpt-5.4",
            objective="demo objective",
            runtime_profile=SimpleNamespace(time_limit="1h", gpu_ids=["0"], gpu_count=1),
        )

        with mock.patch.dict(
            os.environ,
            {
                "AISCI_STUB_LLM": "1",
                "AISCI_STUB_SCENARIO": "submit_sample",
                "AISCI_HOME_ROOT": "/host-only/home",
                "FILE_AS_BUS": "0",
                "LOGS_DIR": "/host-only/logs",
                "HTTP_PROXY": "http://proxy.example:8080",
            },
            clear=True,
        ):
            env = adapter._session_env(job)

        self.assertNotIn("AISCI_HOME_ROOT", env)
        self.assertNotIn("AISCI_MODEL", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertEqual(env["AISCI_JOB_ID"], "job-123")
        self.assertEqual(env["AISCI_OBJECTIVE"], "demo objective")
        self.assertEqual(env["AISCI_STUB_LLM"], "1")
        self.assertEqual(env["AISCI_STUB_SCENARIO"], "submit_sample")
        self.assertEqual(env["LOGS_DIR"], "/home/logs")
        self.assertEqual(env["HTTP_PROXY"], "http://proxy.example:8080")

    def test_ensure_runtime_ready_uses_shared_profile_registry_for_host_checks(self) -> None:
        runtime = mock.Mock()
        runtime.can_use_docker.return_value = True
        adapter = MLEDomainAdapter(runtime=runtime)
        job = SimpleNamespace(llm_profile="gpt-5.4")
        profile = SimpleNamespace(
            name="gpt-5.4",
            backend_name="openai-main",
            provider="openai",
            model="gpt-5.4",
            api_mode="responses",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            job_paths = SimpleNamespace(logs_dir=Path(tmp_dir) / "logs")
            with mock.patch.object(adapter_module, "runtime_uses_stub_llm", return_value=False):
                with mock.patch.object(adapter_module, "resolve_llm_profile", return_value=profile) as resolve_profile:
                    with mock.patch.object(adapter_module, "missing_backend_env_vars", return_value=[]) as missing_env:
                        adapter._ensure_runtime_ready(job, job_paths)

        resolve_profile.assert_called_once_with(
            "gpt-5.4",
            default_for="mle",
            profile_file=adapter_module.domain_llm_profile_file(),
        )
        missing_env.assert_called_once_with(profile)

    def test_run_real_loop_uses_shared_runtime_manager_session_methods(self) -> None:
        runtime = mock.Mock()
        runtime.prepare_image.return_value = "aisci-mle:test"
        runtime.create_session_spec.return_value = mock.sentinel.session_spec
        session = SimpleNamespace(container_name="preview-session")
        runtime.start_session.return_value = session
        adapter = MLEDomainAdapter(runtime=runtime)
        resolved_profile = SimpleNamespace(name="gpt-5.4")
        job = SimpleNamespace(
            id="job-runtime",
            llm_profile="gpt-5.4",
            objective="mle optimization job",
            runtime_profile=SimpleNamespace(
                time_limit="1h",
                gpu_ids=["0"],
                gpu_count=1,
                keep_container_on_failure=False,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            job_paths = SimpleNamespace(
                workspace_dir=tmp_path / "workspace",
                logs_dir=tmp_path / "logs",
                artifacts_dir=tmp_path / "artifacts",
            )
            job_paths.workspace_dir.mkdir(parents=True, exist_ok=True)
            with (
                mock.patch.object(adapter_module, "default_domain_mle_profile", return_value=mock.sentinel.profile),
                mock.patch.object(adapter, "_session_env", return_value={"LOGS_DIR": "/home/logs"}),
                mock.patch.object(adapter, "_build_llm_client", return_value=mock.sentinel.llm),
                mock.patch.object(adapter, "_orchestrator_runtime", return_value=mock.sentinel.runtime_config),
                mock.patch.object(adapter_module, "DockerShellInterface", return_value=mock.sentinel.shell) as shell_cls,
                mock.patch.object(adapter_module, "EmbeddedMLEEngine") as engine_cls,
                mock.patch.object(adapter, "_write_session_info") as write_session_info,
            ):
                engine_cls.return_value.run.return_value = "engine finished"
                adapter._run_real_loop(job, job_paths, resolved_profile)

        runtime.prepare_image.assert_called_once_with(mock.sentinel.profile, job.runtime_profile)
        runtime.create_session_spec.assert_called_once_with(
            job.id,
            job_paths,
            mock.sentinel.profile,
            job.runtime_profile,
            layout=WorkspaceLayout.MLE,
            workdir="/home/code",
            env={"LOGS_DIR": "/home/logs"},
        )
        runtime.start_session.assert_called_once_with(mock.sentinel.session_spec, "aisci-mle:test")
        shell_cls.assert_called_once_with(runtime, session, working_dir="/home/code")
        engine_cls.assert_called_once_with(
            config=mock.sentinel.runtime_config,
            shell=mock.sentinel.shell,
            llm=mock.sentinel.llm,
        )
        engine_cls.return_value.run.assert_called_once_with()
        runtime.exec.assert_not_called()
        write_session_info.assert_called_once_with(
            job_paths,
            session,
            llm_profile="gpt-5.4",
            session_env={"LOGS_DIR": "/home/logs"},
        )
        runtime.cleanup.assert_called_once_with(session)

    def test_submit_tool_warns_once_for_early_submission_then_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = orchestrator_runtime_module.OrchestratorPaths(
                home_root=str(tmp_path),
                data_dir=str(tmp_path / "data"),
                code_dir=str(tmp_path / "code"),
                submission_dir=str(tmp_path / "submission"),
                agent_dir=str(tmp_path / "agent"),
                logs_dir=str(tmp_path / "logs"),
            )
            Path(paths.submission_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.submission_csv_path).write_text("id,target\n1,0\n", encoding="utf-8")

            tool = SubmitTool(
                time_limit=1000,
                start_time=orchestrator_module.time.time() - 100,
                paths=paths,
                llm=None,
            )

            first = tool.execute(None)
            second = tool.execute(None)

        self.assertIn("SUBMIT PRE-CHECK WARNINGS", first)
        self.assertIn("EARLY SUBMISSION", first)
        self.assertIn("call submit() again", first)
        self.assertEqual(second, "✅ Submission accepted. The agent will now terminate.")

    def test_maybe_validate_uses_shared_runtime_manager_validation_path(self) -> None:
        runtime = mock.Mock()
        runtime.can_use_docker.return_value = True
        runtime.prepare_image.return_value = "aisci-mle:test"
        runtime.create_session_spec.return_value = mock.sentinel.session_spec
        runtime.run_validation.return_value = mock.sentinel.validation_report
        adapter = MLEDomainAdapter(runtime=runtime)
        job = SimpleNamespace(
            id="job-validate",
            runtime_profile=SimpleNamespace(run_final_validation=True),
            mode_spec=SimpleNamespace(validation_command="python validate.py"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            submission_dir = tmp_path / "workspace" / "submission"
            submission_dir.mkdir(parents=True, exist_ok=True)
            (submission_dir / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            job_paths = SimpleNamespace(
                workspace_dir=tmp_path / "workspace",
                artifacts_dir=tmp_path / "artifacts",
            )
            with mock.patch.object(adapter_module, "default_domain_mle_profile", return_value=mock.sentinel.profile):
                with mock.patch.object(adapter, "_validation_command", return_value="python validate.py"):
                    report = adapter._maybe_validate(job, job_paths)

        self.assertIs(report, mock.sentinel.validation_report)
        runtime.prepare_image.assert_called_once_with(mock.sentinel.profile, job.runtime_profile)
        runtime.create_session_spec.assert_called_once_with(
            job.id,
            job_paths,
            mock.sentinel.profile,
            job.runtime_profile,
            layout=WorkspaceLayout.MLE,
            workdir="/home/code",
        )
        runtime.run_validation.assert_called_once_with(
            mock.sentinel.session_spec,
            "aisci-mle:test",
            "python validate.py",
            workdir="/home/code",
        )

    def test_maybe_validate_uses_legacy_grade_when_private_cache_is_available(self) -> None:
        runtime = mock.Mock()
        runtime.can_use_docker.return_value = False
        competition_grader = mock.Mock()
        competition_grader.grade_submission.return_value = {
            "competition_id": "demo-competition",
            "score": 0.81234,
            "submission_exists": True,
            "valid_submission": True,
            "gold_threshold": 0.9,
            "silver_threshold": 0.8,
            "bronze_threshold": 0.7,
            "median_threshold": 0.5,
            "gold_medal": False,
            "silver_medal": True,
            "bronze_medal": False,
            "above_median": True,
            "any_medal": True,
            "is_lower_better": False,
            "error": None,
            "created_at": "2026-04-01T00:00:00",
        }
        adapter = MLEDomainAdapter(runtime=runtime, competition_grader=competition_grader)
        job = SimpleNamespace(
            id="job-validate",
            runtime_profile=SimpleNamespace(run_final_validation=True),
            mode_spec=SimpleNamespace(validation_command=None),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            submission_dir = tmp_path / "workspace" / "submission"
            submission_dir.mkdir(parents=True, exist_ok=True)
            submission_path = submission_dir / "submission.csv"
            submission_path.write_text("id,target\n1,0\n", encoding="utf-8")
            job_paths = SimpleNamespace(
                workspace_dir=tmp_path / "workspace",
                artifacts_dir=tmp_path / "artifacts",
            )
            target = adapter_module.LegacyValidationTarget(
                competition_name="demo-competition",
                cache_root=tmp_path / "cache",
                prepared_dir=tmp_path / "cache" / "demo-competition" / "prepared",
            )
            with mock.patch.object(adapter, "_resolve_legacy_validation_target", return_value=nullcontext(target)):
                report = adapter._maybe_validate(job, job_paths)

        self.assertEqual(report.status, "passed")
        self.assertIn("score=0.81234", report.summary)
        self.assertEqual(report.container_image, "host:mlebench")
        competition_grader.grade_submission.assert_called_once_with(
            submission_path,
            competition_name="demo-competition",
            cache_root=tmp_path / "cache",
        )
        runtime.prepare_image.assert_not_called()
        runtime.run_validation.assert_not_called()

    def test_spawn_subagent_uses_effective_logs_dir_from_env(self) -> None:
        created_dirs: list[str] = []

        class FakeSubagent:
            def __init__(self, shell, llm, config):
                self.config = config

            def run(self, context: str) -> SubagentOutput:
                return SubagentOutput(
                    status=SubagentStatus.COMPLETED,
                    content=context,
                    num_steps=1,
                    runtime_seconds=0.1,
                    log_path=self.config.log_dir,
                )

        def fake_makedirs(path: str, exist_ok: bool = False) -> None:
            created_dirs.append(path)

        with mock.patch.dict(os.environ, {"LOGS_DIR": "/home/agent/logs"}, clear=True):
            with mock.patch.dict(spawn_tool_module._TYPE_CLASSES, {"explore": FakeSubagent}):
                with mock.patch.object(spawn_tool_module.os, "makedirs", side_effect=fake_makedirs):
                    tool = spawn_tool_module.SpawnSubagentTool(None, None)
                    result = tool.execute(None, subagent_type="explore", task="inspect state")

        self.assertIn("[EXPLORE Subagent +]", result)
        self.assertTrue(
            any(str(path).startswith("/home/agent/logs/subagent_logs/generic_explore_001_") for path in created_dirs)
        )

    def test_orchestrator_smoke_run_creates_submission_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = build_smoke_runtime_config(
                tmp_dir,
                max_steps=4,
                time_limit=300,
                file_as_bus=False,
            )
            run(runtime)

            submission_path = Path(runtime.paths.submission_csv_path)
            summary_path = Path(runtime.paths.agent_summary_path)
            env_path = Path(runtime.paths.agent_env_path)

            self.assertTrue(submission_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(env_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["submission_exists"])
            self.assertEqual(summary["impl_calls"], 0)
            self.assertEqual(summary["exp_calls"], 0)


if __name__ == "__main__":
    unittest.main()
