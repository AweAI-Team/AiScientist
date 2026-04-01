from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfWriter

from aisci_core.models import (
    JobRecord,
    JobStatus,
    JobType,
    MLESpec,
    NetworkPolicy,
    PaperSpec,
    PullPolicy,
    RunPhase,
    RuntimeProfile,
    UIDGIDMode,
    WorkspaceLayout,
)
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_domain_mle.adapter import MLEDomainAdapter
from aisci_domain_paper.adapter import PaperDomainAdapter
from aisci_runtime_docker.models import ContainerSession, DockerExecutionResult
from aisci_runtime_docker.profiles import default_mle_profile, default_paper_profile
from aisci_runtime_docker.runtime import DockerRuntimeError, DockerRuntimeManager


def _make_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


def _write_llm_config(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "llm_profiles.yaml").write_text(
        """
defaults:
  paper: paper-default
  mle: mle-default
backends:
  openai:
    type: openai
    env:
      api_key:
        var: OPENAI_API_KEY
        required: true
profiles:
  paper-default:
    backend: openai
    model: gpt-5.4
    api: responses
    limits:
      max_completion_tokens: 1024
  mle-default:
    backend: openai
    model: gpt-5.4
    api: responses
    limits:
      max_completion_tokens: 1024
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_image_config(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "image_profiles.yaml").write_text(
        """
defaults:
  paper: paper-default
  mle: mle-default
profiles:
  paper-default:
    image: registry.example/aisci-paper:latest
    pull_policy: if-missing
  mle-default:
    image: registry.example/aisci-mle:latest
    pull_policy: if-missing
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _paper_job(tmp_path: Path, pdf_path: Path, *, run_final_validation: bool = False) -> JobRecord:
    now = __import__("datetime").datetime.now().astimezone()
    return JobRecord(
        id="paper-job",
        job_type=JobType.PAPER,
        status=JobStatus.PENDING,
        phase=RunPhase.INGEST,
        objective="paper objective",
        llm_profile="paper-default",
        runtime_profile=RuntimeProfile(
            run_final_validation=run_final_validation,
            workspace_layout=WorkspaceLayout.PAPER,
        ),
        mode_spec=PaperSpec(pdf_path=str(pdf_path)),
        created_at=now,
        updated_at=now,
    )


def _mle_job(tmp_path: Path, bundle_path: Path) -> JobRecord:
    now = __import__("datetime").datetime.now().astimezone()
    return JobRecord(
        id="mle-job",
        job_type=JobType.MLE,
        status=JobStatus.PENDING,
        phase=RunPhase.INGEST,
        objective="mle objective",
        llm_profile="mle-default",
        runtime_profile=RuntimeProfile(
            run_final_validation=False,
            workspace_layout=WorkspaceLayout.MLE,
        ),
        mode_spec=MLESpec(workspace_bundle_zip=str(bundle_path)),
        created_at=now,
        updated_at=now,
    )


def test_paper_adapter_stages_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: True)
    monkeypatch.setattr(runtime, "prepare_image", lambda profile, runtime_profile: "paper-image:test")  # noqa: ARG001
    monkeypatch.setattr(
        runtime,
        "start_session",
        lambda spec, image_tag: ContainerSession(  # noqa: ARG005
            container_name="paper-test-session",
            image_tag=image_tag,
            profile=spec.profile,
            runtime_profile=spec.runtime_profile,
            workspace_layout=spec.workspace_layout,
            mounts=spec.mounts,
            workdir=spec.workdir,
        ),
    )
    monkeypatch.setattr(runtime, "cleanup", lambda session: None)

    def fake_run_real_loop(job, job_paths, session, profile) -> None:  # noqa: ANN001,ARG001
        analysis_dir = job_paths.workspace_dir / "agent" / "paper_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in ("summary.md", "structure.md", "algorithm.md", "experiments.md", "baseline.md"):
            (analysis_dir / name).write_text(f"# {name}\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "prioritized_tasks.md").write_text("# priorities\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "plan.md").write_text("# plan\n", encoding="utf-8")
        (job_paths.state_dir / "capabilities.json").write_text("{}", encoding="utf-8")
        (job_paths.state_dir / "resolved_llm_config.json").write_text("{}", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "final_self_check.md").write_text("# self-check\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "final_self_check.json").write_text("{}", encoding="utf-8")
        (job_paths.state_dir / "paper_main_prompt.md").write_text("# prompt\n", encoding="utf-8")
        (job_paths.workspace_dir / "submission" / "reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (job_paths.logs_dir / "agent.log").write_text("agent log\n", encoding="utf-8")
        (job_paths.logs_dir / "conversation.jsonl").write_text("{}\n", encoding="utf-8")
        (job_paths.logs_dir / "paper_session_state.json").write_text("{}", encoding="utf-8")

    adapter = PaperDomainAdapter(runtime)
    monkeypatch.setattr(adapter, "_run_real_loop", fake_run_real_loop)
    result = adapter.run(_paper_job(tmp_path, pdf_path))
    assert result["validation_report"].status == "skipped"
    artifact_types = {item.artifact_type for item in result["artifacts"]}
    assert "paper_analysis" in artifact_types
    assert "prioritized_tasks" in artifact_types
    assert "capabilities" in artifact_types
    assert "resolved_llm_config" in artifact_types
    assert "sandbox_session" in artifact_types
    assert "self_check_report" in artifact_types
    job_paths = ensure_job_dirs(resolve_job_paths("paper-job"))
    assert (job_paths.workspace_dir / "submission" / "reproduce.sh").exists()
    assert (job_paths.workspace_dir / "agent" / "paper_analysis" / "summary.md").exists()
    assert (job_paths.state_dir / "capabilities.json").exists()
    assert (job_paths.workspace_dir / "agent" / "final_self_check.md").exists()


def test_default_paper_profile_uses_repo_image_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)

    profile = default_paper_profile()

    assert profile.image == "registry.example/aisci-paper:latest"
    assert profile.pull_policy == PullPolicy.IF_MISSING


def test_runtime_profile_defaults_to_host_network() -> None:
    assert RuntimeProfile().network_policy == NetworkPolicy.HOST


def test_prepare_image_if_missing_pulls_when_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    _write_image_config(tmp_path)
    runtime = DockerRuntimeManager()
    pulled: list[str] = []
    monkeypatch.setattr(runtime, "image_exists", lambda image_ref: False)
    monkeypatch.setattr(runtime, "pull_image", lambda image_ref: pulled.append(image_ref))

    image_ref = runtime.prepare_image(default_paper_profile(), RuntimeProfile())

    assert image_ref == "registry.example/aisci-paper:latest"
    assert pulled == ["registry.example/aisci-paper:latest"]


def test_paper_adapter_reuses_same_image_for_main_and_validation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)

    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: True)

    prepared_images: list[str] = []
    started_specs = []
    started_tags: list[str] = []

    def fake_prepare_image(profile, runtime_profile):  # noqa: ANN001,ARG001
        prepared_images.append(profile.image)
        return "paper-image:123"

    def fake_start_session(spec, image_tag):  # noqa: ANN001
        started_specs.append(spec)
        started_tags.append(image_tag)
        return ContainerSession(
            container_name="paper-test-session",
            image_tag=image_tag,
            profile=spec.profile,
            runtime_profile=spec.runtime_profile,
            workspace_layout=spec.workspace_layout,
            mounts=spec.mounts,
            workdir=spec.workdir,
        )

    monkeypatch.setattr(runtime, "prepare_image", fake_prepare_image)
    monkeypatch.setattr(runtime, "start_session", fake_start_session)
    monkeypatch.setattr(
        runtime,
        "exec",
        lambda session, command, **kwargs: DockerExecutionResult(  # noqa: ARG005
            command=["docker", "exec", command],
            exit_code=0,
            stdout="ok",
            stderr="",
        ),
    )
    monkeypatch.setattr(runtime, "cleanup", lambda session: None)

    adapter = PaperDomainAdapter(runtime)
    monkeypatch.setattr(
        adapter,
        "_run_real_loop",
        lambda job, job_paths, session, profile: (  # noqa: ARG005
            (job_paths.workspace_dir / "submission" / "reproduce.sh").write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
            ),
            (job_paths.state_dir / "resolved_llm_config.json").write_text("{}", encoding="utf-8"),
        ),
    )
    result = adapter.run(_paper_job(tmp_path, pdf_path, run_final_validation=True))

    assert result["validation_report"].status == "passed"
    assert result["validation_report"].container_image == "paper-image:123"
    assert prepared_images == ["registry.example/aisci-paper:latest"]
    assert started_tags == ["paper-image:123", "paper-image:123"]
    assert len(started_specs) == 2
    assert all(spec.profile.image == "registry.example/aisci-paper:latest" for spec in started_specs)
    assert all(dict(spec.env)["HTTP_PROXY"] == "http://proxy.example:3128" for spec in started_specs)
    assert all(dict(spec.env)["HF_TOKEN"] == "hf_test_token" for spec in started_specs)


def test_paper_adapter_only_forwards_optional_runtime_envs_when_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("HF_TOKEN", "hf_secondary")
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    adapter = PaperDomainAdapter(DockerRuntimeManager())

    forwarded = adapter._sandbox_env(_paper_job(tmp_path, pdf_path))

    assert forwarded["AISCI_JOB_ID"] == "paper-job"
    assert forwarded["AISCI_OBJECTIVE"] == "paper objective"
    assert forwarded["LOGS_DIR"] == "/home/logs"
    assert forwarded["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert forwarded["HF_TOKEN"] == "hf_secondary"
    assert "HTTP_PROXY" not in forwarded


def test_paper_adapter_requires_docker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: False)
    adapter = PaperDomainAdapter(runtime)

    try:
        adapter.run(_paper_job(tmp_path, pdf_path))
    except DockerRuntimeError as exc:
        assert "No local fallback loop is available" in str(exc)
    else:
        raise AssertionError("Expected DockerRuntimeError when Docker is unavailable")


def test_mle_adapter_stages_summary_and_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)
    bundle = tmp_path / "workspace.zip"
    with ZipFile(bundle, "w") as zf:
        zf.writestr("description.md", "# task")
        zf.writestr("sample_submission.csv", "id,target\n1,0\n")
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: True)

    def fake_run_real_loop(job, job_paths) -> None:  # noqa: ANN001
        analysis_dir = job_paths.workspace_dir / "agent" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        submission_dir = job_paths.workspace_dir / "submission"
        candidates_dir = submission_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "summary.md").write_text("# analysis\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "prioritized_tasks.md").write_text("# priorities\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "impl_log.md").write_text("# impl\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "exp_log.md").write_text("# exp\n", encoding="utf-8")
        (submission_dir / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
        candidate = candidates_dir / "submission_001_final.csv"
        candidate.write_text("id,target\n1,0\n", encoding="utf-8")
        (submission_dir / "submission_registry.jsonl").write_text(
            '{"event":"system_snapshot"}\n{"event":"candidate_detail"}\n{"event":"champion_selected"}\n',
            encoding="utf-8",
        )
        (job_paths.logs_dir / "agent.log").write_text("agent log\n", encoding="utf-8")
        (job_paths.logs_dir / "conversation.jsonl").write_text("{}\n", encoding="utf-8")
        (job_paths.logs_dir / "summary.json").write_text("{}", encoding="utf-8")
        (job_paths.artifacts_dir / "champion_report.md").write_text("# Champion Report\n", encoding="utf-8")

    adapter = MLEDomainAdapter(runtime)
    monkeypatch.setattr(adapter, "_run_real_loop", fake_run_real_loop)
    result = adapter.run(_mle_job(tmp_path, bundle))
    assert result["validation_report"].status == "skipped"
    artifact_types = {item.artifact_type for item in result["artifacts"]}
    assert "champion_report" in artifact_types
    assert "submission_registry" in artifact_types
    assert "candidate_snapshot" in artifact_types
    job_paths = ensure_job_dirs(resolve_job_paths("mle-job"))
    registry_text = (job_paths.workspace_dir / "submission" / "submission_registry.jsonl").read_text(
        encoding="utf-8"
    )
    assert "champion_selected" in registry_text
    assert (job_paths.workspace_dir / "submission" / "submission.csv").exists()


def test_mle_adapter_requires_docker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_llm_config(tmp_path)
    _write_image_config(tmp_path)
    bundle = tmp_path / "workspace.zip"
    with ZipFile(bundle, "w") as zf:
        zf.writestr("description.md", "# task")
        zf.writestr("sample_submission.csv", "id,target\n1,0\n")
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: False)
    adapter = MLEDomainAdapter(runtime)

    try:
        adapter.run(_mle_job(tmp_path, bundle))
    except DockerRuntimeError as exc:
        assert "No local fallback loop is available" in str(exc)
    else:
        raise AssertionError("Expected DockerRuntimeError when Docker is unavailable")


def test_runtime_session_spec_uses_canonical_mounts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    _write_image_config(tmp_path)
    runtime = DockerRuntimeManager()
    job_paths = ensure_job_dirs(resolve_job_paths("layout-job"))
    runtime.ensure_layout(job_paths, WorkspaceLayout.MLE)
    spec = runtime.create_session_spec(
        "layout-job",
        job_paths,
        default_mle_profile(),
        RuntimeProfile(workspace_layout=WorkspaceLayout.MLE),
        layout=WorkspaceLayout.MLE,
        workdir="/home/code",
    )
    targets = {mount.target for mount in spec.mounts}
    assert "/home" in targets
    assert "/home/logs" in targets
    assert "/workspace/logs" in targets
    assert spec.workdir == "/home/code"


def test_start_session_applies_user_limits_network_and_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    _write_image_config(tmp_path)
    runtime = DockerRuntimeManager()
    job_paths = ensure_job_dirs(resolve_job_paths("runtime-job"))
    runtime.ensure_layout(job_paths, WorkspaceLayout.PAPER)
    monkeypatch.setattr("aisci_runtime_docker.agent_session.os.getuid", lambda: 1234)
    monkeypatch.setattr("aisci_runtime_docker.agent_session.os.getgid", lambda: 2345)
    commands: list[list[str]] = []

    def fake_run(command, check=True, timeout=None):  # noqa: ANN001,ARG001
        commands.append(command)
        return DockerExecutionResult(command=command, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(runtime, "_run", fake_run)
    spec = runtime.create_session_spec(
        "runtime-job",
        job_paths,
        default_paper_profile(),
        RuntimeProfile(
            workspace_layout=WorkspaceLayout.PAPER,
            uid_gid_mode=UIDGIDMode.HOST,
            network_policy=NetworkPolicy.NONE,
            cpu_limit="4",
            memory_limit="8g",
        ),
        layout=WorkspaceLayout.PAPER,
        workdir="/home/submission",
    )

    session = runtime.start_session(spec, "paper-image:test")
    command = commands[0]

    assert session.run_as_user == "1234:2345"
    assert "--user" in command
    assert "1234:2345" in command
    assert "--cpus" in command
    assert "4" in command
    assert "--memory" in command
    assert "8g" in command
    assert "--network" in command
    assert "none" in command
    assert "--label" in command
    assert "aisci.job_id=runtime-job" in command
    assert "aisci.workspace_layout=paper" in command
    assert "aisci.image_profile=paper-default" in command


def test_start_session_prefers_explicit_gpu_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    _write_image_config(tmp_path)
    runtime = DockerRuntimeManager()
    job_paths = ensure_job_dirs(resolve_job_paths("runtime-job-gpu-ids"))
    runtime.ensure_layout(job_paths, WorkspaceLayout.PAPER)
    commands: list[list[str]] = []

    def fake_run(command, check=True, timeout=None):  # noqa: ANN001,ARG001
        commands.append(command)
        return DockerExecutionResult(command=command, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(runtime, "_run", fake_run)
    spec = runtime.create_session_spec(
        "runtime-job-gpu-ids",
        job_paths,
        default_paper_profile(),
        RuntimeProfile(
            workspace_layout=WorkspaceLayout.PAPER,
            gpu_count=2,
            gpu_ids=["4", "5"],
        ),
        layout=WorkspaceLayout.PAPER,
        workdir="/home/submission",
    )

    runtime.start_session(spec, "paper-image:test")
    command = commands[0]

    assert "--gpus" in command
    assert "device=4,5" in command
    assert "2" not in command[command.index("--gpus") + 1]
