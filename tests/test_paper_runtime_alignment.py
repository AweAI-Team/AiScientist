from __future__ import annotations

import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_agent_runtime.tools.constraints import build_blocked_patterns_from_blacklist
from aisci_domain_paper.configs import (
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
    DEFAULT_PRIORITIZATION_CONFIG,
    ENV_SETUP_BASH_DEFAULT_TIMEOUT,
    ENV_SETUP_BASH_MAX_TIMEOUT,
    RESOURCE_DOWNLOAD_BASH_DEFAULT_TIMEOUT,
    RESOURCE_DOWNLOAD_BASH_MAX_TIMEOUT,
)
from aisci_domain_paper.engine import EmbeddedPaperEngine, PaperRuntimeConfig
from aisci_domain_paper.subagents import subagent_class_for_kind
from aisci_domain_paper.subagents.coordinator import SubagentCoordinator, SubagentTask
from aisci_domain_paper.subagents.paper_reader import PaperReaderCoordinator
from aisci_domain_paper.subagents.prioritization import PaperPrioritizationSubagent, PrioritizationRunner
from aisci_domain_paper.state_manager import PaperStateManager
from aisci_domain_paper.tools.basic_tool import (
    LinkSummaryTool,
    ParseRubricTool,
    PriorityWriteTool,
    WebSearchTool,
    build_explore_tools,
    build_general_tools,
    build_implementation_tools,
    build_main_direct_tools,
    build_env_setup_tools,
    build_prioritization_tools,
    build_reader_tools,
    build_plan_tools,
    build_resource_download_tools,
)
from aisci_domain_paper.tools.clean_validation_tool import CleanReproduceValidationTool
from aisci_domain_paper.tools.implementation_tool import ImplementationTool
from aisci_domain_paper.tools.paper_reader_tool import ReadPaperTool
from aisci_domain_paper.tools.prioritization_tool import PrioritizeTasksTool
from aisci_domain_paper.tools.spawn_subagent_tool import build_main_tools, build_spawn_subagent_tool
from aisci_runtime_docker.models import ContainerSession, DockerProfile, SessionMount
from aisci_runtime_docker.shell_interface import DockerShellInterface
from aisci_agent_runtime.tools.shell_tools import ReadFileChunkTool, SearchFileTool


class _StubEngine:
    def _capabilities(self):
        return {
            "online_research": {"available": True},
        }

    def read_paper(self, paper_path: str = "/home/paper/paper.md"):  # noqa: ARG002
        return "read"

    def prioritize_tasks(
        self,
        paper_analysis_dir: str = "/home/agent/paper_analysis",  # noqa: ARG002
        rubric_path: str = "/home/paper/rubric.json",  # noqa: ARG002
        focus_areas: str | None = None,  # noqa: ARG002
    ):
        return "prioritize"

    def run_implementation(self, **kwargs):
        return f"implement:{kwargs['task']}"

    def run_experiment(self, **kwargs):
        return f"experiment:{kwargs['task']}"

    def run_named_subagent(self, **kwargs):
        return f"subagent:{kwargs['subagent_type']}"

    def run_clean_validation(self):  # noqa: ARG002
        return "clean"


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | None, dict | None]] = []

    def event(self, name: str, message: str, phase: str | None = None, payload: dict | None = None) -> None:
        self.events.append((name, message, phase, payload))


def test_main_tools_expose_direct_solver_tools() -> None:
    tools = build_main_tools(_StubEngine())
    names = {tool.name() for tool in tools}
    assert {"bash", "python", "read_file_chunk", "search_file"} <= names
    assert {
        "read_paper",
        "prioritize_tasks",
        "implement",
        "run_experiment",
        "spawn_subagent",
        "clean_reproduce_validation",
        "submit",
    } <= names
    assert "finish_run" not in names
    assert "search_paper" not in names


def test_reader_and_prioritization_configs_match_upstream_scale() -> None:
    assert DEFAULT_PAPER_STRUCTURE_CONFIG is not DEFAULT_PAPER_READER_CONFIG
    assert DEFAULT_PAPER_READER_CONFIG is not DEFAULT_PAPER_SYNTHESIS_CONFIG
    assert DEFAULT_PAPER_STRUCTURE_CONFIG is not DEFAULT_PRIORITIZATION_CONFIG
    assert DEFAULT_PAPER_STRUCTURE_CONFIG.max_steps == 500
    assert DEFAULT_PAPER_READER_CONFIG.max_steps == 500
    assert DEFAULT_PAPER_SYNTHESIS_CONFIG.max_steps == 500
    assert DEFAULT_PRIORITIZATION_CONFIG.max_steps == 500
    assert DEFAULT_PAPER_STRUCTURE_CONFIG.time_limit == 36_000
    assert DEFAULT_PAPER_READER_CONFIG.time_limit == 36_000
    assert DEFAULT_PAPER_SYNTHESIS_CONFIG.time_limit == 36_000
    assert DEFAULT_PRIORITIZATION_CONFIG.time_limit == 36_000


def test_spawn_subagent_contract_matches_upstream_scope() -> None:
    schema = build_spawn_subagent_tool(_StubEngine()).get_tool_schema()
    values = schema["function"]["parameters"]["properties"]["subagent_type"]["enum"]
    assert values == ["explore", "plan", "general"]


def test_generic_subagent_tool_permissions_are_split() -> None:
    explore = {tool.name() for tool in build_explore_tools({"online_research": {"available": True}})}
    plan = {tool.name() for tool in build_plan_tools({"online_research": {"available": True}})}
    general = {tool.name() for tool in build_general_tools({"online_research": {"available": True}})}
    implementation = {tool.name() for tool in build_implementation_tools({"online_research": {"available": True}})}
    reader = {tool.name() for tool in build_reader_tools({"online_research": {"available": True}})}

    assert "edit_file" not in explore
    assert "write_plan" in plan
    assert "edit_file" not in general
    assert "python" in general
    assert "github" not in implementation
    assert "linter" in implementation
    main_direct = {tool.name() for tool in build_main_direct_tools({"online_research": {"available": True}})}
    assert {"web_search", "link_summary"} <= main_direct
    assert "python" not in reader
    assert reader >= {"read_file_chunk", "search_file", "bash", "subagent_complete"}
    prioritization = {tool.name() for tool in build_prioritization_tools({"online_research": {"available": True}})}
    assert prioritization == {"read_file_chunk", "search_file", "parse_rubric", "write_priorities", "subagent_complete"}


def test_validation_subagent_kind_is_not_exposed() -> None:
    with pytest.raises(KeyError):
        subagent_class_for_kind("validation")
    with pytest.raises(KeyError):
        subagent_class_for_kind("reader")
    with pytest.raises(KeyError):
        subagent_class_for_kind("paper_reader")


def test_tool_schemas_match_upstream_main_contract() -> None:
    read_schema = ReadPaperTool(_StubEngine()).get_tool_schema()["function"]["parameters"]
    prioritize_schema = PrioritizeTasksTool(_StubEngine()).get_tool_schema()["function"]["parameters"]
    implement_schema = ImplementationTool(_StubEngine()).get_tool_schema()["function"]["parameters"]
    clean_schema = CleanReproduceValidationTool(_StubEngine()).get_tool_schema()["function"]["parameters"]
    web_schema = WebSearchTool().get_tool_schema()["function"]["parameters"]
    link_schema = LinkSummaryTool().get_tool_schema()["function"]["parameters"]
    priorities_schema = PriorityWriteTool().get_tool_schema()["function"]["parameters"]
    rubric_schema = ParseRubricTool().get_tool_schema()["function"]["parameters"]

    assert set(read_schema["properties"]) == {"paper_path"}
    assert set(prioritize_schema["properties"]) == {"paper_analysis_dir", "rubric_path", "focus_areas"}
    assert implement_schema["required"] == []
    assert set(clean_schema["properties"]) == {"time_budget"}
    assert "query" in web_schema["properties"]
    assert "num" in web_schema["properties"]
    assert set(link_schema["properties"]) == {"url", "goal"}
    assert link_schema["required"] == ["url", "goal"]
    assert set(priorities_schema["properties"]) == {"content"}
    assert set(rubric_schema["properties"]) == {"rubric_path", "max_depth"}


def test_clean_validation_report_is_detailed_for_main_agent(tmp_path: Path) -> None:
    class Engine:
        def __init__(self) -> None:
            self.self_check_path = tmp_path / "final_self_check.md"
            self.self_check_json_path = tmp_path / "final_self_check.json"
            self.trace = _TraceRecorder()

    tool = CleanReproduceValidationTool(Engine())
    report = tool._build_agent_report(  # noqa: SLF001
        result=SubagentOutput(
            status=SubagentStatus.FAILED,
            content="reproduce.sh failed to install dependencies",
            error_message="pip install failed",
            num_steps=4,
            runtime_seconds=12.5,
            log_path="/tmp/clean_val.log",
        ),
        cleanup_summary="$ git clean -fd\n[ok]\nremoved tmp file",
        hardcoded_hits=["src/train.py:12:/home/submission/data"],
        experiment_result="[Clean Validation Experiment | failed]\n\nLog: /tmp/clean_val.log\n\nreproduce.sh failed to install dependencies",
    )

    assert "## Environment Cleanup" in report
    assert "## Hardcoded Path Check" in report
    assert "## Validation Results" in report
    assert "## What's Next?" in report
    assert "implement(mode=\"fix\"" in report


def test_docker_shell_interface_exposes_host_mapping_for_mounted_paths(tmp_path: Path) -> None:
    summary_path = tmp_path / "agent" / "paper_analysis" / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("# Summary\n", encoding="utf-8")
    shell = DockerShellInterface(
        runtime=SimpleNamespace(),
        session=ContainerSession(
            container_name="paper-test",
            image_tag="aisci-paper:latest",
            profile=DockerProfile(name="paper", image="aisci-paper:latest"),
            runtime_profile=SimpleNamespace(),
            workspace_layout="paper",
            mounts=(SessionMount(source=tmp_path, target="/home"),),
            workdir="/home/submission",
        ),
    )

    assert shell.file_exists("/home/agent/paper_analysis/summary.md") is True
    assert shell.read_file("/home/agent/paper_analysis/summary.md") == "# Summary\n"
    assert shell.download("/home/agent/paper_analysis/summary.md") == b"# Summary\n"
    assert shell._mounted_host_path("/home/a/new_dir") == tmp_path / "a" / "new_dir"


def test_state_manager_tracks_session_boundaries_and_recent_history(tmp_path: Path) -> None:
    state = PaperStateManager(
        agent_dir=tmp_path / "agent",
        logs_dir=tmp_path / "logs",
        subagent_logs_dir=tmp_path / "logs" / "subagent_logs",
    )
    state.ensure_logs()
    assert not state.impl_log_path.exists()
    assert not state.exp_log_path.exists()
    assert not state.subagent_logs_dir.exists()
    session = state.create_session("implementation")
    state.append_separator(session)
    state.append_session_note("implementation", "Implemented core loop", "Changed reproduce.sh and training entrypoint.")
    recent = state.recent_impl_history()
    assert "Implement Session 1" in recent
    assert "Implemented core loop" in recent


def test_engine_workspace_bootstrap_uses_lazy_file_creation(tmp_path: Path) -> None:
    engine = EmbeddedPaperEngine(
        config=PaperRuntimeConfig(
            job_id="paper-job",
            objective="reproduce paper",
            llm_profile_name="paper-default",
        ),
        shell=object(),
        llm=None,
        paper_dir=tmp_path / "paper",
        submission_dir=tmp_path / "submission",
        agent_dir=tmp_path / "agent",
        logs_dir=tmp_path / "logs",
        state_dir=tmp_path / "state",
        trace=_TraceRecorder(),
    )

    engine._ensure_workspace()

    assert engine.paper_dir.exists()
    assert engine.submission_dir.exists()
    assert engine.agent_dir.exists()
    assert engine.logs_dir.exists()
    assert (engine.submission_dir / ".gitignore").exists()
    assert (engine.submission_dir / ".git").exists()
    assert not engine.analysis_dir.exists()
    assert not engine.subagent_logs_dir.exists()
    assert not engine.reproduce_path.exists()
    assert not engine.state_manager.impl_log_path.exists()
    assert not engine.state_manager.exp_log_path.exists()
    assert not engine.prompt_path.exists()
    assert not engine.capability_path.exists()
    assert not (engine.submission_dir / "README.md").exists()


def test_engine_subagent_config_uses_host_paths(tmp_path: Path) -> None:
    engine = EmbeddedPaperEngine(
        config=PaperRuntimeConfig(
            job_id="paper-job",
            objective="reproduce paper",
            llm_profile_name="paper-default",
        ),
        shell=object(),
        llm=None,
        paper_dir=tmp_path / "paper",
        submission_dir=tmp_path / "submission",
        agent_dir=tmp_path / "agent",
        logs_dir=tmp_path / "logs",
        state_dir=tmp_path / "state",
        trace=_TraceRecorder(),
    )

    cfg = engine.subagent_config("paper_reader", DEFAULT_PAPER_READER_CONFIG)

    assert cfg.log_dir == str(tmp_path / "logs" / "subagent_logs")
    assert cfg.output_dir == str(tmp_path / "agent")


def test_run_subagent_output_converts_exceptions_to_failed_result(tmp_path: Path) -> None:
    engine = EmbeddedPaperEngine(
        config=PaperRuntimeConfig(
            job_id="paper-job",
            objective="reproduce paper",
            llm_profile_name="paper-default",
        ),
        shell=object(),
        llm=object(),
        paper_dir=tmp_path / "paper",
        submission_dir=tmp_path / "submission",
        agent_dir=tmp_path / "agent",
        logs_dir=tmp_path / "logs",
        state_dir=tmp_path / "state",
        trace=_TraceRecorder(),
    )
    engine._ensure_workspace()

    class BrokenSubagent:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN003
            self.config = args[3]

        def build_context(self) -> str:
            return "context"

        def run(self, context: str = "") -> SubagentOutput:  # noqa: ARG002
            raise PermissionError("simulated host path failure")

    result = engine.run_subagent_output(
        BrokenSubagent,
        objective="broken",
        context="",
        config=DEFAULT_PAPER_READER_CONFIG,
        phase="analyze",
        label="broken_reader",
    )

    assert result.status == SubagentStatus.FAILED
    assert result.error_message == "simulated host path failure"
    assert any(
        name == "subagent_finish" and payload and payload.get("error") == "simulated host path failure"
        for name, _message, _phase, payload in engine.trace.events
    )


def test_reader_coordinator_returns_navigation_summary(tmp_path: Path, monkeypatch) -> None:
    class Engine:
        def __init__(self) -> None:
            self.analysis_dir = tmp_path / "agent" / "paper_analysis"
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            self.shell = object()
            self.llm = object()
            self.trace = _TraceRecorder()

        def reader_context(self) -> str:
            return "Objective:\nReproduce the paper."

        def subagent_config(self, kind: str, base):  # noqa: ARG002
            return base

        def _capabilities(self) -> dict[str, dict[str, bool]]:
            return {
                "online_research": {"available": True},
            }

        def constraints(self) -> dict[str, list[str]]:
            return {"blacklist": []}

    engine = Engine()
    calls: list[tuple[str, int, int]] = []

    def fake_run_stage(self, subagent_cls, *, objective, context, config, phase, label, llm=None):  # noqa: ANN001,ARG001
        calls.append((label, config.max_steps, config.time_limit))
        return SubagentOutput(
            status=SubagentStatus.COMPLETED,
            content=f"{label} findings",
            num_steps=3,
            runtime_seconds=1.5,
            log_path=f"/tmp/{label}.jsonl",
        )

    monkeypatch.setattr(PaperReaderCoordinator, "_run_stage", fake_run_stage, raising=False)

    result = PaperReaderCoordinator(engine).run()

    assert sorted(label for label, _, _ in calls) == sorted([
        "paper_structure",
        "paper_algorithm",
        "paper_experiments",
        "paper_baseline",
        "paper_synthesis",
    ])
    assert all(max_steps == 500 for _, max_steps, _ in calls)
    assert all(time_limit == 36_000 for _, _, time_limit in calls)
    assert result.all_success is True
    assert result.failed_subagents == []
    assert result.executive_summary == "paper_synthesis findings"
    assert "Detailed Analysis Files" in result.summary_with_navigation
    assert "/home/agent/paper_analysis/structure.md" in result.summary_with_navigation
    assert (engine.analysis_dir / "summary.md").exists()
    assert "paper_structure findings" in (engine.analysis_dir / "structure.md").read_text(encoding="utf-8")


def test_subagent_coordinator_runs_parallel_dependency_levels() -> None:
    coordinator = SubagentCoordinator()

    def task(name: str, delay: float, dependencies: list[str] | None = None) -> SubagentTask:
        return SubagentTask(
            name=name,
            dependencies=dependencies or [],
            run_fn=lambda _ctx: _timed_output(name, delay),
        )

    started = time.perf_counter()
    result = coordinator.run(
        [
            task("structure", 0.05),
            task("algorithm", 0.20, ["structure"]),
            task("experiments", 0.20, ["structure"]),
            task("baseline", 0.20, ["structure"]),
            task("synthesis", 0.05, ["algorithm", "experiments", "baseline"]),
        ]
    )
    elapsed = time.perf_counter() - started

    assert result.all_success is True
    assert elapsed < 0.55


def _timed_output(name: str, delay: float) -> SubagentOutput:
    time.sleep(delay)
    return SubagentOutput(
        status=SubagentStatus.COMPLETED,
        content=f"{name} done",
        token_usage={"input": 1, "output": 1},
    )


def test_prioritization_runner_builds_structured_task_description(tmp_path: Path, monkeypatch) -> None:
    class Engine:
        def __init__(self) -> None:
            self.paper_dir = tmp_path / "paper"
            self.analysis_dir = tmp_path / "agent" / "paper_analysis"
            self.prioritized_path = tmp_path / "agent" / "prioritized_tasks.md"
            self.plan_path = tmp_path / "agent" / "plan.md"
            self.paper_dir.mkdir(parents=True, exist_ok=True)
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            self.prioritized_path.parent.mkdir(parents=True, exist_ok=True)
            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            class Shell:
                def file_exists(self_inner, path: str) -> bool:
                    mapping = {
                        "/home/agent/paper_analysis/summary.md": self.analysis_dir / "summary.md",
                        "/home/paper/rubric.json": self.paper_dir / "rubric.json",
                        "/home/paper/addendum.md": self.paper_dir / "addendum.md",
                        "/home/paper/blacklist.txt": self.paper_dir / "blacklist.txt",
                    }
                    return mapping.get(path, Path(path)).exists()

                def read_file(self_inner, path: str) -> str:
                    mapping = {
                        "/home/agent/paper_analysis/summary.md": self.analysis_dir / "summary.md",
                        "/home/paper/rubric.json": self.paper_dir / "rubric.json",
                        "/home/paper/addendum.md": self.paper_dir / "addendum.md",
                        "/home/paper/blacklist.txt": self.paper_dir / "blacklist.txt",
                    }
                    return mapping.get(path, Path(path)).read_text(encoding="utf-8")

            self.shell = Shell()
            self.llm = object()
            self.config = SimpleNamespace(objective="Reproduce the paper")
            self.trace = _TraceRecorder()
            (self.paper_dir / "rubric.json").write_text('{"name": "core", "weight": 10}', encoding="utf-8")
            (self.paper_dir / "addendum.md").write_text("Use official datasets only.", encoding="utf-8")
            (self.paper_dir / "blacklist.txt").write_text("http://blocked.example", encoding="utf-8")
            (self.analysis_dir / "summary.md").write_text("# Summary", encoding="utf-8")
            (self.analysis_dir / "structure.md").write_text("# Structure", encoding="utf-8")
            (self.analysis_dir / "algorithm.md").write_text("# Algorithm", encoding="utf-8")
            (self.analysis_dir / "experiments.md").write_text("# Experiments", encoding="utf-8")
            (self.analysis_dir / "baseline.md").write_text("# Baseline", encoding="utf-8")

        def _capabilities(self) -> dict[str, dict[str, bool]]:
            return {
                "online_research": {"available": True},
            }

        def _read_text(self, path: Path, limit: int = 8_000) -> str:  # noqa: ARG002
            return path.read_text(encoding="utf-8") if path.exists() else ""

        def read_paper(self, paper_path: str = "/home/paper/paper.md"):  # noqa: ARG002
            return "# Summary\n"

        def subagent_config(self, kind: str, base):  # noqa: ARG002
            return base

        def run_subagent_output(
            self,
            cls,  # noqa: ARG002
            *,
            objective: str,  # noqa: ARG002
            context: str,  # noqa: ARG002
            config,  # noqa: ARG002
            phase: str,  # noqa: ARG002
            label: str,  # noqa: ARG002
            session=None,  # noqa: ARG002
            llm_override=None,  # noqa: ARG002
        ) -> SubagentOutput:
            self.prioritized_path.write_text(
                "# Prioritized Implementation Plan\n\n## Executive Summary\n- P0: core method.\n",
                encoding="utf-8",
            )
            return SubagentOutput(
                status=SubagentStatus.COMPLETED,
                content="# Prioritized Implementation Plan\n\n## Executive Summary\n- P0: core method.\n",
                num_steps=6,
                runtime_seconds=2.0,
                log_path=str(tmp_path / "logs" / "subagent_logs" / "prioritization.jsonl"),
            )

        def _capabilities(self) -> dict[str, dict[str, bool]]:
            return {
                "online_research": {"available": True},
            }

        def constraints(self) -> dict[str, list[str]]:
            return {"blacklist": []}

    engine = Engine()

    result = PrioritizationRunner(engine).run()

    assert "parse_rubric" in result.task_description
    assert "write_priorities" in result.task_description
    assert "Treat each model variant as a separate task." in result.task_description
    assert result.prioritized_path.exists()
    assert result.plan_path is None
    assert "Prioritized tasks saved to" in result.summary
    assert "- P0: core method." in result.summary


def test_engine_constraints_build_blocked_search_patterns() -> None:
    blocked = build_blocked_patterns_from_blacklist(
        ["https://github.com/org/repo", "*.blocked.example/path"]
    )
    assert "url" in blocked
    assert any("github\\.com/org/repo" in pattern for pattern in blocked["url"])
    assert any("blocked\\.example/path" in pattern for pattern in blocked["url"])


def test_web_search_filters_blocked_urls(monkeypatch) -> None:
    html = """
    <a class="result__a" href="https://allowed.example/doc">Allowed Result</a>
    <div class="result__snippet">allowed snippet</div>
    <a class="result__a" href="https://blocked.example/doc">Blocked Result</a>
    <div class="result__snippet">blocked snippet</div>
    """

    monkeypatch.setattr("aisci_agent_runtime.tools.research_tools._fetch_raw", lambda _url: html)

    result = WebSearchTool().execute(
        None,
        query="paperbench reader",
        constraints={"blocked_search_patterns": build_blocked_patterns_from_blacklist(["blocked.example"])},
    )

    assert "Allowed Result" in result
    assert "Blocked Result" not in result
    assert "filtered out" in result


def test_link_summary_blocks_blacklisted_url() -> None:
    result = LinkSummaryTool().execute(
        None,
        url="https://blocked.example/resource",
        goal="Inspect the docs",
        constraints={"blocked_search_patterns": build_blocked_patterns_from_blacklist(["blocked.example"])},
    )
    assert "ACCESS DENIED" in result


def test_read_file_chunk_missing_file_is_nonfatal() -> None:
    class Shell:
        def file_exists(self, path: str) -> bool:  # noqa: ARG002
            return False

    result = ReadFileChunkTool().execute(Shell(), "/home/paper/addendum.md")

    assert result == "File not found: /home/paper/addendum.md. It may be optional or not staged for this run."


def test_search_file_missing_path_is_nonfatal() -> None:
    class Shell:
        def file_exists(self, path: str) -> bool:  # noqa: ARG002
            return False

    result = SearchFileTool().execute(Shell(), pattern="PINN", path="/home/paper/addendum.md")

    assert result == "File not found: /home/paper/addendum.md. It may be optional or not staged for this run."


def test_env_setup_and_resource_download_tools_use_configured_timeouts() -> None:
    env_bash = next(tool for tool in build_env_setup_tools() if tool.name() == "bash")
    download_bash = next(tool for tool in build_resource_download_tools() if tool.name() == "bash")

    assert env_bash.default_timeout == ENV_SETUP_BASH_DEFAULT_TIMEOUT
    assert env_bash.max_timeout == ENV_SETUP_BASH_MAX_TIMEOUT
    assert download_bash.default_timeout == RESOURCE_DOWNLOAD_BASH_DEFAULT_TIMEOUT
    assert download_bash.max_timeout == RESOURCE_DOWNLOAD_BASH_MAX_TIMEOUT
