from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from aisci_domain_mle.local_runtime_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

from aisci_agent_runtime.llm_client import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    ToolCallResult,
    create_llm_client,
)
from aisci_domain_mle.constants import is_file_as_bus_enabled

STUB_ENV_KEYS = (
    "AISCI_STUB_LLM",
    "AISCI_STUB_SCENARIO",
    "LOGS_DIR",
    "FILE_AS_BUS",
)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def runtime_uses_stub_llm(env: Mapping[str, str] | None = None) -> bool:
    raw_env = env or os.environ
    return _truthy(raw_env.get("AISCI_STUB_LLM"))


def _parse_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass(frozen=True)
class OrchestratorPaths:
    home_root: str
    data_dir: str
    code_dir: str
    submission_dir: str
    agent_dir: str
    logs_dir: str

    @property
    def description_path(self) -> str:
        return f"{self.data_dir}/description.md"

    @property
    def sample_submission_path(self) -> str:
        return f"{self.data_dir}/sample_submission.csv"

    @property
    def analysis_dir(self) -> str:
        return f"{self.agent_dir}/analysis"

    @property
    def analysis_summary_path(self) -> str:
        return f"{self.analysis_dir}/summary.md"

    @property
    def prioritized_tasks_path(self) -> str:
        return f"{self.agent_dir}/prioritized_tasks.md"

    @property
    def impl_log_path(self) -> str:
        return f"{self.agent_dir}/impl_log.md"

    @property
    def exp_log_path(self) -> str:
        return f"{self.agent_dir}/exp_log.md"

    @property
    def experiments_dir(self) -> str:
        return f"{self.agent_dir}/experiments"

    @property
    def submission_csv_path(self) -> str:
        return f"{self.submission_dir}/submission.csv"

    @property
    def submission_candidates_dir(self) -> str:
        return f"{self.submission_dir}/candidates"

    @property
    def submission_registry_path(self) -> str:
        return f"{self.submission_dir}/submission_registry.jsonl"

    @property
    def agent_env_path(self) -> str:
        return f"{self.agent_dir}/env.json"

    @property
    def agent_summary_path(self) -> str:
        return f"{self.agent_dir}/summary.json"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OrchestratorPaths:
        raw_env = env or os.environ
        home_root = str(Path(raw_env.get("AISCI_HOME_ROOT", "/home")).resolve())
        logs_dir = raw_env.get("LOGS_DIR")
        if logs_dir:
            resolved_logs_dir = str(Path(logs_dir).resolve())
        else:
            resolved_logs_dir = str((Path(home_root) / "logs").resolve())
        return cls(
            home_root=home_root,
            data_dir=str((Path(home_root) / "data").resolve()),
            code_dir=str((Path(home_root) / "code").resolve()),
            submission_dir=str((Path(home_root) / "submission").resolve()),
            agent_dir=str((Path(home_root) / "agent").resolve()),
            logs_dir=resolved_logs_dir,
        )


@dataclass(frozen=True)
class OrchestratorRuntimeConfig:
    time_limit: int
    max_steps: int
    reminder_freq: int
    model: str
    hardware: str
    api_mode: str
    context_reduce_strategy: str
    summary_segment_ratio: float
    summary_min_turns: int
    summary_segment_max_chars: int
    summary_incremental: bool
    file_as_bus: bool
    paths: OrchestratorPaths
    validation_command: str | None = None
    stub_scenario: str | None = None


def load_runtime_config_from_env(env: Mapping[str, str] | None = None) -> OrchestratorRuntimeConfig:
    raw_env = env or os.environ
    paths = OrchestratorPaths.from_env(raw_env)
    return OrchestratorRuntimeConfig(
        time_limit=_parse_int(raw_env, "TIME_LIMIT_SECS", 14400),
        max_steps=_parse_int(raw_env, "AISCI_MAX_STEPS", 500),
        reminder_freq=_parse_int(raw_env, "AISCI_REMINDER_FREQ", 5),
        model=raw_env.get("AISCI_MODEL", "gpt-5.2-2025-12-11"),
        hardware=raw_env.get("HARDWARE", "unknown"),
        api_mode=raw_env.get("AISCI_API_MODE", "completions"),
        context_reduce_strategy=(raw_env.get("AISCI_CONTEXT_REDUCE_STRATEGY", "summary") or "summary").strip().lower(),
        summary_segment_ratio=float(raw_env.get("AISCI_SUMMARY_SEGMENT_RATIO", "0.3")),
        summary_min_turns=_parse_int(raw_env, "AISCI_SUMMARY_MIN_TURNS_TO_SUMMARIZE", 4),
        summary_segment_max_chars=_parse_int(raw_env, "AISCI_SUMMARY_SEGMENT_MAX_CHARS", 25000),
        summary_incremental=(raw_env.get("AISCI_SUMMARY_INCREMENTAL", "true") or "true").strip().lower() in ("true", "1", "yes"),
        file_as_bus=is_file_as_bus_enabled(),
        paths=paths,
        validation_command=(raw_env.get("AISCI_SUBMISSION_VALIDATE_COMMAND") or "").strip() or None,
        stub_scenario=(raw_env.get("AISCI_STUB_SCENARIO") or "submit_sample").strip() or "submit_sample",
    )


def build_task_prompt(config: OrchestratorRuntimeConfig) -> str:
    gpu_line = (
        f"**Hardware**: {config.hardware}"
        if config.hardware != "unknown"
        else "**Hardware**: Check with `nvidia-smi`"
    )
    prompt = (
        f"You have **{_fmt(config.time_limit)}** to solve this Kaggle competition.\n\n"
        f"{gpu_line}\n\n"
        f"Competition data is in `{config.paths.data_dir}/`. Read `{config.paths.description_path}` first.\n"
        f"Your code goes in `{config.paths.code_dir}/` (git repo). Final submission goes to `{config.paths.submission_csv_path}`.\n\n"
        "Start by analysing the data and creating a prioritized task list, then implement and validate."
    )
    if config.validation_command:
        prompt += f"\nFor submission format checks, use `{config.validation_command}`."
    return prompt


def build_smoke_runtime_config(
    run_root: str | Path,
    *,
    scenario: str = "submit_sample",
    max_steps: int = 3,
    time_limit: int = 300,
    file_as_bus: bool = True,
) -> OrchestratorRuntimeConfig:
    root = Path(run_root).resolve()
    home_root = root / "home"
    return OrchestratorRuntimeConfig(
        time_limit=time_limit,
        max_steps=max_steps,
        reminder_freq=max(10, max_steps + 1),
        model="stub-orchestrator",
        hardware="offline-smoke",
        api_mode="stub",
        context_reduce_strategy="summary",
        summary_segment_ratio=0.3,
        summary_min_turns=4,
        summary_segment_max_chars=25000,
        summary_incremental=True,
        file_as_bus=file_as_bus,
        paths=OrchestratorPaths(
            home_root=str(home_root),
            data_dir=str(home_root / "data"),
            code_dir=str(home_root / "code"),
            submission_dir=str(home_root / "submission"),
            agent_dir=str(home_root / "agent"),
            logs_dir=str(root / "logs"),
        ),
        stub_scenario=scenario,
    )


class StubLLMClient(LLMClient):
    """Deterministic offline client for smoke-testing the orchestrator loop."""

    def __init__(self, paths: OrchestratorPaths, *, scenario: str = "submit_sample"):
        super().__init__(
            LLMConfig(
                model="stub-orchestrator",
                api_mode="stub",
                context_window=200_000,
            )
        )
        self._paths = paths
        self._scenario = scenario
        self._main_turn = 0
        self._call_counter = 0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del kwargs
        tool_names = {
            tool_schema.get("function", {}).get("name", "")
            for tool_schema in (tools or [])
        }
        if not tool_names:
            return self._response("Essential Information:\nStub summary.")
        if "submit" in tool_names:
            return self._main_agent_response()
        if "subagent_complete" in tool_names:
            return self._subagent_response()
        return self._response("Stub orchestration idle.")

    def _main_agent_response(self) -> LLMResponse:
        if self._scenario != "submit_sample":
            return self._response(
                f"Unsupported stub scenario: {self._scenario}. Expected submit_sample."
            )

        if self._main_turn == 0:
            self._main_turn += 1
            return self._response(
                tool_calls=[
                    ToolCallResult(
                        call_id=self._next_call_id(),
                        name="python",
                        arguments={
                            "code": self._bootstrap_submission_code(),
                            "timeout": 30,
                        },
                    )
                ]
            )

        if self._main_turn == 1:
            self._main_turn += 1
            return self._response(
                tool_calls=[
                    ToolCallResult(
                        call_id=self._next_call_id(),
                        name="submit",
                        arguments={"confirm": "yes"},
                    )
                ]
            )

        return self._response("Stub orchestration complete.")

    def _subagent_response(self) -> LLMResponse:
        return self._response(
            tool_calls=[
                ToolCallResult(
                    call_id=self._next_call_id(),
                    name="subagent_complete",
                    arguments={"content": "Stub subagent completed without external API access."},
                )
            ]
        )

    def _bootstrap_submission_code(self) -> str:
        data_dir = self._paths.data_dir
        code_dir = self._paths.code_dir
        submission_dir = self._paths.submission_dir
        sample_path = self._paths.sample_submission_path
        description_path = self._paths.description_path
        submission_path = self._paths.submission_csv_path
        return f"""
from pathlib import Path

data_dir = Path({data_dir!r})
code_dir = Path({code_dir!r})
submission_dir = Path({submission_dir!r})
sample_path = Path({sample_path!r})
description_path = Path({description_path!r})
submission_path = Path({submission_path!r})

data_dir.mkdir(parents=True, exist_ok=True)
code_dir.mkdir(parents=True, exist_ok=True)
submission_dir.mkdir(parents=True, exist_ok=True)

if not description_path.exists():
    description_path.write_text(
        "# Stub Competition\\n\\nOffline smoke data for orchestrator validation.\\n",
        encoding="utf-8",
    )

if not sample_path.exists():
    sample_path.write_text(
        "id,target\\n1,0\\n2,0\\n",
        encoding="utf-8",
    )

submission_path.write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")
(code_dir / "README.md").write_text("# Stub code workspace\\n", encoding="utf-8")
print(f"stub submission ready: {{submission_path}}")
""".strip()

    def _next_call_id(self) -> str:
        self._call_counter += 1
        return f"stub-call-{self._call_counter:03d}"

    def _response(
        self,
        text_content: str | None = None,
        *,
        tool_calls: list[ToolCallResult] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text_content=text_content,
            tool_calls=tool_calls or [],
            usage={"input": 0, "output": 0},
            raw_message=None,
        )


def create_orchestrator_llm(config: OrchestratorRuntimeConfig) -> LLMClient:
    if runtime_uses_stub_llm() or config.api_mode == "stub" or config.model == "stub-orchestrator":
        return StubLLMClient(
            config.paths,
            scenario=config.stub_scenario or "submit_sample",
        )
    return create_llm_client()
