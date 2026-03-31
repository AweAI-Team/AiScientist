from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from aisci_core.exporter import export_job_bundle
from aisci_core.logging_utils import append_log
from aisci_core.models import ArtifactRecord, JobStatus, JobType, RunPhase
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.store import JobStore
from aisci_domain_mle.adapter import MLEDomainAdapter
from aisci_domain_paper.adapter import PaperDomainAdapter
from aisci_runtime_docker.runtime import DockerRuntimeManager


class JobRunner:
    def __init__(self, store: JobStore | None = None):
        self.store = store or JobStore()
        self.runtime = DockerRuntimeManager()
        self.paper = PaperDomainAdapter(self.runtime)
        self.mle = MLEDomainAdapter(self.runtime)

    def run_job(self, job_id: str) -> JobStatus:
        job = self.store.get_job(job_id)
        paths = ensure_job_dirs(resolve_job_paths(job_id))
        self.store.mark_running(job_id, os.getpid())
        self.store.append_event(job_id, "status", RunPhase.INGEST, "Worker started.")
        append_log(paths.logs_dir / "job.log", f"worker pid={os.getpid()} started")
        self._write_job_spec(job, paths.root / "job_spec.json")
        try:
            result = self._dispatch(job)
            self._ingest_conversation_events(job_id, paths.logs_dir / "conversation.jsonl")
            artifacts = self._supplement_artifacts(job, paths, result.get("artifacts", []))
            validation_report = result.get("validation_report")
            for artifact in artifacts:
                assert isinstance(artifact, ArtifactRecord)
                self.store.add_artifact(job_id, artifact)
                self.store.append_event(
                    job_id,
                    "artifact",
                    artifact.phase,
                    f"Artifact recorded: {artifact.artifact_type}",
                    {"path": artifact.path, "artifact_type": artifact.artifact_type},
                )
            if validation_report is not None:
                self.store.update_phase(job_id, RunPhase.VALIDATE)
                validation_path = paths.artifacts_dir / "validation_report.json"
                validation_path.write_text(
                    validation_report.model_dump_json(indent=2), encoding="utf-8"
                )
                validation_artifact = ArtifactRecord(
                    artifact_type="validation_report",
                    path=str(validation_path),
                    phase=RunPhase.VALIDATE,
                    size_bytes=validation_path.stat().st_size,
                    metadata={"status": validation_report.status},
                )
                self.store.add_artifact(job_id, validation_artifact)
                self.store.append_event(
                    job_id,
                    "validation",
                    RunPhase.VALIDATE,
                    f"Validation finished with status={validation_report.status}.",
                    validation_report.model_dump(mode="json"),
                )
            self.store.update_phase(job_id, RunPhase.FINALIZE)
            export_path = export_job_bundle(paths, paths.export_dir / f"{job_id}.zip")
            self.store.add_artifact(
                job_id,
                ArtifactRecord(
                    artifact_type="export_bundle",
                    path=str(export_path),
                    phase=RunPhase.FINALIZE,
                    size_bytes=export_path.stat().st_size,
                    metadata={},
                ),
            )
            self.store.append_event(
                job_id,
                "status",
                RunPhase.FINALIZE,
                "Job completed successfully.",
                {"export_bundle": str(export_path)},
            )
            self.store.complete_job(job_id, JobStatus.SUCCEEDED)
            append_log(paths.logs_dir / "job.log", "job succeeded")
            return JobStatus.SUCCEEDED
        except Exception as exc:
            append_log(paths.logs_dir / "job.log", f"job failed: {exc}")
            (paths.logs_dir / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
            self.store.append_event(
                job_id,
                "error",
                job.phase,
                str(exc),
                {"traceback": traceback.format_exc()},
            )
            self.store.complete_job(job_id, JobStatus.FAILED, error=str(exc))
            return JobStatus.FAILED

    def _dispatch(self, job):
        if job.job_type.value == "paper":
            self.store.update_phase(job.id, RunPhase.ANALYZE)
            return self.paper.run(job)
        self.store.update_phase(job.id, RunPhase.ANALYZE)
        return self.mle.run(job)

    def _ingest_conversation_events(self, job_id: str, conversation_path: Path) -> None:
        if not conversation_path.exists():
            return

        current_phase = RunPhase.ANALYZE
        for raw_line in conversation_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_kind = str(record.get("event") or record.get("event_type") or "agent_event")
            inferred_phase = self._infer_phase(record, current_phase)
            current_phase = inferred_phase
            message = self._summarize_event(event_kind, record)
            self.store.append_event(
                job_id,
                event_kind,
                inferred_phase,
                message,
                record,
            )

    def _write_job_spec(self, job, destination: Path) -> None:
        destination.write_text(
            json.dumps(
                {
                    "id": job.id,
                    "job_type": job.job_type.value,
                    "objective": job.objective,
                    "llm_profile": job.llm_profile,
                    "runtime_profile": job.runtime_profile.model_dump(mode="json"),
                    "mode_spec": job.mode_spec.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _supplement_artifacts(self, job, paths, artifacts: list[ArtifactRecord]) -> list[ArtifactRecord]:
        results = list(artifacts)
        seen_paths = {artifact.path for artifact in results}

        def add(artifact_type: str, path: Path, phase: RunPhase, metadata: dict | None = None) -> None:
            if not path.exists():
                return
            resolved = str(path)
            if resolved in seen_paths:
                return
            seen_paths.add(resolved)
            results.append(
                ArtifactRecord(
                    artifact_type=artifact_type,
                    path=resolved,
                    phase=phase,
                    size_bytes=path.stat().st_size,
                    metadata=metadata or {},
                )
            )

        common_files = [
            ("agent_log", paths.logs_dir / "agent.log", RunPhase.FINALIZE),
            ("conversation_log", paths.logs_dir / "conversation.jsonl", RunPhase.FINALIZE),
            ("job_log", paths.logs_dir / "job.log", RunPhase.FINALIZE),
            ("job_spec", paths.root / "job_spec.json", RunPhase.INGEST),
        ]
        for artifact_type, path, phase in common_files:
            add(artifact_type, path, phase)

        if job.job_type == JobType.PAPER:
            paper_files = [
                ("paper_analysis", paths.workspace_dir / "agent" / "paper_analysis" / "summary.md", RunPhase.ANALYZE),
                ("paper_structure", paths.workspace_dir / "agent" / "paper_analysis" / "structure.md", RunPhase.ANALYZE),
                ("paper_algorithm", paths.workspace_dir / "agent" / "paper_analysis" / "algorithm.md", RunPhase.ANALYZE),
                ("paper_experiments", paths.workspace_dir / "agent" / "paper_analysis" / "experiments.md", RunPhase.ANALYZE),
                ("paper_baseline", paths.workspace_dir / "agent" / "paper_analysis" / "baseline.md", RunPhase.ANALYZE),
                ("prioritized_tasks", paths.workspace_dir / "agent" / "prioritized_tasks.md", RunPhase.PRIORITIZE),
                ("impl_log", paths.workspace_dir / "agent" / "impl_log.md", RunPhase.IMPLEMENT),
                ("exp_log", paths.workspace_dir / "agent" / "exp_log.md", RunPhase.VALIDATE),
                ("plan", paths.workspace_dir / "agent" / "plan.md", RunPhase.IMPLEMENT),
                ("capabilities", paths.state_dir / "capabilities.json", RunPhase.ANALYZE),
                ("resolved_llm_config", paths.state_dir / "resolved_llm_config.json", RunPhase.INGEST),
                ("prompt", paths.state_dir / "paper_main_prompt.md", RunPhase.INGEST),
                ("self_check_report", paths.workspace_dir / "agent" / "final_self_check.md", RunPhase.VALIDATE),
                ("self_check_report_json", paths.workspace_dir / "agent" / "final_self_check.json", RunPhase.VALIDATE),
                ("reproduce_script", paths.workspace_dir / "submission" / "reproduce.sh", RunPhase.FINALIZE),
                ("orchestrator_state", paths.logs_dir / "paper_session_state.json", RunPhase.FINALIZE),
                ("sandbox_session", paths.state_dir / "sandbox_session.json", RunPhase.FINALIZE),
            ]
            for artifact_type, path, phase in paper_files:
                add(artifact_type, path, phase)
        else:
            mle_files = [
                ("analysis_summary", paths.workspace_dir / "agent" / "analysis" / "summary.md", RunPhase.ANALYZE),
                ("prioritized_tasks", paths.workspace_dir / "agent" / "prioritized_tasks.md", RunPhase.PRIORITIZE),
                ("impl_log", paths.workspace_dir / "agent" / "impl_log.md", RunPhase.IMPLEMENT),
                ("exp_log", paths.workspace_dir / "agent" / "exp_log.md", RunPhase.VALIDATE),
                ("submission", paths.workspace_dir / "submission" / "submission.csv", RunPhase.FINALIZE),
                ("submission_registry", paths.workspace_dir / "submission" / "submission_registry.jsonl", RunPhase.FINALIZE),
            ]
            for artifact_type, path, phase in mle_files:
                add(artifact_type, path, phase)

        return results

    def _infer_phase(self, record: dict[str, object], fallback: RunPhase) -> RunPhase:
        names: list[str] = []
        if isinstance(record.get("tool"), str):
            names.append(str(record["tool"]))
        tool_call = record.get("tool_call")
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            if isinstance(name, str):
                names.append(name)
        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if isinstance(item, dict):
                    if isinstance(item.get("name"), str):
                        names.append(str(item["name"]))
                    function = item.get("function")
                    if isinstance(function, dict) and isinstance(function.get("name"), str):
                        names.append(str(function["name"]))
        lowered = " ".join(names).lower()
        if any(name in lowered for name in ("read_paper", "analyze_data")):
            return RunPhase.ANALYZE
        if "prioritize" in lowered:
            return RunPhase.PRIORITIZE
        if any(name in lowered for name in ("implement", "spawn_subagent", "bash", "python", "read_file_chunk", "search_file")):
            return RunPhase.IMPLEMENT
        if any(name in lowered for name in ("run_experiment", "clean_reproduce_validation", "validate")):
            return RunPhase.VALIDATE
        if "submit" in lowered:
            return RunPhase.FINALIZE
        return fallback

    def _summarize_event(self, event_kind: str, record: dict[str, object]) -> str:
        if event_kind == "tool_result":
            tool = record.get("tool")
            if isinstance(tool, str):
                return f"Tool completed: {tool}"
            tool_call = record.get("tool_call")
            if isinstance(tool_call, dict) and isinstance(tool_call.get("name"), str):
                return f"Tool completed: {tool_call['name']}"
            return "Tool result recorded."
        if event_kind == "model_response":
            text = record.get("text")
            if isinstance(text, str) and text.strip():
                compact = " ".join(text.split())
                return compact[:160]
            response_messages = record.get("response_messages")
            if isinstance(response_messages, list):
                for item in response_messages:
                    if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip():
                        compact = " ".join(str(item["content"]).split())
                        return compact[:160]
            return "Model response recorded."
        message = record.get("message")
        if isinstance(message, str) and message.strip():
            return message[:160]
        return f"{event_kind} recorded."
