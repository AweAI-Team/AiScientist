from __future__ import annotations

import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from aisci_core.models import (
    ArtifactRecord,
    EventRecord,
    JobRecord,
    JobSpec,
    JobStatus,
    RunPhase,
    RuntimeProfile,
    PaperSpec,
    MLESpec,
)
from aisci_core.paths import database_path


STALE_WORKER_ERROR = "worker exited unexpectedly before final status update"


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _new_job_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


class JobStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or database_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists jobs (
                    id text primary key,
                    job_type text not null,
                    status text not null,
                    phase text not null,
                    objective text not null,
                    llm_profile text not null,
                    runtime_profile_json text not null,
                    mode_spec_json text not null,
                    created_at text not null,
                    updated_at text not null,
                    started_at text,
                    ended_at text,
                    worker_pid integer,
                    error text
                );
                create table if not exists events (
                    id integer primary key autoincrement,
                    job_id text not null,
                    event_type text not null,
                    phase text not null,
                    message text not null,
                    payload_json text not null default '{}',
                    created_at text not null,
                    foreign key(job_id) references jobs(id)
                );
                create table if not exists artifacts (
                    id integer primary key autoincrement,
                    job_id text not null,
                    artifact_type text not null,
                    path text not null,
                    phase text not null,
                    size_bytes integer not null,
                    created_at text not null,
                    metadata_json text not null default '{}',
                    foreign key(job_id) references jobs(id)
                );
                """
            )

    def create_job(self, spec: JobSpec) -> JobRecord:
        now = datetime.now().astimezone()
        job_id = _new_job_id(now)
        with self.connect() as conn:
            conn.execute(
                """
                insert into jobs (
                    id, job_type, status, phase, objective, llm_profile,
                    runtime_profile_json, mode_spec_json, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    spec.job_type.value,
                    JobStatus.PENDING.value,
                    RunPhase.INGEST.value,
                    spec.objective,
                    spec.llm_profile,
                    spec.runtime_profile.model_dump_json(),
                    spec.mode_spec.model_dump_json(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        row = self._fetch_job_row(job_id)
        if row is None:
            raise KeyError(f"unknown job {job_id}")
        job = self._row_to_job(row)
        if self._reconcile_running_job(job):
            refreshed = self._fetch_job_row(job_id)
            if refreshed is None:
                raise KeyError(f"unknown job {job_id}")
            job = self._row_to_job(refreshed)
        return job

    def list_jobs(self) -> list[JobRecord]:
        jobs = [self._row_to_job(row) for row in self._fetch_job_rows()]
        reconciled = False
        for job in jobs:
            reconciled = self._reconcile_running_job(job) or reconciled
        if reconciled:
            jobs = [self._row_to_job(row) for row in self._fetch_job_rows()]
        return jobs

    def append_event(
        self,
        job_id: str,
        event_type: str,
        phase: RunPhase,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        created_at = datetime.now().astimezone()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                insert into events (job_id, event_type, phase, message, payload_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    event_type,
                    phase.value,
                    message,
                    json.dumps(payload or {}, ensure_ascii=True),
                    created_at.isoformat(),
                ),
            )
        return EventRecord(
            id=int(cursor.lastrowid),
            job_id=job_id,
            event_type=event_type,
            phase=phase,
            message=message,
            payload=payload or {},
            created_at=created_at,
        )

    def list_events(self, job_id: str, after_id: int = 0) -> list[EventRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from events where job_id = ? and id > ? order by id asc",
                (job_id, after_id),
            ).fetchall()
        return [
            EventRecord(
                id=int(row["id"]),
                job_id=row["job_id"],
                event_type=row["event_type"],
                phase=RunPhase(row["phase"]),
                message=row["message"],
                payload=json.loads(row["payload_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def add_artifact(self, job_id: str, artifact: ArtifactRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into artifacts (
                    job_id, artifact_type, path, phase, size_bytes, created_at, metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    artifact.artifact_type,
                    artifact.path,
                    artifact.phase.value,
                    artifact.size_bytes,
                    artifact.created_at.isoformat(),
                    json.dumps(artifact.metadata, ensure_ascii=True),
                ),
            )

    def list_artifacts(self, job_id: str) -> list[ArtifactRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from artifacts where job_id = ? order by created_at asc",
                (job_id,),
            ).fetchall()
        return [
            ArtifactRecord(
                artifact_type=row["artifact_type"],
                path=row["path"],
                phase=RunPhase(row["phase"]),
                size_bytes=int(row["size_bytes"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def mark_running(self, job_id: str, worker_pid: int) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                update jobs
                set status = ?, updated_at = ?, started_at = ?, worker_pid = ?
                where id = ?
                """,
                (JobStatus.RUNNING.value, now, now, worker_pid, job_id),
            )

    def update_phase(self, job_id: str, phase: RunPhase) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as conn:
            conn.execute(
                "update jobs set phase = ?, updated_at = ? where id = ?",
                (phase.value, now, job_id),
            )

    def complete_job(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                update jobs
                set status = ?, updated_at = ?, ended_at = ?, error = ?
                where id = ?
                """,
                (status.value, now, now, error, job_id),
            )

    def _fetch_job_row(self, job_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("select * from jobs where id = ?", (job_id,)).fetchone()

    def _fetch_job_rows(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("select * from jobs order by created_at desc").fetchall()

    def _reconcile_running_job(self, job: JobRecord) -> bool:
        if job.status != JobStatus.RUNNING or job.worker_pid is None:
            return False
        if self._pid_exists(job.worker_pid):
            return False
        self._mark_job_stale(job)
        return True

    def _pid_exists(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _mark_job_stale(self, job: JobRecord) -> None:
        now = datetime.now().astimezone()
        payload = {
            "reconciled": True,
            "reason": "stale_worker",
            "worker_pid": job.worker_pid,
            "status": JobStatus.FAILED.value,
        }
        with self.connect() as conn:
            cursor = conn.execute(
                """
                update jobs
                set status = ?, updated_at = ?, ended_at = ?, error = ?
                where id = ? and status = ?
                """,
                (
                    JobStatus.FAILED.value,
                    now.isoformat(),
                    now.isoformat(),
                    STALE_WORKER_ERROR,
                    job.id,
                    JobStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount == 0:
                return
            conn.execute(
                """
                insert into events (job_id, event_type, phase, message, payload_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    "status",
                    job.phase.value,
                    "Worker exited unexpectedly before final status update.",
                    json.dumps(payload, ensure_ascii=True),
                    now.isoformat(),
                ),
            )

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        job_type = row["job_type"]
        if job_type == "paper":
            mode_spec = PaperSpec.model_validate_json(row["mode_spec_json"])
        else:
            mode_spec = MLESpec.model_validate_json(row["mode_spec_json"])
        return JobRecord(
            id=row["id"],
            job_type=job_type,
            status=JobStatus(row["status"]),
            phase=RunPhase(row["phase"]),
            objective=row["objective"],
            llm_profile=row["llm_profile"],
            runtime_profile=RuntimeProfile.model_validate_json(row["runtime_profile_json"]),
            mode_spec=mode_spec,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            started_at=_from_iso(row["started_at"]),
            ended_at=_from_iso(row["ended_at"]),
            worker_pid=row["worker_pid"],
            error=row["error"],
        )
