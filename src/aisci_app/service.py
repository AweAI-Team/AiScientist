from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from aisci_core.exporter import export_job_bundle
from aisci_core.models import JobSpec
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.store import JobStore


class JobService:
    def __init__(self, store: JobStore | None = None):
        self.store = store or JobStore()

    def create_job(self, spec: JobSpec):
        job = self.store.create_job(spec)
        ensure_job_dirs(resolve_job_paths(job.id))
        return job

    def spawn_worker(self, job_id: str, wait: bool = False) -> int:
        command = [sys.executable, "-m", "aisci_app.worker_main", job_id]
        paths = ensure_job_dirs(resolve_job_paths(job_id))
        worker_log = paths.logs_dir / "worker.log"
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        if wait:
            with worker_log.open("ab") as handle:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=handle,
                    stderr=handle,
                )
            return completed.returncode
        handle = worker_log.open("ab")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=handle,
            env=os.environ.copy(),
        )
        handle.close()
        return process.pid

    def export_bundle(self, job_id: str) -> Path:
        paths = ensure_job_dirs(resolve_job_paths(job_id))
        return export_job_bundle(paths, paths.export_dir / f"{job_id}.zip")
