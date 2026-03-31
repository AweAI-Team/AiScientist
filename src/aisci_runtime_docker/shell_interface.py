from __future__ import annotations

import os
import posixpath
import tempfile
from pathlib import Path

from aisci_agent_runtime.shell_interface import (
    ComputerInterface,
    ShellResult,
    TIMEOUT_BUFFER,
    _refuse_broad_python_kill,
    _shell_quote,
)
from aisci_runtime_docker.models import ContainerSession
from aisci_runtime_docker.runtime import DockerRuntimeManager


class DockerShellInterface(ComputerInterface):
    """
    Shell/file facade for a persistent Docker sandbox session.

    Command execution always happens inside the container. File operations prefer
    direct host access for mounted workspace paths and fall back to `docker cp`
    for non-mounted paths such as `/tmp/...`.
    """

    def __init__(
        self,
        runtime: DockerRuntimeManager,
        session: ContainerSession,
        *,
        working_dir: str = "/home/submission",
    ) -> None:
        self.runtime = runtime
        self.session = session
        self.working_dir = working_dir

    def send_shell_command(self, cmd: str, timeout: int = 300) -> ShellResult:
        refusal = _refuse_broad_python_kill(cmd)
        if refusal is not None:
            return ShellResult(output=refusal, exit_code=1)

        wrapped = f"timeout --signal=KILL {timeout} bash -lc {_shell_quote(cmd)}"
        result = self.runtime.exec(
            self.session,
            wrapped,
            workdir=self.working_dir,
            check=False,
            timeout_seconds=timeout + TIMEOUT_BUFFER,
        )
        return ShellResult(output=result.combined_output, exit_code=result.exit_code)

    def send_command(self, cmd: str, timeout: int = 300) -> ShellResult:
        return self.send_shell_command(cmd, timeout=timeout)

    def read_file(self, path: str) -> str:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            return host_path.read_text(encoding="utf-8", errors="replace")
        return self.download(path).decode("utf-8", errors="replace")

    def download(self, path: str) -> bytes:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            return host_path.read_bytes()

        container_path = self._container_path(path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "download.bin"
            self.runtime.copy_from_session(self.session, container_path, tmp_path)
            return tmp_path.read_bytes()

    def write_file(self, path: str, content: str) -> None:
        self.upload(content.encode("utf-8"), path)

    def upload(self, data: bytes, path: str) -> None:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            host_path.write_bytes(data)
            try:
                os.chmod(host_path, 0o644)
            except OSError:
                pass
            return

        container_path = self._container_path(path)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_path = Path(tmp_file.name)
        try:
            self.runtime.copy_to_session(self.session, tmp_path, container_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def append_file(self, path: str, content: str) -> None:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            with host_path.open("a", encoding="utf-8") as handle:
                handle.write(content)
            try:
                os.chmod(host_path, 0o644)
            except OSError:
                pass
            return

        existing = b""
        if self.file_exists(path):
            existing = self.download(path)
        self.upload(existing + content.encode("utf-8"), path)

    def file_exists(self, path: str) -> bool:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            return host_path.exists()
        container_path = self._container_path(path)
        result = self.runtime.exec(
            self.session,
            f"test -e {_shell_quote(container_path)}",
            workdir=self.working_dir,
            check=False,
            timeout_seconds=15,
        )
        return result.exit_code == 0

    def mapped(self, path: str) -> Path:
        host_path = self._mounted_host_path(path)
        if host_path is None:
            raise ValueError(f"{path} is not backed by a mounted host path")
        return host_path

    def _container_path(self, path: str) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(self.working_dir, path))

    def _mounted_host_path(self, path: str) -> Path | None:
        container_path = self._container_path(path)
        mounts = sorted(self.session.mounts, key=lambda mount: len(mount.target), reverse=True)
        for mount in mounts:
            target = mount.target.rstrip("/") or "/"
            if container_path == target:
                return mount.source
            prefix = target if target.endswith("/") else target + "/"
            if container_path.startswith(prefix):
                suffix = container_path[len(prefix):]
                return mount.source / suffix
        return None
