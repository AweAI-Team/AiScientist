from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisci_agent_runtime.shell_interface import (
    ShellResult,
    TIMEOUT_BUFFER,
    _refuse_broad_python_kill,
    _shell_quote,
)


@dataclass(frozen=True)
class ExecutionResult:
    output: bytes
    exit_code: int

    @property
    def unicode_output_best_effort(self) -> str:
        return self.output.decode("utf-8", errors="replace")


class ComputerInterface(ABC):
    @abstractmethod
    async def disable_internet(self) -> None:
        pass

    @abstractmethod
    async def upload(self, file: bytes, destination: str) -> None:
        pass

    @abstractmethod
    async def download(self, file: str) -> bytes:
        pass

    @abstractmethod
    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        pass

    async def check_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        res = await self.send_shell_command(cmd, idempotent=idempotent)
        if res.exit_code != 0:
            raise AssertionError(
                f"Command failed with exit_code={res.exit_code}\n\n{cmd=}\n\n"
                f"{res.unicode_output_best_effort}"
            )
        return res

    @abstractmethod
    async def fetch_container_names(self) -> list[str]:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass


class LocalComputerInterface(ComputerInterface):
    def __init__(self, working_dir: str | Path = "/home"):
        self.working_dir = str(working_dir)

    async def disable_internet(self) -> None:
        return None

    async def upload(self, file: bytes, destination: str) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file)

    async def download(self, file: str) -> bytes:
        return Path(file).read_bytes()

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            "-lc",
            cmd,
            cwd=self.working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        exit_code = process.returncode
        if exit_code < 0:
            exit_code = 128 + abs(exit_code)
        return ExecutionResult(output=(stdout or b"") + (stderr or b""), exit_code=exit_code)

    async def fetch_container_names(self) -> list[str]:
        return []

    async def stop(self) -> None:
        return None


@dataclass(frozen=True)
class PathMapper:
    canonical_to_real: dict[str, Path] = field(default_factory=dict)

    def real_path(self, path: str | os.PathLike[str]) -> Path:
        text = str(path)
        for canonical, real in sorted(self.canonical_to_real.items(), key=lambda item: len(item[0]), reverse=True):
            if text == canonical:
                return real
            if text.startswith(canonical.rstrip("/") + "/"):
                suffix = text[len(canonical.rstrip("/")) :].lstrip("/")
                return real / suffix
        return Path(text)

    def rewrite_command(self, cmd: str) -> str:
        rewritten = cmd
        for canonical, real in sorted(self.canonical_to_real.items(), key=lambda item: len(item[0]), reverse=True):
            rewritten = rewritten.replace(canonical, str(real))
        return rewritten


class MappedShellInterface:
    """Synchronous shell facade that rewrites canonical /home paths to host paths."""

    def __init__(self, working_dir: str | Path, mapper: PathMapper):
        self.mapper = mapper
        self.working_dir = str(self.mapper.real_path(working_dir))

    def send_shell_command(self, cmd: str, timeout: int = 300) -> "ShellResult":
        refusal = _refuse_broad_python_kill(cmd)
        if refusal is not None:
            return ShellResult(output=refusal, exit_code=1)
        rewritten = self.mapper.rewrite_command(cmd)
        wrapped = f"timeout --signal=KILL {timeout} bash --noprofile --norc -lc {_shell_quote(rewritten)}"
        try:
            completed = subprocess.run(
                ["bash", "--noprofile", "--norc", "-lc", wrapped],
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                timeout=timeout + TIMEOUT_BUFFER,
            )
        except subprocess.TimeoutExpired:
            return ShellResult(output=f"ERROR: command timed out after {timeout}s", exit_code=137)
        output = (completed.stdout or "") + (completed.stderr or "")
        return ShellResult(output=output.strip(), exit_code=completed.returncode)

    def send_command(self, cmd: str, timeout: int = 300) -> "ShellResult":
        return self.send_shell_command(cmd, timeout=timeout)

    def read_file(self, path: str | os.PathLike[str]) -> str:
        return self.mapper.real_path(path).read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str | os.PathLike[str], content: str) -> None:
        target = self.mapper.real_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def append_file(self, path: str | os.PathLike[str], content: str) -> None:
        target = self.mapper.real_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)

    def upload(self, data: bytes, path: str | os.PathLike[str]) -> None:
        target = self.mapper.real_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def download(self, path: str | os.PathLike[str]) -> bytes:
        return self.mapper.real_path(path).read_bytes()

    def file_exists(self, path: str | os.PathLike[str]) -> bool:
        return self.mapper.real_path(path).exists()

    def mapped(self, path: str | os.PathLike[str]) -> Path:
        return self.mapper.real_path(path)


async def send_shell_command_with_timeout(
    computer: ComputerInterface,
    cmd: str,
    timeout: int = 300,
) -> ExecutionResult:
    wrapped = f"timeout --signal=KILL {timeout} bash -c {shlex.quote(cmd + ' 2>&1')}"
    try:
        return await asyncio.wait_for(
            computer.send_shell_command(cmd=wrapped),
            timeout=timeout + 30,
        )
    except asyncio.TimeoutError:
        return ExecutionResult(output=f"ERROR: command timed out after {timeout}s".encode("utf-8"), exit_code=137)
