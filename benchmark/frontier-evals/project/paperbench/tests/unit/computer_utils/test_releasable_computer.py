from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

import pytest

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
    JupyterExecutionResult,
)
from paperbench.computer_utils import ReleasableComputer, maybe_release_computer


class DummyComputer(ComputerInterface):
    def __init__(self) -> None:
        self.stopped = False
        self.commands: list[str] = []

    async def disable_internet(self) -> None:
        self.commands.append("disable_internet")

    async def upload(self, file: bytes, destination: str) -> None:
        self.commands.append(f"upload:{destination}")

    async def download(self, file: str) -> bytes:
        self.commands.append(f"download:{file}")
        return b"data"

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        self.commands.append(f"shell:{cmd}")
        return ExecutionResult(output=b"ok", exit_code=0)

    async def fetch_container_names(self) -> list[str]:
        self.commands.append("fetch_container_names")
        return ["container"]

    async def stop(self) -> None:
        self.stopped = True

    async def execute(self, code: str, timeout: int = 120) -> JupyterExecutionResult:
        self.commands.append(f"execute:{code}")
        return JupyterExecutionResult(
            status="success",
            output="",
            final_expression_output=None,
        )


@pytest.mark.asyncio
async def test_releasable_computer_releases_underlying_resource() -> None:
    dummy = DummyComputer()
    cleanup_order: list[str] = []

    @asynccontextmanager
    async def managed_dummy() -> AsyncIterator[ComputerInterface]:
        cleanup_order.append("enter")
        try:
            yield dummy
        finally:
            cleanup_order.append("exit")
            await dummy.stop()

    stack = AsyncExitStack()
    leased_target: ComputerInterface = await stack.enter_async_context(managed_dummy())
    releasable = ReleasableComputer(leased_target, stack)

    await releasable.send_shell_command("echo hi")
    assert dummy.commands == ["shell:echo hi"]

    await releasable.release()

    assert dummy.stopped is True
    assert cleanup_order == ["enter", "exit"]

    with pytest.raises(RuntimeError):
        await releasable.send_shell_command("echo after release")

    # release() should be idempotent
    await releasable.release()


@pytest.mark.asyncio
async def test_releasable_computer_stop_aliases_release() -> None:
    dummy = DummyComputer()
    stack = AsyncExitStack()

    @asynccontextmanager
    async def managed_dummy() -> AsyncIterator[ComputerInterface]:
        yield dummy

    leased_target: ComputerInterface = await stack.enter_async_context(managed_dummy())
    releasable = ReleasableComputer(leased_target, stack)

    await releasable.stop()
    assert dummy.stopped
    with pytest.raises(RuntimeError):
        await releasable.upload(b"", "/tmp/file")


@pytest.mark.asyncio
async def test_maybe_release_computer_calls_release() -> None:
    dummy = DummyComputer()

    @asynccontextmanager
    async def managed_dummy() -> AsyncIterator[ComputerInterface]:
        yield dummy

    stack = AsyncExitStack()
    leased_target: ComputerInterface = await stack.enter_async_context(managed_dummy())
    releasable = ReleasableComputer(leased_target, stack)

    await maybe_release_computer(releasable)

    assert dummy.stopped is True


class DummyNonReleasable(ComputerInterface):
    async def disable_internet(self) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    async def upload(self, file: bytes, destination: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def download(self, file: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    async def send_shell_command(
        self, cmd: str, *, idempotent: bool = False
    ) -> ExecutionResult:  # pragma: no cover
        raise NotImplementedError

    async def fetch_container_names(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    async def stop(self) -> None:
        # Called nowhere in this test
        raise NotImplementedError


@pytest.mark.asyncio
async def test_maybe_release_computer_warns_when_not_leased(
    capsys: pytest.CaptureFixture[str],
) -> None:
    computer = DummyNonReleasable()

    await maybe_release_computer(computer)

    captured = capsys.readouterr()
    assert "non-releasable computer" in captured.out
