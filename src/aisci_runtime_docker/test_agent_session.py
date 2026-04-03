from __future__ import annotations

import unittest

from aisci_core.models import NetworkPolicy, RuntimeProfile, WorkspaceLayout
from aisci_runtime_docker.agent_session import AgentSessionManager
from aisci_runtime_docker.models import DockerExecutionResult, DockerProfile, SessionSpec


class _CapturingAgentSessionManager(AgentSessionManager):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[list[str]] = []

    def _run(
        self,
        command: list[str],
        check: bool = True,
        timeout: int | None = None,
    ) -> DockerExecutionResult:
        del check, timeout
        self.commands.append(command)
        return DockerExecutionResult(command=command, exit_code=0, stdout="container-id", stderr="")


class AgentSessionManagerTests(unittest.TestCase):
    def test_runtime_profile_accepts_legacy_payload_without_new_fields(self) -> None:
        profile = RuntimeProfile.model_validate({"gpu_count": 1, "time_limit": "1h"})

        self.assertEqual(profile.gpu_count, 1)
        self.assertEqual(profile.time_limit, "1h")
        self.assertIsNone(profile.shm_size)
        self.assertIsNone(profile.nano_cpus)

    def test_resource_args_preserve_existing_behavior_when_new_fields_unset(self) -> None:
        manager = AgentSessionManager()
        profile = RuntimeProfile(cpu_limit="2", memory_limit="4g")

        self.assertEqual(manager._resource_args(profile), ["--cpus", "2", "--memory", "4g"])

    def test_start_session_emits_supported_resource_flags_when_configured(self) -> None:
        manager = _CapturingAgentSessionManager()
        profile = RuntimeProfile(
            network_policy=NetworkPolicy.HOST,
            memory_limit="4g",
            shm_size="16G",
            nano_cpus=16000000000,
        )
        spec = SessionSpec(
            job_id="job-123",
            workspace_layout=WorkspaceLayout.MLE,
            profile=DockerProfile(name="mle-default", image="example/mle:latest"),
            runtime_profile=profile,
            mounts=(),
            workdir="/home/code",
        )

        manager.start_session(spec, "example/mle:latest")

        self.assertEqual(len(manager.commands), 1)
        command = manager.commands[0]
        self.assertIn("--network", command)
        self.assertIn("host", command)
        self.assertIn("--cpus", command)
        self.assertIn("16", command)
        self.assertIn("--memory", command)
        self.assertIn("4g", command)
        self.assertIn("--shm-size", command)
        self.assertIn("16G", command)
        self.assertNotIn("--nano-cpus", command)

    def test_explicit_cpu_limit_takes_precedence_over_nano_cpus(self) -> None:
        manager = AgentSessionManager()
        profile = RuntimeProfile(cpu_limit="2", nano_cpus=16000000000)

        self.assertEqual(manager._resource_args(profile), ["--cpus", "2"])


if __name__ == "__main__":
    unittest.main()
