import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import docker
from docker.models.containers import Container
from dotenv import dotenv_values

from agents.registry import Agent
from environment.utils import (
    create_competition_container,
    extract_from_container,
    extract_from_container_sysbox,
)
from mlebench.registry import Competition
from mlebench.utils import purple

CONSTANTS = dotenv_values(Path(__file__).parent.resolve() / ".shared_env")

# Thread-safe port counter for grading server (avoids port conflicts with network_mode=host).
# When multiple run_agent.py processes run in parallel, set MLEBENCH_GRADING_PORT_BASE per process
# so port ranges don't overlap (e.g. first run default 5000, second run export MLEBENCH_GRADING_PORT_BASE=6000).
_port_counter_lock = threading.Lock()
_base_env = os.environ.get("MLEBENCH_GRADING_PORT_BASE", "5000").strip()
_next_grading_port = int(_base_env)
if not (1024 <= _next_grading_port <= 65535):
    raise ValueError(
        f"MLEBENCH_GRADING_PORT_BASE must be a TCP port in [1024, 65535] (got {_next_grading_port!r} from {_base_env!r}). "
        "Values above 65535 cannot bind; use e.g. 6000 or 17000 for a second parallel run_agent."
    )


def _get_next_grading_port() -> int:
    global _next_grading_port
    with _port_counter_lock:
        port = _next_grading_port
        _next_grading_port += 1
    return port


def save_output(container: Container, save_dir: Path, container_config: dict,
                skip_dirs: set | None = None) -> Path:
    """
    Extracts the submission, logs, and code directories from the container
    and saves them to the specified directory.

    Args:
        container: The Docker container.
        save_dir: The directory where the output file will be saved.
        container_config: The container configuration.
        skip_dirs: Set of dir type keys (e.g. {"LOGS_DIR"}) to skip extraction
            for (because they are already volume-mounted to the host).
    Returns:
        Path to the output directory.
    """
    if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
        extraction_fn = extract_from_container_sysbox
    else:
        extraction_fn = extract_from_container

    skip_dirs = skip_dirs or set()
    # Map container dir type -> host path. LOGS_DIR extracted to save_dir/logs so host user owns files
    # (host process writes logs to run_dir; not relying on volume-mounted copy from solver).
    for dir_type in ["SUBMISSION_DIR", "LOGS_DIR", "CODE_DIR"]:
        if dir_type in skip_dirs:
            continue
        container_dir = CONSTANTS[dir_type]
        if dir_type == "LOGS_DIR":
            logs_dir = save_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            extraction_fn(container, container_dir, logs_dir)
        else:
            extraction_fn(container, container_dir, save_dir)

    return save_dir


def _tail_container_until_done(container: Container, logger: logging.Logger) -> None:
    """
    Wait until the agent process inside the container has truly finished.

    Background
    ----------
    The container is kept alive by ``grading_server.py`` which runs forever as
    the container's PID-1 child.  Therefore ``container.status == "running"`` is
    true even after the agent process exits — we cannot use container liveness to
    decide when the agent is done.

    The ``exec_run`` stream (used in ``execute_agent``) ends when the exec session
    closes, which happens exactly when the agent process exits.  So normally we
    don't even reach this function.  We only get here on very long runs where the
    Docker HTTP stream is silently dropped by an intermediate proxy/timeout while
    the agent is still running.

    Strategy
    --------
    Poll ``container.top()`` (which calls ``docker top``) every
    POLL_INTERVAL_S seconds looking for any python/bash process that matches the
    agent start script (``start.sh``).  Once none is found the agent has exited
    and we return — allowing the worker to proceed to ``save_output`` and
    ``clean_up``.

    A hard cap of MAX_WAIT_S prevents infinite blocking even if ``top()`` is
    unreliable.  The value is set above the longest expected agent runtime
    (``TIME_LIMIT_SECS`` defaults to 86400 → 24 h in ``aisci/config.yaml``) plus a buffer.
    """
    POLL_INTERVAL_S = 30
    # Cap must exceed longest TIME_LIMIT_SECS (24 h) if exec stream drops mid-run.
    MAX_WAIT_S = 108_000  # 30 h = 24 h + buffer for finalize / polling jitter

    try:
        container.reload()
        if container.status != "running":
            return

        logger.info(
            "[Container] exec_run stream ended but container still running — "
            "polling for agent process exit (poll every %ds, cap %ds)",
            POLL_INTERVAL_S, MAX_WAIT_S,
        )
        def _agent_still_running() -> bool:
            """Return True if start.sh process is still visible in the container."""
            container.reload()
            if container.status != "running":
                return False
            top_info = container.top()
            processes = top_info.get("Processes") or []
            titles = top_info.get("Titles") or []
            cmd_idx = next(
                (i for i, t in enumerate(titles) if t.upper() in ("CMD", "COMMAND")),
                -1,
            )
            return any(
                "start.sh" in (row[cmd_idx] if cmd_idx >= 0 else str(row))
                for row in processes
            )

        # Check immediately first (exec_run stream may have ended right as agent exited)
        waited = 0
        try:
            if not _agent_still_running():
                logger.info("[Container] agent process (start.sh) already gone — done immediately.")
                return
        except Exception as poll_err:
            logger.warning("[Container] Initial poll error: %s — assuming agent done.", poll_err)
            return

        while waited < MAX_WAIT_S:
            time.sleep(POLL_INTERVAL_S)
            waited += POLL_INTERVAL_S

            try:
                if not _agent_still_running():
                    logger.info(
                        "[Container] agent process (start.sh) no longer found after %ds — done.",
                        waited,
                    )
                    return
                else:
                    logger.info(
                        "[Container] agent still running after %ds — continuing to wait...",
                        waited,
                    )
            except Exception as poll_err:
                logger.warning("[Container] Poll error: %s — assuming agent done.", poll_err)
                return

        logger.warning(
            "[Container] Hard cap of %ds reached — proceeding with save_output anyway.",
            MAX_WAIT_S,
        )
    except Exception as e:
        # Non-fatal: volume-mounted logs are still accessible on the host.
        logger.warning(f"[Container] Log tail ended unexpectedly: {e}")


def execute_agent(
    container: Container,
    agent: Agent,
    logger: logging.Logger,
    *,
    run_as_uid: int | None = None,
    run_as_gid: int | None = None,
):
    """
    Initiates the agent via its start script inside the container.

    When run_as_uid/run_as_gid are set (e.g. host user), the agent runs as that user so that
    files it creates in volume-mounted run_dir/logs are owned by the host user and readable
    without permission issues.
    """
    cmd = ["bash", f"{CONSTANTS['AGENT_DIR']}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    if run_as_uid is not None and run_as_gid is not None:
        user = f"{run_as_uid}:{run_as_gid}"
        logger.info("Running agent as host user %s so log files are readable on host...", user)
        # Give agent a writable HOME so git config --global works (writes to /home/agent/.gitconfig)
        exec_kw = {"user": user, "environment": {"HOME": "/home/agent"}}
    else:
        user = "nonroot"
        logger.info("Running agent...")
        exec_kw = {"user": user}

    # exec_run with stream=True returns a generator backed by a long-lived HTTP
    # connection to the Docker daemon.  On long runs (hours) this connection can
    # silently drop when the container produces no stdout for an extended period,
    # causing the generator to end prematurely while the container keeps running.
    #
    # Workaround: wrap the read loop in a retry that re-attaches via docker logs
    # whenever the exec stream ends but the container is still running.
    exit_code, output = container.exec_run(cmd, stream=True, **exec_kw)

    # Drain the exec_run stream (covers the early startup phase with dense output)
    for chunk in output:
        logger.info(f"[Container] {chunk.decode('utf-8').strip()}")

    # If the container is still alive after the exec stream ended, tail its logs
    # until it actually stops.  This handles the silent-drop case.
    _tail_container_until_done(container, logger)


def clean_up(container: Container, logger: logging.Logger, retain: bool = False) -> bool:
    """
    Stops and removes the container.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Cleaning up container: {container.name}")
    try:
        container.stop()
        if not retain:
            container.remove()
        logger.info(f"Container {container.name} stopped and removed.")
        return True
    except Exception as e:
        logger.error(
            f"Error cleaning up: {e}. You may wish to manually check the status of the {container.name} container."
        )
        return False


def run_in_container(
    client: docker.DockerClient,
    competition: Competition,
    agent: Agent,
    image: str,
    container_config: dict,
    retain_container: bool,
    run_dir: Path,
    logger: logging.Logger,
    gpu_device_id: str | None = None,
) -> Path:
    """
    Runs environment containing the competition and agent for a set maximum amount of time.

    Args:
        client: Docker client.
        competition: The competition to run.
        agent: The agent to run.
        image: The Docker image to use. Assumes the image is built.
        container_config: Configuration for the Docker container.
        retain_container: Whether to retain the container after the run instead of removing it.
        run_dir: Path to the directory where all assets associated with the run are stored.
        logger: Logger for the run.
        gpu_device_id: If provided, assigns this specific GPU (e.g. "0") to the container.

    Returns:
        Path to the output file.
    """
    # Mount run_dir/logs -> /home/logs so agent logs appear on the host in real-time. We run the
    # agent as the host user (run_as_uid/gid in execute_agent) so all files under run_dir/logs
    # are owned by the host user and readable without permission issues.
    logs_host_dir = run_dir / "logs"
    logs_host_dir.mkdir(parents=True, exist_ok=True)
    volumes_config = {
        competition.public_dir.resolve().as_posix(): {
            "bind": "/home/data",
            "mode": "ro",
        },
        competition.private_dir.resolve().as_posix(): {
            "bind": f"/private/data/{competition.id}/prepared/private/",
            "mode": "ro",
        },
        logs_host_dir.resolve().as_posix(): {
            "bind": "/home/logs",
            "mode": "rw",
        },
    }

    grading_port = _get_next_grading_port()
    # Base env from config; then pass through selected runtime env vars from the
    # host so script exports take effect in the container.
    env_vars = {
        "COMPETITION_ID": competition.id,
        "GRADING_PORT": str(grading_port),
        **agent.env_vars,
    }
    for k, v in os.environ.items():
        if (
            k.startswith("AISCI_")
            or k.startswith("GEMINI_")
            or k.startswith("AIDE_")
        ) and v:
            env_vars[k] = v
    # HTTP(S) proxy: agent runs inside the container and calls LLM APIs from there.
    # Host-only `proxy-on` is invisible unless we copy these (host network: 127.0.0.1 works).
    for _proxy_key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        _pv = os.environ.get(_proxy_key)
        if _pv:
            env_vars[_proxy_key] = _pv
    # Orchestrator wall clock (orchestrator.py, start.sh). Host export overrides config.yaml.
    _tl_host = os.environ.get("TIME_LIMIT_SECS", "").strip()
    if _tl_host:
        env_vars["TIME_LIMIT_SECS"] = _tl_host
    # Docker 环境变量应为字符串；yaml 里可能是整型（如 172800），避免部分 runtime 忽略非 str。
    if env_vars.get("TIME_LIMIT_SECS") is not None:
        env_vars["TIME_LIMIT_SECS"] = str(int(env_vars["TIME_LIMIT_SECS"]))
    # AIDE：Omegaconf 的 agent.time_limit 须与 TIME_LIMIT_SECS 一致，否则外层 timeout 已延长而内核仍早停。
    agent_exec = agent
    if (
        agent.kwargs_type == "omegaconf"
        and "agent.time_limit" in agent.kwargs
        and env_vars.get("TIME_LIMIT_SECS") is not None
    ):
        agent_exec = replace(
            agent,
            kwargs={
                **agent.kwargs,
                "agent.time_limit": int(env_vars["TIME_LIMIT_SECS"]),
            },
        )
    # AI Scientist: shared impl_log/exp_log bus (default on inside container; pass through when set on host)
    if "FILE_AS_BUS" in os.environ:
        env_vars["FILE_AS_BUS"] = os.environ["FILE_AS_BUS"]
    container = create_competition_container(
        client=client,
        competition=competition,
        container_config=container_config,
        volumes_config=volumes_config,
        env_vars=env_vars,
        container_image=image,
        privileged=agent.privileged,
        gpu_device_id=gpu_device_id,
    )

    logger.info(purple(f"Run started: {run_dir}"))
    try:
        time_start = time.monotonic()
        container.start()
        exit_code, _ = container.exec_run(
            f'timeout 60s sh -c "while ! curl -s http://localhost:{grading_port}/health > /dev/null; do sleep 1; done"'
        )
        if exit_code != 0:
            raise RuntimeError(
                "The grading server failed to start within 60 seconds. This is likely due to an error in `entrypoint.sh`; check the logs."
            )
        execute_agent(
            container,
            agent_exec,
            logger,
            run_as_uid=os.getuid(),
            run_as_gid=os.getgid(),
        )
        # LOGS_DIR already on host via volume mount; skip redundant extraction
        save_output(container, run_dir, container_config, skip_dirs={"LOGS_DIR"})
        time_end = time.monotonic()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")
        return run_dir
    except Exception as e:
        raise e
    finally:
        clean_up(container, logger, retain_container)
