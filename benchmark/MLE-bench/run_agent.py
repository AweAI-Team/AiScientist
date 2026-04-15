import argparse
import asyncio
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker
from tqdm import tqdm

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run import run_in_container
from environment.defaults import DEFAULT_CONTAINER_CONFIG_PATH
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

logger = get_logger(__name__)

# Global tqdm bar updated by workers
_progress_bar: tqdm | None = None
_progress_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Log watcher: WARNING+ logs from any component are printed to the terminal.
#
# The agent runs inside Docker; a background thread tails agent.log and
# re-prints WARNING/ERROR lines via tqdm.write without disrupting the progress bar.
# ---------------------------------------------------------------------------

# Keywords that mark a line as worth printing even at INFO level
_ALERT_KEYWORDS = (
    "WARNING", "ERROR", "CRITICAL",
    "Content policy", "content_policy", "invalid_prompt", "-4321",
    "Account blocked", "-2005", "达到上限",
    "RateLimitError", "429", "rate_limit",
    "safety_trigger", "Safety trigger",
    "retry", "Retry",
)


class _LogWatcher:
    """
    Background thread that tails a task's agent.log and prints alert lines
    to the terminal in real time via tqdm.write.

    WARNING and above always show on the terminal regardless of the progress bar.
    """

    def __init__(self, log_path: Path, label: str, poll_interval: float = 2.0):
        self._log_path = log_path
        self._label = label          # e.g. "spaceship-titanic [W1]"
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"LogWatcher-{label}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        # Wait up to 30 s for agent.log to appear (agent needs time to start)
        waited = 0.0
        while not self._log_path.exists():
            if self._stop.is_set() or waited >= 30:
                return
            time.sleep(self._poll_interval)
            waited += self._poll_interval

        try:
            with open(self._log_path, "r", errors="replace") as fh:
                # Seek to end so we only see new lines, not historic content
                fh.seek(0, 2)
                while not self._stop.is_set():
                    line = fh.readline()
                    if line:
                        self._maybe_print(line.rstrip())
                    else:
                        time.sleep(self._poll_interval)
        except Exception:
            pass  # log disappeared or unreadable — silently stop

    def _maybe_print(self, line: str) -> None:
        if not line:
            return
        if any(kw in line for kw in _ALERT_KEYWORDS):
            tqdm.write(f"  ⚠  [{self._label}] {line}")


class _RoundRobinGPUAllocator:
    """
    Round-robin GPU allocator (least-loaded GPU at task start).

    Concurrency (how many tasks run at once) is separate from GPU assignment
    (which GPU each task uses). The allocator picks the least-loaded GPU at task-start
    time, so concurrency can freely exceed the number of physical GPUs.

    This simpler version uses round-robin (good enough when tasks have similar duration
    and GPU memory headroom allows sharing, e.g. 8× H20 with lightweight models).
    Tasks call acquire() to get a GPU ID and release() when done.
    """

    def __init__(self, gpu_ids: list[str]):
        self._gpu_ids = gpu_ids
        self._load: dict[str, int] = {g: 0 for g in gpu_ids}
        self._lock = threading.Lock()

    def acquire(self) -> str:
        """Return the least-loaded GPU ID and increment its counter."""
        with self._lock:
            gpu = min(self._load, key=lambda g: self._load[g])
            self._load[gpu] += 1
            return gpu

    def release(self, gpu_id: str) -> None:
        """Decrement the load counter for a GPU after a task finishes."""
        with self._lock:
            self._load[gpu_id] = max(0, self._load[gpu_id] - 1)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    image: str
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition
    container_config: dict[str, Any]


async def worker(
    idx: int,
    queue: asyncio.Queue[Task],
    client: docker.DockerClient,
    tasks_outputs: dict[str, dict[str, Any]],
    gpu_allocator: "_RoundRobinGPUAllocator | None" = None,
) -> None:
    """
    Worker coroutine — pulls tasks from a queue until empty.

    Key difference from a naive worker-per-GPU design:
    - GPU is acquired from the allocator at task-start, not at worker-creation.
    - This allows concurrency > number of GPUs (multiple tasks share a GPU,
      which is fine on H20 when each task only uses a fraction of GPU memory).
    - Workers keep pulling from the queue until it is empty, then exit.
    """
    while True:
        task = await queue.get()

        # --- Acquire GPU dynamically (least-loaded) ---
        gpu_device_id = gpu_allocator.acquire() if gpu_allocator is not None else None

        # Create logger for the run
        run_logger = get_logger(str(task.path_to_run))
        log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
        log_file_handler.setFormatter(
            logging.getLogger().handlers[0].formatter
        )  # match the formatting we have
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        gpu_info = f" on GPU {gpu_device_id}" if gpu_device_id else ""
        run_logger.info(
            f"[Worker {idx}] Running seed {task.seed} for {task.competition.id} "
            f"and agent {task.agent.name}{gpu_info}"
        )
        tqdm.write(f"  ▶ [Worker {idx}{gpu_info}] START  {task.competition.id}")

        # --- Start log watcher (WARNING+ on terminal) ---
        agent_log_path = task.path_to_run / "logs" / "agent.log"
        watcher_label = f"{task.competition.id} W{idx}"
        log_watcher = _LogWatcher(agent_log_path, watcher_label)
        log_watcher.start()

        task_output = {}
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(
                run_in_container,
                client=client,
                competition=task.competition,
                agent=task.agent,
                image=task.agent.name,
                container_config=task.container_config,
                retain_container=args.retain,
                run_dir=task.path_to_run,
                logger=run_logger,
                gpu_device_id=gpu_device_id,
            )
            task_output["success"] = True
            elapsed = time.monotonic() - t0
            run_logger.info(
                f"[Worker {idx}] Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )
            tqdm.write(
                f"  ✅ [Worker {idx}{gpu_info}] DONE   {task.competition.id}  ({elapsed:.0f}s)"
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            stack_trace = traceback.format_exc()
            run_logger.error(type(e))
            run_logger.error(stack_trace)
            run_logger.error(
                f"Run failed for seed {task.seed}, agent {task.agent.id} and competition "
                f"{task.competition.id}"
            )
            task_output["success"] = False
            tqdm.write(
                f"  ❌ [Worker {idx}{gpu_info}] FAILED {task.competition.id}  ({elapsed:.0f}s)  {type(e).__name__}: {e}"
            )
        finally:
            log_watcher.stop()
            # Release GPU back to the pool so the next task can use it
            if gpu_allocator is not None and gpu_device_id is not None:
                gpu_allocator.release(gpu_device_id)
            tasks_outputs[task.run_id] = task_output
            # Advance the global progress bar
            if _progress_bar is not None:
                async with _progress_lock:
                    _progress_bar.update(1)
                    done = sum(1 for v in tasks_outputs.values() if v.get("success"))
                    fail = sum(1 for v in tasks_outputs.values() if not v.get("success", True))
                    _progress_bar.set_postfix(done=done, fail=fail, refresh=True)
            queue.task_done()


async def main(args):
    client = docker.from_env()
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))

    agent = agent_registry.get_agent(args.agent_id)
    if agent.privileged and not (
        os.environ.get("I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS", "False").lower()
        in ("true", "1", "t")
    ):
        raise ValueError(
            "Agent requires running in a privileged container, but the environment variable `I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS` is not set to `True`! "
            "Carefully consider if you wish to run this agent before continuing. See agents/README.md for more details."
        )

    run_group = f"{get_timestamp()}_run-group_{agent.name}"

    # Load competition ids and check all are prepared
    with open(args.competition_set, "r") as f:
        competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]
    for competition_id in competition_ids:
        competition = registry.get_competition(competition_id)
        if not is_dataset_prepared(competition):
            raise ValueError(
                f"Dataset for competition `{competition.id}` is not prepared! "
                f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
            )

    with open(args.container_config, "r") as f:
        container_config = json.load(f)

    # Create tasks for each (competition * seed)
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    for seed in range(args.n_seeds):
        for competition_id in competition_ids:
            competition = registry.get_competition(competition_id)
            run_dir = create_run_dir(competition.id, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                image=agent.name,
                agent=agent,
                competition=competition,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
                container_config=container_config,
            )
            tasks.append(task)

    # Build GPU allocator — concurrency and GPU assignment are independent.
    # The allocator hands out the least-loaded GPU at task-start, so n_workers can freely
    # exceed the number of physical GPUs (multiple tasks share a GPU).
    gpu_allocator: _RoundRobinGPUAllocator | None = None
    gpu_ids: list[str] = []
    if args.gpu_ids:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
        gpu_allocator = _RoundRobinGPUAllocator(gpu_ids)
        logger.info(
            f"GPU allocator ready: {len(gpu_ids)} GPUs × up to {args.n_workers} concurrent tasks "
            f"(up to {args.n_workers // len(gpu_ids) if gpu_ids else '?'} tasks/GPU)"
        )

    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")
    print(f"\n{'='*60}")
    print(f"  Run group  : {run_group}")
    print(f"  Tasks      : {len(tasks)}  ({len(competition_ids)} competitions × {args.n_seeds} seed(s))")
    print(f"  Workers    : {args.n_workers}  (concurrency)")
    print(f"  GPUs       : {', '.join(gpu_ids) if gpu_ids else 'auto'}")
    if gpu_ids:
        ratio = args.n_workers / len(gpu_ids)
        print(f"  Tasks/GPU  : ~{ratio:.1f}  (workers / GPUs — can be > 1 when sharing GPUs)")
    print(f"  Output     : {get_runs_dir() / run_group}")
    print(f"{'='*60}\n")

    # Create queue of tasks, and assign workers to run them
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    workers = []
    tasks_outputs = {}

    global _progress_bar
    _progress_bar = tqdm(
        total=len(tasks),
        desc="Tasks",
        unit="task",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    for idx in range(args.n_workers):
        # GPU is NOT assigned here — the worker acquires it dynamically per-task
        # via the allocator (least-loaded GPU wins).
        w = asyncio.create_task(worker(idx, queue, client, tasks_outputs, gpu_allocator=gpu_allocator))
        workers.append(w)

    # Wait for all tasks to be completed and collect results
    started_at = time.monotonic()
    await queue.join()
    time_taken = time.monotonic() - started_at

    _progress_bar.close()

    for w in workers:
        w.cancel()  # Cancel all workers now that the queue is empty

    await asyncio.gather(*workers, return_exceptions=True)

    # Print final summary to terminal
    n_success = sum(1 for v in tasks_outputs.values() if v.get("success"))
    n_fail = len(tasks_outputs) - n_success
    print(f"\n{'='*60}")
    print(f"  Run group finished in {time_taken:.0f}s")
    print(f"  ✅  Succeeded : {n_success}/{len(tasks)}")
    if n_fail:
        print(f"  ❌  Failed    : {n_fail}/{len(tasks)}")
    print(f"  Output dir  : {get_runs_dir() / run_group}")
    print(f"{'='*60}\n")

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "runs": tasks_outputs,
    }
    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)
    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each competition",
    )
    parser.add_argument(
        "--container-config",
        help="Path to a JSON file with an environment configuration; these args will be passed to `docker.from_env().containers.create`",
        type=str,
        required=False,
        default=DEFAULT_CONTAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        required=False,
        default=None,
        help="Comma-separated GPU device IDs for round-robin assignment to workers "
             "(e.g. '0,1,2,3,4,5,6,7'). Worker 0 gets GPU 0, worker 1 gets GPU 1, etc. "
             "If not provided, Docker assigns GPUs automatically.",
    )
    parser.add_argument(
        "--retain",
        help="Whether to retain the container after the run instead of removing it.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    args = parser.parse_args()
    logger = get_logger(__name__)

    asyncio.run(main(args))
