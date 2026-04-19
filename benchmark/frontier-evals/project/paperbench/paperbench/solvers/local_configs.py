"""
Pre-configured solvers and runtimes for local development.

These configurations work around chz's limitation with nested Pydantic model parameters
(ClusterConfig is a Pydantic model, not a @chz.chz class, so its sub-parameters cannot
be set via command line).

Usage in command line:
    paperbench.solver.computer_runtime=paperbench.solvers.local_configs:LocalComputerRuntime
    paperbench.reproduction.computer_runtime=paperbench.solvers.local_configs:LocalComputerRuntime

Environment Variables for GPU Configuration:
    GPU_CANDIDATE_IDS: Comma-separated list of GPU IDs to use (e.g., "2,4,7")
    GPU_COUNT: Number of GPUs per task (default: 1, use -1 for all in candidate pool)
    GPU_SHARE_MODE: Share mode (default: true) - multiple tasks share GPUs via load balancing
    GPU_AUTO_ALLOCATE: Exclusive mode file locks (default: true, only used when GPU_SHARE_MODE=false)
    
Examples:
    # Share mode (default): 4 GPUs, 16 tasks (load balanced across GPUs)
    export GPU_CANDIDATE_IDS="0,1,2,3"
    export GPU_COUNT=1
    # Then run with: runner.concurrency=16
    
    # Exclusive mode: Use only GPUs 2, 4, 7 (1 task per GPU)
    export GPU_CANDIDATE_IDS="2,4,7"
    export GPU_SHARE_MODE=false
    # Then run with: runner.concurrency=3  (max = number of GPUs)
    
    # Exclusive mode with all 8 GPUs
    export GPU_COUNT=1
    export GPU_SHARE_MODE=false
"""

import os

from alcatraz.clusters.local import LocalConfig
from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
    AlcatrazComputerRuntimeNoJupyter,
)


def _parse_gpu_candidate_ids() -> list[str] | None:
    """Parse GPU_CANDIDATE_IDS environment variable."""
    env_value = os.environ.get("GPU_CANDIDATE_IDS")
    if not env_value:
        return None
    # Parse comma-separated list: "2,4,7" -> ["2", "4", "7"]
    return [x.strip() for x in env_value.split(",") if x.strip()]


def _parse_gpu_count() -> int:
    """Parse GPU_COUNT environment variable."""
    env_value = os.environ.get("GPU_COUNT", "1")
    try:
        return int(env_value)
    except ValueError:
        return 1


def _parse_gpu_auto_allocate() -> bool:
    """Parse GPU_AUTO_ALLOCATE environment variable."""
    env_value = os.environ.get("GPU_AUTO_ALLOCATE", "true").lower()
    return env_value in ("true", "1", "yes")


def _parse_gpu_share_mode() -> bool:
    """Parse GPU_SHARE_MODE environment variable.
    
    Default is True (share mode enabled). Multiple tasks share GPUs via load balancing.
    Set to false for exclusive mode (1 task per GPU).
    
    Share mode is useful when:
    - GPUs have more memory than tasks need (e.g., H20 vs A10)
    - You want concurrency higher than available GPUs
    - Tasks are not GPU memory-intensive
    """
    env_value = os.environ.get("GPU_SHARE_MODE", "true").lower()
    return env_value in ("true", "1", "yes")


# ============================================================================
# Basic configs (no GPU)
# ============================================================================

# Pre-configured LocalConfig without pulling from registry
# local_network=True enables --network host mode for better network access
LocalConfigNoPull = LocalConfig(pull_from_registry=False, local_network=True)

# Pre-configured AlcatrazComputerRuntime for local development
LocalComputerRuntime = AlcatrazComputerRuntime(env=LocalConfigNoPull)


# ============================================================================
# GPU configs (read from environment variables)
# ============================================================================

# GPU config with all GPUs
LocalConfigGPU = LocalConfig(pull_from_registry=False, is_nvidia_gpu_env=True, local_network=True)
LocalComputerRuntimeGPU = AlcatrazComputerRuntime(env=LocalConfigGPU)

# Single GPU config (for parallel task execution, uses env vars)
# Supports both exclusive mode (gpu_auto_allocate) and share mode (gpu_share_mode)
LocalConfigSingleGPU = LocalConfig(
    pull_from_registry=False,
    is_nvidia_gpu_env=True,
    gpu_count=_parse_gpu_count(),
    gpu_candidate_ids=_parse_gpu_candidate_ids(),
    gpu_auto_allocate=_parse_gpu_auto_allocate(),
    gpu_share_mode=_parse_gpu_share_mode(),
    local_network=True,  # Enable --network host mode
)
LocalComputerRuntimeSingleGPU = AlcatrazComputerRuntime(env=LocalConfigSingleGPU)

# Alias for backward compatibility and clearer naming
LocalConfigEnvGPU = LocalConfigSingleGPU
LocalComputerRuntimeEnvGPU = LocalComputerRuntimeSingleGPU

# ============================================================================
# GPU Exclusive Mode configs (1 task per GPU, for GPU-intensive tasks)
# ============================================================================

# Exclusive GPU config: each task gets exclusive access to a GPU
# Use when tasks need full GPU memory or you want strict isolation
LocalConfigExclusiveGPU = LocalConfig(
    pull_from_registry=False,
    is_nvidia_gpu_env=True,
    gpu_count=_parse_gpu_count(),
    gpu_candidate_ids=_parse_gpu_candidate_ids(),
    gpu_auto_allocate=True,   # Enable exclusive allocation with file locks
    gpu_share_mode=False,     # Disable share mode
    local_network=True,
)
LocalComputerRuntimeExclusiveGPU = AlcatrazComputerRuntime(env=LocalConfigExclusiveGPU)


# ============================================================================
# NoJupyter configs (for reproduction - only needs shell commands)
# ============================================================================

# # NoJupyter runtime for reproduction (doesn't require python/pip/jupyter in image)
# LocalComputerRuntimeNoJupyter = AlcatrazComputerRuntimeNoJupyter(env=LocalConfigNoPull)
# LocalComputerRuntimeGPUNoJupyter = AlcatrazComputerRuntimeNoJupyter(env=LocalConfigGPU)
# LocalComputerRuntimeSingleGPUNoJupyter = AlcatrazComputerRuntimeNoJupyter(env=LocalConfigSingleGPU)
