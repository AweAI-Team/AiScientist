"""
Environment Setup Subagent

This subagent handles environment setup for paper reproduction:
- Installing system packages (apt-get)
- Installing Python packages (pip) in a virtual environment
- Setting up virtual environments using venv (NOT conda)
- Configuring environment variables

IMPORTANT: The reproduction environment does NOT have conda installed.
All Python dependencies MUST be installed using pip in a venv.

Design Philosophy:
- Idempotent: Can be called multiple times safely
- Tracks installed packages for status reporting
- Updates reproduce.sh with setup commands
- Maintains /home/agent/env_status.json for state tracking
- Uses python3 -m venv for virtual environments (reproduction compatible)
"""

from __future__ import annotations

import json

from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
    SubagentConfig,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_ENV_SETUP_CONFIG,
    ENV_SETUP_BASH_DEFAULT_TIMEOUT,
    ENV_SETUP_BASH_MAX_TIMEOUT,
)
from paperbench.solvers.aiscientist.tools.basic_tool import BashToolWithTimeout
from paperbench.solvers.basicagent.tools import ReadFileChunk
from paperbench.solvers.basicagent.tools.base import Tool

# =============================================================================
# System Prompt
# =============================================================================

ENV_SETUP_SYSTEM_PROMPT = """You are an Environment Setup Specialist for paper reproduction.

## Your Mission
Set up the required environment for reproducing a research paper. This includes:
1. Installing system packages (apt-get)
2. Creating a Python virtual environment using venv
3. Installing Python packages (pip) in the venv
4. Setting up any required configurations

## CRITICAL: Virtual Environment Requirements

**The reproduction environment does NOT have conda. You MUST use venv.**

```bash
# CORRECT - Use venv (reproduction compatible)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# WRONG - Do NOT use conda (not available in reproduction)
conda create -n myenv python=3.12  # ❌ This will FAIL in reproduction

# WRONG - Do NOT hardcode /home/submission
cd /home/submission  # ❌ Grading runs from /submission, not /home/submission
```

## Guidelines

### Before Installing
1. Check what's already installed using the `check_env_status` tool
2. Read requirements if specified (e.g., from paper analysis)
3. Always run `pip install -r requirements.txt` even if packages appear to be installed

### Installation Strategy
1. **System packages first**: Use apt-get for system-level dependencies
2. **Create venv**: `python3 -m venv venv` in /home/submission
3. **Activate venv**: `source venv/bin/activate`
4. **Install packages**: Use `pip install` within the venv
5. **GPU/CUDA**: The environment has NVIDIA GPU with CUDA pre-installed

### Best Practices
- Always use venv, never conda
- Install specific versions when known (e.g., `torch==2.0.0`)
- Use `pip install -q` for quiet installation
- Handle common issues:
  - If pip fails, try `pip install --upgrade pip` first
  - For CUDA packages, ensure compatible versions (e.g., torch with CUDA 11.8/12.1)

### Recording Setup
After successful installation:
1. Use `record_env_setup` to save the setup commands
2. These commands will be added to reproduce.sh
3. Commands MUST include venv creation and activation

## Available Tools
- `bash`: Execute shell commands
- `read_file_chunk`: Read files for requirements
- `check_env_status`: Check current environment status
- `record_env_setup`: Record setup commands for reproduce.sh
- `subagent_complete`: Signal completion with summary

## Output
When done, call `subagent_complete` with:
- List of installed packages
- Any issues encountered
- Commands recorded for reproduce.sh (must include venv setup)
"""


# =============================================================================
# Specialized Tools
# =============================================================================

class CheckEnvStatusTool(Tool):
    """Check current environment status."""

    def name(self) -> str:
        return "check_env_status"

    async def execute(
        self,
        computer: ComputerInterface,
        check_packages: str = "",
    ) -> str:
        """
        Check environment status and installed packages.

        Args:
            computer: ComputerInterface
            check_packages: Comma-separated list of packages to check

        Returns:
            Environment status report
        """
        results = []

        # Check Python version
        result = await computer.send_shell_command("python3 --version 2>&1")
        results.append(f"Python: {result.output.decode('utf-8').strip()}")

        # Check pip
        result = await computer.send_shell_command("pip --version 2>&1")
        results.append(f"Pip: {result.output.decode('utf-8').strip()}")

        # Check venv status in /home/submission
        result = await computer.send_shell_command("test -d /home/submission/venv && echo 'exists' || echo 'not found'")
        venv_status = result.output.decode('utf-8').strip()
        if venv_status == "exists":
            results.append("Venv: ✅ /home/submission/venv exists")
            # Check venv Python version
            result = await computer.send_shell_command("/home/submission/venv/bin/python --version 2>&1")
            results.append(f"  Venv Python: {result.output.decode('utf-8').strip()}")
        else:
            results.append("Venv: ❌ Not created yet (run: python3 -m venv venv)")

        # Check CUDA
        result = await computer.send_shell_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo 'No GPU'")
        results.append(f"GPU: {result.output.decode('utf-8').strip()}")

        # Check specific packages if requested
        if check_packages:
            packages = [p.strip() for p in check_packages.split(",")]
            results.append("\n## Package Status")
            for pkg in packages:
                result = await computer.send_shell_command(f"pip show {pkg} 2>&1 | head -2 || echo 'Not installed'")
                output = result.output.decode('utf-8').strip()
                if "Not installed" in output or "not found" in output.lower():
                    results.append(f"- {pkg}: ❌ Not installed")
                else:
                    # Extract version
                    version = "installed"
                    for line in output.split("\n"):
                        if line.startswith("Version:"):
                            version = line.split(":")[1].strip()
                    results.append(f"- {pkg}: ✅ {version}")

        # Check env_status.json
        try:
            content = await computer.download("/home/agent/env_status.json")
            status = json.loads(content.decode("utf-8", errors="replace"))
            results.append("\n## Previous Setup Record")
            results.append(f"- Initialized: {status.get('initialized', False)}")
            results.append(f"- Packages installed: {len(status.get('installed_packages', []))}")
            if status.get('installed_packages'):
                results.append(f"- Recent: {', '.join(status.get('installed_packages', [])[-5:])}")
        except Exception:
            results.append("\n## Previous Setup Record")
            results.append("- No previous setup record found")

        return "\n".join(results)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Check environment status and installed packages.

Returns:
- Python version
- Pip version
- GPU info
- Status of specific packages (if requested)
- Previous setup record""",
            parameters={
                "type": "object",
                "properties": {
                    "check_packages": {
                        "type": "string",
                        "description": "Comma-separated list of packages to check (e.g., 'torch,numpy,transformers')",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )


class RecordEnvSetupTool(Tool):
    """Record environment setup commands for reproduce.sh."""

    def name(self) -> str:
        return "record_env_setup"

    async def execute(
        self,
        computer: ComputerInterface,
        commands: str,
        packages: str = "",
        description: str = "",
    ) -> str:
        """
        Record setup commands to be added to reproduce.sh.

        Args:
            computer: ComputerInterface
            commands: Shell commands to record (one per line)
            packages: Comma-separated list of packages installed
            description: Description of what was set up

        Returns:
            Confirmation message
        """
        # Ensure directories exist
        await computer.send_shell_command("mkdir -p /home/agent /home/submission/scripts")

        # Update env_status.json
        try:
            content = await computer.download("/home/agent/env_status.json")
            status = json.loads(content.decode("utf-8", errors="replace"))
        except Exception:
            status = {
                "initialized": False,
                "installed_packages": [],
                "setup_commands": [],
            }

        status["initialized"] = True
        if packages:
            pkg_list = [p.strip() for p in packages.split(",")]
            status["installed_packages"].extend(pkg_list)
            status["installed_packages"] = list(set(status["installed_packages"]))

        cmd_list = [c.strip() for c in commands.strip().split("\n") if c.strip()]
        status["setup_commands"].extend(cmd_list)

        await computer.upload(
            json.dumps(status, indent=2).encode("utf-8"),
            "/home/agent/env_status.json"
        )

        # Update scripts/setup_env.sh
        setup_script_path = "/home/submission/scripts/setup_env.sh"
        try:
            existing = await computer.download(setup_script_path)
            existing_content = existing.decode("utf-8", errors="replace")
        except Exception:
            # Default template with venv setup (reproduction environment has no conda)
            existing_content = """#!/bin/bash
# Environment Setup Script
# This script is sourced by reproduce.sh
#
# IMPORTANT: The reproduction environment does NOT have conda.
# All Python dependencies must be installed using pip in a venv.
#
# NOTE: The grading system may pre-create an empty venv before running
# reproduce.sh. Always install dependencies unconditionally.

set -e

echo "Setting up environment..."

# Create venv if it does not exist
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Always install dependencies (pip skips already-installed packages)
if [ -f requirements.txt ]; then
    pip install -r requirements.txt -q
fi

"""

        # Add new commands with comment
        if description:
            existing_content += f"\n# {description}\n"
        existing_content += commands.strip() + "\n"

        await computer.upload(existing_content.encode("utf-8"), setup_script_path)

        # Make executable
        await computer.send_shell_command(f"chmod +x {setup_script_path}")

        return f"""Setup recorded:
- Commands added to scripts/setup_env.sh
- Packages tracked: {packages if packages else 'N/A'}
- Description: {description if description else 'N/A'}

ACTION REQUIRED: reproduce.sh must include `source scripts/setup_env.sh` to work in the grading container."""

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Record environment setup commands for reproduce.sh.

Commands are saved to /home/submission/scripts/setup_env.sh
which should be sourced by reproduce.sh.""",
            parameters={
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "string",
                        "description": "Shell commands to record (one per line)",
                    },
                    "packages": {
                        "type": "string",
                        "description": "Comma-separated list of packages installed",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what was set up",
                    },
                },
                "required": ["commands"],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# EnvSetup Subagent
# =============================================================================

class EnvSetupSubagent(Subagent):
    """
    Subagent for environment setup.

    Handles:
    - System package installation
    - Python package installation
    - Environment configuration
    - Recording setup for reproduce.sh
    """

    @property
    def name(self) -> str:
        return "env_setup"

    def system_prompt(self) -> str:
        return ENV_SETUP_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            BashToolWithTimeout(
                default_timeout=ENV_SETUP_BASH_DEFAULT_TIMEOUT,
                max_timeout=ENV_SETUP_BASH_MAX_TIMEOUT,
            ),
            ReadFileChunk(),
            CheckEnvStatusTool(),
            RecordEnvSetupTool(),
            SubagentCompleteTool(),
        ]

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Post-process to include setup status."""
        artifacts["env_setup_complete"] = True
        artifacts["setup_script"] = "/home/submission/scripts/setup_env.sh"
        return raw_output, artifacts
