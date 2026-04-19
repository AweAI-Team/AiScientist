"""
Resource Download Subagent

This subagent handles downloading resources for paper reproduction:
- Pre-trained models (primarily via HuggingFace)
- Datasets (HuggingFace datasets, direct downloads)
- Other assets (checkpoints, configs)

Design Philosophy:
- Use HuggingFace for models and datasets when possible
- Track downloads to avoid redundant operations
- Update reproduce.sh with download commands
- Maintain /home/agent/download_status.json for state tracking
- Don't commit large files to git (use .gitignore)
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
    DEFAULT_DOWNLOAD_CONFIG,
)
from paperbench.solvers.aiscientist.tools.basic_tool import BashToolWithTimeout
from paperbench.solvers.basicagent.tools import ReadFileChunk
from paperbench.solvers.basicagent.tools.base import Tool

# =============================================================================
# System Prompt
# =============================================================================

RESOURCE_DOWNLOAD_SYSTEM_PROMPT = """You are a Resource Download Specialist for paper reproduction.

## Your Mission
Download required resources (models, datasets, assets) for reproducing a research paper.

## Guidelines

### Before Downloading
1. Check what's already downloaded using `check_download_status`
2. Verify the download path exists
3. Check available disk space if downloading large files

### Download Methods (in order of preference)

#### 1. HuggingFace (Preferred)
For pre-trained models and datasets:
```python
# Models
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name", cache_dir="./models")

# Or via huggingface-cli
huggingface-cli download model-name --local-dir ./models
```

```python
# Datasets
from datasets import load_dataset
dataset = load_dataset("dataset-name", cache_dir="./data")
```

#### 2. Direct Download
For other resources:
```bash
# Using wget
wget -O output_path URL

# Using curl
curl -L -o output_path URL
```

### API Keys
- `HF_TOKEN` environment variable is available for HuggingFace downloads
- For CLI usage: `huggingface-cli download` will automatically use `HF_TOKEN`

### Best Practices
1. **Specify download paths clearly**: Use ./models, ./data, ./checkpoints
2. **Check file exists before downloading**: Avoid redundant downloads
3. **Handle errors gracefully**: Network issues, missing files
4. **Record downloads**: Use `record_download` to track what was downloaded

### Storage Considerations
- Don't download to /home/submission (will be committed to git)
- Use /home/agent/downloads or /tmp for temporary storage
- For reproduce.sh, download to relative paths (./models, ./data)

## Recording Downloads
After successful download:
1. Use `record_download` to save the download command
2. Commands will be added to scripts/download_resources.sh
3. This script is called by reproduce.sh

## Available Tools
- `bash`: Execute shell commands
- `read_file_chunk`: Read files for requirements
- `check_download_status`: Check what's already downloaded
- `record_download`: Record download commands
- `subagent_complete`: Signal completion with summary

## Output
When done, call `subagent_complete` with:
- List of downloaded resources
- Paths where they were saved
- Any issues encountered
"""


# =============================================================================
# Specialized Tools
# =============================================================================

class CheckDownloadStatusTool(Tool):
    """Check status of downloaded resources."""

    def name(self) -> str:
        return "check_download_status"

    async def execute(
        self,
        computer: ComputerInterface,
        paths: str = "",
    ) -> str:
        """
        Check download status and existing resources.

        Args:
            computer: ComputerInterface
            paths: Comma-separated list of paths to check

        Returns:
            Status report
        """
        results = []

        # Check common download directories
        common_dirs = [
            "/home/submission/models",
            "/home/submission/data",
            "/home/submission/checkpoints",
            "/home/agent/downloads",
        ]

        results.append("## Download Directories")
        for dir_path in common_dirs:
            result = await computer.send_shell_command(f"ls -la {dir_path} 2>&1 | head -5 || echo 'Not exists'")
            output = result.output.decode("utf-8", errors="replace").strip()
            if "Not exists" in output or "No such file" in output:
                results.append(f"- {dir_path}: ❌ Not exists")
            else:
                results.append(f"- {dir_path}: ✅ Exists")
                # Count files
                result = await computer.send_shell_command(f"find {dir_path} -type f 2>/dev/null | wc -l")
                count = result.output.decode("utf-8", errors="replace").strip()
                results.append(f"  Files: {count}")

        # Check specific paths if requested
        if paths:
            results.append("\n## Specific Paths")
            for path in [p.strip() for p in paths.split(",")]:
                result = await computer.send_shell_command(f"ls -lh {path} 2>&1")
                output = result.output.decode("utf-8", errors="replace").strip()
                if "No such file" in output:
                    results.append(f"- {path}: ❌ Not found")
                else:
                    results.append(f"- {path}: ✅ Exists")
                    results.append(f"  {output}")

        # Check download_status.json
        try:
            content = await computer.download("/home/agent/download_status.json")
            status = json.loads(content.decode("utf-8", errors="replace"))
            results.append("\n## Previous Downloads")
            for item in status.get("downloads", [])[-5:]:
                results.append(f"- {item.get('name', 'Unknown')}: {item.get('path', 'N/A')}")
        except Exception:
            results.append("\n## Previous Downloads")
            results.append("- No download record found")

        # Check disk space
        result = await computer.send_shell_command("df -h /home | tail -1")
        results.append(f"\n## Disk Space\n{result.output.decode('utf-8').strip()}")

        return "\n".join(results)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Check download status and existing resources.

Returns:
- Status of common download directories
- Status of specific paths (if requested)
- Previous download record
- Disk space""",
            parameters={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "string",
                        "description": "Comma-separated list of specific paths to check",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )


class RecordDownloadTool(Tool):
    """Record download commands for reproduce.sh."""

    def name(self) -> str:
        return "record_download"

    async def execute(
        self,
        computer: ComputerInterface,
        name: str,
        path: str,
        commands: str,
        source: str = "",
        size: str = "",
    ) -> str:
        """
        Record download commands to be added to reproduce.sh.

        Args:
            computer: ComputerInterface
            name: Name of the resource (e.g., "bert-base-uncased")
            path: Where it was downloaded to
            commands: Shell commands to reproduce the download
            source: Source (e.g., "huggingface", "direct")
            size: Approximate size

        Returns:
            Confirmation message
        """
        # Ensure directories exist
        await computer.send_shell_command("mkdir -p /home/agent /home/submission/scripts")

        # Update download_status.json
        try:
            content = await computer.download("/home/agent/download_status.json")
            status = json.loads(content.decode("utf-8", errors="replace"))
        except Exception:
            status = {"downloads": []}

        status["downloads"].append({
            "name": name,
            "path": path,
            "source": source,
            "size": size,
        })

        await computer.upload(
            json.dumps(status, indent=2).encode("utf-8"),
            "/home/agent/download_status.json"
        )

        # Update scripts/download_resources.sh
        script_path = "/home/submission/scripts/download_resources.sh"
        try:
            existing = await computer.download(script_path)
            existing_content = existing.decode("utf-8", errors="replace")
        except Exception:
            existing_content = """#!/bin/bash
# Resource Download Script
# This script is sourced by reproduce.sh

set -e

echo "Downloading resources..."

"""

        # Add new download with check for existing
        download_block = f"""
# Download: {name}
# Source: {source}, Size: {size}
if [ ! -e "{path}" ]; then
    echo "Downloading {name}..."
{self._indent_commands(commands)}
else
    echo "{name} already exists, skipping..."
fi
"""
        existing_content += download_block

        await computer.upload(existing_content.encode("utf-8"), script_path)
        await computer.send_shell_command(f"chmod +x {script_path}")

        # Update .gitignore to exclude downloaded files
        gitignore_path = "/home/submission/.gitignore"
        try:
            gitignore = await computer.download(gitignore_path)
            gitignore_content = gitignore.decode("utf-8", errors="replace")
        except Exception:
            gitignore_content = """# Auto-generated .gitignore
# Large files should not be committed

"""

        # Add path to gitignore if not already there
        rel_path = path.replace("/home/submission/", "")
        if rel_path not in gitignore_content:
            gitignore_content += f"\n# {name}\n{rel_path}\n"
            await computer.upload(gitignore_content.encode("utf-8"), gitignore_path)

        return f"""Download recorded:
- Name: {name}
- Path: {path}
- Source: {source}
- Added to scripts/download_resources.sh
- Added to .gitignore

ACTION REQUIRED: reproduce.sh must include `source scripts/download_resources.sh` to download data/models in the grading container."""

    def _indent_commands(self, commands: str) -> str:
        """Indent commands for script block."""
        lines = commands.strip().split("\n")
        return "\n".join(f"    {line}" for line in lines)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Record download commands for reproduce.sh.

Commands are saved to /home/submission/scripts/download_resources.sh
which should be sourced by reproduce.sh.

Also adds the downloaded path to .gitignore.""",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the resource (e.g., 'bert-base-uncased')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Where it was downloaded to",
                    },
                    "commands": {
                        "type": "string",
                        "description": "Shell commands to reproduce the download",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source (e.g., 'huggingface', 'direct', 'gdown')",
                    },
                    "size": {
                        "type": "string",
                        "description": "Approximate size (e.g., '420MB')",
                    },
                },
                "required": ["name", "path", "commands"],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# ResourceDownload Subagent
# =============================================================================

class ResourceDownloadSubagent(Subagent):
    """
    Subagent for downloading resources.

    Handles:
    - Pre-trained models (HuggingFace)
    - Datasets
    - Other assets
    - Recording downloads for reproduce.sh
    """

    @property
    def name(self) -> str:
        return "resource_download"

    def system_prompt(self) -> str:
        return RESOURCE_DOWNLOAD_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        # Use BashToolWithTimeout to prevent downloads from hanging indefinitely.
        # default 10min per command, max capped to the subagent's time_limit (20min).
        return [
            BashToolWithTimeout(
                default_timeout=600,
                max_timeout=DEFAULT_DOWNLOAD_CONFIG.time_limit,
            ),
            ReadFileChunk(),
            CheckDownloadStatusTool(),
            RecordDownloadTool(),
            SubagentCompleteTool(),
        ]

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Post-process to include download status."""
        artifacts["download_complete"] = True
        artifacts["download_script"] = "/home/submission/scripts/download_resources.sh"
        return raw_output, artifacts
