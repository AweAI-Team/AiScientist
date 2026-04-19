"""
Implementation Subagent

This is the core subagent responsible for writing code to reproduce papers.
It has access to:
- Paper analysis files (/home/agent/paper_analysis/)
- Prioritized tasks (/home/agent/prioritized_tasks.md)
- Execution state (plan.md, impl_log.md)

It can spawn child subagents for:
- Environment setup (EnvSetupSubagent)
- Resource download (ResourceDownloadSubagent)

Design Philosophy:
- Owns code writing - all implementation happens here
- Can delegate env/resource setup but retains control
- Maintains git history with meaningful commits
- Updates impl_log.md with implementation changelog
- Reads directives from plan.md
"""

from __future__ import annotations

from typing import Any, ClassVar

from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_DOWNLOAD_CONFIG,
    DEFAULT_ENV_SETUP_CONFIG,
    IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
)
from paperbench.solvers.aiscientist.subagents.env_setup import (
    EnvSetupSubagent,
)
from paperbench.solvers.aiscientist.subagents.resource_download import (
    ResourceDownloadSubagent,
)
from paperbench.solvers.aiscientist.subagents.state_manager import (
    AddImplLogTool,
)
from paperbench.solvers.aiscientist.constants import IMPLEMENTATION_WORKSPACE_REFERENCE
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.cus_tools.aweai_mcp.linter import LinterTool
from paperbench.solvers.aiscientist.tools.basic_tool import BashToolWithTimeout
from paperbench.solvers.aiscientist.tools.github_tool import GithubTool
from paperbench.solvers.basicagent.tools import PythonTool, ReadFileChunk, SearchFile
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.google_search import WebSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.link_summary_op import LinkSummaryOpTool

# =============================================================================
# System Prompt
# =============================================================================

IMPLEMENTATION_SYSTEM_PROMPT = f"""You are an Implementation Specialist for reproducing academic papers. You receive either the full prioritization file (Initial Round) or specific fix directives (Fix Round), and you work autonomously through the tasks.

{IMPLEMENTATION_WORKSPACE_REFERENCE}

## How You Work

### Initial Round (mode="full")
You receive the full prioritized task list. **Use a breadth-first strategy** — partial implementations of many components score higher than perfecting one:
1. Read `/home/agent/prioritized_tasks.md` for the complete task list
2. **Phase 1 — Skeleton**: Create the project structure, reproduce.sh skeleton, and basic scaffolding for ALL P0 tasks (file structure, key classes/functions with correct signatures, even if implementations are stubs). Commit this skeleton early.
3. **Phase 2 — Core Implementation**: Fill in real logic for P0 tasks in priority order. For each: implement → test via bash → git_commit → move to next
4. **Phase 3 — Remaining Tasks** (if time permits): Work through P1 → P2 tasks
5. Always ensure reproduce.sh can run end-to-end (even with placeholder outputs) before deep-diving into any single component

### Fix Round (mode="fix")
You receive specific issues from experiment feedback. Focus on:
1. Read the specific fix directives provided
2. Fix the identified issues
3. Test to verify fixes
4. Git commit and complete

## Your Tools

### Information Gathering
- **read_file_chunk** — Read paper analysis, code, configs, experiment logs
- **search_file** — Search within files for keywords, patterns
- **web_search** — Search for library docs, API references, error solutions
- **link_summary** — Extract information from documentation URLs
- **github** — Search GitHub for reference implementations
  - repo mode: `github(keywords="adaptive pruning transformers", stars=">10")`
  - code mode: `github(mode="code", keywords="class PruningScheduler", language="python")`
  - file mode: `github(mode="file", url="https://github.com/owner/repo/blob/main/model.py")`

### Code Writing
- **edit_file** — Create and edit files (preferred for all file operations)
  - `create`: Create new files (parent dirs auto-created)
  - `str_replace`: Replace exact text (old_str must be unique in file)
  - `insert`: Insert text after a specific line number
- **bash** — Shell commands, quick tests. For complex multi-hunk edits, use `apply_patch` via bash:
  ```bash
  apply_patch << 'EOF'
  *** Begin Patch
  *** Update File: src/model.py
  @@
   class Model:
       def __init__(self):
  -        self.lr = 0.001
  +        self.lr = 0.0001
           self.epochs = 100
  *** End Patch
  EOF
  ```
- **python** — Quick Python execution and computation

### Environment & Resources
- **spawn_env_setup** — Install packages, configure environment (uses venv, not conda)
- **spawn_resource_download** — Download models, datasets (prefers HuggingFace; `HF_TOKEN` env var is available)

### Code Quality & Git
- **linter** — Run Ruff linter, optionally auto-fix. Always pass `venv_path` to use project venv.
- **git_commit** — Stage and commit changes. Also manages .gitignore.
- **add_impl_log** — Record changes in `/home/agent/impl_log.md`

## CRITICAL Rules

### 1. reproduce.sh Environment Setup
The reproduction environment does NOT have conda. You MUST use venv.

**pip Source Configuration (IMPORTANT):**
- The system has pre-configured `/etc/pip.conf` with the preferred pip mirror for the environment.
- **NEVER** set `PIP_CONFIG_FILE=/dev/null` in reproduce.sh — this disables the pre-configured mirror and causes pip to fall back to slow external sources, often leading to download timeouts for large packages like PyTorch.
- **NEVER** set `PIP_INDEX_URL` to `pypi.org` blindly if your environment already provides a faster mirror.
- If you must explicitly set a pip index, use the mirror configured for your environment.

**Recommended: Use `spawn_env_setup` for dependency installation.**
`spawn_env_setup` automatically generates `scripts/setup_env.sh` with correct pip configuration. reproduce.sh only needs to source it:
```bash
#!/bin/bash
set -e
# Portable path resolution (works in both development and grading containers)
REPO_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
source scripts/setup_env.sh           # venv + pip install
source scripts/download_resources.sh   # data/model downloads (if applicable)
# ... your training/evaluation commands ...
```

**If you write pip install directly in reproduce.sh** (without `spawn_env_setup`), use this pattern:
```bash
#!/bin/bash
set -e
# Portable path resolution (works in both development and grading containers)
REPO_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
# Create venv if it does not exist
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
# Always install dependencies — pip skips already-installed packages
pip install -r requirements.txt -q
# ... your training/evaluation commands ...
```

**CRITICAL PATH RULE**: NEVER hardcode `/home/submission` in ANY file — not in reproduce.sh, not in Python files, not in configs. The grading system runs submissions from `/submission` (a different path), so any hardcoded `/home/submission` causes immediate failure. Always resolve paths dynamically:
- **reproduce.sh**: Use the `BASH_SOURCE` pattern above; add `export PYTHONPATH="$REPO_DIR:$PYTHONPATH"` after `cd "$REPO_DIR"` so Python scripts can import project modules regardless of where reproduce.sh is run from.
- **Python files**: Do NOT use `sys.path.insert(0, '/home/submission')`. The `PYTHONPATH` set in reproduce.sh already handles this.

**NEVER use "if venv exists then skip pip install".** The grading system may create an empty venv before running reproduce.sh, which would cause all pip installs to be skipped and your code to fail with import errors.

### 2. reproduce.sh Robustness Rules
reproduce.sh runs in a **completely different container** during grading. The grading system may:
1. Delete your venv before running (`rm -rf venv .venv`)
2. Create an EMPTY venv before running (`python3 -m venv venv`)
3. Run `git clean -fd` to delete all untracked files
4. Use Python 3.11 or 3.12 (NOT necessarily the version you used during development)

**Your reproduce.sh and setup_env.sh MUST handle all these scenarios correctly.**

- NEVER use "if venv exists then skip pip install". Always run `pip install -r requirements.txt` unconditionally.
- ALWAYS commit all files that reproduce.sh depends on to git. Untracked files are deleted by `git clean -fd`.
- NEVER build Python from source. Use the system Python (3.11 or 3.12).
- **System libraries**: The grading container has `libgl1` (for `opencv-python`) and `Julia` (for `sbibm` ODE simulators) pre-installed. For other system packages, add `apt-get install -y <package>` at the top of reproduce.sh if needed.
- **HF dataset cache**: NEVER commit `.hf_cache/`, `.cache/`, or `huggingface/` directories to git. Stale cache metadata without actual data files causes `FileNotFoundError`. Always add these to `.gitignore` and let reproduce.sh download data fresh.
- **Large files (models, checkpoints)**: Files >10MB are excluded from the submission tarball. If your pipeline produces checkpoints that are later used for evaluation, reproduce.sh MUST include the training step that generates those checkpoints — do NOT assume checkpoints will be available from a previous run.

### 3. Commit Early, Commit Often
Your session has a time limit. **Uncommitted code is LOST.**
- Implement a small piece → test → `git_commit` → repeat
- Do NOT wait until "everything is done"

### 4. Keep reproduce.sh Updated and Self-Contained
`reproduce.sh` is the ONLY entry point for evaluation. It runs in a **fresh container** — nothing from your current session carries over (venv will be deleted).
- Update it every time you add a runnable component (training, evaluation, new dependencies).
- After creating or updating reproduce.sh, always `git_commit` it immediately — untracked files are deleted by `git clean -fd` during grading.

### 5. Dataset Integrity — NEVER Substitute, ALWAYS Download Fresh
You MUST use the exact datasets specified in the paper. Using different datasets or synthetic/random data scores **ZERO** on the rubric.

- If a dataset download fails, try alternative download methods (different URLs, HuggingFace `datasets` library, `huggingface-cli`, direct `wget`/`curl`). Use `web_search` to find working download links or mirrors.
- **NEVER** replace a dataset with a different one (e.g., substituting MNIST when the paper uses CIFAR-100).
- **NEVER** use `torch.randn()`, `np.random.rand()`, or any synthetic data as a placeholder.
- All data downloads must be reproducible: either use `spawn_resource_download` (which auto-generates `download_resources.sh`), or ensure the download commands are included in `reproduce.sh` so they work in a fresh environment.

**CRITICAL — reproduce.sh Must Include Data Download Steps:**

reproduce.sh runs in a **fresh container with NO pre-existing caches**. Your code must include the data download step — do NOT assume datasets are already available.

**Two rules:**
1. **reproduce.sh must call the download** — e.g., `load_dataset("hellaswag")`, `wget`, `huggingface-cli download`, etc. Normal download commands work fine — the grading container has no cache, so they will download automatically. No need for `force_redownload`.
2. **NEVER commit HF cache directories** — add `.hf_cache/`, `.cache/`, `huggingface/` to `.gitignore` (the git_commit tool does this by default). Committed cache metadata without actual data files causes `FileNotFoundError` during grading.

```python
from datasets import load_dataset
ds = load_dataset("hellaswag")  # downloads automatically in fresh container
```

### 6. Adaptive Hyperparameter Strategy
The "Result Analysis" scoring dimension compares your output metrics against the paper's values. Using toy hyperparameters (e.g., 1 epoch instead of 100, lr=0.1 instead of 0.001) guarantees a zero score even if the code is correct. But using exact paper hyperparameters without considering the 24h time limit causes timeouts that score zero on everything.

**CRITICAL — NEVER substitute a smaller or different model than what the paper specifies.** The grader checks that the exact model name/size from the paper is used. Using a smaller variant "for speed" or "for testing" scores 0 on all related criteria.

**The right approach balances fidelity with feasibility:**

1. **Default to paper's hyperparameters in code**: Read `paper_analysis/experiments.md` for the paper's hyperparameters (learning rate, batch size, epochs, optimizer, scheduler, etc.). **Set these as defaults** in your training scripts.
2. **Smart scaling in reproduce.sh** (only if needed for the 1-GPU/24h constraint):
   - Before running, estimate total training time for ALL experiments
   - If total time > 16h: scale epochs proportionally so everything fits in ~20h
   - If a single experiment > 8h: reduce epochs for THAT experiment only
   - **NEVER reduce to < 10% of paper's epochs** (e.g., paper=100 → minimum 10)
   - **Prefer reducing seeds (use 1 seed) over reducing epochs**
   - **Prefer reducing dataset size slightly over slashing epochs**
   - **IMPORTANT**: "Scaling" means reducing per-experiment training intensity (fewer epochs or seeds), NOT dropping experiment configurations. If the paper sweeps over multiple values (e.g., 4 widths, 3 optimizers, multiple datasets), reproduce.sh must run ALL those configurations — each with potentially fewer epochs. The grader checks that every configuration was actually executed. Running all configurations at reduced epochs scores far higher than running one configuration at full epochs.
3. Ensure training scripts print clear final metrics (e.g., `Final test accuracy: 0.XX`) and save results to output files
4. **Use `tee` in reproduce.sh** to save output: `python train.py 2>&1 | tee results/exp.log`. The grading system checks whether reproduce.sh created or modified ANY files — if it didn't, ALL Result Analysis scores are automatically zero.

### 6b. reproduce.sh Best Practices

reproduce.sh runs during grading with **NO arguments**, on **1 GPU**, with a **24-hour wall clock**:

- **Use `set +e` for experiment sections** — one experiment crashing must NOT kill the rest. Use `set -euo pipefail` for the setup phase, then `set +e` before experiments.
- **Use `| tee results/X.log`** to save each experiment's output to a file.
- **Never gate experiments behind opt-in flags** — `if [ "${{RUN_TABLE4:-0}}" = "1" ]` means Table 4 NEVER runs during grading. All implemented experiments must run unconditionally.
- **Print clear experiment markers**: `echo "=== Experiment 1: Table 1 ==="` before each experiment so the grading judge can identify what ran.

### 7. Blacklist Compliance
Resources in `/home/paper/blacklist.txt` must NOT be accessed. Check before downloading.

### 8. Size Constraints
Committed files must not exceed 1GB total. Use `.gitignore` for models, data, venv/, checkpoints.

### 9. Hardware & GPU
This environment has NVIDIA GPU(s) with CUDA drivers pre-installed. **Always use GPU for training and computation-intensive tasks.**
- Use `--device cuda`, `.to("cuda")`, `device="cuda"`, etc.
- **NEVER** use `--device cpu` for model training — CPU training is orders of magnitude slower and will timeout your session.
- Before starting training, verify GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

### 10. Time Management
Before running any long command (training, large-scale evaluation, dataset processing):
- **Estimate execution time first**. If unsure, run a small-scale test (1 epoch, small data subset) to gauge speed.
- If estimated time exceeds 1 hour, consider: Can you reduce the workload? Is GPU being used? Can you run a faster variant first?
- Your session has a time limit — a single long-running command that times out wastes the entire session.

## CRITICAL: Dependency Consistency Self-Check

**Every time you modify reproduce.sh or add new imports, run this self-check:**

```bash
# 1. Check reproduce.sh syntax
bash -n /home/submission/reproduce.sh

# 2. Find all Python imports used in your code
grep -rh "^import \|^from " /home/submission/src/ /home/submission/*.py 2>/dev/null | \
  awk '{{print $2}}' | cut -d. -f1 | sort -u > /tmp/all_imports.txt

# 3. Check which imports are NOT standard library and NOT in requirements.txt
python3 -c "
import sys
std = set(sys.stdlib_module_names)
with open('/tmp/all_imports.txt') as f:
    imports = {{l.strip() for l in f if l.strip()}}
with open('/home/submission/requirements.txt') as f:
    reqs = {{l.strip().split('==')[0].split('>=')[0].split('<')[0].lower() for l in f if l.strip() and not l.startswith('#')}}
# Map common import names to pip package names
import_to_pkg = {{'cv2':'opencv-python','PIL':'Pillow','sklearn':'scikit-learn','yaml':'pyyaml','bs4':'beautifulsoup4'}}
missing = []
for imp in imports - std:
    pkg = import_to_pkg.get(imp, imp).lower()
    if pkg not in reqs and imp not in reqs:
        missing.append(f'{{imp}} (pip: {{pkg}})')
if missing:
    print('MISSING from requirements.txt:', ', '.join(sorted(missing)))
else:
    print('All imports covered in requirements.txt')
"
```

**Common failure**: Code uses some modules but requirements.txt doesn't include them → `reproduce.sh` fails during grading.

Also do a quick **import sanity check** in the venv after installing packages — pick the key libraries your code uses and verify they actually import:
```bash
source /home/submission/venv/bin/activate && python -c "import torch; import transformers; ..."
```
Replace with your actual key imports. This catches version conflicts and broken installs that the requirements.txt check above cannot detect.

## Best Practices
- **Match paper exactly**: Same hyperparameters, architectures, seeds
- **Reference paper sections**: Add comments like `# Eq. (5) in paper`
- **Use web_search** when you encounter unfamiliar APIs or error messages

## Workflow

1. **Assess current state** (CRITICAL — do this FIRST):
   - Run `git log --oneline -15` to see recent commits
   - Read `/home/agent/exp_log.md` (latest entries) to understand what experiments found, what failed, what metrics were obtained
   - Cross-reference the exp_log with actual code: check if the issues mentioned have already been fixed by a later commit
   - This prevents you from re-fixing already-fixed issues or missing context from previous rounds
2. **Read task(s)**: Understand what needs to be done (full prioritization or specific fixes)
3. **Read paper analysis**: Check `/home/agent/paper_analysis/` for details
4. **Setup** (if needed): Spawn env_setup or resource_download
5. **Implement**: Write code following paper specifications
6. **Test**: Unit test what you just wrote. Quick options:
   - One-liner: `python -c "from module import Foo; print(Foo(...).shape)"`
   - For complex components, write a small pytest file (e.g., `tests/test_model.py`) and run `source venv/bin/activate && pytest tests/test_model.py -x -q`
7. **Commit**: `git_commit` immediately — do not defer
8. **Update reproduce.sh and requirements.txt**: Add new scripts and dependencies to the pipeline. Keep reproduce.sh runnable at all times.
9. **Repeat**: Move to next task. Prioritize **breadth** — ensure all P0 tasks have at least a basic implementation before deep-diving into any single one
10. **Final Check**: Run the dependency consistency self-check (see section above). Verify key imports: `source /home/submission/venv/bin/activate && python -c "import torch; import ..."`
11. **Log & Complete**: Call `add_impl_log`, then `subagent_complete` with summary

## Output Format

When calling subagent_complete:
```
## Summary
[What was implemented]

## Files Changed
- path/to/file.py: [what was done]

## Git Commits
- [hash] [message]

## Status
[completed/partial/blocked]

## Tasks Completed
- [list of P0/P1/P2 tasks completed]

## Issues (if any)
[Description of any problems]
```
"""


# =============================================================================
# Specialized Tools for Implementation
# =============================================================================

class SpawnEnvSetupTool(Tool):
    """Tool to spawn environment setup subagent."""

    # Set by parent
    completer_config: BasicAgentTurnCompleterConfig | None = None
    run_dir: str | None = None
    session_dir: str | None = None  # For logging to impl session directory
    _pending_constraints: dict | None = None  # Set by execute_with_constraints

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "spawn_env_setup"

    def supports_constraints(self) -> bool:
        return True

    async def execute_with_constraints(
        self, computer: ComputerInterface, constraints: dict | None = None, **kwargs: Any
    ) -> str:
        self._pending_constraints = constraints
        try:
            return await self.execute(computer, **kwargs)
        finally:
            self._pending_constraints = None

    async def execute(
        self,
        computer: ComputerInterface,
        requirements: str,
        description: str = "",
    ) -> str:
        """
        Spawn environment setup subagent.

        Args:
            computer: ComputerInterface
            requirements: What needs to be installed (packages, system deps)
            description: Additional context

        Returns:
            Setup result
        """
        if self.completer_config is None:
            return "Error: spawn_env_setup not properly configured."

        task_description = f"""Set up the required environment.

## Requirements
{requirements}

## Additional Context
{description if description else 'N/A'}

## Instructions
1. Check what's already installed
2. Install missing packages
3. Record setup commands for reproduce.sh
4. Report any issues
"""

        # Configure with session_dir for logging
        config = SubagentConfig(
            max_steps=DEFAULT_ENV_SETUP_CONFIG.max_steps,
            time_limit=DEFAULT_ENV_SETUP_CONFIG.time_limit,
            log_dir=self.session_dir,  # Log to impl session directory
        )

        subagent = EnvSetupSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
        )

        result = await subagent.run(
            computer=computer,
            task_description=task_description,
            constraints=self._pending_constraints,
        )

        return self._format_result(result)

    def _format_result(self, result: SubagentOutput) -> str:
        """Format subagent result with actionable reproduce.sh integration guidance."""
        status_icon = "✅" if result.status == SubagentStatus.COMPLETED else "❌"
        log_info = f"\n\n**Log**: {result.log_path}" if result.log_path else ""
        setup_script = result.artifacts.get('setup_script', '/home/submission/scripts/setup_env.sh')
        return f"""[EnvSetup {status_icon}] ({result.num_steps} steps, {result.runtime_seconds:.1f}s)

{result.content}

Setup script: {setup_script}

## ACTION REQUIRED for reproduce.sh

The setup commands have been saved to `{setup_script}`.
reproduce.sh runs in a **fresh container** during grading — it must be self-contained.

Ensure `/home/submission/reproduce.sh` includes this line (after `set -e`):

    source scripts/setup_env.sh

If reproduce.sh does not exist yet, create it now with at minimum:

    #!/bin/bash
    set -e
    REPO_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
    cd "$REPO_DIR"
    source scripts/setup_env.sh
    # ... your training/evaluation commands ...
{log_info}"""

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Spawn a child subagent to set up the environment.

Use this for:
- Installing Python packages
- Installing system dependencies
- Setting up configurations

The subagent will:
1. Check current environment
2. Install required packages
3. Record commands in scripts/setup_env.sh""",
            parameters={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "string",
                        "description": "What needs to be installed (e.g., 'torch, transformers, numpy')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional context about the requirements",
                    },
                },
                "required": ["requirements"],
                "additionalProperties": False,
            },
            strict=False,
        )


class SpawnResourceDownloadTool(Tool):
    """Tool to spawn resource download subagent."""

    completer_config: BasicAgentTurnCompleterConfig | None = None
    run_dir: str | None = None
    session_dir: str | None = None  # For logging to impl session directory
    _pending_constraints: dict | None = None  # Set by execute_with_constraints

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "spawn_resource_download"

    def supports_constraints(self) -> bool:
        return True

    async def execute_with_constraints(
        self, computer: ComputerInterface, constraints: dict | None = None, **kwargs: Any
    ) -> str:
        self._pending_constraints = constraints
        try:
            return await self.execute(computer, **kwargs)
        finally:
            self._pending_constraints = None

    async def execute(
        self,
        computer: ComputerInterface,
        resource_name: str,
        resource_type: str,
        destination: str,
        source_hint: str = "",
    ) -> str:
        """
        Spawn resource download subagent.

        Args:
            computer: ComputerInterface
            resource_name: Name of the resource (e.g., "bert-base-uncased")
            resource_type: Type (model/dataset/checkpoint)
            destination: Where to download to
            source_hint: Hint about where to find it (e.g., HuggingFace model ID)

        Returns:
            Download result
        """
        if self.completer_config is None:
            return "Error: spawn_resource_download not properly configured."

        task_description = f"""Download a resource for paper reproduction.

## Resource Details
- **Name**: {resource_name}
- **Type**: {resource_type}
- **Destination**: {destination}
- **Source Hint**: {source_hint if source_hint else 'Find the best source'}

## Instructions
1. Check if resource already exists
2. Download from appropriate source (prefer HuggingFace)
3. Verify download completed
4. Record download commands for reproduce.sh
5. Add to .gitignore if large
"""

        # Configure with session_dir for logging
        config = SubagentConfig(
            max_steps=DEFAULT_DOWNLOAD_CONFIG.max_steps,
            time_limit=DEFAULT_DOWNLOAD_CONFIG.time_limit,
            log_dir=self.session_dir,  # Log to impl session directory
        )

        subagent = ResourceDownloadSubagent(
            completer_config=self.completer_config,
            config=config,
            run_dir=self.run_dir,
        )

        result = await subagent.run(
            computer=computer,
            task_description=task_description,
            constraints=self._pending_constraints,
        )

        return self._format_result(result)

    def _format_result(self, result: SubagentOutput) -> str:
        """Format subagent result with actionable reproduce.sh integration guidance."""
        status_icon = "✅" if result.status == SubagentStatus.COMPLETED else "❌"
        log_info = f"\n\n**Log**: {result.log_path}" if result.log_path else ""
        download_script = result.artifacts.get('download_script', '/home/submission/scripts/download_resources.sh')
        return f"""[ResourceDownload {status_icon}] ({result.num_steps} steps, {result.runtime_seconds:.1f}s)

{result.content}

Download script: {download_script}

## ACTION REQUIRED for reproduce.sh

The download commands have been saved to `{download_script}`.
reproduce.sh runs in a **fresh container** during grading — data/models won't exist unless downloaded.

Ensure `/home/submission/reproduce.sh` includes this line:

    source scripts/download_resources.sh

If reproduce.sh does not exist yet, create it now with at minimum:

    #!/bin/bash
    set -e
    REPO_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
    cd "$REPO_DIR"
    source scripts/setup_env.sh
    source scripts/download_resources.sh
    # ... your training/evaluation commands ...
{log_info}"""

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Spawn a child subagent to download resources.

Use this for:
- Pre-trained models (prefer HuggingFace)
- Datasets
- Checkpoints

The subagent will:
1. Check if resource exists
2. Download from best source
3. Record commands in scripts/download_resources.sh
4. Add to .gitignore""",
            parameters={
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "Name of the resource (e.g., 'bert-base-uncased')",
                    },
                    "resource_type": {
                        "type": "string",
                        "enum": ["model", "dataset", "checkpoint", "other"],
                        "description": "Type of resource",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Where to download to (e.g., './models/bert')",
                    },
                    "source_hint": {
                        "type": "string",
                        "description": "Hint about source (e.g., 'huggingface:bert-base-uncased')",
                    },
                },
                "required": ["resource_name", "resource_type", "destination"],
                "additionalProperties": False,
            },
            strict=False,
        )


class GitCommitTool(Tool):
    """Tool for git operations with proper commit messages and .gitignore management."""

    # Default patterns that should always be in .gitignore
    DEFAULT_GITIGNORE_PATTERNS: ClassVar[list[str]] = [
        "# Python",
        "venv/",
        ".venv/",
        "__pycache__/",
        "*.pyc",
        "",
        "# Large files (models, data)",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.bin",
        "*.safetensors",
        "",
        "# Common directories",
        "models/",
        "data/",
        "checkpoints/",
        "outputs/",
        "logs/",
        "wandb/",
        "",
        "# Cache directories (stale cache metadata causes failures in grading)",
        ".hf_cache/",
        ".cache/",
        "huggingface/",
        "",
        "# Misc",
        ".DS_Store",
        "*.log",
    ]

    def name(self) -> str:
        return "git_commit"

    async def _ensure_gitignore(self, computer: ComputerInterface, extra_patterns: list[str] | None = None) -> str:
        """
        Ensure .gitignore exists with default patterns and add extra patterns if provided.
        Automatically deduplicates entries.

        Returns:
            Status message about .gitignore updates
        """
        gitignore_path = "/home/submission/.gitignore"
        gitignore_status = []

        # Read existing .gitignore or create default
        try:
            content = await computer.download(gitignore_path)
            existing_content = content.decode("utf-8", errors="replace")
            existing_lines = {line.strip() for line in existing_content.split("\n") if line.strip() and not line.startswith("#")}
        except Exception:
            # Create new .gitignore with default patterns
            existing_content = "\n".join(self.DEFAULT_GITIGNORE_PATTERNS) + "\n"
            existing_lines = {line.strip() for line in self.DEFAULT_GITIGNORE_PATTERNS if line.strip() and not line.startswith("#")}
            gitignore_status.append("Created .gitignore with default patterns")

        # Add extra patterns if provided (with deduplication)
        if extra_patterns:
            new_patterns = []
            for pattern in extra_patterns:
                pattern = pattern.strip()
                if pattern and pattern not in existing_lines and not pattern.startswith("#"):
                    new_patterns.append(pattern)
                    existing_lines.add(pattern)

            if new_patterns:
                # Append new patterns with a comment
                existing_content += "\n# Added by git_commit\n"
                existing_content += "\n".join(new_patterns) + "\n"
                gitignore_status.append(f"Added to .gitignore: {', '.join(new_patterns)}")

        # Write .gitignore
        await computer.upload(existing_content.encode("utf-8"), gitignore_path)

        return "; ".join(gitignore_status) if gitignore_status else ""

    async def execute(
        self,
        computer: ComputerInterface,
        files: str,
        message: str,
        task_id: str = "",
        add_to_gitignore: str = "",
    ) -> str:
        """
        Stage and commit files with a structured message.

        Args:
            computer: ComputerInterface
            files: Files to stage (space-separated or "." for all tracked)
            message: Commit message
            task_id: Task ID to prefix message with
            add_to_gitignore: Patterns to add to .gitignore (comma-separated, e.g., "data/,*.bin,models/")

        Returns:
            Commit result
        """
        # Ensure we're in the submission directory
        await computer.send_shell_command("cd /home/submission && git status >/dev/null 2>&1 || git init")

        # Handle .gitignore
        extra_patterns = [p.strip() for p in add_to_gitignore.split(",") if p.strip()] if add_to_gitignore else None
        gitignore_msg = await self._ensure_gitignore(computer, extra_patterns)

        # Stage files
        if files.strip() == ".":
            # Stage all, but be careful
            result = await computer.send_shell_command(
                "cd /home/submission && git add -A"
            )
        else:
            result = await computer.send_shell_command(
                f"cd /home/submission && git add {files}"
            )

        if result.exit_code != 0:
            return f"Error staging files: {result.output.decode('utf-8')}"

        # Format commit message
        if task_id:
            full_message = f"[{task_id}] {message}"
        else:
            full_message = message

        # Commit using temp file to avoid shell injection from message content
        msg_path = "/tmp/_git_commit_msg.txt"
        await computer.upload(full_message.encode("utf-8"), msg_path)
        result = await computer.send_shell_command(
            f"cd /home/submission && git commit -F {msg_path} && rm -f {msg_path}"
        )

        output = result.output.decode("utf-8", errors="replace")

        if result.exit_code != 0:
            if "nothing to commit" in output:
                return "No changes to commit." + (f"\n\n{gitignore_msg}" if gitignore_msg else "")
            return f"Error committing: {output}"

        # Get commit hash
        hash_result = await computer.send_shell_command(
            "cd /home/submission && git rev-parse --short HEAD"
        )
        commit_hash = hash_result.output.decode("utf-8", errors="replace").strip()

        result_msg = f"Committed: {commit_hash}\nMessage: {full_message}\n\n{output}"
        if gitignore_msg:
            result_msg += f"\n\n.gitignore: {gitignore_msg}"

        return result_msg

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Stage and commit files with a structured message.

Also manages .gitignore - will create with default patterns if missing.
Use add_to_gitignore to exclude large files (models, data, checkpoints).

Best practices:
- Commit specific files, not "." unless necessary
- Use task_id to link commit to task
- Write meaningful commit messages
- Add large file patterns to .gitignore BEFORE committing""",
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "string",
                        "description": "Files to stage (space-separated, or '.' for all)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message describing the change",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to prefix message (e.g., 'P0-1')",
                    },
                    "add_to_gitignore": {
                        "type": "string",
                        "description": "Patterns to add to .gitignore, comma-separated (e.g., 'data/,*.bin,models/'). Use this to exclude large files BEFORE committing.",
                    },
                },
                "required": ["files", "message"],
                "additionalProperties": False,
            },
            strict=False,
        )


class FileEditTool(Tool):
    """Tool for creating and editing files with create, str_replace, and insert commands."""

    def name(self) -> str:
        return "edit_file"

    async def execute(
        self,
        computer: ComputerInterface,
        command: str,
        path: str,
        content: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
    ) -> str:
        """
        Create or edit files.

        Args:
            computer: ComputerInterface
            command: One of "create", "str_replace", "insert"
            path: File path
            content: File content (for create)
            old_str: Text to replace (for str_replace, must be unique in file)
            new_str: Replacement text (for str_replace) or text to insert (for insert)
            insert_line: Line number after which to insert (for insert, 0 = beginning)

        Returns:
            Result message
        """
        if command == "create":
            return await self._create(computer, path, content)
        elif command == "str_replace":
            return await self._str_replace(computer, path, old_str, new_str)
        elif command == "insert":
            return await self._insert(computer, path, insert_line, new_str)
        else:
            return f"Error: Unknown command '{command}'. Use 'create', 'str_replace', or 'insert'."

    async def _create(self, computer: ComputerInterface, path: str, content: str) -> str:
        """Create a new file with content."""
        import os
        import shlex
        dir_path = os.path.dirname(path)
        if dir_path:
            await computer.send_shell_command(f"mkdir -p {shlex.quote(dir_path)}")

        await computer.upload(content.encode("utf-8"), path)

        result = await computer.send_shell_command(f"wc -l < {path}")
        lines = result.output.decode("utf-8", errors="replace").strip()
        return f"Created: {path} ({lines} lines)"

    async def _str_replace(
        self, computer: ComputerInterface, path: str, old_str: str, new_str: str
    ) -> str:
        """Replace exact string in file. old_str must be unique."""
        if not old_str:
            return "Error: old_str is required for str_replace."

        try:
            file_bytes = await computer.download(path)
            file_content = file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error: Cannot read {path}: {e}"

        count = file_content.count(old_str)
        if count == 0:
            # Show a snippet to help the model debug
            preview = file_content[:500] + "..." if len(file_content) > 500 else file_content
            return (
                f"Error: old_str not found in {path}. "
                f"Make sure the text matches exactly (including whitespace/indentation).\n"
                f"File preview:\n```\n{preview}\n```"
            )
        if count > 1:
            return (
                f"Error: old_str appears {count} times in {path}. "
                f"Include more surrounding context to make it unique."
            )

        new_content = file_content.replace(old_str, new_str, 1)
        await computer.upload(new_content.encode("utf-8"), path)

        # Count changed lines for feedback
        old_lines = old_str.count("\n") + 1
        new_lines = new_str.count("\n") + 1
        return f"Edited: {path} (replaced {old_lines} lines with {new_lines} lines)"

    async def _insert(
        self, computer: ComputerInterface, path: str, insert_line: int, new_str: str
    ) -> str:
        """Insert text after a specific line number (0 = beginning of file)."""
        if not new_str:
            return "Error: new_str is required for insert."

        try:
            file_bytes = await computer.download(path)
            file_content = file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error: Cannot read {path}: {e}"

        lines = file_content.splitlines(keepends=True)
        total_lines = len(lines)

        if insert_line < 0 or insert_line > total_lines:
            return f"Error: insert_line {insert_line} out of range (file has {total_lines} lines, valid: 0-{total_lines})."

        # Ensure new_str ends with newline for clean insertion
        insert_text = new_str if new_str.endswith("\n") else new_str + "\n"

        lines.insert(insert_line, insert_text)
        new_content = "".join(lines)
        await computer.upload(new_content.encode("utf-8"), path)

        inserted_count = insert_text.count("\n")
        return f"Inserted {inserted_count} lines after line {insert_line} in {path}"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Create and edit files.

**Commands:**

1. `create` — Create a new file (parent directories auto-created)
   ```
   edit_file(command="create", path="/home/submission/src/model.py", content="import torch\\n...")
   ```

2. `str_replace` — Replace exact text in a file (old_str must be unique)
   ```
   edit_file(command="str_replace", path="/home/submission/src/model.py",
             old_str="self.lr = 0.001", new_str="self.lr = 0.0001")
   ```

3. `insert` — Insert text after a line number (0 = beginning of file)
   ```
   edit_file(command="insert", path="/home/submission/src/model.py",
             insert_line=1, new_str="import numpy as np")
   ```

For complex multi-hunk edits, use `apply_patch` via bash instead.""",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["create", "str_replace", "insert"],
                        "description": "The edit command to run",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute file path",
                    },
                    "content": {
                        "type": "string",
                        "description": "[create] Content for the new file",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "[str_replace] Exact text to find (must be unique in file). Include enough context lines to be unique.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "[str_replace] Replacement text. [insert] Text to insert.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "[insert] Line number after which to insert (0 = beginning of file)",
                    },
                },
                "required": ["command", "path"],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# Implementation Subagent
# =============================================================================

class ImplementationSubagent(Subagent):
    """
    Core subagent for code implementation.

    This subagent:
    - Writes code to reproduce papers
    - Can spawn env_setup and resource_download subagents
    - Manages git commits
    - Updates execution state
    """

    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,
        config: SubagentConfig | None = None,
        run_dir: str | None = None,
        session_dir: str | None = None,
    ):
        super().__init__(completer_config, config, run_dir)
        self.session_dir = session_dir  # For child subagent logs

    @property
    def name(self) -> str:
        return "implementation"

    def system_prompt(self) -> str:
        return IMPLEMENTATION_SYSTEM_PROMPT

    def _construct_reminder(self, elapsed_time: float) -> ChatCompletionMessageParam:
        """Enhanced reminder with GitHub tool nudge and reproduce.sh reminder."""
        hours = elapsed_time / 3600
        time_limit_hours = self.config.time_limit / 3600

        if hours >= 1.0:
            time_str = f"{hours:.1f}h"
        else:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        msg = (
            f"[Implementation Reminder] Time elapsed: {time_str} / {time_limit_hours:.1f}h.\n"
        )

        time_ratio = elapsed_time / self.config.time_limit if self.config.time_limit else 0.0

        if time_ratio >= 0.90:
            msg += (
                "URGENT: Almost out of time! "
                "Commit all changes NOW with git_commit, "
                "ensure reproduce.sh is up to date, then call subagent_complete."
            )
        elif time_ratio >= 0.60:
            msg += (
                "Over 60% time used. Wrap up current work:\n"
                "- git_commit any uncommitted changes\n"
                "- Update reproduce.sh with all runnable components\n"
                "- Verify reproduce.sh dependencies match requirements.txt"
            )
        else:
            msg += (
                "Reminders:\n"
                "- git_commit after each working component\n"
                "- Keep reproduce.sh updated with every new runnable script\n"
                "- Use `github(mode='code', keywords='...')` to find reference implementations "
                "when paper details are unclear or you need baseline code\n"
                "- Ensure requirements.txt includes ALL imports used in your code"
            )

        return {"role": "user", "content": msg}

    def get_tools(self) -> list[Tool]:
        # Create spawn tools with completer config and session_dir
        spawn_env = SpawnEnvSetupTool()
        spawn_env.completer_config = self.completer_config
        spawn_env.run_dir = self.run_dir
        spawn_env.session_dir = self.session_dir  # For child subagent logs

        spawn_download = SpawnResourceDownloadTool()
        spawn_download.completer_config = self.completer_config
        spawn_download.run_dir = self.run_dir
        spawn_download.session_dir = self.session_dir  # For child subagent logs

        tools = [
            # Information gathering
            ReadFileChunk(),
            SearchFile(),
            WebSearchTool(),
            LinkSummaryOpTool(),
            GithubTool(),

            # Code writing
            FileEditTool(),
            BashToolWithTimeout(
                default_timeout=IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
                max_timeout=self.config.time_limit,
            ),
            PythonTool(),

            # Code quality
            LinterTool(),

            # Git
            GitCommitTool(),

            # Spawn child subagents
            spawn_env,
            spawn_download,

            # Logging & completion
            AddImplLogTool(),
            SubagentCompleteTool(),
        ]

        return tools

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Post-process to include implementation status."""
        artifacts["implementation_complete"] = True
        return raw_output, artifacts
