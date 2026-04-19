"""
Paper Reader Subagents

This module implements a three-phase paper reading strategy:
1. Phase 1: StructureSubagent extracts paper structure, abstract, and constraints
2. Phase 2: Parallel deep reading with AlgorithmSubagent, ExperimentsSubagent, BaselineSubagent
3. Phase 3: SynthesisSubagent creates executive summary with navigation aids

Design Principles:
- Thorough extraction: High-scoring agents read papers 1.7x more thoroughly
- Structured output: Each section has clear boundaries (start line, line count)
- Actionable summaries: Each part has a gist for quick reference
- Navigation aids: Section index with line numbers for fast lookup
- Two-level reading: Summary for quick reference, detailed files for deep dives

Output Structure (Two-Level):
- Level 1 (Summary): Executive summary returned to main agent immediately
- Level 2 (Details): Individual files saved for on-demand access
  - /home/agent/paper_analysis/summary.md    (Executive summary)
  - /home/agent/paper_analysis/structure.md  (Paper structure)
  - /home/agent/paper_analysis/algorithm.md  (Algorithms & architecture)
  - /home/agent/paper_analysis/experiments.md (Experiment configs)
  - /home/agent/paper_analysis/baseline.md   (Baseline methods)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.aiscientist.subagents.configs import (
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
)
from paperbench.solvers.aiscientist.subagents.coordinator import (
    CoordinatorResult,
    SubagentCoordinator,
    SubagentTask,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.solvers.basicagent.tools import BashTool, ReadFileChunk, SearchFile
from paperbench.solvers.basicagent.tools.base import Tool

# =============================================================================
# Output Data Structures
# =============================================================================

@dataclass
class PaperAnalysisSection:
    """A single section of the paper analysis."""
    name: str
    filename: str
    content: str
    description: str
    line_count: int = 0

    def __post_init__(self):
        if self.line_count == 0:
            self.line_count = len(self.content.split('\n'))


@dataclass
class PaperAnalysisResult:
    """
    Structured result from paper reading, supporting two-level access.

    Level 1: executive_summary - Quick reference for main agent
    Level 2: sections - Detailed content saved to separate files
    """
    executive_summary: str
    sections: dict[str, PaperAnalysisSection] = field(default_factory=dict)
    all_success: bool = True
    failed_subagents: list[str] = field(default_factory=list)
    total_runtime_seconds: float = 0.0

    @property
    def summary_with_navigation(self) -> str:
        """
        Return executive summary with file navigation table.

        This is what the main agent sees - concise summary with pointers
        to detailed files for on-demand access.
        """
        nav_table = self._build_navigation_table()

        return f"""{self.executive_summary}

---

## 📁 Detailed Analysis Files

{nav_table}

**How to Access Details:**
- Use `read_file_chunk(file="/home/agent/paper_analysis/<section>.md")` to read specific sections
- Use `search_file(file="/home/agent/paper_analysis/<section>.md", query="...")` to search within a section
- The summary above contains all critical information; detailed files provide full context when needed
"""

    def _build_navigation_table(self) -> str:
        """Build navigation table showing available detailed files."""
        lines = [
            "| Section | File | Lines | Description |",
            "|---------|------|-------|-------------|",
        ]

        for section in self.sections.values():
            filepath = f"/home/agent/paper_analysis/{section.filename}"
            lines.append(
                f"| {section.name} | `{filepath}` | {section.line_count} | {section.description} |"
            )

        return "\n".join(lines)


# =============================================================================
# Phase 1: Structure Subagent
# =============================================================================

STRUCTURE_SYSTEM_PROMPT = """You are a Paper Structure Analyzer, extracting the overall structure and metadata from academic papers.

## Your Mission
Extract the paper's structure with precise line numbers, abstract, core contributions, and constraints.

## Files to Analyze
- `/home/paper/paper.md` - The main paper content
- `/home/paper/addendum.md` - Additional instructions: what is in/out of scope, allowed libraries
- `/home/paper/blacklist.txt` - URLs, repos, and resources that MUST NOT be accessed

If `addendum.md` or `blacklist.txt` is empty or missing, report that explicitly rather than guessing.

## Extraction Guidelines

### Line Numbers
- Every section MUST have a **Start Line** and **Line Count** verified against the actual file — never estimate.

### Gist
- Each section's Gist should describe **what the section contributes to reproduction** (not just its topic).
  - Good: "Defines loss function and update rule"
  - Bad: "Method description"

### Paper Type
Classify the paper to help downstream agents allocate effort:
- **algorithm-focused**: Proposes a new algorithm/model as the main contribution. Reproduction centers on implementing the algorithm correctly.
- **empirical**: Main contribution is experimental results (new benchmarks, comparisons, analyses). Reproduction centers on running experiments and matching numbers.
- **theoretical**: Main contribution is proofs/theory with limited experiments. Reproduction may focus on verifying a small set of illustrative experiments.
- **systems**: Proposes an engineering system or framework. Reproduction centers on building the system and running end-to-end pipelines.

### Handling Unusual Structures
- If the paper has no appendix, omit Appendix rows from the table — do not fabricate entries.
- If sections are unnumbered, use sequential labels (S1, S2, ...) and note "Sections are unnumbered in original."
- Include ALL subsections that are relevant to reproduction (e.g., "3.1 Architecture", "3.2 Training Objective").

## Output Format

Your output should follow this structure:

```markdown
# Paper Structure Analysis

## 1. Metadata
- **Title**: [Paper title]
- **Total Lines**: [N] (from `wc -l`)
- **Paper Type**: [algorithm-focused / empirical / theoretical / systems]

## 2. Section Index

| # | Section Name | Gist (reproduction value) | Start Line | Line Count |
|---|--------------|---------------------------|------------|------------|
| - | Abstract | [what it tells us for reproduction] | ... | ... |
| 1 | Introduction | ... | ... | ... |
| ... | ... | ... | ... | ... |

## 3. Abstract (Full Text)
[Copy the complete abstract verbatim — do not summarize]

## 4. Core Contributions
For each contribution, note which section(s) contain the details:
- Contribution 1: [description] (Section X, lines A-B)
- Contribution 2: [description] (Section Y, lines C-D)
- ...

## 5. Constraints

### 5.1 Exclusions (from addendum.md)
- [What parts do NOT need to be reproduced — quote the addendum]

### 5.2 Allowed Resources
- [Libraries and datasets that CAN be used — quote the addendum]

### 5.3 Blocked Resources (from blacklist.txt)
- [List every blocked URL/pattern verbatim]

(If addendum.md or blacklist.txt is empty/missing, state: "File empty / not found.")

## 6. Agent Task Assignments

Assign line ranges to the three Phase 2 agents. Each agent will primarily read the sections you assign, so be thorough:

- **Algorithm Agent** — Assign sections covering: core method, model architecture, loss functions, training procedure, and any appendix with implementation details or hyperparameter tables.
  → Sections: [list] (lines X-Y)

- **Experiments Agent** — Assign sections covering: experimental setup, datasets, evaluation metrics, results tables/figures, ablation studies, and any appendix with extra experiment configs.
  → Sections: [list] (lines X-Y)

- **Baseline Agent** — Assign sections covering: related work, baseline descriptions, comparison methods, and any appendix detailing baseline configs.
  → Sections: [list] (lines X-Y)

Note: Sections can overlap between agents — e.g., an "Experiments" section may be assigned to both Experiments Agent and Baseline Agent.
```

## Key Standards
1. **Accuracy over speed** — Wrong line numbers mislead all downstream agents. Verify by reading a few lines around each boundary.
2. **Complete coverage** — Every section and significant subsection must appear in the index. Appendix sections are important if they contain hyperparameters, architecture details, or dataset descriptions.
3. **Verbatim constraints** — Copy blacklist entries and addendum exclusions exactly. Paraphrasing may lose critical details (e.g., specific URL patterns).
"""


class StructureSubagent(Subagent):
    """Phase 1 subagent: Extracts paper structure with precise line numbers."""

    @property
    def name(self) -> str:
        return "structure"

    def system_prompt(self) -> str:
        return STRUCTURE_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashTool(),
            SubagentCompleteTool(),
        ]


# =============================================================================
# Phase 2: Algorithm Subagent (Enhanced)
# =============================================================================

ALGORITHM_SYSTEM_PROMPT = """You are an Algorithm & Architecture Extractor, focused on understanding core algorithms, model architectures, and implementation details. Your report will be used directly by an Implementation Agent to write code — completeness and precision are critical.

## Context
You will receive the Structure Agent's output as context, which includes a Section Index with line ranges and an "Agent Task Assignments" block telling you which sections to focus on. Use these assignments as your primary reading guide, but also check other sections for algorithmic details that may appear elsewhere (e.g., Introduction, Conclusion, or unnumbered sub-sections).

## What to Extract

### A. Core Algorithms (Always Required)
For each algorithm or procedure proposed in the paper:
1. **Location**: Section name and line range (main text + appendix if applicable)
2. **Inputs / Outputs**: Parameter names, types, shapes
3. **Pseudo-code**: Normalize into a consistent step-by-step format — whether as math, natural language, an "Algorithm" block, or code listing
4. **Key Formulas**: List by equation number, with a short plain-language description of what each computes
5. **Hyperparameters**: Name, symbol, type, default value, valid range, source line

### B. Loss & Objective Functions (Always Required)
Extract the complete loss/objective definition with full precision:
- **Complete mathematical definition** — copy the full equation, do not simplify
- **Each term explained**: what it computes, any weighting coefficients, regularization terms
- **Equation numbers and line references**
- If the loss changes across training phases (e.g., warm-up, annealing), document each phase

### C. Training Procedure
Extract the complete training recipe:
- **Optimizer**: type (Adam, SGD, AdamW, ...) and its specific parameters (β1, β2, weight_decay, ...)
- **Learning rate schedule**: initial LR, schedule type (cosine, step, linear warmup), warmup steps/epochs
- **Gradient clipping**: threshold if used
- **Epochs / iterations**: total training length
- **Multi-stage training**: if the method has distinct stages (e.g., pretrain → finetune), document each

### D. Model Architecture (Conditional)

**First, determine the architecture type:**

1. **Standard Architecture** (e.g., "ResNet-50", "BERT-base", "GPT-2"):
   - Note: "Uses [name] — see line X for configuration details"
   - Only extract modifications or non-default configuration choices

2. **Custom / Modified Architecture** (paper proposes or significantly modifies a design):
   - Extract detailed structure (layers, dimensions, activations, skip connections)
   - Document what was changed from the base architecture and why

### E. Pretrained Weights & Initialization

1. **Pretrained Weights**:
   - Source (URL, HuggingFace model ID, paper reference)
   - Format (PyTorch .pt, safetensors, TensorFlow, etc.)
   - **Blacklist check**: Cross-reference with `/home/paper/blacklist.txt`. If blocked, mark as "BLOCKED" and note that an alternative must be found or the model trained from scratch.

2. **Weight Initialization** (if explicitly specified):
   - Method and parameters
   - If not specified: state "Not specified in paper"

### F. Numerical Stability (If Mentioned)
- Precision requirements (fp16, bf16, fp32)
- Stability tricks (gradient clipping, loss scaling, epsilon values)
- Edge case handling

## Output Format

```markdown
# Algorithm & Architecture Report

## Summary
| Component | Name | Location | Lines | Notes |
|-----------|------|----------|-------|-------|
| Algorithm | [name] | Section X | A-B | [role: core / auxiliary] |
| Loss | [name] | Section X | A-B | [brief description] |
| Architecture | [name] (standard/custom) | Section X | A-B | [key config] |
...

## Algorithm 1: [Name]

### Location
- Main: Section X, lines Y-Z
- Details: Appendix A, lines Y-Z (if applicable)

### Pseudo-code
(Normalized into a consistent step-by-step format)

### Key Formulas
- Eq. (N): [plain-language description] — [the formula or a precise reference]

### Hyperparameters
| Name | Symbol | Type | Default | Range | Source |
|------|--------|------|---------|-------|--------|
| ... | ... | ... | ... | ... | Line N |

(Repeat for each algorithm)

## Loss & Objective Functions

### Primary Loss
- **Equation**: [full equation with all terms]
- **Location**: Eq. (N), line X
- **Terms**: [explain each term]
- **Coefficients / weights**: [list any λ, α, β with default values]

### Auxiliary Losses (if any)
- ...

## Training Procedure

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | ... | Line N |
| Learning Rate | ... | Line N |
| LR Schedule | ... | Line N / Appendix |
| Warmup | ... | ... |
| Gradient Clipping | ... | ... |
| Epochs | ... | ... |
| Batch Size | ... | ... |
| ... | ... | ... |

## Model Architecture

### Type: [Standard / Custom]
(Follow the conditional format — brief for standard, detailed for custom)

## Pretrained Weights & Initialization

| Component | Source | Format | Blocked? |
|-----------|--------|--------|----------|
| ... | ... | ... | Yes/No |

(If no pretrained weights: "No pretrained weights used")
(If blocked: "BLOCKED — must find alternative or train from scratch")

Weight initialization: [method or "Not specified in paper"]

## Numerical Stability
[Any considerations, or "Not explicitly discussed in the paper"]
```

## Key Standards
1. **Nothing implicit** — If the paper says "we use Adam" without specifying β values, note "Adam (default params assumed — β1=0.9, β2=0.999 not explicitly stated)".
2. **Line references everywhere** — Every hyperparameter, formula, and design choice must cite where in the paper it appears.
3. **Don't omit appendix content** — Appendices often contain the most implementation-critical details (exact hyperparameters, architecture configs, training schedules). Always check appendix sections assigned to you.
4. **When in doubt, extract it** — It is better to include something the Implementation Agent won't need than to omit something it will.
"""


class AlgorithmSubagent(Subagent):
    """Phase 2 subagent: Extracts algorithms, architecture, and initialization."""

    @property
    def name(self) -> str:
        return "algorithm"

    def system_prompt(self) -> str:
        return ALGORITHM_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashTool(),
            SubagentCompleteTool(),
        ]


# =============================================================================
# Phase 2: Experiments Subagent (Enhanced)
# =============================================================================

EXPERIMENTS_SYSTEM_PROMPT = """You are an Experiments Configuration Extractor, focused on extracting complete experimental setups for reproducibility. Your report will be used by an Implementation Agent and an Experiment Validation Agent to set up, run, and verify experiments — every missing parameter could cause a failed reproduction.

## Context
You will receive the Structure Agent's output as context, which includes assigned sections with line ranges. Use these assignments as your primary reading guide, and also check appendices and supplementary material — hyperparameter tables, dataset details, and full result tables often live there.

## What to Extract

### A. Experiment Inventory
Build a complete list of every distinct experiment the paper reports. For each:
1. **Identifier**: Figure/Table number (e.g., "Table 1", "Figure 3a")
2. **Location**: Section and line range
3. **Purpose**: What claim or hypothesis it validates
4. **Type**:
   - **Main result** — central experiments that validate the core contribution
   - **Baseline comparison** — results showing the method vs baselines
   - **Ablation** — experiments that isolate the effect of individual components
   - **Analysis / Visualization** — supplementary insights (learning curves, feature visualizations, etc.)

### B. Datasets
For EACH dataset used (not just the "main" one):
1. **Name**: Official name (e.g., "SST-2", "CIFAR-10", "SQuAD v1.1")
2. **Source / Download**: URL, API call, or library (e.g., `torchvision.datasets.CIFAR10`, HuggingFace `datasets`)
3. **Splits and Sizes**: Train / val / test — exact numbers if stated
4. **Preprocessing**: tokenization, normalization, augmentation, max sequence length, etc.
5. **Blacklist check**: Cross-reference with `/home/paper/blacklist.txt` — mark as "BLOCKED" if matched

Many papers evaluate on **multiple datasets** (e.g., GLUE benchmark = 8 tasks). List every one individually.

### C. Training Configuration
For each distinct training setup (there may be different configs for different models/datasets):
- Optimizer type and non-default parameters
- Learning rate, schedule, warmup
- Batch size (per-GPU and effective if specified)
- Epochs / iterations / steps
- Early stopping criteria (if any)
- Gradient clipping (if any)
- Mixed precision / AMP settings (if mentioned)

If a parameter is shared across all experiments, state it once globally. If it varies, note it per-experiment.

### D. Reproducibility Settings
1. **Random Seeds**: number of runs, specific seed values (or "not specified")
2. **Result Aggregation**: mean ± std, median, best-of-N, etc.
3. **Deterministic Mode**: any mentions of `torch.backends.cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`, etc.
4. **Hardware**: GPU type, count, distributed training setup

### E. Evaluation Protocol
For EACH metric:
1. **Name**: e.g., accuracy, F1, BLEU, ROUGE-L, perplexity
2. **Computation details**: micro/macro averaging, tokenization for BLEU, case-sensitive?, etc.
3. **Evaluation set**: which split is used for reporting (val or test?)
4. **Expected target values**: copy the exact numbers from paper tables (these become validation targets)

### F. Expected Outputs
What the code should produce to be considered a successful reproduction:
1. **Result tables**: which metrics for which datasets — copy the paper's table structure
2. **Result figures**: what the axes are, what data they plot
3. **Output file formats**: if the paper specifies particular output formats

## Output Format

```markdown
# Experiments Configuration Report

## Experiment Inventory
| ID | Name | Section | Lines | Type | Priority | Datasets |
|----|------|---------|-------|------|----------|----------|
| Table 1 | Main results | 4.1 | 340-355 | Main result | P0 | Dataset-A, Dataset-B |
| Table 3 | Ablation | 4.3 | 400-415 | Ablation | P1 | Dataset-A |
| Figure 4 | Convergence | 4.2 | 370-380 | Analysis | P2 | Dataset-A |
...

## Datasets
| Dataset | Source | Train | Val | Test | Preprocessing | Blocked? |
|---------|--------|-------|-----|------|---------------|----------|
| Dataset-A | [source URL or library] | 50,000 | 5,000 | 10,000 | [preprocessing details] | No |
| ... | ... | ... | ... | ... | ... | ... |

## Global Training Configuration
(Parameters shared across all experiments)

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | Line N |
| LR | 2e-5 | Line N |
| ... | ... | ... |

## Per-Experiment Overrides
(Parameters that differ from the global config)

### Table 1: Main Results
- Datasets: Dataset-A, Dataset-B
- Epochs: [varies per dataset] (Line N)
- Batch size: 32 for all
- ...

### Table 3: Ablation
- ...

## Reproducibility Settings
- **Seeds**: [N runs, specific values or "not specified"]
- **Aggregation**: [mean ± std / ...]
- **Hardware**: [GPU type × count]
- **Deterministic mode**: [yes/no/not mentioned]

## Evaluation Protocol
| Metric | Computation | Eval Set | Notes |
|--------|-------------|----------|-------|
| Accuracy | exact match | test | ... |
| F1 | macro-averaged | test | ... |
| ... | ... | ... | ... |

## Expected Results (Target Values)
Copy the paper's key result table(s) as-is — these are the targets for validation:

### Table 1: [Title]
| Method | Dataset-A [Metric] | Dataset-B [Metric] |
|--------|---------------------|---------------------|
| Proposed | [value] | [value] |
| Baseline A | [value] | [value] |
| ... | ... | ... | ... |

(Source: Table 1, lines X-Y)

## External Resources
| Resource | Type | Source | Blocked? |
|----------|------|--------|----------|
| [pretrained model] | Model | [source] | No |
| [dataset name] | Dataset | [source] | No |
| ... | ... | ... | ... |
```

## Key Standards
1. **Every dataset individually** — If the paper uses a benchmark suite (e.g., multiple datasets or tasks), list each one as a separate dataset row with its own size and preprocessing.
2. **Copy target numbers verbatim** — The "Expected Results" section should mirror the paper's tables exactly. These numbers are what the Experiment Agent will compare against.
3. **Mark unspecified values explicitly** — If the paper does not state a learning rate schedule, write "Not specified" rather than omitting the row. This avoids the Implementation Agent guessing silently.
4. **Appendix is often critical** — Hyperparameter tables, per-dataset configs, and full result tables are frequently only in appendices. Always check.
"""


class ExperimentsSubagent(Subagent):
    """Phase 2 subagent: Extracts experiment configurations with reproducibility details."""

    @property
    def name(self) -> str:
        return "experiments"

    def system_prompt(self) -> str:
        return EXPERIMENTS_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashTool(),
            SubagentCompleteTool(),
        ]


# =============================================================================
# Phase 2: Baseline Subagent (Enhanced)
# =============================================================================

BASELINE_SYSTEM_PROMPT = """You are a Baseline Methods Extractor, focused on identifying baseline methods and their implementation requirements. Your report helps the Implementation Agent decide HOW to implement each baseline (library call? existing repo? from scratch?) and ensures nothing is missed from the paper's comparison tables.

## Context
You will receive the Structure Agent's output as context, which includes a Section Index and an "Agent Task Assignments" block telling you which sections to focus on. Read `/home/paper/blacklist.txt` early — it directly determines Implementation Category for many baselines. Use the task assignments as your primary reading guide (typically Related Work, Experiments, and appendix sections), and also check for additional baseline comparisons in appendix tables.

## What to Extract

### For Each Baseline:
1. **Identification**: Name, abbreviation, original paper
2. **Implementation Category**:
   - **Library Available**: Can use existing library (PyTorch, scikit-learn, etc.)
   - **Repo Available**: Official/unofficial implementation exists
   - **Custom Required**: Must implement from scratch
   - **Blocked**: In blacklist, cannot use
3. **Configuration**: Hyperparameters for fair comparison
4. **Effort Estimate**: Low/Medium/High implementation effort
5. **Appears In**: Which Tables/Figures this baseline appears in (e.g., "Table 2, Table 4, Figure 3")

Every comparison method counts — even seemingly trivial ones like "vanilla fine-tuning" or "random init" occupy rows in result tables and are scored in the rubric. Do not skip any method that appears in any comparison table or figure.

### Model Variants
Many papers evaluate their method (and baselines) across multiple model architectures or sizes (e.g., ViT-B/16, ViT-L/14, ResNet-50, VisionMamba-S). Each model variant is a **separate grading item** — they are NOT optional configurations of the same experiment. If Table 1 shows results for 3 model sizes, that's 3 independently scored rows. Identify all distinct model variants evaluated in the paper and list them explicitly.

## Output Format

```markdown
# Baseline Methods Report

## Summary Table
| Baseline | Full Name | Category | Source | Effort | Appears In | Blocked? |
|----------|-----------|----------|--------|--------|------------|----------|
| ADVI | Auto-Diff VI | Library | PyMC/Stan | Low | Table 1, Table 3 | No |
| NPE | Neural Post. Est. | Repo | github.com/... | Medium | Table 1 | Check |
| CustomNet | Custom Network | Custom | Paper only | High | Table 2 | N/A |
...

## Implementation Action Items

### Use Existing Library (Low Effort)
| Baseline | Library | Install Command | Notes |
|----------|---------|-----------------|-------|
| ADVI | pymc | pip install pymc | - |
...

### Use Existing Repo (Medium Effort)
| Baseline | Repo URL | Notes |
|----------|----------|-------|
| NPE | github.com/... | Check compatibility |
...

### Implement from Scratch (High Effort)
| Baseline | Paper Reference | Key Components | Estimated LOC |
|----------|-----------------|----------------|---------------|
| CustomNet | Smith et al. 2023 | Encoder, Decoder | 200-300 |
...

### Blocked by Blacklist
| Baseline | Blocked Resource | Alternative |
|----------|------------------|-------------|
| MethodX | github.com/... | Implement from paper |
...

## Model Variants Required

List ALL distinct model architectures/sizes evaluated in the paper. Each variant is independently scored — they are not optional configurations.

| Variant | Type | Appears In | Notes |
|---------|------|------------|-------|
| ViT-B/16 | Architecture | Table 1, Table 3 | Main backbone |
| ViT-L/14 | Architecture | Table 1 | Scaling experiment |
| ResNet-50 | Baseline arch | Table 1, Table 2 | Standard comparison |
...

**Key**: If Table 1 has rows for 3 model sizes × 4 methods = 12 rows, that is 12 independently scored items. Missing any variant means zero on those rubric items.

## Detailed Baseline: [Name]

### Reference
- Paper: [Citation]
- Original Code: [URL if exists]

### Description
[Brief description of the method]

### Configuration for Fair Comparison
| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| Hidden dim | 256 | Table 3 | Same as proposed |
...

### Key Differences from Proposed Method
1. [Difference 1]
2. [Difference 2]

### Implementation Notes
- [Any special considerations]
- [Shared components with other baselines]
```

## Key Standards
1. **Blacklist first** — Read `/home/paper/blacklist.txt` before analyzing any baseline. Cross-reference every repo URL you find against it.
2. **Every comparison method counts** — If a method appears in any result table or figure, it needs an entry. Even trivial baselines (e.g., "Random", "No Augmentation", "Default Config") matter for scoring.
3. **Prioritize libraries over repos** — More reliable and easier to integrate.
4. **Note shared components** — If multiple baselines share modules (e.g., same backbone, shared data loaders, common loss functions), highlight this to avoid duplicate implementation.
5. **Appendix baselines** — Papers often have additional comparisons in appendix tables. Check appendix sections even if not explicitly assigned.
6. **Fair comparison configs** — For each baseline, extract the exact hyperparameters used for comparison. If the paper says "we use the default settings from [repo]", note that explicitly.
"""


class BaselineSubagent(Subagent):
    """Phase 2 subagent: Extracts baseline methods with implementation categorization."""

    @property
    def name(self) -> str:
        return "baseline"

    def system_prompt(self) -> str:
        return BASELINE_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            BashTool(),
            SubagentCompleteTool(),
        ]


# =============================================================================
# Phase 3: Synthesis Subagent (NEW)
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a Synthesis Agent, creating an executive summary from all paper analysis outputs.

## Your Mission
Create a concise, navigable summary that helps the main agent quickly understand and locate information. Your output is the FIRST thing the Main Agent reads — it determines what gets implemented, in what order, and what gets skipped.

## Input
You will receive outputs from:
1. Structure Agent - Paper structure with line numbers
2. Algorithm Agent - Algorithms, architecture, initialization
3. Experiments Agent - Experiment configs, seeds, outputs
4. Baseline Agent - Baseline methods and implementation needs

Some agents may have produced incomplete output (TIMEOUT or FAILED status). If so, note the gap explicitly and recommend which detailed file (e.g., `algorithm.md`, `experiments.md`) the Main Agent should read manually for the missing information.

## Output Format

Create a summary with this structure:

```markdown
# Paper Analysis: Executive Summary

## Quick Reference

### Paper Info
- **Title**: [Title]
- **Type**: [algorithm-focused / empirical / theoretical / systems]
- **Core Contribution**: [One sentence]

### Section Navigator
| Section | Gist | Lines | Read For |
|---------|------|-------|----------|
| Abstract | [3-5 word gist] | 15-40 | Overview |
| Methods | [3-5 word gist] | 180-330 | Algorithm impl |
| Experiments | [3-5 word gist] | 330-530 | Experiment config |
| Appendix A | [3-5 word gist] | 560-660 | Hyperparameters |
...

## Key Takeaways

### Algorithms to Implement
| Algorithm | Location (lines) | Complexity | Dependencies |
|-----------|------------------|------------|--------------|
| BaM | 180-220 | Medium | NumPy, JAX |
...

### Architecture Summary
- **Model Type**: [e.g., Encoder-Decoder]
- **Key Layers**: [e.g., 3x Linear + ReLU]
- **Parameters**: [e.g., ~1M]
- **Initialization**: [e.g., Xavier for Linear, He for Conv]

### Experiments to Run
| Experiment | Type | Datasets | Seeds | Key Config | Target Values |
|------------|------|----------|-------|------------|---------------|
| Table 1 | Main | Dataset-A, Dataset-B | 3 | lr=1e-3, bs=64 | [metric values] |
| Table 3 | Ablation | Dataset-A | 1 | varying components | varies |
...

### Baselines Summary
| Status | Count | Examples |
|--------|-------|----------|
| Library Available | 2 | ADVI (PyMC), SGD |
| Need Implementation | 1 | CustomMethod |
| Blocked | 0 | - |

### Reproducibility Checklist
- [ ] Random seeds: [number] runs needed
- [ ] Hardware: [GPU requirements]
- [ ] Expected training time: [estimate]
- [ ] Output files: [list]

## Constraints Summary
- **Excluded**: [What NOT to reproduce]
- **Allowed**: [Libraries/datasets OK to use]
- **Blocked**: [Resources that MUST NOT be used]

## Suggested Implementation Order
1. **[Component]** - [reason] (lines X-Y, detail: algorithm.md)
2. **[Component]** - [reason] (lines X-Y, detail: experiments.md)
...

## Gaps & Warnings (if any)
- [Any reader agent that timed out or failed — what info is missing]
- [Any contradictions between sections]
- [Blocked resources that need alternative approaches]

---
*Detailed files: /home/agent/paper_analysis/{summary,structure,algorithm,experiments,baseline}.md*
```

## Key Standards
1. **Concise but complete** — Keep it as short as possible without losing actionable information. If a detail can only be found in a detailed file, point to the file instead of repeating it.
2. **Target values are essential** — The "Experiments to Run" table MUST include the paper's reported numbers (from the Experiments Agent output). Without these, the Experiment Agent cannot validate results.
3. **Include line numbers** — For fast navigation back to the paper.
4. **Handle incomplete inputs** — If a reader agent timed out or failed, note what is missing in "Gaps & Warnings" and tell the Main Agent which .md file to read manually.
5. **Implementation Order matters** — Base it on dependencies (data loading before training, core algorithm before baselines) and reference the detail files so the Main Agent knows where to look.
"""


class SynthesisSubagent(Subagent):
    """Phase 3 subagent: Creates executive summary with navigation aids."""

    @property
    def name(self) -> str:
        return "synthesis"

    def system_prompt(self) -> str:
        return SYNTHESIS_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            SubagentCompleteTool(),
        ]


# =============================================================================
# Paper Reader Coordinator (Updated for 3 Phases + Two-Level Output)
# =============================================================================

# Section metadata for two-level output
SECTION_METADATA = {
    "structure": {
        "name": "Paper Structure",
        "filename": "structure.md",
        "description": "Section index, abstract, contributions, constraints",
    },
    "algorithm": {
        "name": "Algorithms & Architecture",
        "filename": "algorithm.md",
        "description": "Core algorithms, pseudo-code, model architecture, initialization",
    },
    "experiments": {
        "name": "Experiments",
        "filename": "experiments.md",
        "description": "Dataset configs, training params, seeds, expected outputs",
    },
    "baseline": {
        "name": "Baseline Methods",
        "filename": "baseline.md",
        "description": "Baseline categorization, implementation requirements",
    },
    "synthesis": {
        "name": "Executive Summary",
        "filename": "summary.md",
        "description": "Quick reference with key takeaways and navigation",
    },
}


def synthesize_paper_analysis(outputs: dict[str, SubagentOutput]) -> str:
    """
    Synthesize outputs from all paper reading subagents into a unified report.

    This function is kept for backward compatibility with the coordinator.
    For two-level output, use synthesize_paper_analysis_structured() instead.
    """
    sections = []

    # Header
    sections.append("# Paper Analysis Report\n")
    sections.append("*Generated by AI Scientist Paper Reader*\n")

    # Status summary
    status_lines = []
    for name, output in outputs.items():
        status = "+" if output.status == SubagentStatus.COMPLETED else "x"
        status_lines.append(f"- {name}: [{status}] ({output.num_steps} steps, {output.runtime_seconds:.1f}s)")
    sections.append("\n## Analysis Status\n" + "\n".join(status_lines) + "\n")

    # Executive Summary (from Synthesis Agent) - THIS GOES FIRST
    if "synthesis" in outputs and outputs["synthesis"].status == SubagentStatus.COMPLETED:
        sections.append("\n---\n")
        sections.append(outputs["synthesis"].content)
        sections.append("\n")

    # Detailed sections
    sections.append("\n---\n")
    sections.append("# Detailed Analysis\n")
    sections.append("*Use the Section Navigator above to jump to specific line numbers.*\n")

    # Structure section (Phase 1)
    if "structure" in outputs:
        sections.append("\n---\n\n## Part 1: Paper Structure\n")
        sections.append(outputs["structure"].content)

    # Algorithm section (Phase 2)
    if "algorithm" in outputs:
        sections.append("\n---\n\n## Part 2: Algorithms & Architecture\n")
        sections.append(outputs["algorithm"].content)

    # Experiments section (Phase 2)
    if "experiments" in outputs:
        sections.append("\n---\n\n## Part 3: Experiment Configurations\n")
        sections.append(outputs["experiments"].content)

    # Baseline section (Phase 2)
    if "baseline" in outputs:
        sections.append("\n---\n\n## Part 4: Baseline Methods\n")
        sections.append(outputs["baseline"].content)

    return "\n".join(sections)


def synthesize_paper_analysis_structured(
    outputs: dict[str, SubagentOutput]
) -> PaperAnalysisResult:
    """
    Synthesize outputs into a structured PaperAnalysisResult.

    This enables two-level reading:
    - Level 1: Executive summary for quick reference
    - Level 2: Detailed files for on-demand deep dives

    Args:
        outputs: Dictionary mapping subagent names to their outputs

    Returns:
        PaperAnalysisResult with structured sections and executive summary
    """
    # Build sections dictionary
    sections: dict[str, PaperAnalysisSection] = {}

    # Track failures
    failed_subagents = []
    total_runtime = 0.0

    for name, output in outputs.items():
        total_runtime += output.runtime_seconds

        if output.status != SubagentStatus.COMPLETED:
            failed_subagents.append(name)
            continue

        if name in SECTION_METADATA:
            meta = SECTION_METADATA[name]

            # Build section content with header
            section_content = _build_section_content(name, output.content)

            sections[name] = PaperAnalysisSection(
                name=meta["name"],
                filename=meta["filename"],
                content=section_content,
                description=meta["description"],
            )

    # Build executive summary
    executive_summary = _build_executive_summary(outputs, sections)

    return PaperAnalysisResult(
        executive_summary=executive_summary,
        sections=sections,
        all_success=len(failed_subagents) == 0,
        failed_subagents=failed_subagents,
        total_runtime_seconds=total_runtime,
    )


def _build_section_content(section_name: str, raw_content: str) -> str:
    """Build formatted content for a section file."""
    meta = SECTION_METADATA.get(section_name, {})
    title = meta.get("name", section_name.title())

    header = f"""# {title}

*Generated by AI Scientist Paper Reader*
*This is a detailed analysis file. For quick reference, see summary.md*

---

"""
    return header + raw_content


def _build_executive_summary(
    outputs: dict[str, SubagentOutput],
    sections: dict[str, PaperAnalysisSection],
) -> str:
    """
    Build executive summary from synthesis agent output.

    The executive summary is designed to be self-contained for quick reference,
    while pointing to detailed files for deeper investigation.
    """
    lines = [
        "# Paper Analysis: Executive Summary",
        "",
        "*Generated by AI Scientist Paper Reader*",
        "",
    ]

    # Analysis status
    lines.append("## Analysis Status")
    lines.append("")
    for name, output in outputs.items():
        status = "✓" if output.status == SubagentStatus.COMPLETED else "✗"
        lines.append(f"- **{name}** [{status}]: {output.num_steps} steps, {output.runtime_seconds:.1f}s")
    lines.append("")

    # Main content from synthesis agent
    if "synthesis" in outputs and outputs["synthesis"].status == SubagentStatus.COMPLETED:
        # The synthesis agent output already contains the formatted summary
        # We include it directly (it follows the SYNTHESIS_SYSTEM_PROMPT format)
        lines.append("---")
        lines.append("")
        lines.append(outputs["synthesis"].content)
    else:
        # Fallback: create minimal summary from available sections
        lines.append("---")
        lines.append("")
        lines.append("*Synthesis agent did not complete. Showing section previews.*")
        lines.append("")

        for name, section in sections.items():
            if name != "synthesis":
                # Show first ~10 lines as preview
                preview_lines = section.content.split('\n')[:10]
                preview = '\n'.join(preview_lines)
                lines.append(f"### {section.name} (Preview)")
                lines.append("")
                lines.append(preview)
                lines.append("...")
                lines.append("")

    return "\n".join(lines)


class PaperReaderCoordinator:
    """
    High-level coordinator for the paper reading phase.

    Implements a three-phase reading strategy:
    1. Phase 1: Structure extraction (serial)
    2. Phase 2: Parallel deep reading (Algorithm, Experiments, Baseline)
    3. Phase 3: Synthesis (creates executive summary)

    Output Modes:
    - read_paper(): Returns CoordinatorResult with single merged output (backward compatible)
    - read_paper_structured(): Returns PaperAnalysisResult with two-level output (recommended)

    Usage:
        coordinator = PaperReaderCoordinator(completer_config)

        # Two-level output (recommended):
        result = await coordinator.read_paper_structured(computer, paper_path)
        print(result.executive_summary)  # Quick reference
        for section in result.sections.values():
            save_file(section.filename, section.content)  # Save detailed files

        # Legacy single-file output:
        result = await coordinator.read_paper(computer, paper_path)
        print(result.synthesized_output)
    """

    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,
        structure_config: SubagentConfig | None = None,
        reader_config: SubagentConfig | None = None,
        synthesis_config: SubagentConfig | None = None,
        run_dir: str | None = None,
    ):
        """
        Initialize the paper reader coordinator.

        Args:
            completer_config: LLM configuration
            structure_config: Config for Phase 1 (structure extraction)
            reader_config: Config for Phase 2 (parallel readers)
            synthesis_config: Config for Phase 3 (synthesis)
            run_dir: Directory for subagent logs
        """
        self.completer_config = completer_config
        self.run_dir = run_dir

        self.structure_config = structure_config or DEFAULT_PAPER_STRUCTURE_CONFIG
        self.reader_config = reader_config or DEFAULT_PAPER_READER_CONFIG
        self.synthesis_config = synthesis_config or DEFAULT_PAPER_SYNTHESIS_CONFIG

        self.coordinator = SubagentCoordinator(
            completer_config=completer_config,
            synthesize_fn=synthesize_paper_analysis,
        )

    def _build_subagent_tasks(
        self,
        paper_path: str,
    ) -> list[SubagentTask]:
        """
        Build the list of subagent tasks for paper reading.

        This is extracted as a method to avoid duplication between
        read_paper() and read_paper_structured().
        """
        # Create subagent instances
        structure_agent = StructureSubagent(
            completer_config=self.completer_config,
            config=self.structure_config,
            run_dir=self.run_dir,
        )
        algorithm_agent = AlgorithmSubagent(
            completer_config=self.completer_config,
            config=self.reader_config,
            run_dir=self.run_dir,
        )
        experiments_agent = ExperimentsSubagent(
            completer_config=self.completer_config,
            config=self.reader_config,
            run_dir=self.run_dir,
        )
        baseline_agent = BaselineSubagent(
            completer_config=self.completer_config,
            config=self.reader_config,
            run_dir=self.run_dir,
        )
        synthesis_agent = SynthesisSubagent(
            completer_config=self.completer_config,
            config=self.synthesis_config,
            run_dir=self.run_dir,
        )

        # Define tasks with dependencies
        # Phase 1: Structure (no dependencies)
        # Phase 2: Algorithm, Experiments, Baseline (depend on Structure, run in parallel)
        # Phase 3: Synthesis (depends on all Phase 2 agents)
        return [
            # Phase 1
            SubagentTask(
                subagent=structure_agent,
                task_description=f"""Analyze the paper structure at {paper_path}.

Extract with PRECISE LINE NUMBERS:
1. Complete section index with Start Line and Line Count for each section
2. A 3-8 word "Gist" summarizing each section's reproduction value
3. Full abstract text
4. Core contributions with section references
5. Constraints from /home/paper/addendum.md and /home/paper/blacklist.txt
6. Agent Task Assignments for Phase 2 readers""",
                dependencies=[],
                context_keys=[],
            ),

            # Phase 2 (parallel)
            SubagentTask(
                subagent=algorithm_agent,
                task_description=f"""Extract algorithms, architecture, and implementation details from {paper_path}.

Focus on:
1. Core algorithms with pseudo-code and hyperparameters
2. Model architecture (layers, dimensions, activations)
3. Initialization strategies (weight init methods)
4. Pretrained weights (sources, formats, check blacklist!)
5. Numerical stability considerations

Include line numbers for all components.""",
                dependencies=["structure"],
                context_keys=["structure"],
            ),
            SubagentTask(
                subagent=experiments_agent,
                task_description=f"""Extract experiment configurations from {paper_path}.

Focus on:
1. All Figure/Table configurations with line numbers
2. Dataset details (name, source, preprocessing)
3. Training configuration (optimizer, lr, batch size, epochs)
4. RANDOM SEEDS - number of runs, specific seed values
5. Expected output files (names, formats, content structure)
6. Evaluation metrics and protocols

Include line numbers for all configurations.""",
                dependencies=["structure"],
                context_keys=["structure"],
            ),
            SubagentTask(
                subagent=baseline_agent,
                task_description=f"""Extract baseline method information from {paper_path}.

Categorize each baseline as:
1. Library Available (can use pip install)
2. Repo Available (existing implementation)
3. Custom Required (implement from paper)
4. Blocked (in blacklist, cannot use)

Check /home/paper/blacklist.txt before suggesting any external resource.
Include effort estimates and shared components.""",
                dependencies=["structure"],
                context_keys=["structure"],
            ),

            # Phase 3 (depends on all Phase 2)
            SubagentTask(
                subagent=synthesis_agent,
                task_description="""Create an executive summary from all the analysis outputs.

Your summary should include:
1. Quick Reference section with paper info
2. Section Navigator table with line numbers and gists
3. Key Takeaways (algorithms, architecture, experiments, baselines)
4. Reproducibility checklist
5. Constraints summary
6. Suggested implementation order

Keep it concise but include all critical information for quick reference.""",
                dependencies=["structure", "algorithm", "experiments", "baseline"],
                context_keys=["structure", "algorithm", "experiments", "baseline"],
            ),
        ]

    async def read_paper(
        self,
        computer: ComputerInterface,
        paper_path: str = "/home/paper/paper.md",
        constraints: dict | None = None,
    ) -> CoordinatorResult:
        """
        Execute the full paper reading workflow (3 phases).

        This is the legacy method that returns a single merged output.
        For two-level output, use read_paper_structured() instead.

        Args:
            computer: ComputerInterface for file access
            paper_path: Path to the paper file
            constraints: Optional blacklist constraints

        Returns:
            CoordinatorResult with all subagent outputs and synthesized analysis
        """
        tasks = self._build_subagent_tasks(paper_path)
        return await self.coordinator.run(computer, tasks, constraints)

    async def read_paper_structured(
        self,
        computer: ComputerInterface,
        paper_path: str = "/home/paper/paper.md",
        constraints: dict | None = None,
    ) -> PaperAnalysisResult:
        """
        Execute paper reading with two-level structured output.

        This is the recommended method for paper reading. It returns:
        - Level 1: Executive summary for quick reference
        - Level 2: Detailed sections for on-demand access

        Args:
            computer: ComputerInterface for file access
            paper_path: Path to the paper file
            constraints: Optional blacklist constraints

        Returns:
            PaperAnalysisResult with structured sections and executive summary
        """
        tasks = self._build_subagent_tasks(paper_path)

        # Run coordinated execution
        coordinator_result = await self.coordinator.run(computer, tasks, constraints)

        # Convert to structured result
        return synthesize_paper_analysis_structured(coordinator_result.outputs)
