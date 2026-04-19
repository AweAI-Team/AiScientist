"""
Prioritization Subagent

This module implements the prioritization phase that bridges paper reading and implementation.
The PrioritizationSubagent analyzes:
1. Paper content and structure (from /home/agent/paper_analysis/ directory)
2. Available rubric information (partial or full)
3. Scope constraints (from addendum.md)

And produces a prioritized task list that guides the implementation phase.

Design Philosophy:
- Evidence-based inference: Priorities are derived from concrete signals, not guesses
- Transparent reasoning: Each priority assignment includes justification
- Actionable output: Tasks are specific enough to be directly implemented

Key Insight from Data Analysis:
- High-priority tasks are ~49% Code Development, ~46% Result Analysis
- Common patterns: core algorithm implementation, main experiments, baseline comparisons
- Appendix-only experiments are explicitly out of scope per instructions
"""

from __future__ import annotations

import json
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
)
from paperbench.solvers.basicagent.tools import ReadFileChunk, SearchFile
from paperbench.solvers.basicagent.tools.base import Tool

# =============================================================================
# Priority Levels
# =============================================================================

class Priority:
    """Priority levels with clear semantics."""
    P0 = "P0-Critical"      # Must complete - core algorithm, main results
    P1 = "P1-Important"     # Should complete - key experiments, baselines
    P2 = "P2-Valuable"      # Complete if time permits - ablations, variants
    P3 = "P3-Optional"      # Low priority - edge cases, supplementary


# =============================================================================
# System Prompt
# =============================================================================

PRIORITIZATION_SYSTEM_PROMPT = """You are a Prioritization Strategist for reproducing academic papers.

## Your Mission
Analyze the paper and available rubric to create a prioritized implementation plan. Your goal is to help the agent maximize their score within limited time by focusing on what matters most.

## Available Inputs

Your task description will include the paper analysis summary with navigation to detailed files.
Use `read_file_chunk` to access specific files when you need more details.

### Key Files
- `/home/paper/rubric.json` - Evaluation rubric (may be partial - only top-level tasks visible)
- `/home/paper/addendum.md` - Scope clarifications and constraints
- `/home/paper/blacklist.txt` - Blocked resources
- `/home/paper/paper.md` - Original paper for cross-reference
- `/home/agent/paper_analysis/` - Detailed analysis files (summary.md, structure.md, algorithm.md, experiments.md, baseline.md)

## Priority Framework

### P0-Critical (Must Complete)
Characteristics:
- Core algorithm that defines the paper's main contribution
- Main experiments shown in the paper's key figures/tables
- High weight in rubric (if visible)
- Explicitly required in rubric top-level tasks
- **Baselines that appear in main-text tables** (Table 1, Table 2, etc.) — these are scored just like core results. A missing baseline row is a zero on that rubric item.
- **Each model variant is a separate P0 task** — if the paper evaluates ViT-Base, ResNet-50, and VisionMamba in Table 1, these are THREE independent P0 tasks, not one "implement models" task. Each variant occupies its own rows in result tables and is graded independently.

### P1-Important (Should Complete)
Characteristics:
- Baselines that appear ONLY in appendix tables (not in any main-text table)
- Secondary experiments that validate the main claims
- Components with medium weight in rubric
- Required for completeness of main results

### P2-Valuable (If Time Permits)
Characteristics:
- Ablation studies
- Sensitivity analyses
- Additional datasets or configurations
- Lower weight rubric items

### P3-Optional (Low Priority)
Characteristics:
- Appendix-only experiments (explicitly out of scope per instructions)
- Edge cases mentioned briefly
- "Nice to have" features
- Not in visible rubric

Note: A lower-priority task that **blocks** higher-priority tasks should be elevated accordingly.

## Analysis Process

### Step 1: Parse Rubric Structure
Read `/home/paper/rubric.json` using `parse_rubric` and extract:
- Top-level task weights (higher weight = higher priority)
- Task categories (Code Development vs Result Analysis)
- Any visible sub-task hints
- Note: Some tasks may not have a rubric file. If not found, infer priorities from paper structure and contributions.

### Step 2: Cross-Reference with Paper
For each rubric item:
- Locate in paper (which section, figure, table?)
- Assess complexity (simple formula vs complex system?)
- Identify dependencies (what must be done first?)

### Step 3: Apply Priority Rules
Use these evidence-based rules:

**Elevate to P0 if:**
- Rubric weight is significantly above average
- Task is core algorithm implementation
- Task mentions "core" or "main" contribution
- Required for other high-weight tasks
- Task is a baseline or model variant that appears in a main-text table (Table 1, 2, 3...) — baselines in main tables are graded with equal weight to the proposed method's results

**Keep at P1 if:**
- Rubric weight is around or above average
- Task is a baseline that appears ONLY in appendix tables (not in any main-text table)
- Task is for Figure/Table in main text but is not a comparison method
- Referenced multiple times in rubric

**Demote to P2/P3 if:**
- Appendix-only content
- Mentioned as "optional" or "extension"
- Very low rubric weight
- Blocked by blacklist constraints

### Step 4: Identify Dependencies
Build a dependency graph:
- Core algorithm → Experiments that use it
- Data loading → Training → Evaluation
- Shared utilities → Multiple components

### Step 5: Estimate Effort
For each task, estimate:
- Complexity: Low / Medium / High
- Risk: Implementation difficulty, unclear specs, potential blockers

## Output Format

Write to `/home/agent/prioritized_tasks.md`:

```markdown
# Prioritized Implementation Plan

## Executive Summary
- **Total Tasks**: N
- **P0 Tasks**: X (estimated Y% of score)
- **Time Budget Recommendation**: [how to allocate time]

## Priority Breakdown

### P0-Critical [Must Complete]

#### Task 1: [Descriptive Name]
- **Rubric Reference**: [ID or description from rubric]
- **Paper Reference**: Section X, Algorithm Y, Figure Z
- **Why P0**: [Evidence-based justification]
- **Dependencies**: [What must be done first]
- **Deliverables**:
  - [ ] Implementation of X
  - [ ] Output file: Y
- **Complexity**: [Low/Medium/High]
- **Estimated Effort**: [Rough guidance]

#### Task 2: ...

### P1-Important [Should Complete]
...

### P2-Valuable [If Time Permits]
...

### P3-Optional [Low Priority]
...

## Dependency Graph
```
[Core Algorithm]
    ├── [Experiment 1]
    ├── [Experiment 2]
    └── [Baseline Comparison]
```

## Risk Assessment
| Task | Risk | Mitigation |
|------|------|------------|
| ... | ... | ... |

## Recommended Execution Order
1. [First task - no dependencies]
2. [Second task - depends on 1]
...

## Time Allocation Strategy
- **Phase 1 (40% of time)**: P0 tasks
- **Phase 2 (35% of time)**: P1 tasks
- **Phase 3 (20% of time)**: P2 tasks
- **Buffer (5% of time)**: Debugging, unexpected issues
```

## Key Standards

1. **Be Specific**: Don't say "implement the algorithm", say "implement BaM batch step per Eq. (6-7)"

2. **Cite Evidence**: Every priority assignment should reference rubric weight, paper section, or explicit instruction

3. **Consider Dependencies**: A P1 task that blocks P0 tasks should be treated as P0

4. **Account for Constraints**: Check blacklist.txt and addendum.md for blocked resources

5. **Think About Grading**: The judge will check:
   - Code correctness (implementation matches paper)
   - Execution success (reproduce.sh runs)
   - Result matching (outputs match paper's claims)

6. **Partial Credit Matters**: Even incomplete implementations get partial credit, so prioritize having something working for each major component over perfecting one component

7. **Model Variants Are Separate Tasks**: If the paper evaluates multiple model sizes or architectures (e.g., ViT-B/16, ViT-L/14, ResNet-50), each one is a separate grading item. Do NOT group them into a single "implement all models" task. Create one task per variant, because each variant occupies distinct rows in result tables and is independently scored.

8. **Baselines in Main Tables Are P0**: A common scoring pitfall is treating baselines as low priority. In reality, baselines appearing in the paper's main-text tables (Table 1, 2, 3...) are scored with equal weight to the proposed method's results. Missing a baseline row means zero on that rubric item. Five methods each implemented at 60% scores far better than one method implemented at 100%.

9. **Cross-Check with baseline.md**: Read `/home/agent/paper_analysis/baseline.md` to identify all baselines and their model variants. Ensure every method listed there has a corresponding task in your plan, and every method that appears in a main-text table is assigned P0.

## When Done
Use the `write_priorities` tool to save your analysis, then call `subagent_complete` with a brief summary.
"""


# =============================================================================
# Specialized Tools
# =============================================================================

class PriorityWriteTool(Tool):
    """Tool for writing the prioritized task list."""

    def name(self) -> str:
        return "write_priorities"

    async def execute(
        self,
        computer: ComputerInterface,
        content: str,
    ) -> str:
        """Write content to /home/agent/prioritized_tasks.md."""
        output_path = "/home/agent/prioritized_tasks.md"

        # Ensure directory exists
        await computer.send_shell_command("mkdir -p /home/agent")

        # Write the prioritized tasks
        await computer.upload(content.encode("utf-8"), output_path)

        return f"Prioritized tasks written to {output_path}"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Save your prioritized task analysis to /home/agent/prioritized_tasks.md",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete prioritized task list in markdown format",
                    },
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            strict=False,
        )


class ParseRubricTool(Tool):
    """Tool for parsing and analyzing rubric.json structure."""

    def name(self) -> str:
        return "parse_rubric"

    async def execute(
        self,
        computer: ComputerInterface,
        rubric_path: str = "/home/paper/rubric.json",
        max_depth: int = 3,
    ) -> str:
        """
        Parse rubric.json and extract structured information.

        Args:
            rubric_path: Path to rubric.json
            max_depth: How deep to traverse (default 3 for root + 2 levels of children).
                       PaperBench rubrics typically have 3-5 levels of nesting.

        Returns:
            Formatted analysis of rubric structure with weights and categories
        """
        try:
            # Read the rubric file
            content = await computer.download(rubric_path)
            rubric = json.loads(content.decode("utf-8", errors="replace"))
        except Exception as e:
            return f"Error reading rubric: {str(e)}. The rubric may not be available."

        def analyze_node(node: dict, depth: int = 0, path: str = "") -> list[dict]:
            """Recursively analyze rubric nodes."""
            results = []

            # Handle different rubric formats
            req = node.get("requirements", node.get("name", ""))
            weight = node.get("weight", 1)
            task_cat = node.get("task_category", node.get("finegrained_task_category"))
            node_id = node.get("id", "")

            children = node.get("sub_tasks", node.get("children", []))

            results.append({
                "id": node_id,
                "depth": depth,
                "path": path,
                "requirement": req[:200] if req else "",
                "weight": weight,
                "category": task_cat,
                "num_children": len(children),
                "is_leaf": len(children) == 0,
            })

            # Recurse into children up to max_depth
            if depth < max_depth:
                for i, child in enumerate(children):
                    child_path = f"{path}/{i}" if path else str(i)
                    results.extend(analyze_node(child, depth + 1, child_path))

            return results

        # First, compute full tree statistics (unlimited depth) for metadata
        def compute_tree_stats(node: dict, depth: int = 0) -> dict:
            """Compute full tree statistics regardless of max_depth."""
            children = node.get("sub_tasks", node.get("children", []))
            if not children:
                return {"max_depth": depth, "total_nodes": 1, "leaf_nodes": 1, "per_level": {depth: 1}}
            stats = {"max_depth": depth, "total_nodes": 1, "leaf_nodes": 0, "per_level": {depth: 1}}
            for child in children:
                child_stats = compute_tree_stats(child, depth + 1)
                stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
                stats["total_nodes"] += child_stats["total_nodes"]
                stats["leaf_nodes"] += child_stats["leaf_nodes"]
                for lvl, cnt in child_stats["per_level"].items():
                    stats["per_level"][lvl] = stats["per_level"].get(lvl, 0) + cnt
            return stats

        tree_stats = compute_tree_stats(rubric)

        nodes = analyze_node(rubric)

        # Calculate statistics
        total_nodes = len(nodes)
        weights = [n["weight"] for n in nodes if n["weight"]]
        avg_weight = sum(weights) / len(weights) if weights else 1
        max_weight = max(weights) if weights else 1

        # Format output
        lines = [
            "# Rubric Analysis",
            "",
            f"**Total visible nodes** (depth ≤ {max_depth}): {total_nodes}",
            f"**Average weight**: {avg_weight:.2f}",
            f"**Max weight**: {max_weight}",
            "",
            "## Top-Level Tasks (by weight)",
            "",
        ]

        # Sort by weight and show top items
        sorted_nodes = sorted(nodes, key=lambda x: (-x["weight"], x["depth"]))

        for node in sorted_nodes[:15]:  # Show top 15
            indent = "  " * node["depth"]
            weight_indicator = "🔴" if node["weight"] >= 2 * avg_weight else "🟡" if node["weight"] >= avg_weight else "⚪"
            cat_str = f"[{node['category']}]" if node["category"] else ""
            children_str = f"({node['num_children']} sub-tasks)" if node["num_children"] > 0 else "(leaf)"

            lines.append(f"{indent}{weight_indicator} **W={node['weight']}** {cat_str} {children_str}")
            lines.append(f"{indent}   {node['requirement'][:150]}...")
            lines.append("")

        # Category distribution
        categories = {}
        for node in nodes:
            cat = node["category"] or "Unspecified"
            categories[cat] = categories.get(cat, 0) + 1

        lines.append("## Category Distribution")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {count}")

        lines.append("")
        lines.append("## Weight Thresholds for Priority")
        lines.append(f"- P0 threshold (≥2x avg): weight ≥ {2 * avg_weight:.1f}")
        lines.append(f"- P1 threshold (≥avg): weight ≥ {avg_weight:.1f}")
        lines.append(f"- P2/P3: weight < {avg_weight:.1f}")

        # Tree statistics (always computed over the full tree)
        lines.append("")
        lines.append("## Tree Statistics")
        lines.append(f"- **Max depth**: {tree_stats['max_depth']}")
        lines.append(f"- **Total nodes (all depths)**: {tree_stats['total_nodes']}")
        lines.append(f"- **Leaf nodes**: {tree_stats['leaf_nodes']}")
        per_level_str = ", ".join(
            f"L{lvl}={cnt}" for lvl, cnt in sorted(tree_stats["per_level"].items())
        )
        lines.append(f"- **Nodes per level**: [{per_level_str}]")
        lines.append(f"- **Shown in this output** (depth ≤ {max_depth}): {total_nodes} / {tree_stats['total_nodes']} nodes")
        if max_depth < tree_stats["max_depth"]:
            lines.append("")
            lines.append(
                f"To see deeper levels, call `parse_rubric(max_depth={tree_stats['max_depth']})`. "
                f"This will reveal {tree_stats['total_nodes'] - total_nodes} additional nodes."
            )

        return "\n".join(lines)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="""Parse and analyze the rubric.json file to understand task weights and structure.

Returns:
- Total number of visible tasks
- Weight statistics (average, max)
- Top tasks sorted by weight with priority indicators
- Category distribution
- Recommended priority thresholds""",
            parameters={
                "type": "object",
                "properties": {
                    "rubric_path": {
                        "type": "string",
                        "description": "Path to rubric.json",
                        "default": "/home/paper/rubric.json",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse (default 3). PaperBench rubrics typically have 3-5 levels. Use higher values to see more detail, lower values for a quick overview.",
                        "default": 3,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
        )


# =============================================================================
# Prioritization Subagent
# =============================================================================

class PrioritizationSubagent(Subagent):
    """
    Subagent that analyzes paper and rubric to create prioritized task list.

    This subagent bridges the paper reading phase and implementation phase,
    ensuring the agent focuses on high-value tasks first.

    Inputs (via context or files):
    - /home/agent/paper_analysis/: Directory with detailed analysis files
      - summary.md, structure.md, algorithm.md, experiments.md, baseline.md
    - rubric.json: Evaluation rubric (may be partial)
    - addendum.md: Scope constraints

    Outputs:
    - prioritized_tasks.md: Structured, prioritized TODO list
    - artifacts: Parsed task structure for programmatic use
    """

    @property
    def name(self) -> str:
        return "prioritization"

    def system_prompt(self) -> str:
        return PRIORITIZATION_SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return [
            ReadFileChunk(),
            SearchFile(),
            ParseRubricTool(),      # Specialized rubric parsing
            PriorityWriteTool(),    # Write prioritized tasks
            SubagentCompleteTool(),
        ]

    def _post_process_output(
        self, raw_output: str, artifacts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Post-process to ensure artifacts contain structured priority info.
        """
        # Add metadata about the prioritization
        artifacts["prioritization_complete"] = True
        artifacts["output_file"] = "/home/agent/prioritized_tasks.md"

        return raw_output, artifacts


# =============================================================================
# Default Task Description Template
# =============================================================================

# Note: The actual task description is built dynamically in PrioritizeTasksTool.execute()
# to include the paper analysis summary as context. This constant is kept for reference.
DEFAULT_PRIORITIZATION_TASK = """Analyze the paper and rubric to create a prioritized implementation plan.

The task description will include:
1. Paper analysis summary with navigation to detailed files
2. Instructions to parse rubric and assign priorities
3. Guidance on using write_priorities tool

See PrioritizeTasksTool.execute() for the complete dynamically-built task.
"""
