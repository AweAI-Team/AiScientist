from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_domain_paper.configs import DEFAULT_PRIORITIZATION_CONFIG
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_prioritization_tools


class PaperPrioritizationSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "prioritization"

    def system_prompt(self) -> str:
        return self.engine.render_subagent_prompt("prioritization")

    def get_tools(self):
        return build_prioritization_tools(self.capabilities)


@dataclass(frozen=True)
class PrioritizationResult:
    prioritized_path: Path
    plan_path: Path | None
    summary: str
    output: SubagentOutput
    task_description: str


class PrioritizationRunner:
    def __init__(self, engine) -> None:
        self.engine = engine

    def run(
        self,
        *,
        paper_analysis_dir: str = "/home/agent/paper_analysis",
        rubric_path: str = "/home/paper/rubric.json",
        focus_areas: str | None = None,
    ) -> PrioritizationResult:
        summary_path = f"{paper_analysis_dir.rstrip('/')}/summary.md"
        if not self.engine.shell.file_exists(summary_path):
            raise FileNotFoundError(
                f"Paper analysis not found at {paper_analysis_dir}/. Run read_paper first to generate the analysis files."
            )

        summary = self.engine.shell.read_file(summary_path)
        rubric = self._safe_read(rubric_path, limit=12_000)
        addendum = self._safe_read("/home/paper/addendum.md", limit=8_000)
        blacklist = self._safe_read("/home/paper/blacklist.txt", limit=4_000)
        task_description = self._build_task_description(
            summary,
            paper_analysis_dir=paper_analysis_dir,
            rubric_path=rubric_path,
            rubric=rubric,
            addendum=addendum,
            blacklist=blacklist,
            focus_areas=focus_areas,
        )
        config = self.engine.subagent_config("prioritization", DEFAULT_PRIORITIZATION_CONFIG)
        output = self.engine.run_subagent_output(
            PaperPrioritizationSubagent,
            objective="Create a prioritized implementation plan for reproducing the paper.",
            context=task_description,
            config=config,
            phase="prioritize",
            label="prioritization",
        )

        prioritized_path = self.engine.prioritized_path
        plan_path = self.engine.plan_path if self.engine.plan_path.exists() else None
        if not prioritized_path.exists():
            output.status = SubagentStatus.FAILED
            output.error_message = (
                "Prioritization subagent finished without writing /home/agent/prioritized_tasks.md. "
                "Use write_priorities to save the plan."
            )

        return PrioritizationResult(
            prioritized_path=prioritized_path,
            plan_path=plan_path,
            summary=self._build_agent_summary(output, prioritized_path, plan_path),
            output=output,
            task_description=task_description,
        )

    def _safe_read(self, canonical_path: str, *, limit: int) -> str:
        if not self.engine.shell.file_exists(canonical_path):
            return ""
        text = self.engine.shell.read_file(canonical_path)
        return text[:limit]

    def _build_task_description(
        self,
        summary: str,
        *,
        paper_analysis_dir: str,
        rubric_path: str,
        rubric: str,
        addendum: str,
        blacklist: str,
        focus_areas: str | None,
    ) -> str:
        task_description = f"""Analyze the paper and rubric to create a prioritized implementation plan.

## Paper Analysis Context

The paper has been analyzed by specialized subagents. The analysis is saved in `{paper_analysis_dir}/`:
- **summary.md** - Executive summary (included below)
- **structure.md** - Paper structure, section index, abstract, constraints
- **algorithm.md** - Core algorithms, pseudo-code, architecture, hyperparameters
- **experiments.md** - Experiment configurations, datasets, training settings, expected outputs
- **baseline.md** - Baseline methods categorized by implementation effort

Here is the executive summary:

---
{summary.strip() or "(paper summary missing)"}
---

## Your Task

1. **Parse the rubric** using `parse_rubric` to analyze `{rubric_path}`
   - If the rubric is missing, infer priorities from the paper structure and contributions.
2. **Review detailed analysis** — use `read_file_chunk` or `search_file` to inspect files in `{paper_analysis_dir}/` when you need more details.
3. **Cross-reference** rubric items (if available) with paper sections and baselines.
4. **Assign priorities** (P0/P1/P2/P3) using evidence:
   - Rubric weights and structure
   - Main-text vs appendix placement
   - Dependencies between tasks
   - Baselines and model variants in the main tables
5. **Identify dependencies** and the best execution order.
6. **Write output** using `write_priorities`.

## Other Files to Check
- `/home/paper/addendum.md` - Scope clarifications and constraints
- `/home/paper/blacklist.txt` - Blocked resources

## Key Considerations

- P0 tasks should represent the core contribution.
- Baselines that appear in main-text tables should be P0 unless a stronger dependency argument exists.
- Treat each model variant as a separate task.
- Account for time constraints and recommend time allocation.
- Flag any risks, blockers, or unclear requirements.

## Rubric Hints
{rubric.strip() or "No rubric.json staged."}

## Addendum Constraints
{addendum.strip() or "No addendum.md staged."}

## Blacklist Constraints
{blacklist.strip() or "No blacklist.txt staged."}

## Required Workflow

1. Use `parse_rubric` when the rubric exists.
2. Read `baseline.md` to verify each baseline and model variant is covered.
3. Use `write_priorities` to write `/home/agent/prioritized_tasks.md`.
4. Return concise findings with `subagent_complete` when done.

## Output Requirements

The prioritized_tasks.md file should contain:
- Executive summary
- P0/P1/P2/P3 breakdown
- Task-specific justification and dependencies
- Explicit baseline and model-variant coverage
- Dependency graph
- Risk assessment
- Recommended execution order
- Time allocation guidance
""".strip()
        if focus_areas:
            task_description += f"\n\n## Focus Areas\nPay special attention to: {focus_areas}"
        return task_description

    def _build_agent_summary(self, output: SubagentOutput, prioritized_path: Path, plan_path: Path | None) -> str:
        status_icon = "✓" if output.status == SubagentStatus.COMPLETED else "✗"
        lines = [
            f"[Prioritization {status_icon}] ({output.num_steps} steps, {output.runtime_seconds:.1f}s)",
            "",
        ]
        if output.status == SubagentStatus.COMPLETED:
            lines.append(f"**Prioritized tasks saved to**: `{prioritized_path}`")
            lines.extend(
                [
                    "",
                    "## Summary",
                    output.content.strip() or "(no subagent output)",
                    "",
                    "---",
                    "",
                    "**Next Steps**:",
                    "1. Review the prioritized tasks in `/home/agent/prioritized_tasks.md`.",
                    "2. Start with P0-Critical tasks.",
                    "3. Use `spawn_subagent(subagent_type='plan')` if you need a separate planning pass for a specific task.",
                ]
            )
        elif output.status == SubagentStatus.FAILED:
            lines.extend(
                [
                    f"Failed: {output.error_message or 'Prioritization failed.'}",
                    "",
                    "Partial output:",
                    output.content.strip() or "(no subagent output)",
                ]
            )
        elif output.status == SubagentStatus.TIMEOUT:
            lines.extend(
                [
                    "Timed out. Partial output:",
                    output.content.strip() or "(no subagent output)",
                ]
            )
        else:
            lines.append(f"Status: {output.status.value}")
        return "\n".join(lines).strip()


__all__ = ["PaperPrioritizationSubagent", "PrioritizationResult", "PrioritizationRunner"]
