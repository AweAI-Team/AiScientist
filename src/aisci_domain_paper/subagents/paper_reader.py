from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from aisci_agent_runtime.llm_client import create_llm_client
from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_domain_paper.configs import (
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
)
from aisci_domain_paper.prompts.templates import (
    ALGORITHM_SYSTEM_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    EXPERIMENTS_SYSTEM_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.subagents.coordinator import SubagentCoordinator, SubagentTask
from aisci_domain_paper.tools import build_reader_tools


class StructureSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_structure"

    def system_prompt(self) -> str:
        return STRUCTURE_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class AlgorithmSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_algorithm"

    def system_prompt(self) -> str:
        return ALGORITHM_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class ExperimentsSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_experiments"

    def system_prompt(self) -> str:
        return EXPERIMENTS_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class BaselineSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_baseline"

    def system_prompt(self) -> str:
        return BASELINE_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class SynthesisSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_synthesis"

    def system_prompt(self) -> str:
        return SYNTHESIS_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


@dataclass(frozen=True)
class PaperAnalysisSection:
    name: str
    filename: str
    content: str
    description: str
    line_count: int = 0

    def __post_init__(self) -> None:
        if self.line_count == 0:
            object.__setattr__(self, "line_count", len(self.content.splitlines()) or 1)


@dataclass
class PaperAnalysisResult:
    executive_summary: str
    sections: dict[str, PaperAnalysisSection] = field(default_factory=dict)
    outputs: dict[str, SubagentOutput] = field(default_factory=dict)
    all_success: bool = True
    failed_subagents: list[str] = field(default_factory=list)
    total_runtime_seconds: float = 0.0

    @property
    def summary_with_navigation(self) -> str:
        nav_table = self._build_navigation_table()
        return "\n\n".join(
            [
                self.executive_summary.strip(),
                "---",
                "## Detailed Analysis Files",
                "",
                nav_table,
                "",
                "**How to Access Details:**",
                "- Use `read_file_chunk(file=\"/home/agent/paper_analysis/<section>.md\")` to read specific sections.",
                "- Use `search_file(file=\"/home/agent/paper_analysis/<section>.md\", query=\"...\")` to search within a section.",
                "- The executive summary above is the quick reference; the detailed files contain the full phase outputs.",
            ]
        ).strip()

    def _build_navigation_table(self) -> str:
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


class PaperReaderCoordinator:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.structure_config = engine.subagent_config("paper_reader", DEFAULT_PAPER_STRUCTURE_CONFIG)
        self.reader_config = engine.subagent_config("paper_reader", DEFAULT_PAPER_READER_CONFIG)
        self.synthesis_config = engine.subagent_config("paper_reader", DEFAULT_PAPER_SYNTHESIS_CONFIG)
        self.coordinator = SubagentCoordinator()

    def run(self) -> PaperAnalysisResult:
        return self.read_paper_structured()

    def read_paper_structured(self) -> PaperAnalysisResult:
        self.engine.analysis_dir.mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        coordinator_result = self.coordinator.run(self._build_subagent_tasks())
        structure_out = coordinator_result.outputs["structure"]
        algorithm_out = coordinator_result.outputs["algorithm"]
        experiments_out = coordinator_result.outputs["experiments"]
        baseline_out = coordinator_result.outputs["baseline"]
        synthesis_out = coordinator_result.outputs["synthesis"]

        sections = {
            "structure": self._section("Structure", "structure.md", structure_out.content, "Paper structure, metadata, and constraints."),
            "algorithm": self._section("Algorithm", "algorithm.md", algorithm_out.content, "Implementation-facing algorithm and architecture details."),
            "experiments": self._section("Experiments", "experiments.md", experiments_out.content, "Datasets, metrics, tables, and evaluation procedure."),
            "baseline": self._section("Baseline", "baseline.md", baseline_out.content, "Baselines, comparison methods, and model variants."),
            "summary": self._section("Executive Summary", "summary.md", synthesis_out.content, "Quick reference for the main agent."),
        }

        result = PaperAnalysisResult(
            executive_summary=synthesis_out.content.strip() or self._fallback_summary(sections),
            sections=sections,
            outputs=coordinator_result.outputs,
            all_success=coordinator_result.all_success,
            failed_subagents=coordinator_result.failed_subagents,
            total_runtime_seconds=coordinator_result.total_runtime_seconds or (perf_counter() - start),
        )

        self._write_outputs(result)
        return result

    def _build_subagent_tasks(self) -> list[SubagentTask]:
        return [
            SubagentTask(
                name="structure",
                run_fn=lambda _ctx: self._run_stage(
                    StructureSubagent,
                    objective="Extract the paper structure, metadata, and constraints.",
                    context=self.engine.reader_context(),
                    config=self.structure_config,
                    phase="analyze",
                    label="paper_structure",
                    llm=self._clone_llm(),
                ),
            ),
            SubagentTask(
                name="algorithm",
                dependencies=["structure"],
                context_keys=["structure"],
                run_fn=lambda ctx: self._run_stage(
                    AlgorithmSubagent,
                    objective="Extract the implementation-facing algorithm and architecture details.",
                    context=self._stage_context(ctx.get("structure", SubagentOutput(SubagentStatus.FAILED, "")).content),
                    config=self.reader_config,
                    phase="analyze",
                    label="paper_algorithm",
                    llm=self._clone_llm(),
                ),
            ),
            SubagentTask(
                name="experiments",
                dependencies=["structure"],
                context_keys=["structure"],
                run_fn=lambda ctx: self._run_stage(
                    ExperimentsSubagent,
                    objective="Extract the paper's datasets, metrics, tables, and evaluation procedure.",
                    context=self._stage_context(ctx.get("structure", SubagentOutput(SubagentStatus.FAILED, "")).content),
                    config=self.reader_config,
                    phase="analyze",
                    label="paper_experiments",
                    llm=self._clone_llm(),
                ),
            ),
            SubagentTask(
                name="baseline",
                dependencies=["structure"],
                context_keys=["structure"],
                run_fn=lambda ctx: self._run_stage(
                    BaselineSubagent,
                    objective="Identify baselines, comparison methods, and model variants that must be reproduced.",
                    context=self._stage_context(ctx.get("structure", SubagentOutput(SubagentStatus.FAILED, "")).content),
                    config=self.reader_config,
                    phase="analyze",
                    label="paper_baseline",
                    llm=self._clone_llm(),
                ),
            ),
            SubagentTask(
                name="synthesis",
                dependencies=["structure", "algorithm", "experiments", "baseline"],
                context_keys=["structure", "algorithm", "experiments", "baseline"],
                run_fn=lambda ctx: self._run_stage(
                    SynthesisSubagent,
                    objective="Synthesize the paper analysis into a concise overview for downstream implementation agents.",
                    context=self._synthesis_context(
                        ctx.get("structure", SubagentOutput(SubagentStatus.FAILED, "")).content,
                        ctx.get("algorithm", SubagentOutput(SubagentStatus.FAILED, "")).content,
                        ctx.get("experiments", SubagentOutput(SubagentStatus.FAILED, "")).content,
                        ctx.get("baseline", SubagentOutput(SubagentStatus.FAILED, "")).content,
                    ),
                    config=self.synthesis_config,
                    phase="analyze",
                    label="paper_synthesis",
                    llm=self._clone_llm(),
                ),
            ),
        ]

    def _run_stage(
        self,
        subagent_cls,
        *,
        objective: str,
        context: str,
        config,
        phase: str,
        label: str,
        llm=None,
    ) -> SubagentOutput:
        return self.engine.run_subagent_output(
            subagent_cls,
            objective=objective,
            context=context,
            config=config,
            phase=phase,
            label=label,
            llm_override=llm if llm is not None else self.engine.llm,
        )

    def _clone_llm(self):
        llm = self.engine.llm
        if llm is None:
            return None
        config = getattr(llm, "config", None)
        if config is None:
            return llm
        return create_llm_client(config)

    def _write_outputs(self, result: PaperAnalysisResult) -> None:
        for section in result.sections.values():
            (self.engine.analysis_dir / section.filename).write_text(section.content.rstrip() + "\n", encoding="utf-8")
        (self.engine.analysis_dir / "summary.md").write_text(result.summary_with_navigation.rstrip() + "\n", encoding="utf-8")

    def _stage_context(self, structure_content: str) -> str:
        return "\n\n".join(
            [
                self.engine.reader_context(),
                "Existing structure analysis:",
                structure_content.strip() or "(no structure analysis available)",
            ]
        ).strip()

    def _synthesis_context(self, structure: str, algorithm: str, experiments: str, baseline: str) -> str:
        return "\n\n".join(
            [
                "Paper analysis inputs:",
                f"Structure:\n{structure.strip()}",
                f"Algorithm:\n{algorithm.strip()}",
                f"Experiments:\n{experiments.strip()}",
                f"Baseline:\n{baseline.strip()}",
            ]
        ).strip()

    def _section(self, name: str, filename: str, content: str, description: str) -> PaperAnalysisSection:
        return PaperAnalysisSection(name=name, filename=filename, content=content, description=description)

    def _fallback_summary(self, sections: dict[str, PaperAnalysisSection]) -> str:
        lines = [
            "# Paper Analysis: Executive Summary",
            "",
            "*Synthesis agent did not complete. Showing section previews.*",
            "",
        ]
        for key in ("structure", "algorithm", "experiments", "baseline"):
            section = sections[key]
            preview = "\n".join(section.content.splitlines()[:12]).strip()
            lines.extend(
                [
                    f"## {section.name}",
                    "",
                    preview or "(empty)",
                    "",
                ]
            )
        return "\n".join(lines).strip()


__all__ = [
    "AlgorithmSubagent",
    "BaselineSubagent",
    "ExperimentsSubagent",
    "PaperAnalysisResult",
    "PaperAnalysisSection",
    "PaperReaderCoordinator",
    "StructureSubagent",
    "SynthesisSubagent",
]
