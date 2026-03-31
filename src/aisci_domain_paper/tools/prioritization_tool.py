from __future__ import annotations

from typing import Any

from aisci_agent_runtime.tools.base import Tool
from aisci_domain_paper.subagents.prioritization import PrioritizationRunner


class PrioritizeTasksTool(Tool):
    def __init__(self, engine) -> None:
        self.engine = engine

    def name(self) -> str:
        return "prioritize_tasks"

    def execute(
        self,
        shell,  # noqa: ARG002
        paper_analysis_dir: str = "/home/agent/paper_analysis",
        rubric_path: str = "/home/paper/rubric.json",
        focus_areas: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        self.engine._ensure_workspace()
        summary_path = f"{paper_analysis_dir.rstrip('/')}/summary.md"
        if not shell.file_exists(summary_path):
            return (
                f"Error: Paper analysis not found at {paper_analysis_dir}/. "
                "Please run read_paper first to generate the paper analysis files."
            )

        result = PrioritizationRunner(self.engine).run(
            paper_analysis_dir=paper_analysis_dir,
            rubric_path=rubric_path,
            focus_areas=focus_areas,
        )
        self.engine.trace.event(
            "subagent_finish",
            "prioritize_tasks completed.",
            phase="prioritize",
            payload={"priorities": str(self.engine.prioritized_path)},
        )
        return result.summary

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "prioritize_tasks",
                "description": "Prioritize the paper implementation work and write a ranked plan to /home/agent/prioritized_tasks.md.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_analysis_dir": {
                            "type": "string",
                            "description": "Directory containing paper analysis files.",
                            "default": "/home/agent/paper_analysis",
                        },
                        "rubric_path": {
                            "type": "string",
                            "description": "Path to rubric.json.",
                            "default": "/home/paper/rubric.json",
                        },
                        "focus_areas": {
                            "type": "string",
                            "description": "Optional comma-separated areas to emphasize in the prioritization.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }


def build_prioritize_tasks_tool(engine):
    return PrioritizeTasksTool(engine)


__all__ = ["PrioritizeTasksTool", "build_prioritize_tasks_tool"]
