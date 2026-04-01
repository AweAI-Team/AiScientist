from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aisci_agent_runtime.tools.base import SubagentCompleteSignal, Tool
from aisci_agent_runtime.tools.research_tools import LinkSummaryTool, LinterTool, WebSearchTool
from aisci_agent_runtime.tools.shell_tools import (
    AddExpLogTool,
    AddImplLogTool,
    BashToolWithTimeout,
    ExecCommandTool,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
)
from aisci_domain_paper.configs import (
    ENV_SETUP_BASH_DEFAULT_TIMEOUT,
    ENV_SETUP_BASH_MAX_TIMEOUT,
    EXPERIMENT_BASH_DEFAULT_TIMEOUT,
    EXPERIMENT_COMMAND_TIMEOUT,
    IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
    MAIN_AGENT_BASH_DEFAULT_TIMEOUT,
    MAIN_AGENT_BASH_MAX_TIMEOUT,
    RESOURCE_DOWNLOAD_BASH_DEFAULT_TIMEOUT,
    RESOURCE_DOWNLOAD_BASH_MAX_TIMEOUT,
)


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


def _capability_enabled(capabilities: dict[str, Any] | None, key: str) -> bool:
    if not capabilities:
        return False
    value = capabilities.get(key)
    if isinstance(value, dict):
        return bool(value.get("available"))
    return bool(value)


class CallbackTool(Tool):
    def __init__(self, spec: _ToolSpec, callback: Callable[..., Any]):
        self._spec = spec
        self._callback = callback

    def name(self) -> str:
        return self._spec.name

    def execute(self, shell, **kwargs) -> str:  # noqa: ANN001
        return str(self._callback(shell=shell, **kwargs))

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._spec.name,
                "description": self._spec.description,
                "parameters": self._spec.parameters,
            },
        }


def callback_tool(name: str, description: str, parameters: dict[str, Any], callback) -> CallbackTool:
    return CallbackTool(_ToolSpec(name=name, description=description, parameters=parameters), callback)


class MappedFileEditTool(Tool):
    def name(self) -> str:
        return "edit_file"

    def execute(
        self,
        shell,
        command: str,
        path: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        **kwargs: Any,
    ) -> str:
        if command == "create":
            shell.write_file(path, file_text)
            lines = file_text.count("\n") + 1 if file_text else 0
            return f"Created {path} ({lines} lines)"
        if command == "str_replace":
            if not old_str:
                return "Error: old_str is required for str_replace"
            if not shell.file_exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            count = content.count(old_str)
            if count == 0:
                return f"Error: old_str not found in {path}. Use read_file_chunk first."
            if count > 1:
                return f"Error: old_str appears {count} times in {path}. Provide more context."
            shell.write_file(path, content.replace(old_str, new_str, 1))
            return f"Replaced in {path}"
        if command == "insert":
            if not shell.file_exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            lines = content.split("\n")
            idx = max(0, min(insert_line, len(lines)))
            lines.insert(idx, new_str)
            shell.write_file(path, "\n".join(lines))
            return f"Inserted at line {idx} in {path}"
        return f"Error: unknown command '{command}'. Use create / str_replace / insert."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Create or edit files with create, str_replace, or insert modes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": ["create", "str_replace", "insert"]},
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
            },
        }


class PaperGitCommitTool(Tool):
    def name(self) -> str:
        return "git_commit"

    def execute(self, shell, message: str, **kwargs: Any) -> str:
        shell.send_shell_command("cd /home/submission && git init 2>/dev/null || true", timeout=10)
        gitignore_path = "/home/submission/.gitignore"
        if not shell.file_exists(gitignore_path):
            shell.write_file(
                gitignore_path,
                "\n".join(
                    [
                        "# Auto-managed by AiScientist paper mode",
                        "venv/",
                        ".venv/",
                        "__pycache__/",
                        "*.pyc",
                        ".cache/",
                        "data/",
                        "models/",
                        "checkpoints/",
                        "",
                    ]
                ),
            )
        shell.write_file("/tmp/_paper_commit_msg.txt", message)
        result = shell.send_shell_command(
            "cd /home/submission && "
            "git config user.email 'aiscientist@local' && "
            "git config user.name 'AiScientist' && "
            "git add -A && (git diff --cached --quiet || git commit -F /tmp/_paper_commit_msg.txt) 2>&1",
            timeout=90,
        )
        return result.output.strip() or "Nothing to commit."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Stage and commit the /home/submission repository with the provided message.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        }


class SubmitTool(Tool):
    def name(self) -> str:
        return "submit"

    def execute(self, shell, summary: str, **kwargs: Any) -> str:  # noqa: ARG002
        raise SubagentCompleteSignal(summary, kwargs)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Finish the paper run after reading, prioritization, implementation, experiments, and self-check are complete.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }


class PlanWriteTool(Tool):
    PATH = Path("/home/agent/plan.md")

    def name(self) -> str:
        return "write_plan"

    def execute(self, shell, content: str, **kwargs: Any) -> str:
        shell.write_file(str(self.PATH), content)
        return f"Wrote {self.PATH}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "write_plan",
                "description": "Write a planning note to /home/agent/plan.md.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        }


class ParseRubricTool(Tool):
    def name(self) -> str:
        return "parse_rubric"

    def execute(
        self,
        shell,
        rubric_path: str = "/home/paper/rubric.json",
        max_depth: int = 3,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        if not shell.file_exists(rubric_path):
            return f"Error reading rubric: {rubric_path} does not exist. The rubric may not be available."
        try:
            payload = json.loads(shell.read_file(rubric_path))
        except Exception as exc:  # noqa: BLE001
            return f"Error reading rubric: {exc}. The rubric may not be available."

        def requirement_text(node: dict[str, Any], fallback: str = "") -> str:
            return str(
                node.get("requirements")
                or node.get("requirement")
                or node.get("name")
                or node.get("title")
                or node.get("description")
                or fallback
            ).strip()

        def children_for(node: dict[str, Any]) -> list[dict[str, Any]]:
            children = node.get("sub_tasks")
            if isinstance(children, list):
                return [child for child in children if isinstance(child, dict)]
            children = node.get("children")
            if isinstance(children, list):
                return [child for child in children if isinstance(child, dict)]
            return []

        def analyze_node(node: dict[str, Any], depth: int = 0, path: str = "") -> list[dict[str, Any]]:
            children = children_for(node)
            try:
                weight = float(node.get("weight", 1))
            except (TypeError, ValueError):
                weight = 1.0
            item = {
                "id": str(node.get("id", "")),
                "depth": depth,
                "path": path,
                "requirement": requirement_text(node, path)[:200],
                "weight": weight,
                "category": str(node.get("task_category") or node.get("finegrained_task_category") or "").strip(),
                "num_children": len(children),
                "is_leaf": len(children) == 0,
            }
            results = [item]
            if depth < max_depth:
                for index, child in enumerate(children):
                    child_path = f"{path}/{index}" if path else str(index)
                    results.extend(analyze_node(child, depth + 1, child_path))
            return results

        def compute_tree_stats(node: dict[str, Any], depth: int = 0) -> dict[str, Any]:
            children = children_for(node)
            if not children:
                return {"max_depth": depth, "total_nodes": 1, "leaf_nodes": 1, "per_level": {depth: 1}}
            stats = {"max_depth": depth, "total_nodes": 1, "leaf_nodes": 0, "per_level": {depth: 1}}
            for child in children:
                child_stats = compute_tree_stats(child, depth + 1)
                stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
                stats["total_nodes"] += child_stats["total_nodes"]
                stats["leaf_nodes"] += child_stats["leaf_nodes"]
                for level, count in child_stats["per_level"].items():
                    stats["per_level"][level] = stats["per_level"].get(level, 0) + count
            return stats

        tree_stats = compute_tree_stats(payload)
        nodes = analyze_node(payload)
        if not nodes:
            return "Rubric parsed but no weighted tasks were found."

        weights = [node["weight"] for node in nodes if node["weight"]]
        avg_weight = sum(weights) / len(weights) if weights else 1.0
        max_weight = max(weights) if weights else 1.0

        lines = [
            "# Rubric Analysis",
            "",
            f"**Total visible nodes** (depth ≤ {max_depth}): {len(nodes)}",
            f"**Average weight**: {avg_weight:.2f}",
            f"**Max weight**: {max_weight:g}",
            "",
            "## Top-Level Tasks (by weight)",
            "",
        ]

        for node in sorted(nodes, key=lambda item: (-item["weight"], item["depth"]))[:15]:
            indent = "  " * int(node["depth"])
            weight = float(node["weight"])
            if weight >= 2 * avg_weight:
                indicator = "🔴"
            elif weight >= avg_weight:
                indicator = "🟡"
            else:
                indicator = "⚪"
            category = f"[{node['category']}]" if node["category"] else ""
            children = f"({node['num_children']} sub-tasks)" if node["num_children"] > 0 else "(leaf)"
            lines.append(f"{indent}{indicator} **W={weight:g}** {category} {children}".rstrip())
            requirement = str(node["requirement"]).strip() or "(no requirement text)"
            lines.append(f"{indent}   {requirement[:150]}...")
            lines.append("")

        categories: dict[str, int] = {}
        for node in nodes:
            category = str(node["category"]).strip() or "Unspecified"
            categories[category] = categories.get(category, 0) + 1

        lines.append("## Category Distribution")
        for category, count in sorted(categories.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {category}: {count}")

        lines.extend(
            [
                "",
                "## Weight Thresholds for Priority",
                f"- P0 threshold (≥2x avg): weight ≥ {2 * avg_weight:.1f}",
                f"- P1 threshold (≥avg): weight ≥ {avg_weight:.1f}",
                f"- P2/P3: weight < {avg_weight:.1f}",
                "",
                "## Tree Statistics",
                f"- Total nodes (full tree): {tree_stats['total_nodes']}",
                f"- Leaf nodes: {tree_stats['leaf_nodes']}",
                f"- Max depth: {tree_stats['max_depth']}",
            ]
        )
        for level, count in sorted(tree_stats["per_level"].items()):
            lines.append(f"- Level {level}: {count} nodes")

        return "\n".join(lines).strip()

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "parse_rubric",
                "description": "Parse rubric.json and extract visible tasks, weights, categories, and tree statistics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rubric_path": {
                            "type": "string",
                            "description": "Path to rubric.json",
                            "default": "/home/paper/rubric.json",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "How deep to traverse the rubric tree.",
                            "default": 3,
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }


class PriorityWriteTool(Tool):
    PRIORITY_PATH = Path("/home/agent/prioritized_tasks.md")

    def name(self) -> str:
        return "write_priorities"

    def execute(self, shell, content: str, **kwargs: Any) -> str:
        shell.write_file(str(self.PRIORITY_PATH), content)
        return f"Prioritized tasks written to {self.PRIORITY_PATH}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "write_priorities",
                "description": "Write the prioritized implementation plan to /home/agent/prioritized_tasks.md.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        }


def build_main_direct_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    tools: list[Tool] = [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=MAIN_AGENT_BASH_DEFAULT_TIMEOUT, max_timeout=MAIN_AGENT_BASH_MAX_TIMEOUT),
        PythonTool(default_timeout=600, max_timeout=7200),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.append(WebSearchTool())
        tools.append(LinkSummaryTool())
    return tools


def build_reader_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(
            default_timeout=MAIN_AGENT_BASH_DEFAULT_TIMEOUT,
            max_timeout=MAIN_AGENT_BASH_MAX_TIMEOUT,
        ),
        SubagentCompleteTool(),
    ]


def build_prioritization_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        SearchFileTool(),
        ParseRubricTool(),
        PriorityWriteTool(),
        SubagentCompleteTool(),
    ]


def build_explore_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_main_direct_tools(capabilities), SubagentCompleteTool()]


def build_plan_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_main_direct_tools(capabilities), PlanWriteTool(), SubagentCompleteTool()]


def build_general_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_main_direct_tools(capabilities), SubagentCompleteTool()]


def build_implementation_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    tools: list[Tool] = [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=IMPLEMENTATION_BASH_DEFAULT_TIMEOUT, max_timeout=36_000),
        PythonTool(default_timeout=1_800, max_timeout=36_000),
        MappedFileEditTool(),
        PaperGitCommitTool(),
        AddImplLogTool(),
        LinterTool(),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.extend([WebSearchTool(), LinkSummaryTool()])
    tools.append(SubagentCompleteTool())
    return tools


def build_experiment_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    tools: list[Tool] = [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=EXPERIMENT_BASH_DEFAULT_TIMEOUT, max_timeout=36_000),
        PythonTool(default_timeout=3_600, max_timeout=36_000),
        MappedFileEditTool(),
        ExecCommandTool(default_timeout=EXPERIMENT_COMMAND_TIMEOUT, max_timeout=18_000),
        PaperGitCommitTool(),
        AddExpLogTool(),
        LinterTool(),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.extend([WebSearchTool(), LinkSummaryTool()])
    tools.append(SubagentCompleteTool())
    return tools


def build_env_setup_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        BashToolWithTimeout(default_timeout=ENV_SETUP_BASH_DEFAULT_TIMEOUT, max_timeout=ENV_SETUP_BASH_MAX_TIMEOUT),
        MappedFileEditTool(),
        SubagentCompleteTool(),
    ]


def build_resource_download_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    tools: list[Tool] = [
        ReadFileChunkTool(),
        BashToolWithTimeout(default_timeout=RESOURCE_DOWNLOAD_BASH_DEFAULT_TIMEOUT, max_timeout=RESOURCE_DOWNLOAD_BASH_MAX_TIMEOUT),
        PythonTool(default_timeout=900, max_timeout=7200),
        MappedFileEditTool(),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.extend([WebSearchTool(), LinkSummaryTool()])
    tools.append(SubagentCompleteTool())
    return tools
