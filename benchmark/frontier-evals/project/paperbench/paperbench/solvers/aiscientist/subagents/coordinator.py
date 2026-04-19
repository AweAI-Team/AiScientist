"""
Subagent Coordinator

This module provides coordination logic for running multiple subagents,
either sequentially or in parallel. It handles:
1. Dependency management between subagents
2. Context passing between subagents
3. Result aggregation and synthesis
4. Error handling and recovery
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentOutput,
    SubagentStatus,
)
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass
class SubagentTask:
    """A task to be executed by a subagent."""
    subagent: Subagent
    task_description: str
    dependencies: list[str] = field(default_factory=list)  # Names of subagents that must complete first
    context_keys: list[str] = field(default_factory=list)  # Which subagent outputs to include in context


@dataclass
class CoordinatorResult:
    """Result of a coordinated multi-subagent execution."""
    outputs: dict[str, SubagentOutput]
    synthesized_output: str
    total_runtime_seconds: float
    total_tokens: dict[str, int]
    all_success: bool
    failed_subagents: list[str]


class SubagentCoordinator:
    """
    Coordinates the execution of multiple subagents.

    The coordinator supports two main patterns:
    1. Sequential execution: Subagents run one after another, with context passing
    2. Parallel execution: Independent subagents run concurrently

    Example usage:
        coordinator = SubagentCoordinator(completer_config)

        # Define tasks
        tasks = [
            SubagentTask(StructureSubagent(...), "Extract paper structure", dependencies=[]),
            SubagentTask(AlgorithmSubagent(...), "Extract algorithms", dependencies=["structure"]),
            SubagentTask(ExperimentsSubagent(...), "Extract experiments", dependencies=["structure"]),
        ]

        # Run coordination
        result = await coordinator.run(computer, tasks, constraints)
    """

    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,
        synthesize_fn: Callable[[dict[str, SubagentOutput]], str] | None = None,
    ):
        """
        Initialize the coordinator.

        Args:
            completer_config: Configuration for the LLM completer
            synthesize_fn: Optional function to synthesize outputs from multiple subagents
        """
        self.completer_config = completer_config
        self.synthesize_fn = synthesize_fn or self._default_synthesize

    def _default_synthesize(self, outputs: dict[str, SubagentOutput]) -> str:
        """Default output synthesis: concatenate all outputs with headers."""
        sections = []
        for name, output in outputs.items():
            status_str = "✓" if output.status == SubagentStatus.COMPLETED else "✗"
            sections.append(f"## {name} [{status_str}]\n\n{output.content}")
        return "\n\n---\n\n".join(sections)

    async def run(
        self,
        computer: ComputerInterface,
        tasks: list[SubagentTask],
        constraints: dict | None = None,
    ) -> CoordinatorResult:
        """
        Execute a list of subagent tasks respecting dependencies.

        Args:
            computer: ComputerInterface for executing commands
            tasks: List of SubagentTask to execute
            constraints: Optional constraints (e.g., blacklist patterns)

        Returns:
            CoordinatorResult with all outputs and synthesized result
        """
        ctx_logger = logger.bind(coordinator="SubagentCoordinator")
        start_time = time.time()

        outputs: dict[str, SubagentOutput] = {}
        total_tokens = {"input": 0, "output": 0}
        failed_subagents = []

        # Topological sort to determine execution order
        execution_order = self._topological_sort(tasks)
        ctx_logger.info(f"Execution order: {execution_order}")

        # Group tasks by their dependency level for parallel execution
        levels = self._group_by_level(tasks)
        ctx_logger.info(f"Execution levels: {levels}")

        for level_idx, level_tasks in enumerate(levels):
            ctx_logger.info(f"Executing level {level_idx}: {[t.subagent.name for t in level_tasks]}")

            # Run tasks at this level in parallel
            async_tasks = []
            for task in level_tasks:
                # Build context from completed dependencies
                context = {}
                for ctx_key in task.context_keys:
                    if ctx_key in outputs:
                        context[ctx_key] = outputs[ctx_key].content
                        if outputs[ctx_key].artifacts:
                            context[f"{ctx_key}_artifacts"] = outputs[ctx_key].artifacts

                async_tasks.append(
                    self._run_single_task(computer, task, context, constraints)
                )

            # Wait for all tasks at this level
            results = await asyncio.gather(*async_tasks, return_exceptions=True)

            # Process results
            for task, result in zip(level_tasks, results):
                if isinstance(result, Exception):
                    ctx_logger.error(f"Subagent {task.subagent.name} raised exception: {result}")
                    outputs[task.subagent.name] = SubagentOutput(
                        subagent_name=task.subagent.name,
                        status=SubagentStatus.FAILED,
                        content="",
                        error_message=str(result),
                    )
                    failed_subagents.append(task.subagent.name)
                else:
                    outputs[task.subagent.name] = result
                    total_tokens["input"] += result.token_usage.get("input", 0)
                    total_tokens["output"] += result.token_usage.get("output", 0)
                    if result.status != SubagentStatus.COMPLETED:
                        failed_subagents.append(task.subagent.name)

        # Synthesize outputs
        synthesized = self.synthesize_fn(outputs)

        return CoordinatorResult(
            outputs=outputs,
            synthesized_output=synthesized,
            total_runtime_seconds=time.time() - start_time,
            total_tokens=total_tokens,
            all_success=len(failed_subagents) == 0,
            failed_subagents=failed_subagents,
        )

    async def _run_single_task(
        self,
        computer: ComputerInterface,
        task: SubagentTask,
        context: dict[str, Any],
        constraints: dict | None,
    ) -> SubagentOutput:
        """Run a single subagent task."""
        return await task.subagent.run(
            computer=computer,
            task_description=task.task_description,
            constraints=constraints,
            context=context if context else None,
        )

    def _topological_sort(self, tasks: list[SubagentTask]) -> list[str]:
        """Topological sort of tasks based on dependencies with cycle detection."""
        task_map = {task.subagent.name: task for task in tasks}
        visited: set[str] = set()
        in_stack: set[str] = set()  # Tracks current DFS path for cycle detection
        result: list[str] = []

        def visit(name: str) -> None:
            if name in in_stack:
                raise ValueError(f"Cyclic dependency detected involving '{name}'")
            if name in visited:
                return
            in_stack.add(name)
            visited.add(name)
            task = task_map.get(name)
            if task:
                for dep in task.dependencies:
                    visit(dep)
            in_stack.discard(name)
            result.append(name)

        for task in tasks:
            visit(task.subagent.name)

        return result

    def _group_by_level(self, tasks: list[SubagentTask]) -> list[list[SubagentTask]]:
        """Group tasks by dependency level for parallel execution."""
        task_map = {task.subagent.name: task for task in tasks}
        levels: dict[str, int] = {}
        in_stack: set[str] = set()  # Tracks current recursion path for cycle detection

        def get_level(name: str) -> int:
            if name in levels:
                return levels[name]
            if name in in_stack:
                raise ValueError(f"Cyclic dependency detected involving '{name}'")
            in_stack.add(name)
            task = task_map.get(name)
            if not task or not task.dependencies:
                levels[name] = 0
            else:
                levels[name] = 1 + max(get_level(dep) for dep in task.dependencies)
            in_stack.discard(name)
            return levels[name]

        for task in tasks:
            get_level(task.subagent.name)

        # Group by level
        max_level = max(levels.values()) if levels else 0
        grouped = [[] for _ in range(max_level + 1)]
        for task in tasks:
            level = levels[task.subagent.name]
            grouped[level].append(task)

        return grouped


class SequentialCoordinator(SubagentCoordinator):
    """
    A simpler coordinator that runs subagents strictly sequentially.

    Useful when:
    - Subagents have strict dependencies
    - Memory/resource constraints prevent parallel execution
    - Debugging subagent behavior
    """

    async def run(
        self,
        computer: ComputerInterface,
        tasks: list[SubagentTask],
        constraints: dict | None = None,
    ) -> CoordinatorResult:
        """Execute tasks strictly sequentially."""
        ctx_logger = logger.bind(coordinator="SequentialCoordinator")
        start_time = time.time()

        outputs: dict[str, SubagentOutput] = {}
        total_tokens = {"input": 0, "output": 0}
        failed_subagents = []

        for task in tasks:
            ctx_logger.info(f"Executing: {task.subagent.name}")

            # Build context from completed dependencies
            context = {}
            for ctx_key in task.context_keys:
                if ctx_key in outputs:
                    context[ctx_key] = outputs[ctx_key].content
                    if outputs[ctx_key].artifacts:
                        context[f"{ctx_key}_artifacts"] = outputs[ctx_key].artifacts

            try:
                result = await task.subagent.run(
                    computer=computer,
                    task_description=task.task_description,
                    constraints=constraints,
                    context=context if context else None,
                )
                outputs[task.subagent.name] = result
                total_tokens["input"] += result.token_usage.get("input", 0)
                total_tokens["output"] += result.token_usage.get("output", 0)
                if result.status != SubagentStatus.COMPLETED:
                    failed_subagents.append(task.subagent.name)
            except Exception as e:
                ctx_logger.error(f"Subagent {task.subagent.name} failed: {e}")
                outputs[task.subagent.name] = SubagentOutput(
                    subagent_name=task.subagent.name,
                    status=SubagentStatus.FAILED,
                    content="",
                    error_message=str(e),
                )
                failed_subagents.append(task.subagent.name)

        synthesized = self.synthesize_fn(outputs)

        return CoordinatorResult(
            outputs=outputs,
            synthesized_output=synthesized,
            total_runtime_seconds=time.time() - start_time,
            total_tokens=total_tokens,
            all_success=len(failed_subagents) == 0,
            failed_subagents=failed_subagents,
        )
