"""
CustomWebBasicAgentSolver - A BasicAgentSolver variant with custom web search tools.

This solver extends BasicAgentSolver to add WebSearchTool and LinkSummaryOpTool
for models that don't support the Response API's built-in web_search_preview
(e.g., GLM, DeepSeek via Completions API).

All other tools are identical to the base BasicAgentSolver.

Usage in launch script:
    paperbench.solver=paperbench.solvers.basicagent.custom_web_solver:CustomWebBasicAgentSolver

Note: Make sure to set the required environment variables for the custom tools:
    - LINK_SUMMARY_MODEL: Model name for link summarization
"""

from typing_extensions import override

import chz
from paperbench.solvers.basicagent.solver import BasicAgentSolver
from paperbench.solvers.basicagent.tools import (
    BashTool,
    PythonTool,
    ReadFileChunk,
    SearchFile,
    SubmitTool,
)
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.google_search import WebSearchTool
from paperbench.solvers.cus_tools.aweai_mcp.link_summary_op import LinkSummaryOpTool


@chz.chz
class CustomWebBasicAgentSolver(BasicAgentSolver):
    """
    BasicAgentSolver + WebSearchTool + LinkSummaryOpTool.

    Identical to BasicAgentSolver except it adds two custom search tools
    for models that can't use the Response API's built-in web_search_preview.
    """

    @override
    def shortname(self) -> str:
        if self.iterative_agent:
            return "customweb_iterativeagent"
        else:
            return "customweb_basicagent"

    @override
    def _get_tools(self) -> list[Tool]:
        if self.iterative_agent:
            tools: list[Tool] = [BashTool(), ReadFileChunk()]
        else:
            tools = [BashTool(), PythonTool(), SearchFile(), ReadFileChunk()]

        tools.extend([WebSearchTool(), LinkSummaryOpTool()])

        if self.use_submit_tool:
            tools.append(SubmitTool())

        return tools
