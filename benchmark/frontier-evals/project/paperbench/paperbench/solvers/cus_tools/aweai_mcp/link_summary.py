"""
Link summary tool with blacklist constraint support.
This tool wraps ByteDance internal LinkSummary service (bandai_mcp_host).

Usage:
    from paperbench.solvers.cus_tools.aweai_mcp.link_summary import LinkSummaryTool
    
    # Add to your solver config:
    solver.basicagent_tools = [LinkSummaryTool()]
    
    # Execute with blacklist constraints:
    constraints = {
        "blocked_search_patterns": {
            "url": [r".*github\\.com/user/repo.*"]
        }
    }
    result = await tool.execute_with_constraints(
        computer,
        constraints=constraints,
        url="https://example.com",
        goal="Summarize the main points"
    )
"""

import json
import re
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool


# System prompt for LinkSummary - objective information extraction
LINK_SUMMARY_PROMPT = """You are a specialized information extraction assistant for an AI research agent working on paper replication tasks. Your job is to summarize web content in a way that maximizes actionable value for the agent.

## Input
- **URL Content**: The parsed content from the target URL
- **Goal**: The specific purpose/question the agent has when accessing this URL

## Output Requirements

### 1. Content Type Identification
First, identify the content type:
- `paper`: Research paper / arXiv / PDF
- `code`: GitHub repo / code documentation
- `docs`: API documentation / library docs
- `tutorial`: How-to guides / blog posts
- `forum`: Stack Overflow / GitHub issues / discussions
- `other`: Other types

### 2. Goal-Oriented Summary
Based on the agent's goal, extract and prioritize information that directly addresses the goal. Structure your response as:

#### Key Information (Required)
- Directly answer the agent's goal if possible
- Extract specific facts, numbers, code snippets, or commands relevant to the goal
- If the goal cannot be answered from this content, explicitly state this

#### Technical Details (If Applicable)
- For **papers**: Key methods, algorithms, hyperparameters, datasets, evaluation metrics, experimental setup
- For **code repos**: Installation steps, dependencies, file structure, key functions/classes, usage examples
- For **docs**: API signatures, parameters, return values, code examples
- For **forums**: Accepted solutions, workarounds, error explanations

#### Actionable Items
- Include exact commands, code snippets, or configurations when available
- Prerequisites or dependencies needed

#### Related Resources (Optional)
- Links or references mentioned that might be useful for the agent's broader task
- Only include if highly relevant to research replication

### 3. Formatting Rules
- Be concise but complete - prioritize information density
- Use bullet points and code blocks for clarity
- Preserve exact technical terms, variable names, and numbers
- Do NOT include: navigation elements, ads, boilerplate text, irrelevant sections
- Maximum length: ~800 tokens (adjust based on content richness)
- Always use fenced code blocks with language tags:
    - Code: ```python, ```bash, ```yaml, ```json, ```cpp, ```cuda, etc.
    - Formulas: ```latex

### 4. Technical Precision
- Preserve exact numerical values, variable names, and technical terms
- Keep formula notation consistent with source (e.g., don't change θ to w)
- Note library/framework versions when code is version-sensitive

### 5. Objectivity and Uncertainty Handling
- **Extract only**: Report only what is explicitly stated in the content. Never infer, assume, or fabricate missing information.
- If information is ambiguous or potentially outdated, flag it clearly
- If the page content is incomplete or lacks details needed to answer the goal, explicitly state what is missing rather than guessing
- Distinguish between direct quotes/facts from the source vs. your interpretation

## Special Instructions
1. **For Paper Replication Tasks**: Pay extra attention to:
   - Exact numerical values (don't approximate)
   - Implementation details that differ from standard practices
   - Clarifications or errata mentioned
   - Version numbers of libraries/frameworks

2. **When Goal is Vague**: If the agent's goal is broad (e.g., "understand this repo"), provide a structured overview focusing on what would be needed to replicate or use the code.

3. **When Content is Dense**: For very long content, focus ruthlessly on the agent's goal. Summarize peripheral information in one line if at all.
"""


class LinkSummaryTool(Tool):
    """
    Link summary tool using ByteDance internal bandai_mcp_host service.
    
    This tool visits a URL and extracts information based on a specified goal.
    Supports blacklist constraints to prevent access to blocked URLs.
    """
    
    # Configuration
    max_attempts: int = 1
    system_prompt: str = LINK_SUMMARY_PROMPT
    
    # Lazy-loaded execute function
    _execute_fn: Any = None
    
    def name(self) -> str:
        return "link_summary"
    
    def _get_execute_fn(self) -> Any:
        """Lazy load the bandai_mcp_host LinkSummary function."""
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools
                self._execute_fn = map_tools("LinkSummary")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for LinkSummaryTool. "
                    "Please install it or use a different implementation."
                ) from e
        return self._execute_fn
    
    async def execute(
        self,
        computer: ComputerInterface,
        url: str,
        goal: str,
    ) -> str:
        """
        Execute link summary to extract information from a URL.
        
        Args:
            computer: ComputerInterface (not used, but required by interface)
            url: The target URL to read (web link or PDF)
            goal: The specific instruction for information extraction
            
        Returns:
            Extracted information as a string
        """
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse
        
        execute_fn = self._get_execute_fn()
        
        params = {
            "url": url,
            "question": goal,
            "system_prompt": self.system_prompt,
        }
        
        # Try with retries
        response: BandaiToolResponse | None = None
        for _ in range(self.max_attempts):
            response = await execute_fn(**params)
            if response is not None and response.status.is_succeeded():
                break
        
        if response is not None and response.status.is_succeeded():
            # Try to extract 'summary' from JSON response
            try:
                content = json.loads(response.result)
                if isinstance(content, dict) and 'summary' in content:
                    return content['summary']
                return response.result
            except json.JSONDecodeError:
                return response.result
        else:
            error_msg = response.result if response is not None else "Unknown error"
            return f"Error reading URL '{url}': {error_msg}"
    
    async def execute_with_constraints(
        self,
        computer: ComputerInterface,
        constraints: dict | None = None,
        url: str = "",
        goal: str = "",
    ) -> str:
        """
        Execute link summary with constraints to block access to blacklisted URLs.
        
        Args:
            computer: ComputerInterface (not used, but required by interface)
            constraints: Execution constraints, can include:
                - blocked_search_patterns: dict[str, list[str]] - Regex patterns for URLs to block
                  (e.g., {"url": ['.*github\\.com/django/django.*']})
            url: The target URL to read
            goal: The specific instruction for information extraction
        
        Returns:
            Extracted information or blocked message
        """
        constraints = constraints or {}
        blocked_patterns = constraints.get('blocked_search_patterns', {})
        
        # Check if URL is blocked
        if blocked_patterns and self._is_url_blocked(url, blocked_patterns):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked because it references "
                f"a blacklisted resource. You are not allowed to access the repository's "
                f"GitHub page, issues, pull requests, or related resources.\n\n"
                f"Please search for alternative sources or try a different approach."
            )
        
        # URL is allowed, proceed with normal execution
        return await self.execute(computer, url=url, goal=goal)
    
    def _is_url_blocked(self, url: str, blocked_patterns: dict[str, list[str]]) -> bool:
        """
        Check if a URL matches any blocked pattern.
        
        Args:
            url: The URL to check
            blocked_patterns: dict with 'url' key containing list of regex patterns
            
        Returns:
            True if URL is blocked, False otherwise
        """
        if not url or not blocked_patterns:
            return False
        
        url_patterns = blocked_patterns.get('url', [])
        for pattern in url_patterns:
            try:
                if re.match(pattern, url, re.IGNORECASE):
                    return True
            except re.error:
                # Invalid regex pattern, skip it
                continue
        
        return False
    
    def supports_constraints(self) -> bool:
        """Returns True as this tool supports blacklist constraint filtering."""
        return True

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Visit and process the content of a specified URL (webpage or PDF document). "
                "It extracts specific information or generates a focused summary based on the provided 'goal'. "
                "Use this tool to read a specific search result, analyze documentation, or extract information from external resources."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The target URL to read. Can be a web link or a PDF url."
                    },
                    "goal": {
                        "type": "string",
                        "description": (
                            "The specific instruction for information extraction. "
                            "Example: 'Summarize the main contribution', 'Extract the installation steps', "
                            "'Find the API usage examples', or 'Extract the pricing table'. "
                            "Be specific to get the best results."
                        )
                    },
                },
                "required": ["url", "goal"],
                "additionalProperties": False,
            },
            strict=False,
        )


# ==============================================================================
# Debug / Test Entry Point
# ==============================================================================

async def _test_basic_link_summary():
    """Test basic link summary without constraints."""
    print("=" * 60)
    print("Test 1: Basic link summary without constraints")
    print("=" * 60)
    
    tool = LinkSummaryTool()
    result = await tool.execute(
        None,  # type: ignore
        url="https://arxiv.org/abs/2405.03553",
        goal="Summarize the main contribution of this paper"
    )
    print(result)
    print()


async def _test_link_summary_with_constraints():
    """Test link summary with blacklist constraints."""
    print("=" * 60)
    print("Test 2: Link summary with blacklist constraints (should be blocked)")
    print("=" * 60)
    
    tool = LinkSummaryTool()
    
    # Simulate a blacklist that blocks a GitHub repo
    blocked_patterns = {
        "url": [
            r'.*github\.com/modichirag/GSM-VI.*',
        ]
    }
    
    constraints = {"blocked_search_patterns": blocked_patterns}
    
    # Try to access a blocked URL
    result = await tool.execute_with_constraints(
        None,  # type: ignore
        constraints=constraints,
        url="https://github.com/modichirag/GSM-VI",
        goal="Get the installation instructions"
    )
    print(result)
    print()


async def _test_link_summary_allowed_url():
    """Test link summary with constraints but allowed URL."""
    print("=" * 60)
    print("Test 3: Link summary with constraints (URL allowed)")
    print("=" * 60)
    
    tool = LinkSummaryTool()
    
    blocked_patterns = {
        "url": [
            r'.*github\.com/modichirag/GSM-VI.*',
        ]
    }
    
    constraints = {"blocked_search_patterns": blocked_patterns}
    
    # Try to access an allowed URL
    result = await tool.execute_with_constraints(
        None,  # type: ignore
        constraints=constraints,
        url="https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ColumnChunkMetaData.html",
        goal="Extract the attributes and methods of ColumnChunkMetaData class"
    )
    print(result)
    print()


async def _test_url_blocking_logic():
    """Test URL blocking logic with various patterns."""
    print("=" * 60)
    print("Test 4: URL blocking logic")
    print("=" * 60)
    
    tool = LinkSummaryTool()
    
    blocked_patterns = {
        "url": [
            r'.*github\.com/modichirag/GSM-VI.*',
            r'.*github\.com/another/repo.*',
        ]
    }
    
    test_urls = [
        "https://github.com/modichirag/GSM-VI",
        "https://github.com/modichirag/GSM-VI/issues/123",
        "https://github.com/modichirag/GSM-VI/pull/456",
        "https://github.com/another/repo/blob/main/README.md",
        "https://github.com/safe/repo",  # Should be allowed
        "https://arxiv.org/abs/2405.03553",  # Should be allowed
        "https://stackoverflow.com/questions/123",  # Should be allowed
    ]
    
    for url in test_urls:
        is_blocked = tool._is_url_blocked(url, blocked_patterns)
        status = "BLOCKED" if is_blocked else "ALLOWED"
        print(f"  {status}: {url}")
    print()


async def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LinkSummaryTool")
    parser.add_argument(
        "--test",
        choices=["basic", "blocked", "allowed", "logic", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Custom URL to summarize"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Summarize the main content",
        help="Goal for information extraction"
    )
    parser.add_argument(
        "--blacklist",
        type=str,
        nargs="+",
        help="Custom blacklist URL patterns"
    )
    
    args = parser.parse_args()
    
    if args.url:
        # Custom URL test
        tool = LinkSummaryTool()
        if args.blacklist:
            from paperbench.solvers.cus_tools.aweai_mcp.utils import build_blocked_patterns_from_blacklist
            patterns = build_blocked_patterns_from_blacklist(args.blacklist)
            result = await tool.execute_with_constraints(
                None,  # type: ignore
                constraints={"blocked_search_patterns": patterns},
                url=args.url,
                goal=args.goal
            )
        else:
            result = await tool.execute(None, url=args.url, goal=args.goal)  # type: ignore
        print(result)
        return
    
    if args.test == "basic" or args.test == "all":
        await _test_basic_link_summary()
    
    if args.test == "blocked" or args.test == "all":
        await _test_link_summary_with_constraints()
    
    if args.test == "allowed" or args.test == "all":
        await _test_link_summary_allowed_url()
    
    if args.test == "logic" or args.test == "all":
        await _test_url_blocking_logic()


if __name__ == "__main__":
    # 运行所有测试
    # python link_summary.py
    
    # 只测试基本功能
    # python link_summary.py --test basic
    
    # 测试被阻止的 URL
    # python link_summary.py --test blocked
    
    # 测试 URL 阻止逻辑
    # python link_summary.py --test logic
    
    # 自定义 URL 测试
    # python link_summary.py --url "https://example.com" --goal "Summarize the content"
    
    # 自定义 URL + blacklist 测试
    # python link_summary.py --url "https://github.com/user/repo" --blacklist "https://github.com/user/repo" --goal "Get README"
    
    import asyncio
    asyncio.run(main())
