"""
Web search tool with blacklist constraint support.
These tools wrap ByteDance internal search services (bandai_mcp_host).

Usage:
    from paperbench.solvers.cus_tools.aweai_mcp.google_search import WebSearchTool
    
    # Add to your solver config:
    solver.basicagent_tools = [WebSearchTool()]
    
    # Execute with blacklist constraints:
    constraints = {
        "blocked_search_patterns": {
            "url": [r".*github\\.com/user/repo.*"]
        }
    }
    result = await tool.execute_with_constraints(computer, query="test", constraints=constraints)
"""

import json
import os
import re
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.utils import build_blocked_patterns_from_blacklist


class WebSearchTool(Tool):
    """
    Web search tool using ByteDance internal bandai_mcp_host service.
    
    This is an alternative to OpenAI's built-in web_search for environments
    that don't support the Responses API. Supports blacklist constraints
    to prevent access to blocked URLs.
    """
    
    # Configuration for the search engine
    engine: str = "google"
    scheme: list[str] = ["position", "title", "description", "snippets", "url"]
    max_attempts: int = 1
    
    # Lazy-loaded search function
    _execute_fn: Any = None
    
    def name(self) -> str:
        return "web_search"
    
    def _get_execute_fn(self) -> Any:
        """Lazy load the bandai_mcp_host search function."""
        if self._execute_fn is None:
            try:
                os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
                from bytedance.bandai_mcp_host import map_tools
                self._execute_fn = map_tools("Search")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for WebSearchTool. "
                    "Please install it or use a different search implementation."
                ) from e
        return self._execute_fn
    
    async def execute(
        self,
        computer: ComputerInterface,
        query: str | list[str],
        num: int = 10,
        start: int = 0,
    ) -> str:
        """
        Execute web search with the given query.
        
        Args:
            computer: ComputerInterface (not used, but required by interface)
            query: Search query string or list of queries
            num: Number of results to return (default: 10)
            start: Pagination start index (default: 0)
            
        Returns:
            Formatted search results as a string
        """
        results = await self._execute_search_str(query, num, start)
        return results
    
    async def execute_with_constraints(
        self,
        computer: ComputerInterface,
        constraints: dict | None = None,
        query: str | list[str] = "",
        num: int = 10,
        start: int = 0,
    ) -> str:
        """
        Execute search with constraints to filter out blocked URLs from results.
        
        Args:
            computer: ComputerInterface (not used, but required by interface)
            constraints: Execution constraints, can include:
                - blocked_search_patterns: dict[str, list[str]] - Regex patterns for fields to filter out
                  (e.g., {"url": ['.*github\\.com/django/django.*'], "title": ['.*django.*']})
            query: Search query string or list of queries
            num: Number of results to return (default: 10)
            start: Pagination start index (default: 0)
        
        Returns:
            Filtered search results as a string
        """
        # First execute the search to get structured results
        results_list = await self._execute_search_list(query, num, start)
        
        constraints = constraints or {}
        blocked_search_patterns = constraints.get('blocked_search_patterns', {})
        
        if not blocked_search_patterns:
            # No constraints, convert back to string format
            constrains_output = []
            for item in results_list:
                constrains_output.append(f"### Search Query: {item['query']}\n{json.dumps(item['content'], ensure_ascii=False, indent=2)}")
            return "\n\n".join(constrains_output)
        
        # Filter the search results
        filtered_outputs, filtered_counts = self._filter_blocked_urls(results_list, blocked_search_patterns)
        
        constrains_output = []
        for item, filtered_count in zip(filtered_outputs, filtered_counts):
            warning = (
                f"\nWARNING: {filtered_count} search result(s) were filtered out because they "
                f"reference blocked resources. You are not allowed to access the blacklists.\n\n"
            ) if filtered_count > 0 else ""

            constrains_output.append(f"### Search Query: {item['query']}\n{warning}{json.dumps(item['content'], ensure_ascii=False, indent=2)}")
        
        return "\n\n".join(constrains_output)

    async def _execute_search_str(
        self,
        query: str | list[str],
        num: int = 10,
        start: int = 0,
    ) -> str:
        """
        Internal method to execute search and return formatted string.
        
        Args:
            query: Search query string or list of queries
            num: Number of results to return
            start: Pagination start index
            
        Returns:
            Search results as formatted string
        """
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse
        
        execute_fn = self._get_execute_fn()
        
        # Support multiple queries
        queries = query if isinstance(query, list) else [query]
        results: list[str] = []
        
        for single_query in queries:
            params = {
                "query": single_query,
                "engine": self.engine,
                "scheme": self.scheme,
                "num": num,
                "start": start,
            }
            
            # Try with retries
            response: BandaiToolResponse | None = None
            for _ in range(self.max_attempts):
                response = await execute_fn(**params)
                if response is not None and response.status.is_succeeded():
                    break
            
            if response is not None and response.status.is_succeeded():
                results.append(f"### Search Query: {single_query}\n{response.result}")
            else:
                error_msg = response.result if response is not None else "Unknown error"
                results.append(f"### Search Query: {single_query}\nError: {error_msg}")
        
        return "\n\n".join(results)

    async def _execute_search_list(
        self,
        query: str | list[str],
        num: int = 10,
        start: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Internal method to execute search and return structured list.
        
        Args:
            query: Search query string or list of queries
            num: Number of results to return
            start: Pagination start index
            
        Returns:
            Search results as list of dicts with 'query' and 'content' keys
        """
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse
        
        execute_fn = self._get_execute_fn()
        
        # Support multiple queries
        queries = query if isinstance(query, list) else [query]
        results: list[dict[str, Any]] = []
        
        for single_query in queries:
            params = {
                "query": single_query,
                "engine": self.engine,
                "scheme": self.scheme,
                "num": num,
                "start": start,
            }
            
            # Try with retries
            response: BandaiToolResponse | None = None
            for _ in range(self.max_attempts):
                response = await execute_fn(**params)
                if response is not None and response.status.is_succeeded():
                    break
            
            if response is not None and response.status.is_succeeded():
                # Parse the response result if it's a string
                content: Any = response.result
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        content = [{"raw": content}]
                results.append({
                    'query': single_query,
                    'content': content,
                })
            else:
                error_msg = response.result if response is not None else "Unknown error"
                results.append({
                    'query': single_query,
                    'content': f"Error: {error_msg}",
                })
        
        return results

    def _filter_blocked_urls(
        self,
        content: list[dict],
        blocked_patterns: dict[str, list[str]]
    ) -> tuple[list[dict], list[int]]:
        """
        Filter out search result entries that match blocked patterns.
        
        Args:
            content: The search result content (list of {query, content} dicts)
            blocked_patterns: dict[str, list[str]] - Regex patterns for fields to filter out
                  (e.g., {
                    "url": ['.*github\\.com/django/django.*'],
                    "title": ['.*django.*']
                  })
        
        Returns:
            Tuple of (filtered_results, filtered_counts)
        """
        if not content or not blocked_patterns:
            return content, [0] * len(content)
        
        final_results = []
        filtered_counts = []
        
        for result in content:
            filtered_count = 0
            filtered_result = []
            
            # Get content list
            content_list = result.get('content', [])
            if isinstance(content_list, str):
                try:
                    content_list = json.loads(content_list)
                except json.JSONDecodeError:
                    content_list = []
            
            if not isinstance(content_list, list):
                # If content is not a list (e.g., error message), keep it as is
                final_results.append(result)
                filtered_counts.append(0)
                continue
            
            for item in content_list:
                if not isinstance(item, dict):
                    filtered_result.append(item)
                    continue
                    
                # Check if this item matches any blocked pattern
                is_blocked = False
                for key, patterns in blocked_patterns.items():
                    if key in item:
                        for pattern in patterns:
                            try:
                                if re.match(pattern, str(item[key]), re.IGNORECASE):
                                    is_blocked = True
                                    break
                            except re.error:
                                # Invalid regex pattern, skip it
                                continue
                    if is_blocked:
                        break
                
                if is_blocked:
                    filtered_count += 1
                else:
                    filtered_result.append(item)
            
            final_results.append({
                'query': result['query'],
                'content': filtered_result,
            })
            filtered_counts.append(filtered_count)
        
        return final_results, filtered_counts
    
    def supports_constraints(self) -> bool:
        """Returns True as this tool supports blacklist constraint filtering."""
        return True

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Execute a web search to retrieve relevant information from the internet. "
                "Use this when you need current information, documentation, or research data. "
                "Supports single or multiple queries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "description": "The search keyword(s) to look up.",
                        "anyOf": [
                            {
                                "type": "string",
                                "description": "A single search query."
                            },
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of search queries."
                            }
                        ]
                    },
                    "num": {
                        "type": "integer",
                        "description": "Number of results to return. Defaults to 10.",
                        "default": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
        )


# ==============================================================================
# Debug / Test Entry Point
# ==============================================================================

async def _test_search_basic():
    """Test basic search without constraints."""
    print("=" * 60)
    print("Test 1: Basic search without constraints")
    print("=" * 60)
    
    tool = WebSearchTool()
    # Note: computer is not used in this tool, so we pass None
    result = await tool.execute(None, query="Python asyncio tutorial", num=3)  # type: ignore
    print(result)
    print()


async def _test_search_with_constraints():
    """Test search with blacklist constraints."""
    print("=" * 60)
    print("Test 2: Search with blacklist constraints")
    print("=" * 60)
    
    tool = WebSearchTool()
    
    # Simulate a blacklist that blocks GitHub astropy repo
    blocked_patterns = {
        "url": [
            r'.*github\.com/[^/]+/astropy(/|$|\?).*',
        ]
    }
    
    constraints = {"blocked_search_patterns": blocked_patterns}
    
    result = await tool.execute_with_constraints(
        None,  # type: ignore
        constraints=constraints,
        query="astropy python astronomy",
        num=5
    )
    print(result)
    print()


async def _test_build_blacklist_patterns():
    """Test building patterns from blacklist file content."""
    print("=" * 60)
    print("Test 3: Build blocked patterns from blacklist")
    print("=" * 60)
    
    # Simulate blacklist.txt content
    blacklist_content = [
        "https://github.com/modichirag/GSM-VI",
        "# This is a comment",
        "",  # Empty line
        "https://github.com/another/repo",
    ]
    
    patterns = build_blocked_patterns_from_blacklist(blacklist_content)
    print(f"Generated patterns: {json.dumps(patterns, indent=2)}")
    print()
    
    # Test filtering with these patterns
    print("Testing filter with mock search results...")
    tool = WebSearchTool()
    
    mock_results = [
        {
            "query": "GSM-VI machine learning",
            "content": [
                {"position": 1, "title": "GSM-VI GitHub", "url": "https://github.com/modichirag/GSM-VI", "description": "Official repo"},
                {"position": 2, "title": "GSM-VI Paper", "url": "https://arxiv.org/abs/xxx", "description": "Paper link"},
                {"position": 3, "title": "Another Repo", "url": "https://github.com/another/repo/issues", "description": "Issues page"},
                {"position": 4, "title": "Safe Result", "url": "https://example.com/gsm", "description": "Some other site"},
            ]
        }
    ]
    
    filtered, counts = tool._filter_blocked_urls(mock_results, patterns)
    print(f"Filtered count: {counts[0]}")
    print(f"Remaining results: {json.dumps(filtered, indent=2)}")
    print()


async def _test_multiple_queries():
    """Test search with multiple queries."""
    print("=" * 60)
    print("Test 4: Multiple queries with constraints")
    print("=" * 60)
    
    tool = WebSearchTool()
    
    blocked_patterns = {
        "url": [
            r'.*github\.com/modichirag/GSM-VI.*',
        ]
    }
    
    result = await tool.execute_with_constraints(
        None,  # type: ignore
        constraints={"blocked_search_patterns": blocked_patterns},
        query=["GSM-VI variational inference", "machine learning sampling"],
        num=3
    )
    print(result)
    print()


async def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WebSearchTool")
    parser.add_argument(
        "--test",
        choices=["basic", "constraints", "blacklist", "multiple", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query to search (only for basic test)"
    )
    parser.add_argument(
        "--blacklist",
        type=str,
        nargs="+",
        help="Custom blacklist URLs (only for constraints test)"
    )
    
    args = parser.parse_args()
    
    if args.test == "basic" or args.test == "all":
        if args.query:
            tool = WebSearchTool()
            result = await tool.execute(None, query=args.query, num=5)  # type: ignore
            print(result)
        else:
            await _test_search_basic()
    
    if args.test == "constraints" or args.test == "all":
        if args.blacklist:
            tool = WebSearchTool()
            patterns = build_blocked_patterns_from_blacklist(args.blacklist)
            result = await tool.execute_with_constraints(
                None,  # type: ignore
                constraints={"blocked_search_patterns": patterns},
                query=args.query or "test search",
                num=5
            )
            print(result)
        else:
            await _test_search_with_constraints()
    
    if args.test == "blacklist" or args.test == "all":
        await _test_build_blacklist_patterns()
    
    if args.test == "multiple" or args.test == "all":
        await _test_multiple_queries()


if __name__ == "__main__":
    # # 运行所有测试
    # python google.py

    # # 只测试基本搜索
    # python google.py --test basic

    # # 测试带约束的搜索
    # python google.py --test constraints

    # # 测试 blacklist 模式构建和过滤逻辑
    # python google.py --test blacklist

    # # 自定义查询
    # python google.py --test basic --query "your search query"

    # # 自定义 blacklist
    # python google.py --test constraints --blacklist "https://github.com/user/repo" --query "test"
    import asyncio
    asyncio.run(main())
