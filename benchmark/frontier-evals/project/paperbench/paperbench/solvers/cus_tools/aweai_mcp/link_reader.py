"""
Link reader tool with blacklist constraint support.
This tool wraps ByteDance internal LinkReader service (bandai_mcp_host).

Unlike LinkSummary, this tool returns the raw content without any processing or summarization.

Usage:
    from paperbench.solvers.cus_tools.aweai_mcp.link_reader import LinkReaderTool
    
    # Add to your solver config:
    solver.basicagent_tools = [LinkReaderTool()]
    
    # Execute with blacklist constraints:
    constraints = {
        "blocked_search_patterns": {
            "url": [r".*github\\.com/user/repo.*"]
        }
    }
    result = await tool.execute_with_constraints(
        computer,
        constraints=constraints,
        url="https://example.com"
    )
    
Environment Variables:
    LINK_READER_MAX_TOKENS: Maximum number of tokens to return (default: 25000)
    LINK_READER_USE_TIKTOKEN: Whether to use tiktoken for accurate token counting (default: "true")
"""

import os
import re
import json
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None

# Environment variable names
ENV_MAX_TOKENS = "LINK_READER_MAX_TOKENS"
ENV_USE_TIKTOKEN = "LINK_READER_USE_TIKTOKEN"

# Default values
DEFAULT_MAX_TOKENS = 100000  # Default max tokens to return
FAST_ESTIMATE_THRESHOLD = 0.8  # If fast estimate < 80% of max, skip tiktoken


def estimate_tokens_fast(text: str) -> int:
    """
    Fast token estimation using heuristics.
    
    Uses a combination of word count and character count:
    - English: ~1.3 tokens per word (words = split by whitespace)
    - Chinese/CJK: ~2 tokens per character
    
    This is a rough estimate but very fast for long texts.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Count words (split by whitespace)
    words = text.split()
    word_count = len(words)
    
    # Count CJK characters (Chinese, Japanese, Korean)
    cjk_count = 0
    for char in text:
        # CJK Unified Ideographs and other CJK ranges
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            cjk_count += 1
        elif '\u3040' <= char <= '\u309f':  # Hiragana
            cjk_count += 1
        elif '\u30a0' <= char <= '\u30ff':  # Katakana
            cjk_count += 1
        elif '\uac00' <= char <= '\ud7af':  # Korean Hangul
            cjk_count += 1
    
    # Estimate tokens:
    # - English words: ~1.3 tokens per word
    # - CJK characters: ~2 tokens per character (but they're already counted in words partially)
    # Simple heuristic: word_count * 1.3 + additional CJK overhead
    english_tokens = int(word_count * 1.3)
    cjk_tokens = int(cjk_count * 0.7)  # Additional overhead for CJK (some already in word count)
    
    return english_tokens + cjk_tokens


def estimate_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """
    Accurate token estimation using tiktoken.
    
    Args:
        text: The text to count tokens for
        model: Model name for encoding (default: gpt-5 uses o200k_base)
        
    Returns:
        Exact token count
    """
    if not HAS_TIKTOKEN or tiktoken is None:
        return estimate_tokens_fast(text)
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to o200k_base encoding (used by GPT-5)
        encoding = tiktoken.get_encoding("o200k_base")
    
    return len(encoding.encode(text))


def truncate_content_by_tokens(
    text: str, 
    max_tokens: int,
    use_tiktoken: bool = True,
) -> tuple[str, bool]:
    """
    Truncate content to fit within max_tokens limit.
    
    Uses a multi-stage approach for efficiency:
    1. Fast estimate - if clearly under limit, return as-is
    2. If near/over limit and tiktoken available, use precise counting
    3. Binary search to find truncation point
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        use_tiktoken: Whether to use tiktoken for precise counting
        
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not text:
        return text, False
    
    # Stage 1: Fast estimate
    fast_estimate = estimate_tokens_fast(text)
    
    # If clearly under limit, return as-is
    if fast_estimate < max_tokens * FAST_ESTIMATE_THRESHOLD:
        return text, False
    
    # If clearly over limit, need to truncate
    # First do a rough character-based truncation
    # Assume ~4 chars per token as starting point
    chars_per_token = 4
    estimated_max_chars = max_tokens * chars_per_token
    
    if len(text) <= estimated_max_chars:
        # Text is short enough, but fast estimate was high
        # Use tiktoken if available for precise check
        if use_tiktoken and HAS_TIKTOKEN:
            actual_tokens = estimate_tokens_tiktoken(text)
            if actual_tokens <= max_tokens:
                return text, False
        else:
            # No tiktoken, trust the fast estimate and truncate conservatively
            pass
    
    # Need to truncate - use binary search for efficiency
    if use_tiktoken and HAS_TIKTOKEN:
        # Precise truncation with tiktoken
        low, high = 0, len(text)
        result = ""
        
        while low < high:
            mid = (low + high + 1) // 2
            candidate = text[:mid]
            tokens = estimate_tokens_tiktoken(candidate)
            
            if tokens <= max_tokens:
                result = candidate
                low = mid
            else:
                high = mid - 1
        
        if not result:
            result = text[:estimated_max_chars]
    else:
        # Fast truncation without tiktoken
        # Use conservative estimate: ~3 chars per token for safety
        result = text[:max_tokens * 3]
    
    # Add truncation marker
    truncation_marker = "\n\n[Content truncated due to length...]"
    result = result + truncation_marker
    
    return result, True


class LinkReaderTool(Tool):
    """
    Link reader tool using ByteDance internal bandai_mcp_host service.
    
    This tool reads and extracts raw content from URLs (webpages or PDF documents)
    without any summarization or processing. Supports blacklist constraints to
    prevent access to blocked URLs.
    
    Content is automatically truncated to fit within token limits. Configure via:
    - LINK_READER_MAX_TOKENS: Maximum tokens (default: 100000)
    - LINK_READER_USE_TIKTOKEN: Use tiktoken for precise counting (default: "true")
    """
    
    # Configuration
    max_attempts: int = 1
    
    # Lazy-loaded execute function
    _execute_fn: Any = None
    
    # Token limit configuration (loaded from environment)
    _max_tokens: int | None = None
    _use_tiktoken: bool | None = None
    
    def name(self) -> str:
        return "link_reader"
    
    def _get_max_tokens(self) -> int:
        """Get max tokens from environment variable or default."""
        if self._max_tokens is None:
            env_value = os.environ.get(ENV_MAX_TOKENS)
            if env_value:
                try:
                    self._max_tokens = int(env_value)
                except ValueError:
                    self._max_tokens = DEFAULT_MAX_TOKENS
            else:
                self._max_tokens = DEFAULT_MAX_TOKENS
        return self._max_tokens
    
    def _get_use_tiktoken(self) -> bool:
        """Get tiktoken usage flag from environment variable or default."""
        if self._use_tiktoken is None:
            env_value = os.environ.get(ENV_USE_TIKTOKEN, "true").lower()
            self._use_tiktoken = env_value in ("true", "1", "yes")
        return self._use_tiktoken
    
    def _truncate_content(self, content: str) -> str:
        """
        Truncate content to fit within token limits.
        
        Args:
            content: Raw content to truncate
            
        Returns:
            Truncated content (with marker if truncated)
        """
        max_tokens = self._get_max_tokens()
        use_tiktoken = self._get_use_tiktoken()
        
        truncated, was_truncated = truncate_content_by_tokens(
            content, max_tokens, use_tiktoken
        )
        return truncated
    
    def _get_execute_fn(self) -> Any:
        """Lazy load the bandai_mcp_host LinkReader function."""
        if self._execute_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools
                self._execute_fn = map_tools("LinkReader")
            except ImportError as e:
                raise ImportError(
                    "bytedance.bandai_mcp_host is required for LinkReaderTool. "
                    "Please install it or use a different implementation."
                ) from e
        return self._execute_fn
    
    async def execute(
        self,
        computer: ComputerInterface,
        url: str,
    ) -> str:
        """
        Read and extract raw content from a URL.
        
        Content is automatically truncated to fit within token limits.
        Configure limits via environment variables:
        - LINK_READER_MAX_TOKENS: Maximum tokens (default: 100000)
        - LINK_READER_USE_TIKTOKEN: Use tiktoken for precise counting (default: "true")
        
        Args:
            computer: ComputerInterface (not used, but required by interface)
            url: The target URL to read (web link or PDF)
            
        Returns:
            Raw content of the URL as a string (truncated if too long)
        """
        from bytedance.bandai_mcp_host import ToolResponse as BandaiToolResponse
        
        execute_fn = self._get_execute_fn()
        
        # Try with retries
        response: BandaiToolResponse | None = None
        for _ in range(self.max_attempts):
            response = await execute_fn(url=url)
            if response is not None and response.status.is_succeeded():
                break
        
        if response is not None and response.status.is_succeeded():
            # Truncate content to fit within token limits
            try:
                content = json.loads(response.result)['content']
            except json.JSONDecodeError:
                content = response.result
            return self._truncate_content(content)
        else:
            error_msg = response.result if response is not None else "Unknown error"
            return f"Error reading URL '{url}': {error_msg}"
    
    async def execute_with_constraints(
        self,
        computer: ComputerInterface,
        constraints: dict | None = None,
        url: str = "",
    ) -> str:
        """
        Read URL content with constraints to block access to blacklisted URLs.
        
        Args:
            computer: ComputerInterface (not used, but required by interface)
            constraints: Execution constraints, can include:
                - blocked_search_patterns: dict[str, list[str]] - Regex patterns for URLs to block
                  (e.g., {"url": ['.*github\\.com/django/django.*']})
            url: The target URL to read
        
        Returns:
            Raw content or blocked message
        """
        constraints = constraints or {}
        blocked_patterns = constraints.get('blocked_search_patterns', {})
        
        # Check if URL is blocked
        if blocked_patterns and self._is_url_blocked(url, blocked_patterns):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked because it references a blacklisted resource."
            )
        
        # URL is allowed, proceed with normal execution
        return await self.execute(computer, url=url)
    
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
                "Read and extract the raw content from a specified URL (webpage or PDF document). "
                "Returns the full text content of the page without any summarization or processing. "
                "Use this tool when you need to access the complete content of a web resource."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The target URL to read. Can be a web link or a PDF url."
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            strict=False,
        )


# ==============================================================================
# Debug / Test Entry Point
# ==============================================================================

async def _test_basic_link_reader():
    """Test basic link reader without constraints."""
    print("=" * 60)
    print("Test 1: Basic link reader without constraints")
    print("=" * 60)
    
    tool = LinkReaderTool()
    result = await tool.execute(
        None,  # type: ignore
        url="https://httpbin.org/html"
    )
    print(f"Content length: {len(result)} characters")
    print(f"Preview: {result[:500]}...")
    print()


async def _test_link_reader_with_constraints():
    """Test link reader with blacklist constraints."""
    print("=" * 60)
    print("Test 2: Link reader with blacklist constraints (should be blocked)")
    print("=" * 60)
    
    tool = LinkReaderTool()
    
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
        url="https://github.com/modichirag/GSM-VI"
    )
    print(result)
    print()


async def _test_link_reader_allowed_url():
    """Test link reader with constraints but allowed URL."""
    print("=" * 60)
    print("Test 3: Link reader with constraints (URL allowed)")
    print("=" * 60)
    
    tool = LinkReaderTool()
    
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
        url="https://httpbin.org/html"
    )
    print(f"Content length: {len(result)} characters")
    print(f"Preview: {result[:300]}...")
    print()


async def _test_url_blocking_logic():
    """Test URL blocking logic with various patterns."""
    print("=" * 60)
    print("Test 4: URL blocking logic")
    print("=" * 60)
    
    tool = LinkReaderTool()
    
    blocked_patterns = {
        "url": [
            r'.*github\.com/modichirag/GSM-VI.*',
            r'.*github\.com/another/repo.*',
        ]
    }
    
    test_urls = [
        "https://github.com/modichirag/GSM-VI",
        "https://github.com/modichirag/GSM-VI/issues/123",
        "https://github.com/modichirag/GSM-VI/blob/main/README.md",
        "https://github.com/another/repo/pull/456",
        "https://github.com/safe/repo",  # Should be allowed
        "https://pytorch.org/docs/stable/",  # Should be allowed
        "https://arxiv.org/abs/2405.03553",  # Should be allowed
    ]
    
    for url in test_urls:
        is_blocked = tool._is_url_blocked(url, blocked_patterns)
        status = "BLOCKED" if is_blocked else "ALLOWED"
        print(f"  {status}: {url}")
    print()


async def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LinkReaderTool")
    parser.add_argument(
        "--test",
        choices=["basic", "blocked", "allowed", "logic", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Custom URL to read"
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
        tool = LinkReaderTool()
        if args.blacklist:
            from paperbench.solvers.cus_tools.aweai_mcp.utils import build_blocked_patterns_from_blacklist
            patterns = build_blocked_patterns_from_blacklist(args.blacklist)
            result = await tool.execute_with_constraints(
                None,  # type: ignore
                constraints={"blocked_search_patterns": patterns},
                url=args.url
            )
        else:
            result = await tool.execute(None, url=args.url)  # type: ignore
        print(f"Content length: {len(result)} characters")
        print(result[:2000] if len(result) > 2000 else result)
        return
    
    if args.test == "basic" or args.test == "all":
        await _test_basic_link_reader()
    
    if args.test == "blocked" or args.test == "all":
        await _test_link_reader_with_constraints()
    
    if args.test == "allowed" or args.test == "all":
        await _test_link_reader_allowed_url()
    
    if args.test == "logic" or args.test == "all":
        await _test_url_blocking_logic()


if __name__ == "__main__":
    # 运行所有测试
    # python link_reader.py
    
    # 只测试基本功能
    # python link_reader.py --test basic
    
    # 测试被阻止的 URL
    # python link_reader.py --test blocked
    
    # 测试 URL 阻止逻辑
    # python link_reader.py --test logic
    
    # 自定义 URL 测试
    # python link_reader.py --url "https://example.com"
    
    # 自定义 URL + blacklist 测试
    # python link_reader.py --url "https://github.com/user/repo" --blacklist "https://github.com/user/repo"
    
    import asyncio
    asyncio.run(main())
