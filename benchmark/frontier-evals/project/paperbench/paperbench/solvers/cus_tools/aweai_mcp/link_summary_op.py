"""
Link summary tool with configurable LLM backend (OpenAI-compatible).

This tool uses LinkReaderTool to fetch URL content, then summarizes it using
a configurable LLM model specified via environment variables and YAML config.

Usage:
    # Set environment variables
    export LINK_SUMMARY_MODEL="deepseek_v32_chat"
    export LINK_SUMMARY_CONFIG_PATH="/path/to/config.yaml"
    
    from paperbench.solvers.cus_tools.aweai_mcp.link_summary_op import LinkSummaryOpTool
    
    tool = LinkSummaryOpTool()
    result = await tool.execute(computer, url="https://example.com", goal="Summarize")
"""

import os
import re
from pathlib import Path
from typing import Any

import logging

import logid
import yaml
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.cus_tools.aweai_mcp.link_reader import LinkReaderTool

# Try to import Ark client for volcengine
try:
    from volcenginesdkarkruntime import AsyncArk
    HAS_ARK = True
except ImportError:
    HAS_ARK = False
    AsyncArk = None


# Environment variable names
ENV_MODEL_NAME = "LINK_SUMMARY_MODEL"
ENV_CONFIG_PATH = "LINK_SUMMARY_CONFIG_PATH"

# Default config path (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"

# Blocked content marker
BLOCKED_CONTENT_MARKER = "ACCESS DENIED:"

# System prompt for summarization
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


def load_llm_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load LLM configurations from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses ENV_CONFIG_PATH or default.
        
    Returns:
        dict with 'agent_llm_configs' key containing model configurations
    """
    if config_path is None:
        config_path = os.environ.get(ENV_CONFIG_PATH)
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"LLM config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_client(client_type: str, client_args: dict[str, Any]) -> AsyncOpenAI | AsyncAzureOpenAI | Any:
    """
    Create an async OpenAI-compatible client based on client_type.
    
    Supported client types:
    - OpenAI: Standard OpenAI API (AsyncOpenAI)
    - AzureOpenAI: Azure OpenAI service (AsyncAzureOpenAI)
    - Ark: Volcengine Ark API (AsyncArk from volcenginesdkarkruntime)
    
    Args:
        client_type: Type of client ('OpenAI', 'AzureOpenAI', 'Ark')
        client_args: Arguments to pass to the client constructor
        
    Returns:
        AsyncOpenAI, AsyncAzureOpenAI, or AsyncArk client
        
    Raises:
        ValueError: If client_type is not supported or Ark is not installed
    """
    if client_type == 'AzureOpenAI':
        return AsyncAzureOpenAI(**client_args)
    elif client_type == 'Ark':
        if not HAS_ARK or AsyncArk is None:
            raise ValueError(
                "Ark client requires volcenginesdkarkruntime package. "
                "Please install it: pip install volcenginesdkarkruntime"
            )
        return AsyncArk(**client_args)
    elif client_type == 'OpenAI':
        return AsyncOpenAI(**client_args)
    else:
        raise ValueError(
            f"Unsupported client_type: '{client_type}'. "
            f"Supported types: 'OpenAI', 'AzureOpenAI', 'Ark'"
        )


class LinkSummaryOpTool(Tool):
    """
    Link summary tool using configurable LLM backend.
    
    This tool:
    1. Uses LinkReaderTool to fetch URL content
    2. Summarizes the content using a configurable LLM model
    
    Configuration is loaded from a YAML file, with the model name
    specified via environment variable.
    """
    
    # Configuration
    max_attempts: int = 3
    system_prompt: str = LINK_SUMMARY_PROMPT
    
    # Lazy-loaded components
    _link_reader: LinkReaderTool | None = None
    _client: AsyncOpenAI | AsyncAzureOpenAI | None = None
    _request_args: dict[str, Any] | None = None
    _config_loaded: bool = False
    
    def name(self) -> str:
        return "link_summary"
    
    def _ensure_config_loaded(self) -> None:
        """Load configuration from YAML file if not already loaded."""
        if self._config_loaded:
            return
        
        # Get model name from environment
        model_name = os.environ.get(ENV_MODEL_NAME)
        if not model_name:
            raise ValueError(
                f"Environment variable {ENV_MODEL_NAME} not set. "
                f"Please set it to a model name defined in the config file."
            )
        
        # Load config
        config = load_llm_config()
        llm_configs = config.get('agent_llm_configs', {})
        
        if model_name not in llm_configs:
            available = list(llm_configs.keys())
            raise ValueError(
                f"Model '{model_name}' not found in config. "
                f"Available models: {available}"
            )
        
        model_config = llm_configs[model_name]
        client_type = model_config.get('client_type', 'OpenAI')
        client_args = model_config.get('client_args', {})
        self._request_args = model_config.get('request_args', {})
        
        # Create client
        self._client = create_client(client_type, client_args)
        self._config_loaded = True
    
    def _get_link_reader(self) -> LinkReaderTool:
        """Get or create LinkReaderTool instance."""
        if self._link_reader is None:
            self._link_reader = LinkReaderTool()
        return self._link_reader
    
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
            Summarized information as a string
        """
        # Ensure config is loaded
        self._ensure_config_loaded()
        
        # Step 1: Fetch URL content using LinkReaderTool
        link_reader = self._get_link_reader()
        content = await link_reader.execute(computer, url=url)
        
        # Check if content is blocked
        if content.startswith(BLOCKED_CONTENT_MARKER):
            return content
        
        # Check if content fetch failed
        if content.startswith("Error reading URL"):
            return content
        
        # Step 2: Summarize content using LLM
        return await self._summarize_content(content, url, goal)
    
    async def _summarize_content(self, content: str, url: str, goal: str) -> str:
        """
        Summarize the fetched content using the configured LLM.
        
        Args:
            content: Raw content from the URL (already truncated by LinkReaderTool if needed)
            url: Original URL (for context)
            goal: User's goal for information extraction
            
        Returns:
            Summarized content
        """
        # Note: Content truncation is now handled by LinkReaderTool
        
        # Build user message
        user_message = f"""## URL
{url}

## Goal
{goal}

## Content
{content}
"""
        
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Request args
        request_args = dict(self._request_args or {})
        
        # Ensure client is initialized
        if self._client is None:
            return "Error: LLM client not initialized"
        
        client = self._client
        
        # Try with retries
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = await client.chat.completions.create(
                    messages=messages,  # type: ignore
                    extra_headers={"X-TT-LOGID": logid.generate_v2()},
                    **request_args,
                )
                
                # Extract response content
                summary_content = response.choices[0].message.content if response.choices and response.choices[0].message.content else "Error: Empty response from model"
                return f"=== Summary Content from {url} ===\n{summary_content}"
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_attempts - 1:
                    # Wait before retry (exponential backoff)
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                continue
        
        return f"=== Summary Content from {url} ===\nError summarizing content after {self.max_attempts} attempts: {last_error}"
    
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
            url: The target URL to read
            goal: The specific instruction for information extraction
        
        Returns:
            Summarized information or blocked message
        """
        # Ensure config is loaded
        self._ensure_config_loaded()
        
        constraints = constraints or {}
        blocked_patterns = constraints.get('blocked_search_patterns', {})
        
        # Check if URL is blocked (pre-check before fetching)
        if blocked_patterns and self._is_url_blocked(url, blocked_patterns):
            return f"ACCESS DENIED: The URL '{url}' is blocked because it references a blacklisted resource."
        
        # Step 1: Fetch URL content using LinkReaderTool with constraints
        link_reader = self._get_link_reader()
        content = await link_reader.execute_with_constraints(
            computer, constraints=constraints, url=url
        )
        
        # Check if content is blocked (in case link_reader blocked it)
        if content.startswith(BLOCKED_CONTENT_MARKER):
            return content
        
        # Check if content fetch failed
        if content.startswith("Error reading URL"):
            return content
        
        # Step 2: Summarize content using LLM
        return await self._summarize_content(content, url, goal)
    
    def _is_url_blocked(self, url: str, blocked_patterns: dict[str, list[str]]) -> bool:
        """Check if a URL matches any blocked pattern."""
        if not url or not blocked_patterns:
            return False
        
        url_patterns = blocked_patterns.get('url', [])
        for pattern in url_patterns:
            try:
                if re.match(pattern, url, re.IGNORECASE):
                    return True
            except re.error:
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

async def _test_basic():
    """Test basic link summary."""
    print("=" * 60)
    print("Test 1: Basic link summary")
    print("=" * 60)
    
    tool = LinkSummaryOpTool()
    result = await tool.execute(
        None,  # type: ignore
        url="https://httpbin.org/html",
        goal="Summarize the content of this page"
    )
    print(result)
    print()


async def _test_with_constraints():
    """Test with blacklist constraints."""
    print("=" * 60)
    print("Test 2: Link summary with blacklist (should be blocked)")
    print("=" * 60)
    
    tool = LinkSummaryOpTool()
    
    blocked_patterns = {
        "url": [r'.*github\.com/modichirag/GSM-VI.*']
    }
    
    result = await tool.execute_with_constraints(
        None,  # type: ignore
        constraints={"blocked_search_patterns": blocked_patterns},
        url="https://github.com/modichirag/GSM-VI",
        goal="Get installation instructions"
    )
    print(result)
    print()


async def _test_real_url():
    """Test with a real documentation URL."""
    print("=" * 60)
    print("Test 3: Real documentation URL")
    print("=" * 60)
    
    tool = LinkSummaryOpTool()
    result = await tool.execute(
        None,  # type: ignore
        url="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html",
        goal="Extract the API signature, parameters, and a usage example for torch.nn.Linear"
    )
    print(result)
    print()


async def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LinkSummaryOpTool")
    parser.add_argument(
        "--test",
        choices=["basic", "blocked", "real", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument("--url", type=str, help="Custom URL to summarize")
    parser.add_argument("--goal", type=str, default="Summarize the main content")
    parser.add_argument("--blacklist", type=str, nargs="+", help="Blacklist URL patterns")
    
    args = parser.parse_args()
    
    # Check environment variables
    if not os.environ.get(ENV_MODEL_NAME):
        print(f"Warning: {ENV_MODEL_NAME} not set. Using default model if available.")
    
    if args.url:
        tool = LinkSummaryOpTool()
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
        await _test_basic()
    
    if args.test == "blocked" or args.test == "all":
        await _test_with_constraints()
    
    if args.test == "real" or args.test == "all":
        await _test_real_url()


if __name__ == "__main__":
    # 使用前请设置环境变量:
    # export LINK_SUMMARY_MODEL="deepseek_v32_chat"
    # export LINK_SUMMARY_CONFIG_PATH="/path/to/config.yaml"
    #
    # 运行测试:
    # python link_summary_op.py --test basic
    # python link_summary_op.py --url "https://example.com" --goal "Summarize"
    
    import asyncio
    asyncio.run(main())
