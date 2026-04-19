from __future__ import annotations

import copy
import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Unpack, cast

import logid

import openai
import structlog.stdlib
import tenacity
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from openai.types.responses.tool_param import ParseableToolParam
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.oai_responses_turn_completer.completer import (
    OpenAIResponsesTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig, retry_predicate
from pydantic import Field

from paperbench.solvers.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


class TimeTrackingRetryConfig(RetryConfig):
    time_spent_retrying: float = 0.0

    def build(self) -> tenacity.AsyncRetrying:
        # just before sleeping, we'll track how long we'll sleep for
        def _track_wait(rs: tenacity.RetryCallState) -> None:
            wait_time = rs.next_action.sleep if rs.next_action else 0
            self.time_spent_retrying += wait_time

        # and we also want to keep the base log behaviour
        base_log_callback = (
            tenacity.before_sleep_log(logger._logger, logging.WARNING) if logger._logger else None
        )

        # so we compose them together
        def _compose(
            a: Callable[[tenacity.RetryCallState], None] | None,
            b: Callable[[tenacity.RetryCallState], None] | None,
        ) -> Callable[[tenacity.RetryCallState], None] | None:
            if a and b:

                def _both(rs: tenacity.RetryCallState) -> None:
                    a(rs)
                    b(rs)

                return _both
            return a or b

        return tenacity.AsyncRetrying(
            wait=tenacity.wait_random_exponential(min=self.wait_min, max=self.wait_max),
            stop=tenacity.stop_after_delay(self.stop_after),
            retry=retry_predicate,
            before_sleep=_compose(_track_wait, base_log_callback),
            reraise=True,
        )


class BasicAgentTurnCompleterConfig(TurnCompleter.Config, ABC):
    """
    Light wrapper around the base `TurnCompleter.Config` Abstract Base Class.

    Adds two attributes expected by BasicAgent-aware completers:
    - basicagent_tools: Optional list of BasicAgent `Tool` instances that a completer
      should convert/forward into its native tool format (e.g., OpenAI tools).
    - retry_config: Defaults to `TimeTrackingRetryConfig`, enabling collection of the
      total time spent in API retries. Completers can ignore this field if unsupported;
      in that case, retry time will simply not be tracked.

    Implementations should handle `basicagent_tools` when present and, where possible,
    leverage `TimeTrackingRetryConfig` to report retry time back to the solver.

    See `OpenAIResponsesTurnCompleterConfig` in this module for a concrete example
    of how a completer integrates with these fields.
    """

    basicagent_tools: list[Tool] | None = None
    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    @abstractmethod
    def build(self) -> TurnCompleter: ...


class OpenAIResponsesTurnCompleterConfig(
    OpenAIResponsesTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    def build(self) -> OpenAIResponsesTurnCompleter:
        if self.basicagent_tools is not None:
            responses_basic_agent_tools = self._basicagent_to_responses_tools(self.basicagent_tools)

            if self._tools_is_set():
                self.tools: list[ParseableToolParam] = (
                    list(self.tools) + responses_basic_agent_tools
                )
            else:
                self.tools = responses_basic_agent_tools

        return OpenAIResponsesTurnCompleter.Config.build(self)

    def _basicagent_to_responses_tools(self, tools: list[Tool]) -> list[ParseableToolParam]:
        tools_responses: list[ParseableToolParam] = [tool.get_oai_tool_call() for tool in tools]

        return tools_responses

    def _tools_is_set(self) -> bool:
        if isinstance(self.tools, NotGiven):
            return False
        else:
            return len(list(self.tools)) > 0


class OpenAICompletionsTurnCompleterConfig(
    OpenAICompletionsTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    """
    BasicAgent-aware config for OpenAI Chat Completions API.
    Use this when your API endpoint only supports Chat Completions API
    (e.g., a custom OpenAI-compatible endpoint or Azure OpenAI).
    """

    def build(self) -> OpenAICompletionsTurnCompleter:
        if self.basicagent_tools is not None:
            completions_basic_agent_tools = self._basicagent_to_completions_tools(
                self.basicagent_tools
            )

            if self._tools_is_set():
                self.tools: list[ChatCompletionToolParam] = (
                    list(self.tools) + completions_basic_agent_tools
                )
            else:
                self.tools = completions_basic_agent_tools

        return OpenAICompletionsTurnCompleter.Config.build(self)

    def _basicagent_to_completions_tools(
        self, tools: list[Tool]
    ) -> list[ChatCompletionToolParam]:
        """
        Convert BasicAgent tools to Chat Completions API format.

        Responses API format (flat):
            {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Chat Completions API format (nested):
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        tools_completions: list[ChatCompletionToolParam] = []
        for tool in tools:
            # Get the Responses API format tool
            responses_tool = tool.get_oai_tool_call()
            # Convert to Chat Completions API format
            function_def: dict[str, Any] = {
                "name": responses_tool["name"],
            }
            # Add parameters if present
            if responses_tool.get("parameters") is not None:
                function_def["parameters"] = responses_tool["parameters"]
            # Add optional fields if present
            if "description" in responses_tool and responses_tool["description"]:
                function_def["description"] = responses_tool["description"]
            if "strict" in responses_tool and responses_tool["strict"] is not None:
                function_def["strict"] = responses_tool["strict"]

            completions_tool = cast(
                ChatCompletionToolParam,
                {"type": "function", "function": function_def},
            )
            tools_completions.append(completions_tool)

        return tools_completions

    def _tools_is_set(self) -> bool:
        if isinstance(self.tools, NotGiven):
            return False
        else:
            return len(list(self.tools)) > 0


# =============================================================================
# Azure OpenAI Chat Completions API Support
# =============================================================================


class AzureOpenAICompletionsTurnCompleter(OpenAICompletionsTurnCompleter):
    """
    OpenAI Chat Completions API completer using Azure OpenAI client.

    This is designed for models that do NOT support the Responses API,
    such as GLM, DeepSeek, and other third-party models exposed via
    Azure-compatible endpoints.

    Uses chat.completions.create() instead of chat.completions.parse()
    to avoid the strict=True requirement for tool definitions.

    Required environment variables:
        AZURE_OPENAI_ENDPOINT: e.g., "https://your-endpoint.openai.azure.com/"
        AZURE_OPENAI_API_KEY: Your API key
        OPENAI_API_VERSION: e.g., "2024-02-01"
    """

    @functools.cached_property
    def _client(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI()

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        """
        Override to use create() instead of parse() to avoid strict tool requirement.

        The parse() method requires all tools to have strict=True, but many tools
        (like bash) are defined with strict=False. Using create() avoids this issue.
        """
        async for attempt in self.retry_config.build():
            with attempt:
                # Build kwargs, only including non-NotGiven values
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": conversation,
                    "extra_headers": {"X-TT-LOGID": logid.generate_v2()},
                }
                if not isinstance(self.temperature, NotGiven):
                    kwargs["temperature"] = self.temperature
                if not isinstance(self.max_tokens, NotGiven):
                    kwargs["max_tokens"] = self.max_tokens
                if not isinstance(self.top_p, NotGiven):
                    kwargs["top_p"] = self.top_p
                if not isinstance(self.tools, NotGiven):
                    kwargs["tools"] = self.tools
                if not isinstance(self.tool_choice, NotGiven):
                    kwargs["tool_choice"] = self.tool_choice
                # Note: response_format for structured output is not commonly used with
                # GLM/DeepSeek models, but can be added here if needed in the future.

                completion = await self._client.chat.completions.create(**kwargs)
        assert isinstance(completion, ChatCompletion)
        return OpenAICompletionsTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[completion.choices[0].message],
            usage=completion.usage,
        )


class AzureOpenAICompletionsTurnCompleterConfig(
    OpenAICompletionsTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    """
    BasicAgent-aware config for Azure OpenAI Chat Completions API.
    
    Use this for models that do NOT support the Responses API, such as:
        - GLM series (glm-4, etc.)
        - DeepSeek series
        - Other third-party models via Azure-compatible endpoints
    
    Example environment setup:
        AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
        AZURE_OPENAI_API_KEY="your-api-key"
        OPENAI_API_VERSION="2024-02-01"
    
    This supports:
        - Chat Completions API format
        - Function calling / tool use
        - Structured output (response_format)
    """

    def build(self) -> AzureOpenAICompletionsTurnCompleter:
        if self.basicagent_tools is not None:
            completions_basic_agent_tools = self._basicagent_to_completions_tools(
                self.basicagent_tools
            )

            if self._tools_is_set():
                self.tools: list[ChatCompletionToolParam] = (
                    list(self.tools) + completions_basic_agent_tools
                )
            else:
                self.tools = completions_basic_agent_tools

        # Build using the parameters directly for Azure completer
        return AzureOpenAICompletionsTurnCompleter(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            response_format=self.response_format,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=self.tools,
            tool_choice=self.tool_choice,
            safety_identifier=self.safety_identifier,
            retry_config=self.retry_config,
            context_window_override=self.context_window_override,
        )

    def _basicagent_to_completions_tools(
        self, tools: list[Tool]
    ) -> list[ChatCompletionToolParam]:
        """
        Convert BasicAgent tools to Chat Completions API format.

        Responses API format (flat):
            {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Chat Completions API format (nested):
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        tools_completions: list[ChatCompletionToolParam] = []
        for tool in tools:
            # Get the Responses API format tool
            responses_tool = tool.get_oai_tool_call()
            # Convert to Chat Completions API format
            function_def: dict[str, Any] = {
                "name": responses_tool["name"],
            }
            # Add parameters if present
            if responses_tool.get("parameters") is not None:
                function_def["parameters"] = responses_tool["parameters"]
            # Add optional fields if present
            if "description" in responses_tool and responses_tool["description"]:
                function_def["description"] = responses_tool["description"]
            if "strict" in responses_tool and responses_tool["strict"] is not None:
                function_def["strict"] = responses_tool["strict"]

            completions_tool = cast(
                ChatCompletionToolParam,
                {"type": "function", "function": function_def},
            )
            tools_completions.append(completions_tool)

        return tools_completions

    def _tools_is_set(self) -> bool:
        if isinstance(self.tools, NotGiven):
            return False
        else:
            return len(list(self.tools)) > 0


# =============================================================================
# Azure OpenAI Responses API Support
# =============================================================================


class AzureOpenAIResponsesTurnCompleter(OpenAIResponsesTurnCompleter):
    """
    OpenAI Responses API completer using Azure OpenAI client.

    This is designed for Azure-compatible Responses API endpoints that require
    the Azure OpenAI client format.

    Required environment variables:
        AZURE_OPENAI_ENDPOINT: e.g., "https://your-azure-endpoint.example.com/"
        AZURE_OPENAI_API_KEY: Your API key
        OPENAI_API_VERSION: e.g., "2024-02-01"
    """

    @functools.cached_property
    def _client(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI()


class AzureOpenAIResponsesTurnCompleterConfig(
    OpenAIResponsesTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    """
    BasicAgent-aware config for Azure OpenAI Responses API.

    Use this for Azure-compatible Responses API endpoints:
        AZURE_OPENAI_ENDPOINT="https://your-azure-endpoint.example.com/"
    
    This supports:
        - Full Responses API features
        - web_search built-in tool
        - reasoning configuration (effort, summary)
    """

    def build(self) -> AzureOpenAIResponsesTurnCompleter:
        if self.basicagent_tools is not None:
            responses_basic_agent_tools = self._basicagent_to_responses_tools(self.basicagent_tools)

            if self._tools_is_set():
                self.tools: list[ParseableToolParam] = (
                    list(self.tools) + responses_basic_agent_tools
                )
            else:
                self.tools = responses_basic_agent_tools

        # Build the base config but return our Azure version
        base_completer = OpenAIResponsesTurnCompleter.Config.build(self)
        return AzureOpenAIResponsesTurnCompleter(
            model=base_completer.model,
            reasoning=base_completer.reasoning,
            text_format=base_completer.text_format,
            tools=base_completer.tools,
            temperature=base_completer.temperature,
            max_output_tokens=base_completer.max_output_tokens,
            top_p=base_completer.top_p,
            safety_identifier=base_completer.safety_identifier,
            retry_config=base_completer.retry_config,
        )

    def _basicagent_to_responses_tools(self, tools: list[Tool]) -> list[ParseableToolParam]:
        tools_responses: list[ParseableToolParam] = [tool.get_oai_tool_call() for tool in tools]
        return tools_responses

    def _tools_is_set(self) -> bool:
        if isinstance(self.tools, NotGiven):
            return False
        else:
            return len(list(self.tools)) > 0


# =============================================================================
# Gemini OpenAI-compatible Chat Completions API Support
# =============================================================================


class GeminiCompletionsTurnCompleter(OpenAICompletionsTurnCompleter):
    """
    OpenAI Chat Completions API completer for Google Gemini models.

    Uses Gemini's OpenAI-compatible endpoint via the standard OpenAI client.
    Handles Gemini-specific features:
      - thought_signature (extra_content) preservation across turns
      - reasoning_effort via extra_body
      - Uses .create() instead of .parse() (no strict tool requirement)

    Required environment variables:
        OPENAI_BASE_URL: "https://generativelanguage.googleapis.com/v1beta/openai/"
        OPENAI_API_KEY: Your Google API key
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Maps tool_call_id → extra_content dict for Gemini thought_signature.
        # Populated after each API response, injected before the next request.
        self._extra_content_map: dict[str, dict[str, Any]] = {}

    @functools.cached_property
    def _client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(timeout=1200.0)

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        """
        Gemini-aware completion with thought_signature preservation.

        Before sending: injects stored extra_content into assistant tool_call dicts.
        After receiving: extracts extra_content from response and stores for next turn.
        """
        patched_conversation = self._inject_extra_content(conversation)
        patched_conversation = self._sanitize_for_gemini(patched_conversation)

        async for attempt in self.retry_config.build():
            with attempt:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": patched_conversation,
                }
                if not isinstance(self.temperature, NotGiven):
                    kwargs["temperature"] = self.temperature
                if not isinstance(self.max_tokens, NotGiven):
                    kwargs["max_tokens"] = self.max_tokens
                if not isinstance(self.top_p, NotGiven):
                    kwargs["top_p"] = self.top_p
                if not isinstance(self.tools, NotGiven):
                    kwargs["tools"] = self._sanitize_tools_for_gemini(self.tools)
                if not isinstance(self.tool_choice, NotGiven):
                    kwargs["tool_choice"] = self.tool_choice

                # Gemini: reasoning_effort via extra_body
                extra_body: dict[str, Any] = {}
                if not isinstance(self.reasoning_effort, NotGiven) and self.reasoning_effort:
                    extra_body["reasoning_effort"] = self.reasoning_effort
                if extra_body:
                    kwargs["extra_body"] = extra_body

                completion = await self._client.chat.completions.create(**kwargs)

        assert isinstance(completion, ChatCompletion)
        message = completion.choices[0].message

        # Extract and store extra_content (thought_signature) from tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                extra = getattr(tc, "extra_content", None)
                if isinstance(extra, dict):
                    self._extra_content_map[tc.id] = extra

        return OpenAICompletionsTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[message],
            usage=completion.usage,
        )

    def _inject_extra_content(self, conversation: list[Any]) -> list[Any]:
        """Inject stored extra_content into assistant messages' tool_call dicts."""
        if not self._extra_content_map:
            return conversation

        patched = []
        for msg in conversation:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                patched.append(msg)
                continue

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                patched.append(msg)
                continue

            needs_patch = any(
                tc.get("id") in self._extra_content_map and "extra_content" not in tc
                for tc in tool_calls
                if isinstance(tc, dict)
            )
            if not needs_patch:
                patched.append(msg)
                continue

            msg_copy = copy.deepcopy(msg)
            for tc in msg_copy["tool_calls"]:
                if isinstance(tc, dict) and tc.get("id") in self._extra_content_map:
                    tc["extra_content"] = self._extra_content_map[tc["id"]]
            patched.append(msg_copy)

        return patched

    @staticmethod
    def _sanitize_for_gemini(conversation: list[Any]) -> list[Any]:
        """Sanitize messages for Gemini's OpenAI-compatible endpoint.

        Gemini rejects:
        - `"content": null` in assistant messages (expects string)
        - `"audio": null`, `"refusal": null` etc. (expects struct, not null)
        Strip any top-level null values from message dicts.
        """
        sanitized = []
        for msg in conversation:
            if isinstance(msg, dict):
                msg = {k: v for k, v in msg.items() if v is not None}
                # Ensure assistant messages always have content
                if msg.get("role") == "assistant" and "content" not in msg:
                    msg["content"] = ""
            sanitized.append(msg)
        return sanitized

    def _sanitize_tools_for_gemini(self, tools: Any) -> Any:
        """Remove fields Gemini doesn't support from tool definitions."""
        if isinstance(tools, NotGiven) or not tools:
            return tools
        sanitized = []
        for tool in tools:
            if not isinstance(tool, dict):
                sanitized.append(tool)
                continue
            tool = copy.deepcopy(tool)
            func = tool.get("function", {})
            # Gemini doesn't support 'strict' field
            func.pop("strict", None)
            sanitized.append(tool)
        return sanitized


class GeminiCompletionsTurnCompleterConfig(
    OpenAICompletionsTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    """
    BasicAgent-aware config for Google Gemini via OpenAI-compatible endpoint.

    Use this for Gemini models (gemini-3-flash, gemini-3.1-pro, etc.)

    Example environment setup:
        OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
        OPENAI_API_KEY="your-google-api-key"

    Features:
        - Chat Completions API via Gemini's OpenAI-compatible endpoint
        - thought_signature (extra_content) automatic preservation
        - reasoning_effort via extra_body
        - No strict tool definition requirement
    """

    def build(self) -> GeminiCompletionsTurnCompleter:
        if self.basicagent_tools is not None:
            completions_basic_agent_tools = self._basicagent_to_completions_tools(
                self.basicagent_tools
            )

            if self._tools_is_set():
                self.tools: list[ChatCompletionToolParam] = (
                    list(self.tools) + completions_basic_agent_tools
                )
            else:
                self.tools = completions_basic_agent_tools

        return GeminiCompletionsTurnCompleter(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            response_format=self.response_format,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=self.tools,
            tool_choice=self.tool_choice,
            safety_identifier=self.safety_identifier,
            retry_config=self.retry_config,
            context_window_override=self.context_window_override,
        )

    def _basicagent_to_completions_tools(
        self, tools: list[Tool]
    ) -> list[ChatCompletionToolParam]:
        tools_completions: list[ChatCompletionToolParam] = []
        for tool in tools:
            responses_tool = tool.get_oai_tool_call()
            function_def: dict[str, Any] = {
                "name": responses_tool["name"],
            }
            if responses_tool.get("parameters") is not None:
                function_def["parameters"] = responses_tool["parameters"]
            if "description" in responses_tool and responses_tool["description"]:
                function_def["description"] = responses_tool["description"]
            if "strict" in responses_tool and responses_tool["strict"] is not None:
                function_def["strict"] = responses_tool["strict"]

            completions_tool = cast(
                ChatCompletionToolParam,
                {"type": "function", "function": function_def},
            )
            tools_completions.append(completions_tool)

        return tools_completions

    def _tools_is_set(self) -> bool:
        if isinstance(self.tools, NotGiven):
            return False
        else:
            return len(list(self.tools)) > 0
