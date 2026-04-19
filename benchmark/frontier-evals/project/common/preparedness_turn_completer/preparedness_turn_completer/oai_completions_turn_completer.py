from __future__ import annotations

import functools
from typing import Any, Literal, Unpack

import logid
import openai
import structlog
import tiktoken
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import (
    RetryConfig,
    get_model_context_window_length,
    should_send_safety_identifier,
    warn_about_non_empty_params,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = structlog.stdlib.get_logger(component=__name__)


class OpenAICompletionsTurnCompleter(TurnCompleter):
    def __init__(
        self,
        model: str,
        reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None | NotGiven = NOT_GIVEN,
        response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        safety_identifier: str | None | NotGiven = NOT_GIVEN,
        retry_config: RetryConfig | None = None,
        context_window_override: int | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.tools = tools
        self.tool_choice = tool_choice
        self.safety_identifier = safety_identifier
        self.encoding_name: str
        self.retry_config = retry_config or RetryConfig()
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            # Fallback to o200k_base
            logger.warning(f"Model {model} not found in tiktoken, using o200k_base")
            self.encoding_name = "o200k_base"
        # Allow overriding context window length for specific use cases (e.g., judge vs solver)
        self.n_ctx: int = context_window_override if context_window_override is not None else get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        """
        Completion configuration. Non-exhaustive.
        Add more configuration options as needed, in a backwards-compatible way.
        """

        # needed for NotGiven type hint
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={NotGiven: lambda v: "NOT_GIVEN"},
        )

        model: str
        reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None | NotGiven = NOT_GIVEN
        response_format: type[BaseModel] | NotGiven = NOT_GIVEN
        temperature: float | None | NotGiven = NOT_GIVEN
        max_tokens: int | None | NotGiven = NOT_GIVEN
        top_p: float | None | NotGiven = NOT_GIVEN
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
        safety_identifier: str | None | NotGiven = NOT_GIVEN
        retry_config: RetryConfig = Field(default_factory=RetryConfig)
        # Override context window length for specific use cases (e.g., ByteDance proxy has 272k limit)
        context_window_override: int | None = None

        def build(self) -> OpenAICompletionsTurnCompleter:
            return OpenAICompletionsTurnCompleter(
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

        @field_validator("*", mode="before")
        @classmethod
        def _decode_not_given(cls: type[OpenAICompletionsTurnCompleter.Config], v: Any) -> Any:
            """
            Turn the string "NOT_GIVEN" back into our sentinel before validation.
            """
            if v == "NOT_GIVEN":
                return NOT_GIVEN
            return v

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncClient:
        return openai.AsyncClient(timeout=1200.0)  # 20 minutes

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        warn_about_non_empty_params(self, **params)

        # Newer OpenAI models (o1/o3/gpt-5.x) require max_completion_tokens instead of max_tokens
        _use_max_completion_tokens = any(
            self.model.startswith(prefix)
            for prefix in ("o1", "o3", "o4", "gpt-5", "gpt-6")
        )

        async for attempt in self.retry_config.build():
            with attempt:
                common_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": conversation,
                    "reasoning_effort": self.reasoning_effort,
                    "response_format": self.response_format,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "tools": self.tools,
                    "tool_choice": self.tool_choice,
                    "extra_headers": {"X-TT-LOGID": logid.generate_v2()},
                }
                if (
                    not isinstance(self.safety_identifier, NotGiven)
                    and self.safety_identifier is not None
                    and should_send_safety_identifier()
                ):
                    common_kwargs["safety_identifier"] = self.safety_identifier
                if _use_max_completion_tokens:
                    completion = await self._client.chat.completions.parse(
                        max_completion_tokens=self.max_tokens if not isinstance(self.max_tokens, NotGiven) else NOT_GIVEN,
                        **common_kwargs,
                    )
                else:
                    completion = await self._client.chat.completions.parse(
                        max_tokens=self.max_tokens,
                        **common_kwargs,
                    )
        assert isinstance(completion, ChatCompletion)
        return OpenAICompletionsTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[completion.choices[0].message],
            usage=completion.usage,
        )
