from __future__ import annotations

import logging
import os
from typing import Unpack

import openai
import structlog.stdlib
import tenacity
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.type_helpers import (
    is_chat_completion_assistant_message_param,
    is_chat_completion_dev_message_param,
    is_chat_completion_function_message_param,
    is_chat_completion_sys_message_param,
    is_chat_completion_tool_message_param,
    is_chat_completion_user_message_param,
)
from pydantic import BaseModel

logger = structlog.stdlib.get_logger(component=__name__)

CONTEXT_WINDOW_LENGTHS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-2024-11-20": 128_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "o1": 200_000,
    "o1-2024-12-17": 200_000,
    "o3": 200_000,
    "o3-mini-2025-01-31": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "o4-mini-deep-research-2025-06-26": 200_000,
    "o4-mini-deep-research": 200_000,
    "o3-deep-research-2025-06-26": 200_000,
    "o3-deep-research": 200_000,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1": 1_047_576,
    "o1-preview": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5-2025-08-07": 400_000,
    "gpt-5-mini-2025-08-07": 400_000,
    "gpt-5-nano-2025-08-07": 400_000,
    "gpt-5-codex": 400_000,
    "gpt-5-pro-2025-10-06": 400_000,
    "gpt-5-pro": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.2-2025-12-11": 400_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-2026-02-27": 1_050_000,
    "glm-4.7": 200_000,
    "glm-5": 200_000,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
}


def get_model_context_window_length(model: str) -> int:
    if model not in CONTEXT_WINDOW_LENGTHS:
        raise ValueError(f"Model {model} not found in context window lengths")
    return CONTEXT_WINDOW_LENGTHS[model]


def should_send_safety_identifier() -> bool:
    """
    Only send `safety_identifier` when using the official OpenAI API path.

    OpenAI-compatible providers often reject unknown request fields. In this codebase,
    those providers are typically selected by setting a custom `OPENAI_BASE_URL` or by
    using Azure-style environment variables.
    """
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    return not openai_base_url and not azure_endpoint


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def is_retryable_bad_request(exc: BaseException) -> bool:
    # retry BadRequestErrors that are actually transient network issues
    if isinstance(exc, openai.BadRequestError):
        err_str = str(exc)
        logger.warning(f"[is_retryable_bad_request] Bad request exception: {err_str}")
        return (
            "Timeout while downloading" in err_str
            or "connection reset by peer" in err_str
            or "TLS handshake timeout" in err_str
        )
    return False


retry_predicate = tenacity.retry_if_exception_type(
    OPENAI_TIMEOUT_EXCEPTIONS
) | tenacity.retry_if_exception(is_retryable_bad_request)


class RetryConfig(BaseModel):
    wait_min: float = 1
    wait_max: float = 300
    stop_after: float = 3600 * 2

    def build(self: RetryConfig) -> tenacity.AsyncRetrying:
        return tenacity.AsyncRetrying(
            wait=tenacity.wait_random_exponential(min=self.wait_min, max=self.wait_max),
            stop=tenacity.stop_after_delay(self.stop_after),
            retry=retry_predicate,
            before_sleep=tenacity.before_sleep_log(logger._logger, logging.WARNING)
            if logger._logger
            else None,
            reraise=True,
        )


def text_from_completion(message: ChatCompletionMessageParam) -> str | list[str]:
    """
    Gets the text content from a chat completion message.
    If a particular message content(part) does not have text context, an empty string is
    returned for that message content(part).
    Useful for any truncation operations that require the text content of a message.
    """
    if "content" not in message or message["content"] is None:
        return ""
    elif (
        is_chat_completion_sys_message_param(message)
        or is_chat_completion_dev_message_param(message)
        or is_chat_completion_tool_message_param(message)
        or is_chat_completion_function_message_param(message)
    ):
        return (
            message["content"]
            if isinstance(message["content"], str)
            else [part["text"] for part in message["content"]]
        )
    elif is_chat_completion_user_message_param(message):
        return (
            message["content"]
            if isinstance(message["content"], str)
            else [part["text"] if part["type"] == "text" else "" for part in message["content"]]
        )
    elif is_chat_completion_assistant_message_param(message):
        return (
            message["content"]
            if isinstance(message["content"], str)
            else [
                part["text"] if part["type"] == "text" else part["refusal"]
                for part in message["content"]
            ]
        )
    else:
        raise ValueError(f"Unknown message role: {message['role']}")


def warn_about_non_empty_params(
    completer: TurnCompleter, **params: Unpack[TurnCompleter.Params]
) -> None:
    """
    We specifically don't want to use `TurnCompleter.Params` in `async_completion`
    because the base (non-abstract) `TurnCompleter.Params` is empty,
    and subclassing it will introduce conflicts or branching in the API.
    """
    if params and os.getenv("TC_DISABLE_EMPTY_PARAMS_WARNING", "false").lower() != "true":
        logger.warning(
            f"{completer.__class__} received params, but they are not used in async_completion."
            " You may disable this warning by setting the environment variable"
            " `TC_DISABLE_CONVERTER_WARNINGS` to `true`.",
            params=params,
        )
