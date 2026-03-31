"""
LLM Client — supports both Chat Completions API and Responses API.

Mirrors PaperBench's dual-completer architecture:
- ``CompletionsLLMClient``: Chat Completions API (``chat.completions.create``)
  Used for GLM, DeepSeek, and any model without Responses API support.
- ``ResponsesLLMClient``: Responses API (``responses.create``)
  Used for GPT with web_search_preview and reasoning support.

Both clients share a common interface so the orchestrator and subagents
are API-agnostic — the same message list works with either backend.

Preserved patterns from PaperBench:
- Retry with wait_random_exponential(min=1, max=300), stop_after_delay(2h)
- Only retry transient errors: RateLimitError, APIConnectionError,
  APITimeoutError, InternalServerError, and select BadRequestErrors
  (download timeout, connection reset, TLS handshake timeout)
- ContentPolicyError and ContextLengthError are NOT retried
- Separate ``total_retry_time`` accumulation so callers can exclude
  retry-wait from the agent's effective time budget (use_real_time_limit)
- Malformed tool-call JSON → skip gracefully instead of crashing
- Token usage tracking per call and cumulative
"""

from __future__ import annotations

import json
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import logid
import structlog
from openai import (
    OpenAI, AzureOpenAI, BadRequestError, PermissionDeniedError,
    RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
)
from openai.types.shared_params.reasoning import Reasoning

logger = structlog.stdlib.get_logger(component=__name__)

# Transient errors that warrant automatic retry (mirrors PaperBench's retry_predicate)
_RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

# Transient BadRequest phrases (PaperBench's is_retryable_bad_request)
_RETRYABLE_BADREQUEST_PHRASES = (
    "Timeout while downloading",
    "connection reset by peer",
    "TLS handshake timeout",
)

# Retry timing parameters aligned with PaperBench's RetryConfig defaults
_RETRY_WAIT_MIN = 1       # seconds
_RETRY_WAIT_MAX = 300     # seconds (5 min cap)
_RETRY_STOP_AFTER = 7200  # seconds (2 hours total retry budget)


# ====================================================================== #
# Shared data structures
# ====================================================================== #


class ContextLengthError(Exception):
    """Raised when the model's context window is exceeded.

    Attributes:
        prune_individual: when True the caller should truncate oversized
            individual messages (not just drop old turns).  Mirrors PaperBench's
            ``PRUNE_INDIVIDUAL_MESSAGES`` sentinel embedded in the exception
            message of ``LengthFinishReasonError``.
    """

    def __init__(self, message: str = "", prune_individual: bool = False):
        super().__init__(message)
        self.prune_individual = prune_individual


class AccountBlockedError(Exception):
    """Raised when the ByteDance gateway blocks the account (-2005).

    Triggered when cumulative content-safety violations for the business
    reach the limit (5 times).  Further requests to o-series models are
    denied until manually unblocked.  Callers should fail immediately.
    """
    pass


class ContentPolicyError(Exception):
    """Raised when the model's content safety policy is triggered.

    Azure o-series models lock the ENTIRE account after 5 cumulative
    violations.  Callers should fail immediately — do NOT retry.

    Attributes:
        dump_path: path to a JSON file containing the triggering messages
                   and error details, for post-mortem analysis of which
                   content triggered the filter.
    """

    def __init__(self, message: str, dump_path: str | None = None):
        super().__init__(message)
        self.dump_path = dump_path


@dataclass
class ToolCallResult:
    call_id: str
    name: str
    arguments: dict[str, Any]
    # Gemini 3 models return thought_signature inside extra_content on each
    # tool_call.  It must be passed back in subsequent requests or the API
    # returns 400.  For non-Gemini models this is always None and the field
    # is simply omitted when building the assistant message dict.
    extra_content: dict[str, Any] | None = None


@dataclass
class LLMResponse:
    text_content: str | None
    tool_calls: list[ToolCallResult]
    usage: dict[str, int]
    raw_message: Any
    time_spent_retrying: float = 0.0
    # GLM / DeepSeek-R1 style reasoning chain, stored for logging/debugging.
    # Maps to `message.reasoning_content` in the Chat Completions response.
    reasoning_content: str | None = None


@dataclass
class LLMConfig:
    provider: str | None = None
    model: str = "gpt-5.4"
    max_tokens: int = 32768
    api_key: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    organization: str | None = None
    project: str | None = None
    # temperature=None → not sent to API (PaperBench default: NOT_GIVEN).
    # GLM-5 and reasoning models should leave this as None.
    # Non-reasoning models like glm-4.7 may pass 0.7 explicitly if desired.
    temperature: float | None = None
    api_mode: str = "completions"
    web_search: bool = False
    # reasoning_effort:
    #   ResponsesLLMClient (GPT o-series): sent as native parameter.
    #   CompletionsLLMClient + Gemini: sent via extra_body={"reasoning_effort": ...}.
    #   CompletionsLLMClient + GLM/DeepSeek: ignored (they have no equivalent).
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    # context_window: maximum model context window from configuration.
    # The runtime derives a smaller prune budget from this value so there is
    # room for completion tokens and, for GLM models, tokenizer mismatch.
    context_window: int | None = None
    # use_phase: GPT-5.4+ feature — add "phase" field to replayed assistant
    # messages in Responses API input.  OpenAI strongly recommends this for
    # multi-step tool flows: historical assistant messages get phase="commentary"
    # so the model does not treat them as final answers (which causes early
    # stopping and degraded quality).  Has no effect on CompletionsLLMClient.
    # Set via AISCI_USE_PHASE env var ("true" to enable).
    use_phase: bool = False
    # clear_thinking: GLM-only (Completions API). When set, CompletionsLLMClient
    # sends extra_body.thinking.clear_thinking to the API. False = preserve
    # reasoning_content across turns (Preserved Thinking on). True = clear each
    # turn. None = do not send (API default). Only GLM-5 profiles should set
    # this; other models (GPT, Kimi, etc.) must not be affected.
    # Set via AISCI_CLEAR_THINKING env var ("false" | "true").
    clear_thinking: bool | None = None

    @property
    def prune_context_window(self) -> int | None:
        if self.context_window is None:
            return None

        input_budget = int(self.context_window) - int(self.max_tokens)
        if input_budget <= 0:
            return max(1, int(self.context_window))

        model_lower = (self.model or "").lower()
        if "glm" in model_lower:
            # GLM tokenization is larger than tiktoken by roughly 1.26x in the
            # workloads we see here, so keep a conservative margin before
            # falling back to retry-driven pruning.
            return max(1024, int(input_budget / 1.26 - 5000))

        return input_budget


# ====================================================================== #
# Abstract base
# ====================================================================== #


class LLMClient(ABC):
    """
    Abstract LLM client interface.

    Both ``CompletionsLLMClient`` and ``ResponsesLLMClient`` accept the same
    Chat-Completions-format messages list and tool schemas — the Responses
    client converts internally.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._total_tokens: dict[str, int] = {"input": 0, "output": 0}
        self._total_retry_time: float = 0.0

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        ...

    @property
    def total_tokens(self) -> dict[str, int]:
        return self._total_tokens.copy()

    @property
    def total_retry_time(self) -> float:
        return self._total_retry_time

    def _retry_loop(self, call_fn):
        """
        Retry logic aligned with PaperBench's TimeTrackingRetryConfig:

        - wait_random_exponential(min=1, max=300): random jitter prevents
          thundering-herd when multiple parallel tasks all hit 429 at once.
        - stop_after_delay(2h): gives up after 2 hours of cumulative waiting.
        - Only retries transient errors (RateLimit, Connection, Timeout,
          InternalServer) and select BadRequestErrors that are truly transient.
        - ContentPolicyError and ContextLengthError immediately re-raised (no retry).
        - Tracks total wait time in ``retry_time`` for use_real_time_limit.
        """
        retry_time = 0.0
        retry_start = time.time()
        attempt = 0

        while True:
            try:
                response = call_fn()
                self._total_retry_time += retry_time
                return response, retry_time

            except PermissionDeniedError as e:
                error_code = str(getattr(e, "code", "") or "")
                error_msg = str(getattr(e, "message", "") or str(e))
                # -2005: ByteDance gateway account-level block after 5 safety violations
                if "-2005" in error_code or "达到上限" in error_msg or "安全拦截" in error_msg:
                    try:
                        req_logid = e.request.headers.get("X-TT-LOGID", "unknown")
                    except Exception:
                        req_logid = "unknown"
                    x_request_id = getattr(e, "request_id", None) or "unknown"
                    logger.error(
                        f"Account blocked by ByteDance gateway (-2005) "
                        f"[tt_logid={req_logid}] [x-request-id={x_request_id}] "
                        f"o-series safety limit reached (5 cumulative violations)",
                        error_code=error_code,
                        tt_logid=req_logid,
                        x_request_id=x_request_id,
                        error_msg=error_msg[:300],
                    )
                    raise AccountBlockedError(
                        f"Account blocked (-2005) "
                        f"[tt_logid={req_logid}] [x-request-id={x_request_id}]: {error_msg}"
                    ) from e
                raise  # other 403 errors re-raised as-is

            except BadRequestError as e:
                error_code = str(getattr(e, "code", "") or "")
                # Fallback: some gateways nest code under body["error"]["code"]
                if not error_code:
                    try:
                        body = getattr(e, "body", None)
                        if isinstance(body, dict):
                            err_obj = body.get("error") if isinstance(body.get("error"), dict) else {}
                            error_code = str(err_obj.get("code") or "")
                    except Exception:
                        pass
                error_msg = str(getattr(e, "message", "") or str(e))
                err_lower = error_msg.lower()

                # Content policy: fail immediately, no retry.
                # Azure locks the ENTIRE account for o-series models after 5
                # cumulative violations. Retrying would waste quota — fail fast.
                if "-4321" in error_code or "invalid_prompt" in err_lower:
                    raise self._make_content_policy_error(e) from e

                # Context length: caller should prune and retry.
                # Keywords cover both OpenAI and ByteDance gateway error messages.
                # PaperBench api.py explicitly checks "input exceeds the context
                # window" from ByteDance's error body and converts it to
                # LengthFinishReasonError.  We do the same here.
                #
                # prune_individual=True when the input itself is over-limit
                # (not just too many turns), so the caller truncates large
                # individual messages via tiktoken — mirrors PaperBench's
                # PRUNE_INDIVIDUAL_MESSAGES path in utils.prune_messages().
                _INDIVIDUAL_PRUNE_KWS = (
                    "input exceeds the context window",
                    "input exceeds",
                    "prompt is too long",
                    "prompt too long",
                    "prompt exceeds max length",  # e.g. error -4316 from some gateways
                    "exceeds max length",
                    "reduce the length",
                    "content too long",  # Gemini OpenAI-compat endpoint
                    "request payload size exceeds",  # Gemini
                )
                _BULK_PRUNE_KWS = (
                    "context_length_exceeded",
                    "context_length",
                    "maximum context",
                    "too many tokens",
                    "max_tokens",
                    "max_output_tokens",
                    "exceeds the limit",
                    "exceeds token limit",
                    "total content token count",  # Gemini native error format
                )
                if any(kw in err_lower for kw in _INDIVIDUAL_PRUNE_KWS):
                    raise ContextLengthError(str(e), prune_individual=True) from e
                if any(kw in err_lower for kw in _BULK_PRUNE_KWS):
                    raise ContextLengthError(str(e)) from e

                # Certain BadRequests are transient (PaperBench: is_retryable_bad_request)
                is_transient = any(phrase in error_msg for phrase in _RETRYABLE_BADREQUEST_PHRASES)
                if not is_transient:
                    raise

                wait = self._next_wait(attempt)
                if time.time() - retry_start + wait > _RETRY_STOP_AFTER:
                    raise RuntimeError(
                        f"LLM retry budget exhausted ({_RETRY_STOP_AFTER}s) on BadRequestError"
                    ) from e
                logger.warning("API error (transient BadRequest)", attempt=attempt + 1,
                               err=error_msg[:200], wait=wait)
                time.sleep(wait)
                retry_time += wait
                attempt += 1

            except (ContextLengthError, ContentPolicyError, AccountBlockedError):
                raise

            except _RETRYABLE_EXCEPTIONS as e:
                err_str = str(e)
                err_lower = err_str.lower()

                # OpenAI account-level ban masquerades as 429 RateLimitError
                # but the message says "terminated due to violation of our
                # policies" or "account_deactivated".  Do NOT retry — the key
                # is permanently revoked and retrying just wastes time.
                _FATAL_429_PHRASES = (
                    "terminated due to violation",
                    "violation of our policies",
                    "account_deactivated",
                    "account deactivated",
                    "access was terminated",
                )
                if isinstance(e, RateLimitError) and any(p in err_lower for p in _FATAL_429_PHRASES):
                    logger.error(
                        "API key terminated by provider (fatal 429) — not retrying",
                        err=err_str[:300],
                    )
                    raise RuntimeError(
                        f"API key terminated by provider: {err_str[:300]}"
                    ) from e

                wait = self._next_wait(attempt)
                if time.time() - retry_start + wait > _RETRY_STOP_AFTER:
                    raise RuntimeError(
                        f"LLM retry budget exhausted ({_RETRY_STOP_AFTER}s)"
                    ) from e
                logger.warning(
                    "API error", attempt=attempt + 1,
                    err=err_str[:200], wait=wait,
                )
                time.sleep(wait)
                retry_time += wait
                attempt += 1

            except Exception:
                raise

    @staticmethod
    def _next_wait(attempt: int) -> float:
        """
        wait_random_exponential(min=1, max=300) — mirrors PaperBench RetryConfig.

        Formula: random.uniform(min, min(max, 2^attempt * min))
        """
        base = min(_RETRY_WAIT_MAX, _RETRY_WAIT_MIN * (2 ** attempt))
        return random.uniform(_RETRY_WAIT_MIN, base)

    def _make_content_policy_error(self, original_error: BadRequestError) -> ContentPolicyError:
        """Build a ContentPolicyError, log both IDs, and dump the triggering context.

        Mirrors PaperBench's subagents/base.py BadRequestError handler:
          req_logid   = e.request.headers.get("X-TT-LOGID", "unknown")
          x_request_id = getattr(e, "request_id", None) or "unknown"
        Both IDs are recorded so the exact request can be found in gateway logs.
        """
        error_code = str(getattr(original_error, "code", "") or "")
        error_msg = str(getattr(original_error, "message", "") or str(original_error))

        # Read back the X-TT-LOGID that was sent with THIS request
        # (PaperBench: e.request.headers.get("X-TT-LOGID", "unknown"))
        try:
            req_logid = original_error.request.headers.get("X-TT-LOGID", "unknown")
        except Exception:
            req_logid = "unknown"

        # x-request-id returned by the gateway in the response
        # (PaperBench: getattr(e, "request_id", None) or "unknown")
        x_request_id = getattr(original_error, "request_id", None) or "unknown"

        dump_path = self._dump_safety_trigger(error_code, error_msg, req_logid, x_request_id)

        logger.error(
            f"Content policy violation [tt_logid={req_logid}] "
            f"[x-request-id={x_request_id}] "
            f"failing immediately to prevent account-level lockout",
            error_code=error_code,
            tt_logid=req_logid,
            x_request_id=x_request_id,
            status_code=original_error.status_code,
            response_body=original_error.body,
            dump_path=dump_path,
        )
        return ContentPolicyError(
            f"Content policy violation ({error_code}) "
            f"[tt_logid={req_logid}] [x-request-id={x_request_id}]: {error_msg}",
            dump_path=dump_path,
        )

    def _dump_safety_trigger(
        self,
        error_code: str,
        error_msg: str,
        tt_logid: str = "unknown",
        x_request_id: str = "unknown",
    ) -> str | None:
        """Dump the triggering request context to disk for post-mortem analysis.

        Saves to $LOGS_DIR/safety_triggers/content_policy_<timestamp>.json.
        Includes both IDs needed to look up the request in ByteDance gateway logs.
        """
        import traceback as _tb
        try:
            logs_dir = os.environ.get("LOGS_DIR", "/home/logs")
            dump_dir = os.path.join(logs_dir, "safety_triggers")
            os.makedirs(dump_dir, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S")
            dump_path = os.path.join(dump_dir, f"content_policy_{ts}.json")

            last_messages = getattr(self, "_last_messages", None)
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_code": error_code,
                "error_message": error_msg,
                "model": self.config.model,
                "api_mode": self.config.api_mode,
                # Both IDs needed to trace the request in ByteDance gateway logs.
                # Mirrors PaperBench's subagents/base.py error handler.
                "tt_logid": tt_logid,        # X-TT-LOGID we sent (read back from e.request.headers)
                "x_request_id": x_request_id, # gateway-assigned ID (from e.request_id)
                "last_messages": last_messages,
                "last_messages_count": len(last_messages) if last_messages else 0,
                "last_tools": getattr(self, "_last_tool_names", None),
            }
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            logger.info(
                "Safety trigger dump saved",
                path=dump_path,
                messages_count=payload["last_messages_count"],
                tt_logid=tt_logid,
                x_request_id=x_request_id,
            )
            return dump_path
        except Exception as ex:
            logger.warning("Failed to dump safety trigger",
                           err=str(ex), traceback=_tb.format_exc())
            return None


# ====================================================================== #
# Chat Completions API client
# ====================================================================== #


class CompletionsLLMClient(LLMClient):
    """
    Chat Completions API client (``chat.completions.create``).

    Used for GLM, DeepSeek, and any model that doesn't support the
    Responses API.  Matches PaperBench's AzureOpenAICompletionsTurnCompleter.

    Client selection is explicit:
    - ``provider=openai``       → ``OpenAI()``
    - ``provider=azure-openai`` → ``AzureOpenAI()``
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        provider = (config.provider or os.environ.get("AISCI_PROVIDER") or "").strip().lower()
        if not provider:
            raise ValueError("AISCI_PROVIDER is required for the completions client.")
        # No default_headers on client — logid is generated fresh per call
        # in extra_headers, matching PaperBench's AzureOpenAICompletionsTurnCompleter.
        if provider == "azure-openai":
            client_kwargs: dict[str, Any] = {"max_retries": 0}
            if config.api_key:
                client_kwargs["api_key"] = config.api_key
            if config.azure_endpoint:
                client_kwargs["azure_endpoint"] = config.azure_endpoint
            if config.api_version:
                client_kwargs["api_version"] = config.api_version
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            self.client = AzureOpenAI(**client_kwargs)
            logger.info(
                "CompletionsLLMClient: using AzureOpenAI",
                endpoint=(config.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""))[:60],
            )
        elif provider == "openai":
            client_kwargs = {"max_retries": 0}
            if config.api_key:
                client_kwargs["api_key"] = config.api_key
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            if config.organization:
                client_kwargs["organization"] = config.organization
            if config.project:
                client_kwargs["project"] = config.project
            self.client = OpenAI(**client_kwargs)
            logger.info(
                "CompletionsLLMClient: using OpenAI",
                base_url=(config.base_url or os.environ.get("OPENAI_BASE_URL", ""))[:60],
            )
        else:
            raise ValueError(f"Unsupported LLM provider for completions client: {provider}")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._last_messages = messages
        self._last_tool_names = [t["function"]["name"] for t in (tools or []) if "function" in t]

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            # X-TT-LOGID: fresh logid per call so each request is individually
            # traceable in ByteDance gateway logs.
            # Mirrors PaperBench's AzureOpenAICompletionsTurnCompleter.
            "extra_headers": {"X-TT-LOGID": logid.generate_v2()},
        }
        # temperature: only send when explicitly set (PaperBench default: NOT_GIVEN).
        # GLM-5 reasoning models must not receive temperature.
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if tools:
            kwargs["tools"] = tools

        extra_body: dict[str, Any] = {}
        # GLM-only: Preserved Thinking.
        if self.config.clear_thinking is not None:
            extra_body["thinking"] = {
                "type": "enabled",
                "clear_thinking": self.config.clear_thinking,
            }
        # Gemini 3: reasoning_effort via extra_body (maps to thinking_level).
        if self.config.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body

        response, retry_time = self._retry_loop(
            lambda: self.client.chat.completions.create(**kwargs)
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            # finish_reason=length means the output was cut off.  This often
            # indicates a single oversized message flooded the context so the
            # model ran out of space for its response.  Signal prune_individual
            # so callers truncate large messages rather than just dropping turns
            # (mirrors PaperBench's PRUNE_INDIVIDUAL_MESSAGES sentinel).
            raise ContextLengthError(
                "finish_reason=length — output truncated, context likely full",
                prune_individual=True,
            )

        usage = self._extract_usage_completions(response)
        tool_calls = self._parse_tool_calls_completions(message)

        # GLM-5 / DeepSeek-R1 return a `reasoning_content` attribute alongside
        # `content`.  The standard OpenAI SDK exposes extra fields via
        # model_extra / __dict__, so we probe with getattr to stay compatible
        # with both vanilla OpenAI models (where it is absent) and GLM series.
        reasoning_content: str | None = getattr(message, "reasoning_content", None)

        return LLMResponse(
            text_content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            raw_message=message,
            time_spent_retrying=retry_time,
            reasoning_content=reasoning_content,
        )

    def _extract_usage_completions(self, response) -> dict[str, int]:
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input": response.usage.prompt_tokens or 0,
                "output": response.usage.completion_tokens or 0,
            }
            self._total_tokens["input"] += usage.get("input", 0)
            self._total_tokens["output"] += usage.get("output", 0)
        return usage

    @staticmethod
    def _parse_tool_calls_completions(message) -> list[ToolCallResult]:
        tool_calls: list[ToolCallResult] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        "Malformed tool-call JSON — skipping",
                        tool=tc.function.name,
                        raw=tc.function.arguments[:200],
                    )
                    continue
                extra = getattr(tc, "extra_content", None)
                tool_calls.append(
                    ToolCallResult(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        extra_content=extra if isinstance(extra, dict) else None,
                    )
                )
        return tool_calls


def _response_output_item_to_dict(item: Any) -> dict[str, Any]:
    """Serialize a Responses API output item to a dict for the next request's input.
    Prefer to_dict() (cookbook) then model_dump() (Pydantic v2).
    """
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "model_dump_json"):
        return json.loads(item.model_dump_json())
    if hasattr(item, "__dict__"):
        return dict(item)
    return dict(item)


# ====================================================================== #
# Responses API client
# ====================================================================== #


class ResponsesLLMClient(LLMClient):
    """
    Responses API client (``responses.create``).

    Used for GPT with web_search_preview and reasoning configuration.
    Matches PaperBench's AzureOpenAIResponsesTurnCompleter.

    Internally converts:
    - Chat Completions messages → Responses API input items
    - Chat Completions tool schemas → Responses API tool format
    - Responses API output → common ``LLMResponse``

    Client selection is explicit:
    - ``provider=openai``       → ``OpenAI()``
    - ``provider=azure-openai`` → ``AzureOpenAI()``
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        provider = (config.provider or os.environ.get("AISCI_PROVIDER") or "").strip().lower()
        if not provider:
            raise ValueError("AISCI_PROVIDER is required for the responses client.")
        # No default_headers on client — logid is generated fresh per call
        # in extra_headers, matching PaperBench's AzureOpenAIResponsesTurnCompleter.
        if provider == "azure-openai":
            client_kwargs: dict[str, Any] = {"max_retries": 0}
            if config.api_key:
                client_kwargs["api_key"] = config.api_key
            if config.azure_endpoint:
                client_kwargs["azure_endpoint"] = config.azure_endpoint
            if config.api_version:
                client_kwargs["api_version"] = config.api_version
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            self.client = AzureOpenAI(**client_kwargs)
            logger.info(
                "ResponsesLLMClient: using AzureOpenAI",
                endpoint=(config.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""))[:60],
            )
        elif provider == "openai":
            client_kwargs = {"max_retries": 0}
            if config.api_key:
                client_kwargs["api_key"] = config.api_key
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            if config.organization:
                client_kwargs["organization"] = config.organization
            if config.project:
                client_kwargs["project"] = config.project
            self.client = OpenAI(**client_kwargs)
            logger.info(
                "ResponsesLLMClient: using OpenAI",
                base_url=(config.base_url or os.environ.get("OPENAI_BASE_URL", ""))[:60],
            )
        else:
            raise ValueError(f"Unsupported LLM provider for responses client: {provider}")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._last_messages = messages
        self._last_tool_names = [t["function"]["name"] for t in (tools or []) if "function" in t]

        input_items = self._convert_messages_to_input(messages)

        responses_tools: list[dict] = []
        if self.config.web_search:
            responses_tools.append({"type": "web_search"})
        if tools:
            responses_tools.extend(self._convert_tool_schemas(tools))

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": input_items,
            "max_output_tokens": self.config.max_tokens,
            # X-TT-LOGID: fresh logid per call so each request is individually
            # traceable in ByteDance gateway logs.
            # Mirrors PaperBench's OpenAIResponsesTurnCompleter.
            "extra_headers": {"X-TT-LOGID": logid.generate_v2()},
        }
        if responses_tools:
            kwargs["tools"] = responses_tools

        # Reasoning config — mirrors PaperBench's ReasoningConfig → Reasoning TypedDict.
        # Uses openai.types.shared_params.reasoning.Reasoning (a TypedDict subclassing dict)
        # so the SDK receives the exact same wire format as PaperBench's completer.
        if self.config.reasoning_effort:
            reasoning_kwargs: dict[str, str] = {"effort": self.config.reasoning_effort}
            if self.config.reasoning_summary:
                reasoning_kwargs["summary"] = self.config.reasoning_summary
            kwargs["reasoning"] = Reasoning(**reasoning_kwargs)
        elif self.config.temperature is not None:
            # Only set temperature when reasoning is NOT enabled
            # (OpenAI disallows temperature with reasoning) and only when
            # explicitly configured (mirrors PaperBench's NOT_GIVEN default).
            kwargs["temperature"] = self.config.temperature

        response, retry_time = self._retry_loop(
            lambda: self.client.responses.create(**kwargs)
        )

        # Check for incomplete / failed responses
        status = getattr(response, "status", "completed")
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = str(details).lower() if details else ""
            # max_output_tokens hit → output truncated, context likely full.
            # Signal prune_individual so caller truncates oversized messages.
            prune_ind = "max_output_tokens" in reason or "length" in reason
            raise ContextLengthError(
                f"Response incomplete: {details}",
                prune_individual=prune_ind,
            )
        if status == "failed":
            error = getattr(response, "error", "unknown error")
            raise RuntimeError(f"Responses API failed: {error}")

        text_content, tool_calls = self._parse_output(response.output)
        usage = self._extract_usage_responses(response)

        # reasoning_content left as None: Responses API preserves reasoning
        # internally across turns; we do not pass it in input or store in messages.
        return LLMResponse(
            text_content=text_content,
            tool_calls=tool_calls,
            usage=usage,
            raw_message=response.output,
            time_spent_retrying=retry_time,
        )

    # ---- format conversions ----

    def _convert_messages_to_input(self, messages: list[dict]) -> list[dict]:
        """
        Convert Chat Completions message list → Responses API input items.

        Mapping:
        - system/user/developer  → same role item
        - assistant (content)    → {"role": "assistant", "content": ...}
        - assistant (tool_calls) → {"type": "function_call", ...} per call
        - tool                   → {"type": "function_call_output", ...}

        When ``config.use_phase`` is True (GPT-5.4+), replayed assistant
        messages get a ``phase`` field per OpenAI's recommendation.

        When an assistant message has ``_response_output`` (list from
        response.output), we inject those raw items (reasoning + message +
        function_calls) instead of converting from content/tool_calls.

        Official docs (platform.openai.com):
        - Function calling: "input_list += response.output" then append
          function_call_output per call; "reasoning items ... must also be
          passed back with tool call outputs."
        - Reasoning guide "Keeping reasoning items in context": "pass in all
          [output] items from a previous response into the input of a new one";
          or "pass reasoning items ... by manually passing in all the output
          items from a past response into the input of a new one."
        """
        use_phase = self.config.use_phase
        items: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            if role in ("system", "user", "developer"):
                items.append({"role": role, "content": msg.get("content") or ""})
            elif role == "assistant":
                raw_output = msg.get("_response_output")
                if isinstance(raw_output, list):
                    for out in raw_output:
                        items.append(_response_output_item_to_dict(out))
                    continue
                has_tool_calls = bool(msg.get("tool_calls"))
                item: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
                if use_phase:
                    item["phase"] = "commentary" if has_tool_calls else "final_answer"
                items.append(item)
                for tc in msg.get("tool_calls") or []:
                    items.append({
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content") or "",
                })
        return items

    @staticmethod
    def _convert_tool_schemas(tools: list[dict]) -> list[dict]:
        """
        Convert Chat Completions tool schemas (nested) → Responses API format (flat).

        Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Responses:   {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        result: list[dict] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                flat: dict[str, Any] = {"type": "function", "name": fn["name"]}
                if fn.get("description"):
                    flat["description"] = fn["description"]
                if fn.get("parameters"):
                    flat["parameters"] = fn["parameters"]
                if fn.get("strict") is not None:
                    flat["strict"] = fn["strict"]
                result.append(flat)
            else:
                result.append(tool)
        return result

    @staticmethod
    def _parse_output(output) -> tuple[str | None, list[ToolCallResult]]:
        """Parse Responses API output items into text content + tool calls."""
        text_content: str | None = None
        tool_calls: list[ToolCallResult] = []

        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) == "output_text":
                        text_content = (text_content or "") + part.text
            elif item_type == "function_call":
                try:
                    args = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Malformed tool-call JSON — skipping",
                        tool=getattr(item, "name", "?"),
                        raw=str(getattr(item, "arguments", ""))[:200],
                    )
                    continue
                tool_calls.append(
                    ToolCallResult(
                        call_id=item.call_id,
                        name=item.name,
                        arguments=args,
                    )
                )
            # web_search_call and other types are silently skipped
        return text_content, tool_calls

    def _extract_usage_responses(self, response) -> dict[str, int]:
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input": getattr(response.usage, "input_tokens", 0) or 0,
                "output": getattr(response.usage, "output_tokens", 0) or 0,
            }
            self._total_tokens["input"] += usage.get("input", 0)
            self._total_tokens["output"] += usage.get("output", 0)
        return usage


# ====================================================================== #
# Factory
# ====================================================================== #


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """
    Create the appropriate LLM client based on configuration.

    If ``config`` is None, reads all settings from environment variables.
    In normal ``aisci`` runs these env vars are materialized from the
    host-side YAML registry before the runtime process starts:
    - AISCI_PROVIDER          → openai | azure-openai
    - AISCI_MODEL             → model name
    - AISCI_API_MODE          → completions | responses
    - AISCI_WEB_SEARCH        → true | false (responses only)
    - AISCI_REASONING_EFFORT  → high | medium | low | xhigh
                                (responses: native param; completions+Gemini: extra_body)
    - AISCI_REASONING_SUMMARY → auto | none (responses only)
    - AISCI_USE_PHASE         → true | false (responses only)
                                GPT-5.4+: add phase="commentary" to replayed
                                assistant messages to prevent early stopping.
    - AISCI_MAX_TOKENS        → max output tokens
    - AISCI_CONTEXT_WINDOW    → maximum model context window from the profile.
                                The runtime derives the actual prune budget locally:
                                  GLM-5:   202752 -> 103901
                                  GPT-5.4: 1000000 -> 868928
    - AISCI_TEMPERATURE       → float temperature, e.g. "0.7"
    - AISCI_CLEAR_THINKING    → "false" | "true" (Completions only, GLM-5). When set,
                                extra_body.thinking.clear_thinking is sent. "true"
                                = clear each turn; "false" = Preserved Thinking.
    """
    if config is None:
        _max_tokens_env = os.environ.get("AISCI_MAX_TOKENS")
        _ctx_window_env = os.environ.get("AISCI_CONTEXT_WINDOW")
        _temperature_env = os.environ.get("AISCI_TEMPERATURE")
        _clear_thinking_env = os.environ.get("AISCI_CLEAR_THINKING", "").strip().lower()
        _provider = (os.environ.get("AISCI_PROVIDER") or "").strip() or None
        _model = (os.environ.get("AISCI_MODEL") or "").strip()
        _api_mode = (os.environ.get("AISCI_API_MODE") or "").strip().lower()
        if _provider is None:
            raise ValueError("AISCI_PROVIDER is required to create an LLM client.")
        if not _model:
            raise ValueError("AISCI_MODEL is required to create an LLM client.")
        if _api_mode not in {"responses", "completions"}:
            raise ValueError("AISCI_API_MODE must be 'responses' or 'completions'.")
        if not _max_tokens_env:
            raise ValueError("AISCI_MAX_TOKENS is required to create an LLM client.")
        clear_thinking: bool | None = None
        if _clear_thinking_env == "false":
            clear_thinking = False
        elif _clear_thinking_env == "true":
            clear_thinking = True
        config = LLMConfig(
            provider=_provider,
            model=_model,
            api_mode=_api_mode,
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or None,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT") or None,
            api_version=os.environ.get("OPENAI_API_VERSION") or None,
            organization=os.environ.get("OPENAI_ORG_ID") or None,
            project=os.environ.get("OPENAI_PROJECT") or None,
            web_search=os.environ.get("AISCI_WEB_SEARCH", "").lower() == "true",
            reasoning_effort=os.environ.get("AISCI_REASONING_EFFORT") or None,
            reasoning_summary=os.environ.get("AISCI_REASONING_SUMMARY") or None,
            max_tokens=int(_max_tokens_env),
            context_window=int(_ctx_window_env) if _ctx_window_env else None,
            temperature=float(_temperature_env) if _temperature_env else None,
            use_phase=os.environ.get("AISCI_USE_PHASE", "").lower() == "true",
            clear_thinking=clear_thinking,
        )

    if config.api_mode == "responses":
        logger.info(
            "Creating Responses API client",
            provider=config.provider,
            model=config.model,
            max_tokens=config.max_tokens,
            context_window=config.context_window,
            prune_context_window=config.prune_context_window,
            web_search=config.web_search,
            reasoning_effort=config.reasoning_effort,
            reasoning_summary=config.reasoning_summary,
            use_phase=config.use_phase,
        )
        return ResponsesLLMClient(config)
    else:
        logger.info(
            "Creating Chat Completions API client",
            provider=config.provider,
            model=config.model,
            max_tokens=config.max_tokens,
            context_window=config.context_window,
            prune_context_window=config.prune_context_window,
            clear_thinking=config.clear_thinking,
        )
        return CompletionsLLMClient(config)
