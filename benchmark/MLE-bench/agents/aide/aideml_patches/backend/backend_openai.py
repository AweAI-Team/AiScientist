"""Backend for OpenAI API — patched for MLE-Bench / aisci parity.

Aligned with ``agents/aisci/llm_client.py`` ``CompletionsLLMClient``:

- Client: ``AzureOpenAI()`` if ``AZURE_OPENAI_ENDPOINT`` is set, else ``OpenAI()``
  (reads ``OPENAI_BASE_URL`` from the environment).
- Gemini: optional ``GEMINI_API_KEY_POOL`` (comma/newline-separated) → per-request
  ``OpenAI(api_key=random.choice(pool), base_url=...)`` like aisci.
- Each request sends ``X-TT-LOGID`` (``uuid.uuid4().hex``) merged with optional ``extra_headers`` from
  the caller (caller keys override the default header).
- ``extra_body`` for GLM (Azure + ``glm-*``): ``thinking`` with
  ``type: "enabled"`` and ``clear_thinking`` from ``AISCI_CLEAR_THINKING`` (same
  rules as aisci's ``create_llm_client`` factory for glm-5).
- ``extra_body`` for Gemini (no Azure): ``reasoning_effort`` from ``AISCI_REASONING_EFFORT``.
- Gemini OpenAI-compat **rejects** ``strict`` (and nested ``strict``) on tool definitions; aideml /
  OpenAI SDK may emit it — we recursively drop ``strict`` from tool dicts (aisci’s own
  ``get_tool_schema()`` never adds ``strict``, so only the AIDE / aideml path needs this).
- Retries: same policy as ``agents/aisci/llm_client.py`` ``LLMClient._retry_loop`` (wait jitter, 24h cap,
  transient BadRequest phrases, fatal 429 phrases, AccountBlocked / content-policy / context-length branches).
- ``finish_reason == "length"``: same as aisci ``CompletionsLLMClient.chat`` when **tools** are used —
  ``ContextLengthError(..., prune_individual=True)``.  For plain text completions, prefer partial assistant
  text when extractable (warn); if nothing usable, raise ``ContextLengthError`` like aisci.
- ``max_tokens`` from ``AISCI_MAX_TOKENS`` when set (overrides aideml default).
- GLM-5: do not send ``temperature`` (aisci omits it for reasoning GLM-5).
- Gemini: if ``AISCI_TEMPERATURE`` is set, it overrides the request temperature (aisci uses 1.0).

Optional env ``AIDE_OPENAI_EXTRA_BODY`` (JSON object) is deep-merged on top of the
above so you can override any field.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
import uuid
from typing import Any

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
)
from funcy import notnone, select_values
from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

logger = logging.getLogger("aide")


def _tt_logid_headers() -> dict[str, str]:
    """Per-request ``X-TT-LOGID`` (UUID v4 hex, no third-party logid package)."""

    return {"X-TT-LOGID": uuid.uuid4().hex}


# Aligned with ``agents/aisci/llm_client.py`` (CompletionsLLMClient / ``_retry_loop``).
_RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
_RETRYABLE_BADREQUEST_PHRASES = (
    "Timeout while downloading",
    "connection reset by peer",
    "TLS handshake timeout",
)
_RETRY_WAIT_MIN = 1
_RETRY_WAIT_MAX = 300
_RETRY_STOP_AFTER = 86400  # 24 hours total retry budget


class ContextLengthError(Exception):
    """Compatible with aisci ``llm_client.ContextLengthError`` for aide-backend callers."""

    def __init__(self, message: str = "", *, prune_individual: bool = False):
        super().__init__(message)
        self.prune_individual = prune_individual


class AccountBlockedError(Exception):
    """Compatible with aisci ``AccountBlockedError`` (-2005 gateway block)."""

    pass


class ContentPolicyError(Exception):
    """Compatible with aisci ``ContentPolicyError``."""

    def __init__(self, message: str, dump_path: str | None = None):
        super().__init__(message)
        self.dump_path = dump_path


def _next_aisci_wait(attempt: int) -> float:
    base = min(_RETRY_WAIT_MAX, _RETRY_WAIT_MIN * (2 ** attempt))
    return random.uniform(_RETRY_WAIT_MIN, base)


def _dump_safety_trigger(
    *,
    messages: list | None,
    model: str,
    error_code: str,
    error_msg: str,
    tt_logid: str,
    x_request_id: str,
) -> str | None:
    try:
        logs_dir = os.environ.get("LOGS_DIR", "/home/logs")
        dump_dir = os.path.join(logs_dir, "safety_triggers")
        os.makedirs(dump_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_path = os.path.join(dump_dir, f"content_policy_{ts}.json")
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_code": error_code,
            "error_message": error_msg,
            "model": model,
            "tt_logid": tt_logid,
            "x_request_id": x_request_id,
            "last_messages": messages,
            "last_messages_count": len(messages) if messages else 0,
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info(
            "Safety trigger dump saved path=%s messages=%s tt_logid=%s x_request_id=%s",
            dump_path,
            payload["last_messages_count"],
            tt_logid,
            x_request_id,
        )
        return dump_path
    except Exception as ex:
        logger.warning("Failed to dump safety trigger: %s", ex)
        return None


def _make_content_policy_error(
    original_error: BadRequestError,
    *,
    messages: list | None,
    model: str,
) -> ContentPolicyError:
    error_code = str(getattr(original_error, "code", "") or "")
    error_msg = str(getattr(original_error, "message", "") or str(original_error))
    try:
        req_logid = original_error.request.headers.get("X-TT-LOGID", "unknown")
    except Exception:
        req_logid = "unknown"
    x_request_id = getattr(original_error, "request_id", None) or "unknown"
    dump_path = _dump_safety_trigger(
        messages=messages,
        model=model,
        error_code=error_code,
        error_msg=error_msg,
        tt_logid=req_logid,
        x_request_id=x_request_id,
    )
    logger.error(
        "Content policy violation [tt_logid=%s] [x-request-id=%s] dump_path=%s code=%s",
        req_logid,
        x_request_id,
        dump_path,
        error_code,
    )
    return ContentPolicyError(
        f"Content policy violation ({error_code}) "
        f"[tt_logid={req_logid}] [x-request-id={x_request_id}]: {error_msg}",
        dump_path=dump_path,
    )


def _completions_retry_loop(
    call_fn,
    *,
    messages_for_dump: list | None,
    model: str,
):
    """AIDE copy of aisci ``LLMClient._retry_loop`` semantics (returns response only)."""
    retry_start = time.time()
    attempt = 0
    while True:
        try:
            return call_fn()

        except PermissionDeniedError as e:
            error_code = str(getattr(e, "code", "") or "")
            error_msg = str(getattr(e, "message", "") or str(e))
            if "-2005" in error_code or "达到上限" in error_msg or "安全拦截" in error_msg:
                try:
                    req_logid = e.request.headers.get("X-TT-LOGID", "unknown")
                except Exception:
                    req_logid = "unknown"
                x_request_id = getattr(e, "request_id", None) or "unknown"
                logger.error(
                    "Account blocked (-2005) tt_logid=%s x_request_id=%s msg=%s",
                    req_logid,
                    x_request_id,
                    error_msg[:300],
                )
                raise AccountBlockedError(
                    f"Account blocked (-2005) "
                    f"[tt_logid={req_logid}] [x-request-id={x_request_id}]: {error_msg}"
                ) from e
            raise

        except BadRequestError as e:
            error_code = str(getattr(e, "code", "") or "")
            if not error_code:
                try:
                    body = getattr(e, "body", None)
                    if isinstance(body, dict):
                        err_obj = (
                            body.get("error")
                            if isinstance(body.get("error"), dict)
                            else {}
                        )
                        error_code = str(err_obj.get("code") or "")
                except Exception:
                    pass
            error_msg = str(getattr(e, "message", "") or str(e))
            err_lower = error_msg.lower()

            if "-4321" in error_code or "invalid_prompt" in err_lower:
                raise _make_content_policy_error(
                    e, messages=messages_for_dump, model=model
                ) from e

            _INDIVIDUAL_PRUNE_KWS = (
                "input exceeds the context window",
                "input exceeds",
                "input token count exceeds",
                "prompt is too long",
                "prompt too long",
                "prompt exceeds max length",
                "exceeds max length",
                "reduce the length",
                "content too long",
                "request payload size exceeds",
            )
            _BULK_PRUNE_KWS = (
                "context_length_exceeded",
                "context_length",
                "maximum context",
                "maximum number of tokens allowed",
                "too many tokens",
                "max_tokens",
                "max_output_tokens",
                "exceeds the limit",
                "exceeds token limit",
                "total content token count",
            )
            if any(kw in err_lower for kw in _INDIVIDUAL_PRUNE_KWS):
                raise ContextLengthError(str(e), prune_individual=True) from e
            if any(kw in err_lower for kw in _BULK_PRUNE_KWS):
                raise ContextLengthError(str(e)) from e

            is_transient = any(
                phrase in error_msg for phrase in _RETRYABLE_BADREQUEST_PHRASES
            )
            if not is_transient:
                raise

            wait = _next_aisci_wait(attempt)
            if time.time() - retry_start + wait > _RETRY_STOP_AFTER:
                raise RuntimeError(
                    f"LLM retry budget exhausted ({_RETRY_STOP_AFTER}s) on BadRequestError"
                ) from e
            logger.warning(
                "API error (transient BadRequest) attempt=%s wait=%s err=%s",
                attempt + 1,
                wait,
                error_msg[:200],
            )
            time.sleep(wait)
            attempt += 1

        except (ContextLengthError, ContentPolicyError, AccountBlockedError):
            raise

        except _RETRYABLE_EXCEPTIONS as e:
            err_str = str(e)
            err_lower = err_str.lower()
            _FATAL_429_PHRASES = (
                "terminated due to violation",
                "violation of our policies",
                "account_deactivated",
                "account deactivated",
                "access was terminated",
            )
            if isinstance(e, RateLimitError) and any(
                p in err_lower for p in _FATAL_429_PHRASES
            ):
                logger.error(
                    "API key terminated by provider (fatal 429): %s",
                    err_str[:300],
                )
                raise RuntimeError(
                    f"API key terminated by provider: {err_str[:300]}"
                ) from e

            wait = _next_aisci_wait(attempt)
            if time.time() - retry_start + wait > _RETRY_STOP_AFTER:
                raise RuntimeError(
                    f"LLM retry budget exhausted ({_RETRY_STOP_AFTER}s)"
                ) from e
            logger.warning(
                "API error attempt=%s wait=%s err=%s",
                attempt + 1,
                wait,
                err_str[:200],
            )
            time.sleep(wait)
            attempt += 1

        except Exception:
            raise


_azure_client: AzureOpenAI | None = None
_openai_client: OpenAI | None = None


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def _drop_strict_keys_gemini(obj: Any) -> Any:
    """Recursively remove ``strict`` — Gemini OpenAI-compat rejects it anywhere under tools."""
    if isinstance(obj, dict):
        return {
            k: _drop_strict_keys_gemini(v)
            for k, v in obj.items()
            if k != "strict"
        }
    if isinstance(obj, list):
        return [_drop_strict_keys_gemini(x) for x in obj]
    return obj


def _gemini_sanitize_openai_tools(tools: list[Any]) -> list[Any]:
    return [_drop_strict_keys_gemini(copy.deepcopy(t)) for t in tools]


def _gemini_sanitize_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return _drop_strict_keys_gemini(copy.deepcopy(tool_choice))


def _get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is None:
        _azure_client = AzureOpenAI(max_retries=0)
    return _azure_client


def _get_openai_singleton() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(max_retries=0)
    return _openai_client


def _client_for_request(model: str) -> OpenAI | AzureOpenAI:
    """Match aisci CompletionsLLMClient._client_for_request + Azure branch."""
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return _get_azure_client()

    pool_raw = os.environ.get("GEMINI_API_KEY_POOL", "")
    if pool_raw.strip() and _is_gemini_model(model):
        pool: list[str] = []
        seen: set[str] = set()
        for item in pool_raw.replace("\n", ",").split(","):
            key = item.strip()
            if key and key not in seen:
                pool.append(key)
                seen.add(key)
        if pool:
            kwargs: dict[str, Any] = {
                "max_retries": 0,
                "api_key": random.choice(pool),
            }
            bu = os.environ.get("OPENAI_BASE_URL", "").strip()
            if bu:
                kwargs["base_url"] = bu
            return OpenAI(**kwargs)

    return _get_openai_singleton()


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if (
            k in dst
            and isinstance(dst[k], dict)
            and isinstance(v, dict)
        ):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _glm_thinking_extra_body(model: str) -> dict[str, Any] | None:
    """Aligned with aisci CompletionsLLMClient.chat + LLMConfig.clear_thinking."""
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return None
    if not model.lower().startswith("glm-"):
        return None
    v = os.environ.get("AISCI_CLEAR_THINKING", "").strip().lower()
    clear_thinking: bool | None = None
    if v == "false":
        clear_thinking = False
    elif v == "true":
        clear_thinking = True
    elif v == "" and "glm-5" in model.lower():
        clear_thinking = True
    if clear_thinking is None:
        return None
    return {"thinking": {"type": "enabled", "clear_thinking": clear_thinking}}


def _gemini_reasoning_extra_body(model: str) -> dict[str, Any] | None:
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return None
    if not _is_gemini_model(model):
        return None
    effort = (os.environ.get("AISCI_REASONING_EFFORT") or "").strip()
    if not effort:
        return None
    return {"reasoning_effort": effort}


def _parse_aide_extra_body_env() -> dict[str, Any] | None:
    raw = os.environ.get("AIDE_OPENAI_EXTRA_BODY", "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        logger.warning("AIDE_OPENAI_EXTRA_BODY is not valid JSON; ignoring")
        return None


def _assistant_visible_text(message: Any) -> str | None:
    """Best-effort assistant text for Chat Completions (non-tool path).

    Gemini 3 (OpenAI-compatible) often returns final text in ``reasoning_content``
    only; the OpenAI SDK leaves ``content`` as None and stashes ``reasoning_content``
    in ``model_extra``.  GLM / DeepSeek expose ``reasoning_content`` alongside
    ``content`` — we prefer non-empty ``content``, then fall back to reasoning.
    """
    c = message.content
    if c is not None and str(c).strip():
        return str(c)

    rc = getattr(message, "reasoning_content", None)
    if rc is not None and str(rc).strip():
        return str(rc)

    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = extra.get(key)
            if v is not None and str(v).strip():
                return str(v)

    refusal = getattr(message, "refusal", None)
    if refusal is not None and str(refusal).strip():
        return str(refusal)

    if c is not None:
        return str(c)
    return None


def _build_extra_body(model: str) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    g = _glm_thinking_extra_body(model)
    if g:
        _deep_merge(merged, g)
    r = _gemini_reasoning_extra_body(model)
    if r:
        _deep_merge(merged, r)
    user = _parse_aide_extra_body_env()
    if user:
        _deep_merge(merged, user)
    return merged or None


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    model = str(filtered_kwargs.get("model", ""))

    client = _client_for_request(model)

    # Gemini (OpenAI-compatible): lone system messages yield empty ``contents`` → 400 INVALID_ARGUMENT.
    # Config sets ``agent.convert_system_to_user``; keep this fallback for older configs / images.
    if _is_gemini_model(model):
        convert_system_to_user = True

    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )

    if func_spec is not None:
        tools = [func_spec.as_openai_tool_dict]
        tc = func_spec.openai_tool_choice_dict
        if _is_gemini_model(model):
            tools = _gemini_sanitize_openai_tools(tools)
            tc = _gemini_sanitize_tool_choice(tc)
        filtered_kwargs["tools"] = tools
        filtered_kwargs["tool_choice"] = tc

    # aisci: GLM-5 reasoning — do not send temperature
    if "glm-5" in model.lower() and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        filtered_kwargs.pop("temperature", None)

    # aisci: AISCI_TEMPERATURE overrides (Gemini 3 uses "1.0")
    _atemp = os.environ.get("AISCI_TEMPERATURE", "").strip()
    if _atemp and _is_gemini_model(model):
        try:
            filtered_kwargs["temperature"] = float(_atemp)
        except ValueError:
            pass

    _mt = os.environ.get("AISCI_MAX_TOKENS", "").strip()
    if _mt:
        try:
            filtered_kwargs["max_tokens"] = int(_mt)
        except ValueError:
            pass

    eb = _build_extra_body(model)
    if eb:
        filtered_kwargs["extra_body"] = eb

    hdr = _tt_logid_headers()
    merged_h = {**hdr, **filtered_kwargs.pop("extra_headers", {})}
    filtered_kwargs["extra_headers"] = merged_h

    t0 = time.time()
    completion = _completions_retry_loop(
        lambda: client.chat.completions.create(
            messages=messages,
            **filtered_kwargs,
        ),
        messages_for_dump=messages,
        model=model,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)

    if func_spec is None:
        if finish_reason == "length":
            output = _assistant_visible_text(choice.message)
            if output is not None:
                logger.warning(
                    "finish_reason=length but partial assistant text retained (aisci would raise ContextLengthError)"
                )
            else:
                raise ContextLengthError(
                    "finish_reason=length — output truncated, context likely full",
                    prune_individual=True,
                )
        else:
            output = _assistant_visible_text(choice.message)
        if output is None:
            fin = getattr(choice, "finish_reason", None)
            extra = getattr(choice.message, "model_extra", None) or {}
            keys = list(extra.keys()) if isinstance(extra, dict) else []
            logger.error(
                "Chat completion has no extractable text (content/reasoning empty); "
                "finish_reason=%s model_extra_keys=%s",
                fin,
                keys,
            )
            raise ValueError(
                "Model returned no assistant text (content is None and no reasoning_content). "
                f"finish_reason={fin!r}"
            )
    else:
        if finish_reason == "length":
            raise ContextLengthError(
                "finish_reason=length — output truncated, context likely full",
                prune_individual=True,
            )
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
