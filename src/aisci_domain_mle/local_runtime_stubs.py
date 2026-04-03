from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import Any


class _NoopLogger:
    def debug(self, *args: Any, **kwargs: Any) -> None:
        return None

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    if importlib.util.find_spec(name) is None:
        return False
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def install_optional_dependency_stubs() -> None:
    """Provide tiny fallbacks for local smoke/test environments.

    The real production runtime is expected to have these dependencies installed.
    """
    if not _module_available("structlog"):
        structlog_module = types.ModuleType("structlog")
        structlog_module.stdlib = types.SimpleNamespace(
            get_logger=lambda **kwargs: _NoopLogger(),
        )
        sys.modules["structlog"] = structlog_module

    if not _module_available("logid"):
        logid_module = types.ModuleType("logid")
        logid_module.generate_v2 = lambda: "stub-logid"
        sys.modules["logid"] = logid_module

    if not _module_available("openai"):
        openai_module = types.ModuleType("openai")

        class _OpenAIStubError(Exception):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args)
                self.code = kwargs.get("code")
                self.message = kwargs.get("message", args[0] if args else "")
                self.request = kwargs.get("request")
                self.request_id = kwargs.get("request_id")

        class OpenAI:
            pass

        class AzureOpenAI:
            pass

        openai_module.OpenAI = OpenAI
        openai_module.AzureOpenAI = AzureOpenAI
        openai_module.BadRequestError = _OpenAIStubError
        openai_module.PermissionDeniedError = _OpenAIStubError
        openai_module.RateLimitError = _OpenAIStubError
        openai_module.APIConnectionError = _OpenAIStubError
        openai_module.APITimeoutError = _OpenAIStubError
        openai_module.InternalServerError = _OpenAIStubError
        sys.modules["openai"] = openai_module

        openai_types = types.ModuleType("openai.types")
        openai_shared_params = types.ModuleType("openai.types.shared_params")
        openai_reasoning = types.ModuleType("openai.types.shared_params.reasoning")

        class Reasoning(dict):
            pass

        openai_reasoning.Reasoning = Reasoning
        sys.modules["openai.types"] = openai_types
        sys.modules["openai.types.shared_params"] = openai_shared_params
        sys.modules["openai.types.shared_params.reasoning"] = openai_reasoning
