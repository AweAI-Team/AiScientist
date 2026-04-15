"""Patched aideml backend router for MLE-Bench.

Upstream aideml v6.3.3 routes:
- ``glm-*`` → OpenRouter (wrong for Azure Chat Completions with GLM).
- ``gemini-*`` → ``google.generativeai`` (wrong for Gemini 3 OpenAI-compatible API).

This fork routes:
- ``glm-*`` → OpenAI-compatible client (``OpenAI`` or ``AzureOpenAI``; see backend_openai).
- ``gemini-3-*`` → OpenAI-compatible Google endpoint (set ``OPENAI_BASE_URL`` to Generative Language OpenAI API).
- other ``gemini-*`` → legacy GDM backend (unchanged).
"""

import logging

from . import backend_anthropic, backend_gdm, backend_openai, backend_openrouter
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("aide")


def determine_provider(model: str) -> str:
    if model.startswith("gpt-") or model.startswith("o1-"):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    # Gemini 3: OpenAI-compatible endpoint (same stack as aisci CompletionsLLMClient).
    elif model.startswith("gemini-3-"):
        return "openai"
    elif model.startswith("gemini-"):
        return "gdm"
    # Azure OpenAI Chat Completions for GLM (same as aisci glm-5).
    elif model.startswith("glm-"):
        return "openai"
    else:
        return "openrouter"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "gdm": backend_gdm.query,
    "openrouter": backend_openrouter.query,
}


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        convert_system_to_user=convert_system_to_user,
        **model_kwargs,
    )
    logger.info(f"response: {output}", extra={"verbose": True})
    logger.info("---Query complete---", extra={"verbose": True})

    return output
