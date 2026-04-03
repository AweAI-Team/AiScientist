from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from aisci_agent_runtime.llm_client import (
    CompletionsLLMClient,
    ResponsesLLMClient,
    create_llm_client,
)
from aisci_agent_runtime.llm_profiles import resolve_llm_profile
from aisci_domain_mle.shared_infra_bridge import (
    build_mle_session_env,
    domain_llm_profile_file,
    mle_runtime_repo_target,
    shared_llm_env,
)


class _FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="ok")],
                )
            ],
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class _FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=5),
        )


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.responses = _FakeResponsesAPI()
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsAPI())
        type(self).instances.append(self)


class _FakeAzureOpenAI:
    instances: list["_FakeAzureOpenAI"] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.responses = _FakeResponsesAPI()
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsAPI())
        type(self).instances.append(self)


class ModelProfileRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeOpenAI.instances.clear()
        _FakeAzureOpenAI.instances.clear()

    def test_profile_registry_loads_all_required_profiles(self) -> None:
        gpt = resolve_llm_profile("gpt-5.4", profile_file=domain_llm_profile_file())
        self.assertEqual(gpt.provider, "openai")
        self.assertEqual(gpt.model, "gpt-5.4")
        self.assertEqual(gpt.api_mode, "responses")
        self.assertTrue(gpt.use_phase)
        self.assertEqual(gpt.max_tokens, 131072)
        self.assertEqual(gpt.context_window, 1000000)

        glm = resolve_llm_profile("glm-5", profile_file=domain_llm_profile_file())
        self.assertEqual(glm.provider, "azure-openai")
        self.assertEqual(glm.model, "glm-5")
        self.assertEqual(glm.api_mode, "completions")
        self.assertEqual(glm.max_tokens, 65536)
        self.assertEqual(glm.context_window, 202752)
        self.assertTrue(glm.clear_thinking)

        gemini = resolve_llm_profile("gemini-3-flash", profile_file=domain_llm_profile_file())
        self.assertEqual(gemini.provider, "openai")
        self.assertEqual(gemini.model, "gemini-3-flash-preview")
        self.assertEqual(gemini.api_mode, "completions")
        self.assertEqual(gemini.reasoning_effort, "high")
        self.assertEqual(gemini.temperature, 1.0)
        self.assertEqual(gemini.max_tokens, 20480)
        self.assertEqual(gemini.context_window, 1000000)

    def test_shared_llm_env_preserves_provider_specific_fields(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-secret",
                "AZURE_OPENAI_ENDPOINT": "https://glm.example.invalid",
                "AZURE_OPENAI_API_KEY": "glm-secret",
            },
            clear=True,
        ):
            gpt_env = shared_llm_env("gpt-5.4")
            self.assertEqual(gpt_env["AISCI_PROVIDER"], "openai")
            self.assertEqual(gpt_env["AISCI_MODEL"], "gpt-5.4")
            self.assertEqual(gpt_env["AISCI_API_MODE"], "responses")
            self.assertEqual(gpt_env["AISCI_WEB_SEARCH"], "false")
            self.assertEqual(gpt_env["AISCI_USE_PHASE"], "true")
            self.assertEqual(gpt_env["AISCI_MAX_TOKENS"], "131072")
            self.assertEqual(gpt_env["AISCI_CONTEXT_WINDOW"], "1000000")
            self.assertNotIn("AISCI_REASONING_EFFORT", gpt_env)
            self.assertNotIn("AISCI_REASONING_SUMMARY", gpt_env)

            glm_env = build_mle_session_env(
                "glm-5",
                time_limit_secs=3600,
                competition_id="glm-job",
                hardware="gpu:0",
            )
            self.assertEqual(glm_env["AISCI_PROVIDER"], "azure-openai")
            self.assertEqual(glm_env["AISCI_MODEL"], "glm-5")
            self.assertEqual(glm_env["AISCI_API_MODE"], "completions")
            self.assertEqual(glm_env["AISCI_CLEAR_THINKING"], "true")
            self.assertEqual(glm_env["AISCI_MAX_TOKENS"], "65536")
            self.assertEqual(glm_env["AISCI_CONTEXT_WINDOW"], "202752")
            self.assertEqual(glm_env["AZURE_OPENAI_ENDPOINT"], "https://glm.example.invalid")
            self.assertEqual(glm_env["AZURE_OPENAI_API_KEY"], "glm-secret")
            self.assertEqual(glm_env["OPENAI_API_VERSION"], "2024-02-01")

            gemini_env = build_mle_session_env(
                "gemini-3-flash",
                time_limit_secs=3600,
                competition_id="demo-job",
                hardware="gpu:0",
            )
            self.assertEqual(gemini_env["AISCI_PROVIDER"], "openai")
            self.assertEqual(gemini_env["AISCI_MODEL"], "gemini-3-flash-preview")
            self.assertEqual(gemini_env["AISCI_API_MODE"], "completions")
            self.assertEqual(gemini_env["AISCI_REASONING_EFFORT"], "high")
            self.assertEqual(gemini_env["AISCI_TEMPERATURE"], "1.0")
            self.assertEqual(gemini_env["AISCI_MAX_TOKENS"], "20480")
            self.assertEqual(gemini_env["AISCI_CONTEXT_WINDOW"], "1000000")
            self.assertEqual(
                gemini_env["OPENAI_BASE_URL"],
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.assertEqual(gemini_env["AISCI_CONTEXT_REDUCE_STRATEGY"], "summary")
            self.assertEqual(gemini_env["TIME_LIMIT_SECS"], "3600")
            self.assertEqual(gemini_env["COMPETITION_ID"], "demo-job")
            self.assertEqual(gemini_env["HARDWARE"], "gpu:0")
            self.assertEqual(gemini_env["LOGS_DIR"], "/home/logs")
            self.assertEqual(gemini_env["AISCI_REPO_ROOT"], mle_runtime_repo_target())
            self.assertEqual(gemini_env["PYTHONPATH"], f"{mle_runtime_repo_target()}/src")
            self.assertNotIn("AISCI_HOME_ROOT", gemini_env)

    def test_gpt54_profile_reaches_responses_client_losslessly(self) -> None:
        with mock.patch(
            "aisci_agent_runtime.llm_client.OpenAI",
            side_effect=_FakeOpenAI,
        ):
            with mock.patch.dict(
                os.environ,
                {
                    **build_mle_session_env(
                        "gpt-5.4",
                        time_limit_secs=7200,
                        competition_id="gpt-job",
                        hardware="gpu:0",
                    ),
                    "OPENAI_API_KEY": "openai-secret",
                },
                clear=True,
            ):
                client = create_llm_client()
                self.assertIsInstance(client, ResponsesLLMClient)
                self.assertTrue(client.config.use_phase)
                self.assertFalse(client.config.web_search)
                self.assertIsNone(client.config.reasoning_effort)
                self.assertIsNone(client.config.reasoning_summary)
                self.assertEqual(client.config.prune_context_window, 868928)

                client.chat(
                    messages=[
                        {"role": "system", "content": "system"},
                        {
                            "role": "assistant",
                            "content": "calling tool",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "python", "arguments": "{}"},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                        {"role": "user", "content": "continue"},
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "python",
                                "description": "run python",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                )

        captured = _FakeOpenAI.instances[-1]
        self.assertEqual(captured.init_kwargs["api_key"], "openai-secret")
        request = captured.responses.calls[-1]
        self.assertEqual(request["model"], "gpt-5.4")
        self.assertNotIn({"type": "web_search"}, request["tools"])
        self.assertNotIn("reasoning", request)
        assistant_items = [
            item for item in request["input"]
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        self.assertTrue(any(item.get("phase") == "commentary" for item in assistant_items))

    def test_glm5_profile_reaches_azure_completions_client_losslessly(self) -> None:
        with mock.patch(
            "aisci_agent_runtime.llm_client.AzureOpenAI",
            side_effect=_FakeAzureOpenAI,
        ):
            with mock.patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_ENDPOINT": "https://glm.example.invalid",
                    "AZURE_OPENAI_API_KEY": "glm-secret",
                    "OPENAI_API_VERSION": "2024-02-01",
                },
                clear=True,
            ):
                env = build_mle_session_env(
                    "glm-5",
                    time_limit_secs=7200,
                    competition_id="glm-job",
                    hardware="gpu:0",
                )
            with mock.patch.dict(os.environ, env, clear=True):
                client = create_llm_client()
                self.assertIsInstance(client, CompletionsLLMClient)
                self.assertEqual(client.config.provider, "azure-openai")
                self.assertTrue(client.config.clear_thinking)
                self.assertEqual(client.config.api_version, "2024-02-01")
                self.assertEqual(client.config.prune_context_window, 103901)

                client.chat(messages=[{"role": "user", "content": "hello"}])

        captured = _FakeAzureOpenAI.instances[-1]
        self.assertEqual(captured.init_kwargs["azure_endpoint"], "https://glm.example.invalid")
        self.assertEqual(captured.init_kwargs["api_version"], "2024-02-01")
        request = captured.chat.completions.calls[-1]
        self.assertEqual(request["model"], "glm-5")
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "enabled", "clear_thinking": True}},
        )
        self.assertNotIn("temperature", request)

    def test_gemini_profile_reaches_openai_compatible_completions_client_losslessly(self) -> None:
        with mock.patch(
            "aisci_agent_runtime.llm_client.OpenAI",
            side_effect=_FakeOpenAI,
        ):
            with mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "gemini-secret",
                    "OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
                },
                clear=True,
            ):
                env = build_mle_session_env(
                    "gemini-3-flash",
                    time_limit_secs=7200,
                    competition_id="gemini-job",
                    hardware="gpu:0",
                )
            with mock.patch.dict(os.environ, env, clear=True):
                client = create_llm_client()
                self.assertIsInstance(client, CompletionsLLMClient)
                self.assertEqual(client.config.provider, "openai")
                self.assertEqual(
                    client.config.base_url,
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self.assertEqual(client.config.temperature, 1.0)
                self.assertEqual(client.config.reasoning_effort, "high")
                self.assertEqual(client.config.prune_context_window, 979520)

                client.chat(messages=[{"role": "user", "content": "hello"}])

        captured = _FakeOpenAI.instances[-1]
        self.assertEqual(
            captured.init_kwargs["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        request = captured.chat.completions.calls[-1]
        self.assertEqual(request["model"], "gemini-3-flash-preview")
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["extra_body"]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
