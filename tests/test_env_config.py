from __future__ import annotations

import os

from aisci_agent_runtime.llm_client import LLMConfig
from aisci_agent_runtime.llm_profiles import default_llm_profile_name, llm_env, resolve_llm_profile
from aisci_core.env_config import load_runtime_env
from aisci_runtime_docker.profiles import default_image_profile_name, resolve_image_profile


def test_load_runtime_env_reads_default_dotenv(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-key",
                "AISCI_MAX_STEPS=123",
                "export AISCI_REMINDER_FREQ=7",
                'QUOTED_VALUE="hello world"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AISCI_MAX_STEPS", raising=False)
    monkeypatch.delenv("AISCI_REMINDER_FREQ", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)

    loaded = load_runtime_env()

    assert loaded == [env_path]
    assert os.environ["OPENAI_API_KEY"] == "file-key"
    assert os.environ["AISCI_MAX_STEPS"] == "123"
    assert os.environ["AISCI_REMINDER_FREQ"] == "7"
    assert os.environ["QUOTED_VALUE"] == "hello world"


def test_load_runtime_env_does_not_override_existing_values(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")

    load_runtime_env()

    assert os.environ["OPENAI_API_KEY"] == "shell-key"


def test_load_runtime_env_allows_local_file_to_override_base_file(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("AISCI_MAX_STEPS=80\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("AISCI_MAX_STEPS=120\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AISCI_MAX_STEPS", raising=False)

    load_runtime_env()

    assert os.environ["AISCI_MAX_STEPS"] == "120"


def test_resolve_llm_profile_supports_gpt54_profile() -> None:
    profile = resolve_llm_profile("gpt-5.4")

    assert profile.model == "gpt-5.4"
    assert profile.api_mode == "responses"
    assert profile.use_phase is True
    assert profile.provider == "openai"
    assert profile.max_tokens == 131072
    assert profile.context_window == 1000000


def test_llm_config_derives_prune_budget_from_total_context_window() -> None:
    gpt = LLMConfig(model="gpt-5.4", provider="openai", max_tokens=131072, context_window=1000000)
    glm = LLMConfig(model="glm-5", provider="azure-openai", max_tokens=65536, context_window=202752)

    assert gpt.prune_context_window == 868928
    assert glm.prune_context_window == 103901


def test_llm_profiles_yaml_supports_defaults_and_extends(tmp_path) -> None:
    profile_file = tmp_path / "llm_profiles.yaml"
    profile_file.write_text(
        """
defaults:
  paper: paper-default
backends:
  openai:
    type: openai
    env:
      api_key:
        var: OPENAI_API_KEY
        required: true
profiles:
  paper-default:
    extends: fast
  fast:
    backend: openai
    model: gpt-fast
    api: responses
    limits:
      max_completion_tokens: 1234
    features:
      use_phase: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert default_llm_profile_name("paper", str(profile_file)) == "paper-default"
    profile = resolve_llm_profile(None, default_for="paper", profile_file=str(profile_file))

    assert profile.name == "paper-default"
    assert profile.model == "gpt-fast"
    assert profile.max_tokens == 1234
    assert profile.use_phase is True


def test_llm_env_uses_provider_env_mapping(tmp_path, monkeypatch) -> None:
    profile_file = tmp_path / "llm_profiles.yaml"
    profile_file.write_text(
        """
defaults:
  default: azure-default
backends:
  azure-openai:
    type: azure-openai
    env:
      endpoint:
        var: AZURE_OPENAI_ENDPOINT
        required: true
      api_key:
        var: AZURE_OPENAI_API_KEY
        required: true
profiles:
  azure-default:
    backend: azure-openai
    model: glm-5
    api: completions
    features:
      clear_thinking: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    env = llm_env("azure-default", profile_file=str(profile_file))

    assert env["AISCI_PROVIDER"] == "azure-openai"
    assert env["AISCI_MODEL"] == "glm-5"
    assert env["AISCI_API_MODE"] == "completions"
    assert env["AISCI_CLEAR_THINKING"] == "true"
    assert env["AZURE_OPENAI_ENDPOINT"] == "https://azure.example"
    assert env["AZURE_OPENAI_API_KEY"] == "test-key"


def test_llm_env_keeps_context_window_as_total_model_limit(tmp_path) -> None:
    profile_file = tmp_path / "llm_profiles.yaml"
    profile_file.write_text(
        """
defaults:
  default: paper-default
backends:
  openai:
    type: openai
    env:
      api_key:
        var: OPENAI_API_KEY
        required: true
profiles:
  paper-default:
    backend: openai
    model: gpt-5.4
    api: responses
    limits:
      max_completion_tokens: 1024
      context_window: 200000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    env = llm_env("paper-default", profile_file=str(profile_file))

    assert env["AISCI_CONTEXT_WINDOW"] == "200000"


def test_image_profiles_yaml_supports_defaults(tmp_path) -> None:
    profile_file = tmp_path / "image_profiles.yaml"
    profile_file.write_text(
        """
defaults:
  paper: paper-default
profiles:
  paper-default:
    image: registry.example/aisci-paper:latest
    pull_policy: if-missing
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert default_image_profile_name("paper", str(profile_file)) == "paper-default"
    profile = resolve_image_profile(None, default_for="paper", profile_file=str(profile_file))

    assert profile.name == "paper-default"
    assert profile.image == "registry.example/aisci-paper:latest"
    assert profile.pull_policy.value == "if-missing"
