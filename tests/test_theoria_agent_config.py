from types import SimpleNamespace

import harness
import pipeline
from pipeline import load_config


ROLE_NAMES = [
    "solver",
    "interpreter",
    "formalizer",
    "citation",
    "problem_given",
    "computation",
    "pedantry",
    "convention_lift",
    "initial_state",
]


def test_theoria_agent_ollama_profile_configures_every_role():
    config = load_config("configs/theoria_agent_ollama.yaml")

    for role in ROLE_NAMES:
        assert config[role]["backend"] == "theoria_agent"
        assert config[role]["model"] == "qwen3:8b"
        assert config[role]["endpoint"] == "http://localhost:11434/v1"
        assert config[role]["allow_shell"] is True
        assert config[role]["allow_host_tools"] is False


def test_theoria_agent_azure_profile_configures_every_role():
    config = load_config("configs/theoria_agent_azure.example.yaml")

    for role in ROLE_NAMES:
        assert config[role]["backend"] == "theoria_agent"
        assert config[role]["model"] == "YOUR_AZURE_DEPLOYMENT"
        assert (
            config[role]["endpoint"]
            == "https://YOUR_RESOURCE.openai.azure.com/openai/v1"
        )
        assert config[role]["allow_shell"] is True
        assert config[role]["allow_host_tools"] is False


def test_searxng_search_profile_is_keyless():
    config = load_config([
        "configs/theoria_agent_ollama.yaml",
        "configs/searxng_search.yaml",
    ])

    assert config["_web_search"] == {
        "provider": "searxng",
        "endpoint": "http://localhost:8888",
    }
    assert "api_key" not in config["_web_search"]
    assert "api_key_env" not in config["_web_search"]


def test_harness_non_codex_backend_applies_to_formalizer(monkeypatch):
    original_config = dict(pipeline.CONFIG)
    pipeline.CONFIG.clear()
    try:
        args = SimpleNamespace(
            config=["configs/theoria_agent_ollama.yaml"],
            backend="theoria_agent",
            codex_model=None,
            watch=False,
            docker=False,
            image=None,
        )

        harness.apply_args(args)

        for role in ROLE_NAMES:
            assert pipeline.CONFIG[role]["backend"] == "theoria_agent"
        assert harness.config_uses_backend("theoria_agent") is True
        assert harness.config_uses_backend("claude") is False
    finally:
        pipeline.CONFIG.clear()
        pipeline.CONFIG.update(original_config)


def test_harness_codex_preset_keeps_formalizer_on_default_claude(monkeypatch):
    original_config = dict(pipeline.CONFIG)
    pipeline.CONFIG.clear()
    try:
        args = SimpleNamespace(
            config=None,
            backend="codex",
            codex_model=None,
            watch=False,
            docker=True,
            image=None,
        )

        harness.apply_args(args)

        assert pipeline.CONFIG["solver"]["backend"] == "codex"
        assert pipeline.CONFIG["formalizer"].get("backend") is None
        assert harness.config_uses_backend("claude") is True
    finally:
        pipeline.CONFIG.clear()
        pipeline.CONFIG.update(original_config)
