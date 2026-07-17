from types import SimpleNamespace

import cli
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


def test_doctor_theoria_agent_config_does_not_require_claude_or_codex():
    args = SimpleNamespace(
        config=["configs/theoria_agent_ollama.yaml"],
        backend="theoria_agent",
        docker=False,
        image=None,
        no_pair_preamble=False,
    )

    config, backend, docker, image = cli._doctor_effective_config(args)

    assert backend == "theoria_agent"
    assert docker is False
    assert image == cli.SAGE_IMAGE
    assert cli._config_uses_backend(config, "theoria_agent") is True
    assert cli._config_uses_backend(config, "claude") is False
    assert cli._config_uses_backend(config, "codex") is False
    assert len(cli._theoria_agent_plans(config)) == 1


def test_doctor_default_codex_config_keeps_claude_formalizer():
    args = SimpleNamespace(
        config=None,
        backend=None,
        docker=None,
        image=None,
        no_pair_preamble=False,
    )

    config, backend, docker, image = cli._doctor_effective_config(args)

    assert backend == "codex"
    assert docker is True
    assert image == cli.SAGE_IMAGE
    assert cli._config_uses_backend(config, "codex") is True
    assert cli._config_uses_backend(config, "claude") is True


def test_doctor_endpoint_validation_rejects_credentials():
    assert cli._valid_http_endpoint("http://localhost:11434/v1") == (True, "")
    ok, hint = cli._valid_http_endpoint("https://user:pass@example.com/openai/v1")

    assert ok is False
    assert "credentials" in hint


def test_doctor_endpoint_validation_rejects_invalid_port():
    ok, hint = cli._valid_http_endpoint("http://localhost:notaport/v1")

    assert ok is False
    assert "Port could not be cast" in hint


def test_doctor_ignores_search_config_when_selected_roles_do_not_use_it():
    config = load_config("configs/searxng_search.yaml")

    assert cli._config_uses_web_search(config) is False


def test_doctor_detects_search_used_by_theoria_agent():
    config = load_config([
        "configs/theoria_agent_ollama.yaml",
        "configs/searxng_search.yaml",
    ])

    assert cli._config_uses_web_search(config) is True
