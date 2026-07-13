from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import llm
import pipeline


ACTIVE_ROLES = {
    "solver",
    "formalizer",
    "citation",
    "problem_given",
    "computation",
    "pedantry",
    "convention_lift",
    "initial_state",
}


def test_ollama_profile_configures_every_pipeline_role():
    root = Path(__file__).parents[1]
    config = pipeline.load_config(root / "configs" / "ollama.yaml")

    for role in ACTIVE_ROLES:
        settings = config[role]
        assert settings["backend"] == "codex"
        assert settings["model"] == "gpt-oss:20b"
        assert settings["oss"] is True
        assert settings["local_provider"] == "ollama"
        assert settings["effort"] is None
        assert settings["search"] is False
        assert settings["schema_retries"] == 1


def test_brave_search_profile_stacks_on_ollama_without_losing_provider():
    root = Path(__file__).parents[1]
    config = pipeline.load_config([
        root / "configs" / "ollama.yaml",
        root / "configs" / "brave_search.yaml",
    ])

    assert llm.web_search_config(config["_web_search"]) == {
        "provider": "brave",
        "api_key_env": "BRAVE_API_KEY",
    }
    for role in ACTIVE_ROLES:
        settings = config[role]
        # Provider wiring from ollama.yaml must survive the stack ...
        assert settings["backend"] == "codex"
        assert settings["oss"] is True
        assert settings["local_provider"] == "ollama"
        # ... while the search profile swaps the no-search policy for the
        # shell helper and keeps Codex's native search tool disabled.
        assert settings["search"] is False
        assert "theoria-search" in settings["prompt_suffix"]


def test_searxng_search_profile_stacks_on_ollama():
    root = Path(__file__).parents[1]
    config = pipeline.load_config([
        root / "configs" / "ollama.yaml",
        root / "configs" / "searxng_search.yaml",
    ])

    assert llm.web_search_config(config["_web_search"]) == {
        "provider": "searxng",
        "endpoint": "http://localhost:8888",
    }
    for role in ACTIVE_ROLES:
        settings = config[role]
        assert settings["oss"] is True
        assert settings["search"] is False
        assert "theoria-search" in settings["prompt_suffix"]


def test_custom_provider_profile_uses_env_reference_not_secret():
    root = Path(__file__).parents[1]
    config = pipeline.load_config(
        root / "configs" / "oss_custom_provider.example.yaml"
    )
    settings = config["solver"]

    assert settings["backend"] == "codex"
    assert settings.get("oss") is not True
    assert settings["provider_env"] == ["OSS_API_KEY"]
    assert settings["codex_config"]["model_providers.oss_custom.env_key"] == (
        "OSS_API_KEY"
    )
    assert "secret" not in repr(settings).lower()


def test_host_provider_opt_in_is_explicit_and_separate():
    root = Path(__file__).parents[1]
    config = pipeline.load_config([
        root / "configs" / "ollama.yaml",
        root / "configs" / "unsafe_host_provider.yaml",
    ])

    assert config["_security"] == {
        "allow_external_provider_host_access": True,
    }
    assert config["solver"]["oss"] is True


def test_config_overlays_merge_nested_mappings_without_aliasing(tmp_path):
    provider = tmp_path / "provider.yaml"
    provider.write_text("""
solver:
  codex_config: &provider
    model_provider: custom
    model_providers.custom.base_url: https://models.example/v1
formalizer:
  codex_config: *provider
""")
    features = tmp_path / "features.yaml"
    features.write_text("""
solver:
  codex_config:
    features.multi_agent: false
""")

    config = pipeline.load_config([provider, features])

    assert config["solver"]["codex_config"] == {
        "model_provider": "custom",
        "model_providers.custom.base_url": "https://models.example/v1",
        "features.multi_agent": False,
    }
    assert config["formalizer"]["codex_config"] == {
        "model_provider": "custom",
        "model_providers.custom.base_url": "https://models.example/v1",
    }
    config["solver"]["codex_config"]["model_provider"] = "changed"
    assert config["formalizer"]["codex_config"]["model_provider"] == "custom"


def test_doctor_parser_accepts_oss_configuration_options():
    args = cli.build_parser().parse_args([
        "doctor",
        "--config",
        "configs/ollama.yaml",
        "--codex-model",
        "local-model",
        "--no-docker",
        "--check-endpoint",
    ])

    assert args.config == ["configs/ollama.yaml"]
    assert args.codex_model == "local-model"
    assert args.docker is False
    assert args.check_endpoint is True


def test_doctor_uses_shared_external_provider_predicate(monkeypatch, capsys):
    monkeypatch.setattr(cli.harness, "load_config", lambda _paths: {
        "_security": {"allow_external_provider_host_access": True},
        "solver": {
            "backend": "codex",
            "model": "local-model",
            "oss": True,
            "local_provider": "ollama",
            "search": False,
        },
    })
    monkeypatch.setattr(
        cli.shutil, "which",
        lambda binary: "/usr/bin/codex" if binary == "codex" else None,
    )

    def fake_run(command, **_kwargs):
        output = (
            "codex 0.133\n"
            "--oss --local-provider PROVIDER --output-schema FILE"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    args = SimpleNamespace(
        config=["ignored.yaml"],
        codex_model=None,
        docker=False,
        image="unused",
        check_endpoint=False,
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_doctor(args)

    output = capsys.readouterr()
    assert exit_info.value.code == 0
    assert "provider credential trust domains are isolated" in output.out
    assert "Traceback" not in output.out + output.err


def test_doctor_requires_local_only_flags_only_for_local_oss():
    assert cli._required_container_codex_capabilities(local_oss=False) == {
        "available",
        "output_schema",
    }
    assert cli._required_container_codex_capabilities(local_oss=True) == {
        "available",
        "output_schema",
        "oss",
        "local_provider",
    }


def test_grade_parser_accepts_codex_model_override():
    args = cli.build_parser().parse_args([
        "grade",
        "runs/example.json",
        "--config",
        "configs/ollama_audit_grader.yaml",
        "--codex-model",
        "local-model",
    ])

    assert args.codex_model == "local-model"
