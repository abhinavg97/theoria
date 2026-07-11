from pathlib import Path

import cli
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
