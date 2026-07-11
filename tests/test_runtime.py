import asyncio
from pathlib import Path
from types import SimpleNamespace

import harness
import sandbox


def test_resolve_runtime_all_oss_needs_no_subscription_auth(monkeypatch):
    monkeypatch.setattr(harness.sys, "platform", "linux")
    monkeypatch.delenv("CODEX_OSS_BASE_URL", raising=False)
    config = {
        "_shared": {"ignored": True},
        "solver": {
            "backend": "codex",
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
            "oss_base_url": "http://localhost:11434/v1",
        },
        "formalizer": {
            "backend": "codex",
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
            "oss_base_url": "http://localhost:11434/v1",
        },
    }

    runtime = harness.resolve_runtime(config)

    assert runtime["active_backends"] == ["codex"]
    assert runtime["requirements"] == {
        "claude_credentials": False,
        "codex_state": True,
        "codex_cloud_auth": False,
        "linux_host_gateway": True,
        "mixed_provider_credentials": False,
        "allow_mixed_provider_credentials": False,
        "external_provider": True,
        "allow_external_provider_host_access": False,
    }


def test_resolve_runtime_mixed_default_requires_both_auth_paths(monkeypatch):
    monkeypatch.delenv("CODEX_OSS_BASE_URL", raising=False)
    runtime = harness.resolve_runtime({
        "solver": {"backend": "codex", "model": "gpt"},
        "formalizer": {"backend": "claude", "model": "opus"},
    })

    assert runtime["requirements"]["claude_credentials"] is True
    assert runtime["requirements"]["codex_cloud_auth"] is True
    assert runtime["requirements"]["mixed_provider_credentials"] is False


def test_external_provider_cannot_share_subscription_credentials(monkeypatch):
    monkeypatch.delenv("CODEX_OSS_BASE_URL", raising=False)
    config = {
        "solver": {
            "backend": "codex",
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
        },
        "formalizer": {"backend": "claude", "model": "opus"},
    }

    runtime = harness.resolve_runtime(config)

    assert runtime["requirements"]["mixed_provider_credentials"] is True
    assert runtime["credential_domains"][0] == "claude"
    assert runtime["credential_domains"][1].startswith("codex:ollama:")


def test_custom_openai_endpoint_is_an_external_domain(monkeypatch):
    monkeypatch.delenv("CODEX_OSS_BASE_URL", raising=False)
    runtime = harness.resolve_runtime({
        "solver": {
            "backend": "codex",
            "model": "model-id",
            "codex_config": {
                "model_provider": "openai",
                "model_providers.openai.base_url": "https://proxy.example/v1",
                "model_providers.openai.env_key": "PROXY_API_KEY",
            },
            "provider_env": ["PROXY_API_KEY"],
        },
        "formalizer": {"backend": "claude", "model": "opus"},
    })

    assert runtime["requirements"]["codex_cloud_auth"] is False
    assert runtime["requirements"]["mixed_provider_credentials"] is True
    assert any(
        domain.startswith("codex:openai:")
        for domain in runtime["credential_domains"]
    )


def test_same_provider_id_with_different_endpoints_is_mixed(monkeypatch):
    monkeypatch.delenv("CODEX_OSS_BASE_URL", raising=False)

    def role(endpoint, env_name):
        return {
            "backend": "codex",
            "model": "model-id",
            "codex_config": {
                "model_provider": "custom",
                "model_providers.custom.base_url": endpoint,
                "model_providers.custom.env_key": env_name,
            },
            "provider_env": [env_name],
        }

    runtime = harness.resolve_runtime({
        "solver": role("https://first.example/v1", "FIRST_API_KEY"),
        "formalizer": role("https://second.example/v1", "SECOND_API_KEY"),
    })

    assert runtime["requirements"]["mixed_provider_credentials"] is True
    assert len(runtime["credential_domains"]) == 2


def test_mixed_provider_credentials_require_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "CONFIG", {
        "solver": {
            "backend": "codex",
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
        },
        "formalizer": {"backend": "claude", "model": "opus"},
    })

    try:
        asyncio.run(harness.run_problems(
            [], parallel=1, save_path=str(tmp_path / "run.json"),
        ))
    except RuntimeError as exc:
        assert "cannot share a run" in str(exc)
    else:
        raise AssertionError("mixed provider credentials were accepted")

    allowed = dict(harness.CONFIG)
    allowed["_security"] = {"allow_mixed_provider_credentials": True}
    runtime = harness.resolve_runtime(allowed)
    assert runtime["requirements"]["mixed_provider_credentials"] is True
    assert runtime["requirements"]["allow_mixed_provider_credentials"] is True


def test_metadata_sanitizer_keeps_env_name_but_removes_secret_value():
    sanitized = harness._sanitize_for_metadata({
        "provider_env": ["MODEL_API_KEY"],
        "api_key": "secret-value",
        "headers.Authorization": "Bearer secret-value",
        "headers.X-Key": "third-secret",
        "access_key": "another-secret",
        "base_url": "https://user:pass@example.test/v1?token=secret",
        "credential_domains": ["codex:custom:fingerprint"],
        "allow_mixed_provider_credentials": True,
    })

    assert sanitized["provider_env"] == ["MODEL_API_KEY"]
    assert sanitized["api_key"] == "<redacted>"
    assert sanitized["headers.Authorization"] == "<redacted>"
    assert sanitized["headers.X-Key"] == "<redacted>"
    assert sanitized["access_key"] == "<redacted>"
    assert "user" not in sanitized["base_url"]
    assert "secret" not in sanitized["base_url"]
    assert sanitized["credential_domains"] == ["codex:custom:fingerprint"]
    assert sanitized["allow_mixed_provider_credentials"] is True


def test_prepare_codex_state_is_writable_without_cloud_auth(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".codex"
    state.mkdir(parents=True)
    (state / "config.toml").write_text('model_provider = "ollama"\n')
    (state / "auth.json").write_text('{"token": "secret"}')
    monkeypatch.setattr(sandbox.os.path, "expanduser", lambda _path: str(state))

    prepared = Path(sandbox.prepare_codex_state_dir(
        include_auth=False, copy_host_state=False,
    ))
    try:
        assert prepared.is_dir()
        assert not (prepared / "config.toml").exists()
        assert not (prepared / "auth.json").exists()
    finally:
        sandbox.cleanup_codex_state_dir(str(prepared))


def test_prepared_cloud_state_is_private_and_excludes_host_config(
    tmp_path, monkeypatch,
):
    state = tmp_path / ".codex"
    state.mkdir()
    (state / "auth.json").write_text('{"token": "secret"}')
    (state / "config.toml").write_text('model_provider = "custom"\n')
    monkeypatch.setattr(sandbox.os.path, "expanduser", lambda _path: str(state))

    prepared = Path(sandbox.prepare_codex_state_dir())
    try:
        assert (prepared / "auth.json").exists()
        assert not (prepared / "config.toml").exists()
        assert prepared.stat().st_mode & 0o777 == 0o700
        assert (prepared / "auth.json").stat().st_mode & 0o777 == 0o600
    finally:
        sandbox.cleanup_codex_state_dir(str(prepared))


def test_start_sandbox_forwards_env_name_and_linux_gateway(monkeypatch, tmp_path):
    captured = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    monkeypatch.setenv("MODEL_API_KEY", "secret")

    container_id = sandbox.start_sandbox(
        "problem",
        "run",
        codex_state_dir=str(tmp_path),
        enable_claude=False,
        provider_env_names=["MODEL_API_KEY"],
        add_host_gateway=True,
    )

    command = captured[0]
    assert container_id == "container-id"
    assert "--add-host=host.docker.internal:host-gateway" in command
    assert command[command.index("-e") + 1] == "MODEL_API_KEY"
    assert "secret" not in command
    assert not any(".credentials.json" in value for value in command)
    assert "--tmpfs" in command


def test_container_inspect_redacts_forwarded_provider_values():
    redacted = sandbox.redact_container_inspect(
        {"Config": {"Env": ["MODEL_API_KEY=secret", "PATH=/usr/bin"]}},
        provider_env_names=["MODEL_API_KEY"],
    )

    assert redacted["Config"]["Env"] == [
        "MODEL_API_KEY=<redacted>",
        "PATH=/usr/bin",
    ]


def test_container_capability_and_endpoint_checks_use_selected_image(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[-2:] == ["exec", "--help"]:
            output = "--oss --local-provider PROVIDER --output-schema FILE"
        else:
            output = "200"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.sys, "platform", "linux")

    assert all(sandbox.codex_oss_capabilities_in_image("sandbox:test").values())
    assert sandbox.endpoint_reachable_from_image(
        "sandbox:test",
        "http://localhost:11434/v1",
        add_host_gateway=True,
    )

    endpoint_command = commands[-1]
    assert "--add-host=host.docker.internal:host-gateway" in endpoint_command
    assert "http://host.docker.internal:11434/v1/models" in endpoint_command


def test_all_oss_batch_skips_claude_credentials(monkeypatch, tmp_path):
    config = {
        "solver": {
            "backend": "codex",
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
        },
        "formalizer": {
            "backend": "codex",
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
        },
    }
    monkeypatch.setattr(harness, "CONFIG", config)
    monkeypatch.setitem(harness._args_ref, "docker", True)
    monkeypatch.setitem(harness._args_ref, "image", "sandbox:test")
    monkeypatch.setattr(
        harness.sbx,
        "refresh_claude_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("Claude auth requested")),
    )
    monkeypatch.setattr(harness.sbx, "image_digest", lambda _image: "sha256:test")
    monkeypatch.setattr(harness.sbx, "container_tool_versions", lambda _image: {})
    monkeypatch.setattr(
        harness, "make_artifact_root", lambda _path: str(tmp_path / "artifacts"),
    )
    monkeypatch.setattr(harness, "update_artifact_root_meta", lambda *_args: None)
    monkeypatch.setattr(harness, "finalize_artifact_root", lambda _root: None)

    results = asyncio.run(harness.run_problems(
        [], parallel=1, save_path=str(tmp_path / "run.json"),
    ))

    assert results == []


def test_batch_fails_fast_when_provider_env_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
    monkeypatch.setattr(harness, "CONFIG", {
        "solver": {
            "backend": "codex",
            "model": "model-id",
            "codex_config": {"model_provider": "custom"},
            "provider_env": ["MISSING_PROVIDER_KEY"],
        },
    })

    try:
        asyncio.run(harness.run_problems(
            [], parallel=1, save_path=str(tmp_path / "run.json"),
        ))
    except RuntimeError as exc:
        assert "MISSING_PROVIDER_KEY" in str(exc)
    else:
        raise AssertionError("missing provider environment was accepted")
