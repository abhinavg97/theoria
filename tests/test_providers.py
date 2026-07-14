import asyncio
import json
import socket
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import harness
import llm
import pipeline
import providers
import sandbox


AZURE_ENDPOINT = "https://unit-test.openai.azure.com/openai/v1"
ACTIVE_ROLES = {
    "solver", "interpreter", "formalizer", "citation", "problem_given",
    "computation", "pedantry", "convention_lift", "initial_state",
}


def azure_settings(**overrides):
    settings = {
        "backend": "codex",
        "model": "gpt-5.3-codex",
        "effort": "medium",
        "provider": {
            "kind": "azure_openai",
            "endpoint": AZURE_ENDPOINT,
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "max_parallel": 4,
        },
    }
    settings.update(overrides)
    return settings


def test_doctor_reports_malformed_azure_endpoint_without_traceback(
    monkeypatch, capsys,
):
    malformed = azure_settings(provider={
        "kind": "azure_openai",
        "endpoint": "http://unit-test.openai.azure.com/openai/v1",
        "api_key_env": "AZURE_OPENAI_API_KEY",
    })
    monkeypatch.setattr(cli.harness, "load_config", lambda _: {
        "solver": malformed,
    })
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    args = SimpleNamespace(
        config=["ignored.yaml"],
        codex_model=None,
        docker=False,
        image="unused",
        check_endpoint=True,
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_doctor(args)

    output = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "provider security configuration is valid" in output.out
    assert "Azure OpenAI endpoint must use https" in output.out
    assert "Traceback" not in output.out + output.err


def config_values(command):
    return [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "-c"
    ]


def test_azure_profile_configures_every_role_and_stacks_with_shell_search():
    root = Path(__file__).parents[1]
    config = pipeline.load_config([
        root / "configs" / "azure_openai.yaml",
        root / "configs" / "brave_search.yaml",
    ])

    for role in ACTIVE_ROLES:
        settings = config[role]
        assert settings["backend"] == "codex"
        assert settings["model"] == "REPLACE_WITH_AZURE_DEPLOYMENT"
        assert settings["provider"]["kind"] == "azure_openai"
        assert settings["provider"]["api_key_env"] == "AZURE_OPENAI_API_KEY"
        assert settings["provider"]["max_parallel"] == 4
        assert settings["search"] is False


def test_structured_azure_provider_compiles_exact_codex_0133_contract():
    spec = providers.resolve_provider_spec(azure_settings())

    assert spec.kind == "azure_openai"
    assert spec.forwarded_env == ("AZURE_OPENAI_API_KEY",)
    assert spec.native_search_default is False
    assert spec.max_parallel == 4
    assert spec.capabilities.as_dict() == {
        "native_web_search": True,
        "namespace_tools": False,
        "unified_exec": True,
        "structured_outputs": True,
    }
    assert spec.codex_config_items[:5] == (
        ("model_provider", "azure"),
        ("model_providers.azure.name", "Azure"),
        ("model_providers.azure.base_url", AZURE_ENDPOINT),
        ("model_providers.azure.env_key", "AZURE_OPENAI_API_KEY"),
        ("model_providers.azure.wire_api", "responses"),
    )


def test_azure_command_uses_deployment_and_capability_defaults_on_initial_and_resume():
    settings = azure_settings()
    initial = llm._build_codex_cmd(
        "prompt", settings, "/tmp/schema.json", "system", None,
        role="formalizer",
    )
    resumed = llm._build_codex_cmd(
        "repair", settings, "/tmp/schema.json", None, "thread-id",
        role="solver",
    )

    for command in (initial, resumed):
        values = config_values(command)
        assert command[command.index("--model") + 1] == "gpt-5.3-codex"
        assert "--oss" not in command
        assert 'model_provider="azure"' in values
        assert 'model_providers.azure.name="Azure"' in values
        assert f'model_providers.azure.base_url="{AZURE_ENDPOINT}"' in values
        assert 'model_providers.azure.env_key="AZURE_OPENAI_API_KEY"' in values
        assert 'model_providers.azure.wire_api="responses"' in values
        assert 'web_search="disabled"' in values
        assert "features.multi_agent=false" in values
        assert "features.multi_agent_v2=false" in values
        assert "features.unified_exec=false" not in values


def test_azure_omitted_search_uses_no_search_prompt_policy():
    settings = azure_settings()

    assert "search" not in settings
    assert pipeline.role_search_policy(settings, {}) == pipeline.NO_SEARCH_POLICY


def test_azure_native_search_is_explicit_and_cannot_be_overridden_indirectly():
    settings = azure_settings(
        search=True,
        codex_config={"web_search": "disabled"},
    )
    command = llm._build_codex_cmd(
        "prompt", settings, None, None, None,
    )
    values = config_values(command)

    assert values.count('web_search="live"') == 1
    assert values[-1] == 'web_search="live"'


@pytest.mark.parametrize("feature", ["multi_agent", "multi_agent_v2"])
def test_azure_namespace_capability_is_an_upper_bound(feature):
    with pytest.raises(ValueError, match="does not support Codex namespace"):
        providers.resolve_codex_role(azure_settings(
            codex_config={f"features.{feature}": True},
        ))


def test_services_endpoint_and_legacy_raw_azure_are_canonicalized():
    endpoint = "https://project.services.ai.azure.com/openai/v1"
    structured = azure_settings(provider={
        "kind": "azure_openai",
        "endpoint": endpoint,
        "api_key_env": "AZURE_OPENAI_API_KEY",
    })
    structured["model"] = "deployment"
    assert providers.resolve_provider_spec(structured).base_url == endpoint

    legacy = {
        "model": "deployment",
        "codex_config": {
            "model_provider": "foundry",
            "model_providers.foundry.name": "Azure OpenAI",
            "model_providers.foundry.base_url": endpoint,
            "model_providers.foundry.env_key": "AZURE_OPENAI_API_KEY",
            "model_providers.foundry.wire_api": "responses",
        },
        "provider_env": ["AZURE_OPENAI_API_KEY"],
    }
    spec = providers.resolve_provider_spec(legacy)
    assert spec.kind == "azure_openai"
    assert spec.max_parallel == 4
    assert providers.resolve_codex_role(legacy).concurrency_key is not None
    assert (
        providers.resolve_codex_role(structured).concurrency_key
        == providers.resolve_codex_role(legacy).concurrency_key
    )
    assert ("model_providers.foundry.name", "Azure") in spec.codex_config_items
    assert ("model_providers.foundry.wire_api", "responses") in spec.codex_config_items


def raw_named_provider(endpoint, *, name="Proxy"):
    return {
        "model": "deployment",
        "codex_config": {
            "model_provider": "proxy",
            "model_providers.proxy.name": name,
            "model_providers.proxy.base_url": endpoint,
            "model_providers.proxy.env_key": "AZURE_OPENAI_API_KEY",
            "model_providers.proxy.wire_api": "responses",
        },
        "provider_env": ["AZURE_OPENAI_API_KEY"],
    }


@pytest.mark.parametrize("name", ["Azure", "aZuRe"])
def test_raw_exact_azure_display_name_cannot_bypass_endpoint_validation(name):
    with pytest.raises(ValueError, match="supported Azure hostname"):
        providers.resolve_provider_spec(raw_named_provider(
            "https://proxy.example/openai/v1", name=name,
        ))


@pytest.mark.parametrize("endpoint", [
    "https://resource.openai.azure.us/openai/v1",
    "https://resource.cognitiveservices.azure.com/openai/v1",
    "https://resource.aoai.azure.net/openai/v1",
    "https://resource.azure-api.net/openai/v1",
    "https://resource.azurefd.net/openai/v1",
    "https://resource.windows.net/openai/v1",
])
def test_raw_codex_azure_url_markers_fail_closed(endpoint):
    with pytest.raises(ValueError, match="supported Azure hostname"):
        providers.resolve_provider_spec(raw_named_provider(endpoint))


@pytest.mark.parametrize("endpoint", [
    "http://resource.openai.azure.com/openai/v1",
    "https://resource.openai.azure.com/openai",
    "https://resource.openai.azure.com/openai/deployments/example",
    "https://resource.openai.azure.com/openai/v1?api-version=2025-01-01",
    "https://example.com/openai/v1",
])
def test_azure_endpoint_validation_rejects_non_ga_shapes(endpoint):
    with pytest.raises(ValueError, match="Azure OpenAI endpoint"):
        providers.resolve_provider_spec(azure_settings(provider={
            "kind": "azure_openai",
            "endpoint": endpoint,
            "api_key_env": "AZURE_OPENAI_API_KEY",
        }))


def test_structured_provider_conflicts_with_raw_provider_selection():
    with pytest.raises(ValueError, match="cannot be combined"):
        providers.resolve_provider_spec(azure_settings(codex_config={
            "model_provider": "custom",
        }))


def test_raw_azure_requires_env_key_and_forwarding():
    settings = {
        "model": "deployment",
        "codex_config": {
            "model_provider": "azure",
            "model_providers.azure.base_url": AZURE_ENDPOINT,
        },
    }
    with pytest.raises(ValueError, match=r"requires model_providers\.azure\.env_key"):
        providers.resolve_provider_spec(settings)


def test_raw_azure_requires_selected_provider_env_key_not_unrelated_one():
    settings = {
        "model": "deployment",
        "codex_config": {
            "model_provider": "azure",
            "model_providers.azure.base_url": AZURE_ENDPOINT,
            "model_providers.other.env_key": "OTHER_API_KEY",
        },
        "provider_env": ["OTHER_API_KEY"],
    }
    with pytest.raises(
        ValueError, match=r"model_providers\.azure\.env_key",
    ):
        providers.resolve_provider_spec(settings)


def test_structured_provider_rejects_unaccounted_advanced_provider_refs():
    settings = azure_settings(codex_config={
        "model_providers.proxy.env_key": "PROXY_API_KEY",
    })
    with pytest.raises(ValueError, match="raw Codex provider selection"):
        providers.resolve_provider_spec(settings)


def test_non_azure_legacy_wire_api_metadata_matches_selected_value():
    spec = providers.resolve_provider_spec({
        "model": "custom-model",
        "codex_config": {
            "model_provider": "custom",
            "model_providers.custom.base_url": "https://models.example/v1",
            "model_providers.custom.wire_api": "chat_completions",
        },
    })
    assert spec.wire_api == "chat_completions"


def test_legacy_oss_environment_endpoint_takes_precedence_and_enters_cache(monkeypatch):
    settings = {
        "model": "local-model",
        "oss": True,
        "local_provider": "ollama",
        "oss_base_url": "http://localhost:11434/v1",
        "search": False,
    }
    monkeypatch.setenv("CODEX_OSS_BASE_URL", "http://localhost:22434/v1")
    spec = providers.resolve_provider_spec(settings)
    identity_kwargs = dict(
        prompt="prompt",
        system=None,
        schema=None,
        role="solver",
        backend="codex",
        settings=settings,
        watch=False,
        resume=None,
        sandboxed=True,
        image_id="sha256:image",
        codex_version="codex-cli 0.133.0",
    )
    identity = llm._call_cache_identity(**identity_kwargs)
    monkeypatch.setenv("CODEX_OSS_BASE_URL", "http://localhost:33434/v1")
    changed_identity = llm._call_cache_identity(**identity_kwargs)

    assert spec.base_url == "http://localhost:22434/v1"
    assert identity["sha256"] != changed_identity["sha256"]
    assert "CODEX_OSS_BASE_URL" not in repr(identity)


def test_runtime_auto_forwards_azure_key_and_records_deployment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-value")
    runtime = harness.resolve_runtime({"solver": azure_settings()})

    assert runtime["requirements"]["codex_cloud_auth"] is False
    assert runtime["requirements"]["external_provider"] is True
    assert runtime["provider_env"] == [
        {"name": "AZURE_OPENAI_API_KEY", "present": True},
    ]
    role = runtime["roles"]["solver"]
    assert role["deployment"] == "gpt-5.3-codex"
    assert role["provider"]["kind"] == "azure_openai"
    assert role["provider"]["native_search"] is False
    assert "secret-value" not in repr(runtime)


def test_azure_cache_identity_tracks_deployment_but_not_key_value(monkeypatch):
    kwargs = dict(
        prompt="prompt",
        system="system",
        schema={"type": "object"},
        role="solver",
        backend="codex",
        settings=azure_settings(search=False),
        watch=False,
        resume=None,
        sandboxed=True,
        image_id="sha256:image",
        codex_version="codex-cli 0.133.0",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "first-secret")
    first = llm._call_cache_identity(**kwargs)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "rotated-secret")
    rotated = llm._call_cache_identity(**kwargs)
    changed = llm._call_cache_identity(**{
        **kwargs,
        "settings": azure_settings(model="another-deployment", search=False),
    })

    assert first == rotated
    assert first["sha256"] != changed["sha256"]
    assert "first-secret" not in repr(first)
    assert "rotated-secret" not in repr(rotated)


def test_cache_identity_tracks_normalized_shell_search():
    base = dict(
        prompt="prompt",
        system=None,
        schema=None,
        role="solver",
        backend="codex",
        settings=azure_settings(search=False),
        watch=False,
        resume=None,
        sandboxed=True,
        image_id="sha256:image",
        codex_version="codex-cli 0.133.0",
    )
    brave = llm._call_cache_identity(
        **base,
        web_search={"provider": "brave", "api_key_env": "BRAVE_API_KEY"},
    )
    searxng = llm._call_cache_identity(
        **base,
        web_search={"provider": "searxng", "endpoint": "http://localhost:8888"},
    )
    assert brave["sha256"] != searxng["sha256"]


def test_azure_provider_gate_limits_parallel_calls(monkeypatch):
    active = 0
    max_active = 0

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            events = [
                {"type": "thread.started", "thread_id": "thread"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "ok"},
                },
            ]
            return "\n".join(json.dumps(event) for event in events).encode(), b""

    async def fake_create_subprocess_exec(*_command, **_kwargs):
        return FakeProcess()

    monkeypatch.setattr(llm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    config = {
        "_security": {"allow_external_provider_host_access": True},
        "judge": azure_settings(provider={
            "kind": "azure_openai",
            "endpoint": AZURE_ENDPOINT,
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "max_parallel": 2,
        }),
    }

    async def run_calls():
        await asyncio.gather(*[
            llm.llm(f"judge {index}", role="judge", config=config)
            for index in range(5)
        ])

    asyncio.run(run_calls())
    assert max_active == 2


def test_provider_gate_changed_limit_is_fresh_for_each_event_loop():
    async def create_gate(limit):
        loop = asyncio.get_running_loop()
        gate = llm._provider_gate("shared-provider", limit)
        registry = getattr(loop, llm._PROVIDER_GATES_ATTR)
        return gate, registry

    first_gate, first_registry = asyncio.run(create_gate(2))
    second_gate, second_registry = asyncio.run(create_gate(5))

    assert first_gate is not second_gate
    assert first_registry is not second_registry
    assert first_gate._value == 2
    assert second_gate._value == 5


def test_provider_gate_limit_conflict_does_not_escape_closed_loop():
    async def conflict_in_first_loop():
        llm._provider_gate("conflicting-provider", 2)
        with pytest.raises(ValueError, match="same max_parallel"):
            llm._provider_gate("conflicting-provider", 5)

    async def use_changed_limit_in_fresh_loop():
        first = llm._provider_gate("conflicting-provider", 5)
        second = llm._provider_gate("conflicting-provider", 5)
        assert first is second
        assert first._value == 5

    asyncio.run(conflict_in_first_loop())
    asyncio.run(use_changed_limit_in_fresh_loop())


def test_azure_gate_key_is_deployment_scoped():
    first = providers.resolve_codex_role(azure_settings(model="deployment-a"))
    same = providers.resolve_codex_role(azure_settings(model="deployment-a"))
    other = providers.resolve_codex_role(azure_settings(model="deployment-b"))

    assert first.concurrency_key == same.concurrency_key
    assert first.concurrency_key != other.concurrency_key


def test_local_gate_is_endpoint_scoped_across_models():
    base = {
        "oss": True,
        "local_provider": "ollama",
        "oss_base_url": "http://localhost:11434/v1",
        "search": False,
    }
    first = providers.resolve_codex_role({**base, "model": "model-a"})
    other = providers.resolve_codex_role({**base, "model": "model-b"})
    assert first.concurrency_key == other.concurrency_key


def test_runtime_rejects_conflicting_limits_for_same_azure_deployment():
    first = azure_settings(provider={
        "kind": "azure_openai",
        "endpoint": AZURE_ENDPOINT,
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "max_parallel": 2,
    })
    second = azure_settings(provider={
        "kind": "azure_openai",
        "endpoint": AZURE_ENDPOINT,
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "max_parallel": 5,
    })

    with pytest.raises(ValueError, match="same max_parallel"):
        harness.resolve_runtime({"solver": first, "citation": second})


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_authenticated_azure_probe_uses_models_and_bearer_without_query():
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    result = providers.probe_azure_endpoint(
        providers.resolve_provider_spec(azure_settings()),
        environ={"AZURE_OPENAI_API_KEY": "test-secret"},
        opener=opener,
    )

    assert result == providers.ProviderProbeResult(True, "ok", 200)
    assert captured["request"].full_url == AZURE_ENDPOINT + "/models"
    assert captured["request"].get_header("Authorization") == "Bearer test-secret"
    assert "test-secret" not in repr(result)


@pytest.mark.parametrize("key", [
    " test-secret",
    "test-secret ",
    "test-secret\n",
    "test-secret\ninjected",
])
def test_authenticated_azure_probe_rejects_invalid_header_without_network(key):
    def opener(*_args, **_kwargs):
        raise AssertionError("invalid credential reached the network opener")

    result = providers.probe_azure_endpoint(
        providers.resolve_provider_spec(azure_settings()),
        environ={"AZURE_OPENAI_API_KEY": key},
        opener=opener,
    )

    assert result == providers.ProviderProbeResult(
        False, "invalid_credential",
    )
    assert "test-secret" not in repr(result)


def test_authenticated_azure_probe_treats_whitespace_only_key_as_missing():
    result = providers.probe_azure_endpoint(
        providers.resolve_provider_spec(azure_settings()),
        environ={"AZURE_OPENAI_API_KEY": " \t\n"},
        opener=lambda *_args, **_kwargs: pytest.fail("opener was called"),
    )

    assert result == providers.ProviderProbeResult(
        False, "missing_credential",
    )


@pytest.mark.parametrize(("error", "category"), [
    (urllib.error.HTTPError("url", 401, "", {}, None), "unauthorized"),
    (urllib.error.HTTPError("url", 403, "", {}, None), "forbidden"),
    (urllib.error.HTTPError("url", 404, "", {}, None), "not_found"),
    (urllib.error.HTTPError("url", 429, "", {}, None), "rate_limited"),
    (urllib.error.URLError(socket.gaierror()), "dns"),
    (urllib.error.URLError(socket.timeout()), "timeout"),
])
def test_authenticated_azure_probe_classifies_failures(error, category):
    def opener(_request, timeout):
        raise error

    result = providers.probe_azure_endpoint(
        providers.resolve_provider_spec(azure_settings()),
        environ={"AZURE_OPENAI_API_KEY": "test-secret"},
        opener=opener,
    )
    assert result.ok is False
    assert result.category == category


def test_docker_azure_probe_passes_only_env_name_and_never_secret(monkeypatch):
    sentinel = "sentinel-super-secret-key"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["result"] = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "category": "ok", "status": 200}),
            stderr="",
        )
        return captured["result"]

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", sentinel)
    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    result = sandbox.azure_endpoint_probe_from_image(
        "sandbox:test", AZURE_ENDPOINT, "AZURE_OPENAI_API_KEY",
    )

    assert result == providers.ProviderProbeResult(True, "ok", 200)
    command = captured["command"]
    assert command[command.index("--env") + 1] == "AZURE_OPENAI_API_KEY"
    assert "AZURE_OPENAI_API_KEY=" not in repr(command)
    assert sentinel not in repr(command)
    assert sentinel not in captured["result"].stdout
    assert sentinel not in captured["result"].stderr
    assert sentinel not in captured["kwargs"].get("input", "")
    assert sentinel not in json.dumps({
        "category": result.category, "status": result.status,
    })


def test_docker_azure_probe_missing_env_does_not_spawn(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("docker probe spawned without its credential")

    monkeypatch.setattr(sandbox.subprocess, "run", unexpected)
    result = sandbox.azure_endpoint_probe_from_image(
        "sandbox:test", AZURE_ENDPOINT, "AZURE_OPENAI_API_KEY",
    )
    assert result.category == "missing_credential"


@pytest.mark.parametrize("key", [
    " test-secret",
    "test-secret ",
    "test-secret\n",
    "test-secret\ninjected",
])
def test_docker_azure_probe_invalid_header_does_not_spawn(monkeypatch, key):
    monkeypatch.setenv(
        "AZURE_OPENAI_API_KEY", key,
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("docker probe spawned with an invalid credential")

    monkeypatch.setattr(sandbox.subprocess, "run", unexpected)
    result = sandbox.azure_endpoint_probe_from_image(
        "sandbox:test", AZURE_ENDPOINT, "AZURE_OPENAI_API_KEY",
    )
    assert result == providers.ProviderProbeResult(
        False, "invalid_credential",
    )
