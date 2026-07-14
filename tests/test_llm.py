import asyncio
import json

import pytest

import llm


def _config_values(command):
    return [command[index + 1] for index, value in enumerate(command) if value == "-c"]


def test_cloud_codex_command_keeps_existing_defaults():
    command = llm._build_codex_cmd(
        "prompt",
        {"model": "opus", "effort": "max", "search": True},
        None,
        "system",
        None,
        role="solver",
    )

    assert command[:4] == ["codex", "exec", "--model", "gpt-5.5"]
    assert "--oss" not in command
    assert 'model_provider="openai"' in _config_values(command)
    assert "model_reasoning_effort=xhigh" in _config_values(command)
    assert 'web_search="live"' in _config_values(command)


def test_oss_codex_initial_command_uses_local_provider_without_effort():
    command = llm._build_codex_cmd(
        "prompt",
        {
            "model": "gpt-oss:20b",
            "effort": None,
            "oss": True,
            "local_provider": "ollama",
            "search": False,
        },
        "/tmp/schema.json",
        "system",
        None,
        role="formalizer",
    )

    assert command[:2] == ["codex", "exec"]
    assert command[command.index("--local-provider") + 1] == "ollama"
    assert "--oss" in command
    assert "--output-schema" in command
    assert not any(value.startswith("model_reasoning_effort=") for value in _config_values(command))
    assert 'web_search="disabled"' in _config_values(command)
    assert "features.multi_agent=false" in _config_values(command)
    assert "features.multi_agent_v2=false" in _config_values(command)
    assert "features.unified_exec=false" in _config_values(command)


def test_oss_codex_resume_reasserts_provider_and_schema():
    command = llm._build_codex_cmd(
        "repair",
        {
            "model": "gpt-oss:20b",
            "effort": None,
            "oss": True,
            "local_provider": "ollama",
            "search": False,
        },
        "/tmp/schema.json",
        None,
        "session-id",
        role="solver",
    )

    assert command[:4] == ["codex", "exec", "resume", "session-id"]
    assert "--oss" not in command
    assert 'model_provider="ollama"' in _config_values(command)
    assert command[command.index("--output-schema") + 1] == "/tmp/schema.json"
    assert 'web_search="disabled"' in _config_values(command)
    assert "features.multi_agent=false" in _config_values(command)
    assert "features.multi_agent_v2=false" in _config_values(command)
    assert "features.unified_exec=false" in _config_values(command)


def test_oss_codex_allows_explicit_feature_overrides():
    command = llm._build_codex_cmd(
        "prompt",
        {
            "model": "local-model",
            "effort": None,
            "oss": True,
            "local_provider": "lmstudio",
            "codex_config": {
                "features.multi_agent": True,
                "features.multi_agent_v2": True,
                "features.unified_exec": True,
            },
        },
        None,
        None,
        None,
    )

    values = _config_values(command)
    assert "features.multi_agent=true" in values
    assert "features.multi_agent=false" not in values
    assert "features.multi_agent_v2=true" in values
    assert "features.multi_agent_v2=false" not in values
    assert "features.unified_exec=true" in values
    assert "features.unified_exec=false" not in values


def test_oss_calls_are_serialized_within_one_event_loop(monkeypatch):
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
                {"type": "thread.started", "thread_id": "t"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "ok"},
                },
            ]
            return "\n".join(json.dumps(e) for e in events).encode(), b""

    async def fake_create_subprocess_exec(*_command, **_kwargs):
        return FakeProcess()

    monkeypatch.setattr(
        llm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
    )
    monkeypatch.delenv("THEORIA_OSS_MAX_PARALLEL", raising=False)
    config = {
        "_security": {"allow_external_provider_host_access": True},
        "judge": {
            "backend": "codex",
            "model": "local-model",
            "oss": True,
            "local_provider": "ollama",
            "search": False,
        },
    }

    async def run_parallel():
        await asyncio.gather(*[
            llm.llm(f"judge {n}", role="judge", config=config)
            for n in range(4)
        ])

    asyncio.run(run_parallel())
    assert max_active == 1

    # Providers that really do handle concurrent requests can raise the
    # limit; a fresh event loop gets a fresh gate with the new value.
    monkeypatch.setenv("THEORIA_OSS_MAX_PARALLEL", "4")
    active = 0
    max_active = 0
    asyncio.run(run_parallel())
    assert max_active > 1


def test_oss_gate_lifetime_is_owned_by_actual_event_loop(monkeypatch):
    async def current_gate():
        return llm._oss_gate()

    # Make any legacy id(loop)-keyed implementation collide deterministically;
    # this test must not depend on CPython happening to recycle an object id.
    monkeypatch.setattr(llm, "id", lambda _value: 7, raising=False)

    monkeypatch.setenv("THEORIA_OSS_MAX_PARALLEL", "1")
    first_loop = asyncio.new_event_loop()
    try:
        first_gate = first_loop.run_until_complete(current_gate())
        assert first_gate._value == 1
        assert getattr(first_loop, llm._OSS_GATE_LOOP_ATTR) is first_gate
    finally:
        first_loop.close()

    monkeypatch.setenv("THEORIA_OSS_MAX_PARALLEL", "4")
    second_loop = asyncio.new_event_loop()
    try:
        second_gate = second_loop.run_until_complete(current_gate())
        assert second_gate is not first_gate
        assert second_gate._value == 4
        assert getattr(second_loop, llm._OSS_GATE_LOOP_ATTR) is second_gate
    finally:
        second_loop.close()


def test_sandboxed_codex_home_avoids_tmp_helper_warning():
    command = llm._build_codex_cmd(
        "prompt",
        {
            "model": "gpt-oss:20b",
            "oss": True,
            "local_provider": "ollama",
            "search": False,
        },
        None,
        None,
        None,
        sandboxed=True,
        role="computation",
    )

    assert command[0] == "bash"
    script = command[2]
    assert "CODEX_HOME=/home/node/.codex-call-" in script
    assert "/tmp/codex-" not in script


def test_custom_provider_config_is_structured_and_rejects_secrets():
    command = llm._build_codex_cmd(
        "prompt",
        {
            "model": "model-id",
            "effort": None,
            "codex_config": {
                "model_provider": "local",
                "model_providers.local.base_url": "https://models.example/v1",
                "model_providers.local.env_key": "MODEL_API_KEY",
                "model_providers.local.wire_api": "responses",
            },
            "provider_env": ["MODEL_API_KEY"],
        },
        None,
        None,
        None,
    )

    values = _config_values(command)
    assert 'model_provider="local"' in values
    assert 'model_providers.local.env_key="MODEL_API_KEY"' in values

    with pytest.raises(ValueError, match="expose a secret"):
        llm._build_codex_cmd(
            "prompt",
            {
                "model": "model-id",
                "codex_config": {"model_providers.local.api_key": "secret"},
            },
            None,
            None,
            None,
        )
    with pytest.raises(ValueError, match="also appear in provider_env"):
        llm._build_codex_cmd(
            "prompt",
            {
                "model": "model-id",
                "codex_config": {
                    "model_provider": "local",
                    "model_providers.local.env_key": "MODEL_API_KEY",
                },
            },
            None,
            None,
            None,
        )
    with pytest.raises(ValueError, match="expose a secret"):
        llm._build_codex_cmd(
            "prompt",
            {
                "model": "model-id",
                "codex_config": {
                    "model_providers.local.http_headers.X-Key": "secret",
                },
            },
            None,
            None,
            None,
        )


def test_external_codex_provider_requires_explicit_model():
    settings = {
        "codex_config": {
            "model_provider": "custom",
            "model_providers.custom.base_url": "https://models.example/v1",
        },
    }

    with pytest.raises(ValueError, match="External Codex provider.*explicit model"):
        llm._build_codex_cmd(
            "prompt", settings, None, None, None,
        )
    with pytest.raises(ValueError, match="External Codex provider.*explicit model"):
        asyncio.run(llm.llm(
            "prompt",
            config={
                "_security": {"allow_external_provider_host_access": True},
                "solver": {"backend": "codex", **settings},
            },
        ))

    # The native OpenAI path intentionally retains its historical default.
    command = llm._build_codex_cmd("prompt", {}, None, None, None)
    assert command[:4] == ["codex", "exec", "--model", "gpt-5.5"]


def test_loopback_endpoint_routes_only_inside_sandbox():
    endpoint = "http://localhost:11434/v1"
    assert llm._route_oss_base_url(endpoint, sandboxed=False) == endpoint
    assert llm._route_oss_base_url(endpoint, sandboxed=True) == (
        "http://host.docker.internal:11434/v1"
    )
    assert llm._route_oss_base_url(
        "http://0.0.0.0:11434/v1", sandboxed=True,
    ) == "http://host.docker.internal:11434/v1"


def test_oss_mode_fails_closed_on_cloud_alias():
    with pytest.raises(ValueError, match="Claude alias"):
        llm._build_codex_cmd(
            "prompt",
            {"model": "opus", "oss": True, "local_provider": "ollama"},
            None,
            None,
            None,
        )


def _cache_identity(
    *,
    system="system",
    schema=None,
    settings=None,
    codex_version="codex-cli 0.133.0",
):
    return llm._call_cache_identity(
        prompt="prompt",
        system=system,
        schema=schema,
        role="solver",
        backend="codex",
        settings=settings or {
            "model": "gpt-5.5",
            "effort": "xhigh",
            "sandbox": "read-only",
            "search": True,
        },
        watch=False,
        resume=None,
        sandboxed=True,
        image_id="sha256:image",
        codex_version=codex_version,
    )


def test_resume_cache_identity_covers_runtime_and_tool_inputs(tmp_path):
    call_dir = tmp_path / "call_000_solver"
    call_dir.mkdir()
    identity = _cache_identity()
    (call_dir / "prompt.txt").write_text("prompt")
    (call_dir / "response.txt").write_text("response")
    (call_dir / "meta.json").write_text(json.dumps({
        "returncode": 0,
        "session_id": "session",
        "cache_identity": identity,
    }))

    cached = llm._try_resume_from_cache(
        str(call_dir), "prompt", None, identity,
    )
    assert cached[0:2] == ("response", "session")

    variants = [
        _cache_identity(system="changed"),
        _cache_identity(schema={"type": "string"}),
        _cache_identity(settings={
            "model": "gpt-5.4",
            "effort": "xhigh",
            "sandbox": "read-only",
            "search": True,
        }),
        _cache_identity(settings={
            "model": "deployment-id",
            "effort": "xhigh",
            "sandbox": "read-only",
            "search": False,
            "codex_config": {
                "model_provider": "custom",
                "model_providers.custom.base_url": "https://models.example/v1",
                "model_providers.custom.env_key": "MODEL_API_KEY",
            },
            "provider_env": ["MODEL_API_KEY"],
        }),
        _cache_identity(codex_version="codex-cli 0.134.0"),
    ]
    for changed_identity in variants:
        with pytest.raises(RuntimeError, match="resume idempotency check failed"):
            llm._try_resume_from_cache(
                str(call_dir), "prompt", None, changed_identity,
            )


def test_resume_cache_identity_never_contains_provider_secret(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "super-secret-value")
    identity = _cache_identity(settings={
        "model": "deployment-id",
        "effort": None,
        "search": False,
        "codex_config": {
            "model_provider": "custom",
            "model_providers.custom.base_url": "https://models.example/v1",
            "model_providers.custom.env_key": "MODEL_API_KEY",
        },
        "provider_env": ["MODEL_API_KEY"],
    })

    assert "super-secret-value" not in json.dumps(identity)
    assert identity["inputs"]["provider"] == "custom"


def test_resume_cache_identity_tracks_effective_oss_endpoint_override(
    monkeypatch,
):
    settings = {
        "model": "gpt-oss:20b",
        "effort": "xhigh",
        "sandbox": "read-only",
        "search": False,
        "oss": True,
        "local_provider": "ollama",
        # The environment override must win over this configured fallback in
        # both the eventual launch and the pre-launch cache fingerprint.
        "oss_base_url": "http://localhost:11434/v1",
    }

    monkeypatch.setenv("CODEX_OSS_BASE_URL", "http://localhost:12434/v1")
    first = _cache_identity(settings=settings)
    monkeypatch.setenv("CODEX_OSS_BASE_URL", "http://localhost:13434/v1")
    changed = _cache_identity(settings=settings)

    assert first["sha256"] != changed["sha256"]
    assert (
        first["inputs"]["invocation_settings_sha256"]
        != changed["inputs"]["invocation_settings_sha256"]
    )

    monkeypatch.delenv("CODEX_OSS_BASE_URL")
    configured = _cache_identity(settings=settings)
    assert configured["sha256"] not in {first["sha256"], changed["sha256"]}


def test_structured_output_validation_reports_path():
    schema = {
        "type": "object",
        "properties": {"accepted": {"type": "boolean"}},
        "required": ["accepted"],
    }

    llm._validate_structured_output({"accepted": True}, schema)
    with pytest.raises(llm.StructuredOutputError, match=r"\$\.accepted"):
        llm._validate_structured_output({"accepted": "yes"}, schema)


def test_schema_failure_retries_once_then_returns_valid_output(
    monkeypatch, tmp_path,
):
    outputs = ["not json", json.dumps({"accepted": True, "reason": "ok"})]
    launches = []

    class FakeProcess:
        returncode = 0

        def __init__(self, message):
            self.message = message

        async def communicate(self):
            events = [
                {"type": "thread.started", "thread_id": "thread-id"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": self.message},
                },
            ]
            return "\n".join(json.dumps(event) for event in events).encode(), b""

    async def fake_create_subprocess_exec(*command, **_kwargs):
        launches.append(command)
        return FakeProcess(outputs[len(launches) - 1])

    monkeypatch.setattr(
        llm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
    )
    schema = {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["accepted", "reason"],
    }

    calls = []
    artifact_token = llm.artifact_dir.set(str(tmp_path))
    log_token = llm.call_log.set(calls)
    try:
        response, session_id = asyncio.run(llm.llm(
            "judge this",
            role="judge",
            schema=schema,
            config={
                "_security": {
                    "allow_external_provider_host_access": True,
                },
                "judge": {
                    "backend": "codex",
                    "model": "gpt-oss:20b",
                    "oss": True,
                    "local_provider": "ollama",
                    "schema_retries": 1,
                    "search": False,
                },
            },
        ))
    finally:
        llm.call_log.reset(log_token)
        llm.artifact_dir.reset(artifact_token)

    assert response == {"accepted": True, "reason": "ok"}
    assert session_id == "thread-id"
    assert len(launches) == 2
    call_dir = tmp_path / "call_000_judge"
    assert (call_dir / "stdout_attempt_001.jsonl.gz").exists()
    assert (call_dir / "stdout.jsonl.gz").exists()
    assert calls[0]["process_attempts"] == 2
    assert calls[0]["schema_retries"] == 1


def test_streaming_schema_retry_preserves_failed_stdout(monkeypatch, tmp_path):
    outputs = ["not json", json.dumps({"accepted": True, "reason": "ok"})]
    launches = []

    class AsyncLines:
        def __init__(self, lines):
            self.lines = iter(lines)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.lines)
            except StopIteration:
                raise StopAsyncIteration

    class EmptyReader:
        async def read(self):
            return b""

    class FakeProcess:
        returncode = 0

        def __init__(self, message):
            events = [
                {"type": "thread.started", "thread_id": "thread-id"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": message},
                },
            ]
            self.stdout = AsyncLines([
                (json.dumps(event) + "\n").encode() for event in events
            ])
            self.stderr = EmptyReader()

        async def wait(self):
            return self.returncode

    async def fake_create_subprocess_exec(*command, **_kwargs):
        launches.append(command)
        return FakeProcess(outputs[len(launches) - 1])

    monkeypatch.setattr(
        llm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
    )
    schema = {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["accepted", "reason"],
    }
    calls = []
    artifact_token = llm.artifact_dir.set(str(tmp_path))
    log_token = llm.call_log.set(calls)
    try:
        response, _ = asyncio.run(llm.llm(
            "judge this",
            role="judge",
            schema=schema,
            watch=True,
            config={
                "_security": {
                    "allow_external_provider_host_access": True,
                },
                "judge": {
                    "backend": "codex",
                    "model": "gpt-oss:20b",
                    "oss": True,
                    "local_provider": "ollama",
                    "schema_retries": 1,
                    "search": False,
                },
            },
        ))
    finally:
        llm.call_log.reset(log_token)
        llm.artifact_dir.reset(artifact_token)

    assert response["accepted"] is True
    assert len(launches) == 2
    assert (tmp_path / "call_000_judge" / "stdout_attempt_001.jsonl.gz").exists()


def test_external_provider_host_access_fails_closed(monkeypatch):
    launched = False

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr(
        llm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
    )

    with pytest.raises(RuntimeError, match="require Docker isolation"):
        asyncio.run(llm.llm(
            "prompt",
            role="solver",
            config={
                "solver": {
                    "backend": "codex",
                    "model": "local-model",
                    "oss": True,
                    "local_provider": "ollama",
                },
            },
        ))

    assert launched is False


def test_streaming_process_failure_preserves_raw_evidence(monkeypatch, tmp_path):
    event = {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "partial response"},
    }

    class AsyncLines:
        def __aiter__(self):
            self.done = False
            return self

        async def __anext__(self):
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return (json.dumps(event) + "\n").encode()

    class ErrorReader:
        async def read(self):
            return b"provider connection failed"

    class FakeProcess:
        returncode = 2
        stdout = AsyncLines()
        stderr = ErrorReader()

        async def wait(self):
            return self.returncode

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return FakeProcess()

    monkeypatch.setattr(
        llm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
    )
    calls = []
    artifact_token = llm.artifact_dir.set(str(tmp_path))
    log_token = llm.call_log.set(calls)
    try:
        with pytest.raises(llm.ProviderProcessError):
            asyncio.run(llm.llm(
                "prompt",
                role="solver",
                watch=True,
                config={
                    "solver": {
                        "backend": "codex",
                        "model": "gpt-5.5",
                        "search": False,
                    },
                },
            ))
    finally:
        llm.call_log.reset(log_token)
        llm.artifact_dir.reset(artifact_token)

    call_dir = tmp_path / "call_000_solver"
    assert (call_dir / "stdout.jsonl.gz").exists()
    assert (call_dir / "stderr.txt.gz").exists()
    assert calls[0]["failed"] is True
    assert calls[0]["returncode"] == 2
