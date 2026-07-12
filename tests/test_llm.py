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


def test_oss_codex_allows_explicit_multi_agent_override():
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
