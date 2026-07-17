import asyncio
import gzip
import json

import pytest

import agent_loop
import llm as llm_module


def test_llm_theoria_agent_writes_standard_artifacts(tmp_path, monkeypatch):
    events = [
        {"type": "thread.started", "thread_id": "sess-1"},
        {
            "type": "item.completed",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": "{\"cmd\":\"printf 4\"}",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "exit_code: 0\nstdout:\n4",
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "4"},
        },
    ]

    async def fake_run_agent(*args, **kwargs):
        assert kwargs["settings"]["model"] == "fake-model"
        assert kwargs["container_id"] == "sandbox123"
        assert kwargs["search_config"] == {"provider": "searxng", "endpoint": "http://search"}
        return agent_loop.AgentRunResult(
            response="4",
            session_id="sess-1",
            metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "total_cost_usd": None,
                "usage_observed": True,
                "usage_complete": True,
                "tool_calls": [{
                    "tool_name": "shell",
                    "input": "{\"cmd\":\"printf 4\"}",
                    "output": "exit_code: 0\nstdout:\n4",
                }],
                "wire_api": "chat-completions",
            },
            events=events,
            raw_stdout=("\n".join(json.dumps(e) for e in events) + "\n").encode(),
            raw_stderr=b"",
            pseudo_cmd=[
                "theoria-agent",
                "--model", "fake-model",
                "--endpoint", "http://endpoint/v1",
                "--wire-api", "chat-completions",
            ],
        )

    monkeypatch.setattr(agent_loop, "run_agent", fake_run_agent)

    log = []
    log_token = llm_module.call_log.set(log)
    artifact_token = llm_module.artifact_dir.set(str(tmp_path))
    sandbox_token = llm_module.sandbox_container.set("sandbox123")
    try:
        response, session_id = asyncio.run(llm_module.llm(
            "What is 2+2?",
            role="solver",
            config={
                "_web_search": {"provider": "searxng", "endpoint": "http://search"},
                "solver": {
                    "backend": "theoria_agent",
                    "model": "fake-model",
                    "endpoint": "http://endpoint/v1",
                },
            },
        ))
    finally:
        llm_module.call_log.reset(log_token)
        llm_module.artifact_dir.reset(artifact_token)
        llm_module.sandbox_container.reset(sandbox_token)

    assert response == "4"
    assert session_id == "sess-1"
    assert log[0]["backend"] == "theoria_agent"
    assert log[0]["model"] == "fake-model"
    assert log[0]["returncode"] == 0
    assert log[0]["usage_complete"] is True
    assert log[0]["wire_api"] == "chat-completions"
    assert log[0]["tool_calls"][0]["tool_name"] == "shell"

    call_dir = tmp_path / "call_000_solver"
    assert json.loads((call_dir / "cmd.json").read_text())[0] == "theoria-agent"
    assert (call_dir / "prompt.txt").read_text() == "What is 2+2?"
    assert (call_dir / "response.txt").read_text() == "4"
    assert json.loads((call_dir / "tool_calls.json").read_text())[0]["tool_name"] == "shell"
    with gzip.open(call_dir / "events.json.gz", "rt") as f:
        assert json.loads(f.read())[0]["type"] == "thread.started"


def test_llm_theoria_agent_records_failed_call_and_partial_artifacts(
    tmp_path, monkeypatch,
):
    events = [{"type": "model.response", "content": "not json"}]
    messages = [
        {"role": "system", "content": "effective system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "not json"},
    ]

    async def fake_run_agent(*args, **kwargs):
        raise agent_loop.AgentRunError(
            "max turns",
            metadata={
                "input_tokens": 3,
                "output_tokens": 2,
                "tool_calls": [],
                "http_retry_count": 1,
                "usage_observed": True,
                "usage_complete": False,
                "failed": True,
            },
            events=events,
            raw_stdout=b'{"type":"model.response"}\n',
            raw_stderr=b"max turns\n",
            pseudo_cmd=[
                "theoria-agent", "--model", "fake-model",
                "--endpoint", "http://endpoint/v1",
            ],
            messages=messages,
        )

    monkeypatch.setattr(agent_loop, "run_agent", fake_run_agent)
    log = []
    log_token = llm_module.call_log.set(log)
    artifact_token = llm_module.artifact_dir.set(str(tmp_path))
    try:
        with pytest.raises(agent_loop.AgentRunError):
            asyncio.run(llm_module.llm(
                "question",
                role="solver",
                config={
                    "solver": {
                        "backend": "theoria_agent",
                        "model": "fake-model",
                        "endpoint": "http://endpoint/v1",
                    },
                },
            ))
    finally:
        llm_module.call_log.reset(log_token)
        llm_module.artifact_dir.reset(artifact_token)

    assert len(log) == 1
    assert log[0]["failed"] is True
    assert log[0]["input_tokens"] == 3
    assert log[0]["usage_observed"] is True
    assert log[0]["usage_complete"] is False
    assert log[0]["retry_count"] == 1
    call_dir = tmp_path / "call_000_solver"
    assert json.loads((call_dir / "meta.json").read_text())["failed"] is True
    assert (call_dir / "messages.json").exists()
    assert (call_dir / "effective_system.txt").read_text() == "effective system"
    assert (call_dir / "attempt_001_stdout.jsonl.gz").exists()


def test_failed_call_retry_preserves_original_evidence(tmp_path, monkeypatch):
    async def failed_agent(*_args, **_kwargs):
        raise agent_loop.AgentRunError(
            "first failure",
            metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "tool_calls": [],
                "usage_observed": True,
                "usage_complete": False,
                "failed": True,
            },
            events=[{"type": "model.response", "content": "bad"}],
            raw_stdout=b"bad\n",
            raw_stderr=b"first failure\n",
            pseudo_cmd=["theoria-agent", "--model", "fake"],
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
        )

    monkeypatch.setattr(agent_loop, "run_agent", failed_agent)
    first_log = []
    log_token = llm_module.call_log.set(first_log)
    artifact_token = llm_module.artifact_dir.set(str(tmp_path))
    try:
        with pytest.raises(agent_loop.AgentRunError):
            asyncio.run(llm_module.llm(
                "question",
                role="solver",
                config={"solver": {
                    "backend": "theoria_agent",
                    "model": "fake",
                    "endpoint": "http://endpoint/v1",
                }},
            ))
    finally:
        llm_module.call_log.reset(log_token)
        llm_module.artifact_dir.reset(artifact_token)

    base = tmp_path / "call_000_solver"
    original_meta = (base / "meta.json").read_bytes()
    original_stdout = (base / "attempt_001_stdout.jsonl.gz").read_bytes()

    async def successful_agent(*_args, **_kwargs):
        return agent_loop.AgentRunResult(
            response="ok",
            session_id="session-2",
            metadata={
                "input_tokens": 2,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "tool_calls": [],
                "usage_observed": True,
                "usage_complete": True,
            },
            events=[{"type": "thread.started", "thread_id": "session-2"}],
            raw_stdout=b'{"type":"thread.started"}\n',
            raw_stderr=b"",
            pseudo_cmd=["theoria-agent", "--model", "fake"],
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "ok"},
            ],
        )

    monkeypatch.setattr(agent_loop, "run_agent", successful_agent)
    second_log = []
    log_token = llm_module.call_log.set(second_log)
    artifact_token = llm_module.artifact_dir.set(str(tmp_path))
    try:
        response, _ = asyncio.run(llm_module.llm(
            "question",
            role="solver",
            config={"solver": {
                "backend": "theoria_agent",
                "model": "fake",
                "endpoint": "http://endpoint/v1",
            }},
        ))
    finally:
        llm_module.call_log.reset(log_token)
        llm_module.artifact_dir.reset(artifact_token)

    assert response == "ok"
    assert (base / "meta.json").read_bytes() == original_meta
    assert (base / "attempt_001_stdout.jsonl.gz").read_bytes() == original_stdout
    retry = base / "retry_001"
    assert retry.is_dir()
    retry_meta = json.loads((retry / "meta.json").read_text())
    assert retry_meta["invocation_count"] == 2
    assert retry_meta["failed_invocations"] == 1
    assert retry_meta["resume_retry_count"] == 1
    assert second_log[0]["retry_count"] == 1


def test_cached_theoria_agent_call_restores_transcript(tmp_path, monkeypatch):
    call_dir = tmp_path / "call_000_solver"
    call_dir.mkdir()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    (call_dir / "prompt.txt").write_text("question")
    (call_dir / "response.txt").write_text("answer")
    (call_dir / "messages.json").write_text(json.dumps(messages))
    (call_dir / "meta.json").write_text(json.dumps({
        "backend": "theoria_agent",
        "returncode": 0,
        "failed": False,
        "session_id": "persisted-session",
    }))
    agent_loop._SESSIONS.clear()

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(agent_loop, "run_agent", should_not_run)
    log = []
    log_token = llm_module.call_log.set(log)
    artifact_token = llm_module.artifact_dir.set(str(tmp_path))
    try:
        response, session_id = asyncio.run(llm_module.llm(
            "question",
            role="solver",
            config={"solver": {
                "backend": "theoria_agent",
                "model": "fake",
            }},
        ))
    finally:
        llm_module.call_log.reset(log_token)
        llm_module.artifact_dir.reset(artifact_token)

    assert response == "answer"
    assert session_id == "persisted-session"
    assert agent_loop._SESSIONS[session_id] == messages
    assert log[0]["resumed_from_cache"] is True


def test_failed_cached_call_rejects_prompt_drift(tmp_path, monkeypatch):
    call_dir = tmp_path / "call_000_solver"
    call_dir.mkdir()
    (call_dir / "prompt.txt").write_text("original")
    (call_dir / "meta.json").write_text(json.dumps({
        "backend": "theoria_agent", "failed": True,
    }))

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("drifted call executed")

    monkeypatch.setattr(agent_loop, "run_agent", should_not_run)
    log_token = llm_module.call_log.set([])
    artifact_token = llm_module.artifact_dir.set(str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="idempotency check failed"):
            asyncio.run(llm_module.llm(
                "changed",
                role="solver",
                config={"solver": {"backend": "theoria_agent", "model": "fake"}},
            ))
    finally:
        llm_module.call_log.reset(log_token)
        llm_module.artifact_dir.reset(artifact_token)


class _AsyncByteLines:
    def __init__(self, lines):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


class _AsyncBytesReader:
    def __init__(self, value=b""):
        self.value = value

    async def read(self):
        return self.value


class _StreamingProcess:
    def __init__(self, lines, stderr=b""):
        self.stdout = _AsyncByteLines(lines)
        self.stderr = _AsyncBytesReader(stderr)
        self.returncode = 0

    async def wait(self):
        return self.returncode


def test_claude_stream_parse_failure_retains_raw_trace():
    event = {"type": "result", "session_id": "s", "result": "not structured"}
    raw = (json.dumps(event) + "\n").encode()
    process = _StreamingProcess([raw], stderr=b"provider diagnostic")

    with pytest.raises(llm_module.ProviderProcessError) as captured:
        asyncio.run(llm_module._run_claude_streaming(
            process, {"type": "object"},
        ))

    assert captured.value.raw_stdout == raw
    assert captured.value.raw_stderr == b"provider diagnostic"
    assert captured.value.events == [event]


def test_codex_stream_parse_failure_retains_raw_trace():
    event = {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "not-json"},
    }
    raw = (json.dumps(event) + "\n").encode()
    process = _StreamingProcess([raw], stderr=b"provider diagnostic")

    with pytest.raises(llm_module.ProviderProcessError) as captured:
        asyncio.run(llm_module._run_codex_streaming(
            process, {"type": "object"},
        ))

    assert captured.value.raw_stdout == raw
    assert captured.value.raw_stderr == b"provider diagnostic"
    assert captured.value.events == [event]
