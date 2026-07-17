import asyncio
import json

import pytest

import agent_loop


class _FakeHTTPResponse:
    def __init__(self, body, headers=None):
        self._body = json.dumps(body).encode()
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def test_chat_completion_sends_declared_sampling_controls(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _FakeHTTPResponse({
            "id": "resp-1",
            "model": "resolved-model",
            "choices": [{
                "message": {"content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }, {"x-request-id": "request-1"})

    monkeypatch.setattr(agent_loop.urllib.request, "urlopen", fake_urlopen)
    content, usage, metadata = agent_loop._chat_completion({
        "endpoint": "https://example.test/v1",
        "model": "deployment",
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.8,
        "seed": 7,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.3,
        "stop": ["END"],
    }, [{"role": "user", "content": "hello"}])

    payload = json.loads(requests[0][0].data)
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8
    assert payload["seed"] == 7
    assert payload["frequency_penalty"] == 0.1
    assert payload["presence_penalty"] == 0.3
    assert payload["stop"] == ["END"]
    assert content == "ok"
    assert usage == {"prompt_tokens": 2, "completion_tokens": 1}
    assert metadata["response_id"] == "resp-1"
    assert metadata["response_headers"] == {"x-request-id": "request-1"}
    assert metadata["usage_reported"] is True


def test_chat_completion_retains_http_attempt_for_malformed_response(
    monkeypatch,
):
    monkeypatch.setattr(
        agent_loop.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse({"choices": []}),
    )

    with pytest.raises(agent_loop.ChatCompletionError) as captured:
        agent_loop._chat_completion({"model": "deployment"}, [])

    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0]["attempt"] == 1
    assert captured.value.attempts[0]["status"] == "success"
    assert captured.value.attempts[0]["duration_ms"] >= 0


def test_chat_completion_retries_read_timeout(monkeypatch):
    calls = 0

    class TimeoutResponse(_FakeHTTPResponse):
        def read(self):
            raise TimeoutError("read timed out")

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TimeoutResponse({})
        return _FakeHTTPResponse({
            "choices": [{
                "message": {"content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    monkeypatch.setattr(agent_loop.urllib.request, "urlopen", fake_urlopen)
    content, _usage, metadata = agent_loop._chat_completion({
        "model": "deployment",
        "max_retries": 1,
        "max_retry_sleep": 0,
    }, [])

    assert content == "ok"
    assert calls == 2
    assert [attempt["status"] for attempt in metadata["http_attempts"]] == [
        "timeout", "success",
    ]
    assert metadata["http_retry_count"] == 1


def test_run_shell_uses_docker_exec_without_login_shell(monkeypatch):
    captured = []

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_create_subprocess_exec(*argv, stdout=None, stderr=None):
        captured.append(argv)
        return FakeProc()

    monkeypatch.setattr(
        agent_loop.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    output, code = asyncio.run(agent_loop._run_shell(
        "python3 -c 'print(42)'",
        container_id="sandbox123",
        allow_host_tools=False,
        timeout=5,
        output_limit=1000,
    ))

    assert code == 0
    assert "ok" in output
    assert captured == [(
        "docker",
        "exec",
        "-w",
        "/workspace",
        "sandbox123",
        "bash",
        "-c",
        "python3 -c 'print(42)'",
    )]


def test_run_agent_executes_shell_tool_with_explicit_host_opt_in(monkeypatch):
    replies = iter([
        {"action": "tool", "tool": "shell", "input": {"cmd": "printf 42"}},
        {"action": "final", "response": "42"},
    ])

    def fake_chat(settings, messages):
        return json.dumps(next(replies)), {
            "prompt_tokens": 3,
            "completion_tokens": 2,
        }

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)

    result = asyncio.run(agent_loop.run_agent(
        "compute",
        settings={
            "model": "fake",
            "allow_shell": True,
            "allow_host_tools": True,
        },
        schema=None,
        system=None,
        resume=None,
        role="solver",
        container_id=None,
        search_config=None,
    ))

    assert result.response == "42"
    assert result.session_id.startswith("theoria-agent-")
    assert result.metadata["input_tokens"] == 6
    assert result.metadata["output_tokens"] == 4
    assert result.metadata["tool_calls"][0]["tool_name"] == "shell"
    assert "42" in result.metadata["tool_calls"][0]["output"]


def test_run_agent_fails_host_shell_closed_by_default(monkeypatch):
    replies = iter([
        {"action": "tool", "tool": "shell", "input": {"cmd": "printf unsafe"}},
        {"action": "final", "response": "done"},
    ])

    def fake_chat(settings, messages):
        return json.dumps(next(replies)), {}

    async def fail_if_subprocess_starts(*argv, stdout=None, stderr=None):
        raise AssertionError(f"unexpected host subprocess: {argv!r}")

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)
    monkeypatch.setattr(
        agent_loop.asyncio,
        "create_subprocess_exec",
        fail_if_subprocess_starts,
    )

    result = asyncio.run(agent_loop.run_agent(
        "compute",
        settings={"model": "fake", "allow_shell": True},
        schema=None,
        system=None,
        resume=None,
        role="solver",
        container_id=None,
        search_config=None,
    ))

    assert result.response == "done"
    assert "unavailable outside Docker" in result.metadata["tool_calls"][0]["output"]


def test_run_agent_exposes_web_search_tool(monkeypatch):
    seen_queries = []
    replies = iter([
        {
            "action": "tool",
            "tool": "web_search",
            "input": {"query": "RFC 9110 title"},
        },
        {"action": "final", "response": "HTTP Semantics"},
    ])

    def fake_chat(settings, messages):
        return json.dumps(next(replies)), {}

    def fake_search(search_config, query, *, max_results):
        seen_queries.append((search_config, query, max_results))
        return "1. RFC 9110\nSnippet: HTTP Semantics"

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)
    monkeypatch.setattr(agent_loop, "_search_web", fake_search)

    result = asyncio.run(agent_loop.run_agent(
        "search",
        settings={"model": "fake", "search": True},
        schema=None,
        system=None,
        resume=None,
        role="citation",
        container_id=None,
        search_config={"provider": "searxng", "endpoint": "http://search"},
    ))

    assert result.response == "HTTP Semantics"
    assert seen_queries == [(
        {"provider": "searxng", "endpoint": "http://search"},
        "RFC 9110 title",
        agent_loop.DEFAULT_SEARCH_RESULTS,
    )]
    assert result.metadata["search_enabled"] is True
    assert result.metadata["tool_calls"][0]["tool_name"] == "web_search"


def test_run_agent_retries_schema_invalid_final(monkeypatch):
    replies = iter([
        {"action": "final", "response": {"accepted": "yes", "reason": "bad"}},
        {"action": "final", "response": {"accepted": True, "reason": "ok"}},
    ])
    seen_messages = []

    def fake_chat(settings, messages):
        seen_messages.append(list(messages))
        return json.dumps(next(replies)), {}

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)

    schema = {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["accepted", "reason"],
    }
    result = asyncio.run(agent_loop.run_agent(
        "judge",
        settings={"model": "fake"},
        schema=schema,
        system=None,
        resume=None,
        role="citation",
        container_id=None,
        search_config=None,
    ))

    assert result.response == {"accepted": True, "reason": "ok"}
    assert len(seen_messages) == 2
    assert "did not match the required JSON schema" in seen_messages[1][-1]["content"]


def test_run_agent_accepts_direct_schema_object(monkeypatch):
    def fake_chat(settings, messages):
        return json.dumps({"accepted": True, "reason": "ok"}), {}

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)

    schema = {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["accepted", "reason"],
    }
    result = asyncio.run(agent_loop.run_agent(
        "judge",
        settings={"model": "fake"},
        schema=schema,
        system=None,
        resume=None,
        role="citation",
        container_id=None,
        search_config=None,
    ))

    assert result.response == {"accepted": True, "reason": "ok"}


def test_run_agent_converts_schema_less_final_to_string(monkeypatch):
    def fake_chat(settings, messages):
        return json.dumps({"action": "final", "response": 4}), {}

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)

    result = asyncio.run(agent_loop.run_agent(
        "solve",
        settings={"model": "fake"},
        schema=None,
        system=None,
        resume=None,
        role="solver",
        container_id=None,
        search_config=None,
    ))

    assert result.response == "4"


def test_run_agent_resume_keeps_transcript(monkeypatch):
    replies = iter([
        {"action": "final", "response": "first"},
        {"action": "final", "response": "second"},
    ])
    seen_messages = []

    def fake_chat(settings, messages):
        seen_messages.append(list(messages))
        return json.dumps(next(replies)), {}

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)

    first = asyncio.run(agent_loop.run_agent(
        "first prompt",
        settings={"model": "fake"},
        schema=None,
        system=None,
        resume=None,
        role="solver",
        container_id=None,
        search_config=None,
    ))
    second = asyncio.run(agent_loop.run_agent(
        "second prompt",
        settings={"model": "fake"},
        schema=None,
        system=None,
        resume=first.session_id,
        role="solver",
        container_id=None,
        search_config=None,
    ))

    assert second.session_id == first.session_id
    assert any(msg["content"] == "first prompt" for msg in seen_messages[1])
    assert seen_messages[1][-1]["content"] == "second prompt"


def test_run_agent_records_provider_response_identity(monkeypatch):
    def fake_chat(settings, messages):
        return json.dumps({"action": "final", "response": "ok"}), {}, {
            "response_id": "resp-1",
            "response_model": "resolved-model-v2",
            "system_fingerprint": "fp-123",
            "http_retry_count": 1,
            "usage_reported": True,
            "http_attempts": [
                {"attempt": 1, "status": "http_error"},
                {"attempt": 2, "status": "success"},
            ],
        }

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)
    result = asyncio.run(agent_loop.run_agent(
        "solve",
        settings={"model": "deployment-alias"},
        schema=None,
        system=None,
        resume=None,
        role="solver",
        container_id=None,
        search_config=None,
    ))

    assert result.metadata["provider_response_ids"] == ["resp-1"]
    assert result.metadata["provider_models"] == ["resolved-model-v2"]
    assert result.metadata["system_fingerprints"] == ["fp-123"]
    assert result.metadata["http_retry_count"] == 1
    assert result.metadata["usage_observed"] is True
    assert result.metadata["usage_complete"] is True
    assert any(event["type"] == "model.response" for event in result.events)
    assert result.messages[0]["role"] == "system"


def test_run_agent_normalizes_provider_cached_token_usage(monkeypatch):
    def fake_chat(settings, messages):
        return json.dumps({"action": "final", "response": "ok"}), {
            "prompt_tokens": 12,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 7},
        }, {"usage_reported": True}

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)
    result = asyncio.run(agent_loop.run_agent(
        "solve",
        settings={"model": "fake"},
        schema=None,
        system=None,
        resume=None,
        role="solver",
        container_id=None,
        search_config=None,
    ))

    assert result.metadata["cache_read_input_tokens"] == 7
    assert result.events[1]["usage"]["cache_read_input_tokens"] == 7


def test_run_agent_failure_keeps_partial_transcript(monkeypatch):
    def fake_chat(settings, messages):
        return (
            "not json",
            {"prompt_tokens": 3, "completion_tokens": 2},
            {"usage_reported": True},
        )

    monkeypatch.setattr(agent_loop, "_chat_completion", fake_chat)

    with pytest.raises(agent_loop.AgentRunError) as captured:
        asyncio.run(agent_loop.run_agent(
            "solve",
            settings={"model": "fake", "max_turns": 1},
            schema=None,
            system=None,
            resume=None,
            role="solver",
            container_id=None,
            search_config=None,
        ))

    error = captured.value
    assert error.metadata["failed"] is True
    assert error.metadata["input_tokens"] == 3
    assert error.metadata["output_tokens"] == 2
    assert error.metadata["usage_observed"] is True
    assert error.metadata["usage_complete"] is False
    assert any(event["type"] == "model.response" for event in error.events)
    assert error.messages[-1]["role"] == "user"
    assert error.raw_stdout
