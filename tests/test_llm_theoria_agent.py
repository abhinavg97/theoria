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
                "cached_input_tokens": 0,
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
