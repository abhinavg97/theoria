"""Provider-neutral Theoria agent loop.

This backend is intentionally small: it talks to an OpenAI-compatible
``/chat/completions`` endpoint, asks the model to emit JSON actions, runs the
requested tool itself, and feeds the result back to the model. It does not use
provider-native tool calling, so it can exercise Azure/Ollama/vLLM models whose
API surface is plain chat but not Codex-compatible tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from json import JSONDecodeError

from jsonschema import ValidationError
from jsonschema.validators import validator_for


DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MAX_TURNS = 8
DEFAULT_TOOL_TIMEOUT_SECS = 120
DEFAULT_TOOL_OUTPUT_CHARS = 12000
DEFAULT_SEARCH_RESULTS = 5

_SESSIONS: dict[str, list[dict[str, str]]] = {}


@dataclass
class AgentRunResult:
    response: str | dict
    session_id: str
    metadata: dict
    events: list[dict]
    raw_stdout: bytes
    raw_stderr: bytes
    pseudo_cmd: list[str]


@dataclass
class SearchResult:
    text: str
    provider: str
    query: str
    result_count: int
    latency_ms: int
    endpoint: str


class SearchFailure(RuntimeError):
    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


def pseudo_command(settings: dict) -> list[str]:
    endpoint = settings.get("endpoint") or settings.get("base_url") or DEFAULT_ENDPOINT
    return [
        "theoria-agent",
        "--model", str(settings.get("model", "qwen3:4b")),
        "--endpoint", _redact_endpoint(str(endpoint)),
        "--wire-api", "chat-completions",
    ]


def _redact_endpoint(endpoint: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return "<invalid endpoint>"
    if not parsed.scheme or not parsed.netloc:
        return endpoint
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        parsed_port = parsed.port
    except ValueError:
        return "<invalid endpoint>"
    port = f":{parsed_port}" if parsed_port is not None else ""
    return urllib.parse.urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))


def _agent_system_prompt(
    *,
    system: str | None,
    schema: dict | None,
    search_enabled: bool,
    shell_enabled: bool,
) -> str:
    tools = []
    if shell_enabled:
        tools.append(
            '- shell: {"action":"tool","tool":"shell","input":{"cmd":"..."}}'
        )
    if search_enabled:
        tools.append(
            '- web_search: {"action":"tool","tool":"web_search","input":{"query":"..."}}'
        )
    tool_text = "\n".join(tools) if tools else "(no tools are available)"
    schema_text = json.dumps(schema, ensure_ascii=False) if schema else "string"
    base = f"""You are running inside Theoria's provider-neutral agent loop.

You must respond with exactly one JSON object and no markdown fence.

Allowed responses:
- Final answer: {{"action":"final","response": ...}}
{tool_text}

Rules:
- Do not use provider-native tool calls. Express tool requests only as JSON text.
- If the user or role prompt asks you to use an available tool, call that
  tool before giving a final answer.
- Use shell for calculations, code execution, and inspecting files when useful.
- Use web_search for current or external factual claims when available.
- After receiving a tool result, continue until you can provide the final answer.
- The final response must be valid for this expected response schema:
{schema_text}
"""
    if system:
        return f"{system}\n\n{base}"
    return base


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    starts = [0] if stripped.startswith("{") else []
    starts.extend(i for i, ch in enumerate(stripped) if ch == "{")
    seen = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"model did not return a JSON object: {text[:500]!r}")


def _validate_schema(value, schema: dict | None) -> None:
    if schema is None:
        return
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)
    validator.validate(value)


def _chat_completion(settings: dict, messages: list[dict[str, str]]) -> tuple[str, dict]:
    endpoint = str(settings.get("endpoint") or settings.get("base_url") or DEFAULT_ENDPOINT)
    url = endpoint.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": settings.get("model", "qwen3:4b"),
        "messages": messages,
    }
    if settings.get("max_tokens") is not None:
        payload["max_tokens"] = settings["max_tokens"]
    if settings.get("temperature") is not None:
        payload["temperature"] = settings["temperature"]

    headers = {"Content-Type": "application/json"}
    api_key_env = settings.get("api_key_env")
    api_key = os.environ.get(api_key_env) if api_key_env else settings.get("api_key")
    if api_key:
        auth_header = settings.get("auth_header", "Authorization")
        if str(auth_header).lower() == "api-key":
            headers["api-key"] = str(api_key)
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    timeout = float(settings.get("request_timeout", 180))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    max_retries = int(settings.get("max_retries", 3))
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = max(1.0, float(retry_after or 0))
                except ValueError:
                    delay = 5.0
                time.sleep(min(delay, float(settings.get("max_retry_sleep", 65))))
                continue
            raise RuntimeError(
                f"chat completion failed with HTTP {exc.code}: {raw[:1500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"chat completion failed: {exc}") from exc
    else:
        raise RuntimeError("chat completion retry loop exited unexpectedly")

    try:
        choice = body["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat completion response: {body!r}") from exc

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None:
        content = ""

    usage = body.get("usage") or {}
    return str(content), usage


async def _run_shell(
    cmd: str,
    *,
    container_id: str | None,
    allow_host_tools: bool,
    timeout: float,
    output_limit: int,
) -> tuple[str, int]:
    if container_id:
        argv = [
            "docker", "exec", "-w", "/workspace", container_id,
            # Do not use a login shell here: it resets the image PATH and
            # hides the sandbox Python/Sage environment from tool commands.
            "bash", "-c", cmd,
        ]
    elif allow_host_tools:
        argv = ["bash", "-lc", cmd]
    else:
        return (
            "shell tool is unavailable outside Docker; rerun with --docker "
            "or set allow_host_tools: true for this backend",
            126,
        )

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"command timed out after {timeout:g}s", 124

    stdout = out.decode(errors="replace")
    stderr = err.decode(errors="replace")
    combined = (
        f"exit_code: {proc.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )
    if len(combined) > output_limit:
        combined = (
            combined[:output_limit]
            + f"\n...[TRUNCATED: {len(combined) - output_limit} chars]"
        )
    return combined, int(proc.returncode or 0)


def _search_web(search_config: dict, query: str, *, max_results: int) -> SearchResult:
    started = time.perf_counter()
    provider = search_config.get("provider", "searxng")
    if provider != "searxng":
        raise SearchFailure(
            "invalid_config",
            "only _web_search.provider: searxng is supported",
        )
    endpoint = search_config.get("endpoint")
    if not endpoint:
        raise SearchFailure(
            "invalid_config",
            "_web_search.endpoint is required for searxng",
        )
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = endpoint.rstrip("/") + "/search?" + params
    request = urllib.request.Request(url, headers={"User-Agent": "theoria-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise SearchFailure(
            f"http_{exc.code}",
            f"web search failed with HTTP {exc.code}: {raw[:500]}",
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        category = "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
        message = "web search timed out" if category == "timeout" else f"web search failed: {exc}"
        raise SearchFailure(category, message) from exc
    except TimeoutError as exc:
        raise SearchFailure("timeout", "web search timed out") from exc

    results = body.get("results") or []
    lines = []
    limited = results[:max_results]
    for idx, result in enumerate(limited, 1):
        title = result.get("title") or "(untitled)"
        url = result.get("url") or ""
        content = result.get("content") or result.get("snippet") or ""
        lines.append(f"{idx}. {title}\nURL: {url}\nSnippet: {content}")
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    return SearchResult(
        text="\n\n".join(lines) if lines else "(no search results)",
        provider=provider,
        query=query,
        result_count=len(limited),
        latency_ms=latency_ms,
        endpoint=_redact_endpoint(str(endpoint)),
    )


def _tool_call_previews(events: list[dict], *, limit: int = 2000) -> list[dict]:
    calls: dict[str, dict] = {}
    order: list[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        call_id = item.get("call_id")
        if not call_id:
            continue
        if item.get("type") == "function_call":
            calls[call_id] = {
                "tool_name": item.get("name"),
                "input": _truncate(str(item.get("arguments") or ""), limit),
            }
            order.append(call_id)
        elif item.get("type") == "function_call_output" and call_id in calls:
            calls[call_id]["output"] = _truncate(str(item.get("output") or ""), limit)
            if isinstance(item.get("metadata"), dict):
                calls[call_id]["metadata"] = item["metadata"]
    return [calls[call_id] for call_id in order]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[TRUNCATED: {len(text) - limit} chars]"


def _search_failure_category(exc: Exception) -> str:
    category = getattr(exc, "category", None)
    if category:
        return str(category)
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
    if isinstance(exc, (JSONDecodeError, UnicodeDecodeError)):
        return "invalid_response"
    return type(exc).__name__


def _is_client_search_failure(category: str) -> bool:
    return category in {
        "invalid_config",
        "invalid_input",
        "missing_query",
        "unavailable",
    }


def _increment_error_category(categories: dict[str, int], category: str) -> None:
    categories[category] = categories.get(category, 0) + 1


async def run_agent(
    prompt: str,
    *,
    settings: dict,
    schema: dict | None,
    system: str | None,
    resume: str | None,
    role: str,
    container_id: str | None,
    search_config: dict | None,
    watch: bool = False,
) -> AgentRunResult:
    session_id = resume if resume in _SESSIONS else f"theoria-agent-{uuid.uuid4().hex}"
    shell_enabled = bool(settings.get("allow_shell", True))
    search_enabled = bool(search_config) and bool(settings.get("search", True))
    allow_host_tools = bool(settings.get("allow_host_tools", False))
    max_turns = int(settings.get("max_turns", DEFAULT_MAX_TURNS))
    tool_timeout = float(settings.get("tool_timeout", DEFAULT_TOOL_TIMEOUT_SECS))
    output_limit = int(settings.get("tool_output_chars", DEFAULT_TOOL_OUTPUT_CHARS))
    max_search_results = int(settings.get("search_results", DEFAULT_SEARCH_RESULTS))

    messages = [dict(m) for m in _SESSIONS.get(session_id, [])]
    if not messages:
        messages.append({
            "role": "system",
            "content": _agent_system_prompt(
                system=system,
                schema=schema,
                search_enabled=search_enabled,
                shell_enabled=shell_enabled,
            ),
        })
    messages.append({"role": "user", "content": prompt})

    events: list[dict] = [{"type": "thread.started", "thread_id": session_id}]
    stderr_lines: list[str] = []
    input_tokens = output_tokens = 0
    final_response: str | dict | None = None
    search_provider = (
        (search_config or {}).get("provider", "searxng")
        if search_config else None
    )
    search_error_categories: dict[str, int] = {}
    search_requests = 0
    search_successes = 0
    search_failures = 0
    search_client_failures = 0
    search_provider_failures = 0
    search_provider_requests = 0
    search_result_count = 0
    search_latency_ms = 0

    for turn in range(max_turns):
        started = time.monotonic()
        content, usage = await asyncio.to_thread(_chat_completion, settings, messages)
        input_tokens += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        events.append({
            "type": "turn.completed",
            "usage": {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0,
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0,
                "duration_ms": int(round((time.monotonic() - started) * 1000)),
            },
        })
        messages.append({"role": "assistant", "content": content})

        try:
            action = _extract_json_object(content)
        except ValueError as exc:
            stderr_lines.append(str(exc))
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not a JSON action object. "
                    "Return exactly one JSON object using the documented format."
                ),
            })
            continue

        if schema is not None:
            try:
                _validate_schema(action, schema)
            except ValidationError:
                pass
            else:
                final_response = action
                events.append({
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": json.dumps(action)},
                })
                break

        if action.get("action") == "final":
            candidate = action.get("response", "")
            if schema is None and not isinstance(candidate, str):
                candidate = json.dumps(candidate, ensure_ascii=False)
            try:
                _validate_schema(candidate, schema)
            except ValidationError as exc:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your final response did not match the required JSON "
                        f"schema: {exc.message}. Return a corrected final "
                        "JSON action only."
                    ),
                })
                continue
            final_response = candidate
            events.append({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": json.dumps(candidate) if isinstance(candidate, (dict, list)) else str(candidate)},
            })
            break

        if action.get("action") != "tool":
            messages.append({
                "role": "user",
                "content": "Unknown action. Use action=\"tool\" or action=\"final\".",
            })
            continue

        tool = action.get("tool")
        raw_tool_input = action.get("input")
        invalid_tool_input = raw_tool_input is not None and not isinstance(
            raw_tool_input, dict
        )
        tool_input = raw_tool_input if isinstance(raw_tool_input, dict) else {}
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        events.append({
            "type": "item.completed",
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": tool,
                "arguments": json.dumps(
                    raw_tool_input if raw_tool_input is not None else {},
                    ensure_ascii=False,
                ),
            },
        })
        if watch:
            print(
                f"      [tool] {tool}({json.dumps(tool_input)[:100]})",
                file=sys.stderr,
            )

        tool_metadata: dict = {}
        try:
            if tool == "shell" and shell_enabled:
                cmd = str(tool_input.get("cmd", ""))
                if not cmd.strip():
                    result = "missing shell input field: cmd"
                    tool_metadata = {"exit_code": 2}
                else:
                    result, exit_code = await _run_shell(
                        cmd,
                        container_id=container_id,
                        allow_host_tools=allow_host_tools,
                        timeout=tool_timeout,
                        output_limit=output_limit,
                    )
                    tool_metadata = {"exit_code": exit_code}
            elif tool == "web_search":
                search_requests += 1
                search_started = time.perf_counter()
                query = str(tool_input.get("query", ""))
                tool_metadata = {
                    "provider": search_provider,
                    "query": query,
                    "ok": False,
                    "provider_request": False,
                    "result_count": 0,
                    "latency_ms": 0,
                }
                if invalid_tool_input:
                    category = "invalid_input"
                    search_failures += 1
                    search_client_failures += 1
                    _increment_error_category(search_error_categories, category)
                    tool_metadata["error_category"] = category
                    result = "web_search input must be a JSON object"
                elif not search_enabled:
                    category = "unavailable"
                    search_failures += 1
                    search_client_failures += 1
                    _increment_error_category(search_error_categories, category)
                    tool_metadata["error_category"] = category
                    result = f"tool {tool!r} is not available"
                elif not query.strip():
                    category = "missing_query"
                    search_failures += 1
                    search_client_failures += 1
                    _increment_error_category(search_error_categories, category)
                    tool_metadata["error_category"] = category
                    result = "missing web_search input field: query"
                else:
                    search_result = await asyncio.to_thread(
                        _search_web, search_config or {}, query,
                        max_results=max_search_results,
                    )
                    if isinstance(search_result, SearchResult):
                        result = search_result.text
                        result_count = search_result.result_count
                        latency = search_result.latency_ms
                        provider = search_result.provider
                        endpoint = search_result.endpoint
                    else:
                        # Tests and third-party monkeypatches may return a
                        # plain string; keep that compatibility while still
                        # marking the search as successful.
                        result = str(search_result)
                        result_count = 0
                        latency = int(round((time.perf_counter() - search_started) * 1000))
                        provider = search_provider
                        endpoint = None
                    search_successes += 1
                    search_provider_requests += 1
                    search_result_count += result_count
                    search_latency_ms += latency
                    tool_metadata.update({
                        "provider": provider,
                        "endpoint": endpoint,
                        "ok": True,
                        "provider_request": True,
                        "result_count": result_count,
                        "latency_ms": latency,
                    })
            else:
                result = f"tool {tool!r} is not available"
        except Exception as exc:
            if tool == "web_search":
                latency = int(round((time.perf_counter() - search_started) * 1000))
                category = _search_failure_category(exc)
                search_failures += 1
                client_failure = _is_client_search_failure(category)
                if client_failure:
                    search_client_failures += 1
                else:
                    search_provider_failures += 1
                    search_provider_requests += 1
                    search_latency_ms += latency
                _increment_error_category(search_error_categories, category)
                tool_metadata.update({
                    "ok": False,
                    "provider_request": not client_failure,
                    "latency_ms": 0 if client_failure else latency,
                    "error_category": category,
                    "status_code": getattr(exc, "status_code", None),
                })
            result = f"tool {tool!r} failed: {type(exc).__name__}: {exc}"

        events.append({
            "type": "item.completed",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
                "metadata": tool_metadata,
            },
        })
        messages.append({
            "role": "user",
            "content": f"Tool result for {tool}:\n{result}",
        })

    if final_response is None:
        raise RuntimeError(f"theoria_agent exceeded max_turns={max_turns} without final response")

    _SESSIONS[session_id] = messages
    raw_stdout = ("\n".join(json.dumps(event) for event in events) + "\n").encode("utf-8")
    raw_stderr = ("\n".join(stderr_lines) + "\n").encode("utf-8") if stderr_lines else b""
    metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": 0,
        "total_cost_usd": None,
        "tool_calls": _tool_call_previews(events),
        "num_turns": len([e for e in events if e.get("type") == "turn.completed"]),
        "provider_endpoint": _redact_endpoint(str(settings.get("endpoint") or settings.get("base_url") or DEFAULT_ENDPOINT)),
        "wire_api": "chat-completions",
        "search_enabled": search_enabled,
        "web_search_provider": search_provider,
        # Backward-compatible: every model-issued web_search action, including
        # malformed, empty, and unavailable-tool attempts.
        "web_search_requests": search_requests,
        "web_search_attempts": search_requests,
        # Requests that passed client-side validation and reached the search
        # provider path. This is the denominator for provider reliability.
        "web_search_provider_requests": search_provider_requests,
        "web_search_successes": search_successes,
        "web_search_failures": search_failures,
        "web_search_client_failures": search_client_failures,
        "web_search_provider_failures": search_provider_failures,
        "web_search_result_count": search_result_count,
        # Cumulative latency for requests that reached the provider. Keep the
        # original field for compatibility and expose the semantics directly.
        "web_search_latency_ms": search_latency_ms,
        "web_search_total_latency_ms": search_latency_ms,
        "web_search_mean_latency_ms": (
            search_latency_ms / search_provider_requests
            if search_provider_requests else None
        ),
        "web_search_error_categories": search_error_categories,
        "shell_enabled": shell_enabled,
        "role": role,
    }
    return AgentRunResult(
        response=final_response,
        session_id=session_id,
        metadata=metadata,
        events=events,
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        pseudo_cmd=pseudo_command(settings),
    )
