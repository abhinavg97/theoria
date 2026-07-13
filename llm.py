"""LLM call layer. Supports claude -p and codex exec backends.

Pipeline.py imports llm() from here. The backend is selected per-role
via the config (backend: claude or backend: codex). Default is claude.

When watch=True, streams events to stderr so you can see tool calls,
thinking, and progress in real time. The return value is the same either way.

llm() returns (response, session_id). Pass resume=session_id on a later
call to continue the same conversation.
"""

from __future__ import annotations

import asyncio
import contextvars
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for


SUBPROCESS_STREAM_LIMIT = 64 * 1024 * 1024  # 64 MiB

TOOL_CALL_INPUT_LIMIT = 500    # chars per tool input (usually a query)
TOOL_CALL_OUTPUT_LIMIT = 2000  # chars per tool output (can be a web search result)


# ── Per-call logging via contextvar ─────────────────────────────
#
# When set to a list (by the harness, per problem), every successful
# llm() call appends a metadata dict to it. When None (the default),
# llm() behaves identically to before — no side effects. The list is
# shared across child asyncio tasks via contextvars.copy_context(),
# so parallel judge calls all append to the same list without locks.

call_log: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "llm_call_log", default=None,
)


# ── Artifact directory via contextvar ───────────────────────────
#
# When set (by the harness, per problem), every llm() call writes the
# exact inputs and raw outputs to <artifact_dir>/call_NNN_<role>/.
# That directory is the source of truth; the existing call_log entries
# keep their truncated previews for scannability.

artifact_dir: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_artifact_dir", default=None,
)


# ── Sandbox container via contextvar ────────────────────────────
#
# When set (by the harness, per problem), every llm() call runs the
# claude/codex binary via `docker exec <container_id>` instead of on
# the host. The container is created/destroyed by the harness; llm()
# just reads this value and wraps the cmd accordingly.

sandbox_container: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_sandbox_container", default=None,
)

# Image digest for whichever image the sandbox container was built
# from. Optional — informational only, stored in call_meta so post-hoc
# analysis can tell which build produced a given call.

sandbox_image_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_sandbox_image_id", default=None,
)

# Effective Codex CLI version for this run.  The harness resolves it once from
# the selected container image (or host CLI) and makes it available to every
# call so cache identity does not require launching another subprocess.

codex_cli_version: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_codex_cli_version", default=None,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# C0 control chars + DEL, except tab/newline/CR which prompts use legitimately.
# LLMs occasionally emit \x00 (and other control bytes) in their output;
# passing those to subprocess argv raises "embedded null byte" (CPython
# #111656). Sanitize at the LLM-output boundary so the same text is safe
# whether downstream uses argv, file content, or JSON.
_BAD_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _sanitize_llm_output(obj):
    """Strip C0 controls (except \\t \\n \\r) and DEL from LLM-generated
    text. Recurses into dicts/lists so structured outputs are cleaned too.
    Non-string scalars pass through unchanged."""
    if isinstance(obj, str):
        return _BAD_CTRL.sub("", obj)
    if isinstance(obj, dict):
        return {k: _sanitize_llm_output(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_llm_output(v) for v in obj]
    return obj


# ── Hang detection ──────────────────────────────────────────────
#
# A subprocess is hung when its per-process CPU ticks (utime + stime
# from /proc/<pid>/stat inside the container) stop accumulating for
# WATCHDOG_SILENCE_SECS. A working LLM call always uses CPU — even
# during server-side reasoning, the client receives streaming bytes,
# parses keepalive pings, schedules tokio timers — all of which
# generate ticks. Literal 0 ticks for many minutes means the HTTP
# connection is silently dead and the client is stuck waiting on it
# forever.
#
# Per-process beats container-wide CPU: when many judge subprocesses
# share one container, container CPU stays nonzero while siblings
# work, masking individual hangs. Per-PID sampling is judge-specific.
#
# Plus a hard wall-time backstop for unanticipated failure modes
# (e.g. busy-spin hangs that consume CPU without producing output).

WATCHDOG_SILENCE_SECS = 90             # 90s of zero CPU ticks → hung
WATCHDOG_HARD_WALL_SECS = 4 * 60 * 60  # 4h absolute ceiling
WATCHDOG_POLL_SECS = 30                # how often the watchdog wakes
WATCHDOG_RETRY_MAX = 3                 # retry attempts after kill
SCHEMA_RETRY_MAX = 3                   # hard cap for opt-in format retries
PROVIDER_TERMINATE_TIMEOUT_SECS = 5
_CONTAINER_CALL_MARKER_ENV = "THEORIA_LLM_CALL_MARKER"


# ── OSS call serialization ──────────────────────────────────────
#
# Local providers (Ollama by default) process one request at a time.
# Theoria's judge fan-out launches every step judge concurrently, so
# on a local provider all but one Codex process sits on an open SSE
# stream receiving zero bytes until a slot frees — which can be hours
# on consumer hardware. Codex's stream-idle timeout then kills and
# re-queues each starved request up to five times before failing the
# call outright, and every abandoned retry still burns a full prefill
# on the server. Observed live: a 13-step proof → 14 parallel judges →
# "stream disconnected before completion: idle timeout waiting for
# SSE" after ~2h, zero judges completed.
#
# Serializing OSS calls per event loop fixes this structurally: a call
# only opens its stream when the provider is actually free, so the
# only idle window left is the model's own prefill. Servers that
# genuinely handle concurrent requests can raise the limit via
# THEORIA_OSS_MAX_PARALLEL.

_OSS_GATE_LOOP_ATTR = "_theoria_oss_provider_gate"


def _oss_max_parallel() -> int:
    raw = os.environ.get("THEORIA_OSS_MAX_PARALLEL", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        print(
            f"[llm] ignoring non-integer THEORIA_OSS_MAX_PARALLEL={raw!r}",
            file=sys.stderr,
        )
        return 1


def _oss_gate() -> asyncio.Semaphore:
    """Per-event-loop semaphore bounding concurrent OSS provider calls."""
    loop = asyncio.get_running_loop()
    gate = getattr(loop, _OSS_GATE_LOOP_ATTR, None)
    if gate is None:
        gate = asyncio.Semaphore(_oss_max_parallel())
        # The loop owns the gate so its lifetime cannot outlive that loop.
        # A process-global id(loop) registry is unsafe because Python may
        # recycle an object's id after asyncio.run() closes and releases it,
        # causing a fresh loop to inherit a semaphore created for a closed
        # loop (and an obsolete THEORIA_OSS_MAX_PARALLEL value).
        setattr(loop, _OSS_GATE_LOOP_ATTR, gate)
    return gate


class WatchdogKilled(RuntimeError):
    """Raised when the watchdog killed the subprocess for being hung.
    Distinguished from generic RuntimeError so the retry loop in llm()
    can treat it as a transient failure (codex CLI hang on a specific
    HTTP connection) rather than a real error."""
    pass


class StructuredOutputError(RuntimeError):
    """A provider response was not valid for the requested JSON schema."""

    def __init__(
        self,
        message: str,
        *,
        raw_stdout: bytes = b"",
        raw_stderr: bytes = b"",
        events: list[dict] | None = None,
        provider_meta: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_stdout = raw_stdout
        self.raw_stderr = raw_stderr
        self.events = events or []
        self.provider_meta = provider_meta or {}


class ProviderProcessError(RuntimeError):
    """A streaming provider process failed after emitting useful evidence."""

    def __init__(
        self,
        message: str,
        *,
        raw_stdout: bytes = b"",
        raw_stderr: bytes = b"",
        events: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_stdout = raw_stdout
        self.raw_stderr = raw_stderr
        self.events = events or []


async def _find_subprocess_pid(
    container_id: str, comm: str, marker: str | None,
) -> int | None:
    """Find the PID of the actual LLM binary inside `container_id`.

    Always filters by `comm` (the short command name — "codex" or
    "claude"). This is critical: the launch chain is
    bash → node → codex, and all three have the schema file in argv.
    Without the comm filter we'd match the node wrapper, whose CPU
    activity tells us nothing about the underlying Rust client.

    `marker` (e.g. a per-call schema file path) further disambiguates
    when many same-comm processes run in parallel (codex judges).
    Without a marker we just take the first matching comm.
    """
    if marker:
        # Filter both by comm and by argv-contains-marker.
        # ps -eo pid,comm,args puts pid first, comm second, full argv
        # rest. Awk uses index() to substring-match the marker
        # anywhere in the full record (which includes args).
        cmd_str = (
            "ps -eo pid,comm,args | awk -v c=" + shlex.quote(comm)
            + " -v m=" + shlex.quote(marker)
            + " '$2 == c && index($0, m) > 0 {print $1; exit}'"
        )
    else:
        cmd_str = f"pgrep -x {shlex.quote(comm)} | head -1"
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, "sh", "-c", cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        text = out.decode().strip()
        if not text:
            return None
        return int(text.split()[0])
    except (asyncio.TimeoutError, ValueError, OSError):
        return None


async def _process_cpu_ticks(container_id: str, pid: int) -> int | None:
    """Read utime+stime (CPU ticks) for `pid` inside `container_id`
    via /proc/<pid>/stat. Fields 14 and 15 of /proc/<pid>/stat per
    proc(5). Returns None if the process is gone or read fails."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, "sh", "-c",
            f"awk '{{print $14, $15}}' /proc/{pid}/stat",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        parts = out.decode().split()
        if len(parts) != 2:
            return None
        return int(parts[0]) + int(parts[1])
    except (asyncio.TimeoutError, ValueError, OSError):
        return None


async def _watchdog(proc, container_id, kill_flag, *,
                     backend: str, marker: str | None = None,
                     label: str = ""):
    """Kill `proc` if its corresponding LLM subprocess inside
    `container_id` accumulates no CPU ticks for SILENCE_SECS, OR after
    HARD_WALL_SECS regardless. `kill_flag` is a one-element [bool] set
    to True before kill so the caller can tell a watchdog kill apart
    from other failures.

    `backend` is "codex" or "claude" — used for the comm fallback when
    `marker` is None.
    """
    started = time.monotonic()
    pid: int | None = None
    last_active = started
    last_ticks: int | None = None
    comm = "codex" if backend == "codex" else "claude"
    tag = f":{label}" if label else ""
    try:
        while proc.returncode is None:
            await asyncio.sleep(WATCHDOG_POLL_SECS)
            now = time.monotonic()
            if now - started > WATCHDOG_HARD_WALL_SECS:
                print(
                    f"[watchdog{tag}] hard wall-time "
                    f"{WATCHDOG_HARD_WALL_SECS}s exceeded — killing",
                    file=sys.stderr,
                )
                kill_flag[0] = True
                try: proc.kill()
                except ProcessLookupError: pass
                return

            # Lazily look up the in-container PID. The subprocess may
            # take a moment to start — if not found yet, wait the
            # next poll. Reset to None if the process has exited so
            # we re-discover (relevant for retries / proc lifecycle).
            if pid is None:
                pid = await _find_subprocess_pid(container_id, comm, marker)
                if pid is None:
                    continue
                last_ticks = await _process_cpu_ticks(container_id, pid)
                last_active = now  # baseline once we have a PID
                continue

            ticks = await _process_cpu_ticks(container_id, pid)
            if ticks is None:
                # Could be transient (docker exec timeout, scheduling
                # delay) or the process is genuinely gone. Don't exit
                # on a single failure — verify the PID is gone by
                # re-running the lookup. If still findable, treat as
                # transient and skip this sample. If missing, exit.
                still_there = await _find_subprocess_pid(
                    container_id, comm, marker,
                )
                if still_there is None:
                    return  # Process really did exit
                # Transient: don't update last_active, don't update
                # last_ticks, just wait for next poll.
                continue

            if last_ticks is not None and ticks != last_ticks:
                last_active = now  # any tick movement = activity
            last_ticks = ticks

            silent_secs = int(now - last_active)
            if silent_secs > WATCHDOG_SILENCE_SECS:
                print(
                    f"[watchdog{tag}] no CPU activity from pid {pid} "
                    f"for {silent_secs}s — killing",
                    file=sys.stderr,
                )
                kill_flag[0] = True
                try: proc.kill()
                except ProcessLookupError: pass
                return
    except asyncio.CancelledError:
        pass


async def _terminate_process_wrapper(proc) -> None:
    """Terminate and reap a host subprocess, escalating after a grace period."""
    if proc is None or proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(
            proc.wait(), timeout=PROVIDER_TERMINATE_TIMEOUT_SECS,
        )
        return
    except asyncio.TimeoutError:
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(
            proc.wait(), timeout=PROVIDER_TERMINATE_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        print(
            "[llm] provider subprocess did not exit after SIGKILL",
            file=sys.stderr,
        )


def _container_termination_script(marker: str) -> str:
    """Build a shell script that terminates one marked container exec tree.

    Docker does not guarantee that terminating its local ``docker exec`` CLI
    wrapper also terminates the already-started process in the container.  A
    unique, non-secret environment marker is inherited by the provider and
    every child in its launch chain.  Scanning ``/proc/*/environ`` lets the
    cleanup exec signal only that call, including schemaless calls where an
    argv marker is unavailable and concurrent calls where ``pkill codex``
    would be unsafe.
    """
    marker_entry = shlex.quote(f"{_CONTAINER_CALL_MARKER_ENV}={marker}")
    return (
        "pids=''; "
        "for envfile in /proc/[0-9]*/environ; do "
        "[ -r \"$envfile\" ] || continue; "
        "if tr '\\0' '\\n' < \"$envfile\" 2>/dev/null "
        f"| grep -Fqx -- {marker_entry}; then "
        "pid=${envfile#/proc/}; pid=${pid%/environ}; "
        "pids=\"$pids $pid\"; fi; done; "
        "[ -z \"$pids\" ] && exit 0; "
        "kill -TERM $pids 2>/dev/null || true; "
        "i=0; alive=\"$pids\"; "
        "while [ -n \"$alive\" ] && [ \"$i\" -lt 50 ]; do "
        "sleep 0.1; next=''; "
        "for pid in $alive; do "
        "kill -0 \"$pid\" 2>/dev/null && next=\"$next $pid\"; "
        "done; alive=\"$next\"; i=$((i + 1)); done; "
        "[ -z \"$alive\" ] || kill -KILL $alive 2>/dev/null || true"
    )


async def _terminate_container_call(container_id: str, marker: str) -> None:
    """Terminate all in-container processes belonging to one provider call."""
    cleanup = None
    try:
        cleanup = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, "sh", "-c",
            _container_termination_script(marker),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(
            cleanup.communicate(),
            timeout=PROVIDER_TERMINATE_TIMEOUT_SECS + 2,
        )
    except (OSError, asyncio.TimeoutError) as error:
        print(
            f"[llm] could not finish in-container provider cleanup: {error}",
            file=sys.stderr,
        )
        if cleanup is not None and cleanup.returncode is None:
            try:
                cleanup.kill()
            except ProcessLookupError:
                pass
            try:
                await cleanup.wait()
            except Exception:
                pass


async def _terminate_cancelled_provider(
    proc,
    *,
    container_id: str | None,
    container_marker: str | None,
) -> None:
    """Finish provider cleanup before cancellation can release its OSS gate."""
    has_container_call = container_id is not None and container_marker is not None
    if not has_container_call and (proc is None or proc.returncode is not None):
        return

    async def cleanup() -> None:
        if has_container_call:
            # Kill the real provider tree first. Merely terminating the local
            # docker CLI can leave Codex/Claude running and consuming the only
            # local-provider slot after the semaphore has been released.
            await _terminate_container_call(container_id, container_marker)
        if proc is not None:
            await _terminate_process_wrapper(proc)

    cleanup_task = asyncio.create_task(cleanup())
    # A second cancel request must not let the gate escape early. The original
    # CancelledError is re-raised by llm() after this helper finishes.
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            continue
    try:
        cleanup_task.result()
    except Exception as error:
        print(f"[llm] provider cancellation cleanup failed: {error}", file=sys.stderr)


# ── Resume from cache ───────────────────────────────────────────
#
# When the same call_dir already contains a successful response (from
# a prior run that we're resuming), reuse it — but ONLY after verifying
# the saved prompt.txt matches the current prompt. Idempotency check
# prevents silent corruption when pipeline code or prompt templates
# have drifted between runs.

def _try_resume_from_cache(
    call_dir: str,
    prompt: str,
    schema: dict | None,
    expected_identity: dict,
):
    """Returns (response, session_id, call_meta) if call_dir has a
    complete and idempotent prior result. Returns None if nothing
    cached. Raises if the cached prompt doesn't match the current
    prompt — that's a state-drift bug, not a fallback case."""
    response_path = os.path.join(call_dir, "response.txt")
    meta_path = os.path.join(call_dir, "meta.json")
    prompt_path = os.path.join(call_dir, "prompt.txt")
    if not (os.path.exists(response_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("failed"):
        return None
    rc = meta.get("returncode")
    if rc not in (0, None):
        return None
    cached_identity = meta.get("cache_identity")
    if not isinstance(cached_identity, dict):
        raise RuntimeError(
            f"resume idempotency check failed for {call_dir}: "
            "cached artifacts predate cache_identity fingerprints. "
            "Refusing to overwrite the original LLM evidence; move or delete "
            "the call directory to force a fresh invocation."
        )
    if cached_identity.get("sha256") != expected_identity.get("sha256"):
        cached_inputs = cached_identity.get("inputs") or {}
        current_inputs = expected_identity.get("inputs") or {}
        changed = sorted(
            key for key in set(cached_inputs) | set(current_inputs)
            if cached_inputs.get(key) != current_inputs.get(key)
        )
        detail = ", ".join(changed) if changed else "invocation inputs"
        raise RuntimeError(
            f"resume idempotency check failed for {call_dir}: "
            f"cached invocation differs in {detail}. "
            f"Pipeline or runtime state has drifted from the original run. "
            f"Either delete {call_dir} to force a fresh LLM call, "
            f"or restore the original configuration and runtime."
        )

    # Keep the human-readable prompt artifact as a second, independent check
    # against a corrupt or manually edited cache directory.
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path) as f:
                cached_prompt = f.read()
        except OSError:
            return None
        if cached_prompt != prompt:
            raise RuntimeError(
                f"resume idempotency check failed for {call_dir}: "
                f"saved prompt.txt ({len(cached_prompt)} chars) does "
                f"not match the current prompt ({len(prompt)} chars). "
                f"Pipeline state has drifted from the original run. "
                f"Either delete {call_dir} to force a fresh LLM call, "
                f"or revert the change that caused the drift."
            )
    # Read response
    try:
        with open(response_path) as f:
            response_text = f.read()
    except OSError:
        return None
    if schema:
        try:
            response = json.loads(response_text)
            _validate_structured_output(response, schema)
        except (json.JSONDecodeError, StructuredOutputError):
            return None
    else:
        response = response_text
    session_id = meta.get("session_id")
    cache_meta = dict(meta)
    cache_meta["resumed_from_cache"] = True
    return response, session_id, cache_meta


# ── Artifact helpers ────────────────────────────────────────────

def _write_artifact(path: str, data: bytes | str, *, gzip_it: bool = False) -> str:
    """Write raw bytes/text to `path`. With gzip_it=True, the stored
    file has a `.gz` suffix. Returns the actual written path.

    No atomic dance — artifacts are written once per call, never
    overwritten by other callers (call index + role make the directory
    name unique), so a plain write is fine. Crash safety here is not
    worth the extra complexity.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if gzip_it:
        out = path + ".gz"
        with gzip.open(out, "wb") as f:
            f.write(data)
        return out
    with open(path, "wb") as f:
        f.write(data)
    return path


def _cmd_hash(cmd: list[str]) -> str:
    """Short hash of an argv list, for grouping identical invocations."""
    return hashlib.sha256("\x1f".join(cmd).encode()).hexdigest()[:16]


def _effective_model(cmd: list[str]) -> str | None:
    """Pull the model name out of the cmd (whatever was actually sent).

    Both backends accept `--model X`. This sidesteps a subtle issue
    where the config says `model: opus` but the codex cmd builder
    translates that to `gpt-5.5`; `settings["model"]` still reads the
    pre-translation value. Recording the effective model avoids
    misleading metadata in the saved call_meta."""
    try:
        i = cmd.index("--model")
        return cmd[i + 1]
    except (ValueError, IndexError):
        return None


def _extract_claude_metadata(result_event: dict) -> dict:
    """Pull per-call tokens/cost/tool-use from a claude result event."""
    usage = result_event.get("usage") or {}
    server_tools = usage.get("server_tool_use") or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "total_cost_usd": result_event.get("total_cost_usd"),
        "num_turns": result_event.get("num_turns"),
        "duration_api_ms": result_event.get("duration_api_ms"),
        "stop_reason": result_event.get("stop_reason"),
        "is_error": result_event.get("is_error", False),
        "web_search_requests": server_tools.get("web_search_requests", 0),
        "web_fetch_requests": server_tools.get("web_fetch_requests", 0),
    }


def _extract_codex_metadata(events: list) -> dict:
    """Pull per-call tokens from codex turn.completed events.

    Codex can emit multiple turn.completed events per call (tool calls
    produce extra turns), so we sum across all of them.

    web_search_requests counts shell tool calls that invoke the sandbox
    `theoria-search` helper (the OSS-compatible search path — see
    web_search_config). Codex reports shell activity as
    `command_execution` items and generic tools as `function_call`
    items; both are checked. It is a substring heuristic over the
    command/arguments text, so a command that merely mentions the
    helper's name is counted too; treat it as approximate usage
    telemetry, parallel to the claude-side server_tool_use counter of
    the same name.
    """
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    web_search_requests = 0
    for event in events:
        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            item_type = item.get("type")
            invoked = (
                item.get("command")
                if item_type == "command_execution"
                else item.get("arguments") if item_type == "function_call"
                else None
            )
            if invoked and WEB_SEARCH_COMMAND in str(invoked):
                web_search_requests += 1
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage") or {}
        input_tokens += usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("output_tokens", 0) or 0
        cached_input_tokens += usage.get("cached_input_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "total_cost_usd": None,  # codex doesn't expose cost
        "web_search_requests": web_search_requests,
    }


_ADDITIVE_PROVIDER_META_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "total_cost_usd",
    "num_turns",
    "duration_api_ms",
    "web_search_requests",
    "web_fetch_requests",
}


def _aggregate_provider_attempts(attempts: list[dict]) -> dict:
    """Combine usage/tool metadata across schema-retry subprocesses."""
    if not attempts:
        return {}
    combined = dict(attempts[-1])
    for key in _ADDITIVE_PROVIDER_META_FIELDS:
        values = [item.get(key) for item in attempts]
        numeric = [value for value in values if isinstance(value, (int, float))]
        if numeric:
            combined[key] = sum(numeric)
        elif key in combined:
            combined[key] = None
    tool_calls = [
        call
        for item in attempts
        for call in (item.get("tool_calls") or [])
    ]
    if tool_calls:
        combined["tool_calls"] = tool_calls
    return combined


def _truncate(s, limit: int) -> str:
    """Truncate with a marker so partial content is obvious in logs."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s) if isinstance(s, (dict, list)) else str(s)
    if len(s) <= limit:
        return s
    return f"{s[:limit]}\n...[TRUNCATED: {len(s) - limit} of {len(s)} chars]"


def _maybe_truncate(s, limit: int, *, truncate: bool):
    """Apply _truncate(s, limit) when truncate=True; otherwise stringify
    without cutting. For the artifact copy we want the full payload."""
    if not truncate:
        if s is None:
            return ""
        if isinstance(s, str):
            return s
        if isinstance(s, (dict, list)):
            return json.dumps(s)
        return str(s)
    return _truncate(s, limit)


def _extract_claude_tool_calls(events: list, *, truncate: bool = True) -> list:
    """Pair claude tool_use events with their tool_result events.

    With truncate=False, tool inputs/outputs are kept in full — used for
    writing the untruncated tool_calls.json artifact.
    """
    uses, order = {}, []
    for e in events:
        for b in ((e.get("message") or {}).get("content") or []):
            if b.get("type") == "tool_use":
                uses[b["id"]] = {
                    "tool_name": b.get("name"),
                    "input": _maybe_truncate(
                        b.get("input"), TOOL_CALL_INPUT_LIMIT, truncate=truncate,
                    ),
                }
                order.append(b["id"])
            elif b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if tid in uses:
                    uses[tid]["output"] = _maybe_truncate(
                        b.get("content"), TOOL_CALL_OUTPUT_LIMIT,
                        truncate=truncate,
                    )
    return [uses[tid] for tid in order]


def _extract_codex_tool_calls(events: list, *, truncate: bool = True) -> list:
    """Normalize codex tool activity into the shared tool-call schema.

    Generic tools arrive as paired function_call / function_call_output
    items; shell activity arrives as self-contained `command_execution`
    items instead, which earlier versions of this extractor dropped —
    leaving tool_calls artifacts empty for shell-only calls.
    """
    calls, order = {}, []
    for e in events:
        if e.get("type") != "item.completed":
            continue
        item = e.get("item") or {}
        cid = item.get("call_id")
        if item.get("type") == "function_call" and cid:
            calls[cid] = {
                "tool_name": item.get("name"),
                "input": _maybe_truncate(
                    item.get("arguments"), TOOL_CALL_INPUT_LIMIT,
                    truncate=truncate,
                ),
            }
            order.append(cid)
        elif item.get("type") == "function_call_output" and cid in calls:
            calls[cid]["output"] = _maybe_truncate(
                item.get("output"), TOOL_CALL_OUTPUT_LIMIT, truncate=truncate,
            )
        elif item.get("type") == "command_execution":
            # Shell activity arrives as one self-contained item carrying
            # the command, its aggregated output, and the exit code.
            key = item.get("id") or f"cmd:{len(order)}"
            calls[key] = {
                "tool_name": "shell",
                "input": _maybe_truncate(
                    item.get("command"), TOOL_CALL_INPUT_LIMIT,
                    truncate=truncate,
                ),
                "output": _maybe_truncate(
                    item.get("aggregated_output"), TOOL_CALL_OUTPUT_LIMIT,
                    truncate=truncate,
                ),
                "exit_code": item.get("exit_code"),
            }
            order.append(key)
    return [calls[cid] for cid in order]


# ── Failure diagnostics ─────────────────────────────────────────

def _format_failure(backend, returncode, stderr_bytes, stdout_bytes=None):
    """Build an informative subprocess failure message.

    Includes returncode and any stderr content. If stdout is also passed
    (batch mode only), includes its tail too. The previous error format
    was just `f"{backend} failed: {stderr}"`, which produced messages like
    `"claude failed: "` (no payload) when stderr was empty — making the
    failure undebuggable.
    """
    parts = [f"{backend} failed (returncode={returncode})"]
    stderr_str = (stderr_bytes or b"").decode(errors="replace").strip()
    if stderr_str:
        parts.append(f"stderr: {stderr_str[:1500]}")
    if stdout_bytes is not None:
        stdout_str = (stdout_bytes or b"").decode(errors="replace").strip()
        if stdout_str:
            tail = stdout_str[-1000:]
            parts.append(f"stdout tail: ...{tail}")
    if len(parts) == 1:
        parts.append("(no stderr or stdout captured)")
    return " | ".join(parts)


# ── Claude ──────────────────────────────────────────────────────

def _build_claude_cmd(prompt, settings, schema, system, watch, resume, *,
                       sandboxed: bool = False):
    model = settings.get("model", "opus")
    effort = settings.get("effort", "max")
    tools = settings.get("tools")

    # Defense-in-depth: even if upstream missed sanitizing, never let a
    # null byte reach subprocess argv (raises "embedded null byte").
    prompt = _BAD_CTRL.sub("", prompt) if isinstance(prompt, str) else prompt
    system = _BAD_CTRL.sub("", system) if isinstance(system, str) else system

    cmd = ["claude", "--model", model, "--effort", effort, "-p", prompt]

    if resume:
        cmd += ["--resume", resume]

    if tools is not None:
        cmd += ["--tools", tools]

    if watch:
        # Stream-json for live events — works with or without schema
        cmd += ["--output-format", "stream-json", "--verbose"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
    else:
        # Always use JSON format so we can capture session_id
        cmd += ["--output-format", "json"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]

    if system:
        cmd += ["--append-system-prompt", system]

    # Inside a Docker container, the container IS the sandbox — skip
    # claude's native permission prompts/Seatbelt. Outside the container
    # we leave behavior unchanged (print mode already doesn't prompt
    # interactively).
    if sandboxed:
        cmd += ["--dangerously-skip-permissions"]

    return cmd


def _parse_claude_output(stdout, schema):
    """Parse non-streaming claude output.

    Returns (response, session_id, metadata, events). The events list is
    returned so callers can persist the full untruncated event stream
    without re-parsing.

    With `--output-format json` (what the batch path uses), claude emits
    a single result object — not a list of events. Older versions, and
    some wrappers, emitted a JSON array of events instead, so we accept
    both shapes.
    """
    parsed = json.loads(stdout)

    if isinstance(parsed, dict):
        # Single result object (current `--output-format json` shape).
        # There are no intermediate tool-use events in this mode — tool
        # calls can only be captured via streaming.
        result_event = parsed
        events = [parsed]
    elif isinstance(parsed, list):
        events = parsed
        try:
            result_event = next(
                e for e in reversed(events)
                if isinstance(e, dict) and e.get("type") == "result"
            )
        except StopIteration:
            raise RuntimeError(
                "claude output contained no result event. "
                f"got {len(events)} items; last={events[-1] if events else None!r}"
            )
    else:
        raise RuntimeError(
            f"unexpected claude output shape: {type(parsed).__name__} "
            f"(first 500 chars: {stdout[:500]!r})"
        )
    session_id = result_event.get("session_id")
    metadata = _extract_claude_metadata(result_event)
    metadata["tool_calls"] = _extract_claude_tool_calls(events)
    if schema:
        if "structured_output" not in result_event:
            # Diagnostic: dump everything we know about the failed result
            raise StructuredOutputError(
                "claude returned a result event without 'structured_output'. "
                f"is_error={result_event.get('is_error')!r} "
                f"subtype={result_event.get('subtype')!r} "
                f"stop_reason={result_event.get('stop_reason')!r} "
                f"result={(result_event.get('result') or '')[:500]!r} "
                f"keys={list(result_event.keys())}",
                events=events,
                provider_meta=metadata,
            )
        return _sanitize_llm_output(result_event["structured_output"]), session_id, metadata, events
    return _sanitize_llm_output(result_event.get("result", "")), session_id, metadata, events


def _print_claude_event(event):
    """Print a claude stream-json event to stderr."""
    t = event.get("type")

    if t == "assistant":
        content = event.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                print(f"      [tool] {block['name']}({json.dumps(block.get('input', {}))[:100]})", file=sys.stderr)
            elif block.get("type") == "text":
                text = block["text"][:200]
                if text.strip():
                    print(f"      [text] {text}", file=sys.stderr)

    elif t == "result":
        cost = event.get("total_cost_usd")
        turns = event.get("num_turns", 0)
        if cost is not None:
            print(f"      [done] {turns} turns, ${cost:.4f}", file=sys.stderr)


async def _run_claude_streaming(proc, schema, *, last_event_ref=None):
    """Read claude stream-json line by line, print events, return
    (response, session_id, metadata, events, raw_stdout).

    `raw_stdout` is the concatenated bytes we read — kept so callers
    can save the untouched provider output as an artifact. `events` is
    the parsed per-line list, for the same reason.
    """
    result_event = None
    session_id = None
    events = []
    raw_chunks: list[bytes] = []

    async for raw_line in proc.stdout:
        if last_event_ref is not None:
            last_event_ref[0] = time.monotonic()
        raw_chunks.append(raw_line)
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        events.append(event)
        _print_claude_event(event)

        if session_id is None and event.get("session_id"):
            session_id = event["session_id"]

        if event.get("type") == "result":
            result_event = event
            # claude's protocol emits exactly one terminal "result" event
            # per CLI invocation. Without this break, the loop continues
            # waiting for stdout EOF — which through `docker exec` can
            # hang indefinitely even after claude itself has exited
            # (manifests as the watch output showing [done] but the
            # pipeline never recording formalizer_returned).
            break

    await proc.wait()
    raw_stdout = b"".join(raw_chunks)

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise ProviderProcessError(
            _format_failure("claude", proc.returncode, stderr, raw_stdout),
            raw_stdout=raw_stdout,
            raw_stderr=stderr,
            events=events,
        )

    if result_event is None:
        if schema:
            raise StructuredOutputError(
                "claude stream ended without result event",
                raw_stdout=raw_stdout,
                events=events,
            )
        raise ProviderProcessError(
            "claude stream ended without result event",
            raw_stdout=raw_stdout,
            events=events,
        )

    metadata = _extract_claude_metadata(result_event)
    metadata["tool_calls"] = _extract_claude_tool_calls(events)
    if schema:
        if "structured_output" not in result_event:
            raise StructuredOutputError(
                "claude returned a result event without 'structured_output'. "
                f"is_error={result_event.get('is_error')!r} "
                f"subtype={result_event.get('subtype')!r} "
                f"stop_reason={result_event.get('stop_reason')!r} "
                f"result={(result_event.get('result') or '')[:500]!r} "
                f"keys={list(result_event.keys())}",
                raw_stdout=raw_stdout,
                events=events,
                provider_meta=metadata,
            )
        return _sanitize_llm_output(result_event["structured_output"]), session_id, metadata, events, raw_stdout
    return _sanitize_llm_output(result_event.get("result", "")), session_id, metadata, events, raw_stdout


# ── Codex ───────────────────────────────────────────────────────

# Codex roles that are invoked sequentially and can be resumed on
# a later call. These must share CODEX_HOME so that `codex exec
# resume <session_id>` can locate the rollout file written by the
# initial call. Roles not in this set are either always-initial or
# run in parallel, and get per-call CODEX_HOME isolation to avoid
# concurrent processes corrupting each other's SQLite state.
_RESUMABLE_CODEX_ROLES = {"solver", "formalizer"}

_CLAUDE_MODEL_ALIASES = {"opus", "sonnet", "haiku"}
_CODEX_LOCAL_PROVIDERS = {"ollama", "lmstudio"}
_CODEX_CONFIG_KEY = re.compile(
    r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$"
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The in-sandbox search helper (sandbox images install search_cli.py at
# /usr/local/bin/theoria-search; `pip install -e .` provides the same
# console script for explicitly opted-in host runs). Agents reach it
# through the ordinary shell function tool, which every Codex provider
# supports — unlike the provider-side `web_search` and MCP `namespace`
# tool types, which local providers reject at the API layer.
WEB_SEARCH_COMMAND = "theoria-search"
_WEB_SEARCH_PROVIDERS = {"brave", "searxng"}
_WEB_SEARCH_FIELDS = {"provider", "api_key_env", "endpoint"}
_WEB_SEARCH_DEFAULT_KEY_ENV = {"brave": "BRAVE_API_KEY"}
_WEB_SEARCH_PROVIDER_ENV = "THEORIA_SEARCH_PROVIDER"
_WEB_SEARCH_ENDPOINT_ENV = "THEORIA_SEARCH_ENDPOINT"
_WEB_SEARCH_API_KEY_NAME_ENV = "THEORIA_SEARCH_API_KEY_ENV"
_SENSITIVE_CODEX_CONFIG_PARTS = {
    "api_key", "authorization", "bearer_token", "cookie", "credential",
    "key", "password", "secret", "token",
}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _toml_scalar(value) -> str:
    """Serialize a scalar for Codex's TOML-parsed `-c key=value` flag."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    raise ValueError(
        "codex_config values must be TOML scalars "
        "(string, boolean, finite number); "
        f"got {type(value).__name__}"
    )


def web_search_config(raw) -> dict:
    """Validate the run-level `_web_search` declaration.

    Returns {} when web search is not configured, otherwise a normalized
    mapping per provider:

      brave   → {"provider", "api_key_env"} — hosted keyed API on a
                fixed endpoint. The credential is referenced by
                environment-variable name only; the harness forwards
                that one variable into the per-problem sandbox, where
                the model-facing theoria-search command reads it. The
                value never appears in YAML, argv, or metadata.
      searxng → {"provider", "endpoint"} — self-hosted metasearch, no
                credential. The endpoint gets the same validation and
                loopback→host.docker.internal routing as an OSS model
                endpoint, and reaches the helper via container env.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("_web_search must be a mapping")
    unknown = sorted(str(field) for field in raw if field not in _WEB_SEARCH_FIELDS)
    if unknown:
        raise ValueError(
            "_web_search has unsupported field(s): " + ", ".join(unknown)
        )
    provider = raw.get("provider")
    if provider not in _WEB_SEARCH_PROVIDERS:
        allowed = ", ".join(sorted(_WEB_SEARCH_PROVIDERS))
        raise ValueError(f"_web_search.provider must be one of: {allowed}")

    if provider == "brave":
        if "endpoint" in raw:
            raise ValueError(
                "_web_search.endpoint is not configurable for brave: the "
                "hosted API endpoint is fixed by design (the "
                "THEORIA_SEARCH_ENDPOINT env override exists for tests only)"
            )
        api_key_env = raw.get(
            "api_key_env", _WEB_SEARCH_DEFAULT_KEY_ENV[provider],
        )
        if not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env):
            raise ValueError(
                "_web_search.api_key_env must be an environment-variable name"
            )
        return {"provider": provider, "api_key_env": api_key_env}

    # searxng
    if "api_key_env" in raw:
        raise ValueError(
            "_web_search.api_key_env is not supported for searxng; the "
            "instance is unauthenticated — keep it bound to localhost"
        )
    endpoint = raw.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError(
            "_web_search.endpoint is required for searxng (the base URL of "
            "your instance, e.g. http://localhost:8888)"
        )
    # Same shape rules as OSS model endpoints: absolute http(s), no
    # credentials/query/fragment. Routing happens at call time.
    _route_oss_base_url(endpoint, sandboxed=False)
    return {"provider": provider, "endpoint": endpoint}


def web_search_env_names(raw) -> list[str]:
    """Return the environment reference declared by `_web_search`."""
    config = web_search_config(raw)
    api_key_env = config.get("api_key_env")
    return [api_key_env] if api_key_env else []


def _is_sensitive_codex_config_key(key: str) -> bool:
    """Return whether an override would put secret material in argv."""
    lower = key.lower()
    if lower.endswith((
        ".env_key", ".env_var", ".env_vars", ".bearer_token_env_var",
    )):
        return False
    parts = set(re.split(r"[._-]", lower))
    if (parts & _SENSITIVE_CODEX_CONFIG_PARTS
            or {"api", "key"}.issubset(parts)):
        return True
    return False


def _codex_config_items(settings: dict) -> list[tuple[str, object]]:
    """Validate and return structured Codex CLI configuration overrides."""
    raw = settings.get("codex_config")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("codex_config must be a mapping of dotted keys to scalars")

    items = []
    for key, value in raw.items():
        if not isinstance(key, str) or not _CODEX_CONFIG_KEY.fullmatch(key):
            raise ValueError(
                "codex_config keys must be non-empty dotted identifiers; "
                f"got {key!r}"
            )
        if key == "mcp_servers" or key.startswith("mcp_servers."):
            raise ValueError(
                "codex_config cannot declare MCP servers: an MCP command is "
                "executable code outside Theoria's trust-domain accounting, "
                "and Codex serializes MCP tools as Responses API namespace "
                "tools that local providers reject. Use the _web_search "
                "shell helper for OSS search instead"
            )
        if _is_sensitive_codex_config_key(key):
            raise ValueError(
                f"codex_config.{key} would expose a secret in process argv; "
                "configure the provider's env_key and list the variable in "
                "provider_env instead"
            )
        _toml_scalar(value)  # validate before constructing any subprocess
        if key.endswith("base_url"):
            _route_oss_base_url(value, sandboxed=False)
        if key == "model_provider" and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError("codex_config.model_provider must be a non-empty string")
        items.append((key, value))
    return items


def _codex_config_override(key: str, value, *, sandboxed: bool) -> str:
    """Serialize one override, routing custom-provider loopback URLs."""
    if key.startswith("model_providers.") and key.endswith(".base_url"):
        value = _route_oss_base_url(value, sandboxed=sandboxed)
    return f"{key}={_toml_scalar(value)}"


def _route_oss_base_url(value, *, sandboxed: bool) -> str | None:
    """Validate an OSS endpoint and route host loopback from Docker."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("oss_base_url must be a non-empty http(s) URL")

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as e:
        raise ValueError(f"invalid oss_base_url: {e}") from e
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("oss_base_url must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "oss_base_url cannot contain credentials, a query, or a fragment; "
            "pass provider credentials through provider_env"
        )

    hostname = parsed.hostname
    if sandboxed and hostname.lower() in _LOOPBACK_HOSTS:
        hostname = "host.docker.internal"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _codex_provider(settings: dict) -> str | None:
    """Return the effective configured provider without exposing secrets."""
    if settings.get("oss"):
        return settings.get("local_provider")
    for key, value in _codex_config_items(settings):
        if key == "model_provider" and isinstance(value, str):
            return value
    return "openai"


def _configured_codex_base_url(settings: dict) -> str | None:
    """Return a validated base URL from OSS or custom-provider settings."""
    if settings.get("oss_base_url") is not None:
        return _route_oss_base_url(
            settings.get("oss_base_url"), sandboxed=False,
        )
    provider = _codex_provider(settings)
    key = f"model_providers.{provider}.base_url"
    for config_key, value in _codex_config_items(settings):
        if config_key == key:
            return _route_oss_base_url(value, sandboxed=False)
    return None


def _effective_codex_base_url_for_identity(settings: dict) -> str | None:
    """Return the endpoint that can affect a cached Codex invocation.

    Local OSS calls prefer the process-level ``CODEX_OSS_BASE_URL`` override
    at launch time.  Cache identity must mirror that precedence; otherwise a
    resumed run can reuse output produced by a different provider endpoint.
    URL validation forbids embedded credentials, query strings, and fragments,
    and the enclosing invocation settings are stored only through a hash.
    """
    if settings.get("oss"):
        env_base_url = os.environ.get("CODEX_OSS_BASE_URL")
        if env_base_url is not None:
            return _route_oss_base_url(env_base_url, sandboxed=False)
    return _configured_codex_base_url(settings)


def _uses_external_codex_provider(settings: dict) -> bool:
    """Return whether Codex is configured outside its native OpenAI path."""
    provider = _codex_provider(settings)
    return bool(
        settings.get("oss")
        or provider not in (None, "openai")
        or _configured_codex_base_url(settings) is not None
        or _provider_env_names(settings)
    )


def uses_external_codex_provider(settings: dict) -> bool:
    """Public wrapper for shared Codex provider security checks."""
    return _uses_external_codex_provider(settings)


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _call_cache_identity(
    *,
    prompt: str,
    system: str | None,
    schema: dict | None,
    role: str,
    backend: str,
    settings: dict,
    watch: bool,
    resume: str | None,
    sandboxed: bool,
    image_id: str | None,
    codex_version: str | None,
) -> dict:
    """Build a versioned, secret-free identity for one model invocation.

    Prompt, system, schema, resume id, and tool/provider settings participate
    only through hashes. Provider credentials never enter this function:
    `provider_env` contributes environment-variable *names*, while Codex
    config has already rejected literal credential values.
    """
    if backend == "codex":
        configured_model = settings.get("model")
        effective_model = (
            "gpt-5.5"
            if configured_model in _CLAUDE_MODEL_ALIASES
            else configured_model
        )
        invocation_settings = {
            "effort": settings.get("effort"),
            "sandbox": settings.get("sandbox"),
            "search": settings.get("search"),
            "full_auto": bool(settings.get("full_auto", False)),
            "oss": bool(settings.get("oss", False)),
            "local_provider": settings.get("local_provider"),
            "base_url": _effective_codex_base_url_for_identity(settings),
            "provider_env": sorted(_provider_env_names(settings)),
            "codex_config": sorted(
                _codex_config_items(settings), key=lambda item: item[0]
            ),
            "schema_retries": settings.get("schema_retries", 0),
        }
        provider = _codex_provider(settings)
    else:
        effective_model = settings.get("model")
        invocation_settings = {
            "effort": settings.get("effort"),
            "tools": settings.get("tools"),
        }
        provider = None

    payload = {
        "version": 1,
        "role": role,
        "backend": backend,
        "model": effective_model,
        "provider": provider,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "system_sha256": hashlib.sha256(
            (system or "").encode("utf-8")
        ).hexdigest(),
        "schema_sha256": _json_sha256(schema),
        "resume_sha256": hashlib.sha256(
            (resume or "").encode("utf-8")
        ).hexdigest(),
        "invocation_settings_sha256": _json_sha256(invocation_settings),
        "sandboxed": sandboxed,
        "image_id": image_id,
        "codex_version": codex_version if backend == "codex" else None,
    }
    return {
        "version": 1,
        "sha256": _json_sha256(payload),
        "inputs": payload,
    }


def _schema_retry_limit(settings: dict) -> int:
    value = settings.get("schema_retries", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("schema_retries must be an integer")
    if not 0 <= value <= SCHEMA_RETRY_MAX:
        raise ValueError(
            f"schema_retries must be between 0 and {SCHEMA_RETRY_MAX}"
        )
    return value


def _provider_env_names(settings: dict) -> list[str]:
    value = settings.get("provider_env", [])
    if isinstance(value, str):
        value = [value]
    if value is None:
        value = []
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) or not _ENV_NAME.fullmatch(name)
        for name in value
    ):
        raise ValueError(
            "provider_env must contain only environment-variable names"
        )
    return list(dict.fromkeys(value))


def _build_codex_cmd(prompt, settings, schema_file, system, resume, *,
                      sandboxed: bool = False, role: str | None = None):
    oss = settings.get("oss", False)
    if not isinstance(oss, bool):
        raise ValueError("oss must be a boolean")

    config_items = _codex_config_items(settings)
    provider_env = _provider_env_names(settings)
    configured_provider = next(
        (value for key, value in config_items if key == "model_provider"),
        None,
    )
    external_provider = _uses_external_codex_provider(settings)
    model = settings.get("model")
    if external_provider and (not isinstance(model, str) or not model.strip()):
        raise ValueError("External Codex provider roles require an explicit model")
    if model is None:
        model = "gpt-5.5"
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Codex model must be a non-empty string")
    sandbox = settings.get("sandbox", "read-only")
    effort = settings.get("effort", "xhigh")
    local_provider = settings.get("local_provider")
    if oss:
        if local_provider not in _CODEX_LOCAL_PROVIDERS:
            allowed = ", ".join(sorted(_CODEX_LOCAL_PROVIDERS))
            raise ValueError(
                f"OSS Codex roles require local_provider to be one of: {allowed}"
            )
        if model in _CLAUDE_MODEL_ALIASES:
            raise ValueError(
                f"OSS Codex model {model!r} is a Claude alias; "
                "configure the explicit local model id"
            )
    elif local_provider is not None:
        raise ValueError("local_provider requires oss: true")
    if not oss and settings.get("oss_base_url") is not None:
        raise ValueError("oss_base_url requires oss: true")

    credential_refs = {
        value
        for key, value in config_items
        if key.lower().endswith((
            ".env_key", ".env_var", ".bearer_token_env_var",
        ))
    }
    missing_credential_refs = sorted(credential_refs - set(provider_env))
    if missing_credential_refs:
        raise ValueError(
            "Codex provider credential environment references must also "
            "appear in provider_env: " + ", ".join(missing_credential_refs)
        )
    if oss and configured_provider not in (None, local_provider):
        raise ValueError(
            "codex_config.model_provider conflicts with local_provider"
        )
    if (configured_provider not in (None, "openai")
            and model in _CLAUDE_MODEL_ALIASES):
        raise ValueError(
            f"Codex model {model!r} is a Claude alias; configure the "
            "explicit provider model id"
        )

    # Defense-in-depth: never let a null byte reach subprocess argv.
    prompt = _BAD_CTRL.sub("", prompt) if isinstance(prompt, str) else prompt
    system = _BAD_CTRL.sub("", system) if isinstance(system, str) else system

    # Translate claude's "max" to codex's "xhigh" (both mean highest reasoning)
    if effort == "max":
        effort = "xhigh"

    # claude uses "opus"/"sonnet"/"haiku" aliases; if the config has a claude
    # model but backend is codex, fall back to a codex model
    if model in _CLAUDE_MODEL_ALIASES:
        model = "gpt-5.5"

    if resume:
        # `codex exec resume` accepts config/model/schema flags but not
        # --oss, --local-provider, or --sandbox. Preserve the local provider
        # explicitly through model_provider; the sandbox is inherited from
        # the original session. The pipeline keeps Codex formalization
        # stateless even though current pinned Codex supports schema resume.
        #
        # It DOES require --skip-git-repo-check and
        # --dangerously-bypass-approvals-and-sandbox when running
        # inside our container: codex re-runs its per-exec trust and
        # approval checks on every resume call, not just the first
        # session, so without these flags a resumed call fails with
        # "Not inside a trusted directory" and hangs waiting for tool
        # approval. The container is our isolation boundary; these
        # flags are safe here for the same reason as on the initial call.
        cmd = ["codex", "exec", "resume", resume]
        cmd += ["--model", model]
        if sandboxed:
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
            cmd += ["--skip-git-repo-check"]
        for key, value in config_items:
            cmd += ["-c", _codex_config_override(
                key, value, sandboxed=sandboxed,
            )]
        if oss and configured_provider is None:
            cmd += ["-c", f"model_provider={_toml_scalar(local_provider)}"]
        elif configured_provider is None:
            cmd += ["-c", 'model_provider="openai"']
        if effort is not None:
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        cmd += ["--json"]
        if schema_file:
            cmd += ["--output-schema", schema_file]
    else:
        cmd = ["codex", "exec"]
        cmd += ["--model", model]
        if oss:
            cmd += ["--oss", "--local-provider", local_provider]
        # Inside a Docker container we trust the container as the
        # sandbox and drop codex's internal Seatbelt/bubblewrap +
        # approval checks. Outside, keep the native sandbox (default
        # read-only) so codex can't accidentally rampage on the host.
        if sandboxed:
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
            # /workspace inside the container isn't a git repo; codex
            # refuses to run otherwise. The flag is safe here because
            # the container is the isolation boundary.
            cmd += ["--skip-git-repo-check"]
        else:
            cmd += ["--sandbox", sandbox]
        for key, value in config_items:
            cmd += ["-c", _codex_config_override(
                key, value, sandboxed=sandboxed,
            )]
        if not oss and configured_provider is None:
            cmd += ["-c", 'model_provider="openai"']
        if effort is not None:
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        cmd += ["--json"]
        if settings.get("full_auto") and not sandboxed:
            # --full-auto is shorthand for --sandbox workspace-write.
            # Redundant (and conflicting) with the bypass flag above.
            cmd += ["--full-auto"]
        if schema_file:
            cmd += ["--output-schema", schema_file]

    search = settings.get("search")
    if search is not None:
        if not isinstance(search, bool):
            raise ValueError("search must be a boolean")
        if search and oss:
            raise ValueError(
                "search: true enables Codex's native web-search tool, a "
                "Responses API tool type that local providers reject before "
                "inference. Keep search: false for OSS roles and stack "
                "configs/brave_search.yaml for shell-based search instead"
            )
        # Codex 0.133 uses the top-level enum to decide whether the native
        # Responses API web-search tool is sent to the provider. Its legacy
        # tools.web_search boolean is ignored; the nested table now only
        # holds options for an enabled tool.
        search_mode = "live" if search else "disabled"
        cmd += ["-c", f"web_search={_toml_scalar(search_mode)}"]

    # Codex enables its multi-agent namespace tool by default. Ollama and LM
    # Studio accept ordinary function tools but reject the Responses API's
    # `namespace` tool type before the model sees the prompt. Keep the agentic
    # shell loop while disabling both incompatible multi-agent implementations
    # for OSS adapters. An explicit provider override remains possible.
    #
    # unified_exec is also disabled for OSS. With it on, Codex advertises
    # shell access as a PTY-session tool pair (exec_command/write_stdin);
    # with it off, as the single classic shell tool. Both are plain
    # function types that providers accept, but the classic tool matches
    # the shell-tool shape local models are trained on, drops a tool from
    # an already-long prompt, and gives up only interactive sessions,
    # which no Theoria role uses.
    if oss:
        configured_keys = {key for key, _ in config_items}
        for feature in ("multi_agent", "multi_agent_v2", "unified_exec"):
            key = f"features.{feature}"
            if key not in configured_keys:
                cmd += ["-c", f"{key}=false"]

    if system and not resume:
        # System prompt only on initial call; resume continues existing context
        cmd.append(f"{system}\n\n{prompt}")
    else:
        cmd.append(prompt)

    # Parallel codex calls (judges, pedantry, state 0 audit) share
    # /home/node/.codex when sandboxed — including sqlite DBs (logs_*.sqlite,
    # state_*.sqlite) that codex mmaps. Concurrent processes truncating or
    # re-initializing these files trigger SIGBUS (exit 135) in the other
    # processes. Give each INITIAL call its own CODEX_HOME under /tmp so
    # their state is fully isolated.
    #
    # EXCEPTION: roles that can be resumed (solver, formalizer) MUST
    # use the shared /home/node/.codex. `codex exec resume <session>`
    # looks up the rollout at $CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl
    # (confirmed via codex docs + session-storage refs). If the initial
    # call wrote the rollout to a per-call /tmp/codex-XXX, the later
    # resume using the default $HOME can't find it and fails with
    # "no rollout found for thread <id>". Keeping these roles on
    # shared CODEX_HOME preserves session continuity. They're called
    # sequentially per problem so no parallel-state corruption risk.
    is_resumable = role in _RESUMABLE_CODEX_ROLES
    if sandboxed and not resume and not is_resumable:
        uid = uuid.uuid4().hex[:12]
        # Under the tmpfs home, not /tmp: codex refuses to create its
        # helper binaries beneath a temporary directory and warns on
        # every call ("Refusing to create helper binaries under
        # temporary dir"). The home tmpfs is mounted exec for exactly
        # this kind of per-call state.
        home = f"/home/node/.codex-call-{uid}"
        inner = " ".join(shlex.quote(a) for a in cmd)
        cmd = [
            "bash", "-c",
            f"mkdir -p {home} && "
            f"cp -f /home/node/.codex/auth.json /home/node/.codex/config.toml "
            f"/home/node/.codex/installation_id {home}/ 2>/dev/null; "
            f"CODEX_HOME={home} {inner}"
        ]

    return cmd


def _parse_codex_output(stdout, schema):
    """Parse non-streaming codex output.

    Returns (response, session_id, metadata, events) — events is the
    parsed JSONL list, kept so callers can save it as an artifact.
    """
    try:
        lines = [
            json.loads(line)
            for line in stdout.strip().split("\n")
            if line.strip()
        ]
    except json.JSONDecodeError as e:
        if schema:
            raise StructuredOutputError(
                f"codex emitted invalid JSONL while schema output was requested: {e}"
            ) from e
        raise
    metadata = _extract_codex_metadata(lines)
    metadata["tool_calls"] = _extract_codex_tool_calls(lines)

    session_id = None
    for event in lines:
        if event.get("type") == "thread.started":
            session_id = event.get("thread_id")
            break

    for event in reversed(lines):
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if schema:
                    try:
                        return _sanitize_llm_output(json.loads(text)), session_id, metadata, lines
                    except json.JSONDecodeError as e:
                        raise StructuredOutputError(
                            f"codex returned non-JSON when schema was requested. "
                            f"text={text[:500]!r} error={e}",
                            events=lines,
                            provider_meta=metadata,
                        ) from e
                return _sanitize_llm_output(text), session_id, metadata, lines

    if schema:
        raise StructuredOutputError(
            "No response found in codex output",
            events=lines,
            provider_meta=metadata,
        )
    raise RuntimeError("No response found in codex output")


def _print_codex_event(event):
    """Print a codex JSONL event to stderr."""
    t = event.get("type")

    if t == "item.completed":
        item = event.get("item", {})
        item_type = item.get("type", "")

        if item_type == "function_call":
            arguments = _truncate(item.get("arguments", ""), 100)
            print(f"      [tool] {item.get('name', '?')}({arguments})", file=sys.stderr)
        elif item_type == "function_call_output":
            output = _truncate(item.get("output", ""), 200)
            print(f"      [result] {output}", file=sys.stderr)
        elif item_type == "command_execution":
            command = _truncate(item.get("command", ""), 160)
            print(f"      [shell] {command}", file=sys.stderr)
            output = _truncate(item.get("aggregated_output", ""), 200)
            if output.strip():
                print(f"      [result] {output}", file=sys.stderr)
        elif item_type == "agent_message":
            text = item.get("text", "")[:200]
            if text.strip():
                print(f"      [text] {text}", file=sys.stderr)

    elif t == "turn.completed":
        usage = event.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        print(f"      [done] {tokens} tokens", file=sys.stderr)


async def _run_codex_streaming(proc, schema, *, last_event_ref=None):
    """Read codex JSONL line by line, print events, return
    (response, session_id, metadata, events, raw_stdout)."""
    last_message_text = None
    session_id = None
    events = []
    raw_chunks: list[bytes] = []

    async for raw_line in proc.stdout:
        if last_event_ref is not None:
            last_event_ref[0] = time.monotonic()
        raw_chunks.append(raw_line)
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        events.append(event)
        _print_codex_event(event)

        if event.get("type") == "thread.started" and session_id is None:
            session_id = event.get("thread_id")

        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                last_message_text = item.get("text", "")

    await proc.wait()
    raw_stdout = b"".join(raw_chunks)

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise ProviderProcessError(
            _format_failure("codex", proc.returncode, stderr, raw_stdout),
            raw_stdout=raw_stdout,
            raw_stderr=stderr,
            events=events,
        )

    if last_message_text is None:
        if schema:
            raise StructuredOutputError(
                "codex stream ended without agent message",
                raw_stdout=raw_stdout,
                events=events,
                provider_meta=_extract_codex_metadata(events),
            )
        raise ProviderProcessError(
            "codex stream ended without agent message",
            raw_stdout=raw_stdout,
            events=events,
        )

    metadata = _extract_codex_metadata(events)
    metadata["tool_calls"] = _extract_codex_tool_calls(events)
    if schema:
        try:
            response = json.loads(last_message_text)
        except json.JSONDecodeError as e:
            raise StructuredOutputError(
                "codex returned non-JSON when schema was requested. "
                f"text={last_message_text[:500]!r} error={e}",
                raw_stdout=raw_stdout,
                events=events,
                provider_meta=metadata,
            ) from e
        return _sanitize_llm_output(response), session_id, metadata, events, raw_stdout
    return _sanitize_llm_output(last_message_text), session_id, metadata, events, raw_stdout


# ── Schema helper ───────────────────────────────────────────────

def _add_additional_properties(schema):
    """Codex requires additionalProperties: false on all objects."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    for key in ("properties", "items"):
        if key in out:
            val = out[key]
            if isinstance(val, dict):
                if key == "items":
                    out[key] = _add_additional_properties(val)
                else:
                    out[key] = {k: _add_additional_properties(v) for k, v in val.items()}
    return out


def _validate_structured_output(value, schema: dict) -> None:
    """Fail explicitly when parsed provider output violates its schema."""
    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except SchemaError as e:
        raise ValueError(f"invalid output schema: {e.message}") from e
    try:
        validator_cls(schema).validate(value)
    except ValidationError as e:
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in e.absolute_path
        )
        raise StructuredOutputError(
            f"structured output violates schema at {path}: {e.message}"
        ) from e


# ── Main entry point ────────────────────────────────────────────

async def llm(
    prompt: str,
    *,
    role: str = "solver",
    schema: dict | None = None,
    system: str | None = None,
    config: dict = {},
    watch: bool = False,
    resume: str | None = None,
) -> tuple[str | dict, str | None]:
    """Call an LLM. Backend (claude/codex) determined by config for the role.

    Returns (response, session_id). Pass resume=session_id on a later call to
    continue the same conversation.

    When watch=True, streams events to stderr in real time.

    When the `artifact_dir` contextvar is set, every call writes:
        <artifact_dir>/call_NNN_<role>/
            cmd.json              — the exact argv run
            prompt.txt            — the full user prompt
            system.txt            — the full system prompt (if any)
            stdout.jsonl[.gz]     — raw provider stdout
            stderr.txt[.gz]       — raw provider stderr
            response.txt          — parsed response text (or JSON)
            events.json.gz        — parsed provider event stream
            tool_calls.json       — untruncated tool call I/O
            meta.json             — call-level metadata (duration, tokens, etc.)

    The call_log entry keeps its existing truncated fields as previews;
    full source of truth is the files on disk.
    """
    role_settings = config.get(role, {})
    if not isinstance(role_settings, dict):
        raise ValueError(f"config for role {role!r} must be a mapping")
    # Defaults are per-call; do not mutate the merged run configuration.
    settings = dict(role_settings)
    backend = settings.get("backend", "claude")

    # Apply backend-specific defaults (config values take precedence)
    if backend == "codex":
        # Native OpenAI retains the historical default. External providers
        # use provider-specific model/deployment ids and must be explicit.
        if not _uses_external_codex_provider(settings):
            settings.setdefault("model", "gpt-5.5")
        settings.setdefault("effort", "xhigh")
        settings.setdefault("sandbox", "read-only")
        settings.setdefault("search", not settings.get("oss", False))
    elif backend == "claude":
        settings.setdefault("model", "opus")
        settings.setdefault("effort", "max")

    # ── Reserve a slot in the call log ───────────────────────────
    #
    # Parallel judges under asyncio.gather all share the same call_log
    # list via contextvars. If we allocated the index at the END of the
    # call (when we append call_meta), two judges running concurrently
    # would race: both would read len(log) at append time and collide.
    # Reserving a placeholder synchronously at the start — before any
    # await — guarantees a unique index per call. We fill it in later.
    log = call_log.get()
    if log is None:
        call_index = None
    else:
        call_index = len(log)
        log.append(None)  # reserve slot; replaced with call_meta below

    # ── Set up the per-call artifact directory ───────────────────
    base_artifact_dir = artifact_dir.get()
    call_dir: str | None = None
    if base_artifact_dir is not None and call_index is not None:
        call_dir = os.path.join(
            base_artifact_dir, f"call_{call_index:03d}_{role}",
        )
        os.makedirs(call_dir, exist_ok=True)

    container_id = sandbox_container.get()
    image_id = sandbox_image_id.get()
    sandboxed = container_id is not None
    effective_codex_version = (
        codex_cli_version.get() if backend == "codex" else None
    )
    cache_identity = _call_cache_identity(
        prompt=prompt,
        system=system,
        schema=schema,
        role=role,
        backend=backend,
        settings=settings,
        watch=watch,
        resume=resume,
        sandboxed=sandboxed,
        image_id=image_id,
        codex_version=effective_codex_version,
    )

    # ── Resume from cache (idempotent) ───────────────────────────
    # If we're resuming a prior run, this call_dir may already contain
    # a successful response. Reuse it without making a new LLM call —
    # but only after verifying the saved prompt matches what we'd send
    # now. Mismatch raises (state drift) rather than silently using a
    # stale cached response.
    if call_dir is not None:
        cached = _try_resume_from_cache(
            call_dir, prompt, schema, cache_identity,
        )
        if cached is not None:
            response, session_id, cache_meta = cached
            print(
                f"[resume] cache hit on call_{call_index:03d}_{role}",
                file=sys.stderr,
            )
            if log is not None and call_index is not None:
                log[call_index] = cache_meta
            return response, session_id

    # ── Docker sandbox wiring ────────────────────────────────────
    # When the harness has started a per-problem container and set the
    # sandbox_container contextvar, every call runs inside that
    # container via `docker exec`.
    #
    # All calls in a problem share cwd = /workspace. This matches the
    # SWE-bench per-task workspace pattern and — crucially — keeps
    # `--resume` working: claude stores per-project session rollouts
    # under ~/.claude/projects/<cwd-encoded>/... so changing cwd
    # between calls makes the resumed session un-findable. We tried
    # per-call /workspace/call_NNN_<role>/ subdirs first; the repair
    # loop broke with "No conversation found with session ID". All
    # agent outputs still get captured via the post-run `docker cp
    # /workspace` snapshot, and the per-call artifact dirs on the
    # host already give us "which call wrote which bytes" attribution.
    container_call_cwd: str | None = "/workspace" if sandboxed else None

    schema_file = None
    child_env = None
    codex_provider = None
    web_search: dict = {}
    web_search_env_vars: dict[str, str] = {}
    configured_base_url = None
    effective_base_url = None
    oss_env_base_url = None
    oss_gate: asyncio.Semaphore | None = None
    oss_gate_held = False
    raw_stdout: bytes = b""
    raw_stderr: bytes = b""
    events: list[dict] = []
    provider_meta: dict = {}
    failed_provider_attempts: list[dict] = []
    cmd: list[str] = []
    proc = None
    container_call_marker: str | None = None
    process_attempts = 0
    schema_retries = 0
    retry_artifacts: list[str] = []
    last_attempt_duration_ms = 0
    started_at = _utc_now_iso()
    call_started = time.perf_counter()
    try:
        # Build command
        if backend == "claude":
            cmd = _build_claude_cmd(
                prompt, settings, schema, system, watch, resume,
                sandboxed=sandboxed,
            )
        elif backend == "codex":
            codex_provider = _codex_provider(settings)
            configured_base_url = _configured_codex_base_url(settings)
            provider_env = _provider_env_names(settings)
            # Run-level shell web search: the helper runs inside the
            # sandbox via the model's shell tool, so the command line
            # needs no changes — but a keyed provider's key must exist
            # here (the host process) to be forwarded, and configured
            # search access makes host execution opt-in, exactly like
            # provider credentials.
            web_search = web_search_config(config.get("_web_search"))
            web_search_key_env = web_search.get("api_key_env")
            if web_search_key_env and not os.environ.get(web_search_key_env):
                raise RuntimeError(
                    "web search is configured but environment variable "
                    f"{web_search_key_env} is not set"
                )
            external_provider = (
                bool(settings.get("oss"))
                or codex_provider not in (None, "openai")
                or configured_base_url is not None
                or bool(provider_env)
                or bool(web_search)
            )
            security = config.get("_security") or {}
            if not isinstance(security, dict):
                raise ValueError("_security must be a mapping")
            allow_external_host = security.get(
                "allow_external_provider_host_access", False,
            )
            if not isinstance(allow_external_host, bool):
                raise ValueError(
                    "_security.allow_external_provider_host_access must be a boolean"
                )
            if external_provider and not sandboxed and not allow_external_host:
                raise RuntimeError(
                    "External Codex providers require Docker isolation by "
                    "default. To run this provider on the host, explicitly set "
                    "_security.allow_external_provider_host_access: true."
                )
            if configured_base_url is not None:
                effective_base_url = _route_oss_base_url(
                    configured_base_url, sandboxed=sandboxed,
                )
            if settings.get("oss"):
                env_base_url = os.environ.get("CODEX_OSS_BASE_URL")
                oss_base_url = env_base_url or settings.get("oss_base_url")
                if oss_base_url is not None:
                    oss_env_base_url = _route_oss_base_url(
                        oss_base_url, sandboxed=sandboxed,
                    )
                    effective_base_url = oss_env_base_url
                child_env = os.environ.copy()
                if oss_env_base_url is not None:
                    child_env["CODEX_OSS_BASE_URL"] = oss_env_base_url
            # Tell the sandbox helper which provider to use. A
            # self-hosted endpoint rides its own env var, loopback-routed
            # the same way as an OSS model endpoint. Docker mode injects
            # these on the exec below; host mode inherits child_env.
            if web_search:
                web_search_env_vars = {
                    _WEB_SEARCH_PROVIDER_ENV: web_search["provider"],
                }
                if web_search.get("api_key_env"):
                    # The helper needs the configured *name* so custom
                    # references such as SEARCH_KEY work.  This selector is
                    # non-secret; the value itself still travels only through
                    # the provider_env allowlist prepared by the harness.
                    web_search_env_vars[_WEB_SEARCH_API_KEY_NAME_ENV] = (
                        web_search["api_key_env"]
                    )
                if web_search.get("endpoint"):
                    web_search_env_vars[_WEB_SEARCH_ENDPOINT_ENV] = (
                        _route_oss_base_url(
                            web_search["endpoint"], sandboxed=sandboxed,
                        )
                    )
                if not sandboxed:
                    child_env = child_env or os.environ.copy()
                    child_env.update(web_search_env_vars)
            if schema:
                codex_schema = _add_additional_properties(schema)
                schema_json = json.dumps(codex_schema).encode("utf-8")
                if sandboxed:
                    # Host tempfiles aren't visible to the container.
                    # Write the schema into a per-call path inside
                    # /workspace (cwd is /workspace; the filename is
                    # unique per call so parallel judges don't stomp
                    # on each other's schema files).
                    schema_file = (
                        f"/workspace/.call_{call_index:03d}_{role}_schema.json"
                    )
                    write = await asyncio.create_subprocess_exec(
                        "docker", "exec", "-i", container_id,
                        "sh", "-c", f"cat > {schema_file}",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, w_err = await write.communicate(input=schema_json)
                    if write.returncode != 0:
                        raise RuntimeError(
                            f"failed to write output schema into container: "
                            f"{w_err.decode(errors='replace')[:300]}"
                        )
                else:
                    f = tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".json", delete=False,
                    )
                    f.write(schema_json)
                    f.close()
                    schema_file = f.name
            cmd = _build_codex_cmd(
                prompt, settings, schema_file, system, resume,
                sandboxed=sandboxed, role=role,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Wrap with `docker exec -w <per-call-cwd> <container>` when
        # sandboxed. The CLI's own cwd becomes the per-call subdir, so
        # scratch files land there.
        if sandboxed:
            docker_prefix = ["docker", "exec"]
            container_call_marker = uuid.uuid4().hex
            docker_prefix += [
                "--env",
                f"{_CONTAINER_CALL_MARKER_ENV}={container_call_marker}",
            ]
            if oss_env_base_url is not None:
                # URL validation above forbids credentials/query parameters;
                # provider secrets travel separately through provider_env.
                docker_prefix += [
                    "--env", f"CODEX_OSS_BASE_URL={oss_env_base_url}",
                ]
            for env_name, env_value in web_search_env_vars.items():
                # Provider name and (already-validated, credential-free)
                # endpoint URL only — never a secret value.
                docker_prefix += ["--env", f"{env_name}={env_value}"]
            cmd = docker_prefix + [
                "-w", container_call_cwd, container_id,
            ] + cmd

        # ── Save pre-call artifacts ──────────────────────────────
        #
        # Write these BEFORE running the subprocess. That way, if the
        # subprocess hangs or the process is killed, we still know
        # exactly what we asked for.
        if call_dir:
            _write_artifact(
                os.path.join(call_dir, "cmd.json"),
                json.dumps(cmd, indent=2, ensure_ascii=False),
            )
            _write_artifact(
                os.path.join(call_dir, "prompt.txt"), prompt,
            )
            if system:
                _write_artifact(
                    os.path.join(call_dir, "system.txt"), system,
                )

        # Retry loop for watchdog-killed subprocesses. Codex CLI
        # sometimes hangs indefinitely on a specific HTTP connection;
        # the watchdog kills it, and we retry from scratch with a
        # fresh subprocess (and therefore fresh codex/claude session).
        # Up to WATCHDOG_RETRY_MAX watchdog retries. Structured-output
        # retries are separately bounded and opt-in per role.
        watchdog_attempts = 0
        schema_retry_limit = (
            _schema_retry_limit(settings) if schema is not None else 0
        )
        call_label = (f"call_{call_index:03d}_{role}"
                      if call_index is not None else role)
        # Local providers handle one request at a time; hold the gate
        # across the whole retry loop so a call only opens its stream
        # when the provider is actually free (see _oss_gate above).
        # Released in the outer finally.
        if backend == "codex" and settings.get("oss"):
            oss_gate = _oss_gate()
            await oss_gate.acquire()
            oss_gate_held = True
        while True:
            process_attempts += 1
            attempt_started = time.perf_counter()
            retrying = False

            # Run
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=SUBPROCESS_STREAM_LIMIT,
                env=child_env,
            )

            # Hang detection. Sandboxed-only: we monitor per-process
            # CPU ticks via /proc/<pid>/stat inside the container. The
            # marker is a unique-per-call token (the schema file path)
            # so we identify the right subprocess when many judges run
            # in parallel; for calls without a schema (solver) we fall
            # back to comm-name matching, which is fine because such
            # calls don't run in parallel within one container.
            last_event_ref = [time.monotonic()]
            watchdog_killed = [False]
            watchdog_task = None
            if container_id is not None:
                marker = schema_file if schema_file else None
                watchdog_task = asyncio.create_task(_watchdog(
                    proc, container_id, watchdog_killed,
                    backend=backend, marker=marker, label=role,
                ))

            raw_stdout = b""
            raw_stderr = b""
            events = []
            provider_meta = {}
            try:
                if watch:
                    # Stream mode: read line by line, print events
                    if backend == "claude":
                        response, session_id, provider_meta, events, raw_stdout = \
                            await _run_claude_streaming(
                                proc, schema, last_event_ref=last_event_ref,
                            )
                    else:
                        response, session_id, provider_meta, events, raw_stdout = \
                            await _run_codex_streaming(
                                proc, schema, last_event_ref=last_event_ref,
                            )
                else:
                    # Batch mode: collect all output, parse at end
                    stdout_bytes, stderr_bytes = await proc.communicate()
                    raw_stdout = stdout_bytes
                    raw_stderr = stderr_bytes

                    if proc.returncode != 0:
                        raise RuntimeError(
                            _format_failure(backend, proc.returncode,
                                            stderr_bytes, stdout_bytes)
                        )

                    output = stdout_bytes.decode()

                    if backend == "claude":
                        response, session_id, provider_meta, events = \
                            _parse_claude_output(output, schema)
                    else:
                        response, session_id, provider_meta, events = \
                            _parse_codex_output(output, schema)
                if schema is not None:
                    _validate_structured_output(response, schema)
                # Success — exit retry loop.
                break
            except StructuredOutputError as e:
                if e.raw_stdout:
                    raw_stdout = e.raw_stdout
                if e.raw_stderr:
                    raw_stderr = e.raw_stderr
                if e.events:
                    events = e.events
                if e.provider_meta:
                    provider_meta = e.provider_meta
                failed_provider_attempts.append(dict(provider_meta))
                if schema_retries < schema_retry_limit:
                    schema_retries += 1
                    retrying = True
                    print(
                        f"[retry] invalid structured output from {call_label}; "
                        f"retrying attempt {schema_retries + 1}/"
                        f"{schema_retry_limit + 1}: {e}",
                        file=sys.stderr,
                    )
                    continue
                raise StructuredOutputError(
                    f"{backend} failed to return valid structured output "
                    f"after {schema_retries + 1} attempt(s): {e}",
                    raw_stdout=raw_stdout,
                    raw_stderr=raw_stderr,
                    events=events,
                    provider_meta=provider_meta,
                ) from e
            except Exception as e:
                if isinstance(e, ProviderProcessError):
                    raw_stdout = e.raw_stdout
                    raw_stderr = e.raw_stderr
                    events = e.events
                # Was this a watchdog kill (transient hang) and do we
                # have retries left? If so, swallow and retry.
                if (watchdog_killed[0]
                        and watchdog_attempts < WATCHDOG_RETRY_MAX):
                    watchdog_attempts += 1
                    retrying = True
                    print(
                        f"[retry] watchdog killed {call_label}; "
                        f"retrying attempt {watchdog_attempts + 1}/"
                        f"{WATCHDOG_RETRY_MAX + 1}",
                        file=sys.stderr,
                    )
                    continue  # finally runs, then loop iterates
                # Non-watchdog failure or out of retries: propagate.
                raise
            finally:
                # Cancel the watchdog so it doesn't leak past this
                # attempt. cancel() is idempotent.
                if watchdog_task is not None:
                    watchdog_task.cancel()
                    try:
                        await watchdog_task
                    except (asyncio.CancelledError, Exception):
                        pass
                last_attempt_duration_ms = int(round(
                    (time.perf_counter() - attempt_started) * 1000
                ))
                # Preserve failed-attempt evidence separately. The final
                # attempt keeps the stable stdout/stderr artifact names used
                # by existing audit tooling.
                if call_dir:
                    suffix = (
                        f"_attempt_{process_attempts:03d}" if retrying else ""
                    )
                    if raw_stdout:
                        path = _write_artifact(
                            os.path.join(call_dir, f"stdout{suffix}.jsonl"),
                            raw_stdout, gzip_it=True,
                        )
                        if retrying:
                            retry_artifacts.append(path)
                    if raw_stderr:
                        path = _write_artifact(
                            os.path.join(call_dir, f"stderr{suffix}.txt"),
                            raw_stderr, gzip_it=True,
                        )
                        if retrying:
                            retry_artifacts.append(path)

        duration_ms = int(round((time.perf_counter() - call_started) * 1000))
        if failed_provider_attempts:
            provider_meta = _aggregate_provider_attempts(
                [*failed_provider_attempts, provider_meta],
            )

        # ── Save post-parse artifacts ────────────────────────────
        artifact_paths: dict[str, str] = {}
        if call_dir:
            response_text = response if isinstance(response, str) else json.dumps(response, indent=2)
            artifact_paths["artifact_dir"] = call_dir
            artifact_paths["cmd_path"] = os.path.join(call_dir, "cmd.json")
            artifact_paths["prompt_path"] = os.path.join(call_dir, "prompt.txt")
            if system:
                artifact_paths["system_path"] = os.path.join(call_dir, "system.txt")
            if raw_stdout:
                artifact_paths["stdout_path"] = os.path.join(call_dir, "stdout.jsonl.gz")
            if raw_stderr:
                artifact_paths["stderr_path"] = os.path.join(call_dir, "stderr.txt.gz")
            if retry_artifacts:
                artifact_paths["retry_artifact_paths"] = retry_artifacts

            _write_artifact(
                os.path.join(call_dir, "response.txt"), response_text,
            )
            artifact_paths["response_path"] = os.path.join(call_dir, "response.txt")

            if events:
                _write_artifact(
                    os.path.join(call_dir, "events.json"),
                    json.dumps(events, ensure_ascii=False, default=str),
                    gzip_it=True,
                )
                artifact_paths["events_path"] = os.path.join(call_dir, "events.json.gz")

            # Full untruncated tool calls — parallel to the truncated
            # ones in call_meta. The truncated list stays in call_meta
            # for scannability; the full list lives on disk.
            if backend == "claude":
                full_tools = _extract_claude_tool_calls(events, truncate=False)
            else:
                full_tools = _extract_codex_tool_calls(events, truncate=False)
            if full_tools:
                _write_artifact(
                    os.path.join(call_dir, "tool_calls.json"),
                    json.dumps(full_tools, ensure_ascii=False,
                               indent=2, default=str),
                )
                artifact_paths["tool_calls_path"] = os.path.join(call_dir, "tool_calls.json")

        # Build call_meta (truncated previews + artifact paths).
        call_meta = None
        if log is not None:
            response_text = response if isinstance(response, str) else json.dumps(response)
            call_meta = {
                "role": role,
                "backend": backend,
                # Record the effective model actually sent to the CLI,
                # not the configured alias — these differ for codex
                # (config model="opus" gets translated to "gpt-5.5").
                "model": _effective_model(cmd) or settings.get("model"),
                "model_config": settings.get("model"),
                "effort": settings.get("effort"),
                "started_at": started_at,
                "ended_at": _utc_now_iso(),
                "duration_ms": duration_ms,
                "last_attempt_duration_ms": last_attempt_duration_ms,
                "process_attempts": process_attempts,
                "session_id": session_id,
                "resumed": bool(resume),
                "has_schema": schema is not None,
                "schema_retries": schema_retries,
                "returncode": proc.returncode,
                "argv_hash": _cmd_hash(cmd),
                "cache_identity": cache_identity,
                # Which sandbox this call ran in (if any). Enables
                # post-hoc reasoning about the container image version.
                "sandboxed": container_id is not None,
                "container_id": container_id,
                "container_cwd": container_call_cwd,
                "image_id": image_id,
                "codex_version": effective_codex_version,
                "oss": bool(settings.get("oss", False)),
                "model_provider": codex_provider,
                "web_search_provider": web_search.get("provider"),
                "base_url": configured_base_url,
                "effective_base_url": effective_base_url,
                "prompt": _truncate(prompt, 8000),
                "system": _truncate(system or "", 8000),
                "response": _truncate(response_text, 8000),
                **artifact_paths,
                **provider_meta,
            }
            # Fill the slot we reserved at the top.
            log[call_index] = call_meta

        # Also write a per-call meta.json artifact so a single call
        # dir is self-describing without reading the batch result.
        if call_dir and call_meta is not None:
            _write_artifact(
                os.path.join(call_dir, "meta.json"),
                json.dumps(call_meta, indent=2, default=str),
            )

        return response, session_id

    except asyncio.CancelledError:
        await _terminate_cancelled_provider(
            proc,
            container_id=container_id,
            container_marker=container_call_marker,
        )
        raise

    except Exception as e:
        # Failure path: at minimum record enough to reconstruct what
        # happened. Raw stdout/stderr were already saved in the inner
        # finally. Write a failure meta.json so the call dir is
        # self-describing for post-mortem.
        duration_ms = int(round((time.perf_counter() - call_started) * 1000))
        returncode = getattr(proc, "returncode", None)
        failure_provider_meta = _aggregate_provider_attempts(
            failed_provider_attempts,
        ) or provider_meta
        failure_meta = {
            "role": role,
            "backend": backend,
            "model": settings.get("model"),
            "effort": settings.get("effort"),
            "started_at": started_at,
            "ended_at": _utc_now_iso(),
            "duration_ms": duration_ms,
            "last_attempt_duration_ms": last_attempt_duration_ms,
            "process_attempts": process_attempts,
            "schema_retries": schema_retries,
            "returncode": returncode,
            "argv_hash": _cmd_hash(cmd) if cmd else None,
            "cache_identity": cache_identity,
            "sandboxed": sandboxed,
            "container_id": container_id,
            "image_id": image_id,
            "codex_version": effective_codex_version,
            "oss": bool(settings.get("oss", False)),
            "model_provider": codex_provider,
            "web_search_provider": web_search.get("provider"),
            "base_url": configured_base_url,
            "effective_base_url": effective_base_url,
            "retry_artifact_paths": retry_artifacts,
            "error_type": type(e).__name__,
            "error": str(e),
            "failed": True,
            **failure_provider_meta,
        }
        if raw_stdout and call_dir:
            failure_meta["stdout_path"] = os.path.join(
                call_dir, "stdout.jsonl.gz",
            )
        if raw_stderr and call_dir:
            failure_meta["stderr_path"] = os.path.join(
                call_dir, "stderr.txt.gz",
            )
        if log is not None and call_index is not None:
            log[call_index] = failure_meta
        if call_dir:
            try:
                _write_artifact(
                    os.path.join(call_dir, "meta.json"),
                    json.dumps(failure_meta, indent=2, default=str),
                )
            except Exception:
                pass
        raise

    finally:
        if oss_gate_held and oss_gate is not None:
            oss_gate.release()
        # Only unlink host-side schema tempfiles. When sandboxed the
        # schema lives at an in-container path; it's cleaned up with
        # the per-call workspace dir when the container is destroyed.
        if schema_file and not sandboxed:
            try:
                os.unlink(schema_file)
            except OSError:
                pass
