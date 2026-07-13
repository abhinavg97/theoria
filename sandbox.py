"""Docker sandbox lifecycle for the Theoria pipeline.

One container per problem, reused across all LLM calls for that problem
via `docker exec`. Matches the dominant benchmark pattern (SWE-bench,
Terminal-Bench, AgentBench all use per-task containers). Containers are
destroyed at problem end.

Authentication
--------------
Claude subscription auth lives in the macOS Keychain as "Claude Code-
credentials". The container can't reach the Keychain, so we extract the
OAuth JSON to a private temp file, then stream it into the container's
tmpfs-backed home as the non-root `node` user.

Codex state is already on disk at ~/.codex. Each problem receives a
writable copy of the small subset of state files the CLI needs. OSS
providers can use the same writable state directory without copying
cloud authentication into the container.

Hardening
---------
Container runs with --cap-drop=ALL, --security-opt=no-new-privileges,
--pids-limit, memory/cpu caps, and a private ephemeral home. This is reasonable
for research-grade isolation without going to gVisor.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_IMAGE = "theoria-sandbox:latest"

# Minimum set of files codex needs in its state dir. Everything else
# (sessions/, archived_sessions/, history.jsonl, cache/) is runtime-
# generated and unnecessary for a fresh call.
_CODEX_STATE_FILES = (
    "auth.json",
    "config.toml",
    "installation_id",
    "internal_storage.json",
    ".codex-global-state.json",
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def refresh_claude_credentials() -> str | None:
    """Extract the Claude OAuth token from the macOS Keychain to a
    private temp file. Returns the path or None if we can't get it (not on
    macOS, not logged in, security CLI missing)."""
    try:
        r = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None

    fd, path = tempfile.mkstemp(prefix="theoria-claude-creds-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(r.stdout)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def prepare_claude_config(retries: int = 5, retry_delay: float = 0.2) -> str | None:
    """Snapshot ~/.claude.json for private transfer into the container
    without racing the host.

    The host may be actively running Claude Code (including the one
    running this pipeline), which writes ~/.claude.json periodically
    via atomic rename. A concurrent read while the file is mid-rewrite can
    yield corrupted-looking JSON (we've observed "Unterminated string"
    errors at this layer). Taking a snapshot, validating, and retrying avoids
    that race entirely.

    Returns the private tempfile path or None if we can't get a
    readable + parseable snapshot after `retries` attempts.
    """
    src = Path(os.path.expanduser("~/.claude.json"))
    if not src.exists():
        return None
    fd, path = tempfile.mkstemp(prefix="theoria-claude-config-", suffix=".json")
    os.close(fd)
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            snapshot = src.read_bytes()
            with open(path, "wb") as f:
                f.write(snapshot)
            os.chmod(path, 0o600)
            with open(path) as f:
                json.load(f)
            return path
        except (json.JSONDecodeError, OSError) as e:
            last_err = e
            time.sleep(retry_delay)
    # Give up; clean up the tempfile so we don't leak.
    try:
        os.unlink(path)
    except OSError:
        pass
    if last_err:
        print(f"WARNING: could not snapshot ~/.claude.json: {last_err}")
    return None


def cleanup_claude_config(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def prepare_codex_state_dir(
    *,
    include_auth: bool = True,
    copy_host_state: bool = True,
    include_config: bool = False,
) -> str:
    """Create a writable, isolated Codex state directory for one problem.

    When host state exists, copy only the minimum files Codex needs. Fully
    OSS/custom-provider runs set ``copy_host_state=False`` so unrelated MCP
    settings, headers, and credentials in the user's config never enter the
    agent container. When ~/.codex is absent, return an empty writable
    directory for later transfer into the container's private tmpfs home.

    Codex writes to ~/.codex during normal operation (trusted-project
    state in config.toml, new session rollouts, etc.). Mounting the
    host ~/.codex :ro causes codex exec to fail. Mounting it :rw
    leaks per-problem state back to the host. This gives each problem
    a fresh, isolated writable copy that's destroyed at cleanup time.

    Returns the newly-created tempdir path.
    """
    src = Path(os.path.expanduser("~/.codex"))
    dst = tempfile.mkdtemp(prefix="theoria-codex-state-")
    if copy_host_state and src.is_dir():
        for name in _CODEX_STATE_FILES:
            if name == "auth.json" and not include_auth:
                continue
            if name == "config.toml" and not include_config:
                continue
            f = src / name
            if f.is_file():
                shutil.copy2(f, Path(dst) / name)
    try:
        os.chmod(dst, 0o700)
        for f in Path(dst).iterdir():
            try:
                os.chmod(f, 0o600)
            except OSError:
                pass
    except OSError:
        pass
    return dst


def validate_provider_env_names(
    names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Validate and deduplicate explicit host environment allowlist names."""
    validated: list[str] = []
    seen: set[str] = set()
    for name in names or ():
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid provider environment variable name: {name!r}")
        if name not in seen:
            validated.append(name)
            seen.add(name)
    return validated


def redact_container_inspect(
    inspect_data: dict,
    *,
    provider_env_names: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Return a copy of ``docker inspect`` with credential env values redacted.

    Docker expands ``-e NAME`` before storing the container config, so the
    otherwise-useful inspection artifact would contain provider secrets. We
    retain environment variable names for reproducibility and replace values
    for every explicitly-forwarded provider variable and common secret names.
    """
    redacted = deepcopy(inspect_data)
    explicit = set(validate_provider_env_names(provider_env_names))
    config = redacted.get("Config")
    if not isinstance(config, dict):
        return redacted
    env = config.get("Env")
    if not isinstance(env, list):
        return redacted

    safe_env: list = []
    for entry in env:
        if not isinstance(entry, str) or "=" not in entry:
            safe_env.append(entry)
            continue
        name, value = entry.split("=", 1)
        upper_name = name.upper()
        is_sensitive = name in explicit or any(
            marker in upper_name for marker in _SENSITIVE_ENV_MARKERS
        )
        safe_env.append(
            f"{name}=<redacted>" if is_sensitive else f"{name}={value}"
        )
    config["Env"] = safe_env
    return redacted


def cleanup_credentials(path: str | None) -> None:
    """Remove a single credential file (extracted claude creds)."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def cleanup_codex_state_dir(path: str | None) -> None:
    """Remove a per-problem codex state tempdir."""
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def image_digest(image: str = DEFAULT_IMAGE) -> str | None:
    """Return the local image digest (sha256:...) so call metadata can
    record which build was actually run. None if the image isn't local.

    Uses `docker images -q` rather than `docker image inspect` because
    the latter fails on containerd-backed Docker daemons for locally-
    built tags that weren't pushed to a registry.
    """
    try:
        r = subprocess.run(
            ["docker", "images", "-q", "--no-trunc", image],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError):
        return None
    if r.returncode != 0:
        return None
    digest = r.stdout.strip()
    return digest or None


def _write_container_file(container_id: str, source: str, destination: str) -> None:
    """Copy one private host snapshot through stdin as the container user."""
    data = Path(source).read_bytes()
    parent = str(Path(destination).parent)
    command = (
        "umask 077; "
        f"mkdir -p {shlex.quote(parent)}; "
        f"cat > {shlex.quote(destination)}"
    )
    result = subprocess.run(
        ["docker", "exec", "-i", container_id, "sh", "-c", command],
        input=data,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to initialize private container state at {destination}: "
            f"{result.stderr.decode(errors='replace')[:300]}"
        )


def _initialize_container_home(
    container_id: str,
    *,
    claude_creds_path: str | None,
    claude_config_path: str | None,
    codex_state_dir: str | None,
) -> None:
    """Populate the container's tmpfs-backed home with private CLI state."""
    result = subprocess.run(
        [
            "docker", "exec", container_id, "sh", "-c",
            "umask 077; mkdir -p /home/node/.claude /home/node/.codex "
            "/home/node/.local",
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to initialize private container home: "
            + result.stderr.decode(errors="replace")[:300]
        )

    if claude_creds_path:
        _write_container_file(
            container_id,
            claude_creds_path,
            "/home/node/.claude/.credentials.json",
        )
    if claude_config_path:
        _write_container_file(
            container_id, claude_config_path, "/home/node/.claude.json",
        )
    if codex_state_dir:
        for source in sorted(Path(codex_state_dir).iterdir()):
            if source.is_file() and not source.is_symlink():
                _write_container_file(
                    container_id, str(source), f"/home/node/.codex/{source.name}",
                )


def start_sandbox(
    pid: str,
    run_id: str,
    *,
    image: str = DEFAULT_IMAGE,
    claude_creds_path: str | None = None,
    claude_config_path: str | None = None,
    codex_state_dir: str | None = None,
    enable_claude: bool = True,
    enable_codex: bool = True,
    provider_env_names: list[str] | tuple[str, ...] | None = None,
    add_host_gateway: bool = False,
    workspace_host_path: str | None = None,
    memory: str = "10g",
    cpus: str = "2",
) -> str:
    """Start a detached sandbox container for one problem. Returns the
    container ID (long form). The container runs `sleep infinity` and is
    intended to be used via `docker exec` until stop_sandbox() kills it.

    Auto-removes on stop thanks to --rm.
    """
    forwarded_env_names = validate_provider_env_names(provider_env_names)

    # Credentials are copied into a private tmpfs home after startup. This
    # avoids world-readable host snapshots and gives both CLIs writable state
    # without persisting it beyond the per-problem container.
    mounts: list[str] = []
    if workspace_host_path:
        os.makedirs(workspace_host_path, exist_ok=True)
        mounts += ["-v", f"{workspace_host_path}:/workspace:rw"]

    hardening = [
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=8192",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        f"--cpus={cpus}",
    ]

    # Passing ``-e NAME`` (without ``=value``) makes Docker copy the value
    # from this process while keeping secrets out of the docker CLI argv.
    # Missing optional variables are not forwarded.
    provider_env_args: list[str] = []
    for name in forwarded_env_names:
        if name in os.environ:
            provider_env_args += ["-e", name]

    network_args: list[str] = []
    if add_host_gateway and sys.platform.startswith("linux"):
        network_args = ["--add-host=host.docker.internal:host-gateway"]

    # Container name helps operators see what's running with `docker ps`.
    # Docker name regex is [a-zA-Z0-9][a-zA-Z0-9_.-]+ so we sanitize the
    # pid in case it has characters Docker rejects.
    safe_pid = "".join(c if c.isalnum() or c in "_.-" else "_" for c in pid)[:64]
    name = f"theoria-{run_id}-{safe_pid}"[:128]

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", name,
        *network_args,
        "--tmpfs", "/home/node:rw,exec,nosuid,size=512m,uid=1000,gid=1000,mode=0700",
        *mounts,
        *hardening,
        *(["-e", "CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000"] if enable_claude else []),
        *provider_env_args,
        "--workdir", "/workspace",
        image,
        "sleep", "infinity",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(
            f"docker run failed (returncode={r.returncode}): "
            f"stderr={r.stderr.strip()!r}"
        )
    container_id = r.stdout.strip()
    if not container_id:
        raise RuntimeError("docker run returned empty container id")
    try:
        _initialize_container_home(
            container_id,
            claude_creds_path=claude_creds_path if enable_claude else None,
            claude_config_path=claude_config_path if enable_claude else None,
            codex_state_dir=codex_state_dir if enable_codex else None,
        )
    except Exception:
        stop_sandbox(container_id)
        raise
    return container_id


def stop_sandbox(container_id: str) -> None:
    """Stop the sandbox (auto-removes thanks to --rm). Idempotent."""
    try:
        subprocess.run(
            ["docker", "stop", "-t", "2", container_id],
            capture_output=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def exec_prefix(container_id: str) -> list[str]:
    """The argv prefix that turns a CLI invocation into a `docker exec`
    in the named container. Caller appends the actual CLI args."""
    return ["docker", "exec", container_id]


def container_tool_versions(image: str = DEFAULT_IMAGE) -> dict:
    """Run a throwaway container to capture the versions of the tools
    actually installed inside the image. These may drift from the host
    versions; the paper methodology should cite these, not the host.

    Returns a dict with one string per tool (or None on failure).
    """
    return {
        "claude": command_in_image(image, ["claude", "--version"]),
        "codex": command_in_image(image, ["codex", "--version"]),
        "python3": command_in_image(image, ["python3", "--version"]),
        "pari_gp": command_in_image(image, ["gp", "--version-short"]),
        "node": command_in_image(image, ["node", "--version"]),
        "theoria_search": command_in_image(
            image, ["theoria-search", "--version"],
        ),
    }


def command_in_image(
    image: str, argv: list[str], *, extra_docker_args: list[str] | None = None,
) -> str | None:
    """Run one non-interactive command in an image and return its output."""
    if not argv:
        raise ValueError("argv must not be empty")
    try:
        r = subprocess.run(
            [
                "docker", "run", "--rm", *(extra_docker_args or []),
                "--entrypoint", argv[0], image, *argv[1:],
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or r.stderr).strip()


def codex_oss_capabilities_in_image(image: str = DEFAULT_IMAGE) -> dict:
    """Inspect the selected image's Codex OSS command-line capabilities."""
    output = command_in_image(image, ["codex", "exec", "--help"])
    return {
        "available": output is not None,
        "oss": bool(output and "--oss" in output),
        "local_provider": bool(output and "--local-provider" in output),
        "output_schema": bool(output and "--output-schema" in output),
    }


def endpoint_reachable_from_image(
    image: str, endpoint: str, *, add_host_gateway: bool = False,
) -> bool:
    """Probe a provider's /models route from the selected sandbox image."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError):
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return False

    hostname = parsed.hostname
    original_hostname = hostname.lower()
    loopback = original_hostname in {
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
    }
    needs_gateway = loopback or original_hostname == "host.docker.internal"
    if loopback:
        hostname = "host.docker.internal"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    base_url = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    target = base_url.rstrip("/") + "/models"

    docker_args = []
    if add_host_gateway and needs_gateway and sys.platform.startswith("linux"):
        docker_args = ["--add-host=host.docker.internal:host-gateway"]
    output = command_in_image(
        image,
        [
            "curl", "--silent", "--show-error", "--output", "/dev/null",
            "--write-out", "%{http_code}", "--max-time", "5", target,
        ],
        extra_docker_args=docker_args,
    )
    return bool(output and output.isdigit() and output != "000")


def url_reachable_from_image(
    image: str, url: str, *, require_ok: bool = False,
) -> bool:
    """Probe one absolute URL from the selected sandbox image.

    By default any HTTP status counts as reachable (an auth failure
    still proves a listening server); 000 means curl could not connect
    at all. With require_ok=True only a 2xx passes — used for endpoints
    that must actually serve the probe (a SearXNG instance whose json
    format is disabled answers 403 but would fail every agent search).
    The caller routes loopback URLs before probing.
    """
    try:
        parsed = urlsplit(url)
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    docker_args = []
    if (
        parsed.hostname.lower() == "host.docker.internal"
        and sys.platform.startswith("linux")
    ):
        docker_args = ["--add-host=host.docker.internal:host-gateway"]
    output = command_in_image(
        image,
        [
            "curl", "--silent", "--show-error", "--output", "/dev/null",
            "--write-out", "%{http_code}", "--max-time", "10", url,
        ],
        extra_docker_args=docker_args,
    )
    if not output or not output.isdigit() or output == "000":
        return False
    if require_ok:
        return output.startswith("2")
    return True


def pip_freeze_in_container(container_id: str) -> str | None:
    """Capture `pip freeze` inside a running sandbox, returning the
    output as a string. Records the exact Python env state (including
    any packages an agent `pip install`-ed at runtime)."""
    try:
        r = subprocess.run(
            ["docker", "exec", container_id, "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def copy_from_container(container_id: str, src: str, dst: str) -> bool:
    """`docker cp <container>:<src> <dst>`. Returns True on success."""
    try:
        r = subprocess.run(
            ["docker", "cp", f"{container_id}:{src}", dst],
            capture_output=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


def inspect_container(container_id: str) -> dict | None:
    """Run `docker inspect <container>` and return the parsed JSON.

    Captures the authoritative exit state — actual returncode, start/
    stop timestamps, OOMKilled flag, actual resource limits applied,
    restart count, and any docker-daemon-level error. If our pipeline
    sees a mysterious nonzero call returncode, this is where we'd find
    out whether the container died of OOM vs something else.

    Returns None on any failure — docker inspect is purely diagnostic,
    the run should continue without it.
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    # `docker inspect` returns a list. For a single ID we want the
    # first (only) element.
    return data[0] if isinstance(data, list) and data else None


def dpkg_list_in_container(container_id: str) -> str | None:
    """Capture the installed apt packages inside the container (the
    system-level equivalent of `pip freeze`). Complements
    pip_freeze_in_container — one run of both tells you the complete
    package state for reproducibility."""
    try:
        r = subprocess.run(
            ["docker", "exec", container_id, "dpkg", "-l"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout
