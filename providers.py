"""Normalized Codex provider configuration.

This module is intentionally independent of the call layer, harness, and CLI.
Those consumers all resolve the same immutable provider/role plan instead of
re-parsing ``oss`` and raw ``codex_config`` fields independently.

The structured public provider mapping currently has one first-class kind::

    provider:
      kind: azure_openai
      endpoint: https://RESOURCE.openai.azure.com/openai/v1
      api_key_env: AZURE_OPENAI_API_KEY
      max_parallel: 4

Legacy PR #3 OSS and raw ``codex_config`` declarations continue to compile to
the same representation.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


CODEX_LOCAL_PROVIDERS = {"ollama", "lmstudio"}
CLAUDE_MODEL_ALIASES = {"opus", "sonnet", "haiku"}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CODEX_CONFIG_KEY = re.compile(
    r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$"
)
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
AZURE_HOST_SUFFIXES = (".openai.azure.com", ".services.ai.azure.com")
CODEX_AZURE_URL_MARKERS = (
    "openai.azure.",
    "cognitiveservices.azure.",
    "aoai.azure.",
    "azure-api.",
    "azurefd.",
    "windows.net/openai",
)
AZURE_PROVIDER_FIELDS = {"kind", "endpoint", "api_key_env", "max_parallel"}
SENSITIVE_CODEX_CONFIG_PARTS = {
    "api_key", "authorization", "bearer_token", "cookie", "credential",
    "key", "password", "secret", "token",
}


@dataclass(frozen=True)
class ProviderCapabilities:
    native_web_search: bool
    namespace_tools: bool
    unified_exec: bool
    structured_outputs: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "native_web_search": self.native_web_search,
            "namespace_tools": self.namespace_tools,
            "unified_exec": self.unified_exec,
            "structured_outputs": self.structured_outputs,
        }


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    kind: str
    display_name: str
    base_url: str | None
    wire_api: str
    auth_kind: str
    credential_env: str | None
    forwarded_env: tuple[str, ...]
    external: bool
    local_adapter: str | None
    requires_explicit_model: bool
    capabilities: ProviderCapabilities
    native_search_default: bool
    max_parallel: int | None
    codex_config_items: tuple[tuple[str, object], ...]
    structured: bool = False

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.display_name,
            "endpoint": self.base_url,
            "wire_api": self.wire_api,
            "auth_kind": self.auth_kind,
            "credential_env": self.credential_env,
            "external": self.external,
            "local_adapter": self.local_adapter,
            "capabilities": self.capabilities.as_dict(),
            "native_search_default": self.native_search_default,
            "max_parallel": self.max_parallel,
        }


@dataclass(frozen=True)
class CodexRolePlan:
    provider: ProviderSpec
    deployment: str
    effort: str | None
    native_search: bool
    codex_config_items: tuple[tuple[str, object], ...]
    forwarded_env: tuple[str, ...]
    concurrency_key: str | None

    def cache_identity(self) -> dict:
        return {
            "provider": self.provider.metadata(),
            "deployment": self.deployment,
            "native_search": self.native_search,
            "codex_config": list(self.codex_config_items),
        }

    def metadata(self) -> dict:
        value = self.provider.metadata()
        value.update({
            "deployment": self.deployment,
            "native_search": self.native_search,
            "concurrency_key": self.concurrency_key,
        })
        return value


@dataclass(frozen=True)
class ProviderProbeResult:
    ok: bool
    category: str
    status: int | None = None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep an Azure bearer token on its already-validated origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_azure_endpoint(
    spec: ProviderSpec,
    *,
    environ: Mapping[str, str] | None = None,
    timeout: float = 5,
    opener=None,
) -> ProviderProbeResult:
    """Authenticate Azure's model-list endpoint without billing inference.

    The key is read into an HTTP header in memory. It is never returned in the
    result, embedded in the target URL, or placed in subprocess argv.

    This mirrors pinned Codex 0.133's ``env_key`` bearer authentication. It
    proves endpoint/auth reachability, not that an arbitrary deployment alias
    is present or Responses-compatible.
    """
    if spec.kind != "azure_openai" or not spec.base_url:
        raise ValueError("authenticated endpoint probe requires Azure OpenAI")
    environ = os.environ if environ is None else environ
    env_name = spec.credential_env
    key = environ.get(env_name or "")
    if not key or not key.strip():
        return ProviderProbeResult(False, "missing_credential")
    if key != key.strip():
        return ProviderProbeResult(False, "invalid_credential")
    target = spec.base_url.rstrip("/") + "/models"
    open_request = (
        urllib.request.build_opener(_NoRedirectHandler()).open
        if opener is None else opener
    )
    try:
        key.encode("latin-1")
        if any(
            ord(character) < 32 or ord(character) == 127
            for character in key
        ):
            return ProviderProbeResult(False, "invalid_credential")
        request = urllib.request.Request(
            target, headers={"Authorization": f"Bearer {key}"}, method="GET",
        )
        with open_request(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return ProviderProbeResult(status == 200, "ok" if status == 200 else "http_error", status)
    except ValueError:
        return ProviderProbeResult(False, "invalid_credential")
    except urllib.error.HTTPError as exc:
        category = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            429: "rate_limited",
        }.get(exc.code, "http_error")
        return ProviderProbeResult(False, category, exc.code)
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            category = "timeout"
        elif isinstance(reason, ssl.SSLError):
            category = "tls"
        elif isinstance(reason, socket.gaierror):
            category = "dns"
        else:
            category = "network"
        return ProviderProbeResult(False, category)
    except (socket.timeout, TimeoutError):
        return ProviderProbeResult(False, "timeout")
    except ssl.SSLError:
        return ProviderProbeResult(False, "tls")
    except OSError:
        return ProviderProbeResult(False, "network")


def toml_scalar(value) -> str:
    """Serialize a scalar for Codex's TOML-parsed ``-c key=value`` flag."""
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


def is_sensitive_codex_config_key(key: str) -> bool:
    lower = key.lower()
    if lower.endswith((
        ".env_key", ".env_var", ".env_vars", ".bearer_token_env_var",
    )):
        return False
    parts = set(re.split(r"[._-]", lower))
    return bool(
        parts & SENSITIVE_CODEX_CONFIG_PARTS
        or {"api", "key"}.issubset(parts)
    )


def normalize_endpoint(value, *, sandboxed: bool, label: str = "provider endpoint") -> str:
    """Validate an absolute credential-free endpoint and route loopback."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty http(s) URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{label} cannot contain credentials, a query, or a fragment; "
            "pass credentials through an environment-variable reference"
        )

    hostname = parsed.hostname
    if sandboxed and hostname.lower() in LOOPBACK_HOSTS:
        hostname = "host.docker.internal"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def normalize_azure_endpoint(value, *, sandboxed: bool = False) -> str:
    endpoint = normalize_endpoint(
        value, sandboxed=sandboxed, label="Azure OpenAI endpoint",
    )
    parsed = urlsplit(endpoint)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ValueError("Azure OpenAI endpoint must use https")
    if not host.endswith(AZURE_HOST_SUFFIXES):
        allowed = " or ".join(f"*{suffix}" for suffix in AZURE_HOST_SUFFIXES)
        raise ValueError(
            "Azure OpenAI endpoint must use a supported Azure hostname "
            f"({allowed})"
        )
    if parsed.path.rstrip("/") != "/openai/v1":
        raise ValueError(
            "Azure OpenAI endpoint must end with /openai/v1; legacy "
            "/deployments URLs and dated api-version query parameters are "
            "not supported"
        )
    return endpoint.rstrip("/")


def codex_config_items(settings: Mapping) -> list[tuple[str, object]]:
    """Validate the legacy/advanced Codex configuration escape hatch."""
    raw = settings.get("codex_config")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("codex_config must be a mapping of dotted keys to scalars")

    items: list[tuple[str, object]] = []
    for key, value in raw.items():
        if not isinstance(key, str) or not CODEX_CONFIG_KEY.fullmatch(key):
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
        if is_sensitive_codex_config_key(key):
            raise ValueError(
                f"codex_config.{key} would expose a secret in process argv; "
                "configure the provider's env_key and list the variable in "
                "provider_env instead"
            )
        toml_scalar(value)
        if key.endswith("base_url"):
            normalize_endpoint(value, sandboxed=False)
        if key == "model_provider" and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError("codex_config.model_provider must be a non-empty string")
        items.append((key, value))
    return items


def codex_config_override(key: str, value, *, sandboxed: bool) -> str:
    if key.startswith("model_providers.") and key.endswith(".base_url"):
        value = normalize_endpoint(value, sandboxed=sandboxed)
    return f"{key}={toml_scalar(value)}"


def explicit_provider_env_names(settings: Mapping) -> list[str]:
    value = settings.get("provider_env", [])
    if isinstance(value, str):
        value = [value]
    if value is None:
        value = []
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) or not ENV_NAME.fullmatch(name)
        for name in value
    ):
        raise ValueError(
            "provider_env must contain only environment-variable names"
        )
    return list(dict.fromkeys(value))


def _credential_refs(items: list[tuple[str, object]]) -> list[str]:
    return list(dict.fromkeys(
        value
        for key, value in items
        if isinstance(value, str) and key.lower().endswith((
            ".env_key", ".env_var", ".bearer_token_env_var",
        ))
    ))


def _positive_parallel(value, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("provider.max_parallel must be a positive integer")
    return value


def _legacy_oss_max_parallel(environ: Mapping[str, str]) -> int:
    raw = environ.get("THEORIA_OSS_MAX_PARALLEL", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _azure_spec(
    structured: Mapping,
    *,
    sandboxed: bool,
    extra_env: list[str],
) -> ProviderSpec:
    unknown = sorted(str(key) for key in structured if key not in AZURE_PROVIDER_FIELDS)
    if unknown:
        raise ValueError(
            "provider has unsupported field(s): " + ", ".join(unknown)
        )
    endpoint = normalize_azure_endpoint(
        structured.get("endpoint"), sandboxed=sandboxed,
    )
    api_key_env = structured.get("api_key_env")
    if not isinstance(api_key_env, str) or not ENV_NAME.fullmatch(api_key_env):
        raise ValueError(
            "provider.api_key_env must be an environment-variable name"
        )
    max_parallel = _positive_parallel(
        structured.get("max_parallel"), default=4,
    )
    forwarded = tuple(dict.fromkeys([api_key_env, *extra_env]))
    items = (
        ("model_provider", "azure"),
        ("model_providers.azure.name", "Azure"),
        ("model_providers.azure.base_url", endpoint),
        ("model_providers.azure.env_key", api_key_env),
        ("model_providers.azure.wire_api", "responses"),
    )
    return ProviderSpec(
        id="azure",
        kind="azure_openai",
        display_name="Azure",
        base_url=endpoint,
        wire_api="responses",
        auth_kind="api_key",
        credential_env=api_key_env,
        forwarded_env=forwarded,
        external=True,
        local_adapter=None,
        requires_explicit_model=True,
        capabilities=ProviderCapabilities(True, False, True, True),
        native_search_default=False,
        max_parallel=max_parallel,
        codex_config_items=items,
        structured=True,
    )


def resolve_provider_spec(
    settings: Mapping,
    *,
    sandboxed: bool = False,
    environ: Mapping[str, str] | None = None,
) -> ProviderSpec:
    """Resolve structured or legacy role settings into one provider spec."""
    environ = os.environ if environ is None else environ
    explicit_env = explicit_provider_env_names(settings)
    structured = settings.get("provider")
    legacy_items = codex_config_items(settings)

    if structured is not None:
        if not isinstance(structured, dict):
            raise ValueError("provider must be a mapping")
        if settings.get("oss") or settings.get("local_provider") is not None \
                or settings.get("oss_base_url") is not None:
            raise ValueError(
                "structured provider cannot be combined with oss, "
                "local_provider, or oss_base_url"
            )
        conflicting = [
            key for key, _ in legacy_items
            if key == "model_provider" or key.startswith("model_providers.")
        ]
        if conflicting:
            raise ValueError(
                "structured provider cannot be combined with raw Codex "
                "provider selection: " + ", ".join(sorted(conflicting))
            )
        kind = structured.get("kind")
        if kind != "azure_openai":
            raise ValueError("provider.kind must be azure_openai")
        spec = _azure_spec(
            structured, sandboxed=sandboxed, extra_env=explicit_env,
        )
        advanced_refs = _credential_refs(legacy_items)
        missing_refs = sorted(set(advanced_refs) - set(spec.forwarded_env))
        if missing_refs:
            raise ValueError(
                "advanced Codex credential environment references must also "
                "appear in provider_env: " + ", ".join(missing_refs)
            )
        # Advanced non-provider overrides remain available alongside the
        # generated, audited provider block.
        return ProviderSpec(
            **{
                **spec.__dict__,
                "codex_config_items": (
                    *spec.codex_config_items, *legacy_items,
                ),
            }
        )

    oss = settings.get("oss", False)
    if not isinstance(oss, bool):
        raise ValueError("oss must be a boolean")
    configured_provider = next(
        (value for key, value in legacy_items if key == "model_provider"),
        None,
    )

    if oss:
        adapter = settings.get("local_provider")
        if adapter not in CODEX_LOCAL_PROVIDERS:
            allowed = ", ".join(sorted(CODEX_LOCAL_PROVIDERS))
            raise ValueError(
                f"OSS Codex roles require local_provider to be one of: {allowed}"
            )
        if configured_provider not in (None, adapter):
            raise ValueError(
                "codex_config.model_provider conflicts with local_provider"
            )
        base = environ.get("CODEX_OSS_BASE_URL") or settings.get("oss_base_url")
        if base is not None:
            base = normalize_endpoint(base, sandboxed=sandboxed)
        refs = _credential_refs(legacy_items)
        missing = sorted(set(refs) - set(explicit_env))
        if missing:
            raise ValueError(
                "Codex provider credential environment references must also "
                "appear in provider_env: " + ", ".join(missing)
            )
        return ProviderSpec(
            id=str(adapter),
            kind="local",
            display_name=str(adapter),
            base_url=base,
            wire_api="responses",
            auth_kind="environment" if refs else "none",
            credential_env=refs[0] if len(refs) == 1 else None,
            forwarded_env=tuple(explicit_env),
            external=True,
            local_adapter=str(adapter),
            requires_explicit_model=True,
            capabilities=ProviderCapabilities(False, False, False, True),
            native_search_default=False,
            max_parallel=_legacy_oss_max_parallel(environ),
            codex_config_items=tuple(legacy_items),
        )

    if settings.get("local_provider") is not None:
        raise ValueError("local_provider requires oss: true")
    if settings.get("oss_base_url") is not None:
        raise ValueError("oss_base_url requires oss: true")

    provider_id = str(configured_provider or "openai")
    base_key = f"model_providers.{provider_id}.base_url"
    base = next((value for key, value in legacy_items if key == base_key), None)
    if base is not None:
        base = normalize_endpoint(base, sandboxed=sandboxed)
    refs = _credential_refs(legacy_items)
    selected_env_key = next((
        value for key, value in legacy_items
        if key == f"model_providers.{provider_id}.env_key"
        and isinstance(value, str)
    ), None)
    wire_key = f"model_providers.{provider_id}.wire_api"
    selected_wire_api = next((
        value for key, value in legacy_items if key == wire_key
    ), None)
    name_key = f"model_providers.{provider_id}.name"
    selected_name = next((
        value for key, value in legacy_items if key == name_key
    ), None)
    missing = sorted(set(refs) - set(explicit_env))
    if missing:
        raise ValueError(
            "Codex provider credential environment references must also "
            "appear in provider_env: " + ", ".join(missing)
        )
    external = bool(
        provider_id != "openai" or base is not None or explicit_env
    )
    base_lower = base.lower() if isinstance(base, str) else ""
    azure = bool(
        provider_id.lower() == "azure"
        or (
            isinstance(selected_name, str)
            and selected_name.strip().lower() == "azure"
        )
        or (
            isinstance(base, str)
            and (urlsplit(base).hostname or "").lower().endswith(
                AZURE_HOST_SUFFIXES
            )
        )
        or any(marker in base_lower for marker in CODEX_AZURE_URL_MARKERS)
    )
    if azure:
        if base is None:
            raise ValueError("Azure OpenAI provider requires a base_url")
        base = normalize_azure_endpoint(base, sandboxed=sandboxed)
    if azure and not selected_env_key:
        raise ValueError(
            "Azure OpenAI provider requires "
            f"model_providers.{provider_id}.env_key"
        )
    if azure:
        # Codex 0.133 only recognizes arbitrary Azure hostname families by
        # exact provider name. Canonicalize legacy raw declarations too, so
        # services.ai.azure.com receives Azure's required `store: true` path.
        legacy_items = [
            (key, "Azure" if key == name_key else value)
            for key, value in legacy_items
        ]
        if not any(key == name_key for key, _ in legacy_items):
            legacy_items.append((name_key, "Azure"))
        legacy_items = [
            (key, "responses" if key == wire_key else value)
            for key, value in legacy_items
        ]
        if not any(key == wire_key for key, _ in legacy_items):
            legacy_items.append((wire_key, "responses"))
        selected_wire_api = "responses"

    capabilities = (
        ProviderCapabilities(True, False, True, True)
        if azure else ProviderCapabilities(True, True, True, True)
    )
    return ProviderSpec(
        id=provider_id,
        kind="azure_openai" if azure else ("custom" if external else "openai"),
        display_name="Azure" if azure else provider_id,
        base_url=base,
        wire_api=str(selected_wire_api or "responses"),
        auth_kind="api_key" if refs else ("environment" if explicit_env else "codex"),
        credential_env=selected_env_key if azure else (
            refs[0] if len(refs) == 1 else None
        ),
        forwarded_env=tuple(explicit_env),
        external=external,
        local_adapter=None,
        requires_explicit_model=external,
        capabilities=capabilities,
        # Preserve PR #3 custom-provider behavior. Azure is the one hosted
        # provider whose native search is deliberately opt-in because it can
        # cross the configured Azure data/compliance boundary.
        native_search_default=False if azure else True,
        max_parallel=4 if azure else None,
        codex_config_items=tuple(legacy_items),
    )


def resolve_codex_role(
    settings: Mapping,
    *,
    sandboxed: bool = False,
    environ: Mapping[str, str] | None = None,
) -> CodexRolePlan:
    spec = resolve_provider_spec(
        settings, sandboxed=sandboxed, environ=environ,
    )
    model = settings.get("model")
    if spec.requires_explicit_model and (
        not isinstance(model, str) or not model.strip()
    ):
        raise ValueError("External Codex provider roles require an explicit model")
    if model is None:
        model = "gpt-5.5"
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Codex model must be a non-empty string")
    if spec.external and model in CLAUDE_MODEL_ALIASES:
        raise ValueError(
            f"Codex model {model!r} is a Claude alias; configure the explicit "
            "provider model or deployment id"
        )
    if not spec.external and model in CLAUDE_MODEL_ALIASES:
        model = "gpt-5.5"

    search = settings.get("search", spec.native_search_default)
    if not isinstance(search, bool):
        raise ValueError("search must be a boolean")
    if search and not spec.capabilities.native_web_search:
        raise ValueError(
            "search: true enables Codex's native web-search tool, which the "
            f"{spec.display_name} provider does not support. Keep search: "
            "false and stack a shell web-search profile instead"
        )

    effort = settings.get("effort", "xhigh")
    configured_keys = {key for key, _ in spec.codex_config_items}
    if spec.kind == "azure_openai":
        unsafe_namespace_overrides = [
            key for key, value in spec.codex_config_items
            if key in {
                "features.multi_agent", "features.multi_agent_v2",
            } and value is not False
        ]
        if unsafe_namespace_overrides:
            raise ValueError(
                "Azure OpenAI does not support Codex namespace tools; "
                "remove or set false: "
                + ", ".join(sorted(unsafe_namespace_overrides))
            )
    config_items = list(spec.codex_config_items)
    if not spec.capabilities.namespace_tools:
        for feature in ("multi_agent", "multi_agent_v2"):
            key = f"features.{feature}"
            if key not in configured_keys:
                config_items.append((key, False))
    if not spec.capabilities.unified_exec:
        key = "features.unified_exec"
        if key not in configured_keys:
            config_items.append((key, False))

    concurrency_key = None
    if spec.max_parallel is not None:
        raw = json.dumps(
            {
                "kind": spec.kind,
                # Azure's scheduler identity is endpoint + deployment. Raw
                # configs may choose an arbitrary provider id (for example
                # `foundry`) while compiling to the same Azure transport.
                "provider": None if spec.kind == "azure_openai" else spec.id,
                "endpoint": spec.base_url,
                # Azure quotas are deployment-scoped. A local server is one
                # shared scheduler even when roles select different models.
                "deployment": model if spec.kind == "azure_openai" else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        concurrency_key = sha256(raw.encode()).hexdigest()[:20]

    return CodexRolePlan(
        provider=spec,
        deployment=model,
        effort=effort,
        native_search=search,
        codex_config_items=tuple(config_items),
        forwarded_env=spec.forwarded_env,
        concurrency_key=concurrency_key,
    )
