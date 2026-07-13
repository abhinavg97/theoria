# Observability and research telemetry

Theoria keeps two different records on purpose:

- **Operational logs** explain what the running process is doing. They are
  written to stderr as human-readable text or one JSON object per line.
- **Research telemetry** is persisted with run artifacts. It records measured
  provider usage, timing, retries, failures, tools, and cost provenance.

## Operational logging

Use CLI flags or their environment-variable equivalents:

```bash
theoria hle 1 --log-format json --log-level INFO
THEORIA_LOG_FORMAT=json THEORIA_LOG_LEVEL=WARNING theoria hle 1
```

JSON logs include UTC timestamps, stable event names, severity, component,
run/problem/call identifiers where available, and event-specific fields.
Known secret-bearing field names are recursively redacted. Prompts, responses,
credentials, and command arguments are never added to structured operational
events. Common inline bearer tokens and key/value credentials are also
redacted from live tool previews and the truncated previews in result JSON.
Full research artifacts remain under the gitignored `runs/artifacts/` tree;
they intentionally preserve exact inputs and outputs and must be treated as
sensitive data.

## Per-call telemetry

Every `call_NNN_<role>/meta.json` and every result's `calls` array contains:

- backend, effective model, configured model, role, effort, and schema use;
- UTC start/end timestamps, wall duration, retry count, return code, and error;
- input, output, cache-read, cache-creation, and reasoning-output tokens;
- the provider's unmodified usage object for forward compatibility;
- tool calls, session/resume state, sandbox image, and correlation identifiers;
- cost amount, currency, source, pricing version, and completeness.

The normalized schema is versioned with `telemetry_schema_version`. Provider
usage remains the source of truth; a missing dimension is recorded as zero only
when the CLI did not report that dimension.

Problem-level metrics are stored in the result JSON. Run-level totals are
written to `runs/artifacts/<run_id>/telemetry.json` with `run_started_at`,
`run_ended_at`, and `run_duration_ms`. `theoria grade` uses the same call
schema and writes a separate `grade_<source-run>_<timestamp>` artifact root.

## Cost semantics

Cost is not inferred by default. Claude may report a call cost; Codex
subscription calls currently report token usage but no monetary cost. A
partially known total is dangerous in a paper, so `total_cost_usd` remains
`null` unless every call has complete pricing. The nested `cost` object always
reports coverage and source counts.

To calculate a reproducible API-equivalent estimate, supply a versioned catalog:

```yaml
pricing_id: vendor-prices-2026-07-12
models:
  "codex:gpt-example":
    # Codex/OpenAI input totals commonly include cache-read tokens.
    input_token_semantics: includes_cache_read
    input_per_million_usd: 1.25
    output_per_million_usd: 10.0
    cache_read_per_million_usd: 0.125
    cache_creation_per_million_usd: 1.25
```

```bash
theoria hle 1 --pricing ./pricing.yaml
# or: THEORIA_PRICING_FILE=./pricing.yaml theoria hle 1
```

Use exact effective model names from call metadata. Catalog values are not
bundled because vendor prices change and subscription cost is not equivalent
to token-based API cost. Archive the catalog alongside a paper's analysis and
cite its `pricing_id`. Every model must declare `input_token_semantics` as
`excludes_cache`, `includes_cache_read`, or `includes_all_cache`; this prevents
cached tokens from being billed twice across providers with different usage
semantics.

## Recommended paper reporting

Report the telemetry schema version, run IDs, git SHA, container digest, CLI
versions, model names, pricing ID, cost source, and cost coverage. Distinguish:

1. measured tokens from provider events;
2. provider-reported cost;
3. model-pricing estimates; and
4. fixed subscription spend, which cannot be allocated accurately per call
   without an explicitly documented allocation policy.

Also retain rejected/failed calls and retries. Excluding them understates both
compute and cost.

For HLE runs, pin the source when possible:

```bash
theoria hle 1 --dataset-revision <hugging-face-commit>
```

Every loaded problem records the requested revision and the resolved
`datasets` fingerprint. Run metadata also includes canonical configuration and
per-role prompt SHA-256 hashes, pricing-file SHA-256, git state, tool versions,
and the sandbox image digest. Failed calls and problems write traceback files
inside their artifact directories.
