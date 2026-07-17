# Research audit contract

The audit artifacts are designed so every reported experiment row can be
traced to an exact cohort, code/config state, model deployment, call trace,
tool action, repair outcome, and grading decision.

## Run identity

`runs/artifacts/<run_id>/meta.json` records:

- the exact selected problem IDs and hashes of all problem records;
- the pinned HLE repository revision and resolved `datasets` fingerprint;
- the effective merged config, config sources, and full SHA-256 hashes;
- effective per-role backend, model, sampling, context, tool, and turn limits;
- every resolved CLI condition argument plus problem/provider concurrency;
- prompt, schema, endpoint, and config fingerprints;
- Git commit, dirty diff hash/patch, and untracked-file hashes;
- host runtime, CLI versions, requested and actual Docker image digests, and
  container tool versions;
- stable runtime model identities, including local Ollama content digests and
  nonvolatile descriptors from local OpenAI-compatible servers;
- explicit `running`, `completed`, `failed`, or `interrupted` status and counts;
- aggregate calls by role/model, cache/resume use, failures, retries, tokens,
  mechanistic tool evidence, search, cost coverage, and repair metrics when
  available.

Resume is fail-closed. A changed experiment argument, concurrency setting,
config, code state, sandbox image digest, runtime model identity, problem ID
list, problem record hash, missing cohort manifest, or unreadable prior
metadata must start a new run instead of rewriting the original identity. The
`--resume` value itself is the only CLI argument excluded from comparison.
Local runs whose model probe did not resolve cannot be resumed.

## Model identity

Local Ollama models are resolved to their content digest, format,
quantization, parameter size, and context length. Local OpenAI-compatible
servers such as vLLM are queried through `/models`. Provider responses retain
response IDs, returned model names, system fingerprints, service tier, finish
reason, request IDs, and retry records when supplied by the API.

Remote services cannot always reveal their underlying weights or serving
configuration. For official Azure or hosted-vLLM experiments, declare the
following fields in the shared role config when the provider does not return
them:

```yaml
model_source: azure-foundry
model_revision: exact-catalog-model-version
deployment_version: immutable-deployment-version
dtype: bfloat16
quantization: none
server_hardware: 8x-H100-80GB
```

These fields are retained in the effective settings and config identity hash.

## Call evidence

Every successful or failed call has a `call_NNN_<role>/` directory containing
the exact prompt, system prompt, response schema, command, response, provider
events, tool inputs/outputs, retry-attempt stdout/stderr, transcript for the
provider-neutral loop, traceback on failure, and self-describing `meta.json`.
Failed calls remain in aggregate metrics. Retry usage and tool calls are
counted across attempts rather than only from the final successful attempt.
Token accounting distinguishes calls with any observed usage from calls with
complete usage. A timeout can retain tokens from completed turns while leaving
the timed-out provider request unreported; in that case `usage_observed` is
true and `usage_complete` is false.

Tool counts are evidence, not proof of correctness. The audit distinguishes
successful, failed, and unknown-status tool calls by role. Search-provider
latency, result counts, and error categories are populated when the configured
search backend exposes them. A search snippet is not equivalent to a fetched
primary source, and analyses must not label it as one.

## Grading evidence

`theoria grade` creates a separate audit run linked to the source run by
SHA-256. It retains the exact grader prompt/schema/config, pinned rationale
dataset and fingerprint, every grader prompt/response, failures/retries, and
aggregate usage. Automatic grades are explicitly marked non-authoritative;
manual adjudication remains a separate research artifact and must be joined by
problem ID for headline paper numbers.

## Integrity and release checklist

Each completed pipeline or grading run writes `artifact_manifest.json` with
the size and SHA-256 of every retained artifact and external result file.

Before using a run in a paper table:

1. Require `meta.json.status == "completed"` and matching requested/completed
   counts.
2. Prefer a clean Git state; otherwise archive the retained patch and
   untracked files identified by hash.
3. Verify immutable model provenance, especially for remote deployments.
4. Freeze prompt/config hashes before evaluation and tune only on a dev split.
   Use `--experiment-phase dev` for calibration and
   `--experiment-phase evaluation --experiment-id <condition> --trial <n>`
   for frozen runs.
5. Keep failed problems and failed/retried calls in denominators.
6. Report token usage only when `usage_complete_fraction == 1`, and cost only
   with complete cost coverage; partial token/cost observations must not be
   presented as totals.
7. Archive the automatic grader trace and any blinded human adjudication used
   to compute certified precision.
