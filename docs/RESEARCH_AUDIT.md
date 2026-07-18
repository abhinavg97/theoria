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
- explicit `running`, `completed`, `completed_with_errors`, `failed`, or
  `interrupted` status and counts;
- aggregate calls by role/model, cache/resume use, failures, retries, tokens,
  mechanistic tool evidence, search, cost coverage, and repair metrics when
  available.

Resume is fail-closed. For a cleanly finalized run, Theoria verifies the prior
global manifest, every internal artifact, the exact internal file set, and each
external result file. The verified seal is archived before a replacement seal
is written.

Every completed problem also has an atomic `_problem_checkpoint.json` covering
its exact artifact set and `result.json`. If a process is hard-killed while the
run status is still `running`, the next resume may proceed without a final
global manifest, but it reuses only completed problems whose independent
checkpoint verifies. Incomplete problem directories are moved under
`_interrupted_attempts/` for forensic retention and are never read as caches.
A present but corrupted completed-problem checkpoint aborts the resume rather
than silently rerunning or resealing it. This supports SIGKILL, OOM, preemption,
and power-loss recovery without trusting unsealed responses.
Completed results that contain an execution error are not skipped: their valid
prior checkpoint is retained under `_retry_checkpoints/`, and the problem is
rerun so successful earlier calls can be reused while the failed call retries.

A changed experiment argument, concurrency setting, config, code state, Python
executable/version, installed-package fingerprint, CLI version, sandbox image
digest, runtime model identity, problem ID list, problem record hash, missing
cohort manifest, or unreadable prior metadata must start a new run instead of
rewriting the original identity. The `--resume` value itself is the only CLI
argument excluded from comparison. Local runs whose model probe did not resolve
cannot be resumed.

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
deployment_upgrade_policy: NoAutoUpgrade
azure_subscription_id: exact-subscription-id
azure_resource_group: exact-resource-group
azure_account: exact-AI-Services-account
dtype: bfloat16
quantization: none
server_hardware: 8x-H100-80GB
```

These fields are retained in the effective settings and config identity hash.
For `model_source: azure-foundry`, Theoria also queries the Azure control plane
at run start and marks runtime identity incomplete unless the live catalog
model/version, deployment etag, running state, and upgrade policy match the
declarations.

Provider-neutral runs also retain `system_role`, `temperature`, `top_p`,
`top_k`, `min_p`, `seed`, `reasoning_effort`, and chat-template settings when
configured. This matters for reasoning models whose published protocol forbids
a system role or distinguishes thinking from non-thinking mode.

## Call evidence

Every successful or failed call has a `call_NNN_<role>/` directory containing
the exact prompt, system prompt, response schema, command, response, provider
events, tool inputs/outputs, retry-attempt stdout/stderr, transcript for the
provider-neutral loop, traceback on failure, and self-describing `meta.json`.
Failed calls remain in aggregate metrics. A later execution retry is written to
`call_NNN_<role>/retry_NNN/`; it never overwrites the failed invocation. Retry
usage and tool calls are counted across attempts rather than only from the final
successful attempt. Provider-neutral transcripts and Docker-side Claude/Codex
session state are persisted so a new process can continue the same conversation
instead of silently starting without its earlier context.
Token accounting distinguishes calls with any observed usage from calls with
complete usage. A timeout can retain tokens from completed turns while leaving
the timed-out provider request unreported; in that case `usage_observed` is
true and `usage_complete` is false.

Tool counts are evidence, not proof of correctness. The audit distinguishes
successful, failed, and unknown-status tool calls by role. Search-provider
latency, result counts, and error categories are populated when the configured
search backend exposes them. Search attempts and client-side failures are kept
separate from requests that reached the provider and provider-side failures, so
provider reliability must use `web_search_provider_requests` as its denominator.
A search snippet is not equivalent to a fetched primary source, and analyses
must not label it as one.
When a role declares `required_tools`, the provider-neutral loop refuses its
final response until every declared tool has succeeded. Per-call and run-level
artifacts separately record required calls, obligations, satisfied tools, and
compliance fractions. This proves execution of the configured mechanism, not
that the tool result was interpreted correctly.

## Grading evidence

`theoria grade` creates a separate audit run linked to one stable source-run
snapshot by SHA-256. The exact bytes that were parsed are copied into the
grading artifact root; later source-file changes cannot alter the recorded
identity. If source-run metadata exists, its run identity, completion status,
and artifact seal must match the supplied path. Each grading invocation gets a
unique default output rather than overwriting an earlier grade. It retains the
exact grader prompt/schema/config, pinned rationale dataset and fingerprint,
every grader prompt/response, failures/retries, and aggregate usage. Automatic
grades are explicitly marked non-authoritative; manual adjudication remains a
separate research artifact and must be joined by problem ID for headline paper
numbers.
The grading target is explicit in metadata. `final` grades the shipped answer;
`first_attempt` grades the answer selected by the first semantic
formalize-and-verify attempt; and `solver_initial` grades the first retained
solver response from the same sealed run. Final and first-attempt grading use
the identical condition-neutral rubric, and the evaluator receives neither the
target name nor internal verification/repair traces. A missing selected response
fails its grade row by default. Explicit `--missing-as-incorrect`
intent-to-treat policy records a deterministic wrong grade instead and makes no
model call.

Local JSON benchmarks carry the SHA-256 of the complete source file into every
problem record. Repair metrics retain exact and NFKC/case/whitespace-normalized
before/after answer hashes so answer flips can be audited without relying on a
display-only comparison. `repair_attempted` records any current-loop retry;
`semantic_repair_attempted`, post-judge activity, formalizer-invalid retries,
provider retries, and action/schema reprompts remain separate fields.

## Integrity and release checklist

Each completed pipeline or grading run writes `artifact_manifest.json` with
the size and SHA-256 of every retained artifact and external result file.
Evaluation and ablation runs additionally retain `pip freeze`; smoke and dev
runs use the equivalent installed-distribution inventory without spawning pip.
Resume invocations verify the recorded host-runtime identity and reference the
original package inventory instead of running `pip freeze` again.

Before using a run in a paper table:

1. For a pipeline run, require `meta.json.status` to be `completed` or
   `completed_with_errors`, matching requested/completed counts, one sealed
   result row per requested problem, and a valid artifact manifest. Under the
   preregistered intent-to-treat policy, per-problem execution errors in a
   `completed_with_errors` pipeline run are declines and remain in the
   denominator. `failed` or `interrupted` pipeline runs, missing result rows,
   and unsealed artifacts are incomplete and ineligible. Report execution-error
   counts and any missing repair telemetry explicitly; do not silently impute
   repair-process totals. A grading run used for precision must have status
   `completed`: a grader failure is a missing outcome label, not a Theoria
   decline, and must be rerun or resolved by the frozen adjudication policy.
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
