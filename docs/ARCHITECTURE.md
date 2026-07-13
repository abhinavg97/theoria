# Architecture

Theoria is small. There are four engine files, three thin layers on top, and a
folder of prompts. If you read `pipeline.py` top to bottom you've understood
the system.

## The files

```
pipeline.py    THE METHOD. The solve → formalize → judge → filter → repair
               loop, plus the proof/verdict data types. Read this first.
llm.py         Backend plumbing. Calls the `claude` or `codex` CLI (optionally
               inside Docker), enforces JSON-schema output, streams events,
               writes per-call artifacts, and survives hangs.
sandbox.py     One hardened Docker container per problem; streams only the
               credentials/config needed by active providers into a private
               tmpfs home; tears it down.
harness.py     Runs many problems with bounded parallelism, saves incrementally,
               and records reproducibility metadata for each run.

loaders.py     Builds the `problem` dicts: HLE problems or an ad-hoc question.
grade.py       LLM-judge grader (answer vs. expected) — a stronger check than
               the naive `correct` flag, not ground truth. Reads a run JSON.
export_problem.py   Renders a run JSON into readable markdown.
cli.py         The `theoria` command that ties it all together.

configs/       Every role's prompt and model, as data. defaults.yaml is the
               spec of what each judge actually does.
sandbox/, sandbox-sage/   The two Docker images (standard, and Sage for math).
```

## The verification loop

```
                         ┌─────────────────────────────────────────┐
                         │                problem                   │
                         └─────────────────────────────────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │    SOLVE     │  solver answers
                                   └──────────────┘  (web search ok)
                                          │
                                          ▼
                                   ┌──────────────┐   reject
                                   │  FORMALIZE   │──────────────┐
                                   │  → Proof     │              │ (back to
                                   └──────────────┘              │  solver for
                                          │ proof                │  a new answer)
                                          ▼                      │
              ┌───────────────────────────────────────────┐     │
              │   JUDGE  (all steps + state 0, in parallel)│     │
              │   computation · citation · problem_given   │     │
              └───────────────────────────────────────────┘     │
                                          │ any rejections?      │
                            ┌─────────────┴───────────┐          │
                          no│                          │yes       │
                            ▼                          ▼          │
                     ┌────────────┐            ┌──────────────┐   │
                     │JUDGE-PASSED│            │  PEDANTRY    │   │
                     │     ✓      │            │  filter      │   │ over the
                     └────────────┘            └──────────────┘   │ attempt
                                                      │ still bad │ limit?
                                                      ▼           │
                                               ┌──────────────┐   │
                                               │ CONVENTION   │   │
                                               │ lift         │   │
                                               └──────────────┘   │
                                                      │ still bad │
                                                      └───────────┘
                                                       repair loop
```

- **Three justification types**, each with its own judge prompt:
  `computation` (an operation that was actually performed),
  `citation` (a theorem / identity / definition that licenses the step),
  `problem_given` (a fact taken directly from the problem text).
- **State 0** (the proof's initial state) is audited by its own judge in
  parallel with the steps — it catches content smuggled in as a premise.
- **Pedantry filter**: brutal judges over-reject informally-worded problems.
  This pass asks, per rejection, "real error or nitpick?" and overrides
  nitpicks (tagged `[PEDANTRY OVERRIDE]`).
- **Convention lift**: a rejection that's a *real* gap may still be closed by
  one standard, citable convention (e.g. "assume an inertial frame"). If so
  the step is accepted *under that assumption*, which is recorded — such a
  result is `verified` but not `verified_unconditionally`.
- **Repair loop**: failed verdicts go back to the formalizer, bounded by
  `max_verify_attempts` and `max_solver_answers` (defaults 3 and 3).

**A note on terminology.** A pass means *every step's justification was
accepted by an independent LLM judge* — it is **not** formal/axiomatic
verification. So the outward-facing verdict is reported as **JUDGE-PASSED /
REJECTED**, not "verified." The internal result field is still named
`verified` (a boolean), and at the system level the decline behavior is
*abstention*: Theoria withholds an answer rather than ship one it can't justify.

**What it's for.** Theoria verifies questions that have a **definite,
checkable answer** — a value, expression, multiple-choice letter, or other
conclusion a proof can land on. Open-ended or subjective prompts ("what do you
think of…", "why is the sky blue?") don't fit the proof model; the formalizer
is forced into an awkward conclusion and such questions are typically REJECTED.
That's the system working as intended, not a failure.

## Backends

`llm.py` dispatches each role to the **Claude** or **Codex** CLI from its
merged settings. The audited default uses Codex for the solver and judges and
Claude for the formalizer, so it requires both subscriptions. The Claude
formalizer repairs a proof by resuming its structured-output session.

A config may instead put the formalizer on Codex. That path is deliberately
stateless: every repair call includes the solution, prior proof, and failed
verdicts in a fresh structured-output request. It does not depend on whether a
particular Codex release accepts an output schema while resuming a session.

Codex-backed roles can use OpenAI cloud models, Codex's built-in Ollama or LM
Studio adapters, or an explicitly configured Codex `model_provider`. Keeping
Codex as the runtime preserves the agent tool loop and artifacts; a bare HTTP
chat-completions backend would not. The supplied OSS provider profiles disable
Codex's native search and take the conservative path when an external claim
cannot be verified. Fetching a known URL is direct retrieval, not discovery.

Native search cannot simply be enabled for local providers: Codex serializes
its `web_search` tool — and every MCP server — as Responses API tool types
(`web_search`, `namespace`) that Ollama and LM Studio reject before
inference, so the call layer fails closed on `search: true` for OSS roles and
rejects MCP declarations in `codex_config`. The stackable `brave_search.yaml`
and `searxng_search.yaml` profiles restore discovery through the one tool
path all providers share: a `theoria-search` command inside the sandbox
image, invoked through the shell function tool, calling the configured
search backend directly from the container. `_web_search` is run-level
configuration with two providers: `brave` (hosted keyed API on a fixed
endpoint — the key is referenced by environment-variable name and joins the
same forwarding allowlist and redaction path as provider credentials) and
`searxng` (self-hosted, credential-free — the instance endpoint gets the same
validation and loopback routing as an OSS model endpoint). Runtime metadata
records credential issuers independently from model/search data destinations:
SearXNG is a destination but not a credential domain, while Brave is both.
This makes a keyed custom model plus Brave an explicit mixed-credential run,
without misclassifying a key-free model/search combination. LLM answer engines
are deliberately not backends: judges
must verify primary sources, not another model's synthesis. Helper
invocations are counted as `web_search_requests` in call metadata. Search
results are untrusted leads: one centrally derived prompt policy requires
agents to ignore embedded instructions and fetch and inspect the underlying
source before accepting a claim. Provider prompt suffixes cannot replace that
policy when configuration profiles are stacked.

Credentials are scoped to a per-problem sandbox, not to an individual role.
The runtime therefore rejects configurations that expose a credential across
an external model trust boundary or combine multiple external credential
issuers by default. The
`_security.allow_mixed_provider_credentials` escape hatch requires an explicit
acknowledgement that every role can access every credential in that run.
External-provider host execution is also rejected by default because an
agent's shell tools can inspect host-readable files and environment values.
`configs/unsafe_host_provider.yaml` is the explicit development-only opt-in;
Docker remains the supported isolation boundary.

Only the default Claude/Codex configuration and published traces are audited.
All-OSS and custom-provider configurations can change both coverage and
precision and must not be presented as reproductions of the published result.

## What a run leaves behind

```
runs/<command>_<tag>_<timestamp>.json     the results (one entry per problem)
runs/partial/<id>.json                    crash-safe live snapshot
runs/artifacts/<run_id>/
    meta.json                             argv, config, model + CLI versions,
                                          git sha, sandbox image digest
    <problem_id>/call_NNN_<role>/         every prompt, response, tool call,
                                          and raw event stream, per LLM call
```

Everything needed to reproduce or audit a run is on disk.
