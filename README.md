# Theoria

**Don't trust an LLM's answer — make it prove the answer, then have independent judges check every step of the proof.**

Open source — and it doesn't ask for *your* trust either: every number reproduces from a committed database, and every reasoning trace is public.

Theoria is a verified-reasoning harness. Instead of asking a model a question and hoping the answer is right, Theoria makes the model produce a *step-by-step proof* of its answer — a chain of explicit state-to-state transitions, each carrying a justification — and then audits every transition with independent, skeptical judges. If every step survives the judges, Theoria marks the answer **JUDGE-PASSED**. If any step can't be justified, it **declines** rather than shipping a guess.

The point is calibration: a system that tells you *when it knows* — it judge-passes only the answers whose every step survives the judges, and declines the rest. On a text-only sample of HLE-Verified Gold (n=184), Theoria judge-passed **57%** (106/184) and declined the rest; of the ones it passed, **88%** (94/106) matched the answer key — rising to **~99%\*** (105/106) once you credit cases where Theoria's answer was right and the *key* was wrong. Run it yourself and see.

> \*Raw key-match precision is **94/106 = 88.68%** (both independent graders agreeing). Crediting two answers the harness mis-extracted but had proved correctly raises it to **96/106 = 90.57%**; the **~99%** figure (105/106) additionally counts answers that disagreed with the key but appear to have caught an error in the *key*. Every number here has a one-line query in [`paper/audit_queries.sql`](paper/audit_queries.sql).

> One further problem ran but the two graders split on whether its answer matched the key (p527); rather than guess, the headline is over the **184** problems both graders scored the same way. The audit DB keeps all **185** that ran — the paper reports the full set and details p527.

**You don't have to trust any of these numbers. Reproduce them in ~30 seconds, no benchmark access or API keys needed:**

```bash
sqlite3 paper/audit.db < paper/audit_queries.sql   # every figure prints with its expected value
```

`audit.db` ships in the repo (numbers-only, no benchmark text). And the complete per-problem evidence behind it (every proof, every judge verdict, both graders' reasoning) is published in full as a public gated (auto approve) dataset: **[i-eat-food/theoria-hle-audit](https://huggingface.co/datasets/i-eat-food/theoria-hle-audit)**.

> **Heads up:** I used an LLM to port my original implementation into this cleaner, public repo. It's been reviewed but not exhaustively tested — if you hit a bug, open an issue or ping me [@theoriacy](https://twitter.com/theoriacy).

---

## How it works

For each problem Theoria runs a loop of specialized LLM roles. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture; in brief:

1. **Solve** — a solver answers the question (web search encouraged in the audited profile).
2. **Formalize** — a formalizer rewrites the solution as a `Proof`: an `initial_state` plus `steps`, where each step is a new `state` justified by exactly one of three types — `computation` (an operation actually carried out), `citation` (a theorem/identity/definition), or `problem_given` (a fact from the problem text).
3. **Judge** — every step is audited *in parallel* by a judge specialized to its justification type, plus a judge for the initial state. Judges are deliberately brutal: they recompute, verify citations when search is available, and reject hidden assumptions.
4. **Pedantry filter** — re-examines each rejection: real error, or just nitpicking? Pedantic rejections are overridden.
5. **Convention lift** — a surviving rejection gets one last check: does a single standard, citable domain convention fully resolve it? If so, the step is accepted *under that explicitly-recorded assumption*.
6. **Repair loop** — anything still failing goes back to the formalizer, which fixes the proof or sends the solver back for a new answer.

A result is **verified** only if the initial state and every step are accepted.

The method lives in [`pipeline.py`](pipeline.py); every role's actual prompt is data in [`configs/defaults.yaml`](configs/defaults.yaml).

---

## Requirements

The **audited default configuration** runs on subscription access to two CLIs:

- **[Claude Code](https://claude.com/claude-code) (`claude`)** — runs the formalizer.
- **[Codex](https://developers.openai.com/codex/cli/) (`codex`)** — runs the solver and judges.

That configuration requires both subscriptions, **Docker**, **Python >= 3.12**, and **macOS**.

> **Why macOS for the audited default:** Theoria reads Claude subscription credentials from the macOS Keychain. The all-Codex OSS profile below does not use Claude credentials and is cross-platform.

The optional OSS path requires Codex plus a compatible Ollama, LM Studio, or custom model-provider endpoint. Those runs are **unaudited** and do not reproduce the published results.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python ≥3.12 (3.13 matches the reference runs)
pip install -e .          # installs the `theoria` command + dependencies
theoria doctor              # check your setup (CLIs, Docker, images)
theoria build               # build the two Docker sandbox images
theoria hle 1               # reproduce HLE problem 1 (streams live by default)
```

`theoria doctor` tells you exactly what's missing and how to fix it. `theoria build` builds both sandbox images (`theoria-sandbox` and the Sage math/science variant `theoria-sandbox-sage`); the Sage image installs SageMath via conda, so its first build takes several minutes.

Results land in `runs/<command>_<tag>_<timestamp>.json`; full per-call artifacts (every prompt, response, tool call, and event stream) under `runs/artifacts/`. Both are gitignored.

### Reproducing the HLE results

`theoria hle` runs the **audited default configuration**: every role on Codex except the Claude formalizer, `--docker`, the Sage image, and the default prompts. The preamble (the environment description handed to the solver and judges) is matched to the image automatically, so on the default Sage image they're told about Sage rather than the Debian sandbox. The published HLE v2 runs predate this and used the Debian preamble on the Sage image; pass `--no-pair-preamble` to reproduce them verbatim. (The HLE-Verified dataset is pulled live from Hugging Face and isn't version-pinned, so upstream dataset edits can shift results over time.) To run a span of problems:

```bash
theoria hle --skip 100 --max 50 --tag myrun --watch
```

Each problem runs in its own container; `--parallel N` only changes throughput, not per-problem results.

---

## Run your own question

Same engine, same setup as HLE — just pass a question with a definite, checkable answer:

```bash
theoria custom "Prove there are infinitely many primes"
```

---

## Open-source models (unaudited)

The cheapest OSS extension keeps **Codex as the agent runtime** and changes its model provider. This preserves Theoria's shell tools, structured outputs, sessions, sandboxing, and artifacts. A basic OpenAI-compatible Chat Completions server is not enough: the selected model and endpoint must support the tool-calling and structured-output behavior Codex uses.

For Ollama, start the server and make the model available, then select the all-role profile and provide the explicit model id:

```bash
theoria doctor --config configs/ollama.yaml \
  --codex-model YOUR_OLLAMA_MODEL --check-endpoint

theoria custom "Prove there are infinitely many primes" \
  --config configs/ollama.yaml --codex-model YOUR_OLLAMA_MODEL
```

`configs/ollama.yaml` points to `http://localhost:11434/v1`. Set `CODEX_OSS_BASE_URL` to override it. In Docker mode, Theoria maps a loopback endpoint to `host.docker.internal`; `--no-docker` uses the URL directly. The profile sets every role, including the formalizer, to `backend: codex`. Codex formalizer repairs are stateless: each repair call includes the prior proof and failed verdicts.

[`configs/oss_custom_provider.example.yaml`](configs/oss_custom_provider.example.yaml) shows the corresponding `model_providers` wiring for a hosted or self-hosted provider. It forwards only the environment-variable names in `provider_env`; credentials must remain in the environment and must never be written into YAML. Provider compatibility must be tested individually.

Theoria uses one sandbox per problem, so every role in that problem can read the credentials made available to the sandbox. An external provider therefore cannot be mixed with Claude, Codex cloud, or a second external provider by default. Keep each run within one provider trust domain. The escape hatch `_security.allow_mixed_provider_credentials: true` is intentionally explicit and should be used only when every configured provider is trusted with every credential in the run.

Keep Docker enabled for external providers. With `--no-docker`, the provider-controlled agent can inspect host-readable files and environment values through shell tools, so host execution fails closed. For deliberate local compatibility testing, stack [`configs/unsafe_host_provider.yaml`](configs/unsafe_host_provider.yaml) after the provider profile; selecting it explicitly accepts that exposure.

Search is deliberately off in the supplied OSS profiles. Fetching a known URL can verify that source, but it is not web search or source discovery. Citation-heavy problems will therefore decline more often and have lower verification coverage. A Codex-supported search facility or an MCP search server available inside the container can restore discovery, but that integration must be tested with the selected provider before treating its results as comparable.

The OSS answer-grader preset is [`configs/ollama_audit_grader.yaml`](configs/ollama_audit_grader.yaml). It is also unaudited and uses the same conservative no-search policy:

```bash
theoria grade runs/hle_<timestamp>.json \
  --config configs/ollama_audit_grader.yaml \
  --config configs/unsafe_host_provider.yaml \
  --codex-model YOUR_OLLAMA_MODEL
```

---

## Read and grade your results

```bash
# Render a run as readable markdown: a verified answer + its proof, plus a
# full trace of every attempt, judge verdict, and filter decision.
theoria show runs/custom_<timestamp>.json        # one run file at a time

# Grade answers vs. expected with an LLM grader (handles equivalent
# notations/forms) — a stronger check than the built-in `correct` substring
# flag, though still an LLM judgment, not ground truth.
theoria grade runs/hle_<timestamp>.json          # one run file at a time
```

`theoria grade` writes per-problem `key_match` + `dispute_category` to `runs/grades/`.

---

## Configuration

`theoria` defaults to the audited HLE setup; advanced users can override individual knobs:

| Flag | Default | Effect |
|---|---|---|
| `--docker` / `--no-docker` | on | Per-problem container isolation (keep on). |
| `--image NAME` | sage | Which sandbox image when Docker is on. |
| `--config PATH` | none | Stack an extra YAML on `configs/defaults.yaml` (repeatable). |
| `--codex-model MODEL` | role config | Override the model for every Codex-backed role. |
| `--no-pair-preamble` | off | Don't auto-match the preamble to the image (reproduces the published v2 runs). |
| `--parallel N` | 1 | Concurrency (does not affect per-problem results). |
| `--watch` / `--no-watch` | on | Stream live LLM events to stderr. |

Stackable config presets in `configs/`: `defaults.yaml` (base — every role's prompt), `sage.yaml`, `restricted.yaml` (tool allowlists), `tuned.yaml`, and `audit_grader.yaml` (the audited grader role). `ollama.yaml`, `ollama_audit_grader.yaml`, and `oss_custom_provider.example.yaml` are unaudited OSS profiles/templates. `unsafe_host_provider.yaml` is an explicit security opt-in for external-provider runs without Docker.

`theoria doctor` accepts repeatable `--config` options and checks only the CLIs, credentials, environment variables, Docker image, and provider mode used by the merged configuration. Add `--check-endpoint` for a reachability probe; the command does not print the configured endpoint or any credential value. Use `--no-docker` to check a host-only run.

---

## Scope and honesty notes

- **The judges are LLMs.** Verification is looped LLM auditing plus real tool/code execution in the sandbox — not a fixed formal proof checker. It raises the bar far above a single model's say-so; it is not a theorem prover.
- **The built-in `correct` flag is a naive substring check**, for scanning only. The real signals are the `verified` flag and `theoria grade`.
- **Answers are extracted** as the first element of the proof's final state; occasionally a correct conclusion sits behind a placeholder. Read the proof, not just the headline.
- Every run records its exact models, CLI versions, and sandbox image in `runs/artifacts/<run_id>/meta.json` for reproducibility.
- **Published numbers apply only to the audited Claude/Codex configuration.** OSS profiles, custom providers, and their grader are unaudited and can have materially different coverage and precision.
- **Grading.** The bundled `theoria grade` shows the LLM grader slightly more per-attempt context than the internal audit grader it's ported from, so its grades may differ marginally on borderline cases — the JUDGE-PASSED/REJECTED verdict itself is unaffected.

---

## The paper & full results

The full paper — **[`paper/theoria_white_paper_draft.md`](paper/theoria_white_paper_draft.md)** — covers the method, the HLE-Verified Gold audit, every number, and nine figures.

Every number in the paper reproduces from a committed, numbers-only database — no benchmark access needed:

```bash
sqlite3 paper/audit.db < paper/audit_queries.sql   # prints each figure with its expected value
python paper/generate_figures.py                   # redraws all 9 figures from that DB
```

The complete per-problem traces (full proofs, every judge verdict, both graders' reasoning, and the HLE-Verified question/answer text) are published as a **public gated (auto approve)** Hugging Face dataset: **[i-eat-food/theoria-hle-audit](https://huggingface.co/datasets/i-eat-food/theoria-hle-audit)**. See [`DATA.md`](DATA.md) for provenance and licenses.

---

## Data & credits

HLE problems come from **[HLE-Verified](https://arxiv.org/abs/2602.13964)** (`skylenage/HLE-Verified` on Hugging Face) — an expert-verified version of *Humanity's Last Exam*. Theoria pulls it live from Hugging Face at runtime and does not redistribute it; consult the dataset's page for its own terms of use. By default Theoria uses the **Gold** subset, text-only.

---

## License

Copyright 2026 Michael Saldivar. Released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
