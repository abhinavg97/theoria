# Open-weight model experiments

Use [`PUBLICATION_COOKBOOK.md`](PUBLICATION_COOKBOOK.md) for the frozen
protocol and exact commands. The short examples below are readiness checks,
not a substitute for that protocol.

## HLE cohorts

- `smoke-1`: one protocol check.
- `smoke-5`: five domains; nested inside `dev-30`.
- `dev-30`: prompt calibration only, disjoint from all 200 paper items.
- `paper-ran-185`: exact IDs behind the arXiv headline denominator. Ordered-ID
  SHA-256: `31e912acbee669cce95ea10c9fb8a5241a23011cc174b0c2c28d6a87c2e54cea`.
- `paper-sampled-200`: all sampled paper items, including nine historical
  crashes and six policy blocks. Ordered-ID SHA-256:
  `d8a5e29289350b59d50909bbe3a11b006fa3942831cfaaa2ab439300bda1cc43`.
- `paper-recoverable-194`: only IDs directly retained in `paper/audit.db`.

`paper-185` and `paper-200` remain compatibility aliases. Never tune on any
paper cohort.

The 185-item cohort is the direct historical comparison. The 200-item cohort
is the intent-to-treat sensitivity analysis. The committed DB contains 106
certifications; arXiv reports the post-fix bucket of 105 by excluding p134.
The reported 96/105 credits extraction-only p133 and p509; the strict selected-
candidate comparator used here is 94/105.

## Source audit

The benchmark artifacts needed by the paper are on the companion analysis
repository's `experiments` branch, not `master`:

```bash
git clone --branch experiments https://github.com/zaladbar/theoria_analysis.git \
  ../theoria_analysis
git -C ../theoria_analysis checkout --detach \
  52344a6bd2e301b0657fb16a9e9ccd9ad41bbb81

python3 experiments/audit_source_artifacts.py \
  --analysis-repo ../theoria_analysis \
  --out runs/source_audit.json
```

The audit deliberately surfaces two historical limitations: the GPQA-65 set
is outcome-filtered and mixes model stacks, and the structured poison results
reject 12/15 generated controls. Use GPQA-fixed-100 for new OOD claims and
report all poison controls.

## Search and sandbox

```bash
docker compose -f experiments/searxng/compose.yaml up -d
theoria build --sage
```

SearXNG search results are snippets, not fetched-source evidence. The
`mechanistic_tools.yaml` overlay is therefore a tool-execution diagnostic, not
evidence-gated R2 repair.

## Local readiness example

```bash
ollama pull qwen3:8b-q8_0

python experiments/run_hle_condition.py \
  --cohort smoke-5 \
  --base-config configs/theoria_agent_ollama.yaml \
  --model-config experiments/configs/ollama_qwen3_8b_q8.yaml \
  --experiment-id qwen3-8b-q8-smoke-r0 \
  --phase smoke --trial 1 --parallel 1 --no-repair
```

Ollama tags can move and Q8 is a distinct condition. Main paper runs should use
the immutable Hugging Face revisions in `model_matrix.yaml` through the pinned
vLLM recipe.

## What is implemented

- Audited all-role provider-neutral runs on HLE or local JSON benchmarks.
- Explicit R0 and current-loop R1, including paired first-pass metrics.
- Solver-initial, first-attempt, and final grading targets.
- Wilson intervals, per-domain metrics, repair aggregates, answer hashes, tool
  telemetry, protocol reprompt counts, model/runtime identity, and sealed
  artifacts.
- Deterministic, trace-validated GPQA summaries with paired repair inference.
- Hash-bound two-reviewer HLE adjudication for R0/R1 and paired inference.
- Audited weak-solver/strong-verifier config composition for conditional role
  ablations.

R2 structured evidence-gated repair, R3 matched-budget fresh resampling, a new
holistic matched-coverage runner, and an audited direct-proof poison runner are
not implemented. Do not assign those labels to existing output.
