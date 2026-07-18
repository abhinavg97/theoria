# Open-weight model experiments

This directory contains the versioned inputs for Theoria's open-weight model
experiments. Official runs must use a clean Git checkout and must retain the
complete `runs/artifacts/<run_id>/` directory alongside the result JSON.

## Cohorts

- `smoke-1`: one bounded math readiness check.
- `smoke-5`: five HLE-Verified Gold problems spanning five domains.
- `dev-30`: a fixed prompt-calibration set disjoint from every recoverable
  problem in the paper audit.
- `paper-200`: the complete paper cohort (v1=96, v2=104). Of these IDs, 194
  come directly from `paper/audit.db`; six content-policy-blocked v2 IDs are
  reconstructed by position against the pinned HLE revision. The positional
  join reproduces all 194 known IDs exactly. The ordered cohort SHA-256 is
  `d8a5e29289350b59d50909bbe3a11b006fa3942831cfaaa2ab439300bda1cc43`.
- `paper-194`: only the directly retained IDs, for diagnostic comparisons.

Never tune prompts on `paper-200` or `paper-194`. The runner refuses missing or
duplicate ids before creating a run.

The OSS experiments grade every item against the answer in the pinned
HLE-Verified dataset revision. They do not rewrite an item label to reproduce
an earlier paper adjudication. In particular, the paper's post-fix treatment
of p134 must be reported separately when comparing against the published
baseline; it is not silently applied to new model runs. This keeps the new raw
grades reproducible while making the cross-paper denominator convention
explicit.

## Search service

Start the pinned SearXNG image:

```bash
docker compose -f experiments/searxng/compose.yaml up -d
curl -fsS 'http://localhost:8888/search?q=RFC+9110&format=json' | jq '.results | length'
```

The local-only service disables SearXNG's reverse-proxy rate limiter because
Theoria connects directly rather than through a proxy. JSON output is enabled
explicitly. The image digest and upstream revision are pinned in
`compose.yaml`.

## Local Ollama example

```bash
ollama pull qwen3:8b
theoria build --sage

python experiments/run_hle_condition.py \
  --cohort smoke-5 \
  --base-config configs/theoria_agent_ollama.yaml \
  --experiment-id qwen3-8b-smoke \
  --phase smoke \
  --trial 1
```

Use a model override after the Ollama base config for another Qwen size:

```bash
python experiments/run_hle_condition.py \
  --cohort dev-30 \
  --base-config configs/theoria_agent_ollama.yaml \
  --model-config experiments/configs/ollama_qwen3_14b.yaml \
  --experiment-id qwen3-14b-dev \
  --phase dev \
  --trial 1
```

For the explicit no-repair validation condition, add `--no-repair`. The
repair-enabled result also records `first_attempt_verified`, allowing a paired
first-pass operating point to be calculated without regenerating the initial
trace.

## Azure gpt-oss-120b example

The `gpt-oss-120b-v1-theoria-eval` Azure deployment uses catalog model version
`1` and `versionUpgradeOption=NoAutoUpgrade`. Load the key without writing it to
the shell history, then run a bounded readiness cohort:

```bash
export AZURE_OPENAI_API_KEY="$(az cognitiveservices account keys list \
  -g rg-soccer826-8608 -n theoria-resource --query key1 -o tsv)"

python experiments/run_hle_condition.py \
  --cohort smoke-5 \
  --base-config experiments/configs/azure_gpt_oss_120b_v1.yaml \
  --experiment-id gpt-oss-120b-v1-smoke-r0 \
  --phase smoke \
  --trial 1 \
  --no-repair
```

Before each official evaluation, verify that Azure still reports model
`gpt-oss-120b`, version `1`, and `NoAutoUpgrade`; the deployment etag in the
config must also match the live deployment.

Grade the final Theoria answer and the initial solver response independently.
The target-specific prompts and all grader calls are retained in separate,
sealed grading runs:

```bash
theoria grade runs/<run-id>.json \
  --target final \
  --config experiments/configs/grader_azure_gpt54.yaml \
  --config configs/searxng_search.yaml \
  --no-watch

theoria grade runs/<run-id>.json \
  --target solver_initial \
  --config experiments/configs/grader_azure_gpt54.yaml \
  --config configs/searxng_search.yaml \
  --no-watch
```

Repeat both targets with
`experiments/configs/grader_azure_deepseek_v4_flash.yaml`. Report each grader,
their conservative all-grader strict consensus, and every disagreement. The
automatic favorable category is a review queue, not an authoritative paper
label; manually adjudicate certified mismatches before making a favorable
precision claim.

## Rules for evaluation runs

1. Freeze and commit the model configuration and prompts before evaluation.
2. Use the same cohort, sandbox image, problem concurrency, and grading policy
   for every compared condition.
3. Use `--phase evaluation --trial N`; never use evaluation outputs to tune.
4. Resume only by repeating the identical command with `--resume <run_id>`.
5. Require `meta.json.status == "completed"` and verify the artifact manifest
   before including a run in a paper table.
6. Label models as open-source or open-weight according to their actual
   licenses; do not use the terms interchangeably.
