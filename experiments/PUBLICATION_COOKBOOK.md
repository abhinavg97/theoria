# Publication cookbook: open-weight Theoria

This protocol evaluates selective prediction, not leaderboard accuracy. The
primary estimands are strict certified precision and coverage. Treat every
execution failure as a decline in the intent-to-treat denominator.

All main conditions use Theoria's `/chat/completions` provider-neutral loop.
Models emit validated JSON actions; Theoria performs search and executes shell
commands inside its Docker sandbox. Provider-native function calling is not
part of the experimental contract.

## 1. Audit verdict on the earlier Codex commit

The branch-only commit is `a58c0d9` on top of merged audit code at `4a4947e`.
The reason is not recorded beyond its commit message, so it must be inferred
from the diff and adjacent ignored run artifacts. Pre-commit Qwen and DeepSeek
smokes exposed two real failures: unresolved `ANSWER` placeholders could pass,
and direct role JSON was confused with the outer action protocol. The diff also
addressed a reproducibility gap where a later failure could discard partial
solver/proof state. A subsequent gpt-oss smoke still exhausted the action loop,
confirming the need to measure protocol compliance rather than assume it. The
cohort, solver-only grading, artifact sealing, model probing, partial-state
retention, and protocol/tool telemetry are useful and should remain.

The clearest local evidence is
`runs/custom_deepseek-v4-flash-tool-smoke_20260717_170804.json`, which records
`answer="ANSWER"` with `verified=true`. The Qwen smoke variants record either
the same unresolved slot or action-loop exhaustion. These files are ignored
local diagnostics, not benchmark results.

The commit did not make the requested study publication-ready:

- `paper-200` was not the paper's 185-item headline denominator.
- GPQA, poisoned-proof, matched-coverage, R2, and R3 runners were absent.
- Search-result snippets were counted as tool use but are not fetched evidence.
- The short recipe implied more completeness than the code provided.
- Core verifier fixes change the evaluated pipeline. Call this frozen version
  `Theoria-OSS-v1`; do not call it a bit-for-bit paper replication.

This branch keeps the defensible changes, adds `paper-ran-185`, preserves
`paper-sampled-200`, and labels unsupported conditions as implementation gates.
The new strict metric grades the selected answer itself: placeholders receive
no extraction credit. Accordingly, cite the paper's 96/105 exactly as reported,
but also disclose that its database has 94/105 selected-candidate key matches;
p133 and p509 account for the two extraction-only credits.

## 2. Freeze the research questions before spending compute

Use these conditions and names exactly:

| ID | Meaning | Current support |
|---|---|---|
| R0 | One solver answer, one semantic formalization/judge cycle | Yes |
| R1 | Current Theoria repair loop | Yes |
| R2 | Structured, evidence-gated localized repair | No; implementation gate |
| R3 | Fresh resample matched to R2 calls/tokens | No; implementation gate |

For R1, the `r0_first_pass` operating point comes from the same trace and is
the primary paired R0 comparison. Run the explicit R0 condition as a validation
that disabling repair produces the same first-pass decisions, not as the only
R0 estimate. In summaries, `repair_attempted` means any current-loop retry;
`semantic_repair_attempted`, formalizer-invalid retries, provider retries, and
JSON/schema reprompts are reported separately.

Primary benchmarks:

1. `paper-ran-185`: exact IDs behind the arXiv HLE denominator. This is the
   direct historical comparison.
2. `paper-sampled-200`: all intended paper items, including the nine historical
   crashes and six content-policy blocks. This is the intent-to-treat
   sensitivity analysis.
3. `gpqa-fixed-100`: the 25 first-batch and all 75 prespecified second-batch
   GPQA rows. Use this for new OOD claims.
4. `gpqa-legacy-paper-65`: appendix only. It retained 40 second-batch rows
   after successful execution, counted one failed first-batch item, and mixes
   two different historical model stacks.
5. Poisoned proofs: all 95 poisons plus all 15 controls. Report sensitivity and
   control specificity together. Do not headline this benchmark until the
   poison and control labels receive blinded human validation.

## 3. Pin and audit every source

Run this and every later section in Bash on the same Linux GPU host that will
serve vLLM, run Theoria, run Docker, and host SearXNG. Keeping every endpoint on
loopback avoids an undocumented network hop. Start in the Theoria checkout:

```bash
set -euo pipefail
export THEORIA_ROOT="$(git rev-parse --show-toplevel)"
export ARTIFACT_ROOT="$THEORIA_ROOT/../theoria-publication-artifacts"
mkdir -p "$ARTIFACT_ROOT"

git clone https://github.com/zaladbar/theoria-research.git \
  "$ARTIFACT_ROOT/theoria-research"
git -C "$ARTIFACT_ROOT/theoria-research" checkout --detach \
  6b7b395d98e12f1a20bac8a777e6888bd9575db3

git clone --branch experiments \
  https://github.com/zaladbar/theoria_analysis.git \
  "$ARTIFACT_ROOT/theoria_analysis"
git -C "$ARTIFACT_ROOT/theoria_analysis" checkout --detach \
  52344a6bd2e301b0657fb16a9e9ccd9ad41bbb81

cd "$THEORIA_ROOT"
python3 experiments/audit_source_artifacts.py \
  --analysis-repo "$ARTIFACT_ROOT/theoria_analysis" \
  --out runs/source_audit.json

test "$(git -C "$ARTIFACT_ROOT/theoria-research" rev-parse HEAD)" = \
  6b7b395d98e12f1a20bac8a777e6888bd9575db3
test "$(git -C "$ARTIFACT_ROOT/theoria_analysis" rev-parse HEAD)" = \
  52344a6bd2e301b0657fb16a9e9ccd9ad41bbb81

python3 - <<'PY'
import json
from pathlib import Path

r = json.loads(Path("runs/source_audit.json").read_text())
assert r["hle"]["sampled"]["count"] == 200
assert r["hle"]["ran"]["count"] == 185
assert r["hle"]["committed_db_certified"] == 106
assert r["hle"]["arxiv_postfix_certified_excluding_p134"] == 105
assert r["hle"]["arxiv_postfix_strict_correct"] == 96
assert r["hle"]["arxiv_postfix_selected_candidate_key_match"] == 94
assert r["hle"]["arxiv_postfix_extraction_only_problem_numbers"] == [133, 509]
assert r["gpqa"]["paper65_count"] == 65
assert r["gpqa"]["step7"] == {
    "selected_before_run": 75,
    "results_written": 68,
    "retained_clean": 40,
    "errored_results": 28,
    "never_ran": 7,
}
assert r["poisoned_proofs"]["poisoned"] == 95
assert r["poisoned_proofs"]["controls"] == 15
assert r["poisoned_proofs"]["structured_control_rejections"] == 12
PY
```

The audit JSON must contain all of the following before any model run:

```text
HLE sampled=200, ran=185, DB certified=106, arXiv post-fix=105/96
HLE selected-candidate-only historical count=94/105; p133/p509 are extraction-only
GPQA legacy=65, step7 selected=75, retained clean=40, errored=28, never ran=7
Poison=95, controls=15, structured control rejections=12
```

Pin arXiv v3 independently:

```bash
curl -fL https://arxiv.org/pdf/2607.01223v3 \
  -o "$ARTIFACT_ROOT/2607.01223v3.pdf"
curl -fL https://arxiv.org/e-print/2607.01223v3 \
  -o "$ARTIFACT_ROOT/2607.01223v3.tar"
printf '%s  %s\n' \
  e9408646a8613502393b0bacd0d3a5a28a8920db00d31b0098e27262e0b6a8d1 \
  "$ARTIFACT_ROOT/2607.01223v3.pdf" \
  a3ef485d4903556120beae1ca69db6dda5c58f6301cc7dfe5952730870e88db8 \
  "$ARTIFACT_ROOT/2607.01223v3.tar" \
  | sha256sum -c -
```

Expected hashes:

```text
e9408646a8613502393b0bacd0d3a5a28a8920db00d31b0098e27262e0b6a8d1  PDF
a3ef485d4903556120beae1ca69db6dda5c58f6301cc7dfe5952730870e88db8  source
```

## 4. Create the Python and sandbox environments

```bash
cd "$THEORIA_ROOT"
uv --version | grep '^uv 0\.11\.28 '
uv venv --python 3.13.14 .venv
source .venv/bin/activate
uv sync --frozen --extra test
test "$(command -v theoria)" = "$THEORIA_ROOT/.venv/bin/theoria"
python - <<'PY'
from pathlib import Path
import cli

assert Path(cli.__file__).resolve() == Path("cli.py").resolve()
PY
theoria --help >/dev/null
python -m pytest -q

docker compose -f experiments/searxng/compose.yaml up -d
curl -fsS 'http://127.0.0.1:8888/search?q=RFC+9110&format=json' \
  | python -c 'import json,sys; assert json.load(sys.stdin)["results"]'
docker image inspect \
  searxng/searxng@sha256:11ffedd387dc9cf99e881250c67861470384e55194a86f76df76aa0034a28a1a \
  > runs/searxng.inspect.json
docker save \
  searxng/searxng@sha256:11ffedd387dc9cf99e881250c67861470384e55194a86f76df76aa0034a28a1a \
  | gzip -n > "$ARTIFACT_ROOT/searxng.tar.gz"
sha256sum "$ARTIFACT_ROOT/searxng.tar.gz"

theoria build --sage
docker images -q --no-trunc theoria-sandbox-sage:latest
docker image inspect theoria-sandbox-sage:latest \
  > runs/theoria-sandbox-sage.inspect.json
docker save theoria-sandbox-sage:latest \
  | gzip -n > "$ARTIFACT_ROOT/theoria-sandbox-sage.tar.gz"
sha256sum "$ARTIFACT_ROOT/theoria-sandbox-sage.tar.gz"
```

The Dockerfiles contain mutable bases and unpinned package solves. The saved
OCI image, its SHA-256, and the run-recorded image ID are therefore required
release artifacts. Rebuilding later is not an exact reproduction.

## 5. Serve immutable open-weight checkpoints

Use vLLM BF16/native MXFP4 for main tables. Use Ollama quantized weights only
as a separately labeled appendix condition.

The supported main-table topology is co-located: vLLM listens on
`127.0.0.1:8000`, SearXNG listens on `127.0.0.1:8888`, and the Theoria runner
uses those loopback endpoints on this same Linux GPU host. Do not run the
following server on one host and the generated runner config on another. A
remote topology is a different deployment and needs a separately frozen tunnel
or authenticated endpoint recipe.

In a dedicated vLLM server shell, set the same checkout and artifact roots, then
capture the complete serving environment before starting the server:

```bash
set -euo pipefail
export THEORIA_ROOT="$(git rev-parse --show-toplevel)"
export ARTIFACT_ROOT="$THEORIA_ROOT/../theoria-publication-artifacts"
cd "$THEORIA_ROOT"

export VLLM_ENV="$ARTIFACT_ROOT/envs/vllm-0.25.1"
uv venv --python 3.12 "$VLLM_ENV"
source "$VLLM_ENV/bin/activate"
uv pip install 'vllm==0.25.1' --torch-backend=auto
python -c 'import vllm; assert vllm.__version__ == "0.25.1"'

export MODEL=qwen3-8b
export CUDA_VISIBLE_DEVICES=0
export SERVER_RUN_ID="${MODEL}-official-v1"
export SERVER_DIR="$ARTIFACT_ROOT/servers/$SERVER_RUN_ID"
test ! -e "$SERVER_DIR"
mkdir -p "$SERVER_DIR"
python --version > "$SERVER_DIR/python-version.txt"
uv --version > "$SERVER_DIR/uv-version.txt"
uv pip freeze > "$SERVER_DIR/python-freeze.txt"
uname -a > "$SERVER_DIR/uname.txt"
cp /etc/os-release "$SERVER_DIR/os-release.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" \
  > "$SERVER_DIR/cuda-visible-devices.txt"
nvidia-smi -q > "$SERVER_DIR/nvidia-smi-q.txt"
sha256sum experiments/model_matrix.yaml experiments/serve_vllm.py \
  > "$SERVER_DIR/server-inputs.sha256"

read -r HF_ID HF_REV < <(python - "$MODEL" <<'PY'
import sys
import yaml

spec = yaml.safe_load(open("experiments/model_matrix.yaml"))["models"][sys.argv[1]]
print(spec["hf_id"], spec["revision"])
PY
)
export HF_ID HF_REV
MODEL_SNAPSHOT="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=os.environ["HF_ID"], revision=os.environ["HF_REV"]))
PY
)"
export MODEL_SNAPSHOT
printf '%s\n' "$HF_ID@$HF_REV" > "$SERVER_DIR/model-revision.txt"
printf '%s\n' "$MODEL_SNAPSHOT" > "$SERVER_DIR/model-snapshot-path.txt"
(cd "$MODEL_SNAPSHOT" && find -L . -type f -print0 | sort -z \
  | xargs -0 sha256sum) > "$SERVER_DIR/model-snapshot-files.sha256"

python experiments/serve_vllm.py "$MODEL" > "$SERVER_DIR/command.txt"
cat "$SERVER_DIR/command.txt"

# This process stays in the foreground. Keep the shell open for the condition.
python experiments/serve_vllm.py "$MODEL" --execute 2>&1 \
  | tee "$SERVER_DIR/server.log"
```

Replace `qwen3-8b` with a key from `experiments/model_matrix.yaml`. The script
pins weight and tokenizer revisions, context 32768, engine seed 20260718,
single-sequence scheduling, Qwen thinking mode, reasoning parser, and Mistral
loader flags. `--tensor-parallel-size` may be changed only before the config
freeze and must be reported. When changing tensor parallelism, memory
utilization, model length, or sequence count, pass the same frozen values to
both `serve_vllm.py` and `make_model_config.py`. Set `CUDA_VISIBLE_DEVICES` to
the exact devices used by that tensor-parallel configuration and retain it with
the server record.

Do not resolve a fresh dependency environment between trials of one condition.
Release `python-freeze.txt`, the GPU/driver report, command, input hashes, and
server log with the run. The served model name contains the pinned Hugging Face
revision, but that name alone is not weight attestation; retain the vLLM log and
downloaded snapshot metadata as well.

In a second shell on the same host, reactivate the runner environment and create
the matching Theoria config. Record actual hardware, not a desired
configuration:

```bash
set -euo pipefail
export THEORIA_ROOT="$(git rev-parse --show-toplevel)"
export ARTIFACT_ROOT="$THEORIA_ROOT/../theoria-publication-artifacts"
cd "$THEORIA_ROOT"
source .venv/bin/activate
test "$(command -v theoria)" = "$THEORIA_ROOT/.venv/bin/theoria"

export MODEL=qwen3-8b
export CUDA_VISIBLE_DEVICES=0
export HARDWARE="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=name,memory.total \
  --format=csv,noheader | paste -sd ';' -)"
mkdir -p experiments/frozen_configs

python experiments/make_model_config.py "$MODEL" \
  --server-hardware "$HARDWARE" \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --out "experiments/frozen_configs/${MODEL}.yaml"

theoria doctor --backend theoria_agent \
  --config "experiments/frozen_configs/${MODEL}.yaml" \
  --config configs/searxng_search.yaml \
  --docker --image theoria-sandbox-sage:latest --check-endpoint
```

Starting calibration values, to be frozen rather than mixed silently:

| Track | Temperature | top-p | top-k | Other |
|---|---:|---:|---:|---|
| Qwen3 hybrid | 0.6 | 0.95 | 20 | thinking enabled |
| DeepSeek R1 Distill | 0.6 | 0.95 | unset | system instructions folded into user |
| Qwen3 2507 non-thinking | 0.7 | 0.8 | 20 | separate ablation |
| Mistral Small 3.2 | 0.15 | 1.0 | unset | Mistral loader |
| gpt-oss | 1.0 | 1.0 | unset | `--reasoning-effort high` |

Examples:

```bash
python experiments/make_model_config.py deepseek-r1-distill-qwen-14b \
  --server-hardware '1x-H100-80GB' \
  --temperature 0.6 --top-p 0.95 \
  --out experiments/frozen_configs/deepseek-r1-distill-qwen-14b.yaml

python experiments/make_model_config.py mistral-small-3.2-24b-instruct-2506 \
  --server-hardware '2x-H100-80GB' \
  --temperature 0.15 --top-p 1.0 \
  --out experiments/frozen_configs/mistral-small-3.2-24b.yaml

python experiments/make_model_config.py gpt-oss-120b \
  --server-hardware '2x-H100-80GB' \
  --temperature 1.0 --top-p 1.0 --reasoning-effort high \
  --out experiments/frozen_configs/gpt-oss-120b.yaml
```

Do not invent `Qwen3-8B/14B/32B-Instruct-2507` rows; those official Qwen
checkpoints do not exist. The clean dense scale spine is hybrid
`Qwen3-{4B,8B,14B,32B}`. Call the umbrella set open-weight. Llama 3.3 uses a
custom community license and must not be labeled OSS.

## 6. Calibrate without touching evaluation items

The smoke cohorts are nested in `dev-30`; all are disjoint from the 200 paper
items. Limit each family to at most three prompt candidates. Use two separate
dev conditions per candidate: the ordinary verifier for quality selection, and
the mechanistic overlay only as a tool-compliance gate. Never use the gated
condition's precision or coverage as the ordinary verifier's estimate.

A candidate may rewrite the actual solver, formalizer, typed-judge, pedantry,
convention, and repair prompts, including examples, decomposition, verbosity,
search/shell instructions, and failure paths. It may also freeze a different
turn/token/context/reasoning budget. Treat the whole prompt/config bundle as
one versioned candidate; do not quietly carry individual edits between
candidates or retune a selected bundle on evaluation items.

The frozen lexicographic rule is:

1. sealed 5-item smoke and ordinary 30-item dev outputs with zero execution
   failures; policy failures in the separate tool gate remain in its compliance
   denominator;
2. 100% solver tool-obligation completion in smoke and at least 95% typed-judge
   tool-obligation completion in the separate dev gate;
3. highest all-grader-consensus strict certified-precision Wilson lower bound
   on the ordinary dev run;
4. highest coverage only as the final tie-breaker.

Freeze the role-ablation trigger now: run the 8B-solver/32B-verifier condition
only if the selected Qwen3-8B ordinary `dev-30` row has consensus strict
precision below 0.90 or R1 coverage below 0.20, and Qwen3-32B passes the
frozen five-item protocol smoke with no execution error. Evaluate the trigger
on dev only and record its boolean result before any HLE-185 output is opened.

The two Azure grader deployments are private, fixed research infrastructure.
The deployment owner must export the key and be logged into the Azure CLI so
the grading artifact can verify the live control plane. A public reproducer may
substitute other graders only as a separately named, fully frozen condition.
Define this helper once in the runner shell; sections 7, 8, and 10 reuse it:

```bash
set -euo pipefail
cd "$THEORIA_ROOT"
source .venv/bin/activate
: "${AZURE_OPENAI_API_KEY:?export the fixed grader deployment key}"
command -v az >/dev/null
az account show >/dev/null

export GRADER_GPT54=experiments/configs/grader_azure_gpt54.yaml
export GRADER_DEEPSEEK=experiments/configs/grader_azure_deepseek_v4_flash.yaml
export SEARCH_CONFIG=configs/searxng_search.yaml

grade_hle_run() {
  local run="$1"
  local summary_out="$2"
  local stem grade_dir
  local gpt_final deepseek_final gpt_first deepseek_first gpt_solver deepseek_solver
  stem="$(basename "$run" .json)"
  grade_dir="runs/grades/$stem"
  gpt_final="$grade_dir/gpt54.final.json"
  deepseek_final="$grade_dir/deepseek.final.json"
  gpt_first="$grade_dir/gpt54.first.json"
  deepseek_first="$grade_dir/deepseek.first.json"
  gpt_solver="$grade_dir/gpt54.solver.json"
  deepseek_solver="$grade_dir/deepseek.solver.json"
  mkdir -p "$grade_dir" "$(dirname "$summary_out")"
  test ! -e "$summary_out"

  theoria grade "$run" --target final --missing-as-incorrect \
    --config "$GRADER_GPT54" --config "$SEARCH_CONFIG" --no-watch \
    --out "$gpt_final"
  theoria grade "$run" --target final --missing-as-incorrect \
    --config "$GRADER_DEEPSEEK" --config "$SEARCH_CONFIG" --no-watch \
    --out "$deepseek_final"
  theoria grade "$run" --target first_attempt --missing-as-incorrect \
    --config "$GRADER_GPT54" --config "$SEARCH_CONFIG" --no-watch \
    --out "$gpt_first"
  theoria grade "$run" --target first_attempt --missing-as-incorrect \
    --config "$GRADER_DEEPSEEK" --config "$SEARCH_CONFIG" --no-watch \
    --out "$deepseek_first"
  theoria grade "$run" --target solver_initial --missing-as-incorrect \
    --config "$GRADER_GPT54" --config "$SEARCH_CONFIG" --no-watch \
    --out "$gpt_solver"
  theoria grade "$run" --target solver_initial --missing-as-incorrect \
    --config "$GRADER_DEEPSEEK" --config "$SEARCH_CONFIG" --no-watch \
    --out "$deepseek_solver"

  python - "$gpt_final" "$deepseek_final" "$gpt_first" "$deepseek_first" \
    "$gpt_solver" "$deepseek_solver" <<'PY'
import json
import sys
from pathlib import Path

for grade_path in map(Path, sys.argv[1:]):
    grade = json.loads(grade_path.read_text())
    root = Path(grade["artifact_root"])
    if not root.is_absolute():
        root = Path.cwd() / root
    meta = json.loads((root / "meta.json").read_text())
    assert meta["status"] == "completed", (grade_path, meta["status"])
    assert meta["research_audit"]["model_runtime_identity_complete"] is True
PY

  python experiments/summarize_run.py "$run" \
    --final-grade "$gpt_final" --final-grade "$deepseek_final" \
    --first-grade "$gpt_first" --first-grade "$deepseek_first" \
    --solver-grade "$gpt_solver" --solver-grade "$deepseek_solver" \
    --out "$summary_out"
}
```

Choose and preregister one calibration anchor per model family, then run the
following three conditions for each candidate at that anchor, changing only
`CANDIDATE` and `PROMPT`. Freeze the resulting family-to-prompt mapping before
evaluation. The committed `qwen-v1` profile is the initial candidate, not a
preselected winner:

```bash
export MODEL=qwen3-8b
export CFG="experiments/frozen_configs/${MODEL}.yaml"
mkdir -p "$ARTIFACT_ROOT/preregistration"
export CANDIDATE_MANIFEST="$ARTIFACT_ROOT/preregistration/${MODEL}.prompt-candidates.txt"
test ! -e "$CANDIDATE_MANIFEST"
# Write one to three IDs here before running or inspecting any candidate.
printf '%s\n' qwen-v1 > "$CANDIDATE_MANIFEST"
sha256sum "$CANDIDATE_MANIFEST"

export CANDIDATE=qwen-v1
grep -Fxq "$CANDIDATE" "$CANDIDATE_MANIFEST"
export PROMPT="experiments/prompts/${CANDIDATE}.yaml"
export POINTER_DIR=runs/pointers
mkdir -p "$POINTER_DIR" runs/summaries/dev

# Outer action protocol and solver shell/search discipline.
SMOKE_PTR="$POINTER_DIR/${MODEL}.${CANDIDATE}.smoke.path"
rm -f "$SMOKE_PTR"

python experiments/run_hle_condition.py \
  --cohort smoke-5 --base-config "$CFG" \
  --config "$PROMPT" --config experiments/configs/tool_smoke.yaml \
  --experiment-id "${MODEL}-${CANDIDATE}-protocol-smoke" \
  --tag "${MODEL}-${CANDIDATE}-protocol-smoke-t1" \
  --phase smoke --trial 1 --parallel 1 --no-repair \
  --run-path-file "$SMOKE_PTR"
test -s "$SMOKE_PTR"
SMOKE_RUN="$(cat "$SMOKE_PTR")"
python experiments/summarize_run.py "$SMOKE_RUN" \
  --out "runs/summaries/dev/${MODEL}.${CANDIDATE}.smoke.json"

# Ordinary verifier quality run. This is the only dev row used for precision
# and coverage selection.
DEV_PTR="$POINTER_DIR/${MODEL}.${CANDIDATE}.quality.path"
rm -f "$DEV_PTR"
python experiments/run_hle_condition.py \
  --cohort dev-30 --base-config "$CFG" \
  --config "$PROMPT" \
  --experiment-id "${MODEL}-${CANDIDATE}-dev-quality" \
  --tag "${MODEL}-${CANDIDATE}-dev-quality-t1" \
  --phase dev --trial 1 --parallel 1 \
  --run-path-file "$DEV_PTR"
test -s "$DEV_PTR"
DEV_RUN="$(cat "$DEV_PTR")"
grade_hle_run "$DEV_RUN" \
  "runs/summaries/dev/${MODEL}.${CANDIDATE}.quality.json"

# Separate typed-judge compliance gate. Its verifier outcomes are diagnostic.
TOOLS_PTR="$POINTER_DIR/${MODEL}.${CANDIDATE}.tools.path"
rm -f "$TOOLS_PTR"
python experiments/run_hle_condition.py \
  --cohort dev-30 --base-config "$CFG" \
  --config "$PROMPT" --config experiments/configs/mechanistic_tools.yaml \
  --experiment-id "${MODEL}-${CANDIDATE}-dev-tools" \
  --tag "${MODEL}-${CANDIDATE}-dev-tools-t1" \
  --phase dev --trial 1 --parallel 1 --no-repair \
  --run-path-file "$TOOLS_PTR"
test -s "$TOOLS_PTR"
TOOLS_RUN="$(cat "$TOOLS_PTR")"
python experiments/summarize_run.py "$TOOLS_RUN" \
  --out "runs/summaries/dev/${MODEL}.${CANDIDATE}.tools.json"
```

Create prompt overlays only after examining role-level failures. Never use
`paper-ran-185`, `paper-sampled-200`, GPQA, or poison outcomes to select a
prompt. After all one-to-three candidates have completed, execute the selection
rule and seal its inputs:

```bash
python - "$MODEL" "$CANDIDATE_MANIFEST" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

model = sys.argv[1]
manifest_path = Path(sys.argv[2])
candidates = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
if not 1 <= len(candidates) <= 3 or len(candidates) != len(set(candidates)):
    raise SystemExit("candidate manifest must contain one to three unique IDs")
if any(not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", item) for item in candidates):
    raise SystemExit("invalid candidate ID in manifest")
root = Path("runs/summaries/dev")
records = []
for candidate in candidates:
    quality_path = root / f"{model}.{candidate}.quality.json"
    smoke_path = root / f"{model}.{candidate}.smoke.json"
    tools_path = root / f"{model}.{candidate}.tools.json"
    if not all(path.is_file() for path in (quality_path, smoke_path, tools_path)):
        raise SystemExit(f"missing gate summary for {candidate}")
    quality = json.loads(quality_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    tools = json.loads(tools_path.read_text())
    strict = quality["operating_points"]["r1_final"]["certified_precision"][
        "all_graders_consensus"
    ]["strict_correct"]
    smoke_tool = smoke["mechanistic_evidence"][
        "required_tool_obligation_compliance"
    ]
    judge_tool = tools["mechanistic_evidence"][
        "required_tool_obligation_compliance"
    ]
    gates = {
        "quality_problems_30": quality["cohort"]["problems"] == 30,
        "smoke_problems_5": smoke["cohort"]["problems"] == 5,
        "tools_problems_30": tools["cohort"]["problems"] == 30,
        "quality_execution_errors_zero": quality["cohort"]["execution_errors"] == 0,
        "smoke_execution_errors_zero": smoke["cohort"]["execution_errors"] == 0,
        "smoke_tool_obligations_complete": (
            smoke_tool["denominator"] > 0 and smoke_tool["value"] == 1.0
        ),
        "judge_tool_obligations_at_least_95pct": (
            judge_tool["denominator"] > 0 and judge_tool["value"] >= 0.95
        ),
        "certified_precision_defined": strict["wilson_95"] is not None,
    }
    records.append({
        "candidate": candidate,
        "eligible": all(gates.values()),
        "gates": gates,
        "strict_precision_wilson_lower": (
            strict["wilson_95"][0] if strict["wilson_95"] else None
        ),
        "coverage": quality["operating_points"]["r1_final"]["coverage"]["value"],
        "tool_gate_execution_errors": tools["cohort"]["execution_errors"],
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (quality_path, smoke_path, tools_path)
        },
    })

if not 1 <= len(records) <= 3:
    raise SystemExit(f"expected one to three candidates, found {len(records)}")
eligible = [record for record in records if record["eligible"]]
if not eligible:
    raise SystemExit("no prompt candidate passed the frozen readiness gates")
eligible.sort(key=lambda record: (
    -record["strict_precision_wilson_lower"],
    -record["coverage"],
    record["candidate"],
))
selected = eligible[0]["candidate"]
report = {
    "model": model,
    "selection_rule": "gates, then precision Wilson lower bound, then coverage, then candidate id",
    "selected": selected,
    "selected_prompt": f"experiments/prompts/{selected}.yaml",
    "candidate_manifest": str(manifest_path.resolve()),
    "candidate_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "candidates": records,
}
out = root / f"{model}.selection.json"
if out.exists():
    raise SystemExit(f"refusing to overwrite selection report: {out}")
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(report["selected_prompt"])
PY

export PROMPT="$(python -c \
  'import json,sys; print(json.load(open(f"runs/summaries/dev/{sys.argv[1]}.selection.json"))["selected_prompt"])' \
  "$MODEL")"
test -f "$PROMPT"
mkdir -p "$ARTIFACT_ROOT/preregistration"
test ! -e "$ARTIFACT_ROOT/preregistration/${MODEL}.prompt-selection.json"
cp "runs/summaries/dev/${MODEL}.selection.json" \
  "$ARTIFACT_ROOT/preregistration/${MODEL}.prompt-selection.json"
sha256sum "$ARTIFACT_ROOT/preregistration/${MODEL}.prompt-selection.json"
```

Apply that frozen family prompt to every scale in the family, then repeat only
the five-item protocol smoke block for each scale. This is a preregistered
readiness check, not a new selection opportunity: a failed scale is reported as
protocol-incompatible and its prompt is not retuned on evaluation-adjacent data.

Once selected:

```bash
git add experiments/frozen_configs experiments/prompts experiments/model_matrix.yaml
git commit -m 'Freeze open-weight evaluation configs and prompts'
test -z "$(git status --porcelain)"
git rev-parse HEAD | tee "$ARTIFACT_ROOT/preregistration/theoria-commit.txt"
sha256sum "$CFG" "$PROMPT" experiments/model_matrix.yaml \
  > "$ARTIFACT_ROOT/preregistration/${MODEL}.condition-inputs.sha256"
```

The clean-status assertion must pass for every evaluation run. Archive the exact
commit SHA and condition-input hashes in the preregistration.

## 7. Run HLE R1 and paired first-pass R0

`model_matrix.yaml` is a checkpoint catalog, not an instruction to run every
optional row. Before evaluation, preregister the exact main-model list and name
the small, medium, and frontier repeatability anchors. A defensible minimum main
spine is `qwen3-8b`, `qwen3-14b`, and `qwen3-32b`; additions must be frozen
before looking at evaluation outcomes.

```bash
cat > "$ARTIFACT_ROOT/preregistration/main-models.txt" <<'EOF'
qwen3-8b
qwen3-14b
qwen3-32b
EOF
cat > "$ARTIFACT_ROOT/preregistration/repeatability-anchors.txt" <<'EOF'
qwen3-8b
qwen3-32b
EOF
sha256sum "$ARTIFACT_ROOT/preregistration/main-models.txt" \
  "$ARTIFACT_ROOT/preregistration/repeatability-anchors.txt"
```

Every generated config defaults to request seed `20260718`, and the vLLM engine
seed is also fixed. `--trial` records a replicate label; it does not change the
inference seed. Trials 2 and 3 with the same config are therefore same-seed
repeatability checks for predeclared anchors, not independent statistical
samples. Do not pool them as `n=3`. A different-seed sensitivity analysis must
use separately frozen configs, condition names, and seed values. Keep problem
parallelism and vLLM `--max-num-seqs` at one.

The example below is trial 1 for one model. Use an atomic run-path receipt for
every condition; never recover a result with `ls -t`, which can select a stale
run after a failure.

```bash
set -euo pipefail
export THEORIA_ROOT="$(git rev-parse --show-toplevel)"
export ARTIFACT_ROOT="$THEORIA_ROOT/../theoria-publication-artifacts"
cd "$THEORIA_ROOT"
source .venv/bin/activate
export SEARCH_CONFIG=configs/searxng_search.yaml
export MODEL=qwen3-8b
export TRIAL=1
export CFG="experiments/frozen_configs/${MODEL}.yaml"
export PROMPT_SELECTION_KEY=qwen3-8b
# Re-read this from the frozen selection artifact in a fresh shell.
export PROMPT="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_prompt"])' \
  "$ARTIFACT_ROOT/preregistration/${PROMPT_SELECTION_KEY}.prompt-selection.json")"
export POINTER_DIR=runs/pointers
mkdir -p "$POINTER_DIR" runs/logs

R1_CONDITION="${MODEL}-hle185-r1"
R1_TAG="${R1_CONDITION}-t${TRIAL}"
R1_PTR="$POINTER_DIR/${R1_TAG}.path"
R1_LOG="runs/logs/${R1_TAG}.log"
rm -f "$R1_PTR"

python experiments/run_hle_condition.py \
  --cohort paper-ran-185 --base-config "$CFG" --config "$PROMPT" \
  --experiment-id "$R1_CONDITION" --tag "$R1_TAG" \
  --phase evaluation --trial "$TRIAL" --parallel 1 \
  --run-path-file "$R1_PTR" 2>&1 | tee "$R1_LOG"
test -s "$R1_PTR"
export RUN="$(cat "$R1_PTR")"
export RUN_ID="$(basename "$RUN" .json)"
```

The resulting summary's `r0_first_pass` and `r1_final` share the initial trace.
Run explicit R0 as a consistency check:

```bash
R0_CONDITION="${MODEL}-hle185-r0"
R0_TAG="${R0_CONDITION}-t${TRIAL}"
R0_PTR="$POINTER_DIR/${R0_TAG}.path"
R0_LOG="runs/logs/${R0_TAG}.log"
rm -f "$R0_PTR"

python experiments/run_hle_condition.py \
  --cohort paper-ran-185 --base-config "$CFG" --config "$PROMPT" \
  --experiment-id "$R0_CONDITION" --tag "$R0_TAG" \
  --phase evaluation --trial "$TRIAL" --parallel 1 --no-repair \
  --run-path-file "$R0_PTR" 2>&1 | tee "$R0_LOG"
test -s "$R0_PTR"
export R0_RUN="$(cat "$R0_PTR")"
```

For anchor models, run the intent-to-treat sensitivity cohort:

```bash
ITT_CONDITION="${MODEL}-hle200-r1-itt"
ITT_TAG="${ITT_CONDITION}-t${TRIAL}"
ITT_PTR="$POINTER_DIR/${ITT_TAG}.path"
ITT_LOG="runs/logs/${ITT_TAG}.log"
rm -f "$ITT_PTR"

python experiments/run_hle_condition.py \
  --cohort paper-sampled-200 --base-config "$CFG" --config "$PROMPT" \
  --experiment-id "$ITT_CONDITION" --tag "$ITT_TAG" \
  --phase evaluation --trial "$TRIAL" --parallel 1 \
  --run-path-file "$ITT_PTR" 2>&1 | tee "$ITT_LOG"
test -s "$ITT_PTR"
export ITT_RUN="$(cat "$ITT_PTR")"
```

The receipt is written only after a sealed run returns. If a process is
interrupted before that, recover the run ID from its dedicated log, then resume
with the exact original arguments, tag, trial, and pointer:

```bash
export RUN_ID="$(sed -n \
  's|.*Saving to runs/\([^ ]*\)\.json.*|\1|p' "$R1_LOG" | tail -n 1)"
test -n "$RUN_ID"

python experiments/run_hle_condition.py \
  --cohort paper-ran-185 --base-config "$CFG" --config "$PROMPT" \
  --experiment-id "$R1_CONDITION" --tag "$R1_TAG" \
  --phase evaluation --trial "$TRIAL" --parallel 1 \
  --resume "$RUN_ID" --run-path-file "$R1_PTR" \
  2>&1 | tee -a "$R1_LOG"
test -s "$R1_PTR"
export RUN="$(cat "$R1_PTR")"
```

Never resume into a changed prompt, config, model server, image, or cohort.

## 8. Grade HLE independently and summarize it

Use two fixed graders developed independently from the internal judges. Keep
grader models, prompts, search, and revisions identical for every condition.
The committed Azure configs are deployment snapshots, not portable public
endpoints; use them only if the live control-plane probe still matches. The
helper produces separate `final`, `first_attempt`, and `solver_initial` grade
runs. R0 precision must use `first_attempt` grades, never the repaired final
answer's grade. Final and first-attempt candidates use the identical frozen,
condition-neutral rubric and expose no repair status or internal trace to the
grader; only the selected candidate differs.

```bash
# Re-declare grade_hle_run from section 6 after opening a fresh shell.
mkdir -p runs/summaries/evaluation
export R1_SUMMARY="runs/summaries/evaluation/${R1_TAG}.json"
export R0_SUMMARY="runs/summaries/evaluation/${R0_TAG}.json"

grade_hle_run "$RUN" "$R1_SUMMARY"
grade_hle_run "$R0_RUN" "$R0_SUMMARY"
if [[ -n "${ITT_RUN:-}" ]]; then
  export ITT_SUMMARY="runs/summaries/evaluation/${ITT_TAG}.json"
  grade_hle_run "$ITT_RUN" "$ITT_SUMMARY"
fi

# Record, rather than assume, whether the independent explicit R0 reproduced
# the paired first-pass certification decisions from R1.
export R0_CONSISTENCY="runs/summaries/evaluation/${MODEL}-hle185-r0-consistency-t${TRIAL}.json"
test ! -e "$R0_CONSISTENCY"
python - "$RUN" "$R0_RUN" \
  "$R0_CONSISTENCY" <<'PY'
import json
import sys
from pathlib import Path

r1 = json.loads(Path(sys.argv[1]).read_text())
r0 = json.loads(Path(sys.argv[2]).read_text())
r1_by_id = {str(row["id"]): row for row in r1}
r0_by_id = {str(row["id"]): row for row in r0}
assert set(r1_by_id) == set(r0_by_id)
paired = {
    pid for pid, row in r1_by_id.items()
    if (row.get("repair_metrics") or {}).get("first_attempt_verified")
}
explicit = {pid for pid, row in r0_by_id.items() if row.get("verified")}
report = {
    "paired_r0_certified": len(paired),
    "explicit_r0_certified": len(explicit),
    "agreement": len(set(r1_by_id) - (paired ^ explicit)),
    "paired_only": sorted(paired - explicit),
    "explicit_only": sorted(explicit - paired),
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n")
PY
```

Blindly adjudicate every certified disagreement, every automatically identified
certified error, and 10% of certified agreements rounded up. Select that 10% by
sorting `SHA256("20260718:" + problem_id)` so the audit sample is independent of
file order and reproducible. Freeze the queue before giving it to reviewers and
hide automatic labels and condition names from their forms. Preserve both
automatic grades and a problem-ID-keyed adjudication table; never overwrite
them. The favorable category is a review queue, not a headline label.

Build separate hash-bound queues for R1 and paired R0 from the same sealed run:

```bash
GRADE_DIR="runs/grades/$(basename "$RUN" .json)"
FINAL_QUEUE="runs/adjudication/${R1_TAG}.final"
FIRST_QUEUE="runs/adjudication/${R1_TAG}.first"

python experiments/build_adjudication_queue.py "$RUN" --target final \
  --grade "$GRADE_DIR/gpt54.final.json" \
  --grade "$GRADE_DIR/deepseek.final.json" \
  --seed 20260718 --audit-fraction 0.10 --out-dir "$FINAL_QUEUE"

python experiments/build_adjudication_queue.py "$RUN" --target first_attempt \
  --grade "$GRADE_DIR/gpt54.first.json" \
  --grade "$GRADE_DIR/deepseek.first.json" \
  --seed 20260718 --audit-fraction 0.10 --out-dir "$FIRST_QUEUE"

sha256sum "$FINAL_QUEUE"/* "$FIRST_QUEUE"/*
cp "$FINAL_QUEUE/decisions.template.json" \
  "$FINAL_QUEUE/decisions.completed.json"
cp "$FIRST_QUEUE/decisions.template.json" \
  "$FIRST_QUEUE/decisions.completed.json"
```

Give reviewers only each `cases.json` and its decision template. Two reviewers
must assess every case independently, resolve disagreements, then set
`status="completed"`, a boolean `strict_correct`, both opaque `reviewer_ids`,
and concise reasoning in the corresponding `decisions.completed.json`. A
placeholder remains incorrect even if hidden proof text contained the answer.
Apply the completed decisions without modifying the queues:

```bash
python experiments/apply_adjudication.py "$RUN" --target final \
  --grade "$GRADE_DIR/gpt54.final.json" \
  --grade "$GRADE_DIR/deepseek.final.json" \
  --queue-dir "$FINAL_QUEUE" \
  --decisions "$FINAL_QUEUE/decisions.completed.json" \
  --out "runs/summaries/evaluation/${R1_TAG}.final.adjudicated.json"

python experiments/apply_adjudication.py "$RUN" --target first_attempt \
  --grade "$GRADE_DIR/gpt54.first.json" \
  --grade "$GRADE_DIR/deepseek.first.json" \
  --queue-dir "$FIRST_QUEUE" \
  --decisions "$FIRST_QUEUE/decisions.completed.json" \
  --out "runs/summaries/evaluation/${R1_TAG}.first.adjudicated.json"

python experiments/compare_adjudicated.py \
  --first-attempt \
    "runs/summaries/evaluation/${R1_TAG}.first.adjudicated.json" \
  --final "runs/summaries/evaluation/${R1_TAG}.final.adjudicated.json" \
  --out "runs/summaries/evaluation/${R1_TAG}.paired.adjudicated.json"
```

## 9. Run the outcome-independent GPQA cohort

Materialize benchmark text locally; never commit or re-host it:

```bash
export HF_TOKEN='<token with approved Idavidrein/gpqa access>'
python experiments/prepare_companion_benchmarks.py \
  --analysis-repo "$ARTIFACT_ROOT/theoria_analysis" \
  --out-dir runs/benchmarks --verify-hf

python -c 'from loaders import load_json_benchmark; p=load_json_benchmark("runs/benchmarks/gpqa-fixed-100.json"); assert len(p)==100'
```

Run R1 and derive paired R0 from its first pass:

```bash
GPQA_CONDITION="${MODEL}-gpqa100-r1"
GPQA_TAG="${GPQA_CONDITION}-t${TRIAL}"
GPQA_PTR="$POINTER_DIR/${GPQA_TAG}.path"
GPQA_LOG="runs/logs/${GPQA_TAG}.log"
rm -f "$GPQA_PTR"

theoria benchmark runs/benchmarks/gpqa-fixed-100.json \
  --backend theoria_agent --docker \
  --image theoria-sandbox-sage:latest \
  --config "$CFG" --config "$PROMPT" --config "$SEARCH_CONFIG" \
  --parallel 1 --no-watch \
  --experiment-id "$GPQA_CONDITION" \
  --experiment-phase evaluation --trial "$TRIAL" \
  --tag "$GPQA_TAG" --run-path-file "$GPQA_PTR" \
  2>&1 | tee "$GPQA_LOG"
test -s "$GPQA_PTR"
export GPQA_RUN="$(cat "$GPQA_PTR")"
```

Grade final multiple-choice answers without an LLM:

```bash
test ! -e "runs/summaries/evaluation/${GPQA_TAG}.mcq.json"
python experiments/summarize_gpqa.py "$GPQA_RUN" \
  --out "runs/summaries/evaluation/${GPQA_TAG}.mcq.json"
```

The MCQ report contains raw solver accuracy, paired `r0_first_pass_*`, final R1
coverage/precision, paired bootstrap intervals and exact McNemar tests, repair
transitions, certification stage, answer flips, extraction-review flags,
telemetry, execution failures, and per-domain results. The frozen deterministic
parser is the primary GPQA evaluator; unparsed responses count as incorrect.
Report extraction-review flags as a sensitivity analysis and do not alter the
headline parser after observing evaluation outputs.

Run `gpqa-legacy-paper-65.json` only for an explicitly labeled appendix
comparison. Do not compare its 33/34 directly as though it came from one frozen
stack or a pre-outcome cohort.

## 10. Mechanistic checks

`experiments/configs/mechanistic_tools.yaml` records that the configured tool
returned transport-level success before a role finalized. It does not prove
that a shell command performed the required computation, that a primary source
was fetched, or that the result was used correctly. A SearXNG snippet is a lead,
not evidence.

Therefore:

- shell-required execution is a valid tool-policy ablation, not proof of a
  relevant computation;
- search-required citation is diagnostic only;
- do not claim evidence-gated factual verification until a URL fetch tool,
  fetched-content hash, source URL, status, timestamp, and claim-to-evidence
  link are recorded and independently checked.

Run the strict diagnostic as a separately named condition:

```bash
TOOLS_CONDITION="${MODEL}-hle185-r1-tools-diagnostic"
TOOLS_TAG="${TOOLS_CONDITION}-t${TRIAL}"
TOOLS_PTR="$POINTER_DIR/${TOOLS_TAG}.path"
TOOLS_LOG="runs/logs/${TOOLS_TAG}.log"
rm -f "$TOOLS_PTR"

python experiments/run_hle_condition.py \
  --cohort paper-ran-185 --base-config "$CFG" --config "$PROMPT" \
  --config experiments/configs/mechanistic_tools.yaml \
  --experiment-id "$TOOLS_CONDITION" --tag "$TOOLS_TAG" \
  --phase ablation --trial "$TRIAL" --parallel 1 \
  --run-path-file "$TOOLS_PTR" 2>&1 | tee "$TOOLS_LOG"
test -s "$TOOLS_PTR"
export TOOLS_RUN="$(cat "$TOOLS_PTR")"

grade_hle_run "$TOOLS_RUN" \
  "runs/summaries/evaluation/${TOOLS_TAG}.json"
```

Do not merge this row with the prompt-only verifier row.

### Conditional role ablation

Trigger this only from the preregistered dev readiness rule, never after seeing
evaluation outcomes. The example assigns Qwen3-8B only to `solver` and Qwen3-32B
to the formalizer and every verifier role. Both servers must run concurrently.
Repeat section 5 explicitly for `qwen3-32b` on port 8000; its earlier example
served `qwen3-8b` and cannot be reused. Start Qwen3-8B on port 8001 with the same
captured-server recipe, and place the processes on recorded, non-overlapping
GPUs. Confirm each `/v1/models` response before composing the config.

```bash
# Server shell A: use the full section-5 capture recipe with these values.
export MODEL=qwen3-32b
export CUDA_VISIBLE_DEVICES=0,1
test ! -e "$ARTIFACT_ROOT/servers/qwen3-32b-role-ablation"
mkdir -p "$ARTIFACT_ROOT/servers/qwen3-32b-role-ablation"
python experiments/serve_vllm.py "$MODEL" --port 8000 --execute 2>&1 \
  | tee "$ARTIFACT_ROOT/servers/qwen3-32b-role-ablation/server.log"

# Server shell B: use the full section-5 capture recipe with these values.
export MODEL=qwen3-8b
export CUDA_VISIBLE_DEVICES=2
test ! -e "$ARTIFACT_ROOT/servers/qwen3-8b-role-ablation"
mkdir -p "$ARTIFACT_ROOT/servers/qwen3-8b-role-ablation"
python experiments/serve_vllm.py "$MODEL" --port 8001 --execute 2>&1 \
  | tee "$ARTIFACT_ROOT/servers/qwen3-8b-role-ablation/server.log"

# Runner shell, after both health checks pass.
curl -fsS http://127.0.0.1:8000/v1/models > \
  "$ARTIFACT_ROOT/servers/qwen3-32b-role-ablation/models.json"
curl -fsS http://127.0.0.1:8001/v1/models > \
  "$ARTIFACT_ROOT/servers/qwen3-8b-role-ablation/models.json"
export WEAK_SERVER_HARDWARE="$(nvidia-smi -i 2 \
  --query-gpu=name,memory.total --format=csv,noheader)"

python experiments/make_model_config.py qwen3-8b \
  --endpoint http://127.0.0.1:8001/v1 \
  --server-hardware "$WEAK_SERVER_HARDWARE" \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --out experiments/frozen_configs/qwen3-8b-port8001.yaml

python experiments/make_role_ablation_config.py \
  --solver-config experiments/frozen_configs/qwen3-8b-port8001.yaml \
  --verifier-config experiments/frozen_configs/qwen3-32b.yaml \
  --out experiments/frozen_configs/qwen3-8b-solver__qwen3-32b-verifier.yaml

theoria doctor --backend theoria_agent \
  --config experiments/frozen_configs/qwen3-8b-solver__qwen3-32b-verifier.yaml \
  --config configs/searxng_search.yaml \
  --docker --image theoria-sandbox-sage:latest --check-endpoint
```

Run the complete section 6 smoke, ordinary-dev, tool-gate, grading, and prompt
selection procedure with this mixed config before freezing it. Then run
sections 7-8 unchanged with
`MODEL=qwen3-8b-solver__qwen3-32b-verifier` and
`PROMPT_SELECTION_KEY=qwen3-8b-solver__qwen3-32b-verifier`; do not retain the
weak all-role selection key from the earlier shell. The role table must include the
weak all-role row, strong all-role row, and this mixed row on identical cohorts.

## 11. Poisoned-proof protocol

The frozen artifact hash is:

```text
dc8190d61642cf8054dae94376fe2f6b49ef70f53b67b18b0323ef167e44ab43
```

The paper's 90/95 versus 79/95 sensitivity comparison omits a critical
specificity result: structured raw judges reject 12/15 generated controls,
while the holistic judge rejects 1/15. The labels were model-generated without
an independent validation pass.

Before running open-weight models:

1. Have two blinded domain reviewers independently label all 110 items for
   proof validity, planted-error uniqueness, planted step, and control validity.
2. Resolve disagreements without showing historical judge outcomes.
3. Freeze the adjudicated manifest and its SHA-256.
4. Implement an audited direct-proof runner in this repository that captures
   the same model/config/sandbox/call evidence as normal runs.
5. Run both raw typed judges and the complete judge/pedantry/convention stack.
6. Report poison sensitivity, control specificity, balanced accuracy, paired
   McNemar intervals, and results by error type.

The current repository has no publication-grade direct-proof runner. Do not
substitute the analysis branch's `run_evaluation.py`: it hardcodes GPT-5.5,
uses a different API loop, and would change the experimental object.

## 12. R2 and R3 implementation gate

Do not label the current repair loop R2. It lacks structured objections,
objection validation, preserved evidence references, localized changed-step
records, blind independent re-verification, and a matched-budget fresh-resample
fork.

Before any R2/R3 result, add and test:

- immutable initial solver/formalizer/judge trace export and replay;
- objection schema with type, affected step, required evidence, evidence IDs,
  severity, and confidence;
- fetched-source validation for factual/citation objections;
- sandbox execution validation for computational objections;
- format-only, formalization-only, local-logic, computation, factual, and
  solver-level routing;
- changed step IDs, changed claims, exact and normalized answer hashes;
- fresh independent full-verifier calls after repair;
- R3 resampling from the same initial trace with matched calls and tokens;
- explicit outcomes: certified, declined, invalid objection, exhausted, failed.

Until those tests exist, publish R0 versus R1 only and describe R2/R3 as future
work or a preregistered follow-up.

## 13. Baselines and statistics

Solver-only is supported by `--target solver_initial`. The holistic judge,
explicit-abstention judge, deterministic matched-coverage tie rule, and
overlap/ensemble analysis are not yet implemented for new model traces. Do not
reuse historical confidence files: they used different solvers.

Required implementation before a matched-coverage claim:

- score exactly the same initial solver response used by Theoria;
- choose thresholds on dev only;
- on evaluation, include all items above threshold and resolve cutoff ties by a
  preregistered hash of problem ID, not completion order;
- report the full precision-coverage curve and the point matched to Theoria;
- run an explicit-abstention prompt as a separate baseline;
- use paired bootstrap intervals and McNemar tests on shared items;
- apply Holm correction across the predeclared model comparisons.

For every main table report counts before percentages:

- ran/requested and execution failures;
- certified/ran coverage with Wilson 95% CI;
- strict-correct/certified precision with Wilson 95% CI;
- certified wrong count;
- solver correct/ran;
- declined wrong rate and certified/declined asymmetry;
- R0 and R1, repair attempts, yield, precision among repair certifications,
  answer flips, semantic-repair attempts, nonsemantic retry certifications,
  solver retries, judge rounds, and formalizer rejects/invalids;
- complete-case and missing-error repair-yield bounds, plus rows with missing
  repair telemetry;
- JSON-action, role-schema, unknown-action, and required-tool policy reprompts;
- per-domain counts, coverage, and precision;
- calls, tokens, latency, and missing-usage fraction, with cost secondary.

The problem is the statistical unit. Provider retries are not independent
samples. Report trial-to-trial variability separately from binomial uncertainty.

### Terminus/Harbor scope

The main tables must use Theoria's provider-neutral action loop. Terminus/Harbor
adds tmux state, command batching, parser recovery, summarization, and completion
semantics, so it changes the evaluated system. It is eligible only as a
separately frozen appendix condition named "generic terminal agent with the same
model." Pin both repository commits and the terminal image, match the model and
token budget, and do not describe that row as canonical Theoria. This repository
does not currently contain an audited Terminus adapter, so no such result is
publication-eligible yet.

## 14. Execution order and stop rules

1. Complete artifact audit and human poison validation.
2. Smoke 4B, 14B, and the strongest available model on five HLE items.
3. Calibrate at most three prompt candidates per family on `dev-30`.
4. Freeze Git commit, prompts, configs, model revisions, images, graders, and
   analysis plan.
5. Run all-role Qwen R1 on HLE-185; derive paired first-pass R0.
6. Run explicit R0 validation.
7. Add DeepSeek, Mistral, and gpt-oss only after protocol success is acceptable.
8. Run GPQA-fixed-100 on the predeclared anchor models.
9. Run poison+controls only after the direct-proof runner and label audit exist.
10. Run role ablations only if all-role OSS fails a preregistered readiness bar.
11. Add Terminus/Harbor only as a generic-terminal-agent appendix.

Stop a model before full evaluation if smoke has any silent missing output,
unsealed artifact, wrong model identity, unresolved placeholder certification,
or systematic action-protocol failure. Do not stop merely for low accuracy;
that is a result.

## 15. Release bundle

Release, subject to benchmark licenses and gating terms:

```text
git commit SHA and clean-status proof
model_matrix.yaml and every frozen config/prompt
model server command/version, dependency freeze, runtime descriptor, and
checkpoint file-hash manifest
sandbox OCI hash and inspect output
source_audit.json and ID/index-only cohort manifests
run meta.json, artifact manifests, telemetry, and sealed result JSONs
separate grader runs and blinded adjudication table
machine-readable summaries and analysis code
README with exclusions, failures, and denominator policy
```

Do not redistribute GPQA question text. HLE-Verified has no explicit dataset
license of its own; confirm redistribution rights rather than relying only on
the inference that unchanged Gold rows inherit HLE's MIT license.

## 16. Paper tables and claim discipline

Build the paper from the sealed JSON counts, not copied percentages:

1. Model matrix: exact checkpoint revision, license, serving backend/version,
   dtype/quantization, context, sampling, prompt hash, role assignment, GPU.
2. Benchmark summary: solver accuracy, R0 and R1 coverage, strict precision,
   Wilson intervals, errors shipped, execution failures, and paired changes.
3. Repair table: attempts, yield, repaired precision, correctness transitions,
   answer flips, retries, formalizer failures, and certification stage.
4. HLE domain table: denominator, R0/R1 coverage, strict precision, errors.
5. Role ablation table: weak all-role, strong all-role, and mixed-role only when
   the preregistered trigger fired.
6. Mechanistic table: obligations, transport-level compliance, tool failures,
   result counts, and latency; do not call snippets fetched evidence.
7. Poison table: sensitivity, specificity, balanced accuracy, and error type,
   only after label validation and the direct-proof runner gate.

The defensible primary claim is selective precision at retained coverage for a
frozen open-weight stack. Do not claim an exact paper replication, evidence-
gated repair, matched-coverage superiority, or poison robustness until the
corresponding gates above are complete.
