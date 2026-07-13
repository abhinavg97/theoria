"""LLM-based answer grader.

The pipeline's built-in `correct` flag is a naive substring match — fine for
a quick scan. This grader is a stronger (LLM-judge) check: it
shows an LLM grader the problem, the expected answer, the system's final
answer, and every attempt, and asks whether they conceptually match
(handling equivalent notations, algebraic forms, unordered sets, etc.).

It is a faithful port of the internal audit grader, decoupled from the audit
database: it reads a run JSON directly instead of SQLite. Same prompt
(grader_prompt.md), same structured output (key_match + dispute_category).

Output per problem:
    final.key_match         — did the shipped answer match the key?
    final.dispute_category  — if not a plain match, why (convention,
                              interpretation, tighter_bound, edge_case,
                              extraction, pipeline_drift, other, none)
    attempts[].key_match    — same judgment per individual attempt
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from llm import (
    artifact_dir,
    call_log,
    llm as _llm_call,
    telemetry_context,
)
from pipeline import load_config
from telemetry import aggregate_calls


# ── Structured output schema (identical to the internal audit grader) ──

ATTEMPT_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "attempt_index": {"type": "integer"},
        "key_match": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["attempt_index", "key_match", "reasoning"],
}

FINAL_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_match": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "dispute_category": {
            "type": "string",
            "enum": [
                "none", "convention", "interpretation", "tighter_bound",
                "edge_case", "extraction", "pipeline_drift", "other",
            ],
        },
    },
    "required": ["key_match", "reasoning", "dispute_category"],
}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "final": FINAL_GRADE_SCHEMA,
        "attempts": {"type": "array", "items": ATTEMPT_GRADE_SCHEMA},
    },
    "required": ["final", "attempts"],
}

PROMPT_PATH = Path(__file__).parent / "grader_prompt.md"
DEFAULT_GRADER_CONFIG = str(Path(__file__).parent / "configs" / "audit_grader.yaml")


def load_prompt() -> tuple[str, str]:
    """Return (grader system prompt, its short sha256 for provenance)."""
    text = PROMPT_PATH.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    return text, sha


# ── Turn one run-JSON result into the grader's text input ─────────

def _attempt_summary(attempt: dict) -> tuple[str, str]:
    """Reduce one attempt to (terminal_state, justification) for the grader.
    Shows the full terminal state (see format_input for how this differs from
    the internal audit grader)."""
    if attempt.get("phase") == "formalizer_reject":
        return "(formalizer rejected the solution)", attempt.get("reject_reason", "")
    proof = attempt.get("proof") or {}
    steps = proof.get("steps") or []
    if steps:
        last = steps[-1]
        return str(last.get("state")), str(last.get("justification", ""))
    return "(no steps)", ""


def fetch_rationales(ids: set[str]) -> dict[str, dict]:
    """Pull the canonical HLE rationale (and HLE's reviewer validity flags)
    for the given problem ids from the `skylenage/HLE-Verified` dataset, so
    the grader sees the same context the internal audit grader had.

    Returns {id: {"rationale", "is_valid", "error_type"}}. Degrades to {} if
    `datasets` isn't installed or the fetch fails (e.g. a custom-only run, or
    offline) — the grader then just sees no rationale, which is correct for
    questions that have none.
    """
    if not ids:
        return {}
    try:
        from datasets import load_dataset
        ds = load_dataset("skylenage/HLE-Verified", split="train")
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for ex in ds:
        if ex["id"] not in ids:
            continue
        try:
            meta = json.loads(ex["json"])
        except Exception:
            meta = {}
        out[ex["id"]] = {
            "rationale": meta.get("rationale") or "",
            "is_valid": ex.get("rationale_is_valid"),
            "error_type": ex.get("rationale_error_type"),
        }
    return out


def format_input(result: dict, rationale: dict | None = None) -> str:
    """Format the per-problem context that goes to the grader.

    Follows the internal audit grader's layout (problem / expected /
    rationale / final answer / attempts). One intentional difference: each
    attempt shows its full terminal state (the internal grader saw only the
    final answer slot), so reproduced grades may differ marginally on
    borderline cases. The HLE rationale block is fetched separately; for
    custom questions it renders as "(no rationale available)".
    """
    rat_text = (rationale or {}).get("rationale") or "(no rationale available)"
    riv = (rationale or {}).get("is_valid")
    ret = (rationale or {}).get("error_type")
    rationale_flags = ""
    if riv is not None or ret is not None:
        rationale_flags = (
            f"(HLE's own reviewers flagged: "
            f"rationale_is_valid={riv!r} rationale_error_type={ret!r})\n"
        )

    parts = [
        "PROBLEM TEXT:",
        result.get("problem") or "(missing)",
        "",
        "EXPECTED ANSWER:",
        result.get("expected") or "(missing)",
        "",
        "HLE RATIONALE:",
        rationale_flags + rat_text,
        "",
        "SYSTEM FINAL ANSWER:",
        result.get("answer") or "(no final answer)",
        "",
        "SYSTEM ATTEMPTS:",
    ]
    attempts = result.get("attempts") or []
    if not attempts:
        parts.append("(no attempts recorded)")
    else:
        for i, a in enumerate(attempts):
            phase = a.get("phase") or "?"
            all_ok = a.get("all_ok")
            state, just = _attempt_summary(a)
            parts.append(f"--- attempt {i} (phase={phase}, all_ok={all_ok}) ---")
            parts.append(f"  state: {state}")
            parts.append(f"  justification: {just}")
    return "\n".join(parts)


# ── Grading ───────────────────────────────────────────────────────

async def grade_one(
    result: dict, config: dict, prompt_text: str, rationale: dict | None = None,
    watch: bool = True,
) -> dict:
    """Grade one problem result. Returns the parsed structured verdict."""
    user_prompt = format_input(result, rationale)
    response, _session = await _llm_call(
        user_prompt,
        role="audit_grader",
        schema=GRADE_SCHEMA,
        system=prompt_text,
        config=config,
        watch=watch,
    )
    if not isinstance(response, dict):
        raise RuntimeError(
            f"grader returned {type(response).__name__}, expected dict"
        )
    return response


async def grade_run(
    run_file: str,
    config_paths: list[str] | None = None,
    out_path: str | None = None,
    *,
    watch: bool = True,
    pricing_path: str | None = None,
) -> dict:
    """Grade every problem in a run JSON. Writes a grades JSON and returns a
    summary dict. Defaults the output to runs/grades/<run-stem>.json."""
    results = json.load(open(run_file))
    if isinstance(results, dict):
        results = [results]

    config = load_config(config_paths or [DEFAULT_GRADER_CONFIG])
    if pricing_path:
        config.setdefault("_telemetry", {})["pricing_file"] = pricing_path
    if "audit_grader" not in config:
        raise SystemExit(
            "No 'audit_grader' role in the loaded config. Pass "
            "--config configs/audit_grader.yaml."
        )
    prompt_text, prompt_sha = load_prompt()
    settings = config["audit_grader"]
    grader_model = (
        f"{settings.get('backend', 'claude')}:"
        f"{settings.get('model')}:{settings.get('effort')}"
    )
    grade_started_at = datetime.now(timezone.utc).isoformat()
    grade_started_perf = time.perf_counter()
    grade_run_id = (
        f"grade_{Path(run_file).stem}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    grade_artifact_root = Path("runs/artifacts") / grade_run_id
    grade_artifact_root.mkdir(parents=True, exist_ok=True)
    (grade_artifact_root / "meta.json").write_text(json.dumps({
        "run_id": grade_run_id,
        "kind": "audit_grading",
        "source_run_file": run_file,
        "started_at": grade_started_at,
        "grader_model": grader_model,
        "grader_prompt_sha256": hashlib.sha256(
            prompt_text.encode()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest(),
        "config": config,
    }, indent=2, default=str))

    # Fetch canonical HLE rationales for any benchmark problems in this run
    # (no-op for custom questions, which aren't in the dataset).
    rationales = fetch_rationales({r.get("id") for r in results if r.get("id")})
    if rationales:
        print(f"Loaded {len(rationales)} HLE rationale(s) for grader context.")

    graded = []
    all_calls: list[dict] = []
    n_match = 0
    for i, result in enumerate(results, start=1):
        pid = result.get("id", f"#{i}")
        print(f"[{i}/{len(results)}] grading {pid}...", flush=True)
        safe_pid = re.sub(r"[^\w.-]", "_", str(pid))
        problem_artifact_dir = grade_artifact_root / safe_pid
        problem_artifact_dir.mkdir(parents=True, exist_ok=True)
        calls: list[dict] = []
        call_token = call_log.set(calls)
        artifact_token = artifact_dir.set(str(problem_artifact_dir))
        trace_token = telemetry_context.set({
            "run_id": grade_run_id,
            "problem_id": str(pid),
        })
        problem_started_at = datetime.now(timezone.utc).isoformat()
        problem_started_perf = time.perf_counter()
        error: Exception | None = None
        verdict = None
        try:
            verdict = await grade_one(
                result, config, prompt_text, rationales.get(pid), watch=watch,
            )
        except Exception as e:
            error = e
        finally:
            call_log.reset(call_token)
            artifact_dir.reset(artifact_token)
            telemetry_context.reset(trace_token)
        problem_metrics = aggregate_calls(
            calls,
            problem_started_at=problem_started_at,
            problem_ended_at=datetime.now(timezone.utc).isoformat(),
            problem_duration_ms=int(round(
                (time.perf_counter() - problem_started_perf) * 1000
            )),
        )
        all_calls.extend(calls)
        if error is not None:
            e = error
            print(f"  ! failed: {e}")
            graded.append({
                "id": pid,
                "error": str(e),
                "calls": calls,
                "metrics": problem_metrics,
            })
            continue
        assert verdict is not None
        final = verdict["final"]
        n_match += 1 if final["key_match"] else 0
        print(
            f"  key_match={final['key_match']} "
            f"dispute={final.get('dispute_category')!r}"
        )
        graded.append({
            "id": pid,
            "expected": result.get("expected"),
            "answer": result.get("answer"),
            "verified": result.get("verified"),
            "grader_model": grader_model,
            "grader_prompt_sha": prompt_sha,
            "calls": calls,
            "metrics": problem_metrics,
            **verdict,
        })

    summary = {
        "run_file": run_file,
        "graded": len(results),
        "key_match": n_match,
        "grader_model": grader_model,
        "grader_prompt_sha": prompt_sha,
        "artifact_root": str(grade_artifact_root),
        "results": graded,
    }

    if out_path is None:
        Path("runs/grades").mkdir(parents=True, exist_ok=True)
        out_path = f"runs/grades/{Path(run_file).stem}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    grade_ended_at = datetime.now(timezone.utc).isoformat()
    run_metrics = aggregate_calls(
        all_calls,
        problem_started_at=grade_started_at,
        problem_ended_at=grade_ended_at,
        problem_duration_ms=int(round(
            (time.perf_counter() - grade_started_perf) * 1000
        )),
    )
    run_metrics.update({
        "scope": "grading_run",
        "graded_problems": len(results),
        "successful_grades": len(results) - sum(
            1 for item in graded if item.get("error")
        ),
    })
    (grade_artifact_root / "telemetry.json").write_text(
        json.dumps(run_metrics, indent=2, default=str)
    )
    meta_path = grade_artifact_root / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update({
        "finished_at": grade_ended_at,
        "telemetry_path": str(grade_artifact_root / "telemetry.json"),
        "output_path": out_path,
    })
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    summary["metrics"] = run_metrics
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_out, out_path)

    print(
        f"\nGraded {len(results)} problems: {n_match} matched the key. "
        f"Wrote {out_path}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grade a run JSON.")
    parser.add_argument("run_file", help="Path to a run JSON")
    parser.add_argument("--config", action="append", default=None,
                        help="Grader config YAML (repeatable). "
                             f"Default: {DEFAULT_GRADER_CONFIG}")
    parser.add_argument("--out", default=None, help="Output grades JSON path")
    args = parser.parse_args()
    asyncio.run(grade_run(args.run_file, args.config, args.out))
