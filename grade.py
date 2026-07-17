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
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import harness
from llm import artifact_dir, call_log, effective_settings, llm as _llm_call
from loaders import HLE_DATASET_NAME, HLE_DATASET_REVISION, HLE_DATASET_SPLIT
from pipeline import load_config


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
    """Return the grader system prompt and its full SHA-256."""
    text = PROMPT_PATH.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
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


def fetch_rationales(
    ids: set[str],
    revision: str = HLE_DATASET_REVISION,
    *,
    include_provenance: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], dict]:
    """Pull the canonical HLE rationale (and HLE's reviewer validity flags)
    for the given problem ids from the `skylenage/HLE-Verified` dataset, so
    the grader sees the same context the internal audit grader had.

    Returns {id: {"rationale", "is_valid", "error_type"}}. Degrades to {} if
    `datasets` isn't installed or the fetch fails (e.g. a custom-only run, or
    offline) — the grader then just sees no rationale, which is correct for
    questions that have none.
    """
    if not ids:
        empty_provenance = {
            "name": HLE_DATASET_NAME,
            "split": HLE_DATASET_SPLIT,
            "revision": revision,
            "fingerprint": None,
            "available": False,
            "records": 0,
        }
        return ({}, empty_provenance) if include_provenance else {}
    try:
        from datasets import load_dataset
        ds = load_dataset(
            HLE_DATASET_NAME,
            split=HLE_DATASET_SPLIT,
            revision=revision,
        )
    except Exception:
        return ({}, {
            "name": HLE_DATASET_NAME,
            "split": HLE_DATASET_SPLIT,
            "revision": revision,
            "fingerprint": None,
            "available": False,
        }) if include_provenance else {}
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
    provenance = {
        "name": HLE_DATASET_NAME,
        "split": HLE_DATASET_SPLIT,
        "revision": revision,
        "fingerprint": getattr(ds, "_fingerprint", None),
        "available": True,
        "records": len(out),
    }
    return (out, provenance) if include_provenance else out


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
    *,
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


def _grade_call_metrics(calls: list[dict], duration_ms: int) -> dict:
    calls = [call for call in calls if call is not None]
    input_tokens = sum(int(call.get("input_tokens", 0) or 0) for call in calls)
    output_tokens = sum(int(call.get("output_tokens", 0) or 0) for call in calls)
    failed = sum(bool(call.get("failed")) for call in calls)
    usage_observed = sum(bool(call.get("usage_observed")) for call in calls)
    usage_complete = sum(bool(call.get("usage_complete")) for call in calls)
    return {
        "duration_ms": duration_ms,
        "num_calls": len(calls),
        "successful_calls": len(calls) - failed,
        "failed_calls": failed,
        "total_retries": sum(
            int(call.get("retry_count", 0) or 0) for call in calls
        ),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_available_calls": usage_observed,
        "usage_coverage_fraction": (
            usage_observed / len(calls) if calls else None
        ),
        "usage_complete_calls": usage_complete,
        "usage_complete_fraction": (
            usage_complete / len(calls) if calls else None
        ),
        "calls_by_model": dict(Counter(
            f"{call.get('backend', '?')}:{call.get('model') or '?'}"
            for call in calls
        )),
    }


async def grade_run(
    run_file: str,
    config_paths: list[str] | None = None,
    out_path: str | None = None,
    *,
    watch: bool = True,
) -> dict:
    """Grade every problem in a run JSON. Writes a grades JSON and returns a
    summary dict. Defaults the output to runs/grades/<run-stem>.json."""
    with open(run_file) as f:
        results = json.load(f)
    if isinstance(results, dict):
        results = [results]

    resolved_config_paths = list(config_paths or [DEFAULT_GRADER_CONFIG])
    config = load_config(resolved_config_paths)
    if "audit_grader" not in config:
        raise SystemExit(
            "No 'audit_grader' role in the loaded config. Pass "
            "--config configs/audit_grader.yaml."
        )
    prompt_text, prompt_sha = load_prompt()
    settings = effective_settings(
        config["audit_grader"],
        web_search_config=config.get("_web_search"),
    )
    grader_model = (
        f"{settings.get('backend', 'claude')}:"
        f"{settings.get('model')}:{settings.get('effort')}"
    )

    if out_path is None:
        Path("runs/grades").mkdir(parents=True, exist_ok=True)
        out_path = f"runs/grades/{Path(run_file).stem}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()
    grade_run_id = (
        f"grade_{Path(run_file).stem}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    grade_artifact_root = Path("runs/artifacts") / grade_run_id
    grade_artifact_root.mkdir(parents=True, exist_ok=False)
    audit = harness._research_audit_metadata(config, {
        "config_paths": resolved_config_paths,
        "backend": settings.get("backend"),
        "watch": watch,
        "docker": False,
        "image": None,
    })
    source_run_hash = harness._file_sha256(run_file)
    source_meta_path = (
        Path("runs/artifacts") / Path(run_file).stem / "meta.json"
    )
    meta = {
        "audit_schema_version": harness.AUDIT_SCHEMA_VERSION,
        "run_id": grade_run_id,
        "kind": "audit_grading",
        "status": "running",
        "started_at": started_at,
        "source_run_file": run_file,
        "source_run_sha256": source_run_hash,
        "source_run_meta_path": (
            str(source_meta_path) if source_meta_path.exists() else None
        ),
        "source_run_meta_sha256": (
            harness._file_sha256(str(source_meta_path))
            if source_meta_path.exists() else None
        ),
        "output_path": out_path,
        "grader_model": grader_model,
        "grader_prompt_sha256": prompt_sha,
        "grader_schema_sha256": harness._sha256_json(GRADE_SCHEMA),
        "evaluation_policy": {
            "automatic_grader": True,
            "manual_adjudication_included": False,
            "automatic_grade_is_authoritative": False,
        },
        "config": harness._redact_config(config),
        "config_sha256": audit["config_sha256"],
        "research_audit": audit,
        "git": harness._safe_git_state(
            str(grade_artifact_root), "grading",
        ),
        "host_runtime": harness._host_runtime_metadata(),
        "host_environment": harness._capture_host_environment(
            str(grade_artifact_root), "grading",
        ),
    }
    harness._write_meta(str(grade_artifact_root), meta)
    (grade_artifact_root / "grader_prompt.txt").write_text(prompt_text)
    (grade_artifact_root / "grader_schema.json").write_text(
        json.dumps(GRADE_SCHEMA, indent=2)
    )

    # Fetch canonical HLE rationales for any benchmark problems in this run
    # (no-op for custom questions, which aren't in the dataset).
    dataset_revisions = {
        result.get("dataset_revision") for result in results
        if result.get("dataset_revision")
    }
    if len(dataset_revisions) > 1:
        error = RuntimeError(
            "source run mixes HLE dataset revisions; grade each revision "
            "separately so rationale provenance remains unambiguous"
        )
        meta.update({
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "run_error": {"type": type(error).__name__, "message": str(error)},
        })
        harness._write_meta(str(grade_artifact_root), meta)
        harness.write_artifact_manifest(
            str(grade_artifact_root), external_paths=[run_file],
        )
        raise error
    rationale_revision = (
        next(iter(dataset_revisions))
        if dataset_revisions else HLE_DATASET_REVISION
    )
    fetched = fetch_rationales(
        {r.get("id") for r in results if r.get("id")},
        revision=rationale_revision,
        include_provenance=True,
    )
    if isinstance(fetched, tuple):
        rationales, rationale_provenance = fetched
    else:
        rationales, rationale_provenance = fetched, {}
    rationale_path = grade_artifact_root / "rationales.json"
    rationale_path.write_text(json.dumps(rationales, indent=2, default=str))
    rationale_provenance["records_sha256"] = harness._sha256_json(rationales)
    harness.update_artifact_root_meta(
        str(grade_artifact_root),
        {"rationale_dataset": rationale_provenance},
        strict=True,
    )
    if rationales:
        print(f"Loaded {len(rationales)} HLE rationale(s) for grader context.")

    graded = []
    all_calls: list[dict] = []
    n_match = 0
    run_error: BaseException | None = None
    completed = 0
    try:
      for i, result in enumerate(results, start=1):
        pid = result.get("id", f"#{i}")
        print(f"[{i}/{len(results)}] grading {pid}...", flush=True)
        safe_pid = re.sub(r"[^\w.-]", "_", str(pid))
        problem_artifact_dir = grade_artifact_root / safe_pid
        problem_artifact_dir.mkdir(parents=True, exist_ok=True)
        calls: list[dict] = []
        call_token = call_log.set(calls)
        artifact_token = artifact_dir.set(str(problem_artifact_dir))
        problem_started = time.perf_counter()
        verdict = None
        error = None
        try:
            verdict = await grade_one(
                result, config, prompt_text, rationales.get(pid),
                watch=watch,
            )
        except Exception as e:
            error = e
            (problem_artifact_dir / "traceback.txt").write_text(
                traceback.format_exc()
            )
        finally:
            call_log.reset(call_token)
            artifact_dir.reset(artifact_token)
        calls = [call for call in calls if call is not None]
        duration_ms = int(round(
            (time.perf_counter() - problem_started) * 1000
        ))
        metrics = _grade_call_metrics(calls, duration_ms)
        all_calls.extend(calls)
        completed += 1
        if error is not None:
            print(f"  ! failed: {error}")
            graded.append({
                "id": pid,
                "error_type": type(error).__name__,
                "error": str(error),
                "calls": calls,
                "metrics": metrics,
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
            "metrics": metrics,
            **verdict,
        })
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        run_metrics = _grade_call_metrics(
            all_calls,
            int(round((time.perf_counter() - started_perf) * 1000)),
        )
        run_metrics.update({
            "requested_grades": len(results),
            "completed_grades": completed,
            "successful_grades": sum(
                1 for item in graded if "error" not in item
            ),
        })
        status = (
            "completed" if run_error is None and completed == len(results)
            else "interrupted" if isinstance(
                run_error, (asyncio.CancelledError, KeyboardInterrupt)
            )
            else "failed"
        )
        final_meta = harness._read_meta(str(grade_artifact_root))
        final_meta.update({
            "status": status,
            "finished_at": finished_at,
            "metrics": run_metrics,
            "completed_problem_ids": [str(item.get("id")) for item in graded],
        })
        if run_error is not None:
            final_meta["run_error"] = {
                "type": type(run_error).__name__,
                "message": str(run_error),
            }
        harness._write_meta(str(grade_artifact_root), final_meta)

    summary = {
        "run_file": run_file,
        "run_file_sha256": source_run_hash,
        "graded": len(results),
        "key_match": n_match,
        "grader_model": grader_model,
        "grader_prompt_sha": prompt_sha,
        "artifact_root": str(grade_artifact_root),
        "rationale_dataset": rationale_provenance,
        "evaluation_policy": meta["evaluation_policy"],
        "metrics": run_metrics,
        "results": graded,
    }
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_out, out_path)
    harness.write_artifact_manifest(
        str(grade_artifact_root),
        external_paths=[out_path, run_file],
    )

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
