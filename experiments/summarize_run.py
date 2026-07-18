#!/usr/bin/env python3
"""Build a reproducible paper-metric summary from one sealed Theoria run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import harness
from grade import FINAL_PROMPT_PATH, SOLVER_PROMPT_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return [center - half, center + half]


def _rate(k: int, n: int) -> dict:
    return {
        "numerator": k,
        "denominator": n,
        "value": k / n if n else None,
        "wilson_95": wilson_interval(k, n),
    }


def _strict_correct(grade_result: dict) -> bool:
    final = grade_result.get("final") or {}
    return bool(
        final.get("key_match")
        or final.get("dispute_category") == "extraction"
    )


def _key_match(grade_result: dict) -> bool:
    return bool((grade_result.get("final") or {}).get("key_match"))


def _favorable_correct(grade_result: dict) -> bool:
    final = grade_result.get("final") or {}
    return _strict_correct(grade_result) or final.get("dispute_category") in {
        "convention", "interpretation", "tighter_bound", "edge_case", "other",
    }


def _load_sealed_run(path: Path) -> tuple[list[dict], dict, str]:
    path = path.resolve(strict=True)
    results = json.loads(path.read_text())
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not all(isinstance(row, dict) for row in results):
        raise RuntimeError("run JSON must contain an object or list of objects")
    root = path.parent / "artifacts" / path.stem
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise RuntimeError(f"source run is not complete: {meta.get('status')!r}")
    manifest = harness.verify_artifact_manifest(str(root))
    source_sha = _sha256(path)
    matching_entries = [
        entry for entry in manifest.get("entries") or []
        if entry.get("external")
        and Path(entry.get("path", "")).resolve() == path
    ]
    if len(matching_entries) != 1:
        raise RuntimeError(f"source run file is not uniquely sealed: {path}")
    if matching_entries[0].get("sha256") != source_sha:
        raise RuntimeError(f"source run hash does not match its seal: {path}")
    ids = [str(row.get("id")) for row in results]
    problem_set = meta.get("problem_set") or {}
    if problem_set.get("ids") != ids:
        raise RuntimeError("source result order does not match its problem manifest")
    if (meta.get("research_audit") or {}).get(
        "model_runtime_identity_complete"
    ) is not True:
        raise RuntimeError("source model runtime identity is incomplete")
    return results, meta, source_sha


def _load_grades(
    paths: list[Path], source_sha256: str, target: str, expected_ids: set[str],
) -> dict[str, dict[str, dict]]:
    graders: dict[str, dict[str, dict]] = {}
    for path in paths:
        path = path.resolve(strict=True)
        summary = json.loads(path.read_text())
        actual_target = summary.get("grading_target") or "final"
        if actual_target != target:
            raise RuntimeError(
                f"grade target mismatch for {path}: {actual_target!r} != {target!r}"
            )
        if summary.get("run_file_sha256") != source_sha256:
            raise RuntimeError(f"grade source hash mismatch: {path}")
        artifact_root = Path(summary["artifact_root"])
        if not artifact_root.is_absolute():
            artifact_root = (path.parent.parent.parent / artifact_root).resolve()
            if not artifact_root.exists():
                artifact_root = Path(summary["artifact_root"]).resolve()
        manifest = harness.verify_artifact_manifest(str(artifact_root))
        meta = json.loads((artifact_root / "meta.json").read_text())
        if meta.get("status") != "completed":
            raise RuntimeError(f"grade run is not complete: {path}")
        matching_entries = [
            entry for entry in manifest.get("entries") or []
            if entry.get("external")
            and Path(entry.get("path", "")).resolve() == path
        ]
        if len(matching_entries) != 1:
            raise RuntimeError(f"grade file is not uniquely sealed: {path}")
        if matching_entries[0].get("sha256") != _sha256(path):
            raise RuntimeError(f"grade file hash does not match its seal: {path}")
        expected_prompt_path = (
            FINAL_PROMPT_PATH if target == "final" else SOLVER_PROMPT_PATH
        )
        expected_prompt_sha = _sha256(expected_prompt_path)
        if summary.get("grader_prompt_sha") != expected_prompt_sha:
            raise RuntimeError(f"unexpected grader prompt for {target}: {path}")
        if meta.get("grader_prompt_sha256") != expected_prompt_sha:
            raise RuntimeError(f"grade metadata prompt mismatch: {path}")
        rationale = summary.get("rationale_dataset") or {}
        if expected_ids and (
            rationale.get("available") is not True
            or int(rationale.get("records", 0) or 0) != len(expected_ids)
        ):
            raise RuntimeError(f"grade rationale coverage is incomplete: {path}")
        grader = str(summary.get("grader_model") or path.stem)
        if grader in graders:
            raise RuntimeError(f"duplicate grader for {target}: {grader}")
        rows = summary.get("results") or []
        if any(row.get("error") for row in rows):
            raise RuntimeError(f"grade run contains failed problems: {path}")
        by_id = {str(row.get("id")): row for row in rows}
        if set(by_id) != expected_ids:
            raise RuntimeError(f"grade ids do not match source run: {path}")
        graders[grader] = by_id
    return graders


def _graded_metric(
    graders: dict[str, dict[str, dict]], ids: list[str], eligible: set[str],
    *,
    solver_only: bool = False,
) -> dict:
    output: dict[str, dict] = {}
    selected = [pid for pid in ids if pid in eligible]
    for grader, rows in graders.items():
        exact = sum(_key_match(rows[pid]) for pid in selected)
        strict = exact if solver_only else sum(
            _strict_correct(rows[pid]) for pid in selected
        )
        favorable = strict if solver_only else sum(
            _favorable_correct(rows[pid]) for pid in selected
        )
        output[grader] = {
            "surface_key_match": _rate(exact, len(selected)),
            "strict_correct": _rate(strict, len(selected)),
            "favorable_unadjudicated": _rate(favorable, len(selected)),
            "strict_errors": len(selected) - strict,
        }
    if graders:
        consensus_exact = sum(
            all(_key_match(rows[pid]) for rows in graders.values())
            for pid in selected
        )
        consensus_strict = consensus_exact if solver_only else sum(
            all(_strict_correct(rows[pid]) for rows in graders.values())
            for pid in selected
        )
        favorable_any = consensus_strict if solver_only else sum(
            any(_favorable_correct(rows[pid]) for rows in graders.values())
            for pid in selected
        )
        output["all_graders_consensus"] = {
            "surface_key_match": _rate(consensus_exact, len(selected)),
            "strict_correct": _rate(consensus_strict, len(selected)),
            "favorable_unadjudicated": _rate(favorable_any, len(selected)),
            "strict_errors": len(selected) - consensus_strict,
        }
    return output


def summarize(
    run_path: Path,
    final_grade_paths: list[Path],
    solver_grade_paths: list[Path],
) -> dict:
    results, meta, source_sha = _load_sealed_run(run_path)
    ids = [str(row.get("id")) for row in results]
    if len(ids) != len(set(ids)):
        raise RuntimeError("source run has duplicate problem ids")
    id_set = set(ids)
    final_grades = _load_grades(
        final_grade_paths, source_sha, "final", id_set,
    ) if final_grade_paths else {}
    solver_grades = _load_grades(
        solver_grade_paths, source_sha, "solver_initial", id_set,
    ) if solver_grade_paths else {}

    r0_ids = {
        str(row.get("id")) for row in results
        if (row.get("repair_metrics") or {}).get("first_attempt_verified")
    }
    r1_ids = {str(row.get("id")) for row in results if row.get("verified")}
    execution_errors = sum(bool(row.get("error")) for row in results)
    repair_rows = [row.get("repair_metrics") or {} for row in results]
    if any(not isinstance(row.get("repair_metrics"), dict) for row in results):
        raise RuntimeError("source run is missing repair metrics")
    if any(not isinstance(row.get("metrics"), dict) for row in results):
        raise RuntimeError("source run is missing per-problem telemetry")
    repair_attempted = sum(bool(row.get("repair_attempted")) for row in repair_rows)
    certified_by_repair = sum(bool(row.get("certified_by_repair")) for row in repair_rows)
    if not r0_ids <= r1_ids:
        raise RuntimeError("invalid operating points: R0 certification is not in R1")
    if len(r1_ids - r0_ids) != certified_by_repair:
        raise RuntimeError(
            "certified_by_repair does not match the R1 minus R0 transition"
        )
    research_audit = meta.get("research_audit") or {}
    run_args = research_audit.get("run_args") or {}

    tools = Counter()
    evidence_by_role: dict[str, Counter] = {}
    usage_complete_calls = 0
    calls = 0
    required_tool_calls = 0
    compliant_required_tool_calls = 0
    required_tool_obligations = 0
    satisfied_required_tool_obligations = 0
    for row in results:
        metrics = row.get("metrics") or {}
        tools.update(metrics.get("tool_calls_by_name") or {})
        calls += int(metrics.get("num_calls", 0) or 0)
        usage_complete_calls += int(metrics.get("usage_complete_calls", 0) or 0)
        required_tool_calls += int(metrics.get("required_tool_calls", 0) or 0)
        compliant_required_tool_calls += int(
            metrics.get("compliant_required_tool_calls", 0) or 0
        )
        required_tool_obligations += int(
            metrics.get("required_tool_obligations", 0) or 0
        )
        satisfied_required_tool_obligations += int(
            metrics.get("satisfied_required_tool_obligations", 0) or 0
        )
        for role, counts in (metrics.get("mechanistic_evidence_by_role") or {}).items():
            evidence_by_role.setdefault(role, Counter()).update(counts or {})

    report = {
        "schema_version": 1,
        "source_run": str(run_path.resolve()),
        "source_run_sha256": source_sha,
        "run_id": meta.get("run_id"),
        "experiment": {
            "id": run_args.get("experiment_id"),
            "phase": run_args.get("experiment_phase"),
            "trial": run_args.get("trial"),
        },
        "model_provenance": {
            "roles": research_audit.get("role_model_manifest"),
            "runtime_identity": research_audit.get("model_runtime_identity"),
            "runtime_identity_complete": research_audit.get(
                "model_runtime_identity_complete"
            ),
        },
        "sandbox": meta.get("sandbox"),
        "cohort": {
            "problems": len(results),
            "ordered_ids_sha256": harness._sha256_json(ids),
            "dataset_revisions": (meta.get("problem_set") or {}).get(
                "dataset_revisions"
            ),
            "dataset_fingerprints": (meta.get("problem_set") or {}).get(
                "dataset_fingerprints"
            ),
            "execution_errors": execution_errors,
            "intent_to_treat_denominator": len(results),
        },
        "solver_only": {
            "accuracy": _graded_metric(
                solver_grades, ids, id_set, solver_only=True,
            ),
        },
        "operating_points": {
            "r0_first_pass": {
                "coverage": _rate(len(r0_ids), len(results)),
                "certified_precision": _graded_metric(final_grades, ids, r0_ids),
            },
            "r1_final": {
                "coverage": _rate(len(r1_ids), len(results)),
                "certified_precision": _graded_metric(final_grades, ids, r1_ids),
            },
        },
        "repair": {
            "attempted": repair_attempted,
            "certified_by_repair": certified_by_repair,
            "yield": _rate(certified_by_repair, repair_attempted),
            "answer_changed": sum(
                value is True
                for value in (
                    row.get("answer_changed_during_repair") for row in repair_rows
                )
            ),
            "answer_change_unobservable": sum(
                row.get("repair_attempted")
                and not row.get("answer_change_observable")
                for row in repair_rows
            ),
        },
        "mechanistic_evidence": {
            "tool_calls_by_name": dict(sorted(tools.items())),
            "by_role": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(evidence_by_role.items())
            },
            "required_tool_call_compliance": _rate(
                compliant_required_tool_calls, required_tool_calls,
            ),
            "required_tool_obligation_compliance": _rate(
                satisfied_required_tool_obligations, required_tool_obligations,
            ),
            "note": "Tool counts are evidence of use, not proof of compliance.",
        },
        "telemetry": {
            "calls": calls,
            "usage_complete_calls": usage_complete_calls,
            "usage_complete_fraction": (
                usage_complete_calls / calls if calls else None
            ),
        },
        "graders": {
            "final": sorted(final_grades),
            "solver_initial": sorted(solver_grades),
            "automatic_grades_are_authoritative": False,
        },
    }
    return report


def _format_rate(rate: dict) -> str:
    if rate["value"] is None:
        return "n/a"
    low, high = rate["wilson_95"]
    return (
        f"{rate['numerator']}/{rate['denominator']} "
        f"({100 * rate['value']:.1f}%, 95% CI {100 * low:.1f}-{100 * high:.1f}%)"
    )


def print_report(report: dict) -> None:
    print(f"run: {report['run_id']}")
    print(f"problems: {report['cohort']['problems']}")
    for name, point in report["operating_points"].items():
        print(f"{name} coverage: {_format_rate(point['coverage'])}")
        for grader, values in point["certified_precision"].items():
            print(
                f"{name} strict precision [{grader}]: "
                f"{_format_rate(values['strict_correct'])}"
            )
    for grader, values in report["solver_only"]["accuracy"].items():
        print(
            f"solver accuracy [{grader}]: "
            f"{_format_rate(values['strict_correct'])}"
        )
    print(
        "repair yield: "
        f"{_format_rate(report['repair']['yield'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--final-grade", type=Path, action="append", default=[])
    parser.add_argument("--solver-grade", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(args.run, args.final_grade, args.solver_grade)
    print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
