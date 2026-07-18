#!/usr/bin/env python3
"""Verify completed blinded decisions and compute adjudicated HLE metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

if __package__:
    from experiments import summarize_run
    from experiments.build_adjudication_queue import (
        CASES_FILENAME,
        DECISION_TEMPLATE_FILENAME,
        MANIFEST_FILENAME,
        SCHEMA_VERSION,
        SUPPORTED_TARGETS,
        _candidate_answer,
        _certified_rows,
        _grade_input_records,
        _grade_signature,
        _identity_sha256,
        _is_hle_row,
        _load_frozen_rationales,
        _sha256_file,
        _target_certified,
        _write_new,
        select_adjudication_cases,
        validate_blinded_cases,
    )
else:  # Support `python experiments/apply_adjudication.py ...`.
    import summarize_run
    from build_adjudication_queue import (
        CASES_FILENAME,
        DECISION_TEMPLATE_FILENAME,
        MANIFEST_FILENAME,
        SCHEMA_VERSION,
        SUPPORTED_TARGETS,
        _candidate_answer,
        _certified_rows,
        _grade_input_records,
        _grade_signature,
        _identity_sha256,
        _is_hle_row,
        _load_frozen_rationales,
        _sha256_file,
        _target_certified,
        _write_new,
        select_adjudication_cases,
        validate_blinded_cases,
    )


_DECISION_KEYS = {
    "case_id",
    "status",
    "strict_correct",
    "reviewer_ids",
    "reasoning",
}
_DECISIONS_TOP_LEVEL_KEYS = {
    "schema_version",
    "queue_id",
    "cases_sha256",
    "decision_standard",
    "review_protocol",
    "decisions",
}


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc


def _verify_queue_artifacts(queue_dir: Path) -> tuple[dict, dict, dict]:
    manifest_path = queue_dir / MANIFEST_FILENAME
    cases_path = queue_dir / CASES_FILENAME
    template_path = queue_dir / DECISION_TEMPLATE_FILENAME
    manifest = _load_json(manifest_path)
    cases_payload = _load_json(cases_path)
    template = _load_json(template_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported adjudication manifest")
    queue_id = manifest.get("queue_id")
    if not isinstance(queue_id, str) or not queue_id:
        raise RuntimeError("adjudication manifest has no queue_id")
    expected_queue_id = "adj_" + _identity_sha256(
        manifest.get("queue_identity")
    )[:24]
    if queue_id != expected_queue_id:
        raise RuntimeError("adjudication manifest queue identity is invalid")

    artifacts = manifest.get("artifacts") or {}
    for name, path in (
        (CASES_FILENAME, cases_path),
        (DECISION_TEMPLATE_FILENAME, template_path),
    ):
        expected = artifacts.get(name) or {}
        if expected.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"adjudication artifact hash mismatch: {name}")
        if expected.get("size_bytes") != path.stat().st_size:
            raise RuntimeError(f"adjudication artifact size mismatch: {name}")

    cases = validate_blinded_cases(cases_payload, queue_id=queue_id)
    if not isinstance(template, dict):
        raise RuntimeError("decision template is not an object")
    if template.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported decision template schema")
    if template.get("queue_id") != queue_id:
        raise RuntimeError("decision template queue_id mismatch")
    if template.get("cases_sha256") != _sha256_file(cases_path):
        raise RuntimeError("decision template cases hash mismatch")
    template_rows = template.get("decisions")
    if not isinstance(template_rows, list):
        raise RuntimeError("decision template has no decision rows")
    if [row.get("case_id") for row in template_rows] != [
        case["case_id"] for case in cases
    ]:
        raise RuntimeError("decision template does not align with blinded cases")
    return manifest, cases_payload, template


def _load_completed_decisions(
    path: Path,
    *,
    queue_id: str,
    cases_sha256: str,
    expected_case_ids: list[str],
    template: dict,
) -> dict[str, dict]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or set(payload) != _DECISIONS_TOP_LEVEL_KEYS:
        raise RuntimeError("completed decisions have an unexpected top-level shape")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported completed-decisions schema")
    if payload.get("queue_id") != queue_id:
        raise RuntimeError("completed decisions belong to a different queue")
    if payload.get("cases_sha256") != cases_sha256:
        raise RuntimeError("completed decisions do not bind the blinded cases")
    for field in ("decision_standard", "review_protocol"):
        if payload.get(field) != template.get(field):
            raise RuntimeError(f"completed decisions changed the {field}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise RuntimeError("completed decisions must contain a decisions list")

    by_id = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != _DECISION_KEYS:
            raise RuntimeError("completed decision has an unexpected shape")
        case_id = decision.get("case_id")
        if case_id in by_id:
            raise RuntimeError(f"duplicate completed decision: {case_id}")
        if decision.get("status") != "completed":
            raise RuntimeError(f"adjudication decision is not completed: {case_id}")
        if type(decision.get("strict_correct")) is not bool:
            raise RuntimeError(f"adjudication decision is not boolean: {case_id}")
        reviewer_ids = decision.get("reviewer_ids")
        if (
            not isinstance(reviewer_ids, list)
            or len(reviewer_ids) < 2
            or any(not isinstance(value, str) or not value.strip() for value in reviewer_ids)
            or len({value.strip() for value in reviewer_ids}) != len(reviewer_ids)
        ):
            raise RuntimeError(
                f"adjudication decision needs two distinct reviewer_ids: {case_id}"
            )
        if not isinstance(decision.get("reasoning"), str) or not decision[
            "reasoning"
        ].strip():
            raise RuntimeError(f"adjudication decision has no reasoning: {case_id}")
        by_id[case_id] = decision
    if set(by_id) != set(expected_case_ids) or len(by_id) != len(expected_case_ids):
        missing = sorted(set(expected_case_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_case_ids))
        raise RuntimeError(
            f"completed decisions do not cover the queue: missing={missing}, extra={extra}"
        )
    return by_id


def _domain_report(
    rows: list[dict], correct_by_id: dict[str, bool], *, target: str,
) -> dict:
    output = {}
    domains = sorted({str(row.get("category") or "unknown") for row in rows})
    for domain in domains:
        domain_rows = [
            row for row in rows
            if str(row.get("category") or "unknown") == domain
        ]
        certified = [
            row for row in domain_rows
            if _target_certified(row, target)
        ]
        correct = sum(correct_by_id[str(row.get("id"))] for row in certified)
        output[domain] = {
            "problems": len(domain_rows),
            "certified": len(certified),
            "coverage": summarize_run._rate(len(certified), len(domain_rows)),
            "strict_correct_certified": correct,
            "strict_certified_precision": summarize_run._rate(
                correct, len(certified),
            ),
            "errors_shipped": len(certified) - correct,
        }
    return output


def apply_decisions(
    run_path: Path,
    grade_paths: list[Path],
    queue_dir: Path,
    decisions_path: Path,
    *,
    target: str = "final",
    out_path: Path | None = None,
) -> dict:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported adjudication target: {target}")
    queue_dir = queue_dir.resolve(strict=True)
    manifest, cases_payload, template = _verify_queue_artifacts(queue_dir)
    queue_id = manifest["queue_id"]
    cases = validate_blinded_cases(cases_payload, queue_id=queue_id)
    cases_sha256 = _sha256_file(queue_dir / CASES_FILENAME)
    decisions = _load_completed_decisions(
        decisions_path.resolve(strict=True),
        queue_id=queue_id,
        cases_sha256=cases_sha256,
        expected_case_ids=[case["case_id"] for case in cases],
        template=template,
    )

    results, run_meta, source_sha256 = summarize_run._load_sealed_run(run_path)
    ids = [str(row.get("id")) for row in results]
    created_from = manifest.get("created_from") or {}
    if created_from.get("grading_target") != target:
        raise RuntimeError("adjudication manifest belongs to a different target")
    if created_from.get("source_run_sha256") != source_sha256:
        raise RuntimeError("adjudication manifest belongs to a different source run")
    ordered_ids_sha = summarize_run.harness._sha256_json(ids)
    if created_from.get("ordered_problem_ids_sha256") != ordered_ids_sha:
        raise RuntimeError("source problem order differs from the adjudication queue")
    if created_from.get("problem_count") != len(results):
        raise RuntimeError("source problem count differs from the adjudication queue")
    non_hle = [str(row.get("id")) for row in results if not _is_hle_row(row)]
    if non_hle:
        raise RuntimeError(f"adjudication report accepts only HLE rows: {non_hle}")

    grade_paths = [Path(path) for path in grade_paths]
    grade_inputs = _grade_input_records(grade_paths, target=target)
    if _grade_signature(grade_inputs) != _grade_signature(
        created_from.get("grades") or []
    ):
        raise RuntimeError("grade inputs differ from the adjudication queue")
    graders = summarize_run._load_grades(
        grade_paths, source_sha256, target, set(ids),
    )
    if set(graders) != {record["grader"] for record in grade_inputs}:
        raise RuntimeError("verified grade identities do not match grade files")

    policy = manifest.get("selection_policy") or {}
    agreement_policy = policy.get("agreement_audit") or {}
    rationales, rationales_sha256 = _load_frozen_rationales(
        grade_paths, set(ids),
    )
    expected_queue_identity = {
        "schema_version": SCHEMA_VERSION,
        "source_run_sha256": source_sha256,
        "ordered_problem_ids_sha256": ordered_ids_sha,
        "grading_target": target,
        "grades": _grade_signature(grade_inputs),
        "canonical_rationales_sha256": rationales_sha256,
        "selection_seed": str(agreement_policy.get("seed")),
        "agreement_audit_fraction": float(agreement_policy.get("fraction")),
    }
    if manifest.get("queue_identity") != expected_queue_identity:
        raise RuntimeError("adjudication queue identity does not match sealed inputs")
    if created_from.get("canonical_rationales_sha256") != rationales_sha256:
        raise RuntimeError("canonical rationales differ from the adjudication queue")
    if created_from.get("certified_count") != len(_certified_rows(results, target)):
        raise RuntimeError("source certification count differs from the queue")
    recomputed = select_adjudication_cases(
        results,
        graders,
        target=target,
        seed=str(agreement_policy.get("seed")),
        audit_fraction=float(agreement_policy.get("fraction")),
    )
    manifest_cases = manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise RuntimeError("adjudication manifest has no case records")
    manifest_by_problem = {
        str(case.get("problem_id")): case for case in manifest_cases
    }
    if len(manifest_by_problem) != len(manifest_cases):
        raise RuntimeError("adjudication manifest contains duplicate problem ids")
    if set(manifest_by_problem) != {
        selection["problem_id"] for selection in recomputed
    }:
        raise RuntimeError("adjudication manifest omits or adds selected cases")
    for selection in recomputed:
        stored = manifest_by_problem[selection["problem_id"]]
        for field in (
            "category",
            "automatic_strict_correct",
            "automatic_consensus_strict_correct",
            "grader_disagreement",
            "apparent_certified_error",
            "agreement_audit_hash",
            "selection_reasons",
        ):
            if stored.get(field) != selection[field]:
                raise RuntimeError(
                    f"adjudication manifest selection mismatch: "
                    f"{selection['problem_id']} {field}"
                )

    blinded_by_id = {case["case_id"]: case for case in cases}
    manifest_by_case = {case.get("case_id"): case for case in manifest_cases}
    if set(blinded_by_id) != set(manifest_by_case):
        raise RuntimeError("private manifest does not align with blinded cases")
    for case_id, blinded in blinded_by_id.items():
        private = manifest_by_case[case_id]
        if private.get("blinded_case_sha256") != _identity_sha256(blinded):
            raise RuntimeError(f"blinded case payload mismatch: {case_id}")
        problem_id = str(private.get("problem_id"))
        source_row = next(
            (row for row in results if str(row.get("id")) == problem_id), None,
        )
        if source_row is None:
            raise RuntimeError(f"blinded case has no source row: {case_id}")
        expected_case = {
            "case_id": case_id,
            "category": str(source_row.get("category") or "unknown"),
            "problem": source_row.get("problem"),
            "expected_answer": str(source_row.get("expected")),
            "candidate_answer": str(_candidate_answer(source_row, target)),
            "canonical_rationale": rationales.get(problem_id),
        }
        if blinded != expected_case:
            raise RuntimeError(f"blinded case differs from sealed inputs: {case_id}")

    decision_by_problem = {
        case["problem_id"]: decisions[case["case_id"]]
        for case in manifest_cases
    }
    certified = _certified_rows(results, target)
    correct_by_id = {}
    automatic_by_id = {}
    adjudicated_rows = []
    for row in certified:
        problem_id = str(row.get("id"))
        automatic = all(
            summarize_run._key_match(rows[problem_id])
            for rows in graders.values()
        )
        automatic_by_id[problem_id] = automatic
        decision = decision_by_problem.get(problem_id)
        if decision is None:
            if not automatic:
                raise RuntimeError(
                    f"automatic certified error escaped adjudication: {problem_id}"
                )
            correct_by_id[problem_id] = True
            continue
        correct_by_id[problem_id] = decision["strict_correct"]
        case = manifest_by_problem[problem_id]
        adjudicated_rows.append({
            "case_id": case["case_id"],
            "problem_id": problem_id,
            "category": str(row.get("category") or "unknown"),
            "selection_reasons": case["selection_reasons"],
            "automatic_consensus_strict_correct": automatic,
            "adjudicated_strict_correct": decision["strict_correct"],
            "changed_automatic_label": automatic != decision["strict_correct"],
            "reviewer_ids": decision["reviewer_ids"],
        })

    strict_correct = sum(correct_by_id.values())
    automatic_correct = sum(automatic_by_id.values())
    ordered_outcomes = [
        {
            "id": str(row.get("id")),
            "certified": str(row.get("id")) in correct_by_id,
            "strict_correct": correct_by_id.get(str(row.get("id"))),
        }
        for row in results
    ]
    selection_counts = Counter(
        reason for case in manifest_cases for reason in case["selection_reasons"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "grading_target": target,
        "source": {
            "run": str(run_path.resolve()),
            "run_id": run_meta.get("run_id"),
            "run_sha256": source_sha256,
            "ordered_problem_ids_sha256": ordered_ids_sha,
            "grades": grade_inputs,
        },
        "adjudication_provenance": {
            "manifest": str((queue_dir / MANIFEST_FILENAME).resolve()),
            "manifest_sha256": _sha256_file(queue_dir / MANIFEST_FILENAME),
            "cases_sha256": cases_sha256,
            "decision_template_sha256": _sha256_file(
                queue_dir / DECISION_TEMPLATE_FILENAME
            ),
            "completed_decisions": str(decisions_path.resolve()),
            "completed_decisions_sha256": _sha256_file(decisions_path.resolve()),
            "queued_cases": len(manifest_cases),
            "completed_cases": len(decisions),
            "selection_reason_counts": dict(sorted(selection_counts.items())),
        },
        "cohort": {
            "problems": len(results),
            "execution_errors": sum(bool(row.get("error")) for row in results),
            "certified": len(certified),
        },
        "adjudicated_metrics": {
            "coverage": summarize_run._rate(len(certified), len(results)),
            "strict_correct_certified": strict_correct,
            "strict_certified_precision": summarize_run._rate(
                strict_correct, len(certified),
            ),
            "errors_shipped": len(certified) - strict_correct,
        },
        "automatic_consensus_reference": {
            "strict_correct_certified": automatic_correct,
            "strict_certified_precision": summarize_run._rate(
                automatic_correct, len(certified),
            ),
            "errors_shipped": len(certified) - automatic_correct,
        },
        "per_domain": _domain_report(results, correct_by_id, target=target),
        "ordered_outcomes": ordered_outcomes,
        "adjudicated_cases": sorted(
            adjudicated_rows, key=lambda row: row["case_id"],
        ),
    }
    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_new(out_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), default="final")
    parser.add_argument("--grade", type=Path, action="append", required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = apply_decisions(
        args.run,
        args.grade,
        args.queue_dir,
        args.decisions,
        target=args.target,
        out_path=args.out,
    )
    precision = report["adjudicated_metrics"]["strict_certified_precision"]
    print(args.out)
    print(
        f"strict certified precision: {precision['numerator']}/"
        f"{precision['denominator']}"
    )


if __name__ == "__main__":
    main()
