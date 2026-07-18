#!/usr/bin/env python3
"""Compare adjudicated first-attempt and final HLE operating points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from experiments import summarize_run
    from experiments.build_adjudication_queue import _sha256_file, _write_new
else:  # Support `python experiments/compare_adjudicated.py ...`.
    import summarize_run
    from build_adjudication_queue import _sha256_file, _write_new


SCHEMA_VERSION = 1
_OUTCOME_KEYS = {"id", "certified", "strict_correct"}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid adjudication report JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"adjudication report is not an object: {path}")
    return payload


def _metric_count(metric: object, field: str, expected: int, path: Path) -> None:
    if (
        not isinstance(metric, dict)
        or type(metric.get(field)) is not int
        or metric.get(field) != expected
    ):
        raise RuntimeError(
            f"adjudication report metric mismatch for {field}: {path}"
        )


def _validated_report(path: Path, *, target: str) -> tuple[dict, list[dict]]:
    report = _load_json(path)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported adjudication report schema: {path}")
    if report.get("grading_target") != target:
        raise RuntimeError(
            f"adjudication report target mismatch: expected {target}: {path}"
        )
    if not isinstance(report.get("queue_id"), str) or not report["queue_id"]:
        raise RuntimeError(f"adjudication report has no queue identity: {path}")

    source = report.get("source")
    cohort = report.get("cohort")
    metrics = report.get("adjudicated_metrics")
    outcomes = report.get("ordered_outcomes")
    if not isinstance(source, dict) or not isinstance(cohort, dict):
        raise RuntimeError(f"adjudication report has no source/cohort: {path}")
    if not isinstance(metrics, dict) or not isinstance(outcomes, list):
        raise RuntimeError(f"adjudication report has no metrics/outcomes: {path}")
    if not isinstance(source.get("run_sha256"), str) or not source["run_sha256"]:
        raise RuntimeError(f"adjudication report has no sealed run hash: {path}")
    order_sha = source.get("ordered_problem_ids_sha256")
    if not isinstance(order_sha, str) or not order_sha:
        raise RuntimeError(f"adjudication report has no problem-order hash: {path}")

    problem_count = cohort.get("problems")
    if type(problem_count) is not int or problem_count <= 0:
        raise RuntimeError(f"adjudication report has invalid cohort size: {path}")
    execution_errors = cohort.get("execution_errors")
    if (
        type(execution_errors) is not int
        or execution_errors < 0
        or execution_errors > problem_count
    ):
        raise RuntimeError(
            f"adjudication report has invalid execution-error count: {path}"
        )
    if len(outcomes) != problem_count:
        raise RuntimeError(f"adjudication outcome count differs from cohort: {path}")

    ids = []
    certified_count = 0
    correct_count = 0
    for outcome in outcomes:
        if not isinstance(outcome, dict) or set(outcome) != _OUTCOME_KEYS:
            raise RuntimeError(f"adjudication outcome has an unexpected shape: {path}")
        problem_id = outcome.get("id")
        certified = outcome.get("certified")
        strict_correct = outcome.get("strict_correct")
        if not isinstance(problem_id, str) or not problem_id:
            raise RuntimeError(f"adjudication outcome has no problem id: {path}")
        if type(certified) is not bool:
            raise RuntimeError(f"adjudication outcome certification is not boolean: {path}")
        if certified and type(strict_correct) is not bool:
            raise RuntimeError(f"certified outcome has no strict label: {path}")
        if not certified and strict_correct is not None:
            raise RuntimeError(f"declined outcome has a strict label: {path}")
        ids.append(problem_id)
        certified_count += int(certified)
        correct_count += int(strict_correct is True)

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"adjudication outcomes contain duplicate ids: {path}")
    if summarize_run.harness._sha256_json(ids) != order_sha:
        raise RuntimeError(f"adjudication outcomes do not match the order hash: {path}")
    if type(cohort.get("certified")) is not int or (
        cohort.get("certified") != certified_count
    ):
        raise RuntimeError(f"adjudication certified count mismatch: {path}")
    _metric_count(metrics.get("coverage"), "numerator", certified_count, path)
    _metric_count(metrics.get("coverage"), "denominator", problem_count, path)
    if type(metrics.get("strict_correct_certified")) is not int or (
        metrics.get("strict_correct_certified") != correct_count
    ):
        raise RuntimeError(f"adjudication strict-correct count mismatch: {path}")
    precision = metrics.get("strict_certified_precision")
    _metric_count(precision, "numerator", correct_count, path)
    _metric_count(precision, "denominator", certified_count, path)
    if type(metrics.get("errors_shipped")) is not int or (
        metrics.get("errors_shipped") != certified_count - correct_count
    ):
        raise RuntimeError(f"adjudication errors-shipped count mismatch: {path}")
    return report, outcomes


def _operating_point(outcomes: list[dict]) -> dict:
    certified = sum(outcome["certified"] for outcome in outcomes)
    correct = sum(outcome["strict_correct"] is True for outcome in outcomes)
    return {
        "problems": len(outcomes),
        "certified": certified,
        "correct_and_certified": correct,
        "coverage": summarize_run._rate(certified, len(outcomes)),
        "certified_precision": summarize_run._rate(correct, certified),
        "errors_shipped": certified - correct,
    }


def _paired_changes(records: list[tuple[bool, bool, bool, bool]]) -> dict:
    def coverage_difference(sample):
        return sum(r1 - r0 for r0, r1, _c0, _c1 in sample) / len(sample)

    def correct_certified_difference(sample):
        return sum(c1 - c0 for _r0, _r1, c0, c1 in sample) / len(sample)

    def precision_difference(sample):
        r0_total = sum(r0 for r0, _r1, _c0, _c1 in sample)
        r1_total = sum(r1 for _r0, r1, _c0, _c1 in sample)
        if not r0_total or not r1_total:
            return None
        return (
            sum(c1 for _r0, _r1, _c0, c1 in sample) / r1_total
            - sum(c0 for _r0, _r1, c0, _c1 in sample) / r0_total
        )

    r0_total = sum(record[0] for record in records)
    r1_total = sum(record[1] for record in records)
    return {
        "bootstrap_seed": summarize_run.BOOTSTRAP_SEED,
        "bootstrap_resamples": summarize_run.BOOTSTRAP_RESAMPLES,
        "coverage_r1_minus_r0": {
            "estimate": sum(record[1] - record[0] for record in records) / len(records),
            "paired_bootstrap_95": summarize_run._paired_bootstrap_interval(
                records, coverage_difference,
            ),
            "mcnemar": summarize_run._mcnemar_exact(
                [record[0] for record in records],
                [record[1] for record in records],
            ),
        },
        "correct_and_certified_rate_r1_minus_r0": {
            "estimate": sum(record[3] - record[2] for record in records) / len(records),
            "paired_bootstrap_95": summarize_run._paired_bootstrap_interval(
                records, correct_certified_difference, seed_offset=1,
            ),
            "mcnemar": summarize_run._mcnemar_exact(
                [record[2] for record in records],
                [record[3] for record in records],
            ),
        },
        "certified_precision_r1_minus_r0": {
            "estimate": (
                sum(record[3] for record in records) / r1_total
                - sum(record[2] for record in records) / r0_total
                if r0_total and r1_total else None
            ),
            "paired_bootstrap_95": (
                summarize_run._paired_bootstrap_interval(
                    records, precision_difference, seed_offset=2,
                )
                if r0_total and r1_total else None
            ),
        },
    }


def compare_reports(
    first_attempt_path: Path,
    final_path: Path,
    *,
    out_path: Path | None = None,
) -> dict:
    first_attempt_path = first_attempt_path.resolve(strict=True)
    final_path = final_path.resolve(strict=True)
    first, r0_outcomes = _validated_report(
        first_attempt_path, target="first_attempt",
    )
    final, r1_outcomes = _validated_report(final_path, target="final")

    first_source = first["source"]
    final_source = final["source"]
    for field in ("run_sha256", "ordered_problem_ids_sha256", "run_id"):
        if first_source.get(field) != final_source.get(field):
            raise RuntimeError(f"adjudication reports use different source {field}")
    for field in ("problems", "execution_errors"):
        if first["cohort"].get(field) != final["cohort"].get(field):
            raise RuntimeError(f"adjudication reports use different cohort {field}")

    r0_ids = [outcome["id"] for outcome in r0_outcomes]
    r1_ids = [outcome["id"] for outcome in r1_outcomes]
    if r0_ids != r1_ids:
        raise RuntimeError("adjudication reports have different problem order")

    records = []
    for r0, r1 in zip(r0_outcomes, r1_outcomes):
        if r0["certified"] and not r1["certified"]:
            raise RuntimeError(
                f"R0 certification is not a subset of R1: {r0['id']}"
            )
        records.append((
            r0["certified"],
            r1["certified"],
            r0["strict_correct"] is True,
            r1["strict_correct"] is True,
        ))

    report = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "run_id": first_source.get("run_id"),
            "run_sha256": first_source["run_sha256"],
            "ordered_problem_ids_sha256": first_source[
                "ordered_problem_ids_sha256"
            ],
        },
        "input_reports": {
            "first_attempt": {
                "path": str(first_attempt_path),
                "sha256": _sha256_file(first_attempt_path),
                "queue_id": first.get("queue_id"),
            },
            "final": {
                "path": str(final_path),
                "sha256": _sha256_file(final_path),
                "queue_id": final.get("queue_id"),
            },
        },
        "cohort": {
            "problems": len(records),
            "execution_errors": first["cohort"].get("execution_errors"),
        },
        "operating_points": {
            "r0_first_attempt": _operating_point(r0_outcomes),
            "r1_final": _operating_point(r1_outcomes),
        },
        "paired_changes": _paired_changes(records),
    }
    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_new(out_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-attempt", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(
        args.first_attempt,
        args.final,
        out_path=args.out,
    )
    paired = report["paired_changes"]
    print(args.out)
    print(
        "coverage R1-R0: "
        f"{paired['coverage_r1_minus_r0']['estimate']:.6f}"
    )


if __name__ == "__main__":
    main()
