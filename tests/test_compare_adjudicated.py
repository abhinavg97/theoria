import json
from pathlib import Path

import pytest

from experiments import compare_adjudicated
from experiments import summarize_run


def _outcome(problem_id: str, certified: bool, correct: bool | None) -> dict:
    return {
        "id": problem_id,
        "certified": certified,
        "strict_correct": correct,
    }


def _report(target: str, outcomes: list[dict], *, source_sha: str = "a" * 64) -> dict:
    certified = sum(outcome["certified"] for outcome in outcomes)
    correct = sum(outcome["strict_correct"] is True for outcome in outcomes)
    return {
        "schema_version": 1,
        "queue_id": f"queue-{target}",
        "grading_target": target,
        "source": {
            "run": "/sealed/run.json",
            "run_id": "same-run",
            "run_sha256": source_sha,
            "ordered_problem_ids_sha256": summarize_run.harness._sha256_json(
                [outcome["id"] for outcome in outcomes]
            ),
            "grades": [],
        },
        "cohort": {
            "problems": len(outcomes),
            "execution_errors": 0,
            "certified": certified,
        },
        "adjudicated_metrics": {
            "coverage": summarize_run._rate(certified, len(outcomes)),
            "strict_correct_certified": correct,
            "strict_certified_precision": summarize_run._rate(correct, certified),
            "errors_shipped": certified - correct,
        },
        "ordered_outcomes": outcomes,
    }


def _write_report(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def test_compare_reports_computes_paired_adjudicated_inference(tmp_path):
    r0 = [
        _outcome("p1", True, True),
        _outcome("p2", True, False),
        _outcome("p3", False, None),
        _outcome("p4", False, None),
        _outcome("p5", False, None),
    ]
    r1 = [
        _outcome("p1", True, True),
        _outcome("p2", True, True),
        _outcome("p3", True, True),
        _outcome("p4", True, False),
        _outcome("p5", False, None),
    ]
    first_path = _write_report(
        tmp_path / "first.json", _report("first_attempt", r0),
    )
    final_path = _write_report(tmp_path / "final.json", _report("final", r1))
    out_path = tmp_path / "paired.json"

    report = compare_adjudicated.compare_reports(
        first_path, final_path, out_path=out_path,
    )
    paired = report["paired_changes"]

    assert report["operating_points"]["r0_first_attempt"]["certified"] == 2
    assert report["operating_points"]["r1_final"]["certified"] == 4
    coverage = paired["coverage_r1_minus_r0"]
    assert coverage["estimate"] == pytest.approx(0.4)
    assert coverage["paired_bootstrap_95"] is not None
    assert coverage["mcnemar"] == {
        "lost": 0,
        "gained": 2,
        "discordant": 2,
        "two_sided_exact_p": 0.5,
    }
    correct = paired["correct_and_certified_rate_r1_minus_r0"]
    assert correct["estimate"] == pytest.approx(0.4)
    assert correct["paired_bootstrap_95"] is not None
    assert correct["mcnemar"]["gained"] == 2
    precision = paired["certified_precision_r1_minus_r0"]
    assert precision["estimate"] == pytest.approx(0.25)
    assert precision["paired_bootstrap_95"] is not None
    assert json.loads(out_path.read_text())["paired_changes"] == paired

    repeated = compare_adjudicated.compare_reports(first_path, final_path)
    assert repeated["paired_changes"] == paired


def test_compare_reports_rejects_different_source_order_or_cohort(tmp_path):
    outcomes = [_outcome("p1", False, None), _outcome("p2", False, None)]
    first_path = _write_report(
        tmp_path / "first.json", _report("first_attempt", outcomes),
    )
    final = _report("final", outcomes, source_sha="b" * 64)
    final_path = _write_report(tmp_path / "final.json", final)

    with pytest.raises(RuntimeError, match="different source run_sha256"):
        compare_adjudicated.compare_reports(first_path, final_path)

    final = _report("final", list(reversed(outcomes)))
    final_path.write_text(json.dumps(final))
    with pytest.raises(RuntimeError, match="different source ordered_problem_ids_sha256"):
        compare_adjudicated.compare_reports(first_path, final_path)

    final = _report("final", outcomes)
    final["cohort"]["execution_errors"] = 1
    final_path.write_text(json.dumps(final))
    with pytest.raises(RuntimeError, match="different cohort execution_errors"):
        compare_adjudicated.compare_reports(first_path, final_path)


def test_compare_reports_rejects_non_subset_and_invalid_outcomes(tmp_path):
    r0 = [_outcome("p1", True, True), _outcome("p2", False, None)]
    r1 = [_outcome("p1", False, None), _outcome("p2", True, True)]
    first_path = _write_report(
        tmp_path / "first.json", _report("first_attempt", r0),
    )
    final_path = _write_report(tmp_path / "final.json", _report("final", r1))

    with pytest.raises(RuntimeError, match="R0 certification is not a subset"):
        compare_adjudicated.compare_reports(first_path, final_path)

    invalid = _report("final", r1)
    invalid["ordered_outcomes"][0]["strict_correct"] = False
    final_path.write_text(json.dumps(invalid))
    with pytest.raises(RuntimeError, match="declined outcome has a strict label"):
        compare_adjudicated.compare_reports(first_path, final_path)
