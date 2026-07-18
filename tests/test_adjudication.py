import json
from pathlib import Path

import pytest

from experiments import apply_adjudication
from experiments import build_adjudication_queue
from loaders import HLE_DATASET_NAME


def _grade(correct: bool, dispute_category: str = "none") -> dict:
    return {
        "final": {
            "key_match": correct,
            "dispute_category": dispute_category,
        },
    }


def _publication_inputs(tmp_path: Path, monkeypatch):
    rows = [
        {
            "id": "disagreement",
            "dataset_name": HLE_DATASET_NAME,
            "problem": "Disagreement question",
            "expected": "A",
            "answer": "B",
            "verified": True,
            "category": "Math",
            "metrics": {},
        },
        {
            "id": "consensus-error",
            "dataset_name": HLE_DATASET_NAME,
            "problem": "Error question",
            "expected": "A",
            "answer": "B",
            "verified": True,
            "category": "Science",
            "metrics": {},
        },
    ]
    for index in range(10):
        rows.append({
            "id": f"agreement-{index}",
            "dataset_name": HLE_DATASET_NAME,
            "problem": f"Agreement question {index}",
            "expected": "A",
            "answer": "A",
            "verified": True,
            "category": "Math" if index % 2 == 0 else "Science",
            "metrics": {},
        })
    rows.append({
        "id": "declined",
        "dataset_name": HLE_DATASET_NAME,
        "problem": "Declined question",
        "expected": "A",
        "answer": "B",
        "verified": False,
        "category": "Math",
        "metrics": {},
    })

    grader_one = {str(row["id"]): _grade(True) for row in rows}
    grader_two = {str(row["id"]): _grade(True) for row in rows}
    grader_one["disagreement"] = _grade(True)
    grader_two["disagreement"] = _grade(False)
    grader_one["consensus-error"] = _grade(False)
    grader_two["consensus-error"] = _grade(False)
    graders = {"grader-one": grader_one, "grader-two": grader_two}
    rationales = {
        str(row["id"]): {
            "rationale": f"Canonical rationale for {row['id']}",
            "is_valid": True,
            "error_type": None,
        }
        for row in rows
    }

    run_path = tmp_path / "run.json"
    run_path.write_text("[]")
    grade_paths = []
    for grader in graders:
        path = tmp_path / f"{grader}.json"
        path.write_text(json.dumps({
            "grading_target": "final",
            "grader_model": grader,
        }))
        grade_paths.append(path)

    monkeypatch.setattr(
        build_adjudication_queue.summarize_run,
        "_load_sealed_run",
        lambda _path: (rows, {"run_id": "condition-label"}, "source-sha"),
    )
    monkeypatch.setattr(
        build_adjudication_queue.summarize_run,
        "_load_grades",
        lambda _paths, _sha, _target, _ids: graders,
    )
    frozen = lambda _paths, _ids: (rationales, "canonical-rationales-sha")
    monkeypatch.setattr(
        build_adjudication_queue, "_load_frozen_rationales", frozen,
    )
    monkeypatch.setattr(apply_adjudication, "_load_frozen_rationales", frozen)
    return rows, graders, run_path, grade_paths


def _complete_decisions(queue_dir: Path, manifest: dict) -> Path:
    template = json.loads(
        (queue_dir / build_adjudication_queue.DECISION_TEMPLATE_FILENAME).read_text()
    )
    by_case = {case["case_id"]: case for case in manifest["cases"]}
    for decision in template["decisions"]:
        private = by_case[decision["case_id"]]
        reasons = private["selection_reasons"]
        decision.update({
            "status": "completed",
            "strict_correct": "grader_disagreement" in reasons,
            "reviewer_ids": ["reviewer-opaque-1", "reviewer-opaque-2"],
            "reasoning": "Reviewed against the benchmark key and rationale.",
        })
    completed = queue_dir / "decisions.completed.json"
    completed.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
    return completed


def test_build_queue_is_blinded_complete_and_hash_bound(tmp_path, monkeypatch):
    _rows, _graders, run_path, grade_paths = _publication_inputs(
        tmp_path, monkeypatch,
    )
    queue_dir = tmp_path / "queue"

    manifest = build_adjudication_queue.build_queue(
        run_path, grade_paths, queue_dir,
    )

    reasons = {
        case["problem_id"]: set(case["selection_reasons"])
        for case in manifest["cases"]
    }
    assert "grader_disagreement" in reasons["disagreement"]
    assert "apparent_certified_error" in reasons["disagreement"]
    assert reasons["consensus-error"] == {"apparent_certified_error"}
    sampled = [
        problem_id for problem_id, selected_reasons in reasons.items()
        if "agreement_audit_sample" in selected_reasons
    ]
    assert len(sampled) == 1
    assert manifest["selection_counts"]["queued"] == 3

    cases_path = queue_dir / build_adjudication_queue.CASES_FILENAME
    cases_payload = json.loads(cases_path.read_text())
    cases = build_adjudication_queue.validate_blinded_cases(
        cases_payload, queue_id=manifest["queue_id"],
    )
    assert all(set(case) == build_adjudication_queue._BLINDED_CASE_KEYS for case in cases)
    assert all("problem_id" not in case for case in cases)
    assert all("canonical_rationale" in case for case in cases)
    assert all("certified_proof" not in case and "proof" not in case for case in cases)
    assert "condition-label" not in cases_path.read_text()
    assert "grader-one" not in cases_path.read_text()
    assert manifest["artifacts"][cases_path.name]["sha256"] == (
        build_adjudication_queue._sha256_file(cases_path)
    )

    template = json.loads((
        queue_dir / build_adjudication_queue.DECISION_TEMPLATE_FILENAME
    ).read_text())
    assert template["cases_sha256"] == build_adjudication_queue._sha256_file(
        cases_path
    )
    assert all(decision["status"] == "pending" for decision in template["decisions"])
    assert all(decision["strict_correct"] is None for decision in template["decisions"])
    assert all(decision["reviewer_ids"] == [] for decision in template["decisions"])


def test_apply_decisions_reports_adjudicated_metrics_and_domains(
    tmp_path, monkeypatch,
):
    rows, _graders, run_path, grade_paths = _publication_inputs(
        tmp_path, monkeypatch,
    )
    queue_dir = tmp_path / "queue"
    manifest = build_adjudication_queue.build_queue(
        run_path, grade_paths, queue_dir,
    )
    completed = _complete_decisions(queue_dir, manifest)
    out_path = tmp_path / "adjudicated.json"

    report = apply_adjudication.apply_decisions(
        run_path,
        grade_paths,
        queue_dir,
        completed,
        out_path=out_path,
    )

    assert report["cohort"]["problems"] == len(rows) == 13
    assert report["cohort"]["certified"] == 12
    assert report["adjudicated_metrics"]["coverage"]["numerator"] == 12
    assert report["adjudicated_metrics"]["coverage"]["denominator"] == 13
    assert report["adjudicated_metrics"]["strict_correct_certified"] == 10
    precision = report["adjudicated_metrics"]["strict_certified_precision"]
    assert precision["numerator"] == 10
    assert precision["denominator"] == 12
    assert precision["wilson_95"] is not None
    assert report["adjudicated_metrics"]["errors_shipped"] == 2
    assert sum(
        domain["errors_shipped"] for domain in report["per_domain"].values()
    ) == 2
    assert sum(
        domain["certified"] for domain in report["per_domain"].values()
    ) == 12
    assert sum(
        case["changed_automatic_label"]
        for case in report["adjudicated_cases"]
    ) == 2
    outcomes = report["ordered_outcomes"]
    assert [outcome["id"] for outcome in outcomes] == [
        str(row["id"]) for row in rows
    ]
    assert sum(outcome["certified"] for outcome in outcomes) == 12
    assert sum(outcome["strict_correct"] is True for outcome in outcomes) == 10
    assert outcomes[-1] == {
        "id": "declined",
        "certified": False,
        "strict_correct": None,
    }
    assert json.loads(out_path.read_text())["queue_id"] == manifest["queue_id"]


def test_apply_decisions_fails_closed_on_incomplete_or_tampered_artifacts(
    tmp_path, monkeypatch,
):
    _rows, _graders, run_path, grade_paths = _publication_inputs(
        tmp_path, monkeypatch,
    )
    queue_dir = tmp_path / "queue"
    manifest = build_adjudication_queue.build_queue(
        run_path, grade_paths, queue_dir,
    )
    completed = _complete_decisions(queue_dir, manifest)
    payload = json.loads(completed.read_text())
    payload["decisions"][0]["reviewer_ids"] = ["reviewer-opaque-1"]
    completed.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="two distinct reviewer_ids"):
        apply_adjudication.apply_decisions(
            run_path, grade_paths, queue_dir, completed,
        )

    payload["decisions"][0]["reviewer_ids"] = [
        "reviewer-opaque-1", "reviewer-opaque-2",
    ]
    payload["decisions"][0]["status"] = "pending"
    completed.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="not completed"):
        apply_adjudication.apply_decisions(
            run_path, grade_paths, queue_dir, completed,
        )

    cases_path = queue_dir / build_adjudication_queue.CASES_FILENAME
    cases_path.write_text(cases_path.read_text() + " ")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        apply_adjudication.apply_decisions(
            run_path, grade_paths, queue_dir, completed,
        )


def test_selection_treats_extraction_only_grade_as_candidate_error():
    rows = [{
        "id": "placeholder",
        "dataset_name": HLE_DATASET_NAME,
        "verified": True,
        "answer": "See reasoning above",
    }]
    graders = {
        "grader-one": {"placeholder": _grade(False, "extraction")},
        "grader-two": {"placeholder": _grade(False, "extraction")},
    }

    selected = build_adjudication_queue.select_adjudication_cases(
        rows,
        graders,
        target="final",
        seed="fixed",
        audit_fraction=0,
    )

    assert len(selected) == 1
    assert selected[0]["automatic_consensus_strict_correct"] is False
    assert selected[0]["selection_reasons"] == ["apparent_certified_error"]


def test_first_attempt_target_uses_first_candidate_and_certification(
    tmp_path, monkeypatch,
):
    rows = [
        {
            "id": "first-certified",
            "dataset_name": HLE_DATASET_NAME,
            "problem": "First-attempt question",
            "expected": "A",
            "answer": "B",
            "verified": True,
            "category": "Math",
            "repair_metrics": {
                "first_attempt_answer": "A",
                "first_attempt_verified": True,
            },
        },
        {
            "id": "final-only",
            "dataset_name": HLE_DATASET_NAME,
            "problem": "Final-only question",
            "expected": "B",
            "answer": "B",
            "verified": True,
            "category": "Science",
            "repair_metrics": {
                "first_attempt_answer": "A",
                "first_attempt_verified": False,
            },
        },
        {
            "id": "errored",
            "dataset_name": HLE_DATASET_NAME,
            "problem": "Errored question",
            "expected": "A",
            "answer": "A",
            "verified": True,
            "category": "Math",
            "error": "execution failed",
            "repair_metrics": {
                "first_attempt_answer": "A",
                "first_attempt_verified": True,
            },
        },
    ]
    graders = {
        grader: {str(row["id"]): _grade(True) for row in rows}
        for grader in ("grader-one", "grader-two")
    }
    rationales = {
        str(row["id"]): {
            "rationale": f"Rationale for {row['id']}",
            "is_valid": True,
            "error_type": None,
        }
        for row in rows
    }
    run_path = tmp_path / "run.json"
    run_path.write_text("[]")
    grade_paths = []
    for grader in graders:
        path = tmp_path / f"{grader}.json"
        path.write_text(json.dumps({
            "grading_target": "first_attempt",
            "grader_model": grader,
        }))
        grade_paths.append(path)

    monkeypatch.setattr(
        build_adjudication_queue.summarize_run,
        "_load_sealed_run",
        lambda _path: (rows, {"run_id": "repair-condition"}, "source-sha"),
    )
    monkeypatch.setattr(
        build_adjudication_queue.summarize_run,
        "_load_grades",
        lambda _paths, _sha, _target, _ids: graders,
    )
    frozen = lambda _paths, _ids: (rationales, "canonical-rationales-sha")
    monkeypatch.setattr(
        build_adjudication_queue, "_load_frozen_rationales", frozen,
    )
    monkeypatch.setattr(apply_adjudication, "_load_frozen_rationales", frozen)

    queue_dir = tmp_path / "queue-first-attempt"
    manifest = build_adjudication_queue.build_queue(
        run_path,
        grade_paths,
        queue_dir,
        target="first_attempt",
        audit_fraction=1,
    )
    cases = json.loads((queue_dir / "cases.json").read_text())["cases"]
    assert manifest["created_from"]["grading_target"] == "first_attempt"
    assert manifest["created_from"]["certified_count"] == 1
    assert len(cases) == 1
    assert cases[0]["candidate_answer"] == "A"

    completed = _complete_decisions(queue_dir, manifest)
    payload = json.loads(completed.read_text())
    payload["decisions"][0]["strict_correct"] = True
    completed.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = apply_adjudication.apply_decisions(
        run_path,
        grade_paths,
        queue_dir,
        completed,
        target="first_attempt",
    )

    assert report["grading_target"] == "first_attempt"
    assert report["cohort"]["certified"] == 1
    assert report["adjudicated_metrics"]["coverage"]["numerator"] == 1
    assert report["adjudicated_metrics"]["coverage"]["denominator"] == 3
    assert report["adjudicated_metrics"]["strict_correct_certified"] == 1
    assert report["ordered_outcomes"] == [
        {"id": "first-certified", "certified": True, "strict_correct": True},
        {"id": "final-only", "certified": False, "strict_correct": None},
        {"id": "errored", "certified": False, "strict_correct": None},
    ]
