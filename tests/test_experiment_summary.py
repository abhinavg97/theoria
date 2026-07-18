import json
from pathlib import Path

import pytest

from experiments import summarize_run


def test_wilson_interval_handles_empty_and_extreme_counts():
    assert summarize_run.wilson_interval(0, 0) is None
    low, high = summarize_run.wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.2775, abs=0.0001)
    low, high = summarize_run.wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=0.0001)
    assert high == pytest.approx(1.0)


def test_mcnemar_exact_reports_paired_gains_and_losses():
    result = summarize_run._mcnemar_exact(
        [True, True, False, False],
        [False, True, True, True],
    )
    assert result == {
        "lost": 1,
        "gained": 2,
        "discordant": 3,
        "two_sided_exact_p": 1.0,
    }


def test_strict_correct_requires_selected_candidate_key_match():
    assert summarize_run._strict_correct({"final": {"key_match": True}})
    extraction = {
        "final": {"key_match": False, "dispute_category": "extraction"},
    }
    assert not summarize_run._strict_correct(extraction)
    assert summarize_run._favorable_correct(extraction)
    assert not summarize_run._strict_correct({
        "final": {"key_match": False, "dispute_category": "interpretation"},
    })


def test_wrong_rates_and_asymmetry_use_strict_correctness():
    certified = {
        "grader": {"strict_correct": summarize_run._rate(9, 10)},
    }
    declined = {
        "grader": {"strict_correct": summarize_run._rate(5, 10)},
    }

    certified_wrong = summarize_run._wrong_rates(certified)
    declined_wrong = summarize_run._wrong_rates(declined)

    assert certified_wrong["grader"]["numerator"] == 1
    assert declined_wrong["grader"]["numerator"] == 5
    assert summarize_run._wrong_rate_asymmetry(
        certified_wrong, declined_wrong,
    )["grader"] == pytest.approx(5.0)


def test_wrong_rate_asymmetry_is_undefined_with_zero_certified_errors():
    certified = {"grader": summarize_run._rate(0, 10)}
    declined = {"grader": summarize_run._rate(2, 10)}

    assert summarize_run._wrong_rate_asymmetry(
        certified, declined,
    )["grader"] is None


def test_load_grades_rejects_source_hash_mismatch(tmp_path, monkeypatch):
    grade_path = tmp_path / "grade.json"
    grade_path.write_text(json.dumps({
        "grading_target": "final",
        "run_file_sha256": "wrong",
        "grader_model": "test",
        "results": [],
    }))
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        summarize_run._load_grades(
            [grade_path], "expected", "final", set(),
        )


def test_summary_keeps_execution_errors_in_itt_denominator(monkeypatch, tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text("[]")
    rows = [
        {
            "id": "certified",
            "verified": True,
            "category": "Math",
            "repair_metrics": {
                "first_attempt_verified": True,
                "repair_attempted": False,
                "certified_by_repair": False,
            },
            "metrics": {},
        },
        {
            "id": "transport-error",
            "verified": False,
            "error": "timeout",
            "category": "Math",
            "repair_metrics": None,
            "metrics": {},
        },
    ]
    monkeypatch.setattr(
        summarize_run,
        "_load_sealed_run",
        lambda _path: (rows, {"research_audit": {}}, "source-sha"),
    )

    report = summarize_run.summarize(run_path, [], [])

    assert report["cohort"]["intent_to_treat_denominator"] == 2
    assert report["cohort"]["execution_errors"] == 1
    assert report["cohort"]["repair_metrics_complete_problems"] == 1
    assert report["cohort"]["repair_metrics_missing_execution_errors"] == 1
    assert report["operating_points"]["r1_final"]["coverage"]["numerator"] == 1
    assert report["operating_points"]["r1_final"]["coverage"]["denominator"] == 2
    assert report["repair"]["metric_scope"]["missing_execution_error_rows"] == 1
    assert report["repair"]["yield_missing_errors_as_attempted"]["denominator"] == 1


def test_summary_rejects_missing_repair_metrics_without_execution_error(
    monkeypatch, tmp_path,
):
    run_path = tmp_path / "run.json"
    run_path.write_text("[]")
    rows = [{
        "id": "bad",
        "verified": False,
        "repair_metrics": None,
        "metrics": {},
    }]
    monkeypatch.setattr(
        summarize_run,
        "_load_sealed_run",
        lambda _path: (rows, {"research_audit": {}}, "source-sha"),
    )

    with pytest.raises(RuntimeError, match="missing repair metrics"):
        summarize_run.summarize(run_path, [], [])


def test_summary_uses_first_attempt_grades_for_r0_and_repair_transitions(
    monkeypatch, tmp_path,
):
    run_path = tmp_path / "run.json"
    run_path.write_text("[]")
    rows = [{
        "id": "repaired",
        "verified": True,
        "repair_metrics": {
            "first_attempt_verified": False,
            "repair_attempted": True,
            "certified_by_repair": True,
        },
        "metrics": {},
    }, {
        "id": "first-pass",
        "verified": True,
        "repair_metrics": {
            "first_attempt_verified": True,
            "repair_attempted": False,
            "certified_by_repair": False,
        },
        "metrics": {},
    }]
    monkeypatch.setattr(
        summarize_run,
        "_load_sealed_run",
        lambda _path: (rows, {"research_audit": {}}, "source-sha"),
    )
    final = {
        "repaired": {"answer": "42", "final": {"key_match": True}},
        "first-pass": {"answer": "42", "final": {"key_match": True}},
    }
    first = {
        "repaired": {"answer": "41", "final": {"key_match": False}},
        "first-pass": {"answer": "41", "final": {"key_match": False}},
    }

    def fake_load(_paths, _sha, target, _ids):
        if target == "final":
            return {"grader": final}
        if target == "first_attempt":
            return {"grader": first}
        return {}

    monkeypatch.setattr(summarize_run, "_load_grades", fake_load)

    report = summarize_run.summarize(
        run_path, [tmp_path / "final"], [], [tmp_path / "first"],
    )

    r0 = report["operating_points"]["r0_first_pass"]["certified_precision"]
    r1 = report["operating_points"]["r1_final"]["certified_precision"]
    assert r0["grader"]["strict_correct"]["numerator"] == 0
    assert r1["grader"]["strict_correct"]["numerator"] == 2
    transitions = report["repair"]["correctness_transitions"]["grader"]
    assert transitions["transition_counts"] == {"incorrect_to_correct": 1}
    consensus = report["repair"]["correctness_transitions"][
        "all_graders_consensus"
    ]
    assert consensus["transition_counts"] == {"incorrect_to_correct": 1}
    assert "all_graders_consensus" in report["paired_changes"]["by_grader"]


def test_repair_transitions_separate_missing_candidate_from_incorrect():
    first = {"grader": {
        "p1": {
            "answer": None,
            "grade_source": "deterministic_missing_target",
            "final": {"key_match": False},
        },
    }}
    final = {"grader": {
        "p1": {"answer": "42", "final": {"key_match": True}},
    }}

    transitions = summarize_run._paired_correctness_transitions(
        first, final, ["p1"],
    )["grader"]

    assert transitions["transition_counts"] == {"missing_to_correct": 1}
    assert transitions["incorrect_to_correct_rate"]["denominator"] == 0


def test_summary_rejects_different_graders_across_targets(monkeypatch, tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text("[]")
    rows = [{
        "id": "p1",
        "verified": True,
        "repair_metrics": {
            "first_attempt_verified": True,
            "repair_attempted": False,
            "certified_by_repair": False,
        },
        "metrics": {},
    }]
    monkeypatch.setattr(
        summarize_run,
        "_load_sealed_run",
        lambda _path: (rows, {"research_audit": {}}, "source-sha"),
    )

    def fake_load(_paths, _sha, target, _ids):
        grader = "final-grader" if target == "final" else "first-grader"
        return {grader: {"p1": {"answer": "A", "final": {"key_match": True}}}}

    monkeypatch.setattr(summarize_run, "_load_grades", fake_load)

    with pytest.raises(RuntimeError, match="identities differ"):
        summarize_run.summarize(
            run_path, [tmp_path / "final"], [], [tmp_path / "first"],
        )
