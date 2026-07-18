import pytest

from experiments import summarize_gpqa
from experiments.summarize_gpqa import (
    _bare_letter_option_text_collision,
    _options_from_question,
    extract_choice,
)


QUESTION = "Q\nOptions:\nA) one\nB) two\nC) three\nD) four"


def _proof(answer, passed):
    return {
        "phase": "verify",
        "all_ok": passed,
        "proof": {"steps": [{"state": [answer], "justification": "test"}]},
    }


def _repair(
    first_answer,
    first_verified,
    *,
    attempted,
    certified,
    verify_attempts,
    solver_retries=0,
    rejects=0,
    invalids=0,
    changed=False,
    observable=True,
):
    return {
        "first_attempt_answer": first_answer,
        "first_attempt_verified": first_verified,
        "repair_attempted": attempted,
        "certified_by_repair": certified,
        "verify_attempts": verify_attempts,
        "judge_repair_rounds": max(0, verify_attempts - 1),
        "solver_retries": solver_retries,
        "formalizer_reject_count": rejects,
        "formalizer_invalid_count": invalids,
        "answer_change_observable": observable,
        "answer_changed_during_repair": changed if observable else None,
        "normalized_answer_changed_during_repair": changed if observable else None,
        "solver_solution_text_changed_during_repair": solver_retries > 0,
    }


def _telemetry(
    calls=1,
    *,
    available=None,
    complete=None,
    priced=0,
    input_tokens=10,
    output_tokens=5,
    llm_ms=100,
    problem_ms=150,
    partial_cost=None,
):
    return {
        "num_calls": calls,
        "usage_available_calls": calls if available is None else available,
        "usage_complete_calls": calls if complete is None else complete,
        "priced_calls": priced,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cache_read_input_tokens": 1,
        "total_llm_duration_ms": llm_ms,
        "problem_duration_ms": problem_ms,
        "partial_cost_usd": partial_cost,
    }


def _result(
    problem_id,
    expected,
    answer,
    verified,
    attempts,
    repair,
    *,
    telemetry=None,
):
    return {
        "id": problem_id,
        "problem": QUESTION,
        "category": "science",
        "expected": expected,
        "answer": answer,
        "verified": verified,
        "solver_solutions": [f"Answer: {expected.lower()} because solver"],
        "attempts": attempts,
        "repair_metrics": repair,
        "metrics": telemetry or _telemetry(),
    }


def test_extract_choice_accepts_safe_lowercase_explicit_answers():
    assert extract_choice("Answer: a because this follows") == "A"
    assert extract_choice("Answer: d because this follows") == "D"
    assert extract_choice("The final answer is c because this follows") == "C"
    assert extract_choice("answer=d, since the calculation gives it") == "D"


def test_extract_choice_does_not_treat_prose_as_lowercase_label():
    assert extract_choice("The answer is a function of pressure") is None
    assert extract_choice("The final answer is a function of pressure") is None
    assert extract_choice("Answer: density changes with temperature") is None


def test_extract_choice_and_option_helpers_preserve_existing_policy():
    assert extract_choice("B") == "B"
    assert extract_choice("Answer: (c)") == "C"
    assert extract_choice("A. Derivation. Final answer: C") == "C"
    options = _options_from_question(
        "Q\nOptions:\nA) first\ncontinued\nB) second\nC) d\nD) c\n"
    )
    assert options["A"] == "first\ncontinued"
    assert extract_choice("second", options) == "B"
    assert _bare_letter_option_text_collision("C", options) is True


def test_summary_reports_repair_transitions_stages_flips_and_telemetry(
    monkeypatch, tmp_path,
):
    run = tmp_path / "gpqa.json"
    run.write_text("[]")
    results = [
        _result(
            "repair-correct", "B", "Answer: b because fixed", True,
            [_proof("A", False), _proof("B", True)],
            _repair(
                "A", False, attempted=True, certified=True,
                verify_attempts=2, changed=True,
            ),
            telemetry=_telemetry(
                calls=2, priced=2, input_tokens=20, output_tokens=8,
                partial_cost=0.2,
            ),
        ),
        _result(
            "first-pass", "C", "C", True,
            [_proof("C", True)],
            _repair(
                "C", True, attempted=False, certified=False,
                verify_attempts=1,
            ),
            telemetry=_telemetry(
                calls=1, priced=0, input_tokens=6, output_tokens=2,
            ),
        ),
        _result(
            "repair-damage", "A", "B", False,
            [_proof("A", False), _proof("B", False)],
            _repair(
                "A", False, attempted=True, certified=False,
                verify_attempts=2, changed=True,
            ),
            telemetry=_telemetry(calls=2, available=1, complete=1),
        ),
        _result(
            "missing-recovered", "D", "D", True,
            [
                {"phase": "formalizer_reject", "reject_reason": "retry"},
                _proof("D", True),
            ],
            _repair(
                None, False, attempted=True, certified=True,
                verify_attempts=1, solver_retries=1, rejects=1,
                observable=False,
            ),
        ),
        _result(
            "repair-wrong", "A", "B", True,
            [_proof("C", False), _proof("B", True)],
            _repair(
                "C", False, attempted=True, certified=True,
                verify_attempts=2, changed=True,
            ),
        ),
    ]
    monkeypatch.setattr(
        summarize_gpqa,
        "_load_sealed_run",
        lambda _path: (results, {"run_id": "run"}, "sha"),
    )

    report = summarize_gpqa.summarize(run)
    metrics = report["metrics"]

    assert metrics["r0_first_pass_coverage"]["numerator"] == 1
    assert metrics["coverage"]["numerator"] == 4
    assert metrics["repair"]["certified_by_repair"] == 3
    assert metrics["repair"]["precision_among_certifications"]["numerator"] == 2
    transitions = metrics["repair"]["correctness_transitions"]
    assert transitions["transition_counts"] == {
        "missing_to_incorrect": 0,
        "missing_to_correct": 1,
        "incorrect_to_incorrect": 1,
        "incorrect_to_correct": 1,
        "correct_to_incorrect": 1,
        "correct_to_correct": 0,
    }
    assert transitions["incorrect_to_correct_rate"]["value"] == 0.5
    assert transitions["correct_to_incorrect_damage_rate"]["value"] == 1.0

    stages = metrics["certification_stages"]
    assert stages["r0_first_pass"]["certified"] == 1
    assert stages["repair"]["certified"] == 3
    assert stages["repair"]["errors_shipped"] == 1
    assert metrics["certification_by_pipeline_attempt"]["1"]["certified"] == 1
    assert metrics["certification_by_pipeline_attempt"]["2"]["certified"] == 3
    assert metrics["certification_by_verify_attempt"]["1"]["certified"] == 2
    assert metrics["certification_by_verify_attempt"]["2"]["certified"] == 2
    assert metrics["certification_by_pipeline_attempt"]["2"][
        "certified_precision"
    ]["numerator"] == 2

    paired = metrics["paired_changes"]
    assert paired["coverage_r1_minus_r0"]["estimate"] == 0.6
    assert paired["coverage_r1_minus_r0"]["paired_bootstrap_95"] is not None
    assert paired["coverage_r1_minus_r0"]["mcnemar"] == {
        "lost": 0,
        "gained": 3,
        "discordant": 3,
        "two_sided_exact_p": 0.25,
    }
    correct_certified = paired["correct_and_certified_rate_r1_minus_r0"]
    assert correct_certified["estimate"] == 0.4
    assert correct_certified["paired_bootstrap_95"] is not None
    assert correct_certified["mcnemar"]["gained"] == 2
    precision_change = paired["certified_precision_r1_minus_r0"]
    assert precision_change["estimate"] == -0.25
    assert precision_change["paired_bootstrap_95"] is not None

    flips = metrics["repair"]["certified_answer_flips"]
    assert flips == {
        "certified_by_repair": 3,
        "observable": 2,
        "unobservable": 1,
        "exact_changed": 2,
        "exact_unchanged": 0,
        "normalized_changed": 2,
        "normalized_unchanged": 0,
    }
    assert metrics["repair"]["solver_retries"] == 1
    assert metrics["repair"]["formalizer_reject_count"] == 1

    telemetry = metrics["telemetry"]
    assert telemetry["calls"] == 7
    assert telemetry["total_input_tokens"] == 56
    assert telemetry["total_output_tokens"] == 25
    assert telemetry["usage_complete_calls"] == 6
    assert telemetry["priced_calls"] == 2
    assert telemetry["partial_cost_usd"] == pytest.approx(0.2)
    assert telemetry["total_cost_usd"] is None


def test_summary_rejects_certified_by_repair_mismatch(monkeypatch, tmp_path):
    run = tmp_path / "gpqa.json"
    run.write_text("[]")
    result = _result(
        "bad", "B", "B", True,
        [_proof("A", False), _proof("B", True)],
        _repair(
            "A", False, attempted=True, certified=False,
            verify_attempts=2, changed=True,
        ),
    )
    monkeypatch.setattr(
        summarize_gpqa,
        "_load_sealed_run",
        lambda _path: ([result], {"run_id": "run"}, "sha"),
    )

    with pytest.raises(RuntimeError, match="certified_by_repair"):
        summarize_gpqa.summarize(run)


def test_summary_rejects_r0_certification_that_disappears(monkeypatch, tmp_path):
    run = tmp_path / "gpqa.json"
    run.write_text("[]")
    result = _result(
        "bad", "B", "B", False,
        [_proof("B", True)],
        _repair(
            "B", True, attempted=False, certified=False,
            verify_attempts=1,
        ),
    )
    monkeypatch.setattr(
        summarize_gpqa,
        "_load_sealed_run",
        lambda _path: ([result], {"run_id": "run"}, "sha"),
    )

    with pytest.raises(RuntimeError, match="declined row contains a passing attempt"):
        summarize_gpqa.summarize(run)


def test_summary_keeps_execution_errors_in_itt_and_yield_bound(monkeypatch, tmp_path):
    run = tmp_path / "gpqa.json"
    run.write_text("[]")
    completed = _result(
        "complete", "B", "B", True,
        [_proof("B", True)],
        _repair(
            "B", True, attempted=False, certified=False,
            verify_attempts=1,
        ),
    )
    failed = {
        "id": "failed",
        "problem": QUESTION,
        "category": "science",
        "expected": "A",
        "answer": None,
        "verified": False,
        "solver_solutions": [],
        "attempts": [],
        "repair_metrics": None,
        "metrics": _telemetry(calls=0, available=0, complete=0),
        "error": "provider timeout",
    }
    monkeypatch.setattr(
        summarize_gpqa,
        "_load_sealed_run",
        lambda _path: ([completed, failed], {"run_id": "run"}, "sha"),
    )

    metrics = summarize_gpqa.summarize(run)["metrics"]

    assert metrics["coverage"]["denominator"] == 2
    assert metrics["execution_errors"] == 1
    scope = metrics["repair"]["metric_scope"]
    assert scope["missing_execution_error_rows"] == 1
    conservative = metrics["repair"]["yield_missing_errors_as_attempted"]
    assert conservative["denominator"] == 1
