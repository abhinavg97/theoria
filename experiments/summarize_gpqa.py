#!/usr/bin/env python3
"""Build deterministic publication metrics for a sealed GPQA run."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.summarize_run import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    _load_sealed_run,
    _mcnemar_exact,
    _paired_bootstrap_interval,
    _rate,
)


BARE_CHOICE_RE = re.compile(
    r"^\s*[\[(]?([A-D])[\])\].:]?\s*$",
    re.IGNORECASE,
)
LABELED_CHOICE_RE = re.compile(
    r"^\s*[\[(]?([A-D])[\])\].:]\s*\S.*$",
    re.IGNORECASE | re.DOTALL,
)
EXPLICIT_ANSWER_RE = re.compile(
    r"(?i:\b(?:final\s+)?answer\s*(?:(?:is|=|:)\s*)?)"
    r"(?:[\[(]([A-Da-d])[\])\].:]?|([A-D])(?:[\])\].:]|\b)"
    r"|([a-d])(?:[\])\].:]|\s*$))",
)
# A delimiter makes a lowercase A-D explicit even when an explanation follows.
LOWERCASE_DELIMITED_ANSWER_RE = re.compile(
    r"(?i:\b(?:final\s+)?answer\s*[:=]\s*)"
    r"[\[(]?([a-d])[\])\].:]?(?=\s|[,;!?]|\.$|$)",
)
# With "final answer is", lowercase b-d remain unambiguous. Lowercase "a" is
# excluded because it can be an English article ("the answer is a function").
LOWERCASE_FINAL_IS_ANSWER_RE = re.compile(
    r"(?i:\bfinal\s+answer\s+is\s+)"
    r"[\[(]?([b-d])[\])\].:]?(?=\s|[,;!?]|\.$|$)",
)

TRANSITION_KEYS = (
    "missing_to_incorrect",
    "missing_to_correct",
    "incorrect_to_incorrect",
    "incorrect_to_correct",
    "correct_to_incorrect",
    "correct_to_correct",
)


def _normalize(value: object) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).casefold().split()
    ).strip(" .")


def extract_choice(answer: object, options: dict[str, str] | None = None) -> str | None:
    if answer is None:
        return None
    text = str(answer)
    normalized = _normalize(text)
    bare = BARE_CHOICE_RE.match(text)
    if bare:
        return bare.group(1).upper()
    matches = [
        letter for letter, option in (options or {}).items()
        if normalized == _normalize(option)
    ]
    if len(matches) == 1:
        return matches[0]

    explicit_matches: list[tuple[int, str]] = []
    for match in EXPLICIT_ANSWER_RE.finditer(text):
        explicit_matches.append((
            match.start(),
            next(group.upper() for group in match.groups() if group is not None),
        ))
    for pattern in (
        LOWERCASE_DELIMITED_ANSWER_RE,
        LOWERCASE_FINAL_IS_ANSWER_RE,
    ):
        for match in pattern.finditer(text):
            explicit_matches.append((match.start(), match.group(1).upper()))
    if explicit_matches:
        return max(explicit_matches, key=lambda item: item[0])[1]

    match = LABELED_CHOICE_RE.match(text)
    return match.group(1).upper() if match else None


def _options_from_question(question: str) -> dict[str, str]:
    matches = re.finditer(
        r"(?ms)^[ \t]*([A-D])\)[ \t]*(.*?)"
        r"(?=^[ \t]*[A-D]\)[ \t]*|\Z)",
        question,
    )
    return {match.group(1): match.group(2).strip() for match in matches}


def _bare_letter_option_text_collision(
    answer: object, options: dict[str, str],
) -> bool:
    if answer is None:
        return False
    match = BARE_CHOICE_RE.match(str(answer))
    if not match:
        return False
    label = match.group(1).upper()
    normalized = _normalize(answer)
    return any(
        letter != label and normalized == _normalize(option)
        for letter, option in options.items()
    )


def _attempt_answer(attempt: dict) -> str | None:
    proof = attempt.get("proof") or {}
    steps = proof.get("steps") or []
    if not steps:
        return None
    state = steps[-1].get("state") or []
    if not state:
        return None
    return str(state[0])


def _certification_attempts(attempts: list[dict]) -> list[tuple[int, int]]:
    passed = []
    verify_ordinal = 0
    for pipeline_attempt, attempt in enumerate(attempts, 1):
        if attempt.get("phase") != "verify":
            continue
        verify_ordinal += 1
        if attempt.get("all_ok") is True:
            passed.append((pipeline_attempt, verify_ordinal))
    return passed


def _validate_repair_trace(result: dict, repair: dict) -> tuple[int | None, int | None]:
    problem_id = str(result.get("id"))
    attempts = result.get("attempts") or []
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise RuntimeError(f"{problem_id}: attempts must be a list of objects")

    first = attempts[0] if attempts else None
    derived_first_verified = bool(
        first and first.get("phase") == "verify" and first.get("all_ok") is True
    )
    if bool(repair.get("first_attempt_verified")) != derived_first_verified:
        raise RuntimeError(f"{problem_id}: first_attempt_verified disagrees with trace")
    derived_first_answer = (
        _attempt_answer(first)
        if first and first.get("phase") == "verify"
        else None
    )
    reported_first_answer = repair.get("first_attempt_answer")
    if reported_first_answer is not None:
        reported_first_answer = str(reported_first_answer)
    if reported_first_answer != derived_first_answer:
        raise RuntimeError(f"{problem_id}: first_attempt_answer disagrees with trace")

    verify_attempts = sum(item.get("phase") == "verify" for item in attempts)
    if "verify_attempts" in repair and int(repair.get("verify_attempts") or 0) != verify_attempts:
        raise RuntimeError(f"{problem_id}: verify_attempts disagrees with trace")
    expected_judge_rounds = max(0, verify_attempts - 1)
    if int(repair.get("judge_repair_rounds") or 0) != expected_judge_rounds:
        raise RuntimeError(f"{problem_id}: judge_repair_rounds disagrees with trace")
    for field, phase in (
        ("formalizer_reject_count", "formalizer_reject"),
        ("formalizer_invalid_count", "formalizer_invalid"),
    ):
        expected = sum(item.get("phase") == phase for item in attempts)
        if int(repair.get(field) or 0) != expected:
            raise RuntimeError(f"{problem_id}: {field} disagrees with trace")

    solver_retries = int(repair.get("solver_retries") or 0)
    expected_repair_attempted = bool(
        expected_judge_rounds or solver_retries or len(attempts) > 1
    )
    if bool(repair.get("repair_attempted")) != expected_repair_attempted:
        raise RuntimeError(f"{problem_id}: repair_attempted disagrees with trace")

    observable = bool(repair.get("answer_change_observable"))
    exact_changed = repair.get("answer_changed_during_repair")
    normalized_changed = repair.get("normalized_answer_changed_during_repair")
    if observable and not all(
        isinstance(value, bool) for value in (exact_changed, normalized_changed)
    ):
        raise RuntimeError(f"{problem_id}: observable answer change needs booleans")
    if not observable and any(
        value is not None for value in (exact_changed, normalized_changed)
    ):
        raise RuntimeError(f"{problem_id}: unobservable answer change has a verdict")

    passed = _certification_attempts(attempts)
    verified = bool(result.get("verified") and not result.get("error"))
    if verified and len(passed) != 1:
        raise RuntimeError(f"{problem_id}: verified row needs exactly one passing attempt")
    if not verified and passed:
        raise RuntimeError(f"{problem_id}: declined row contains a passing attempt")
    return passed[0] if passed else (None, None)


def _telemetry(rows: list[dict]) -> dict:
    calls = 0
    usage_available = 0
    usage_complete = 0
    priced_calls = 0
    input_tokens = 0
    output_tokens = 0
    cache_tokens = 0
    llm_duration_ms = 0
    problem_duration_ms = 0
    partial_costs = []
    partial_cost_complete = True
    for row in rows:
        metrics = row["telemetry"]
        row_calls = int(metrics.get("num_calls", 0) or 0)
        row_available = int(metrics.get("usage_available_calls", 0) or 0)
        row_complete = int(metrics.get("usage_complete_calls", 0) or 0)
        row_priced = int(metrics.get("priced_calls", 0) or 0)
        if not 0 <= row_complete <= row_available <= row_calls:
            raise RuntimeError(f"{row['id']}: invalid usage coverage telemetry")
        if not 0 <= row_priced <= row_calls:
            raise RuntimeError(f"{row['id']}: invalid pricing coverage telemetry")
        calls += row_calls
        usage_available += row_available
        usage_complete += row_complete
        priced_calls += row_priced
        input_tokens += int(metrics.get("total_input_tokens", 0) or 0)
        output_tokens += int(metrics.get("total_output_tokens", 0) or 0)
        cache_tokens += int(metrics.get("total_cache_read_input_tokens", 0) or 0)
        llm_duration_ms += int(metrics.get("total_llm_duration_ms", 0) or 0)
        problem_duration_ms += int(metrics.get("problem_duration_ms", 0) or 0)
        if row_priced:
            partial_cost = metrics.get("partial_cost_usd")
            if partial_cost is None:
                partial_cost_complete = False
            else:
                partial_costs.append(float(partial_cost))

    partial_cost = (
        sum(partial_costs)
        if priced_calls and partial_cost_complete
        else None
    )
    return {
        "calls": calls,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_cache_read_input_tokens": cache_tokens,
        "usage_available_calls": usage_available,
        "usage_available_fraction": usage_available / calls if calls else None,
        "usage_complete_calls": usage_complete,
        "usage_complete_fraction": usage_complete / calls if calls else None,
        "total_llm_duration_ms": llm_duration_ms,
        "total_problem_duration_ms": problem_duration_ms,
        "mean_problem_duration_ms": problem_duration_ms / len(rows) if rows else None,
        "priced_calls": priced_calls,
        "cost_coverage_fraction": priced_calls / calls if calls else None,
        "partial_cost_usd": partial_cost,
        "total_cost_usd": (
            partial_cost if calls and priced_calls == calls else None
        ),
    }


def _certification_summary(certified: list[dict], denominator: int) -> dict:
    correct = sum(row["correct"] for row in certified)
    return {
        "certified": len(certified),
        "correct": correct,
        "errors_shipped": len(certified) - correct,
        "marginal_coverage": _rate(len(certified), denominator),
        "certified_precision": _rate(correct, len(certified)),
    }


def _by_certification_attempt(
    rows: list[dict], field: str,
) -> dict[str, dict]:
    output = {}
    values = sorted({row[field] for row in rows if row["verified"]})
    for value in values:
        certified = [row for row in rows if row["verified"] and row[field] == value]
        output[str(value)] = _certification_summary(certified, len(rows))
    return output


def _repair_transitions(rows: list[dict]) -> dict:
    counts = Counter({key: 0 for key in TRANSITION_KEYS})
    for row in rows:
        before = (
            "missing" if row["first_attempt_missing"]
            else "correct" if row["first_attempt_correct"]
            else "incorrect"
        )
        after = "correct" if row["correct"] else "incorrect"
        counts[f"{before}_to_{after}"] += 1
    return {
        "problems": len(rows),
        "transition_counts": {key: counts[key] for key in TRANSITION_KEYS},
        "incorrect_to_correct_rate": _rate(
            counts["incorrect_to_correct"],
            counts["incorrect_to_correct"] + counts["incorrect_to_incorrect"],
        ),
        "correct_to_incorrect_damage_rate": _rate(
            counts["correct_to_incorrect"],
            counts["correct_to_incorrect"] + counts["correct_to_correct"],
        ),
        "missing_to_correct_rate": _rate(
            counts["missing_to_correct"],
            counts["missing_to_correct"] + counts["missing_to_incorrect"],
        ),
    }


def _paired_changes(rows: list[dict]) -> dict:
    records = [(
        row["first_attempt_verified"],
        row["verified"],
        row["first_attempt_verified"] and row["first_attempt_correct"],
        row["verified"] and row["correct"],
    ) for row in rows]

    def coverage_difference(sample):
        return sum(r1 - r0 for r0, r1, _c0, _c1 in sample) / len(sample)

    def correct_certified_rate_difference(sample):
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
    coverage_estimate = (
        sum(record[1] - record[0] for record in records) / len(records)
        if records else None
    )
    correct_certified_estimate = (
        sum(record[3] - record[2] for record in records) / len(records)
        if records else None
    )
    precision_estimate = (
        sum(record[3] for record in records) / r1_total
        - sum(record[2] for record in records) / r0_total
        if r0_total and r1_total else None
    )
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "coverage_r1_minus_r0": {
            "estimate": coverage_estimate,
            "paired_bootstrap_95": _paired_bootstrap_interval(
                records, coverage_difference,
            ),
            "mcnemar": _mcnemar_exact(
                [record[0] for record in records],
                [record[1] for record in records],
            ),
        },
        "correct_and_certified_rate_r1_minus_r0": {
            "estimate": correct_certified_estimate,
            "paired_bootstrap_95": _paired_bootstrap_interval(
                records, correct_certified_rate_difference, seed_offset=1,
            ),
            "mcnemar": _mcnemar_exact(
                [record[2] for record in records],
                [record[3] for record in records],
            ),
        },
        "certified_precision_r1_minus_r0": {
            "estimate": precision_estimate,
            "paired_bootstrap_95": (
                _paired_bootstrap_interval(
                    records, precision_difference, seed_offset=2,
                )
                if r0_total and r1_total else None
            ),
        },
    }


def _metrics(selected: list[dict]) -> dict:
    certified = [row for row in selected if row["verified"]]
    declined = [row for row in selected if not row["verified"]]
    first_pass = [row for row in selected if row["first_attempt_verified"]]
    repair_attempted = [row for row in selected if row["repair_attempted"]]
    repair_certified = [row for row in selected if row["certified_by_repair"]]
    repair_complete = [row for row in selected if row["repair_metrics_complete"]]
    certified_observable = [
        row for row in repair_certified if row["answer_change_observable"]
    ]
    certified_exact_changed = sum(
        row["answer_changed_during_repair"] is True for row in repair_certified
    )
    certified_normalized_changed = sum(
        row["normalized_answer_changed_during_repair"] is True
        for row in repair_certified
    )

    return {
        "problems": len(selected),
        "coverage": _rate(len(certified), len(selected)),
        "final_candidate_accuracy": _rate(
            sum(row["correct"] for row in selected), len(selected),
        ),
        "raw_solver_accuracy": _rate(
            sum(row["solver_correct"] for row in selected), len(selected),
        ),
        "solver_missing": sum(row["solver_missing"] for row in selected),
        "solver_needs_extraction_review": sum(
            row["solver_needs_extraction_review"] for row in selected
        ),
        "r0_first_pass_coverage": _rate(len(first_pass), len(selected)),
        "r0_first_pass_certified_precision": _rate(
            sum(row["first_attempt_correct"] for row in first_pass),
            len(first_pass),
        ),
        "r0_first_pass_errors_shipped": sum(
            not row["first_attempt_correct"] for row in first_pass
        ),
        "first_attempt_missing": sum(row["first_attempt_missing"] for row in selected),
        "first_attempt_needs_extraction_review": sum(
            row["first_attempt_needs_extraction_review"] for row in selected
        ),
        "certified_precision": _rate(
            sum(row["correct"] for row in certified), len(certified),
        ),
        "paired_changes": _paired_changes(selected),
        "declined_wrong_rate": _rate(
            sum(not row["correct"] for row in declined), len(declined),
        ),
        "errors_shipped": sum(not row["correct"] for row in certified),
        "certification_stages": {
            "r0_first_pass": _certification_summary(first_pass, len(selected)),
            "repair": _certification_summary(repair_certified, len(selected)),
            "final": _certification_summary(certified, len(selected)),
        },
        "certification_by_pipeline_attempt": _by_certification_attempt(
            selected, "certification_pipeline_attempt",
        ),
        "certification_by_verify_attempt": _by_certification_attempt(
            selected, "certification_verify_attempt",
        ),
        "repair": {
            "attempted": len(repair_attempted),
            "semantic_repair_attempted": sum(
                row["semantic_repair_attempted"] for row in selected
            ),
            "post_judge_repair_attempted": sum(
                row["post_judge_repair_attempted"] for row in selected
            ),
            "formalizer_invalid_retry_attempted": sum(
                row["formalizer_invalid_retry_attempted"] for row in selected
            ),
            "certified_after_nonsemantic_retry": sum(
                row["certified_after_nonsemantic_retry"] for row in selected
            ),
            "certified_by_repair": len(repair_certified),
            "yield": _rate(len(repair_certified), len(repair_attempted)),
            "yield_missing_errors_as_attempted": _rate(
                len(repair_certified),
                len(repair_attempted) + len(selected) - len(repair_complete),
            ),
            "precision_among_certifications": _rate(
                sum(row["correct"] for row in repair_certified),
                len(repair_certified),
            ),
            "correctness_transitions": _repair_transitions(repair_attempted),
            "solver_retries": sum(row["solver_retries"] for row in selected),
            "judge_repair_rounds": sum(
                row["judge_repair_rounds"] for row in selected
            ),
            "formalizer_reject_count": sum(
                row["formalizer_reject_count"] for row in selected
            ),
            "formalizer_invalid_count": sum(
                row["formalizer_invalid_count"] for row in selected
            ),
            "answer_changed": sum(
                row["answer_changed_during_repair"] is True for row in selected
            ),
            "normalized_answer_changed": sum(
                row["normalized_answer_changed_during_repair"] is True
                for row in selected
            ),
            "answer_change_unobservable": sum(
                row["repair_attempted"] and not row["answer_change_observable"]
                for row in selected
            ),
            "solver_solution_text_changed": sum(
                row["solver_solution_text_changed_during_repair"] is True
                for row in selected
            ),
            "certified_answer_flips": {
                "certified_by_repair": len(repair_certified),
                "observable": len(certified_observable),
                "unobservable": len(repair_certified) - len(certified_observable),
                "exact_changed": certified_exact_changed,
                "exact_unchanged": len(certified_observable) - certified_exact_changed,
                "normalized_changed": certified_normalized_changed,
                "normalized_unchanged": (
                    len(certified_observable) - certified_normalized_changed
                ),
            },
            "metric_scope": {
                "complete_rows": len(repair_complete),
                "missing_execution_error_rows": len(selected) - len(repair_complete),
                "note": (
                    "Repair process counts use complete rows and are lower bounds. "
                    "Complete-case yield can be optimistic; the conservative yield "
                    "treats missing execution-error rows as attempted failures."
                ),
            },
        },
        "execution_errors": sum(row["execution_error"] for row in selected),
        "needs_extraction_review": sum(
            row["needs_extraction_review"] for row in selected
        ),
        "telemetry": _telemetry(selected),
    }


def summarize(run_path: Path) -> dict:
    results, meta, source_sha = _load_sealed_run(run_path)
    ids = [str(result.get("id")) for result in results]
    if len(ids) != len(set(ids)):
        raise RuntimeError("source run has duplicate problem ids")

    rows = []
    domains: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        problem_id = str(result.get("id"))
        expected = str(result.get("expected") or "").strip().upper()
        if expected not in {"A", "B", "C", "D"}:
            raise RuntimeError(
                f"{problem_id}: expected answer is not one of A/B/C/D"
            )
        options = _options_from_question(str(result.get("problem") or ""))
        if set(options) != {"A", "B", "C", "D"}:
            raise RuntimeError(f"{problem_id}: could not recover exactly four options")
        telemetry = result.get("metrics")
        if not isinstance(telemetry, dict):
            raise RuntimeError(f"{problem_id}: missing per-problem telemetry")

        execution_error = bool(result.get("error"))
        if execution_error and result.get("verified"):
            raise RuntimeError(f"{problem_id}: execution-error row cannot be verified")
        repair_metrics_complete = isinstance(result.get("repair_metrics"), dict)
        if not repair_metrics_complete and not execution_error:
            raise RuntimeError(f"{problem_id}: successful row is missing repair metrics")
        repair = result.get("repair_metrics") or {}
        certification_pipeline_attempt = None
        certification_verify_attempt = None
        if repair_metrics_complete:
            (
                certification_pipeline_attempt,
                certification_verify_attempt,
            ) = _validate_repair_trace(result, repair)
        attempts = result.get("attempts") or []
        post_judge_repair_attempted = repair_metrics_complete and any(
            attempt.get("phase") == "verify"
            and attempt.get("all_ok") is False
            and index + 1 < len(attempts)
            for index, attempt in enumerate(attempts)
        )
        semantic_repair_attempted = bool(
            int(repair.get("solver_retries") or 0)
            or post_judge_repair_attempted
        )
        formalizer_invalid_retry_attempted = bool(
            repair_metrics_complete
            and any(
                item.get("phase") == "formalizer_invalid" for item in attempts
            )
            and len(attempts) > 1
        )
        certified_after_nonsemantic_retry = bool(
            result.get("verified")
            and repair.get("repair_attempted")
            and not repair.get("first_attempt_verified")
            and not semantic_repair_attempted
            and not execution_error
        )
        for field, derived in (
            ("post_judge_repair_attempted", post_judge_repair_attempted),
            ("semantic_repair_attempted", semantic_repair_attempted),
            (
                "formalizer_invalid_retry_attempted",
                formalizer_invalid_retry_attempted,
            ),
            (
                "certified_after_nonsemantic_retry",
                certified_after_nonsemantic_retry,
            ),
        ):
            if field in repair and bool(repair[field]) != derived:
                raise RuntimeError(f"{problem_id}: {field} disagrees with trace")

        answer = result.get("answer")
        extracted = extract_choice(answer, options)
        solver_solutions = result.get("solver_solutions") or []
        solver_answer = solver_solutions[0] if solver_solutions else None
        solver_extracted = extract_choice(solver_answer, options)
        first_attempt_answer = repair.get("first_attempt_answer")
        first_attempt_extracted = extract_choice(first_attempt_answer, options)
        final_collision = _bare_letter_option_text_collision(answer, options)
        solver_collision = _bare_letter_option_text_collision(solver_answer, options)
        first_collision = _bare_letter_option_text_collision(
            first_attempt_answer, options,
        )
        first_attempt_missing = (
            first_attempt_answer is None or not str(first_attempt_answer).strip()
        )

        row = {
            "id": problem_id,
            "category": str(result.get("category") or "unknown"),
            "verified": bool(result.get("verified") and not execution_error),
            "expected": expected,
            "extracted": extracted,
            "correct": extracted == expected,
            "solver_extracted": solver_extracted,
            "solver_correct": solver_extracted == expected,
            "solver_missing": solver_answer is None or not str(solver_answer).strip(),
            "solver_needs_extraction_review": bool(
                solver_answer is not None
                and str(solver_answer).strip()
                and (solver_extracted is None or solver_collision)
            ),
            "solver_bare_letter_option_text_collision": solver_collision,
            "first_attempt_extracted": first_attempt_extracted,
            "first_attempt_correct": first_attempt_extracted == expected,
            "first_attempt_missing": first_attempt_missing,
            "first_attempt_verified": bool(
                repair.get("first_attempt_verified") and not execution_error
            ),
            "first_attempt_needs_extraction_review": bool(
                not first_attempt_missing
                and (first_attempt_extracted is None or first_collision)
            ),
            "first_attempt_bare_letter_option_text_collision": first_collision,
            "repair_attempted": bool(repair.get("repair_attempted")),
            "semantic_repair_attempted": semantic_repair_attempted,
            "post_judge_repair_attempted": post_judge_repair_attempted,
            "formalizer_invalid_retry_attempted": (
                formalizer_invalid_retry_attempted
            ),
            "certified_after_nonsemantic_retry": (
                certified_after_nonsemantic_retry
            ),
            "certified_by_repair": bool(
                repair.get("certified_by_repair") and not execution_error
            ),
            "solver_retries": int(repair.get("solver_retries") or 0),
            "judge_repair_rounds": int(repair.get("judge_repair_rounds") or 0),
            "formalizer_reject_count": int(
                repair.get("formalizer_reject_count") or 0
            ),
            "formalizer_invalid_count": int(
                repair.get("formalizer_invalid_count") or 0
            ),
            "answer_change_observable": bool(repair.get("answer_change_observable")),
            "answer_changed_during_repair": repair.get(
                "answer_changed_during_repair"
            ),
            "normalized_answer_changed_during_repair": repair.get(
                "normalized_answer_changed_during_repair"
            ),
            "solver_solution_text_changed_during_repair": repair.get(
                "solver_solution_text_changed_during_repair"
            ),
            "answer_before_repair_sha256": repair.get(
                "answer_before_repair_sha256"
            ),
            "answer_before_repair_normalized_sha256": repair.get(
                "answer_before_repair_normalized_sha256"
            ),
            "final_answer_sha256": repair.get("final_answer_sha256"),
            "final_answer_normalized_sha256": repair.get(
                "final_answer_normalized_sha256"
            ),
            "certification_pipeline_attempt": certification_pipeline_attempt,
            "certification_verify_attempt": certification_verify_attempt,
            "repair_metrics_complete": repair_metrics_complete,
            "needs_extraction_review": bool(
                answer is not None
                and str(answer).strip()
                and (extracted is None or final_collision)
            ),
            "bare_letter_option_text_collision": final_collision,
            "execution_error": execution_error,
            "telemetry": telemetry,
        }
        rows.append(row)
        domains[row["category"]].append(row)

    r0_ids = {row["id"] for row in rows if row["first_attempt_verified"]}
    r1_ids = {row["id"] for row in rows if row["verified"]}
    repair_ids = {row["id"] for row in rows if row["certified_by_repair"]}
    if not r0_ids <= r1_ids:
        raise RuntimeError("invalid operating points: R0 certification is not in R1")
    if repair_ids != r1_ids - r0_ids:
        raise RuntimeError(
            "certified_by_repair does not match the R1 minus R0 transition"
        )
    for row in rows:
        expected_certified_by_repair = bool(
            row["verified"]
            and row["repair_attempted"]
            and not row["first_attempt_verified"]
        )
        if row["certified_by_repair"] != expected_certified_by_repair:
            raise RuntimeError(
                f"{row['id']}: certified_by_repair is inconsistent"
            )

    return {
        "schema_version": 1,
        "extraction_policy": (
            "A bare A-D token is interpreted as an option label. Exact option-text "
            "matching applies only to longer responses. Lowercase A-D after an "
            "answer delimiter and lowercase b-d after 'final answer is' are labels. "
            "Ambiguous bare tokens that equal another option's text are review flags."
        ),
        "run_id": meta.get("run_id"),
        "source_run": str(run_path.resolve()),
        "source_run_sha256": source_sha,
        "metrics": _metrics(rows),
        "per_domain": {
            domain: _metrics(domain_rows)
            for domain, domain_rows in sorted(domains.items())
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    report = summarize(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
