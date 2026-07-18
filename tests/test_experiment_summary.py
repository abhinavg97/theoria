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


def test_strict_correct_credits_extraction_only():
    assert summarize_run._strict_correct({"final": {"key_match": True}})
    assert summarize_run._strict_correct({
        "final": {"key_match": False, "dispute_category": "extraction"},
    })
    assert not summarize_run._strict_correct({
        "final": {"key_match": False, "dispute_category": "interpretation"},
    })


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
