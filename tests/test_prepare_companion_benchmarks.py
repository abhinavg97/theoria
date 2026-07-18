import json
import sys
from types import SimpleNamespace

import pytest

from experiments import prepare_companion_benchmarks as prepare


class _Dataset(list):
    _fingerprint = "test-fingerprint"


def _hf_row():
    return {
        "Question": "Which option is correct?",
        "Correct Answer": "the correct answer",
        "Incorrect Answer 1": "first distractor",
        "Incorrect Answer 2": "second distractor",
        "Incorrect Answer 3": "third distractor",
    }


def _problem(answer="B"):
    return {
        "id": "gpqa_test",
        "source_row_index_0based": 0,
        "question": (
            "Which option is correct?\n\n"
            "A) first distractor\n"
            "B) the correct\n   answer\n"
            "C) second distractor\n"
            "D) third distractor"
        ),
        "correct_text": "the correct answer",
        "answer": answer,
    }


def _install_fake_datasets(monkeypatch):
    dataset = _Dataset([_hf_row(), *({} for _ in range(197))])
    module = SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset)
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_verify_hf_checks_shuffled_answer_key(monkeypatch):
    _install_fake_datasets(monkeypatch)

    assert prepare._verify_against_huggingface([_problem()]) == "test-fingerprint"


def test_verify_hf_rejects_shuffled_answer_key_mismatch(monkeypatch):
    _install_fake_datasets(monkeypatch)

    with pytest.raises(RuntimeError, match="shuffled answer key mismatch"):
        prepare._verify_against_huggingface([_problem(answer="A")])


def test_legacy_step7_rows_are_labeled_as_outcome_filtered(monkeypatch, tmp_path):
    step5 = {
        "metadata": {"sample_indices": list(range(25))},
        "problems": [
            {"id": f"gpqa_{index:03d}", "question": "Q", "answer": "A"}
            for index in range(25)
        ],
    }
    step7 = {
        "metadata": {"sample_indices": list(range(25, 100))},
        "problems": [
            {"id": f"gpqa_{index:03d}", "question": "Q", "answer": "A"}
            for index in range(25, 100)
        ],
    }
    step7_clean = [{"id": f"gpqa_{index:03d}"} for index in range(25, 65)]

    def fake_git_json(_repo, path):
        if path.endswith("step5_gpqa_ood/gpqa_problems.json"):
            return step5
        if path.endswith("step7_gpqa_expanded/gpqa_problems.json"):
            return step7
        if path.endswith("step7_gpqa_expanded/gpqa_api_run_clean.json"):
            return step7_clean
        raise AssertionError(path)

    monkeypatch.setattr(prepare, "_git_json", fake_git_json)
    prepare.materialize(tmp_path, tmp_path / "out")

    legacy = json.loads(
        (tmp_path / "out/gpqa-legacy-paper-65.json").read_text()
    )["problems"]
    legacy_step7 = [row for row in legacy if int(row["id"][5:]) >= 25]
    assert len(legacy_step7) == 40
    assert {row["paper65_source"] for row in legacy_step7} == {
        "step7_outcome_filtered_historical_subset_retained_after_success"
    }

    fixed = json.loads((tmp_path / "out/gpqa-fixed-100.json").read_text())[
        "problems"
    ]
    fixed_step7 = [row for row in fixed if int(row["id"][5:]) >= 25]
    assert {row["paper65_source"] for row in fixed_step7} == {
        "step7_fixed_before_run"
    }
