import hashlib
import json
import sys
import types

import pytest

from loaders import (
    HLE_DATASET_NAME,
    HLE_DATASET_REVISION,
    build_question,
    load_hle,
    load_json_benchmark,
)


def test_hle_cli_defaults_to_pinned_dataset_revision():
    from cli import build_parser

    args = build_parser().parse_args(["hle", "1"])
    assert args.dataset_revision == HLE_DATASET_REVISION


def test_build_question_default_id():
    problems = build_question("What is 2 + 2?")
    assert problems == [{
        "id": "question",
        "question": "What is 2 + 2?",
        "answer": "",
        "dataset": "custom",
        "dataset_name": "custom",
    }]


def test_build_question_custom_id():
    problems = build_question("Prove there are infinitely many primes", problem_id="primes")
    assert len(problems) == 1
    assert problems[0]["id"] == "primes"
    assert problems[0]["answer"] == ""
    assert problems[0]["dataset"] == "custom"
    assert problems[0]["dataset_name"] == "custom"


def test_hle_loader_pins_revision_and_records_fingerprint(monkeypatch):
    seen = {}

    class FakeInfo:
        config_name = "default"

    class FakeDataset(list):
        _fingerprint = "resolved-fingerprint"
        info = FakeInfo()

    def fake_load_dataset(name, *, split, revision):
        seen.update(name=name, split=split, revision=revision)
        return FakeDataset([{
            "id": "p1",
            "question": "Q",
            "answer": "A",
            "category": "Math",
            "Verified_Classes": "Gold subset",
            "json": json.dumps({"image": None, "answer_type": "exact"}),
        }])

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    problems = load_hle(max_questions=1)

    assert seen["revision"] == HLE_DATASET_REVISION
    assert problems[0]["dataset_revision"] == HLE_DATASET_REVISION
    assert problems[0]["dataset_name"] == HLE_DATASET_NAME
    assert problems[0]["dataset_fingerprint"] == "resolved-fingerprint"
    assert problems[0]["dataset_config"] == "default"


def test_hle_loader_preserves_explicit_id_order(monkeypatch):
    rows = [{
        "id": pid,
        "question": f"Q-{pid}",
        "answer": f"A-{pid}",
        "category": "Math",
        "Verified_Classes": "Gold subset",
        "json": json.dumps({"image": None, "answer_type": "exact"}),
    } for pid in ("first", "second", "third")]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *_args, **_kwargs: rows),
    )

    problems = load_hle(ids=["third", "first"])

    assert [problem["id"] for problem in problems] == ["third", "first"]


def test_hle_loader_rejects_missing_explicit_id(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *_args, **_kwargs: []),
    )

    with pytest.raises(ValueError, match="absent"):
        load_hle(ids=["missing"])


def test_json_benchmark_records_source_hash_and_requested_order(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({
        "metadata": {
            "name": "GPQA fixed",
            "dataset_name": "Idavidrein/gpqa",
            "dataset_revision": "revision",
        },
        "problems": [
            {"id": "one", "question": "Q1", "answer": "A"},
            {"id": "two", "question": "Q2", "answer": "B"},
        ],
    }))

    problems = load_json_benchmark(path, ids=["two", "one"])

    assert [problem["id"] for problem in problems] == ["two", "one"]
    assert problems[0]["dataset_name"] == "Idavidrein/gpqa"
    assert problems[0]["dataset_revision"] == "revision"
    assert problems[0]["benchmark_file_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_json_benchmark_preserves_numeric_zero_answer(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([
        {"id": "zero", "question": "1 - 1?", "answer": 0},
    ]))

    assert load_json_benchmark(path)[0]["answer"] == "0"


def test_json_benchmark_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([
        {"id": "same", "question": "Q1", "answer": "A"},
        {"id": "same", "question": "Q2", "answer": "B"},
    ]))

    try:
        load_json_benchmark(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate benchmark ids were accepted")


def test_json_benchmark_rejects_duplicate_ids_with_requested_order(tmp_path):
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([
        {"id": "same", "question": "Q1", "answer": "A"},
        {"id": "same", "question": "Q2", "answer": "B"},
    ]))

    with pytest.raises(ValueError, match="duplicate"):
        load_json_benchmark(path, ids=["same"])
