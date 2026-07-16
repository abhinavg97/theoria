import json
import sys
import types

from loaders import HLE_DATASET_REVISION, build_question, load_hle


def test_build_question_default_id():
    problems = build_question("What is 2 + 2?")
    assert problems == [{
        "id": "question",
        "question": "What is 2 + 2?",
        "answer": "",
        "dataset": "custom",
    }]


def test_build_question_custom_id():
    problems = build_question("Prove there are infinitely many primes", problem_id="primes")
    assert len(problems) == 1
    assert problems[0]["id"] == "primes"
    assert problems[0]["answer"] == ""
    assert problems[0]["dataset"] == "custom"


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
    assert problems[0]["dataset_fingerprint"] == "resolved-fingerprint"
    assert problems[0]["dataset_config"] == "default"
