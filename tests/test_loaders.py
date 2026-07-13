from __future__ import annotations

import json
import sys
import types
from unittest.mock import patch

import loaders
from loaders import build_question

import json
import sys
import types
import unittest
from unittest.mock import patch

class FakeDataset(list):
    _fingerprint = "dataset-fingerprint"


def test_build_question_default_id():
    problems = build_question("What is 2 + 2?")
    assert problems == [{
        "id": "question",
        "question": "What is 2 + 2?",
        "answer": "",
    }]


def test_build_question_custom_id():
    problems = build_question(
        "Prove there are infinitely many primes", problem_id="primes",
    )
    assert len(problems) == 1
    assert problems[0]["id"] == "primes"
    assert problems[0]["answer"] == ""


def test_hle_records_revision_and_resolved_fingerprint():
    seen = {}

    def load_dataset(name, *, split, revision):
        seen.update(name=name, split=split, revision=revision)
        return FakeDataset([{
            "id": "p1",
            "json": json.dumps({"image": None, "answer_type": "exact"}),
            "Verified_Classes": "Gold subset",
            "category": "Math",
            "question": "Q",
            "answer": "A",
        }])

    module = types.SimpleNamespace(load_dataset=load_dataset)
    with patch.dict(sys.modules, {"datasets": module}):
        problems = loaders.load_hle(revision="commit-sha")

    assert seen["revision"] == "commit-sha"
    assert problems[0]["dataset_revision"] == "commit-sha"
    assert problems[0]["dataset_fingerprint"] == "dataset-fingerprint"
