from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import patch

import loaders


class FakeDataset(list):
    _fingerprint = "dataset-fingerprint"


class LoaderProvenanceTests(unittest.TestCase):
    def test_hle_records_revision_and_resolved_fingerprint(self):
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

        self.assertEqual(seen["revision"], "commit-sha")
        self.assertEqual(problems[0]["dataset_revision"], "commit-sha")
        self.assertEqual(
            problems[0]["dataset_fingerprint"], "dataset-fingerprint"
        )


if __name__ == "__main__":
    unittest.main()
