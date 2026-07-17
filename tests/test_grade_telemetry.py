from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import grade
from llm import artifact_dir, call_log, telemetry_context
from telemetry import enrich_call


class GraderTelemetryTests(unittest.TestCase):
    def test_grade_run_captures_calls_and_artifacts(self):
        async def fake_grade_one(*_args, **_kwargs):
            trace = telemetry_context.get()
            self.assertTrue(Path(artifact_dir.get()).is_dir())
            call_log.get().append(enrich_call({
                "call_id": "grader-call",
                **trace,
                "role": "audit_grader",
                "backend": "claude",
                "model": "test-model",
                "input_tokens": 10,
                "output_tokens": 4,
                "provider_usage": {"input_tokens": 10, "output_tokens": 4},
                "total_cost_usd": 0.01,
                "duration_ms": 5,
            }))
            return {
                "final": {
                    "key_match": True,
                    "reasoning": "equivalent",
                    "dispute_category": "none",
                },
                "attempts": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                run_path = Path("source.json")
                run_path.write_text(json.dumps([{
                    "id": "p1",
                    "problem": "question",
                    "expected": "answer",
                    "answer": "answer",
                    "attempts": [],
                }]))
                with (
                    patch("grade.fetch_rationales", return_value={}),
                    patch("grade.grade_one", side_effect=fake_grade_one),
                ):
                    summary = asyncio.run(grade.grade_run(
                        str(run_path), watch=False,
                    ))
                self.assertEqual(summary["metrics"]["total_tokens"], 14)
                self.assertEqual(len(summary["results"][0]["calls"]), 1)
                artifact_root = Path(summary["artifact_root"])
                self.assertTrue((artifact_root / "telemetry.json").exists())
                self.assertTrue((artifact_root / "meta.json").exists())
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
