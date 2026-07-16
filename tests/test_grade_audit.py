import asyncio
import json
from pathlib import Path

import grade
from llm import artifact_dir, call_log


def test_grade_run_captures_calls_and_reproducibility_artifacts(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 2+2?",
        "expected": "4",
        "answer": "4",
        "verified": True,
        "attempts": [],
    }]))

    async def fake_grade_one(*_args, **_kwargs):
        assert Path(artifact_dir.get()).is_dir()
        call_log.get().append({
            "role": "audit_grader",
            "backend": "claude",
            "model": "test-grader",
            "input_tokens": 10,
            "output_tokens": 4,
            "duration_ms": 5,
            "retry_count": 0,
            "failed": False,
            "usage_observed": True,
            "usage_complete": True,
        })
        return {
            "final": {
                "key_match": True,
                "reasoning": "equivalent",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade_one)
    monkeypatch.setattr(
        grade,
        "fetch_rationales",
        lambda *_args, **_kwargs: ({}, {
            "revision": grade.HLE_DATASET_REVISION,
            "fingerprint": "fake-fingerprint",
        }),
    )

    summary = asyncio.run(grade.grade_run(str(source), watch=False))

    assert summary["run_file_sha256"]
    assert summary["metrics"]["total_tokens"] == 14
    assert summary["metrics"]["usage_complete_fraction"] == 1.0
    assert len(summary["results"][0]["calls"]) == 1
    assert summary["evaluation_policy"] == {
        "automatic_grader": True,
        "manual_adjudication_included": False,
        "automatic_grade_is_authoritative": False,
    }

    root = Path(summary["artifact_root"])
    meta = json.loads((root / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["source_run_sha256"] == summary["run_file_sha256"]
    assert meta["grader_prompt_sha256"] == summary["grader_prompt_sha"]
    assert meta["rationale_dataset"]["fingerprint"] == "fake-fingerprint"
    assert (root / "grader_prompt.txt").exists()
    assert (root / "grader_schema.json").exists()
    assert (root / "rationales.json").exists()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    assert any(entry["path"] == "meta.json" for entry in manifest["entries"])
