import asyncio
import json
from pathlib import Path

import pytest

import grade
from llm import artifact_dir, call_log


def _stub_audit_environment(monkeypatch):
    monkeypatch.setattr(grade.harness, "_probe_ollama_models", lambda _config: [])
    monkeypatch.setattr(
        grade.harness, "_probe_openai_compatible_models", lambda _config: [],
    )
    monkeypatch.setattr(grade.harness, "_safe_git_state", lambda *_a, **_k: {})
    monkeypatch.setattr(grade.harness, "_safe_version", lambda *_a, **_k: None)
    monkeypatch.setattr(
        grade.harness,
        "_capture_host_environment",
        lambda *_a, **_k: {"available": True, "sha256": "environment"},
    )


def test_grade_run_captures_calls_and_reproducibility_artifacts(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
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
        "missing_target_as_incorrect": False,
    }

    root = Path(summary["artifact_root"])
    meta = json.loads((root / "meta.json").read_text())
    assert meta["audit_schema_version"] == grade.harness.AUDIT_SCHEMA_VERSION
    assert meta["status"] == "completed"
    assert meta["source_run_sha256"] == summary["run_file_sha256"]
    assert meta["grader_prompt_sha256"] == summary["grader_prompt_sha"]
    assert meta["rationale_dataset"]["fingerprint"] == "fake-fingerprint"
    assert meta["grading_target"] == "final"
    assert (root / "grader_prompt.txt").exists()
    assert (root / "grader_schema.json").exists()
    assert (root / "rationales.json").exists()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    assert any(entry["path"] == "meta.json" for entry in manifest["entries"])


def test_grade_run_can_grade_initial_solver_response(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 2+2?",
        "expected": "4",
        "answer": "4",
        "verified": True,
        "solver_solutions": ["I initially answered 5", "I repaired it to 4"],
        "attempts": [{"phase": "verify"}],
    }]))

    solver_prompt = grade.SOLVER_PROMPT_PATH.read_text()

    async def fake_grade_one(result, _config, prompt_text, *_args, **_kwargs):
        assert result["answer"] == "I initially answered 5"
        assert result["verified"] is None
        assert result["attempts"] == []
        assert prompt_text == solver_prompt
        return {
            "final": {
                "key_match": False,
                "reasoning": "wrong",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade_one)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(
        str(source), target="solver_initial", watch=False,
    ))

    assert summary["grading_target"] == "solver_initial"
    assert summary["results"][0]["target"] == "solver_initial"
    assert summary["results"][0]["answer"] == "I initially answered 5"
    meta = json.loads(
        (Path(summary["artifact_root"]) / "meta.json").read_text()
    )
    assert meta["grading_target"] == "solver_initial"
    assert meta["grader_prompt_sha256"] == grade.hashlib.sha256(
        solver_prompt.encode()
    ).hexdigest()
    assert (Path(summary["artifact_root"]) / "grader_prompt.txt").read_text() == (
        solver_prompt
    )
    assert Path(summary["artifact_root"]).name.startswith("grade_solver_initial_")


def test_grade_run_can_blindly_grade_first_formalized_candidate(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    first = {"phase": "verify", "all_ok": False, "proof": {"steps": []}}
    later = {"phase": "verify", "all_ok": True, "proof": {"steps": []}}
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 2+2?",
        "expected": "4",
        "answer": "4",
        "verified": True,
        "attempts": [first, later],
        "repair_metrics": {
            "first_attempt_answer": "5",
            "first_attempt_verified": False,
        },
    }]))

    async def fake_grade_one(result, _config, _prompt_text, *_args, **_kwargs):
        assert result["answer"] == "5"
        assert result["attempts"] == []
        return {
            "final": {
                "key_match": False,
                "reasoning": "wrong",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade_one)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(
        str(source), target="first_attempt", watch=False,
    ))

    assert summary["grading_target"] == "first_attempt"
    assert summary["results"][0]["answer"] == "5"
    assert Path(summary["artifact_root"]).name.startswith("grade_first_attempt_")


def test_solver_baseline_can_count_missing_response_as_incorrect(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 2+2?",
        "expected": "4",
        "answer": None,
        "verified": False,
        "solver_solutions": [],
        "error": "provider timed out",
    }]))
    monkeypatch.setattr(
        grade,
        "grade_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing response must not call a grader")
        ),
    )
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(
        str(source),
        target="solver_initial",
        missing_as_incorrect=True,
        watch=False,
    ))

    assert summary["key_match"] == 0
    assert summary["evaluation_policy"]["missing_target_as_incorrect"] is True
    row = summary["results"][0]
    assert row["grade_source"] == "deterministic_missing_target"
    assert row["final"]["key_match"] is False
    assert row["calls"] == []
    root = Path(summary["artifact_root"])
    assert list(root.glob("problem_*/deterministic_verdict.json"))


def test_solver_baseline_uses_only_initial_response_for_missing_policy(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 2+2?",
        "expected": "4",
        "answer": "4",
        "verified": True,
        "solver_solutions": ["", "4"],
    }]))
    monkeypatch.setattr(
        grade,
        "grade_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing initial response must not call a grader")
        ),
    )
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(
        str(source), target="solver_initial", missing_as_incorrect=True,
        watch=False,
    ))

    assert summary["results"][0]["grade_source"] == "deterministic_missing_target"


def test_final_grade_can_count_missing_response_as_incorrect(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 2+2?",
        "expected": "4",
        "answer": None,
        "verified": False,
    }]))
    monkeypatch.setattr(
        grade,
        "grade_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing final response must not call a grader")
        ),
    )
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(
        str(source), target="final", missing_as_incorrect=True, watch=False,
    ))

    assert summary["results"][0]["grade_source"] == "deterministic_missing_target"


def test_numeric_zero_is_not_treated_as_a_missing_final_response(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "p1",
        "problem": "What is 1-1?",
        "expected": "0",
        "answer": 0,
        "verified": True,
    }]))
    called = False

    async def fake_grade_one(*_args, **_kwargs):
        nonlocal called
        called = True
        return {
            "final": {
                "key_match": True,
                "reasoning": "exact",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade_one)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    asyncio.run(grade.grade_run(
        str(source), target="final", missing_as_incorrect=True, watch=False,
    ))

    assert called is True


def test_malformed_verdict_fails_only_its_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([
        {"id": "bad", "problem": "Q1", "expected": "A", "answer": "A"},
        {"id": "good", "problem": "Q2", "expected": "B", "answer": "B"},
    ]))
    calls = 0

    async def fake_grade_one(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"attempts": []}
        return {
            "final": {
                "key_match": True,
                "reasoning": "match",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade_one)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_args, **_kwargs: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(str(source), watch=False))

    assert calls == 2
    assert summary["key_match"] == 1
    assert summary["metrics"]["successful_grades"] == 1
    assert "missing a valid final grade" in summary["results"][0]["error"]
    assert summary["results"][1]["final"]["key_match"] is True
    meta = json.loads(
        (Path(summary["artifact_root"]) / "meta.json").read_text()
    )
    assert meta["status"] == "completed_with_errors"


def test_hle_grade_fails_closed_without_canonical_rationale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([{
        "id": "hle-id",
        "dataset_name": grade.HLE_DATASET_NAME,
        "dataset_revision": grade.HLE_DATASET_REVISION,
        "problem": "Question",
        "expected": "Answer",
        "answer": "Answer",
    }]))
    monkeypatch.setattr(
        grade,
        "fetch_rationales",
        lambda *_a, **_k: ({}, {"available": False}),
    )

    with pytest.raises(RuntimeError, match="canonical rationale coverage"):
        asyncio.run(grade.grade_run(str(source), watch=False))

    roots = list(Path("runs/artifacts").glob("grade_source_*"))
    assert len(roots) == 1
    meta = json.loads((roots[0] / "meta.json").read_text())
    assert meta["status"] == "failed"


def test_grade_run_uses_one_immutable_source_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    original = [{
        "id": "p1", "problem": "Q", "expected": "A", "answer": "A",
        "attempts": [],
    }]
    source.write_text(json.dumps(original))
    original_sha = grade.hashlib.sha256(source.read_bytes()).hexdigest()

    async def mutate_source(*_args, **_kwargs):
        source.write_text(json.dumps([{"id": "different"}]))
        return {
            "final": {
                "key_match": True,
                "reasoning": "match",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", mutate_source)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_a, **_k: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(str(source), watch=False))

    assert summary["run_file_sha256"] == original_sha
    assert json.loads(Path(summary["source_snapshot"]).read_text()) == original
    manifest = json.loads(
        (Path(summary["artifact_root"]) / "artifact_manifest.json").read_text()
    )
    assert not any(
        entry.get("external") and Path(entry["path"]).resolve() == source.resolve()
        for entry in manifest["entries"]
    )


def test_grade_run_isolates_unsafe_and_colliding_problem_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps([
        {"id": "..", "problem": "Q1", "expected": "A", "answer": "A"},
        {"id": "a/b", "problem": "Q2", "expected": "A", "answer": "A"},
        {"id": "a?b", "problem": "Q3", "expected": "A", "answer": "A"},
    ]))

    async def fake_grade(result, *_args, **_kwargs):
        if result["id"] == "..":
            return {"attempts": []}
        return {
            "final": {
                "key_match": True,
                "reasoning": "match",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_a, **_k: ({}, {}),
    )

    summary = asyncio.run(grade.grade_run(str(source), watch=False))

    subdirs = [item["artifact_subdir"] for item in summary["results"]]
    assert len(set(subdirs)) == 3
    assert all("/" not in name and name not in {".", ".."} for name in subdirs)
    assert summary["results"][0]["error_type"] == "RuntimeError"
    meta = json.loads((Path(summary["artifact_root"]) / "meta.json").read_text())
    assert meta["status"] == "completed_with_errors"


def test_grade_run_never_overwrites_previous_grade(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps({
        "id": "p1", "problem": "Q", "expected": "A", "answer": "A",
    }))

    async def fake_grade(*_args, **_kwargs):
        return {
            "final": {
                "key_match": True,
                "reasoning": "match",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_a, **_k: ({}, {}),
    )

    first = asyncio.run(grade.grade_run(str(source), watch=False))
    second = asyncio.run(grade.grade_run(str(source), watch=False))

    first_output = Path(json.loads(
        (Path(first["artifact_root"]) / "meta.json").read_text()
    )["output_path"])
    second_output = Path(json.loads(
        (Path(second["artifact_root"]) / "meta.json").read_text()
    )["output_path"])
    assert first_output != second_output
    assert first_output.exists() and second_output.exists()


def test_grade_rejects_source_metadata_for_a_different_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "runs" / "source.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"id": "p1"}))
    meta_path = tmp_path / "runs" / "artifacts" / "source" / "meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps({
        "run_id": "source",
        "cwd": str(tmp_path),
        "save_path": "runs/different.json",
        "status": "completed",
    }))

    with pytest.raises(RuntimeError, match="different run file"):
        asyncio.run(grade.grade_run(str(source), watch=False))


def test_grade_manifest_failure_marks_run_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_audit_environment(monkeypatch)
    source = Path("source.json")
    source.write_text(json.dumps({
        "id": "p1", "problem": "Q", "expected": "A", "answer": "A",
    }))

    async def fake_grade(*_args, **_kwargs):
        return {
            "final": {
                "key_match": True,
                "reasoning": "match",
                "dispute_category": "none",
            },
            "attempts": [],
        }

    monkeypatch.setattr(grade, "grade_one", fake_grade)
    monkeypatch.setattr(
        grade, "fetch_rationales", lambda *_a, **_k: ({}, {}),
    )
    monkeypatch.setattr(
        grade.harness,
        "write_artifact_manifest",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("seal failed")),
    )

    with pytest.raises(OSError, match="seal failed"):
        asyncio.run(grade.grade_run(str(source), watch=False))

    roots = list((tmp_path / "runs" / "artifacts").glob("grade_source_*"))
    assert len(roots) == 1
    meta = json.loads((roots[0] / "meta.json").read_text())
    assert meta["status"] == "failed"
    assert meta["run_error"]["message"] == "seal failed"
