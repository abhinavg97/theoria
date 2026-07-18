#!/usr/bin/env python3
"""Build a blinded, hash-bound adjudication queue for a sealed HLE run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from loaders import HLE_DATASET_NAME

if __package__:
    from experiments import summarize_run
else:  # Support `python experiments/build_adjudication_queue.py ...`.
    import summarize_run


SCHEMA_VERSION = 1
DEFAULT_SEED = "20260718"
DEFAULT_AUDIT_FRACTION = 0.10
CASES_FILENAME = "cases.json"
DECISION_TEMPLATE_FILENAME = "decisions.template.json"
MANIFEST_FILENAME = "manifest.json"
SUPPORTED_TARGETS = {"final", "first_attempt"}

_CASES_TOP_LEVEL_KEYS = {"schema_version", "queue_id", "cases"}
_BLINDED_CASE_KEYS = {
    "case_id",
    "category",
    "problem",
    "expected_answer",
    "candidate_answer",
    "canonical_rationale",
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_new(path: Path, value: object) -> tuple[str, int]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite adjudication artifact: {path}")
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return _sha256_bytes(payload), len(payload)


def _is_hle_row(row: dict) -> bool:
    return row.get("dataset_name") == HLE_DATASET_NAME


def _grade_input_records(paths: list[Path], *, target: str) -> list[dict]:
    records = []
    seen_graders = set()
    for supplied_path in paths:
        path = supplied_path.resolve(strict=True)
        summary = json.loads(path.read_text())
        actual_target = summary.get("grading_target") or "final"
        if actual_target != target:
            raise RuntimeError(
                f"grade target mismatch for {path}: {actual_target!r} != {target!r}"
            )
        grader = str(summary.get("grader_model") or "").strip()
        if not grader:
            raise RuntimeError(f"grade file has no grader identity: {path}")
        if grader in seen_graders:
            raise RuntimeError(f"duplicate grader identity: {grader}")
        seen_graders.add(grader)
        records.append({
            "grader": grader,
            "path": str(path),
            "sha256": _sha256_file(path),
        })
    return sorted(records, key=lambda row: row["grader"])


def _grade_signature(records: list[dict]) -> list[dict]:
    return [
        {"grader": row["grader"], "sha256": row["sha256"]}
        for row in sorted(records, key=lambda row: row["grader"])
    ]


def _grade_artifact_root(grade_path: Path, summary: dict) -> Path:
    root = Path(summary.get("artifact_root") or "")
    if root.is_absolute():
        return root.resolve(strict=True)
    candidate = (grade_path.parent.parent.parent / root).resolve()
    if candidate.exists():
        return candidate
    return root.resolve(strict=True)


def _load_frozen_rationales(
    grade_paths: list[Path], expected_ids: set[str],
) -> tuple[dict[str, dict], str]:
    """Load and cross-check the rationale snapshots sealed by every grader."""
    snapshots: list[tuple[str, dict[str, dict]]] = []
    for supplied_path in grade_paths:
        path = supplied_path.resolve(strict=True)
        summary = json.loads(path.read_text())
        root = _grade_artifact_root(path, summary)
        rationale_path = root / "rationales.json"
        manifest = summarize_run.harness.verify_artifact_manifest(str(root))
        matching = [
            entry for entry in (manifest.get("entries") or [])
            if not entry.get("external")
            and (root / str(entry.get("path", ""))).resolve() == rationale_path.resolve()
        ]
        if len(matching) != 1:
            raise RuntimeError(f"grader rationale snapshot is not uniquely sealed: {path}")
        if matching[0].get("sha256") != _sha256_file(rationale_path):
            raise RuntimeError(f"grader rationale snapshot hash mismatch: {path}")
        payload = json.loads(rationale_path.read_text())
        if not isinstance(payload, dict) or set(map(str, payload)) != expected_ids:
            raise RuntimeError(f"grader rationale ids do not match the source run: {path}")
        normalized = {str(problem_id): value for problem_id, value in payload.items()}
        if any(not isinstance(value, dict) for value in normalized.values()):
            raise RuntimeError(f"grader rationale snapshot has invalid rows: {path}")
        snapshots.append((_identity_sha256(normalized), normalized))
    if not snapshots:
        raise RuntimeError("at least one frozen rationale snapshot is required")
    hashes = {digest for digest, _payload in snapshots}
    if len(hashes) != 1:
        raise RuntimeError("grader rationale snapshots disagree")
    return snapshots[0][1], snapshots[0][0]


def _target_certified(row: dict, target: str) -> bool:
    if row.get("error"):
        return False
    if target == "final":
        return bool(row.get("verified"))
    if target == "first_attempt":
        repair = row.get("repair_metrics")
        return bool(
            isinstance(repair, dict) and repair.get("first_attempt_verified")
        )
    raise ValueError(f"unsupported adjudication target: {target}")


def _candidate_answer(row: dict, target: str) -> object | None:
    if target == "final":
        return row.get("answer")
    if target == "first_attempt":
        repair = row.get("repair_metrics")
        return repair.get("first_attempt_answer") if isinstance(repair, dict) else None
    raise ValueError(f"unsupported adjudication target: {target}")


def _certified_rows(results: list[dict], target: str) -> list[dict]:
    return [row for row in results if _target_certified(row, target)]


def select_adjudication_cases(
    results: list[dict],
    graders: dict[str, dict[str, dict]],
    *,
    target: str,
    seed: str,
    audit_fraction: float,
) -> list[dict]:
    """Return private queue records, including automatic labels and reasons."""
    if len(graders) < 2:
        raise RuntimeError("publication adjudication requires at least two graders")
    if not 0 <= audit_fraction <= 1:
        raise ValueError("audit_fraction must be between 0 and 1")

    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported adjudication target: {target}")
    certified = _certified_rows(results, target)
    automatic: dict[str, dict] = {}
    agreement_pool = []
    for row in certified:
        problem_id = str(row.get("id"))
        labels = {
            grader: summarize_run._key_match(rows[problem_id])
            for grader, rows in graders.items()
        }
        disagreement = len(set(labels.values())) > 1
        apparent_error = not all(labels.values())
        record = {
            "problem_id": problem_id,
            "category": str(row.get("category") or "unknown"),
            "automatic_strict_correct": dict(sorted(labels.items())),
            "automatic_consensus_strict_correct": all(labels.values()),
            "grader_disagreement": disagreement,
            "apparent_certified_error": apparent_error,
            "agreement_audit_hash": hashlib.sha256(
                f"{seed}:{problem_id}".encode("utf-8")
            ).hexdigest(),
            "selection_reasons": [],
        }
        if disagreement:
            record["selection_reasons"].append("grader_disagreement")
        if apparent_error:
            record["selection_reasons"].append("apparent_certified_error")
        if not disagreement and not apparent_error:
            agreement_pool.append(record)
        automatic[problem_id] = record

    sample_size = (
        math.ceil(audit_fraction * len(agreement_pool)) if agreement_pool else 0
    )
    sampled = sorted(
        agreement_pool,
        key=lambda row: (row["agreement_audit_hash"], row["problem_id"]),
    )[:sample_size]
    for record in sampled:
        record["selection_reasons"].append("agreement_audit_sample")

    return sorted(
        (
            record for record in automatic.values()
            if record["selection_reasons"]
        ),
        key=lambda row: row["problem_id"],
    )


def validate_blinded_cases(payload: object, *, queue_id: str | None = None) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != _CASES_TOP_LEVEL_KEYS:
        raise RuntimeError("blinded cases file has an unexpected top-level shape")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported blinded cases schema version")
    if queue_id is not None and payload.get("queue_id") != queue_id:
        raise RuntimeError("blinded cases queue_id does not match the manifest")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("blinded cases must be a list")
    case_ids = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != _BLINDED_CASE_KEYS:
            raise RuntimeError("blinded case contains provenance or automatic-label fields")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("blinded case has no opaque case_id")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("blinded cases contain duplicate case_ids")
    return cases


def build_queue(
    run_path: Path,
    grade_paths: list[Path],
    out_dir: Path,
    *,
    target: str = "final",
    seed: str = DEFAULT_SEED,
    audit_fraction: float = DEFAULT_AUDIT_FRACTION,
) -> dict:
    results, run_meta, source_sha256 = summarize_run._load_sealed_run(run_path)
    ids = [str(row.get("id")) for row in results]
    if len(ids) != len(set(ids)):
        raise RuntimeError("source run contains duplicate problem ids")
    non_hle = [str(row.get("id")) for row in results if not _is_hle_row(row)]
    if non_hle:
        raise RuntimeError(f"adjudication queue accepts only HLE rows: {non_hle}")

    grade_paths = [Path(path) for path in grade_paths]
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported adjudication target: {target}")
    grade_inputs = _grade_input_records(grade_paths, target=target)
    graders = summarize_run._load_grades(
        grade_paths, source_sha256, target, set(ids),
    )
    if set(graders) != {record["grader"] for record in grade_inputs}:
        raise RuntimeError("verified grade identities do not match grade files")
    rationales, rationales_sha256 = _load_frozen_rationales(
        grade_paths, set(ids),
    )

    selected = select_adjudication_cases(
        results,
        graders,
        target=target,
        seed=seed,
        audit_fraction=audit_fraction,
    )
    queue_identity = {
        "schema_version": SCHEMA_VERSION,
        "source_run_sha256": source_sha256,
        "ordered_problem_ids_sha256": summarize_run.harness._sha256_json(ids),
        "grading_target": target,
        "grades": _grade_signature(grade_inputs),
        "canonical_rationales_sha256": rationales_sha256,
        "selection_seed": seed,
        "agreement_audit_fraction": audit_fraction,
    }
    queue_id = f"adj_{_identity_sha256(queue_identity)[:24]}"

    rows_by_id = {str(row.get("id")): row for row in results}
    manifest_cases = []
    blinded_cases = []
    for selection in selected:
        problem_id = selection["problem_id"]
        row = rows_by_id[problem_id]
        problem = row.get("problem")
        expected = row.get("expected")
        candidate = _candidate_answer(row, target)
        if not isinstance(problem, str) or not problem.strip():
            raise RuntimeError(f"certified case has no problem text: {problem_id}")
        if expected is None or not str(expected).strip():
            raise RuntimeError(f"certified case has no expected answer: {problem_id}")
        if candidate is None or not str(candidate).strip():
            raise RuntimeError(f"certified case has no candidate answer: {problem_id}")
        rationale = rationales.get(problem_id)
        if not isinstance(rationale, dict):
            raise RuntimeError(f"certified case has no canonical rationale: {problem_id}")
        case_id = "case_" + hashlib.sha256(
            f"{queue_id}:{problem_id}".encode("utf-8")
        ).hexdigest()[:20]
        case = {
            "case_id": case_id,
            "category": str(row.get("category") or "unknown"),
            "problem": problem,
            "expected_answer": str(expected),
            "candidate_answer": str(candidate),
            "canonical_rationale": rationale,
        }
        blinded_cases.append(case)
        manifest_cases.append({
            **selection,
            "case_id": case_id,
            "blinded_case_sha256": _identity_sha256(case),
        })

    blinded_cases.sort(key=lambda case: case["case_id"])
    manifest_cases.sort(key=lambda case: case["case_id"])
    cases_payload = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "cases": blinded_cases,
    }
    validate_blinded_cases(cases_payload, queue_id=queue_id)
    decisions_payload = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "cases_sha256": _sha256_bytes(_json_bytes(cases_payload)),
        "decision_standard": (
            "Set strict_correct=true only when the candidate answer itself "
            "conceptually matches the expected answer. A placeholder, refusal, "
            "or answer recoverable only from hidden reasoning is incorrect."
        ),
        "review_protocol": (
            "At least two blinded reviewers must assess each case independently. "
            "Resolve disagreements before completion and list every participating "
            "reviewer in reviewer_ids."
        ),
        "decisions": [
            {
                "case_id": case["case_id"],
                "status": "pending",
                "strict_correct": None,
                "reviewer_ids": [],
                "reasoning": None,
            }
            for case in blinded_cases
        ],
    }

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    cases_sha, cases_size = _write_new(out_dir / CASES_FILENAME, cases_payload)
    template_sha, template_size = _write_new(
        out_dir / DECISION_TEMPLATE_FILENAME, decisions_payload,
    )
    if decisions_payload["cases_sha256"] != cases_sha:
        raise RuntimeError("internal cases hash mismatch")

    reason_counts = Counter(
        reason for case in manifest_cases for reason in case["selection_reasons"]
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "queue_identity": queue_identity,
        "created_from": {
            "source_run": str(run_path.resolve()),
            "source_run_sha256": source_sha256,
            "source_run_id": run_meta.get("run_id"),
            "grading_target": target,
            "ordered_problem_ids_sha256": summarize_run.harness._sha256_json(ids),
            "problem_count": len(results),
            "certified_count": len(_certified_rows(results, target)),
            "grades": grade_inputs,
            "canonical_rationales_sha256": rationales_sha256,
        },
        "selection_policy": {
            "automatic_label": (
                "strict correctness of the selected candidate uses key_match only; "
                "extraction and other dispute categories receive no credit"
            ),
            "grader_disagreement": "all selected-candidate key-match disagreements",
            "apparent_certified_error": "any grader marks the candidate key_match false",
            "agreement_audit": {
                "fraction": audit_fraction,
                "rounding": "ceiling",
                "ranking": "SHA256(seed + ':' + problem_id)",
                "seed": seed,
            },
        },
        "selection_counts": {
            "queued": len(manifest_cases),
            "by_reason": dict(sorted(reason_counts.items())),
        },
        "cases": manifest_cases,
        "artifacts": {
            CASES_FILENAME: {"sha256": cases_sha, "size_bytes": cases_size},
            DECISION_TEMPLATE_FILENAME: {
                "sha256": template_sha,
                "size_bytes": template_size,
            },
        },
    }
    _write_new(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), default="final")
    parser.add_argument("--grade", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--audit-fraction", type=float, default=DEFAULT_AUDIT_FRACTION,
    )
    args = parser.parse_args()
    manifest = build_queue(
        args.run,
        args.grade,
        args.out_dir,
        target=args.target,
        seed=args.seed,
        audit_fraction=args.audit_fraction,
    )
    print(args.out_dir / MANIFEST_FILENAME)
    print(f"queued {manifest['selection_counts']['queued']} certified cases")


if __name__ == "__main__":
    main()
