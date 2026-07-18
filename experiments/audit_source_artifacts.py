#!/usr/bin/env python3
"""Audit the frozen paper artifacts without copying benchmark text."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_COMMIT = "52344a6bd2e301b0657fb16a9e9ccd9ad41bbb81"
HLE_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
GPQA_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
EXPECTED_FILES = {
    "paper/audit.db": (
        "0069ef11b1d4a03f0d84f202449be2d3e5dffcc9b21e16bd9d9cf2df70f5dd50"
    ),
    "step5_gpqa_ood/gpqa_problems.json": (
        "a9754f68c5c6129980fd14538c1466ca7449a29c48e6bd5c9b978212bf85ec9f"
    ),
    "step5_gpqa_ood/gpqa_ood_run.json": (
        "db63a2bf7a6269d78294377dc308a2858cae7aa0788b661c0df1f00181816f75"
    ),
    "step7_gpqa_expanded/gpqa_problems.json": (
        "be9411a99c3bf8a8003e2e1d5553d317318e01da1cf0c7bf68f20bd4f53ca2c2"
    ),
    "step7_gpqa_expanded/gpqa_api_run.json": (
        "bda5a8d63940cd04fa1d217f51a6de2415271c0fd7bb10caa134f6fe0187b4d0"
    ),
    "step7_gpqa_expanded/gpqa_api_run_clean.json": (
        "b8f8b3363e97e741aad54715fea727ef36ff9ba54f4517689a19ff9b7126b6f2"
    ),
    "step6_expanded_adversarial/generated_proofs.json": (
        "dc8190d61642cf8054dae94376fe2f6b49ef70f53b67b18b0323ef167e44ab43"
    ),
    "step6_expanded_adversarial/structured_results.json": (
        "c96488aad1c098c12eb0fc6239acfefbf893e248f3e854f84f7963729c1d352e"
    ),
    "step6_expanded_adversarial/holistic_results.json": (
        "11e59f5313e6be683478a2bbf40563e82832786d9bece6327079d1a093e9f599"
    ),
}
BLOCKED_HLE_IDS = {
    115: "66ff063787bfb80443d02df6",
    246: "671bc0c855449c636f4bbd36",
    376: "6725a933e10373a976b7e2a2",
    389: "6726941826b7fc6a39fbe581",
    541: "6759a235c0c22e78a0758d86",
    562: "677da0a433769e54d305f23c",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _git_blob(repo: Path, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{ANALYSIS_COMMIT}:{path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot read {path} at {ANALYSIS_COMMIT}: "
            + completed.stderr.decode(errors="replace").strip()
        )
    digest = _sha256(completed.stdout)
    if digest != EXPECTED_FILES[path]:
        raise RuntimeError(
            f"artifact hash mismatch for {path}: {digest} != {EXPECTED_FILES[path]}"
        )
    return completed.stdout


def _git_json(repo: Path, path: str):
    return json.loads(_git_blob(repo, path))


def audit_hle(repo: Path) -> dict:
    database_bytes = _git_blob(repo, "paper/audit.db")
    database_sha = _sha256(database_bytes)
    local_database = ROOT / "paper/audit.db"
    local_database_sha = _sha256(local_database.read_bytes())
    if local_database_sha != database_sha:
        raise RuntimeError(
            "local paper/audit.db differs from the analysis source of truth: "
            f"{local_database_sha} != {database_sha}"
        )
    with tempfile.NamedTemporaryFile(suffix=".db") as database:
        database.write(database_bytes)
        database.flush()
        with sqlite3.connect(database.name) as connection:
            rows = connection.execute(
                "SELECT problem_number, dataset, hle_id, status, verified "
                "FROM problems ORDER BY dataset, problem_number"
            ).fetchall()
            strict_rows = connection.execute(
                "SELECT DISTINCT problem_number FROM grades "
                "WHERE target = 'final' "
                "AND (key_match = 1 OR dispute_category = 'extraction')"
            ).fetchall()
            key_match_rows = connection.execute(
                "SELECT DISTINCT problem_number FROM grades "
                "WHERE target = 'final' AND key_match = 1"
            ).fetchall()
            extraction_rows = connection.execute(
                "SELECT DISTINCT problem_number FROM grades "
                "WHERE target = 'final' AND dispute_category = 'extraction'"
            ).fetchall()
    strict_ids = {int(row[0]) for row in strict_rows}
    key_match_ids = {int(row[0]) for row in key_match_rows}
    extraction_ids = {int(row[0]) for row in extraction_rows}
    sampled_ids = [
        str(hle_id or BLOCKED_HLE_IDS[int(number)])
        for number, _dataset, hle_id, _status, _verified in rows
    ]
    ran_rows = [row for row in rows if row[3] == "ran"]
    ran_ids = [str(row[2]) for row in ran_rows]
    db_certified = [row for row in ran_rows if row[4] == 1]
    arxiv_certified = [row for row in db_certified if row[0] != 134]
    return {
        "dataset_revision": HLE_REVISION,
        "audit_db_sha256": database_sha,
        "local_audit_db_sha256": local_database_sha,
        "sampled": {
            "count": len(sampled_ids),
            "ordered_ids_sha256": _canonical_sha256(sampled_ids),
        },
        "ran": {
            "count": len(ran_ids),
            "ordered_ids_sha256": _canonical_sha256(ran_ids),
        },
        "status_counts": dict(Counter(str(row[3]) for row in rows)),
        "committed_db_certified": len(db_certified),
        "arxiv_postfix_certified_excluding_p134": len(arxiv_certified),
        "arxiv_postfix_strict_correct": sum(
            int(row[0]) in strict_ids for row in arxiv_certified
        ),
        "arxiv_postfix_selected_candidate_key_match": sum(
            int(row[0]) in key_match_ids for row in arxiv_certified
        ),
        "arxiv_postfix_extraction_only_problem_numbers": sorted(
            int(row[0]) for row in arxiv_certified
            if int(row[0]) in extraction_ids
            and int(row[0]) not in key_match_ids
        ),
        "warning": (
            "The committed DB has 106 certifications; the arXiv 105/185 "
            "headline is a post-fix bucket that excludes p134. Its 96 correct "
            "credits two extraction-only cases; the selected-candidate-only "
            "count is 94."
        ),
    }


def audit_gpqa(repo: Path) -> dict:
    step5_problems = _git_json(repo, "step5_gpqa_ood/gpqa_problems.json")
    step5_run = _git_json(repo, "step5_gpqa_ood/gpqa_ood_run.json")
    step7_problems = _git_json(repo, "step7_gpqa_expanded/gpqa_problems.json")
    step7_full = _git_json(repo, "step7_gpqa_expanded/gpqa_api_run.json")
    step7_clean = _git_json(repo, "step7_gpqa_expanded/gpqa_api_run_clean.json")

    if step5_problems["metadata"].get("dataset_revision") != GPQA_REVISION:
        raise RuntimeError("unexpected GPQA revision in step5 metadata")
    step7_indices = step7_problems["metadata"]["sample_indices"]
    step7_by_id = {
        str(problem["id"]): step7_indices[int(str(problem["id"])[5:]) - 25]
        for problem in step7_problems["problems"]
    }
    manifest = [
        {
            "id": str(problem["id"]),
            "source": "step5_fixed_before_run",
            "source_row_index_0based": source_index,
        }
        for problem, source_index in zip(
            step5_problems["problems"],
            step5_problems["metadata"]["sample_indices"],
        )
    ]
    manifest.extend({
        "id": str(row["id"]),
        "source": "step7_retained_after_success",
        "source_row_index_0based": step7_by_id[str(row["id"])],
    } for row in step7_clean)

    combined = list(step5_run) + list(step7_clean)
    return {
        "dataset_revision": GPQA_REVISION,
        "paper65_count": len(manifest),
        "paper65_manifest_sha256": _canonical_sha256(manifest),
        "step5": {
            "selected_before_run": len(step5_problems["problems"]),
            "results": len(step5_run),
            "transport_errors": sum(bool(row.get("error")) for row in step5_run),
        },
        "step7": {
            "selected_before_run": len(step7_problems["problems"]),
            "results_written": len(step7_full),
            "retained_clean": len(step7_clean),
            "errored_results": sum(bool(row.get("error")) for row in step7_full),
            "never_ran": len(step7_problems["problems"]) - len(step7_full),
        },
        "raw_result_fields": {
            "certified": sum(bool(row.get("verified")) for row in combined),
            "certified_and_substring_correct": sum(
                bool(row.get("verified")) and bool(row.get("correct"))
                for row in combined
            ),
            "substring_correct": sum(bool(row.get("correct")) for row in combined),
        },
        "paper_reported_after_external_grading": {
            "certified": 34,
            "certified_correct": 33,
            "solver_correct": 59,
        },
        "warnings": [
            "The second-batch 40 were retained after observing run success.",
            "Only 64/65 items completed; one transport failure is counted in the denominator.",
            "Step5 and step7 used different model stacks.",
            "Use this set only as a labeled legacy replication stratum.",
        ],
    }


def audit_poison(repo: Path) -> dict:
    proofs = _git_json(repo, "step6_expanded_adversarial/generated_proofs.json")
    structured = _git_json(
        repo, "step6_expanded_adversarial/structured_results.json"
    )
    holistic = _git_json(repo, "step6_expanded_adversarial/holistic_results.json")
    structured_by_id = {str(row["problem_id"]): row for row in structured}
    holistic_by_id = {str(row["problem_id"]): row for row in holistic}
    ids = [str(row["problem_id"]) for row in proofs]
    if set(ids) != set(structured_by_id) or set(ids) != set(holistic_by_id):
        raise RuntimeError("poison proof and result ids do not match")

    poisoned = [row for row in proofs if row.get("error_type") != "control"]
    controls = [row for row in proofs if row.get("error_type") == "control"]
    pairs = Counter()
    for row in poisoned:
        pid = str(row["problem_id"])
        structured_caught = bool(structured_by_id[pid].get("caught_by_judge"))
        holistic_caught = bool(holistic_by_id[pid].get("caught"))
        key = (
            "both" if structured_caught and holistic_caught
            else "structured_only" if structured_caught
            else "holistic_only" if holistic_caught
            else "neither"
        )
        pairs[key] += 1
    return {
        "artifact_sha256": EXPECTED_FILES[
            "step6_expanded_adversarial/generated_proofs.json"
        ],
        "poisoned": len(poisoned),
        "controls": len(controls),
        "poison_counts_by_type": dict(Counter(
            str(row.get("error_type")) for row in poisoned
        )),
        "paired_detection": dict(pairs),
        "structured_detected": sum(
            bool(structured_by_id[str(row["problem_id"])].get("caught_by_judge"))
            for row in poisoned
        ),
        "holistic_detected": sum(
            bool(holistic_by_id[str(row["problem_id"])].get("caught"))
            for row in poisoned
        ),
        "structured_control_rejections": sum(
            bool(structured_by_id[str(row["problem_id"])].get("caught_by_judge"))
            for row in controls
        ),
        "holistic_control_rejections": sum(
            bool(holistic_by_id[str(row["problem_id"])].get("caught"))
            for row in controls
        ),
        "warnings": [
            "The generated poison/control labels were not independently validated.",
            "The paper reports poison sensitivity but omits the 12/15 structured control rejection rate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.analysis_repo.resolve(strict=True)
    report = {
        "schema_version": 1,
        "analysis_commit": ANALYSIS_COMMIT,
        "hle": audit_hle(repo),
        "gpqa": audit_gpqa(repo),
        "poisoned_proofs": audit_poison(repo),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
