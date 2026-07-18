#!/usr/bin/env python3
"""Materialize local GPQA inputs from the pinned analysis commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_source_artifacts import (
    ANALYSIS_COMMIT,
    GPQA_REVISION,
    _git_json,
)


def _write(path: Path, metadata: dict, problems: list[dict]) -> str:
    payload = {"metadata": metadata, "problems": problems}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_problem(problem: dict, source_index: int, source: str) -> dict:
    output = dict(problem)
    output.update({
        "id": str(problem["id"]),
        "category": str(problem.get("subdomain") or "unknown"),
        "answer_type": "multiple_choice",
        "source_row_index_0based": int(source_index),
        "paper65_source": source,
    })
    return output


def _options_from_question(question: str) -> dict[str, str]:
    matches = re.finditer(
        r"(?ms)^[ \t]*([A-D])\)[ \t]*(.*?)"
        r"(?=^[ \t]*[A-D]\)[ \t]*|\Z)",
        question,
    )
    return {match.group(1): match.group(2).strip() for match in matches}


def _normalize_option_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _verify_against_huggingface(problems: list[dict]) -> str | None:
    from datasets import load_dataset

    dataset = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        revision=GPQA_REVISION,
    )
    if len(dataset) != 198:
        raise RuntimeError(f"unexpected GPQA Diamond size: {len(dataset)}")
    for problem in problems:
        source = dataset[int(problem["source_row_index_0based"])]
        if problem.get("correct_text") != source["Correct Answer"]:
            raise RuntimeError(f"GPQA answer mismatch for {problem['id']}")
        if not str(problem["question"]).startswith(source["Question"]):
            raise RuntimeError(f"GPQA question mismatch for {problem['id']}")
        option_texts = [
            source["Correct Answer"],
            source["Incorrect Answer 1"],
            source["Incorrect Answer 2"],
            source["Incorrect Answer 3"],
        ]
        options = _options_from_question(str(problem["question"]))
        if set(options) != {"A", "B", "C", "D"}:
            raise RuntimeError(f"GPQA option labels are invalid for {problem['id']}")
        if sorted(map(_normalize_option_text, options.values())) != sorted(
            map(_normalize_option_text, option_texts)
        ):
            raise RuntimeError(f"GPQA option mismatch for {problem['id']}")
        answer_key = str(problem.get("answer") or "").strip().upper()
        if answer_key not in options:
            raise RuntimeError(f"GPQA answer key is invalid for {problem['id']}")
        if _normalize_option_text(options[answer_key]) != _normalize_option_text(
            problem["correct_text"]
        ):
            raise RuntimeError(
                f"GPQA shuffled answer key mismatch for {problem['id']}"
            )
    return getattr(dataset, "_fingerprint", None)


def materialize(
    repo: Path, out_dir: Path, *, verify_huggingface: bool = False,
) -> dict[str, dict]:
    step5 = _git_json(repo, "step5_gpqa_ood/gpqa_problems.json")
    step7 = _git_json(repo, "step7_gpqa_expanded/gpqa_problems.json")
    step7_clean = _git_json(
        repo, "step7_gpqa_expanded/gpqa_api_run_clean.json"
    )
    step5_rows = [
        _normalized_problem(problem, source_index, "step5_fixed_before_run")
        for problem, source_index in zip(
            step5["problems"], step5["metadata"]["sample_indices"]
        )
    ]
    step7_rows = [
        _normalized_problem(problem, source_index, "step7_fixed_before_run")
        for problem, source_index in zip(
            step7["problems"], step7["metadata"]["sample_indices"]
        )
    ]
    fixed100 = sorted(step5_rows + step7_rows, key=lambda row: row["id"])
    clean_ids = {str(row["id"]) for row in step7_clean}
    legacy_step7_rows = []
    for row in step7_rows:
        if row["id"] not in clean_ids:
            continue
        legacy_row = dict(row)
        legacy_row["paper65_source"] = (
            "step7_outcome_filtered_historical_subset_retained_after_success"
        )
        legacy_step7_rows.append(legacy_row)
    paper65 = sorted(
        step5_rows + legacy_step7_rows,
        key=lambda row: row["id"],
    )
    if len(fixed100) != 100 or len({row["id"] for row in fixed100}) != 100:
        raise RuntimeError("fixed GPQA cohort is not 100 unique problems")
    if len(paper65) != 65 or len({row["id"] for row in paper65}) != 65:
        raise RuntimeError("legacy GPQA cohort is not 65 unique problems")
    dataset_fingerprint = (
        _verify_against_huggingface(fixed100) if verify_huggingface else None
    )

    common = {
        "dataset_name": "Idavidrein/gpqa",
        "dataset_revision": GPQA_REVISION,
        "source": f"theoria_analysis@{ANALYSIS_COMMIT}",
        "answer_format": "shuffled_multiple_choice_letter",
        "dataset_fingerprint": dataset_fingerprint,
        "huggingface_revision_verified": verify_huggingface,
    }
    outputs = {}
    for name, problems, extra in (
        (
            "gpqa-fixed-100",
            fixed100,
            {
                "selection": "25 initial plus all 75 prespecified second-batch rows",
                "headline_eligible": True,
            },
        ),
        (
            "gpqa-legacy-paper-65",
            paper65,
            {
                "selection": "25 initial plus 40 retained after second-batch success",
                "headline_eligible": False,
                "warning": "Outcome-filtered and combines historical model stacks.",
            },
        ),
    ):
        path = out_dir / f"{name}.json"
        metadata = {**common, **extra, "name": name, "count": len(problems)}
        outputs[name] = {
            "path": str(path),
            "sha256": _write(path, metadata, problems),
            "count": len(problems),
        }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-hf", action="store_true",
        help="Verify every row against the gated pinned GPQA revision.",
    )
    args = parser.parse_args()
    outputs = materialize(
        args.analysis_repo.resolve(strict=True),
        args.out_dir.resolve(),
        verify_huggingface=args.verify_hf,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
