#!/usr/bin/env python3
"""Validate a frozen HLE cohort and launch one audited condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

from loaders import HLE_DATASET_REVISION, load_hle


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "smoke-1": ROOT / "experiments/cohorts/hle_smoke_1_ids.txt",
    "smoke-5": ROOT / "experiments/cohorts/hle_smoke_5_ids.txt",
    "dev-30": ROOT / "experiments/cohorts/hle_dev_30_ids.txt",
}
PAPER_BLOCKED_ID_RECONSTRUCTION = {
    115: "66ff063787bfb80443d02df6",
    246: "671bc0c855449c636f4bbd36",
    376: "6725a933e10373a976b7e2a2",
    389: "6726941826b7fc6a39fbe581",
    541: "6759a235c0c22e78a0758d86",
    562: "677da0a433769e54d305f23c",
}
PAPER_200_SHA256 = "d8a5e29289350b59d50909bbe3a11b006fa3942831cfaaa2ab439300bda1cc43"
PAPER_RAN_185_SHA256 = (
    "31e912acbee669cce95ea10c9fb8a5241a23011cc174b0c2c28d6a87c2e54cea"
)


def _paper_ids(*, include_reconstructed: bool) -> list[str]:
    database = ROOT / "paper/audit.db"
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT problem_number, hle_id FROM problems "
            "ORDER BY dataset, problem_number"
        ).fetchall()
    if include_reconstructed:
        ids = [
            str(hle_id or PAPER_BLOCKED_ID_RECONSTRUCTION[problem_number])
            for problem_number, hle_id in rows
        ]
        digest = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != PAPER_200_SHA256:
            raise RuntimeError(
                f"paper cohort hash mismatch: {digest} != {PAPER_200_SHA256}"
            )
        return ids
    return [str(hle_id) for _, hle_id in rows if hle_id is not None]


def _paper_ran_ids() -> list[str]:
    """Return the 185 items behind the paper's headline denominator."""
    database = ROOT / "paper/audit.db"
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT hle_id FROM problems WHERE status = 'ran' "
            "ORDER BY dataset, problem_number"
        ).fetchall()
    ids = [str(hle_id) for (hle_id,) in rows]
    if any(not problem_id or problem_id == "None" for problem_id in ids):
        raise RuntimeError("paper ran cohort contains a missing HLE id")
    digest = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    if len(ids) != 185 or digest != PAPER_RAN_185_SHA256:
        raise RuntimeError(
            "paper ran cohort mismatch: "
            f"count={len(ids)}, sha256={digest}, "
            f"expected_count=185, expected_sha256={PAPER_RAN_185_SHA256}"
        )
    return ids


def _file_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_cohort(value: str) -> list[str]:
    if value in {"paper-ran-185", "paper-185"}:
        ids = _paper_ran_ids()
    elif value in {"paper-sampled-200", "paper-200"}:
        ids = _paper_ids(include_reconstructed=True)
    elif value in {"paper-194", "paper-recoverable", "paper-recoverable-194"}:
        ids = _paper_ids(include_reconstructed=False)
    else:
        path = COHORTS.get(value, Path(value))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise SystemExit(f"cohort does not exist: {path}")
        ids = _file_ids(path)
    if not ids:
        raise SystemExit("cohort is empty")
    if len(ids) != len(set(ids)):
        raise SystemExit("cohort contains duplicate ids")
    return ids


def validate_cohort(ids: list[str]) -> None:
    loaded = load_hle(ids=ids, revision=HLE_DATASET_REVISION)
    found = [problem["id"] for problem in loaded]
    if found != ids:
        requested_set = set(ids)
        found_set = set(found)
        missing = sorted(requested_set - found_set)
        unexpected = sorted(found_set - requested_set)
        raise SystemExit(
            "cohort does not match pinned HLE revision or requested order: "
            f"missing={missing}, unexpected={unexpected}, "
            f"order_matches={found == ids}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--model-config")
    parser.add_argument(
        "--config", action="append", default=[],
        help="Additional config overlay; repeat in the desired merge order.",
    )
    parser.add_argument("--search-config", default="configs/searxng_search.yaml")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--phase", required=True,
        choices=("smoke", "dev", "evaluation", "ablation"),
    )
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--image", default="theoria-sandbox-sage:latest")
    parser.add_argument("--tag")
    parser.add_argument("--run-path-file")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.trial < 1 or args.parallel < 1:
        raise SystemExit("--trial and --parallel must be positive")
    ids = load_cohort(args.cohort)
    validate_cohort(ids)
    cohort_sha256 = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()

    configs = [args.base_config]
    if args.model_config:
        configs.append(args.model_config)
    configs.extend(args.config)
    if args.search_config:
        configs.append(args.search_config)
    if args.no_repair:
        configs.append("configs/no_repair.yaml")

    command = [
        sys.executable, "-m", "cli", "hle",
        "--ids", ",".join(ids),
        "--backend", "theoria_agent",
        "--image", args.image,
        "--parallel", str(args.parallel),
        "--experiment-phase", args.phase,
        "--experiment-id", args.experiment_id,
        "--trial", str(args.trial),
        "--tag", args.tag or args.experiment_id,
        "--dataset-revision", HLE_DATASET_REVISION,
        "--watch" if args.watch else "--no-watch",
    ]
    for config in configs:
        command.extend(("--config", config))
    if args.resume:
        command.extend(("--resume", args.resume))
    if args.run_path_file:
        command.extend(("--run-path-file", args.run_path_file))

    print(
        f"Validated {len(ids)} unique problem ids "
        f"(cohort_sha256={cohort_sha256})."
    )
    print("Command:")
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
