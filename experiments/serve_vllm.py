#!/usr/bin/env python3
"""Print or execute the pinned vLLM command for one matrix entry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "experiments/model_matrix.yaml"


def build_command(args: argparse.Namespace) -> list[str]:
    matrix = yaml.safe_load(MATRIX_PATH.read_text())
    try:
        spec = matrix["models"][args.model_key]
    except KeyError as exc:
        choices = ", ".join(sorted(matrix.get("models") or {}))
        raise SystemExit(f"unknown model key {args.model_key!r}; choose one of: {choices}") from exc
    reference = matrix["serving_reference"]
    served_model_name = (
        args.served_model_name
        or f"{spec['hf_id']}@{spec['revision']}"
    )
    command = [
        "vllm", "serve", spec["hf_id"],
        "--revision", spec["revision"],
        "--tokenizer-revision", spec["revision"],
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(
            args.tensor_parallel_size or spec["tensor_parallel_size"]
        ),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len or reference["max_model_len"]),
        "--max-num-seqs", str(args.max_num_seqs or reference["max_num_seqs"]),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--generation-config", "vllm",
        "--seed", str(
            args.engine_seed
            if args.engine_seed is not None
            else reference["engine_seed"]
        ),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if spec.get("reasoning_parser"):
        command.extend(("--reasoning-parser", spec["reasoning_parser"]))
    if spec.get("mode") == "hybrid_thinking_enabled":
        command.extend((
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": True}, separators=(",", ":")),
        ))
    if spec.get("vllm_loader") == "mistral":
        command.extend((
            "--tokenizer-mode", "mistral",
            "--config-format", "mistral",
            "--load-format", "mistral",
        ))
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("model_key")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--served-model-name")
    result.add_argument("--tensor-parallel-size", type=int)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--max-model-len", type=int)
    result.add_argument("--max-num-seqs", type=int)
    result.add_argument("--engine-seed", type=int)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    return result


def main() -> None:
    args = parser().parse_args()
    for name in ("tensor_parallel_size", "max_model_len", "max_num_seqs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.execute:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
