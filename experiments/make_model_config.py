#!/usr/bin/env python3
"""Render one explicit all-role provider config from the frozen model matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "experiments/model_matrix.yaml"
ROLES = (
    "solver",
    "interpreter",
    "formalizer",
    "citation",
    "problem_given",
    "computation",
    "pedantry",
    "convention_lift",
    "initial_state",
)


def build_config(args: argparse.Namespace) -> dict:
    matrix_bytes = MATRIX_PATH.read_bytes()
    matrix = yaml.safe_load(matrix_bytes)
    try:
        spec = matrix["models"][args.model_key]
    except KeyError as exc:
        choices = ", ".join(sorted(matrix.get("models") or {}))
        raise SystemExit(f"unknown model key {args.model_key!r}; choose one of: {choices}") from exc

    quantization = args.quantization
    if quantization is None:
        quantization = "native-mxfp4" if args.model_key.startswith("gpt-oss-") else "none"
    system_role = args.system_role or spec.get("system_role", "system")
    served_model_name = (
        args.served_model_name
        or f"{spec['hf_id']}@{spec['revision']}"
    )
    tensor_parallel_size = (
        args.tensor_parallel_size or spec["tensor_parallel_size"]
    )
    max_num_seqs = args.max_num_seqs or matrix["serving_reference"]["max_num_seqs"]
    common = {
        "backend": "theoria_agent",
        "model": served_model_name,
        "model_source": "huggingface",
        "model_revision": spec["revision"],
        "dtype": args.dtype,
        "quantization": quantization,
        "server_hardware": args.server_hardware,
        "endpoint": args.endpoint,
        "context_length": args.context_length,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "system_role": system_role,
        "allow_shell": True,
        "allow_host_tools": False,
        "tool_timeout": args.tool_timeout,
        "tool_output_chars": args.tool_output_chars,
    }
    if args.top_k is not None:
        common["top_k"] = args.top_k
    if args.min_p is not None:
        common["min_p"] = args.min_p
    if args.reasoning_effort:
        common["reasoning_effort"] = args.reasoning_effort
    if spec.get("mode") == "hybrid_thinking_enabled":
        common["chat_template_kwargs"] = {"enable_thinking": True}

    return {
        "_experiment_model": {
            "matrix_key": args.model_key,
            "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "track": spec["track"],
            "hf_id": spec["hf_id"],
            "revision": spec["revision"],
            "license": spec["license"],
            "serving_backend": "vllm",
            "serving_backend_version": matrix["serving_reference"]["version"],
            "server": {
                "served_model_name": served_model_name,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_seqs": max_num_seqs,
                "max_model_len": args.context_length,
                "engine_seed": args.engine_seed,
                "endpoint": args.endpoint,
            },
        },
        **{role: copy.deepcopy(common) for role in ROLES},
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("model_key")
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--served-model-name")
    result.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    result.add_argument("--server-hardware", required=True)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--quantization")
    result.add_argument("--tensor-parallel-size", type=int)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--max-num-seqs", type=int)
    result.add_argument("--engine-seed", type=int, default=20260718)
    result.add_argument("--context-length", type=int, default=32768)
    result.add_argument("--max-turns", type=int, default=16)
    result.add_argument("--max-tokens", type=int, default=8192)
    result.add_argument("--temperature", type=float, required=True)
    result.add_argument("--top-p", type=float, required=True)
    result.add_argument("--top-k", type=int)
    result.add_argument("--min-p", type=float)
    result.add_argument("--seed", type=int, default=20260718)
    result.add_argument("--system-role", choices=("system", "user"))
    result.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    result.add_argument("--tool-timeout", type=float, default=120)
    result.add_argument("--tool-output-chars", type=int, default=12000)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if (
        args.context_length < 1
        or args.max_turns < 1
        or args.max_tokens < 1
        or (args.tensor_parallel_size is not None and args.tensor_parallel_size < 1)
        or (args.max_num_seqs is not None and args.max_num_seqs < 1)
    ):
        raise SystemExit("context length, max turns, and max tokens must be positive")
    if (
        not 0 <= args.temperature
        or not 0 < args.top_p <= 1
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        raise SystemExit("temperature must be nonnegative and top-p must be in (0, 1]")
    config = build_config(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
