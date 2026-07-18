import argparse

from experiments import serve_vllm


def _args(model_key):
    return argparse.Namespace(
        model_key=model_key,
        execute=False,
        served_model_name=None,
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=None,
        max_num_seqs=None,
        engine_seed=None,
        gpu_memory_utilization=0.9,
        host="127.0.0.1",
        port=8000,
    )


def test_qwen_command_pins_weights_tokenizer_and_thinking_mode():
    command = serve_vllm.build_command(_args("qwen3-8b"))

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert command[command.index("--revision") + 1] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert command[command.index("--tokenizer-revision") + 1] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert command[command.index("--served-model-name") + 1] == (
        "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert command[command.index("--seed") + 1] == "20260718"
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true}'
    )


def test_mistral_command_uses_official_loader_flags():
    command = serve_vllm.build_command(
        _args("mistral-small-3.2-24b-instruct-2506")
    )

    assert command[command.index("--tokenizer-mode") + 1] == "mistral"
    assert command[command.index("--config-format") + 1] == "mistral"
    assert command[command.index("--load-format") + 1] == "mistral"


def test_engine_seed_can_be_frozen_for_a_sensitivity_condition():
    args = _args("qwen3-8b")
    args.engine_seed = 17

    command = serve_vllm.build_command(args)

    assert command[command.index("--seed") + 1] == "17"
