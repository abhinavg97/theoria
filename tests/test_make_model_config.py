import argparse

from experiments import make_model_config


def _args(**overrides):
    values = {
        "model_key": "qwen3-8b",
        "served_model_name": None,
        "endpoint": "http://127.0.0.1:8000/v1",
        "server_hardware": "1x-H100-80GB",
        "dtype": "bfloat16",
        "quantization": None,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": None,
        "engine_seed": 20260718,
        "context_length": 32768,
        "max_turns": 16,
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": None,
        "seed": 20260718,
        "system_role": None,
        "reasoning_effort": None,
        "tool_timeout": 120,
        "tool_output_chars": 12000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_config_resolves_frozen_qwen_revision_for_every_role():
    config = make_model_config.build_config(_args())

    assert config["_experiment_model"]["revision"] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert config["_experiment_model"]["server"] == {
        "served_model_name": (
            "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218"
        ),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 1,
        "max_model_len": 32768,
        "engine_seed": 20260718,
        "endpoint": "http://127.0.0.1:8000/v1",
    }
    for role in make_model_config.ROLES:
        assert config[role]["model"] == (
            "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218"
        )
        assert config[role]["chat_template_kwargs"] == {"enable_thinking": True}
        assert config[role]["top_k"] == 20


def test_build_config_folds_deepseek_system_prompt_into_user_role():
    config = make_model_config.build_config(_args(
        model_key="deepseek-r1-distill-qwen-7b",
    ))

    assert config["solver"]["system_role"] == "user"


def test_build_config_records_gpt_oss_native_quantization():
    config = make_model_config.build_config(_args(
        model_key="gpt-oss-20b",
        reasoning_effort="high",
    ))

    assert config["solver"]["quantization"] == "native-mxfp4"
    assert config["solver"]["reasoning_effort"] == "high"
