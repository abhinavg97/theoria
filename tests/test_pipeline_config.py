import yaml

from pipeline import load_config


def test_load_config_defaults_only():
    config = load_config()
    assert "solver" in config
    assert "_preamble" in config
    assert config["solver"]["model"] == "opus"


def test_load_config_dict_merge_overrides_only_given_keys(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({"solver": {"effort": "low"}}))

    config = load_config([str(override)])

    # `effort` is replaced, but sibling keys from defaults.yaml survive —
    # role overrides are a shallow dict-merge, not a wholesale replace.
    assert config["solver"]["effort"] == "low"
    assert config["solver"]["model"] == "opus"
    assert "prompt" in config["solver"]


def test_load_config_scalar_override_replaces_wholesale(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({"_preamble": "custom preamble"}))

    config = load_config([str(override)])

    assert config["_preamble"] == "custom preamble"


def test_load_config_multiple_overrides_apply_in_order_last_wins(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump({"solver": {"effort": "low"}}))
    second.write_text(yaml.safe_dump({"solver": {"effort": "high"}}))

    config = load_config([str(first), str(second)])

    assert config["solver"]["effort"] == "high"


def test_load_config_accepts_single_path_not_wrapped_in_list(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump({"solver": {"effort": "low"}}))

    config = load_config(str(override))

    assert config["solver"]["effort"] == "low"
