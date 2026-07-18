from pathlib import Path

import yaml

from experiments.make_role_ablation_config import VERIFIER_ROLES, compose


def _write(path: Path, model: str) -> None:
    common = {"backend": "theoria_agent", "model": model}
    payload = {
        "_experiment_model": {"matrix_key": model},
        "solver": dict(common),
        **{role: dict(common) for role in VERIFIER_ROLES},
    }
    path.write_text(yaml.safe_dump(payload))


def test_compose_changes_only_solver_role_and_records_sources(tmp_path):
    weak = tmp_path / "weak.yaml"
    strong = tmp_path / "strong.yaml"
    _write(weak, "weak-model")
    _write(strong, "strong-model")

    config = compose(weak, strong)

    assert config["solver"]["model"] == "weak-model"
    assert all(config[role]["model"] == "strong-model" for role in VERIFIER_ROLES)
    assert "_experiment_model" not in config
    audit = config["_experiment_role_ablation"]
    assert audit["solver_config_sha256"]
    assert audit["verifier_config_sha256"]
    assert audit["role_assignment"]["solver"] == "solver_config"
    assert all(
        audit["role_assignment"][role] == "verifier_config"
        for role in VERIFIER_ROLES
    )
