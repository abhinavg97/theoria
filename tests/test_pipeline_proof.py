import pipeline
from pipeline import (
    Proof,
    Step,
    Verdict,
    _format_failed_verdicts,
    _format_proof,
    _proof_from_dict,
    _proof_to_dict,
    max_solver_answers,
    max_verify_attempts,
)


def _sample_proof() -> Proof:
    return Proof(
        initial_state=["n is an integer"],
        steps=[
            Step(
                state=["n^2 is an integer"],
                justification_type="computation",
                justification="n * n",
            ),
        ],
    )


def test_proof_dict_round_trip():
    proof = _sample_proof()
    data = _proof_to_dict(proof)
    restored = _proof_from_dict(data)

    assert restored == proof


def test_proof_to_dict_shape():
    data = _proof_to_dict(_sample_proof())

    assert data == {
        "initial_state": ["n is an integer"],
        "steps": [{
            "state": ["n^2 is an integer"],
            "justification_type": "computation",
            "justification": "n * n",
        }],
    }


def test_format_proof_includes_every_state_and_justification():
    text = _format_proof(_sample_proof())

    assert "State 0:" in text
    assert "['n is an integer']" in text
    assert "State 1:" in text
    assert "[computation] n * n" in text


def test_format_failed_verdicts_skips_accepted_steps():
    proof = _sample_proof()
    verdicts = [Verdict(accepted=True, reason="fine")]

    text = _format_failed_verdicts(proof, verdicts)

    assert text == ""


def test_format_failed_verdicts_reports_rejected_step_and_state0():
    proof = _sample_proof()
    verdicts = [Verdict(accepted=False, reason="unjustified")]
    state0_verdict = Verdict(accepted=False, reason="n is undefined")

    text = _format_failed_verdicts(proof, verdicts, state0_verdict)

    assert "State 0 (initial_state) FAILED" in text
    assert "n is undefined" in text
    assert "Step 1 [computation] FAILED: n * n" in text
    assert "unjustified" in text


def test_limits_fall_back_to_defaults_when_config_empty(monkeypatch):
    monkeypatch.setitem(pipeline.CONFIG, "_limits", {})

    assert max_verify_attempts() == 3
    assert max_solver_answers() == 3


def test_limits_read_from_config_when_present(monkeypatch):
    monkeypatch.setitem(pipeline.CONFIG, "_limits", {
        "max_verify_attempts": 5,
        "max_solver_answers": 1,
    })

    assert max_verify_attempts() == 5
    assert max_solver_answers() == 1
