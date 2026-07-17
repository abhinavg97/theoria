import asyncio

import pipeline
from jsonschema import validate
from pipeline import (
    Proof,
    Step,
    Verdict,
    _format_failed_verdicts,
    _format_proof,
    _formalizer_decision_feedback,
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


def test_formalizer_schema_accepts_proof_without_reject_reason():
    validate({
        "action": "proof",
        "proof": {
            "initial_state": ["ANSWER"],
            "steps": [],
        },
    }, pipeline._formalizer_decision_schema())


def test_formalizer_missing_proof_is_reprompted(monkeypatch):
    calls = []

    proof = {
        "initial_state": ["ANSWER"],
        "steps": [{
            "state": ["42"],
            "justification_type": "computation",
            "justification": "6 * 7 = 42",
        }],
    }

    async def fake_llm(
        prompt,
        *,
        role="solver",
        schema=None,
        system=None,
        resume=None,
    ):
        calls.append({"role": role, "prompt": prompt, "resume": resume})
        if role == "solver":
            return "6 * 7 = 42", "solver-session"
        if role == "formalizer":
            formalizer_calls = [c for c in calls if c["role"] == "formalizer"]
            if len(formalizer_calls) == 1:
                return {"action": "proof"}, "formalizer-session"
            assert resume == "formalizer-session"
            assert "omitted the required 'proof' object" in prompt
            return {"action": "proof", "proof": proof}, "formalizer-session"
        if role in {"initial_state", "computation"}:
            return {"accepted": True, "reason": "ok"}, None
        raise AssertionError(f"unexpected role {role}")

    monkeypatch.setattr(pipeline, "llm", fake_llm)

    result = asyncio.run(pipeline.run("Compute 6*7."))

    assert result["verified"] is True
    assert result["answer"] == "42"
    assert result["attempts"][0]["phase"] == "formalizer_invalid"
    assert result["attempts"][0]["decision"] == {"action": "proof"}
    assert result["attempts"][1]["phase"] == "verify"
    assert [c["role"] for c in calls].count("formalizer") == 2


def test_formalizer_decision_feedback_reports_missing_action_fields():
    assert "omitted the required 'proof' object" in _formalizer_decision_feedback({
        "action": "proof",
    })
    assert "did not match the required schema" in _formalizer_decision_feedback({
        "action": "proof",
        "proof": {},
    })
    assert "reject_reason" in _formalizer_decision_feedback({
        "action": "reject",
    })
