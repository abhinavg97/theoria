import asyncio

import pytest

from llm import StructuredOutputError
import pipeline


def test_agent_prompt_appends_provider_suffix(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        {
            "_preamble": "environment",
            "citation": {
                "prompt": "judge prompt",
                "preamble": True,
                "prompt_suffix": "provider policy",
            },
        },
    )

    assert pipeline.agent_prompt("citation") == (
        "environment\n\njudge prompt\n\nprovider policy"
    )


def test_coerce_formalizer_decision_accepts_serialized_object():
    assert pipeline._coerce_formalizer_decision(
        '{"action": "reject", "reject_reason": "bad"}'
    ) == {"action": "reject", "reject_reason": "bad"}


@pytest.mark.parametrize("value", ["not json", [], None])
def test_coerce_formalizer_decision_rejects_provider_failures(value):
    with pytest.raises(RuntimeError):
        pipeline._coerce_formalizer_decision(value)


def test_codex_formalizer_repair_is_stateless(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        {"formalizer": {"backend": "codex", "prompt": "formalize"}},
    )
    calls = []

    async def fake_llm(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return {"action": "reject", "reject_reason": "still wrong"}, None

    monkeypatch.setattr(pipeline, "llm", fake_llm)
    prior = pipeline.Proof(
        initial_state=["ANSWER = ?"],
        steps=[
            pipeline.Step(
                state=["ANSWER = 1"],
                justification_type="computation",
                justification="computed",
            )
        ],
    )

    asyncio.run(
        pipeline._formalizer_call(
            "problem",
            "solution",
            "unused-session",
            "step 1 failed",
            prior_proof=prior,
        )
    )

    prompt, kwargs = calls[0]
    assert "Your previous proof attempt" in prompt
    assert "step 1 failed" in prompt
    assert kwargs["role"] == "formalizer"
    assert "schema" in kwargs
    assert "resume" not in kwargs


def test_judge_schema_failure_becomes_failed_verdict(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        {"computation": {"prompt": "Check {step_number}"}},
    )

    async def fail_llm(*_args, **_kwargs):
        raise StructuredOutputError("bad provider output")

    monkeypatch.setattr(pipeline, "llm", fail_llm)
    step = pipeline.Step(
        state=["ANSWER = 1"],
        justification_type="computation",
        justification="computed",
    )
    proof = pipeline.Proof(initial_state=["ANSWER = ?"], steps=[step])

    verdict, _ = asyncio.run(
        pipeline.judge(step, proof.initial_state, "problem", proof, 1)
    )

    assert verdict.accepted is False
    assert verdict.infrastructure_failure is True
    assert "STRUCTURED OUTPUT FAILURE" in verdict.reason


def test_judge_ignores_provider_attempt_to_set_internal_failure_flag(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        {"computation": {"prompt": "Check {step_number}"}},
    )

    async def fake_llm(*_args, **_kwargs):
        return {
            "accepted": True,
            "reason": "valid",
            "infrastructure_failure": True,
        }, None

    monkeypatch.setattr(pipeline, "llm", fake_llm)
    step = pipeline.Step(
        state=["ANSWER = 1"],
        justification_type="computation",
        justification="computed",
    )
    proof = pipeline.Proof(initial_state=["ANSWER = ?"], steps=[step])

    verdict, _ = asyncio.run(
        pipeline.judge(step, proof.initial_state, "problem", proof, 1)
    )

    assert verdict.accepted is True
    assert verdict.infrastructure_failure is False


def test_pedantry_schema_failure_does_not_override_rejection(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        {"pedantry": {"prompt": "Check rejection"}},
    )

    async def fail_llm(*_args, **_kwargs):
        raise StructuredOutputError("bad provider output")

    monkeypatch.setattr(pipeline, "llm", fail_llm)
    step = pipeline.Step(
        state=["ANSWER = 1"],
        justification_type="computation",
        justification="computed",
    )
    proof = pipeline.Proof(initial_state=["ANSWER = ?"], steps=[step])

    is_pedantic, reason = asyncio.run(pipeline._pedantry_check(
        pipeline.Verdict(False, "wrong"),
        step,
        proof.initial_state,
        1,
        "problem",
        proof,
    ))

    assert is_pedantic is False
    assert "STRUCTURED OUTPUT FAILURE" in reason


def test_infrastructure_verdict_is_not_sent_to_pedantry(monkeypatch):
    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("pedantry must not override infrastructure failure")

    monkeypatch.setattr(pipeline, "_pedantry_check", fail_if_called)
    proof = pipeline.Proof(
        initial_state=["ANSWER = ?"],
        steps=[pipeline.Step(
            state=["ANSWER = 1"],
            justification_type="computation",
            justification="computed",
        )],
    )
    verdict = pipeline.Verdict(
        accepted=False,
        reason="provider failed",
        infrastructure_failure=True,
    )

    updated, _, records = asyncio.run(
        pipeline._filter_pedantic(proof, [verdict], "problem")
    )

    assert updated == [verdict]
    assert records == []


def test_infrastructure_verdict_is_non_actionable_repair_text():
    proof = pipeline.Proof(
        initial_state=["ANSWER = ?"],
        steps=[
            pipeline.Step(
                state=["ANSWER = 1"],
                justification_type="computation",
                justification="computed",
            ),
            pipeline.Step(
                state=["ANSWER = 2"],
                justification_type="algebra",
                justification="added one",
            ),
        ],
    )

    text = pipeline._format_failed_verdicts(
        proof,
        [
            pipeline.Verdict(
                accepted=False,
                reason="provider schema retries exhausted",
                infrastructure_failure=True,
            ),
            pipeline.Verdict(accepted=False, reason="algebra does not follow"),
        ],
    )

    assert "Step 1 [computation] INFRASTRUCTURE FAILURE" in text
    assert "not a proof objection" in text
    assert "Do not revise this step" in text
    assert "Step 2 [algebra] FAILED" in text
