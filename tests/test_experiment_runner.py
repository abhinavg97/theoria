import hashlib
import json

from experiments import run_hle_condition


def test_paper_cohort_is_frozen_and_complete():
    ids = run_hle_condition.load_cohort("paper-200")

    assert len(ids) == 200
    assert len(set(ids)) == 200
    assert hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest() == run_hle_condition.PAPER_200_SHA256


def test_paper_ran_cohort_matches_headline_denominator():
    ids = run_hle_condition.load_cohort("paper-ran-185")

    assert len(ids) == 185
    assert len(set(ids)) == 185
    assert hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest() == run_hle_condition.PAPER_RAN_185_SHA256


def test_paper_cohort_aliases_are_stable():
    assert run_hle_condition.load_cohort("paper-185") == (
        run_hle_condition.load_cohort("paper-ran-185")
    )
    assert run_hle_condition.load_cohort("paper-200") == (
        run_hle_condition.load_cohort("paper-sampled-200")
    )


def test_paper_direct_id_subset_remains_available():
    ids = run_hle_condition.load_cohort("paper-194")

    assert len(ids) == 194
    assert len(set(ids)) == 194
    assert not set(run_hle_condition.PAPER_BLOCKED_ID_RECONSTRUCTION.values()) & set(ids)
    assert ids == run_hle_condition.load_cohort("paper-recoverable-194")
