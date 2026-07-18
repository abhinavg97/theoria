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


def test_paper_direct_id_subset_remains_available():
    ids = run_hle_condition.load_cohort("paper-194")

    assert len(ids) == 194
    assert len(set(ids)) == 194
    assert not set(run_hle_condition.PAPER_BLOCKED_ID_RECONSTRUCTION.values()) & set(ids)
