import json

from export_problem import _header_id, fmt_state, latest_run_file, load_run


def test_header_id_int_gets_p_prefix():
    assert _header_id(3) == "p3"


def test_header_id_non_int_passthrough():
    assert _header_id("custom-question") == "custom-question"


def test_fmt_state_empty():
    assert fmt_state([]) == "(empty)"
    assert fmt_state(None) == "(empty)"


def test_fmt_state_lists_each_entry_indexed():
    text = fmt_state(["a", "b"])

    assert text == "  [0] a\n  [1] b"


def test_latest_run_file_picks_most_recent_timestamp(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "hle_p7_20260101_090000.json").write_text("{}")
    (runs_dir / "hle_p7_20260301_120000.json").write_text("{}")
    (runs_dir / "hle_p9_20260501_000000.json").write_text("{}")

    result = latest_run_file(7, runs_dir=str(runs_dir))

    assert result == str(runs_dir / "hle_p7_20260301_120000.json")


def test_latest_run_file_returns_none_when_no_match(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    assert latest_run_file(42, runs_dir=str(runs_dir)) is None


def test_load_run_returns_first_element_of_list(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps([{"id": "p1"}, {"id": "p2"}]))

    assert load_run(str(path)) == {"id": "p1"}


def test_load_run_returns_none_for_empty_list(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps([]))

    assert load_run(str(path)) is None


def test_load_run_returns_none_for_missing_file(tmp_path):
    assert load_run(str(tmp_path / "does_not_exist.json")) is None


def test_load_run_returns_none_for_non_list_json(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"id": "p1"}))

    assert load_run(str(path)) is None
