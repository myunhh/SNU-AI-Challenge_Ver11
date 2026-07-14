import csv

import pytest

from snuai11 import perm
from snuai11.submission import format_answer, parse_answer, write_submission


def test_format_has_spaces():
    assert format_answer((0, 1, 2, 3)) == "[1, 2, 3, 4]"
    assert format_answer((3, 2, 0, 1)) == "[4, 3, 1, 2]"


def test_parse_roundtrip():
    for rank in perm.ALL_PERMS:
        assert parse_answer(format_answer(rank)) == rank


def test_parse_rejects_bad():
    with pytest.raises(ValueError):
        parse_answer("[1, 1, 2, 3]")
    with pytest.raises(ValueError):
        parse_answer("[1, 2, 3]")


def test_write_submission_follows_sample_order(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    with open(sample, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Answer"])
        for i in ["b", "a", "c"]:
            w.writerow([i, "[1, 2, 3, 4]"])

    rows = [("a", (0, 1, 2, 3)), ("c", (3, 2, 1, 0)), ("b", (1, 0, 2, 3))]
    out = write_submission(rows, tmp_path / "submission.csv", sample)
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["Id"] for r in got] == ["b", "a", "c"]
    assert got[0]["Answer"] == "[2, 1, 3, 4]"
    assert got[2]["Answer"] == "[4, 3, 2, 1]"


def test_write_submission_id_mismatch_raises(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    with open(sample, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Answer"])
        w.writerow(["only", "[1, 2, 3, 4]"])
    with pytest.raises(ValueError):
        write_submission([("other", (0, 1, 2, 3))], tmp_path / "s.csv", sample)
