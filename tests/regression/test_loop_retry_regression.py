from __future__ import annotations

from skillmoo.loop import classify_quarantine


def test_quarantine_regression_matrix() -> None:
    matrix = {
        ("execution_error", 0, 1): False,
        ("execution_error", 1, 1): True,
        ("infra_error", 2, 2): True,
        ("pass", 2, 2): False,
    }
    for key, expected in matrix.items():
        status, used, limit = key
        assert classify_quarantine(status, retries_used=used, max_retries=limit) is expected
