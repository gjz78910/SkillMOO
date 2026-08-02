from __future__ import annotations

from skillmoo.loop import classify_quarantine


def test_classify_quarantine_true_for_retryable_terminal_state() -> None:
    assert classify_quarantine("execution_error", retries_used=1, max_retries=1) is True
    assert classify_quarantine("timeout", retries_used=2, max_retries=2) is True


def test_classify_quarantine_false_for_non_retryable_state() -> None:
    assert classify_quarantine("fail", retries_used=1, max_retries=1) is False
