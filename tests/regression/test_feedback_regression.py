from __future__ import annotations

from skillmoo.feedback import extract_feedback


def test_feedback_payload_regression_snapshot() -> None:
    payload = {
        "status": "execution_error",
        "error_class": "network_error",
        "timed_out": False,
        "verifier_stdout": "",
    }
    signal = extract_feedback(payload).to_dict()
    assert signal == {
        "status": "execution_error",
        "error_class": "network_error",
        "timeout": False,
        "retryable": True,
        "recommended_family": "cost",
        "failure_summary": "",
        "pattern_types": (),
    }
