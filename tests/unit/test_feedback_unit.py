from __future__ import annotations

from skillmoo.feedback import extract_feedback, should_retry


def test_extract_feedback_marks_compilation_as_pass_family() -> None:
    payload = {
        "status": "fail",
        "error_class": "none",
        "timed_out": False,
        "verifier": {
            "stdout": "FAILED tests/test_x.py::test_a - AssertionError\nE   AssertionError: x\n",
        },
    }
    signal = extract_feedback(payload)
    assert signal.recommended_family == "pass"
    assert "assertion" in signal.pattern_types
    assert signal.retryable is False


def test_extract_feedback_timeout_is_retryable_and_cost_family() -> None:
    payload = {"status": "timeout", "error_class": "none", "timed_out": True}
    signal = extract_feedback(payload)
    assert signal.retryable is True
    assert signal.recommended_family == "cost"
    assert should_retry(signal, retry_count=0, max_retries=1) is True
    assert should_retry(signal, retry_count=1, max_retries=1) is False


def test_agent_setup_failure_is_nonfatal_execution_error() -> None:
    payload = {
        "exception_info": {
            "exception_type": "RuntimeError",
            "exception_message": "Agent setup failed with exit code 127. See logs in jobs/demo/agent/setup",
        }
    }
    signal = extract_feedback(payload)
    assert signal.status == "execution_error"
    assert signal.error_class == "agent_setup_error"
    assert signal.retryable is True
    assert "Agent setup failed" in signal.failure_summary
