from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

RETRYABLE_STATUSES = {"execution_error", "infra_error", "timeout"}


@dataclass(frozen=True)
class FeedbackSignal:
    status: str
    error_class: str
    timeout: bool
    retryable: bool
    recommended_family: str
    failure_summary: str
    pattern_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_feedback(payload: dict[str, Any]) -> FeedbackSignal:
    status = str(payload.get("status") or "")
    error_class = str(payload.get("error_class") or "none")
    exception_info = payload.get("exception_info")
    if exception_info and status not in {"pass", "fail", "timeout"}:
        error_class = _exception_error_class(exception_info)
        if error_class in {"NonZeroAgentExitCodeError", "agent_setup_error"}:
            status = "execution_error"
        else:
            status = "infra_error"
    timeout = bool(payload.get("timed_out", False) or status == "timeout")
    retryable = status in RETRYABLE_STATUSES
    verifier_output = _extract_verifier_output(payload)
    if not verifier_output:
        verifier_output = _extract_exception_message(exception_info)
    failure_summary = _extract_failure_summary(verifier_output)
    failures = _parse_test_failures(verifier_output)
    pattern_types = tuple(sorted({str(item.get("pattern_type", "unknown")) for item in failures}))
    return FeedbackSignal(
        status=status,
        error_class=error_class,
        timeout=timeout,
        retryable=retryable,
        recommended_family=_recommend_family(status, error_class, timeout, pattern_types),
        failure_summary=failure_summary,
        pattern_types=pattern_types,
    )


def _exception_error_class(exception_info: Any) -> str:
    if not isinstance(exception_info, dict):
        return "infra_error"
    message = str(exception_info.get("exception_message") or "").lower()
    if "agent setup failed" in message:
        return "agent_setup_error"
    if "docker" in message or "compose" in message or "buildkit" in message:
        return "docker_error"
    if "registry-1.docker.io" in message or "deadline exceeded" in message:
        return "network_error"
    return str(exception_info.get("exception_type") or "infra_error")


def _extract_exception_message(exception_info: Any) -> str:
    if not isinstance(exception_info, dict):
        return ""
    return str(exception_info.get("exception_message") or "")


def should_retry(signal: FeedbackSignal, retry_count: int, max_retries: int) -> bool:
    if retry_count >= max_retries:
        return False
    return signal.retryable


def _recommend_family(
    status: str,
    error_class: str,
    timeout: bool,
    pattern_types: tuple[str, ...],
) -> str:
    if timeout:
        return "cost"
    if status == "pass":
        return "length"
    if "compilation" in pattern_types or "assertion" in pattern_types:
        return "pass"
    if error_class in {"network_error", "setup_rate_limit", "infra_error"}:
        return "cost"
    return "pass"


def _extract_verifier_output(payload: dict[str, Any]) -> str:
    verifier = payload.get("verifier")
    if isinstance(verifier, dict):
        stdout = verifier.get("stdout")
        if isinstance(stdout, str):
            return stdout
    direct = payload.get("verifier_stdout")
    if isinstance(direct, str):
        return direct
    return ""


def _extract_failure_summary(content: str, max_chars: int = 10_000) -> str:
    if not content:
        return ""
    lines = content.splitlines()
    picked: list[str] = []
    patterns = [
        re.compile(r"^FAILED"),
        re.compile(r"^ERROR"),
        re.compile(r"AssertionError"),
        re.compile(r"Agent setup failed"),
        re.compile(r"cannot find symbol"),
        re.compile(r"not found:"),
    ]
    for line in lines:
        if any(pattern.search(line) for pattern in patterns):
            picked.append(line)
    text = "\n".join(picked)
    return text[:max_chars]


def _parse_test_failures(verifier_output: str) -> list[dict[str, str]]:
    if not verifier_output:
        return []
    out: list[dict[str, str]] = []
    for line in verifier_output.splitlines():
        if "AssertionError" in line:
            out.append({"test_name": "unknown_test", "error_pattern": "AssertionError", "pattern_type": "assertion"})
        elif "cannot find symbol" in line or "not found:" in line:
            out.append(
                {
                    "test_name": "unknown_test",
                    "error_pattern": "compilation_error",
                    "pattern_type": "compilation",
                }
            )
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in out:
        key = f"{row['pattern_type']}::{row['error_pattern']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
