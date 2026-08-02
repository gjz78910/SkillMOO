from __future__ import annotations

from skillmoo.feedback import FeedbackSignal
from skillmoo.metrics import aggregate_experiment, summarize_task
from skillmoo.selection import EvaluationRecord


def test_summarize_task_aggregates_means_and_counts() -> None:
    records = [
        EvaluationRecord("task-a", 0, 0, "pass", 1.0, 1.2, 100.0, 2, "pass", "none", False, 0),
        EvaluationRecord("task-a", 0, 1, "fail", 0.5, 0.8, 80.0, 1, "cost", "none", False, 1),
    ]
    feedback = [
        FeedbackSignal("pass", "none", False, False, "length", "", ("none",)),
        FeedbackSignal("fail", "none", False, False, "pass", "", ("assertion",)),
    ]
    summary = summarize_task("task-a", records, feedback)
    assert summary.pass_rate_mean == 0.75
    assert summary.cost_usd_mean == 1.0
    assert summary.status_counts == {"pass": 1, "fail": 1}
    assert summary.failure_type_counts == {"assertion": 1, "none": 1}


def test_aggregate_experiment_has_expected_shape() -> None:
    records = [EvaluationRecord("task-a", 0, 0, "pass", 1.0, 1.2, 100.0, 2, "pass", "none", False, 0)]
    feedback = [FeedbackSignal("pass", "none", False, False, "length", "", ("none",))]
    summary = summarize_task("task-a", records, feedback)
    payload = aggregate_experiment([summary])
    assert payload["task_count"] == 1
    assert payload["overall"]["pass_rate_mean"] == 1.0
    assert payload["rows"][0]["task_id"] == "task-a"
