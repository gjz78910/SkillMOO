from __future__ import annotations

from skillmoo.feedback import FeedbackSignal
from skillmoo.metrics import aggregate_experiment, summarize_task
from skillmoo.selection import EvaluationRecord


def test_metrics_schema_regression() -> None:
    records = [EvaluationRecord("task-a", 0, 0, "fail", 0.4, 0.7, 50.0, 2, "pass", "none", False, 0)]
    feedback = [FeedbackSignal("fail", "none", False, False, "pass", "FAILED", ("assertion",))]
    summary = summarize_task("task-a", records, feedback)
    payload = aggregate_experiment([summary])
    assert sorted(payload.keys()) == ["generated_at", "overall", "rows", "task_count"]
    assert sorted(payload["overall"].keys()) == ["cost_usd_mean", "duration_sec_mean", "pass_rate_mean"]
    assert sorted(payload["rows"][0].keys()) == [
        "best_bundle_size",
        "best_cost_usd",
        "best_duration_sec",
        "best_pass_rate",
        "best_selected_skill_ids",
        "cost_usd_mean",
        "duration_sec_mean",
        "failure_type_counts",
        "n_evaluations",
        "pass_rate_mean",
        "status_counts",
        "task_id",
    ]
