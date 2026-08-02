from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from .feedback import FeedbackSignal
from .io import now_iso
from .selection import EvaluationRecord, best_record, status_counts


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    n_evaluations: int
    pass_rate_mean: float
    cost_usd_mean: float
    duration_sec_mean: float
    best_pass_rate: float
    best_cost_usd: float
    best_duration_sec: float
    best_bundle_size: int
    best_selected_skill_ids: tuple[str, ...]
    status_counts: dict[str, int]
    failure_type_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_task(task_id: str, records: list[EvaluationRecord], feedback: list[FeedbackSignal]) -> TaskSummary:
    pass_rates = [item.pass_rate for item in records] or [0.0]
    costs = [item.cost_usd for item in records] or [0.0]
    durations = [item.duration_sec for item in records] or [0.0]
    best = best_record(records)
    return TaskSummary(
        task_id=task_id,
        n_evaluations=len(records),
        pass_rate_mean=round(mean(pass_rates), 6),
        cost_usd_mean=round(mean(costs), 6),
        duration_sec_mean=round(mean(durations), 6),
        best_pass_rate=round(best.pass_rate, 6) if best else 0.0,
        best_cost_usd=round(best.cost_usd, 6) if best else 0.0,
        best_duration_sec=round(best.duration_sec, 6) if best else 0.0,
        best_bundle_size=best.bundle_size if best else 0,
        best_selected_skill_ids=best.selected_skill_ids if best else tuple(),
        status_counts=status_counts(records),
        failure_type_counts=_failure_type_counts(feedback),
    )


def aggregate_experiment(summaries: list[TaskSummary]) -> dict[str, Any]:
    rows = [summary.to_dict() for summary in summaries]
    overall_pass = mean([row["pass_rate_mean"] for row in rows]) if rows else 0.0
    overall_cost = mean([row["cost_usd_mean"] for row in rows]) if rows else 0.0
    overall_duration = mean([row["duration_sec_mean"] for row in rows]) if rows else 0.0
    return {
        "generated_at": now_iso(),
        "task_count": len(rows),
        "overall": {
            "pass_rate_mean": round(overall_pass, 6),
            "cost_usd_mean": round(overall_cost, 6),
            "duration_sec_mean": round(overall_duration, 6),
        },
        "rows": rows,
    }


def summary_rows_for_csv(summaries: list[TaskSummary]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summaries:
        row = summary.to_dict()
        row["status_counts"] = _flat_map(summary.status_counts)
        row["failure_type_counts"] = _flat_map(summary.failure_type_counts)
        row["best_selected_skill_ids"] = ";".join(summary.best_selected_skill_ids)
        out.append(row)
    return out


def _failure_type_counts(signals: list[FeedbackSignal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        keys = signal.pattern_types or ("none",)
        for name in keys:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _flat_map(items: dict[str, int]) -> str:
    if not items:
        return ""
    return ";".join(f"{key}:{items[key]}" for key in sorted(items))
