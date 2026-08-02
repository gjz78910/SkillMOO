from __future__ import annotations

from skillmoo.selection import EvaluationRecord, nsga2_survivor_order


def _record(pass_rate: float, cost_usd: float, candidate_id: int) -> EvaluationRecord:
    return EvaluationRecord(
        task_id="task-a",
        generation_id=0,
        candidate_id=candidate_id,
        status="fail",
        pass_rate=pass_rate,
        cost_usd=cost_usd,
        duration_sec=10.0,
        bundle_size=1,
        operator_family="pass",
        error_class="none",
        quarantined=False,
        retries_used=0,
    )


def test_nsga2_survivor_order_prefers_non_dominated_records() -> None:
    records = [
        _record(0.8, 0.8, 0),
        _record(0.7, 0.2, 1),
        _record(0.5, 0.7, 2),
    ]

    ordered = nsga2_survivor_order(records)

    assert set(ordered[:2]) == {0, 1}
    assert ordered[-1] == 2


def test_nsga2_survivor_order_uses_crowding_inside_front() -> None:
    records = [
        _record(0.9, 0.9, 0),
        _record(0.7, 0.5, 1),
        _record(0.5, 0.1, 2),
    ]

    ordered = nsga2_survivor_order(records)

    assert set(ordered[:2]) == {0, 2}
