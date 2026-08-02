from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EvaluationRecord:
    task_id: str
    generation_id: int
    candidate_id: int
    status: str
    pass_rate: float
    cost_usd: float
    duration_sec: float
    bundle_size: int
    operator_family: str
    error_class: str
    quarantined: bool
    retries_used: int
    selected_skill_ids: tuple[str, ...] = tuple()
    result_json_path: str = ""
    materialized_task_path: str = ""
    method: str = ""
    seed: int = 0
    model_name: str = ""
    timeout_sec: int = 0
    bundle_operation: str = ""


def best_record(records: list[EvaluationRecord]) -> EvaluationRecord | None:
    if not records:
        return None
    return sorted(
        records,
        key=lambda item: (
            -item.pass_rate,
            item.cost_usd,
            item.bundle_size,
            item.duration_sec,
            item.status != "pass",
        ),
    )[0]


def status_counts(records: list[EvaluationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in records:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def nsga2_survivor_order(records: list[EvaluationRecord]) -> list[int]:
    if not records:
        return []

    fronts = _non_dominated_fronts(records)
    ordered: list[int] = []
    for front in fronts:
        distance = _crowding_distance(records, front)
        ordered.extend(
            sorted(
                front,
                key=lambda idx: (
                    -distance[idx],
                    -records[idx].pass_rate,
                    records[idx].cost_usd,
                    records[idx].bundle_size,
                    records[idx].duration_sec,
                ),
            )
        )
    return ordered


def _non_dominated_fronts(records: list[EvaluationRecord]) -> list[list[int]]:
    dominates: dict[int, list[int]] = {idx: [] for idx in range(len(records))}
    dominated_by_count: dict[int, int] = {idx: 0 for idx in range(len(records))}
    first_front: list[int] = []

    for left in range(len(records)):
        for right in range(len(records)):
            if left == right:
                continue
            if _dominates(records[left], records[right]):
                dominates[left].append(right)
            elif _dominates(records[right], records[left]):
                dominated_by_count[left] += 1
        if dominated_by_count[left] == 0:
            first_front.append(left)

    fronts: list[list[int]] = []
    current = first_front
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for left in current:
            for right in dominates[left]:
                dominated_by_count[right] -= 1
                if dominated_by_count[right] == 0:
                    next_front.append(right)
        current = next_front
    return fronts


def _dominates(left: EvaluationRecord, right: EvaluationRecord) -> bool:
    pass_at_least = left.pass_rate >= right.pass_rate
    cost_at_most = left.cost_usd <= right.cost_usd
    strictly_better = left.pass_rate > right.pass_rate or left.cost_usd < right.cost_usd
    return pass_at_least and cost_at_most and strictly_better


def _crowding_distance(records: list[EvaluationRecord], front: list[int]) -> dict[int, float]:
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = math.inf
        return distance

    objectives = (
        lambda record: record.pass_rate,
        lambda record: -record.cost_usd,
    )
    for objective in objectives:
        ordered = sorted(front, key=lambda idx: objective(records[idx]))
        distance[ordered[0]] = math.inf
        distance[ordered[-1]] = math.inf
        min_value = objective(records[ordered[0]])
        max_value = objective(records[ordered[-1]])
        if max_value == min_value:
            continue
        for pos in range(1, len(ordered) - 1):
            if math.isinf(distance[ordered[pos]]):
                continue
            prev_value = objective(records[ordered[pos - 1]])
            next_value = objective(records[ordered[pos + 1]])
            distance[ordered[pos]] += (next_value - prev_value) / (max_value - min_value)
    return distance
