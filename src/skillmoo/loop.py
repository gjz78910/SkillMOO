from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import time
from typing import Any

from .candidates import (
    SkillCandidate,
    build_initial_population,
    discover_task_skills,
    materialize_candidate_task,
)
from .config import RunConfig
from .feedback import FeedbackSignal, extract_feedback, should_retry
from .io import read_json, write_json
from .operators import OperatorPolicy
from .optimizer import propose_bundle_edit
from .selection import EvaluationRecord, nsga2_survivor_order


def run_task_generations(
    task_id: str,
    cfg: RunConfig,
    rng: random.Random,
    policy: OperatorPolicy,
    output_root: Path,
) -> tuple[list[EvaluationRecord], list[FeedbackSignal]]:
    runner = _build_runner(cfg, output_root)
    task_root = output_root / task_id
    task_root.mkdir(parents=True, exist_ok=True)
    task_base = cfg.skillsbench_root / "tasks" / task_id
    all_skill_ids = discover_task_skills(task_base)
    # population is list of (SkillCandidate, bundle_operation) after gen 0
    init_candidates = _initial_population(all_skill_ids, cfg, rng)
    population: list[tuple[SkillCandidate, str]] = [(c, "") for c in init_candidates]
    records: list[EvaluationRecord] = []
    feedback_signals: list[FeedbackSignal] = []

    for generation_id in range(cfg.num_generations):
        generation_records: list[EvaluationRecord] = []
        generation_family = policy.pick_family(rng)
        for candidate_id, (candidate, bundle_operation) in enumerate(population):
            record, signal = _evaluate_candidate(
                runner=runner,
                cfg=cfg,
                task_id=task_id,
                task_root=task_root,
                generation_id=generation_id,
                candidate_id=candidate_id,
                candidate=candidate,
                family=generation_family,
                bundle_operation=bundle_operation,
            )
            generation_records.append(record)
            records.append(record)
            feedback_signals.append(signal)
        policy.update(generation_family, success=any(r.status == "pass" for r in generation_records))

        write_json(
            task_root / f"generation_{generation_id:02d}.json",
            {
                "generation_id": generation_id,
                "population_size": len(population),
                "operator_family": generation_family,
                "records": [asdict(item) for item in generation_records],
            },
        )
        if generation_id >= cfg.num_generations - 1 or cfg.method != "skillmoo":
            continue
        population = _next_population(
            generation_records=generation_records,
            current_population=[c for c, _ in population],
            all_skill_ids=all_skill_ids,
            target_size=cfg.population_size,
            rng=rng,
            family_hint=generation_family,
        )
    return records, feedback_signals


def _initial_population(
    all_skill_ids: list[str],
    cfg: RunConfig,
    rng: random.Random,
) -> list[SkillCandidate]:
    if cfg.method == "no_skill":
        return [SkillCandidate(selected_skill_ids=tuple())]
    if cfg.method == "original_skills":
        return [SkillCandidate(selected_skill_ids=tuple(sorted(all_skill_ids)))]
    return build_initial_population(all_skill_ids, cfg.population_size, rng)


def classify_quarantine(status: str, retries_used: int, max_retries: int) -> bool:
    return status in {"execution_error", "infra_error", "timeout"} and retries_used >= max_retries


def _load_payload(result: Any) -> dict[str, Any]:
    try:
        payload = read_json(result.result_json_path)
    except Exception:
        payload = asdict(result)
        payload["result_json_path"] = str(result.result_json_path)
        payload["source_result_json_path"] = (
            str(result.source_result_json_path) if result.source_result_json_path else None
        )
        return payload

    # Some SkillsBench payloads do not include top-level status/duration.
    # Backfill from the runner result object to keep exported summaries consistent.
    status = payload.get("status")
    if not isinstance(status, str) or not status.strip():
        fallback = getattr(result, "status", "")
        if isinstance(fallback, str):
            payload["status"] = fallback

    duration = payload.get("duration_sec")
    if duration in (None, 0, 0.0, ""):
        fallback_duration = getattr(result, "duration_sec", None)
        if fallback_duration is not None:
            payload["duration_sec"] = float(fallback_duration)

    for key in ("cost_usd", "test_pass_ratio", "error_class", "timed_out"):
        current = payload.get(key)
        if current is None or current == "":
            fallback_value = getattr(result, key, None)
            if fallback_value is not None:
                payload[key] = fallback_value
    return payload


def _record_payload(record: EvaluationRecord, signal: FeedbackSignal) -> dict[str, Any]:
    payload = asdict(record)
    payload["feedback"] = signal.to_dict()
    return payload


def _evaluate_candidate(
    *,
    runner: Any,
    cfg: RunConfig,
    task_id: str,
    task_root: Path,
    generation_id: int,
    candidate_id: int,
    candidate: SkillCandidate,
    family: str,
    bundle_operation: str = "",
) -> tuple[EvaluationRecord, FeedbackSignal]:
    retries_used = 0
    latest_signal: FeedbackSignal | None = None
    latest_result: Any = None
    while True:
        task_path = materialize_candidate_task(
            base_task_path=cfg.skillsbench_root / "tasks" / task_id,
            candidate=candidate,
            output_root=task_root,
            generation_id=generation_id,
            candidate_id=candidate_id,
        )
        result = runner.run_task(
            task_id=task_id,
            task_path=task_path,
            model_name=cfg.model_name,
            mock=cfg.mock,
            timeout_sec=cfg.timeout_sec,
            retries=0,
            out_dir=task_root / "raw",
            strict_formal=cfg.strict_formal,
            agent_name=cfg.agent_name,
            agent_kwargs={"skillmoo_family": family},
        )
        payload = _load_payload(result)
        signal = extract_feedback(payload)
        latest_signal = signal
        latest_result = result
        if cfg.fail_fast_infra and signal.status == "infra_error":
            message = "Harbor/SkillsBench infrastructure failure"
            exception_info = payload.get("exception_info")
            if isinstance(exception_info, dict):
                raw_message = str(exception_info.get("exception_message") or "")
                if raw_message:
                    message = raw_message
            raise RuntimeError(
                f"{message}\n"
                f"task_id={task_id} generation={generation_id} candidate={candidate_id}\n"
                f"result_json_path={getattr(result, 'result_json_path', '')}"
            )
        if not should_retry(signal, retries_used, cfg.max_retries_per_trial):
            break
        retries_used += 1

    if latest_result is None or latest_signal is None:
        raise RuntimeError("candidate evaluation ended without result")
    quarantined = classify_quarantine(
        latest_signal.status,
        retries_used=retries_used,
        max_retries=cfg.max_retries_per_trial,
    )
    record = EvaluationRecord(
        task_id=task_id,
        generation_id=generation_id,
        candidate_id=candidate_id,
        status=latest_signal.status,
        pass_rate=float(latest_result.test_pass_ratio),
        cost_usd=float(latest_result.cost_usd),
        duration_sec=float(latest_result.duration_sec or 0.0),
        bundle_size=candidate.bundle_size,
        operator_family=family,
        error_class=latest_signal.error_class,
        quarantined=quarantined,
        retries_used=retries_used,
        selected_skill_ids=candidate.selected_skill_ids,
        result_json_path=str(latest_result.result_json_path),
        materialized_task_path=str(task_path),
        method=cfg.method,
        seed=cfg.seed,
        model_name=cfg.model_name,
        timeout_sec=cfg.timeout_sec,
        bundle_operation=bundle_operation,
    )
    write_json(task_root / f"g{generation_id:02d}_c{candidate_id:03d}.json", _record_payload(record, latest_signal))
    return record, latest_signal


def _next_population(
    *,
    generation_records: list[EvaluationRecord],
    current_population: list[SkillCandidate],
    all_skill_ids: list[str],
    target_size: int,
    rng: random.Random,
    family_hint: str,
) -> list[tuple[SkillCandidate, str]]:
    """Return list of (candidate, bundle_operation) for the next generation."""
    survivor_order = nsga2_survivor_order(generation_records)
    survivors = [(current_population[idx], generation_records[idx]) for idx in survivor_order]
    children: list[tuple[SkillCandidate, str]] = []
    for parent, parent_record in survivors:
        evidence = {
            "pass_rate": parent_record.pass_rate,
            "cost_usd": parent_record.cost_usd,
            "status": parent_record.status,
            "failure_summary": parent_record.error_class,
        }
        child, operation = propose_bundle_edit(
            candidate=parent,
            all_skill_ids=all_skill_ids,
            evidence=evidence,
            rng=rng,
            family_hint=family_hint,
        )
        children.append((child, operation))
        if len(children) >= target_size:
            break
    while len(children) < target_size:
        parent, parent_record = survivors[rng.randrange(len(survivors))] if survivors else (current_population[0], generation_records[0])
        evidence = {
            "pass_rate": parent_record.pass_rate,
            "cost_usd": parent_record.cost_usd,
            "status": parent_record.status,
            "failure_summary": parent_record.error_class,
        }
        child, operation = propose_bundle_edit(
            candidate=parent,
            all_skill_ids=all_skill_ids,
            evidence=evidence,
            rng=rng,
            family_hint=family_hint,
        )
        children.append((child, operation))
    return children


def _build_runner(cfg: RunConfig, output_root: Path) -> Any:
    try:
        from skillmoo.skillsbench_runner import SkillsBenchRunner

        return SkillsBenchRunner(cfg.skillsbench_root)
    except Exception as exc:
        if not cfg.mock:
            raise RuntimeError(
                "Failed to import skillmoo.skillsbench_runner. "
                "Use Python >=3.10 or run with --mock."
            ) from exc
        return _MockRunner(output_root)


class _MockRunner:
    def __init__(self, output_root: Path):
        self.output_root = output_root

    def run_task(self, *, task_id: str, out_dir: Path, **_: Any) -> Any:
        elapsed = float(int(time.time()) % 30 + 1)
        pass_ratio = 0.5
        payload = {
            "status": "fail",
            "test_pass_ratio": pass_ratio,
            "cost_usd": 0.01,
            "duration_sec": elapsed,
            "error_class": "none",
            "timed_out": False,
            "verifier_stdout": "FAILED tests/test_outputs.py::test_mock - AssertionError",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / f"{task_id}_mock_result.json"
        write_json(result_path, payload)
        return _MockResult(task_id=task_id, result_json_path=result_path, **payload)


class _MockResult:
    def __init__(
        self,
        *,
        task_id: str,
        result_json_path: Path,
        status: str,
        test_pass_ratio: float,
        cost_usd: float,
        duration_sec: float,
        error_class: str,
        timed_out: bool,
        verifier_stdout: str,
    ):
        self.task_id = task_id
        self.status = status
        self.test_pass_ratio = test_pass_ratio
        self.cost_usd = cost_usd
        self.duration_sec = duration_sec
        self.error_class = error_class
        self.timed_out = timed_out
        self.verifier_stdout = verifier_stdout
        self.result_json_path = result_json_path
        self.source_result_json_path = result_json_path
