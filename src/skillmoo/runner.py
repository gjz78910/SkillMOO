from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Any

from .config import RunConfig
from .io import read_json, write_csv, write_json
from .loop import run_task_generations
from .metrics import TaskSummary, aggregate_experiment, summarize_task, summary_rows_for_csv
from .operators import OperatorPolicy
from .tasks import resolve_tasks, write_task_manifest


def run_experiment(cfg: RunConfig) -> dict[str, Any]:
    manifest = resolve_tasks(cfg.repo_root, cfg.skillsbench_root, cfg.task_ids)
    if not manifest.is_valid:
        raise FileNotFoundError("Missing task directories:\n" + "\n".join(manifest.missing_task_paths))

    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[skillmoo] output root: {root}")
    print(f"[skillmoo] tasks: {len(manifest.task_ids)}")
    print(
        "[skillmoo] "
        f"population_size={cfg.population_size} num_generations={cfg.num_generations}"
    )
    write_task_manifest(root / "task_manifest.json", manifest.task_ids)
    print(f"[skillmoo] wrote task manifest: {root / 'task_manifest.json'}")

    summaries: list[TaskSummary] = []
    total = len(manifest.task_ids)
    for index, task_id in enumerate(manifest.task_ids, start=1):
        task_summary_path = root / task_id / "summary.json"
        if task_summary_path.is_file():
            summary = _load_task_summary(task_summary_path)
            summaries.append(summary)
            print(
                "[skillmoo] "
                f"[{index}/{total}] skip existing: {task_id} "
                f"(n={summary.n_evaluations}, pass_mean={summary.pass_rate_mean:.4f}, cost_mean={summary.cost_usd_mean:.4f})"
            )
            continue
        print(f"[skillmoo] [{index}/{total}] running task: {task_id}")
        policy = OperatorPolicy()
        records, feedback = run_task_generations(task_id, cfg, _task_rng(cfg.seed, task_id), policy, root)
        summary = summarize_task(task_id, records, feedback)
        summaries.append(summary)
        write_json(task_summary_path, summary.to_dict())
        write_json(root / task_id / "operator_policy.json", policy.as_dict())
        print(
            "[skillmoo] "
            f"[{index}/{total}] done: {task_id} "
            f"(n={summary.n_evaluations}, pass_mean={summary.pass_rate_mean:.4f}, cost_mean={summary.cost_usd_mean:.4f})"
        )

    aggregate = aggregate_experiment(summaries)
    write_json(root / "summary.json", aggregate)
    rows = summary_rows_for_csv(summaries)
    write_csv(
        root / "summary.csv",
        [
            "task_id",
            "n_evaluations",
            "pass_rate_mean",
            "cost_usd_mean",
            "duration_sec_mean",
            "best_pass_rate",
            "best_cost_usd",
            "best_duration_sec",
            "best_bundle_size",
            "best_selected_skill_ids",
            "status_counts",
            "failure_type_counts",
        ],
        rows,
    )
    print(f"[skillmoo] wrote summary: {root / 'summary.json'}")
    print(f"[skillmoo] wrote csv: {root / 'summary.csv'}")
    print("[skillmoo] finished.")
    return aggregate


def _task_rng(seed: int, task_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _load_task_summary(path: Path) -> TaskSummary:
    payload = read_json(path)
    return TaskSummary(
        task_id=str(payload["task_id"]),
        n_evaluations=int(payload["n_evaluations"]),
        pass_rate_mean=float(payload["pass_rate_mean"]),
        cost_usd_mean=float(payload["cost_usd_mean"]),
        duration_sec_mean=float(payload["duration_sec_mean"]),
        best_pass_rate=float(payload["best_pass_rate"]),
        best_cost_usd=float(payload["best_cost_usd"]),
        best_duration_sec=float(payload["best_duration_sec"]),
        best_bundle_size=int(payload["best_bundle_size"]),
        best_selected_skill_ids=tuple(payload.get("best_selected_skill_ids") or ()),
        status_counts={str(key): int(value) for key, value in dict(payload.get("status_counts") or {}).items()},
        failure_type_counts={
            str(key): int(value) for key, value in dict(payload.get("failure_type_counts") or {}).items()
        },
    )
