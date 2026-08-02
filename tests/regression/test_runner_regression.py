from __future__ import annotations

import csv
import json
from pathlib import Path

from skillmoo.config import RunConfig
from skillmoo.runner import run_experiment


def _make_task(skillsbench_root: Path, task_id: str, skill_ids: list[str]) -> None:
    for sid in skill_ids:
        skill_dir = skillsbench_root / "tasks" / task_id / "environment" / "skills" / sid
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"# {sid}\n", encoding="utf-8")


def test_run_experiment_produces_expected_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skillsbench_root = repo_root / "skillsbench"
    output_root = repo_root / "out"

    task_ids = ("alpha-task", "beta-task")
    _make_task(skillsbench_root, "alpha-task", ["skill-x", "skill-y"])
    _make_task(skillsbench_root, "beta-task", ["skill-p"])

    cfg = RunConfig(
        repo_root=repo_root,
        skillsbench_root=skillsbench_root,
        output_root=output_root,
        model_name="glm-5",
        agent_name="claude-code",
        population_size=1,
        num_generations=1,
        max_retries_per_trial=0,
        timeout_sec=60,
        strict_formal=False,
        mock=True,
        task_ids=task_ids,
    )

    result = run_experiment(cfg)

    # Top-level summary.json has expected shape
    assert result["task_count"] == 2
    assert "overall" in result
    assert "pass_rate_mean" in result["overall"]

    # summary.json on disk matches returned value
    summary_path = output_root / "summary.json"
    assert summary_path.is_file()
    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk["task_count"] == 2

    # summary.csv has one row per task
    csv_path = output_root / "summary.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert {row["task_id"] for row in rows} == set(task_ids)

    # task manifest exists
    assert (output_root / "task_manifest.json").is_file()

    # per-task artifacts exist
    for task_id in task_ids:
        assert (output_root / task_id / "summary.json").is_file()
        assert (output_root / task_id / "operator_policy.json").is_file()
        assert (output_root / task_id / "generation_00.json").is_file()
