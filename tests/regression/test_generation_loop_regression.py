from __future__ import annotations

from pathlib import Path
import random

from skillmoo.config import RunConfig
from skillmoo.loop import run_task_generations
from skillmoo.operators import OperatorPolicy


def test_generation_loop_outputs_two_generations(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    task_root = repo_root / "skillsbench" / "tasks" / "task-a"
    for sid in ("alpha", "beta", "gamma"):
        skill_dir = task_root / "environment" / "skills" / sid
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"# {sid}\n", encoding="utf-8")

    cfg = RunConfig(
        repo_root=repo_root,
        skillsbench_root=repo_root / "skillsbench",
        output_root=repo_root / "out",
        model_name="glm-5",
        agent_name="claude-code",
        population_size=1,
        num_generations=2,
        max_retries_per_trial=0,
        timeout_sec=60,
        strict_formal=False,
        mock=True,
        task_ids=("task-a",),
    )
    records, _feedback = run_task_generations(
        task_id="task-a",
        cfg=cfg,
        rng=random.Random(42),
        policy=OperatorPolicy(),
        output_root=cfg.output_root,
    )
    assert len(records) == 2
    assert [item.generation_id for item in records] == [0, 1]
    assert (cfg.output_root / "task-a" / "generation_00.json").is_file()
    assert (cfg.output_root / "task-a" / "generation_01.json").is_file()


def test_generation_loop_materializes_method_bundles(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    task_root = repo_root / "skillsbench" / "tasks" / "task-a"
    for sid in ("alpha", "beta"):
        skill_dir = task_root / "environment" / "skills" / sid
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"# {sid}\n", encoding="utf-8")

    for method, expected_size in (("no_skill", 0), ("original_skills", 2)):
        cfg = RunConfig(
            repo_root=repo_root,
            skillsbench_root=repo_root / "skillsbench",
            output_root=repo_root / f"out-{method}",
            model_name="glm-5",
            agent_name="claude-code",
            population_size=1,
            num_generations=1,
            max_retries_per_trial=0,
            timeout_sec=60,
            strict_formal=False,
            mock=True,
            task_ids=("task-a",),
            method=method,
            seed=0,
        )
        records, _feedback = run_task_generations(
            task_id="task-a",
            cfg=cfg,
            rng=random.Random(0),
            policy=OperatorPolicy(),
            output_root=cfg.output_root,
        )
        assert len(records) == 1
        assert records[0].bundle_size == expected_size
        assert records[0].method == method
