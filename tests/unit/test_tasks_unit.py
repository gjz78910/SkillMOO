from __future__ import annotations

from pathlib import Path

from skillmoo.config import load_default_se_task_pool
from skillmoo.tasks import resolve_tasks


def test_load_default_se_task_pool_has_16_tasks() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pool = load_default_se_task_pool(repo_root)
    assert len(pool) == 16
    assert "fix-build-agentops" in pool
    assert "spring-boot-jakarta-migration" in pool


def test_resolve_tasks_validates_task_paths(tmp_path: Path) -> None:
    repo_root = tmp_path
    skillsbench_root = repo_root / "skillsbench"
    (skillsbench_root / "tasks" / "fix-build-agentops").mkdir(parents=True)
    manifest = resolve_tasks(repo_root, skillsbench_root, ("fix-build-agentops", "missing-task"))
    assert manifest.task_ids == ("fix-build-agentops", "missing-task")
    assert len(manifest.missing_task_paths) == 1
    assert "missing-task" in manifest.missing_task_paths[0]
