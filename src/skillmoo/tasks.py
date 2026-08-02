from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import load_default_se_task_pool
from .io import now_iso, write_json


@dataclass(frozen=True)
class TaskManifest:
    task_ids: tuple[str, ...]
    missing_task_paths: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_task_paths


def resolve_tasks(repo_root: Path, skillsbench_root: Path, override: tuple[str, ...] | None) -> TaskManifest:
    task_ids = override if override else load_default_se_task_pool(repo_root)
    missing: list[str] = []
    for task_id in task_ids:
        task_dir = skillsbench_root / "tasks" / task_id
        if not task_dir.is_dir():
            missing.append(str(task_dir))
    return TaskManifest(task_ids=tuple(task_ids), missing_task_paths=tuple(missing))


def write_task_manifest(path: Path, task_ids: tuple[str, ...]) -> None:
    payload = {
        "generated_at": now_iso(),
        "task_count": len(task_ids),
        "task_ids": list(task_ids),
    }
    write_json(path, payload)
