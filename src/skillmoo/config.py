from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

DEFAULT_METHOD = "skillmoo"
SEARCH_METHODS = ("skillmoo",)
STATIC_METHODS = ("no_skill", "original_skills")
METHODS = STATIC_METHODS + SEARCH_METHODS
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class RunConfig:
    repo_root: Path
    skillsbench_root: Path
    output_root: Path
    model_name: str
    agent_name: str
    population_size: int
    num_generations: int
    max_retries_per_trial: int
    timeout_sec: int
    strict_formal: bool
    mock: bool
    fail_fast_infra: bool = True
    task_ids: tuple[str, ...] | None = None
    method: str = DEFAULT_METHOD
    seed: int = 42

    @classmethod
    def from_args(cls, args: Any) -> "RunConfig":
        repo_root = Path(args.repo_root).resolve()
        skillsbench_root = (repo_root / "skillsbench").resolve()
        output_root = Path(args.output_root).resolve()
        return cls(
            repo_root=repo_root,
            skillsbench_root=skillsbench_root,
            output_root=output_root,
            model_name=str(args.model_name),
            agent_name=str(args.agent_name),
            population_size=max(1, int(args.population_size)),
            num_generations=max(1, int(args.num_generations)),
            max_retries_per_trial=max(0, int(args.max_retries_per_trial)),
            timeout_sec=max(1, int(args.timeout_sec)),
            strict_formal=bool(args.strict_formal),
            mock=bool(args.mock),
            fail_fast_infra=not bool(getattr(args, "keep_going_on_infra_error", False)),
            task_ids=_parse_task_ids(getattr(args, "task_ids", "")),
            method=_validate_method(str(getattr(args, "method", DEFAULT_METHOD))),
            seed=int(getattr(args, "seed", 42)),
        )


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TASKS_MANIFEST = _PACKAGE_ROOT / "tasks_manifest.json"


def load_default_se_task_pool(repo_root: Path) -> tuple[str, ...]:
    manifest = _find_tasks_manifest(repo_root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw_pool = payload.get("se_task_pool", [])
    return tuple(str(item) for item in raw_pool if str(item).strip())


def _parse_task_ids(raw: str) -> tuple[str, ...] | None:
    items = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not items:
        return None
    return tuple(items)


def _find_tasks_manifest(repo_root: Path) -> Path:
    candidates = (repo_root / "tasks_manifest.json", _DEFAULT_TASKS_MANIFEST)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find tasks_manifest.json in repo root ({repo_root}) or package root ({_PACKAGE_ROOT})"
    )


def _validate_method(method: str) -> str:
    if method not in METHODS:
        allowed = ", ".join(METHODS)
        raise ValueError(f"Unknown method '{method}'. Expected one of: {allowed}")
    return method
