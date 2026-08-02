from __future__ import annotations

from pathlib import Path
import random

from skillmoo.candidates import (
    SkillCandidate,
    build_initial_population,
    build_random_population,
    materialize_candidate_task,
)


def test_build_initial_population_respects_size() -> None:
    population = build_initial_population(
        skill_ids=["a", "b", "c"],
        population_size=3,
        rng=random.Random(7),
    )
    assert len(population) == 3
    assert all(isinstance(item, SkillCandidate) for item in population)
    assert all(item.bundle_size >= 1 for item in population)


def test_build_initial_population_pads_when_few_unique_bundles() -> None:
    population = build_initial_population(
        skill_ids=["solo"],
        population_size=4,
        rng=random.Random(0),
    )
    assert len(population) == 4
    assert all(item.selected_skill_ids == ("solo",) for item in population)


def test_build_initial_population_two_skills_fills_target_size() -> None:
    population = build_initial_population(
        skill_ids=["x", "y"],
        population_size=4,
        rng=random.Random(1),
    )
    assert len(population) == 4


def test_materialize_candidate_task_filters_skills(tmp_path: Path) -> None:
    base = tmp_path / "skillsbench" / "tasks" / "task-a"
    for sid in ("s1", "s2", "s3"):
        skill_dir = base / "environment" / "skills" / sid
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"# {sid}\n", encoding="utf-8")

    output_root = tmp_path / "out"
    candidate = SkillCandidate(selected_skill_ids=("s1", "s3"))
    materialized = materialize_candidate_task(
        base_task_path=base,
        candidate=candidate,
        output_root=output_root,
        generation_id=0,
        candidate_id=0,
    )
    kept = sorted(p.name for p in (materialized / "environment" / "skills").iterdir() if p.is_dir())
    assert kept == ["s1", "s3"]


def test_materialize_candidate_task_supports_empty_and_full_bundles(tmp_path: Path) -> None:
    base = tmp_path / "skillsbench" / "tasks" / "task-a"
    for sid in ("s1", "s2"):
        skill_dir = base / "environment" / "skills" / sid
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"# {sid}\n", encoding="utf-8")

    empty = materialize_candidate_task(
        base_task_path=base,
        candidate=SkillCandidate(selected_skill_ids=tuple()),
        output_root=tmp_path / "empty",
        generation_id=0,
        candidate_id=0,
    )
    assert list((empty / "environment" / "skills").iterdir()) == []

    full = materialize_candidate_task(
        base_task_path=base,
        candidate=SkillCandidate(selected_skill_ids=("s1", "s2")),
        output_root=tmp_path / "full",
        generation_id=0,
        candidate_id=0,
    )
    kept = sorted(p.name for p in (full / "environment" / "skills").iterdir() if p.is_dir())
    assert kept == ["s1", "s2"]


def test_materialize_candidate_task_hardens_ubuntu_dockerfile(tmp_path: Path) -> None:
    base = tmp_path / "skillsbench" / "tasks" / "task-a"
    env_dir = base / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN apt-get update\n",
        encoding="utf-8",
    )

    materialized = materialize_candidate_task(
        base_task_path=base,
        candidate=SkillCandidate(selected_skill_ids=tuple()),
        output_root=tmp_path / "out",
        generation_id=0,
        candidate_id=0,
    )
    dockerfile = (materialized / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "skillmoo build hardening" in dockerfile
    assert "mirrors.tuna.tsinghua.edu.cn/ubuntu" in dockerfile
    assert "mirrors.tuna.tsinghua.edu.cn/debian" in dockerfile
    assert "PIP_TIMEOUT=120" in dockerfile
    assert "PIP_DEFAULT_TIMEOUT=120" in dockerfile


def test_materialize_candidate_task_hardens_python_slim_dockerfile(tmp_path: Path) -> None:
    base = tmp_path / "skillsbench" / "tasks" / "task-a"
    env_dir = base / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\nRUN apt-get update\n",
        encoding="utf-8",
    )

    materialized = materialize_candidate_task(
        base_task_path=base,
        candidate=SkillCandidate(selected_skill_ids=tuple()),
        output_root=tmp_path / "out",
        generation_id=0,
        candidate_id=0,
    )
    dockerfile = (materialized / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "skillmoo build hardening" in dockerfile
    assert "mirrors.tuna.tsinghua.edu.cn/debian" in dockerfile


def test_random_population_is_deterministic_for_fixed_seed() -> None:
    first = build_random_population(["a", "b", "c", "d"], 4, random.Random(5))
    second = build_random_population(["a", "b", "c", "d"], 4, random.Random(5))
    assert first == second
    assert len(first) == 4
