from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

DOCKERFILE_HARDENING_MARKER = "# skillmoo build hardening"
DOCKERFILE_HARDENING_BLOCK = """\
# skillmoo build hardening
ARG SKILLMOO_APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG SKILLMOO_SECURITY_APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG SKILLMOO_DEBIAN_APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/debian
ARG SKILLMOO_DEBIAN_SECURITY_APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/debian-security
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \\
    PIP_TIMEOUT=120 \\
    PIP_DEFAULT_TIMEOUT=120 \\
    PIP_RETRIES=10 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN set -eux; \\
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources; do \\
        [ -e "$f" ] || continue; \\
        sed -i -E \\
            -e "s#https?://(archive|[a-z][a-z]\\.archive)\\.ubuntu\\.com/ubuntu#${SKILLMOO_APT_MIRROR}#g" \\
            -e "s#https?://security\\.ubuntu\\.com/ubuntu#${SKILLMOO_SECURITY_APT_MIRROR}#g" \\
            -e "s#https?://deb\\.debian\\.org/debian#${SKILLMOO_DEBIAN_APT_MIRROR}#g" \\
            -e "s#https?://security\\.debian\\.org/debian-security#${SKILLMOO_DEBIAN_SECURITY_APT_MIRROR}#g" \\
            "$f"; \\
    done
"""


@dataclass(frozen=True)
class SkillCandidate:
    selected_skill_ids: tuple[str, ...]

    @property
    def bundle_size(self) -> int:
        return len(self.selected_skill_ids)


def discover_task_skills(task_root: Path) -> list[str]:
    skills_dir = task_root / "environment" / "skills"
    if not skills_dir.is_dir():
        return []
    out: list[str] = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if (skill_dir / "SKILL.md").is_file():
            out.append(skill_dir.name)
    return out


def build_initial_population(skill_ids: list[str], population_size: int, rng: random.Random) -> list[SkillCandidate]:
    if population_size <= 0:
        return []
    if not skill_ids:
        return [SkillCandidate(selected_skill_ids=tuple()) for _ in range(population_size)]
    population: list[SkillCandidate] = []
    for _ in range(population_size):
        target_k = rng.randint(1, max(1, min(4, len(skill_ids))))
        chosen = tuple(sorted(rng.sample(skill_ids, k=target_k)))
        population.append(SkillCandidate(selected_skill_ids=chosen))
    return _fill_population(dedupe_population(population, skill_ids), skill_ids, population_size, rng)


def build_random_population(skill_ids: list[str], population_size: int, rng: random.Random) -> list[SkillCandidate]:
    population = build_initial_population(skill_ids, population_size, rng)
    while len(population) < population_size:
        population.extend(build_initial_population(skill_ids, population_size - len(population), rng))
        population = dedupe_population(population, skill_ids)
        if _max_unique_bundle_count(skill_ids) <= len(population):
            break
    return _pad_population_to_size(population[:population_size], population_size, rng)


def mutate_candidate(
    candidate: SkillCandidate,
    all_skill_ids: list[str],
    rng: random.Random,
    family_hint: str,
) -> SkillCandidate:
    selected = list(candidate.selected_skill_ids)
    if not all_skill_ids:
        return candidate
    family = family_hint or "pass"
    if family in {"cost", "length"} and len(selected) > 1:
        removed = rng.choice(selected)
        selected = [item for item in selected if item != removed]
    elif family == "pass":
        pool = [sid for sid in all_skill_ids if sid not in selected]
        if pool:
            selected.append(rng.choice(pool))
        elif selected:
            selected[rng.randrange(len(selected))] = rng.choice(all_skill_ids)
    else:
        if selected:
            selected[rng.randrange(len(selected))] = rng.choice(all_skill_ids)
        else:
            selected = [rng.choice(all_skill_ids)]
    # Heuristic path always normalises to sorted order; reorder edits are only
    # meaningful when the LLM optimizer (optimizer.py) proposes a permutation.
    selected = sorted(set(selected))
    if not selected and all_skill_ids:
        selected = [all_skill_ids[0]]
    return SkillCandidate(selected_skill_ids=tuple(selected))


def materialize_candidate_task(
    base_task_path: Path,
    candidate: SkillCandidate,
    output_root: Path,
    generation_id: int,
    candidate_id: int,
) -> Path:
    task_copy = output_root / "materialized" / f"gen_{generation_id:02d}" / f"cand_{candidate_id:03d}"
    if task_copy.exists():
        shutil.rmtree(task_copy)
    shutil.copytree(base_task_path, task_copy)
    _filter_skills(task_copy / "environment" / "skills", set(candidate.selected_skill_ids))
    _harden_dockerfile_for_network(task_copy / "environment" / "Dockerfile")
    return task_copy


def _harden_dockerfile_for_network(dockerfile: Path) -> None:
    if not dockerfile.is_file():
        return
    text = dockerfile.read_text(encoding="utf-8")
    if DOCKERFILE_HARDENING_MARKER in text:
        return
    lines = text.splitlines()
    patched: list[str] = []
    for line in lines:
        patched.append(line)
        parts = line.strip().split()
        image = parts[1].lower() if len(parts) >= 2 and parts[0].upper() == "FROM" else ""
        if image.startswith("ubuntu:") or image.startswith("python:"):
            patched.append(DOCKERFILE_HARDENING_BLOCK.rstrip())
    dockerfile.write_text("\n".join(patched) + "\n", encoding="utf-8")


def dedupe_population(population: list[SkillCandidate], all_skill_ids: list[str]) -> list[SkillCandidate]:
    seen: set[tuple[str, ...]] = set()
    out: list[SkillCandidate] = []
    for item in population:
        key = tuple(sorted(item.selected_skill_ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(SkillCandidate(selected_skill_ids=key))
    if out:
        return out
    if all_skill_ids:
        return [SkillCandidate(selected_skill_ids=(all_skill_ids[0],))]
    return [SkillCandidate(selected_skill_ids=tuple())]


def _max_unique_bundle_count(skill_ids: list[str]) -> int:
    if not skill_ids:
        return 1
    capped = min(len(skill_ids), 4)
    total = 0
    for size in range(1, capped + 1):
        total += _n_choose_k(len(skill_ids), size)
    return total


def _fill_population(
    population: list[SkillCandidate],
    all_skill_ids: list[str],
    population_size: int,
    rng: random.Random,
) -> list[SkillCandidate]:
    max_unique = _max_unique_bundle_count(all_skill_ids)
    out = list(population)
    attempts = 0
    while len(out) < population_size and len(out) < max_unique and attempts < 100:
        attempts += 1
        if not all_skill_ids:
            candidate = SkillCandidate(selected_skill_ids=tuple())
        else:
            target_k = rng.randint(1, max(1, min(4, len(all_skill_ids))))
            candidate = SkillCandidate(selected_skill_ids=tuple(sorted(rng.sample(all_skill_ids, k=target_k))))
        out = dedupe_population(out + [candidate], all_skill_ids)
    out = out[:population_size]
    return _pad_population_to_size(out, population_size, rng)


def _pad_population_to_size(
    population: list[SkillCandidate], population_size: int, rng: random.Random
) -> list[SkillCandidate]:
    if population_size <= 0:
        return []
    seq = list(population[:population_size])
    if not seq:
        return []
    while len(seq) < population_size:
        seq.append(seq[rng.randrange(len(seq))])
    return seq


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    out = 1
    for i in range(1, k + 1):
        out = out * (n - i + 1) // i
    return out


def _filter_skills(skills_root: Path, keep_ids: set[str]) -> None:
    if not skills_root.is_dir():
        return
    for skill_dir in skills_root.iterdir():
        if not skill_dir.is_dir():
            continue
        if skill_dir.name not in keep_ids:
            shutil.rmtree(skill_dir, ignore_errors=True)
