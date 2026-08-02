#!/usr/bin/env python3
"""Prepare the exact SkillsBench source and task overlay used by the ASE NIER rerun."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tarfile


SKILLSBENCH_URL = "https://github.com/benchflow-ai/skillsbench.git"
SKILLSBENCH_COMMIT = "593b0c6a3d95e0d4acc813788b12b6c044560b43"


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _extract_overlay(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.name}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="skillsbench")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    destination = (repo_root / args.destination).resolve()
    overlay = repo_root / "patches" / "skillsbench-ase-nier-overlay.tar.gz"
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if not overlay.is_file():
        raise FileNotFoundError(f"Missing task overlay: {overlay}")

    _run("git", "clone", SKILLSBENCH_URL, str(destination))
    _run("git", "checkout", "--detach", SKILLSBENCH_COMMIT, cwd=destination)
    _extract_overlay(overlay, destination)
    _run("git", "diff", "--check", cwd=destination)
    print(f"SkillsBench prepared at {destination}")
    print(f"Base commit: {SKILLSBENCH_COMMIT}")


if __name__ == "__main__":
    main()
