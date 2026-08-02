from __future__ import annotations

import importlib
from pathlib import Path


def test_full_rerun_executor_and_overlay_are_packaged() -> None:
    module = importlib.import_module("skillmoo.skillsbench_runner")
    assert hasattr(module, "SkillsBenchRunner")
    archive = Path(__file__).resolve().parents[2] / "patches" / "skillsbench-ase-nier-overlay.tar.gz"
    assert archive.is_file()
