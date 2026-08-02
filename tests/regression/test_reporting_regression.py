from __future__ import annotations

from pathlib import Path

from skillmoo.io import write_json
from skillmoo.reporting import build_matrix_report


def _write_summary(root: Path, method: str, seed: int, rows: list[tuple[str, float]]) -> None:
    write_json(
        root / method / f"seed_{seed}" / "summary.json",
        {
            "rows": [
                {
                    "task_id": task_id,
                    "n_evaluations": 1,
                    "pass_rate_mean": pass_rate,
                    "best_pass_rate": pass_rate,
                    "cost_usd_mean": 1.0,
                    "best_cost_usd": 1.0,
                    "duration_sec_mean": 10.0,
                    "best_duration_sec": 10.0,
                    "best_bundle_size": 2,
                }
                for task_id, pass_rate in rows
            ]
        },
    )


def test_reporting_absolute_and_relative_deltas(tmp_path: Path) -> None:
    _write_summary(tmp_path, "original_skills", 0, [("low-base", 0.16), ("high-base", 0.97)])
    _write_summary(tmp_path, "skillmoo", 0, [("low-base", 0.37), ("high-base", 0.99)])

    report = build_matrix_report(tmp_path)
    by_key = {(row["task_id"], row["method"]): row for row in report["rows"]}

    low = by_key[("low-base", "skillmoo")]
    assert low["absolute_delta_vs_original"] == 0.21
    assert low["relative_delta_pct_vs_original"] == 131.25

    high = by_key[("high-base", "skillmoo")]
    assert high["absolute_delta_vs_original"] == 0.02
    assert high["relative_delta_pct_vs_original"] == 2.061856

    assert (tmp_path / "ase_nier_report.csv").is_file()
    assert (tmp_path / "ase_nier_overall.csv").is_file()
