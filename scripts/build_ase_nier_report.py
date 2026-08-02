#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_paths(repo_root: Path) -> None:
    path = repo_root / "src"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _default_records_csv(repo_root: Path) -> Path | None:
    path = (repo_root / "reports" / "results_records.csv").resolve()
    return path if path.is_file() else None


def _default_reports_dir(repo_root: Path) -> Path:
    return repo_root / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--input-root", default="experiments/ase_nier_rerun")
    parser.add_argument("--baseline-method", default="original_skills")
    parser.add_argument(
        "--records-csv",
        default="",
        help="Optional per-record CSV (defaults to <input-root>/results_records.csv if present).",
    )
    parser.add_argument(
        "--records-diagnostics-only",
        action="store_true",
        help="Only write ase_nier_records_diagnostics.* from --records-csv; no summary.json required.",
    )
    parser.add_argument(
        "--records-diagnostics-output",
        default="",
        help="Output directory for --records-diagnostics-only (default: reports next to --repo-root).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _add_paths(repo_root)

    from skillmoo.reporting import build_matrix_report, write_records_diagnostics_only

    records_arg = str(args.records_csv or "").strip()
    records_csv: Path | None = Path(records_arg).resolve() if records_arg else None

    if bool(args.records_diagnostics_only):
        rec_path = records_csv if records_csv is not None else _default_records_csv(repo_root)
        if rec_path is None or not rec_path.is_file():
            raise FileNotFoundError("Missing results_records.csv; pass --records-csv explicitly.")
        out_arg = str(args.records_diagnostics_output or "").strip()
        out_dir = Path(out_arg).resolve() if out_arg else _default_reports_dir(repo_root)
        write_records_diagnostics_only(rec_path, out_dir)
        print(f"[ase-nier] records diagnostics -> {out_dir / 'ase_nier_records_diagnostics.csv'}")
        return

    input_root = (repo_root / args.input_root).resolve()
    if records_csv is None:
        records_csv = _default_records_csv(repo_root)
    report = build_matrix_report(input_root, baseline_method=args.baseline_method, records_csv=records_csv)
    print(
        "[ase-nier] report rebuilt: "
        f"tasks={report['task_count']} methods={report['method_count']} "
        f"path={input_root / 'ase_nier_report.csv'}"
    )


if __name__ == "__main__":
    main()
