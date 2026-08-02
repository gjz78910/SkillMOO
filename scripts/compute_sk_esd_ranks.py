#!/usr/bin/env python3
"""Compute per-task Scott-Knott ESD ranks (r_p on pass_rate, r_c on cost_usd) across the
three methods (no_skill, original_skills, skillmoo) from the 10 independent per-seed runs
in reports/results_records.csv, using the official CRAN ScottKnottESD package. Reproduces
the r_p/r_c columns in the paper's Table 2.

Requires R + the ScottKnottESD package + rpy2 (not part of the core skillmoo package
dependencies, since this is a one-off statistical post-processing step). See README.md
section 6a for install instructions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

_SK_PKG = importr("ScottKnottESD")
_METHODS = ("no_skill", "original_skills", "skillmoo")


def _sk_esd_ranks(values_by_method: dict[str, list[float]]) -> dict[str, int]:
    """Return {method: rank} where rank 1 is the best-performing SK-ESD group."""
    pooled = [v for vals in values_by_method.values() for v in vals]
    if len(set(pooled)) <= 1:
        # ScottKnottESD's ANOVA step is undefined with zero variance across all groups
        # (e.g. every run scores pass_rate=0.0); all methods tie for rank 1.
        return {method: 1 for method in values_by_method}
    safe = {f"M{i}": name for i, name in enumerate(values_by_method)}
    r_df = ro.DataFrame({sk: FloatVector(values_by_method[name]) for sk, name in safe.items()})
    try:
        r_sk = _SK_PKG.sk_esd(r_df, version="np")
    except RRuntimeError:
        # ScottKnottESD's ANOVA step also fails (division by zero) when some, but not
        # all, groups have zero within-group variance (e.g. two methods both score a
        # constant 0.0 while a third varies). Fall back to grouping methods with an
        # identical run-by-run distribution, then ranking the distinct groups by mean.
        return _degenerate_group_ranks(values_by_method)
    raw_groups = [int(x) for x in r_sk.rx2("groups")]
    nms = list(r_sk.rx2("nms"))
    method_by_col = {sk: name for sk, name in safe.items()}
    raw_group_of_method = {method_by_col[sk]: g for sk, g in zip(nms, raw_groups)}
    # R's own group numbering order is not guaranteed to run best-to-worst; remap so
    # rank 1 always corresponds to the group with the highest mean value.
    method_mean = {m: sum(vals) / len(vals) for m, vals in values_by_method.items()}
    groups_present = set(raw_group_of_method.values())
    group_mean = {
        g: sum(method_mean[m] for m in values_by_method if raw_group_of_method[m] == g)
        / sum(1 for m in values_by_method if raw_group_of_method[m] == g)
        for g in groups_present
    }
    rank_of_group = {
        g: rank for rank, g in enumerate(sorted(group_mean, key=lambda g: -group_mean[g]), start=1)
    }
    return {method: rank_of_group[raw_group_of_method[method]] for method in values_by_method}


def _degenerate_group_ranks(values_by_method: dict[str, list[float]]) -> dict[str, int]:
    by_distribution: dict[tuple[float, ...], list[str]] = {}
    for method, values in values_by_method.items():
        key = tuple(sorted(values))
        by_distribution.setdefault(key, []).append(method)
    ordered = sorted(by_distribution.items(), key=lambda kv: -(sum(kv[0]) / len(kv[0])))
    ranks: dict[str, int] = {}
    for rank, (_, members) in enumerate(ordered, start=1):
        for method in members:
            ranks[method] = rank
    return ranks


def compute_ranks(records_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(records_csv)
    rows = []
    for task_id, task_df in df.groupby("task_id"):
        pass_by_method = {m: task_df.loc[task_df.method == m, "pass_rate"].tolist() for m in _METHODS}
        cost_by_method = {m: task_df.loc[task_df.method == m, "cost_usd"].tolist() for m in _METHODS}
        # Higher pass_rate is better -> rank directly; lower cost is better -> rank negated cost.
        r_p = _sk_esd_ranks(pass_by_method)
        r_c = _sk_esd_ranks({m: [-v for v in vals] for m, vals in cost_by_method.items()})
        for method in _METHODS:
            rows.append({"task_id": task_id, "method": method, "r_p": r_p[method], "r_c": r_c[method]})
    return pd.DataFrame(rows).sort_values(["task_id", "method"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-csv",
        default=str(Path(__file__).resolve().parents[1] / "reports" / "results_records.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parents[1] / "reports" / "rq1_sk_esd_ranks.csv"),
    )
    args = parser.parse_args()

    ranks = compute_ranks(Path(args.records_csv))
    ranks.to_csv(args.output_csv, index=False)
    print(ranks.to_string(index=False))
    print(f"\n[sk-esd] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
