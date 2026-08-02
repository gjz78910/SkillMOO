"""Guards that `reports/*.csv` still reproduce the exact numbers printed in the paper's
Tables 2-4. If these fail after regenerating reports, the paper tables must be
re-typeset from the new data before submission.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / name)


def test_rq1_summary_has_all_tasks_and_methods() -> None:
    df = _read("results_summary.csv")
    assert df["task_id"].nunique() == 16
    assert set(df["method"]) == {"no_skill", "original_skills", "skillmoo"}
    assert len(df) == 16 * 3


def test_rq1_avg_pass_and_cost_matches_table2() -> None:
    df = _read("results_summary.csv")
    avg = df.groupby("method")[["pass_mean", "cost_mean"]].mean().round(2)
    assert avg.loc["skillmoo", "pass_mean"] == 0.39
    assert avg.loc["skillmoo", "cost_mean"] == 0.71
    assert avg.loc["original_skills", "pass_mean"] == 0.14
    assert avg.loc["original_skills", "cost_mean"] == 0.89
    assert avg.loc["no_skill", "pass_mean"] == 0.21
    assert avg.loc["no_skill", "cost_mean"] == 0.64


def test_rq2_hv_matches_table3_exactly() -> None:
    df = _read("rq2_hv.csv").set_index("task_id")
    expected = {
        "citation-check": (4.1539, 0.0016, 0.0798, 5031),
        "data-to-d3": (4.1437, 0.0015, 0.0723, 4580),
        "dialogue-parser": (3.3556, 0.0017, 0.0489, 2828),
        "enterprise-information-search": (4.2755, 0.0020, 0.0612, 2892),
        "fix-build-agentops": (2.2676, 0.0081, 0.1783, 2110),
        "fix-build-google-auto": (13.7338, 0.0009, 0.0183, 1845),
        "gh-repo-analytics": (1.9664, 0.0012, 0.0044, 263),
        "jax-computing-basics": (2.8053, 0.0016, 0.0319, 1903),
        "parallel-tfidf-search": (9.6070, 0.0053, 0.0480, 807),
        "python-scala-translation": (1.8611, 0.0056, 0.0296, 430),
        "react-performance-debugging": (9.9575, 0.0055, 0.1342, 2322),
        "spring-boot-jakarta-migration": (1.7641, 0.0078, 0.0311, 301),
    }
    assert set(df.index) == set(expected)
    for task_id, (opt_cost, hv_ori, hv_skillmoo, delta_pct) in expected.items():
        row = df.loc[task_id]
        assert row["opt_cost_usd"] == opt_cost
        assert row["hv_ori_skill"] == hv_ori
        assert row["hv_skillmoo"] == hv_skillmoo
        assert round(row["delta_hv_pct"]) == delta_pct


def test_rq2_breakeven_reuse_count_matches_section3() -> None:
    summary = _read("results_summary.csv")
    hv = _read("rq2_hv.csv")
    skillmoo = summary[summary.method == "skillmoo"][["task_id", "cost_mean"]].rename(
        columns={"cost_mean": "cost_skillmoo"}
    )
    original = summary[summary.method == "original_skills"][["task_id", "cost_mean"]].rename(
        columns={"cost_mean": "cost_ori"}
    )
    merged = hv.merge(skillmoo, on="task_id").merge(original, on="task_id")
    merged["per_run_saving"] = merged["cost_ori"] - merged["cost_skillmoo"]
    merged["breakeven_reuses"] = (merged["opt_cost_usd"] / merged["per_run_saving"]).apply(math.ceil)
    assert merged["breakeven_reuses"].min() == 5
    assert merged["breakeven_reuses"].max() == 682
    assert merged["breakeven_reuses"].median() == 45


# Multi-skill ("larger skill bundle") tasks, per the pool sizes in README.md section 6.
# The abstract/RQ1-answer claim "pass rate gains up to X percentage points" is scoped to
# this subset, not all 16 tasks (several single-skill tasks show larger raw pp gains).
_MULTI_SKILL_TASKS = {
    "fix-build-agentops",
    "fix-build-google-auto",
    "flink-query",
    "parallel-tfidf-search",
    "python-scala-translation",
    "react-performance-debugging",
    "simpo-code-reproduction",
    "spring-boot-jakarta-migration",
    "trend-anomaly-causal-inference",
}


def test_max_pass_rate_gain_matches_abstract_and_rq1_answer() -> None:
    df = _read("results_summary.csv")
    piv = df.pivot(index="task_id", columns="method", values="pass_mean").loc[list(_MULTI_SKILL_TASKS)]
    gain_pp = (piv["skillmoo"] - piv["original_skills"]) * 100
    assert round(gain_pp.max(), 1) == 42.0
    assert gain_pp.idxmax() == "fix-build-google-auto"


def test_max_cost_reduction_matches_abstract_and_rq1_answer() -> None:
    df = _read("results_summary.csv")
    piv = df.pivot(index="task_id", columns="method", values="cost_mean").loc[list(_MULTI_SKILL_TASKS)]
    cost_reduction_pct = (piv["original_skills"] - piv["skillmoo"]) / piv["original_skills"] * 100
    assert round(cost_reduction_pct.max(), 1) == 32.1
    assert cost_reduction_pct.idxmax() == "fix-build-agentops"


def test_rq3_operation_totals_match_table4() -> None:
    df = _read("rq3_summary.csv")
    totals = df.groupby("operation")["n_edits"].sum().to_dict()
    assert totals == {
        "Add Skills": 9,
        "Edit Skill Content": 4,
        "Remove Skills": 16,
        "Reorder Bundle": 2,
        "Replace Skills": 7,
    }
    assert df["n_edits"].sum() == 38
