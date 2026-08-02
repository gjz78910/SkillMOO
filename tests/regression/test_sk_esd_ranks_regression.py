"""Regression test for scripts/compute_sk_esd_ranks.py against Table 2's r_p/r_c columns.

Skipped unless R + rpy2 + ScottKnottESD are installed (not a core skillmoo dependency).
See README.md section 6a for how to set this up.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

pytest.importorskip("rpy2")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from compute_sk_esd_ranks import compute_ranks  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

EXPECTED_RANKS = {
    ("citation-check", "skillmoo"): (1, 1),
    ("citation-check", "original_skills"): (3, 2),
    ("citation-check", "no_skill"): (2, 2),
    ("data-to-d3", "skillmoo"): (1, 1),
    ("data-to-d3", "original_skills"): (3, 2),
    ("data-to-d3", "no_skill"): (2, 2),
    ("dialogue-parser", "skillmoo"): (1, 2),
    ("dialogue-parser", "original_skills"): (1, 3),
    ("dialogue-parser", "no_skill"): (2, 1),
    ("enterprise-information-search", "skillmoo"): (1, 1),
    ("enterprise-information-search", "original_skills"): (3, 1),
    ("enterprise-information-search", "no_skill"): (2, 1),
    ("fix-build-agentops", "skillmoo"): (1, 1),
    ("fix-build-agentops", "original_skills"): (2, 2),
    ("fix-build-agentops", "no_skill"): (3, 2),
    ("fix-build-google-auto", "skillmoo"): (1, 2),
    ("fix-build-google-auto", "original_skills"): (2, 2),
    ("fix-build-google-auto", "no_skill"): (2, 1),
    ("flink-query", "skillmoo"): (1, 2),
    ("flink-query", "original_skills"): (1, 3),
    ("flink-query", "no_skill"): (1, 1),
    ("gh-repo-analytics", "skillmoo"): (2, 1),
    ("gh-repo-analytics", "original_skills"): (2, 2),
    ("gh-repo-analytics", "no_skill"): (1, 2),
    ("jax-computing-basics", "skillmoo"): (1, 2),
    ("jax-computing-basics", "original_skills"): (3, 3),
    ("jax-computing-basics", "no_skill"): (2, 1),
    ("parallel-tfidf-search", "skillmoo"): (1, 2),
    ("parallel-tfidf-search", "original_skills"): (2, 3),
    ("parallel-tfidf-search", "no_skill"): (2, 1),
    ("python-scala-translation", "skillmoo"): (1, 2),
    ("python-scala-translation", "original_skills"): (1, 2),
    ("python-scala-translation", "no_skill"): (2, 1),
    ("react-performance-debugging", "skillmoo"): (1, 2),
    ("react-performance-debugging", "original_skills"): (2, 3),
    ("react-performance-debugging", "no_skill"): (3, 1),
    ("simpo-code-reproduction", "skillmoo"): (1, 2),
    ("simpo-code-reproduction", "original_skills"): (1, 3),
    ("simpo-code-reproduction", "no_skill"): (1, 1),
    ("spring-boot-jakarta-migration", "skillmoo"): (1, 2),
    ("spring-boot-jakarta-migration", "original_skills"): (2, 3),
    ("spring-boot-jakarta-migration", "no_skill"): (3, 1),
    ("taxonomy-tree-merge", "skillmoo"): (1, 2),
    ("taxonomy-tree-merge", "original_skills"): (1, 3),
    ("taxonomy-tree-merge", "no_skill"): (1, 1),
    ("trend-anomaly-causal-inference", "skillmoo"): (1, 2),
    ("trend-anomaly-causal-inference", "original_skills"): (1, 3),
    ("trend-anomaly-causal-inference", "no_skill"): (1, 1),
}


def test_rq1_ranks_match_table2_exactly() -> None:
    ranks = compute_ranks(REPORTS_DIR / "results_records.csv").set_index(["task_id", "method"])
    assert len(ranks) == len(EXPECTED_RANKS)
    for key, (expected_r_p, expected_r_c) in EXPECTED_RANKS.items():
        assert ranks.loc[key, "r_p"] == expected_r_p, key
        assert ranks.loc[key, "r_c"] == expected_r_c, key


def test_skillmoo_achieves_top_pass_rank_on_11_of_12_nonzero_tasks() -> None:
    ranks = compute_ranks(REPORTS_DIR / "results_records.csv").set_index(["task_id", "method"])
    zero_pass_tasks = {"flink-query", "simpo-code-reproduction", "taxonomy-tree-merge", "trend-anomaly-causal-inference"}
    nonzero_tasks = [t for t in EXPECTED_RANKS if t[1] == "skillmoo" and t[0] not in zero_pass_tasks]
    top_rank_count = sum(1 for task_id, method in nonzero_tasks if ranks.loc[(task_id, method), "r_p"] == 1)
    assert top_rank_count == 11
