from __future__ import annotations

from pathlib import Path

from skillmoo.config import load_default_se_task_pool


def test_se_task_pool_regression_snapshot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected = (
        "citation-check",
        "data-to-d3",
        "dialogue-parser",
        "enterprise-information-search",
        "fix-build-agentops",
        "fix-build-google-auto",
        "flink-query",
        "gh-repo-analytics",
        "jax-computing-basics",
        "parallel-tfidf-search",
        "python-scala-translation",
        "react-performance-debugging",
        "simpo-code-reproduction",
        "spring-boot-jakarta-migration",
        "taxonomy-tree-merge",
        "trend-anomaly-causal-inference",
    )
    assert load_default_se_task_pool(repo_root) == expected


def test_se_task_pool_prefers_repo_root_override(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks_manifest.json"
    manifest.write_text('{"se_task_pool": ["a", "b"]}', encoding="utf-8")
    assert load_default_se_task_pool(tmp_path) == ("a", "b")


def test_se_task_pool_falls_back_to_bundled_manifest(tmp_path: Path) -> None:
    assert load_default_se_task_pool(tmp_path)[0] == "citation-check"
