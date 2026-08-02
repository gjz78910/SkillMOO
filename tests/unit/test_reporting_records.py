from __future__ import annotations

from pathlib import Path

from skillmoo.reporting import row_eligible_for_rq1, summarize_records_diagnostics


def test_row_eligible_for_rq1_pass() -> None:
    assert row_eligible_for_rq1("pass", "none") is True


def test_row_eligible_for_rq1_infra_status() -> None:
    assert row_eligible_for_rq1("infra_error", "infra_error") is False
    assert row_eligible_for_rq1("timeout", "agent_timeout") is False


def test_row_eligible_for_rq1_infra_failure_type() -> None:
    assert row_eligible_for_rq1("fail", "verifier_incomplete") is False
    assert row_eligible_for_rq1("execution_error", "docker_error") is False


def test_summarize_records_diagnostics(tmp_path: Path) -> None:
    p = tmp_path / "results_records.csv"
    p.write_text(
        "task_id,method,rank,selection_tier,seed,generation_id,candidate_id,pass_rate,cost_usd,cost_usd_raw,duration_sec,status,failure_type\n"
        "a,no_skill,1,1,0,0,0,0.0,1,1,1,pass,none\n"
        "a,no_skill,2,1,0,0,0,0.0,1,1,1,infra_error,infra_error\n",
        encoding="utf-8",
    )
    rows = summarize_records_diagnostics(p)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "a"
    assert rows[0]["n_total"] == 2
    assert rows[0]["n_eligible"] == 1
    assert rows[0]["eligible_for_rq1"] is True
