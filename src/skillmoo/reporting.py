from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Any

from .config import METHODS
from .io import read_csv_rows, read_json, write_csv, write_json


INFRA_FAILURE_TYPES = frozenset(
    {
        "infra_error",
        "docker_error",
        "verifier_incomplete",
        "agent_setup_error",
    }
)
HARNESS_INELIGIBLE_STATUS = frozenset({"infra_error", "timeout"})


@dataclass(frozen=True)
class MethodTaskRows:
    task_id: str
    method: str
    values: list[dict[str, Any]]


def build_matrix_report(
    input_root: Path,
    baseline_method: str = "original_skills",
    records_csv: Path | None = None,
) -> dict[str, Any]:
    groups = _load_groups(input_root)
    rows = [_summarize_group(group) for group in groups]
    _attach_deltas(rows, baseline_method)
    diagnostics_path = _resolve_records_path(input_root, records_csv)
    diag_map: dict[tuple[str, str], dict[str, Any]] = {}
    if diagnostics_path is not None and diagnostics_path.is_file():
        diag_rows = summarize_records_diagnostics(diagnostics_path)
        diag_map = {(str(r["task_id"]), str(r["method"])): r for r in diag_rows}
        write_csv(input_root / "ase_nier_records_diagnostics.csv", _diagnostics_fields(), diag_rows)
        write_json(input_root / "ase_nier_records_diagnostics.json", {"rows": diag_rows})
        for row in rows:
            key = (str(row["task_id"]), str(row["method"]))
            d = diag_map.get(key)
            if d:
                row["status_counts"] = d.get("status_counts", "")
                row["failure_type_counts"] = d.get("failure_type_counts", "")
                row["n_record_runs"] = d.get("n_total", 0)
                row["n_eligible_runs"] = d.get("n_eligible", 0)
                row["eligible_for_rq1"] = bool(d.get("eligible_for_rq1", False))
    overall = _overall_rows(rows)
    payload = {
        "task_count": len({row["task_id"] for row in rows}),
        "method_count": len({row["method"] for row in rows}),
        "baseline_method": baseline_method,
        "rows": rows,
        "overall": overall,
        "records_diagnostics_path": str(diagnostics_path) if diagnostics_path else "",
    }
    write_json(input_root / "ase_nier_report.json", payload)
    write_csv(input_root / "ase_nier_report.csv", _report_fields(), rows)
    write_csv(input_root / "ase_nier_overall.csv", _overall_fields(), overall)
    return payload


def _resolve_records_path(input_root: Path, records_csv: Path | None) -> Path | None:
    if records_csv is not None:
        return records_csv
    candidate = input_root / "results_records.csv"
    if candidate.is_file():
        return candidate
    return None


def row_eligible_for_rq1(status: str, failure_type: str) -> bool:
    st = (status or "").strip()
    ft = (failure_type or "").strip()
    if st == "pass":
        return True
    if st in HARNESS_INELIGIBLE_STATUS:
        return False
    if ft in INFRA_FAILURE_TYPES:
        return False
    return st in ("fail", "execution_error")


def summarize_records_diagnostics(records_path: Path) -> list[dict[str, Any]]:
    raw = read_csv_rows(records_path)
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for rec in raw:
        task_id = str(rec.get("task_id", "")).strip()
        method = str(rec.get("method", "")).strip()
        if not task_id or not method:
            continue
        buckets.setdefault((task_id, method), []).append(rec)

    out: list[dict[str, Any]] = []
    for (task_id, method), items in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        status_hist: dict[str, int] = {}
        failure_hist: dict[str, int] = {}
        eligible = 0
        for item in items:
            s = str(item.get("status", "")).strip()
            f = str(item.get("failure_type", "")).strip()
            status_hist[s] = status_hist.get(s, 0) + 1
            failure_hist[f] = failure_hist.get(f, 0) + 1
            if row_eligible_for_rq1(s, f):
                eligible += 1
        n_total = len(items)
        out.append(
            {
                "task_id": task_id,
                "method": method,
                "n_total": n_total,
                "n_eligible": eligible,
                "eligible_for_rq1": eligible > 0,
                "eligible_fraction": round(eligible / n_total, 6) if n_total else 0.0,
                "status_counts": _hist_to_str(status_hist),
                "failure_type_counts": _hist_to_str(failure_hist),
            }
        )
    return out


def _hist_to_str(h: dict[str, int]) -> str:
    if not h:
        return ""
    return ";".join(f"{k}:{h[k]}" for k in sorted(h))


def write_records_diagnostics_only(records_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = summarize_records_diagnostics(records_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ase_nier_records_diagnostics.csv", _diagnostics_fields(), rows)
    write_json(output_dir / "ase_nier_records_diagnostics.json", {"rows": rows})
    return rows


def _diagnostics_fields() -> list[str]:
    return [
        "task_id",
        "method",
        "n_total",
        "n_eligible",
        "eligible_for_rq1",
        "eligible_fraction",
        "status_counts",
        "failure_type_counts",
    ]


def _load_groups(input_root: Path) -> list[MethodTaskRows]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for summary_path in sorted(input_root.glob("*/seed_*/summary.json")):
        method = summary_path.parents[1].name
        seed_name = summary_path.parent.name
        seed = int(seed_name.removeprefix("seed_"))
        summary = read_json(summary_path)
        for row in summary.get("rows", []):
            item = dict(row)
            item["seed"] = seed
            item["method"] = method
            buckets.setdefault((str(item["task_id"]), method), []).append(item)
    order = {method: index for index, method in enumerate(METHODS)}
    return [
        MethodTaskRows(task_id=task, method=method, values=values)
        for (task, method), values in sorted(buckets.items(), key=lambda item: (item[0][0], order.get(item[0][1], 99)))
    ]


def _summarize_group(group: MethodTaskRows) -> dict[str, Any]:
    pass_values = [float(item.get("best_pass_rate", item.get("pass_rate_mean", 0.0))) for item in group.values]
    cost_values = [float(item.get("best_cost_usd", item.get("cost_usd_mean", 0.0))) for item in group.values]
    duration_values = [
        float(item.get("best_duration_sec", item.get("duration_sec_mean", 0.0))) for item in group.values
    ]
    bundle_values = [float(item.get("best_bundle_size", 0)) for item in group.values]
    eval_values = [float(item.get("n_evaluations", 0)) for item in group.values]
    ci_low, ci_high = _bootstrap_ci(pass_values)
    return {
        "task_id": group.task_id,
        "method": group.method,
        "n_seeds": len(group.values),
        "eval_count_mean": round(mean(eval_values), 6) if eval_values else 0.0,
        "pass_rate_mean": _mean(pass_values),
        "pass_rate_sd": _sd(pass_values),
        "pass_rate_ci_low": ci_low,
        "pass_rate_ci_high": ci_high,
        "cost_usd_mean": _mean(cost_values),
        "duration_sec_mean": _mean(duration_values),
        "bundle_size_mean": _mean(bundle_values),
        "absolute_delta_vs_original": "",
        "relative_delta_pct_vs_original": "",
        "status_counts": "",
        "failure_type_counts": "",
        "n_record_runs": 0,
        "n_eligible_runs": 0,
        "eligible_for_rq1": False,
    }


def _attach_deltas(rows: list[dict[str, Any]], baseline_method: str) -> None:
    baselines = {
        row["task_id"]: float(row["pass_rate_mean"])
        for row in rows
        if row["method"] == baseline_method
    }
    for row in rows:
        baseline = baselines.get(row["task_id"])
        if baseline is None:
            continue
        delta = float(row["pass_rate_mean"]) - baseline
        row["absolute_delta_vs_original"] = round(delta, 6)
        if baseline == 0.0:
            row["relative_delta_pct_vs_original"] = ""
        else:
            row["relative_delta_pct_vs_original"] = round((delta / baseline) * 100.0, 6)


def _overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    out: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        out.append(
            {
                "method": method,
                "task_count": len(items),
                "pass_rate_mean": _mean([float(item["pass_rate_mean"]) for item in items]),
                "cost_usd_mean": _mean([float(item["cost_usd_mean"]) for item in items]),
                "duration_sec_mean": _mean([float(item["duration_sec_mean"]) for item in items]),
                "bundle_size_mean": _mean([float(item["bundle_size_mean"]) for item in items]),
                "eval_count_mean": _mean([float(item["eval_count_mean"]) for item in items]),
            }
        )
    return out


def _bootstrap_ci(values: list[float], n_samples: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        value = round(values[0], 6)
        return value, value
    rng = random.Random(12345)
    means = []
    for _ in range(n_samples):
        sample = [rng.choice(values) for _ in values]
        means.append(mean(sample))
    means.sort()
    return round(means[int(0.025 * n_samples)], 6), round(means[int(0.975 * n_samples) - 1], 6)


def _mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _sd(values: list[float]) -> float:
    return round(stdev(values), 6) if len(values) > 1 else 0.0


def _report_fields() -> list[str]:
    return [
        "task_id",
        "method",
        "n_seeds",
        "eval_count_mean",
        "pass_rate_mean",
        "pass_rate_sd",
        "pass_rate_ci_low",
        "pass_rate_ci_high",
        "absolute_delta_vs_original",
        "relative_delta_pct_vs_original",
        "cost_usd_mean",
        "duration_sec_mean",
        "bundle_size_mean",
        "status_counts",
        "failure_type_counts",
        "n_record_runs",
        "n_eligible_runs",
        "eligible_for_rq1",
    ]


def _overall_fields() -> list[str]:
    return [
        "method",
        "task_count",
        "pass_rate_mean",
        "cost_usd_mean",
        "duration_sec_mean",
        "bundle_size_mean",
        "eval_count_mean",
    ]
