"""Minimal public Harbor adapter used for SkillMOO task evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any


_SECRET = re.compile(r"api[_-]?key|token|secret|password|authorization", re.I)
_PYTEST = re.compile(r"(?P<n>\d+)\s+(?P<k>passed|failed|error|errors|skipped|xfailed|xpassed)", re.I)


@dataclass
class TaskRunResult:
    task_id: str
    status: str
    test_pass_ratio: float
    cost_usd: float
    duration_sec: float
    error_class: str
    result_json_path: Path
    source_result_json_path: Path | None


def _redact(value: Any, key: str = "") -> Any:
    if _SECRET.search(key) or key.lower() == "env":
        return "***REDACTED***"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _seconds(start: Any, finish: Any) -> float:
    try:
        a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(finish).replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds())
    except ValueError:
        return 0.0


def _ratio(trial_dir: Path, payload: dict[str, Any]) -> float:
    log = trial_dir / "verifier" / "test-stdout.txt"
    if log.is_file():
        counts = {"passed": 0, "failed": 0, "error": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
        for n, kind in _PYTEST.findall(log.read_text(encoding="utf-8", errors="ignore")):
            counts[kind.lower()] += int(n)
        total = sum(counts.values())
        if total:
            return (counts["passed"] + counts["xpassed"]) / total
    rewards = payload.get("verifier_result", {}).get("rewards", {})
    return float(rewards.get("reward", 0.0) or 0.0)


class SkillsBenchRunner:
    def __init__(self, skillsbench_root: str | Path, jobs_dir_name: str = "jobs"):
        self.skillsbench_root = Path(skillsbench_root)
        self.jobs_dir_name = jobs_dir_name

    def run_task(self, *, task_id: str, model_name: str, task_path: str | Path | None = None,
                 out_dir: str | Path | None = None, agent_name: str = "terminus-2",
                 timeout_sec: int = 900, retries: int = 0, mock: bool = False, **_: Any) -> TaskRunResult:
        task = Path(task_path) if task_path else self.skillsbench_root / "tasks" / task_id
        if not task.is_dir():
            raise FileNotFoundError(f"Missing task directory: {task}")
        output = Path(out_dir or self.skillsbench_root / "skillmoo_runs")
        output.mkdir(parents=True, exist_ok=True)
        if mock:
            return self._persist(task_id, output, {"status": "fail", "test_pass_ratio": 0.5, "cost_usd": 0.01, "duration_sec": 0.0}, None)
        if not shutil.which("harbor"):
            raise RuntimeError("Harbor is required for a non-mock rerun. Install the pinned SkillsBench dependencies first.")
        latest: TaskRunResult | None = None
        for attempt in range(retries + 1):
            job = f"skillmoo-{task_id}-{int(time.time() * 1000)}-{attempt}"
            jobs_dir = output / self.jobs_dir_name
            cmd = ["harbor", "run", "-p", str(task), "-a", agent_name, "-m", model_name, "--job-name", job, "--jobs-dir", str(jobs_dir), "--n-attempts", "1"]
            try:
                subprocess.run(cmd, cwd=self.skillsbench_root, env=os.environ.copy(), timeout=timeout_sec, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                source = self._trial_result(jobs_dir, task_id)
                payload = json.loads(source.read_text(encoding="utf-8")) if source else {"exception_info": {"exception_type": "MissingResult"}}
                latest = self._persist(task_id, output, payload, source)
            except subprocess.TimeoutExpired:
                latest = self._persist(task_id, output, {"exception_info": {"exception_type": "Timeout"}, "timed_out": True}, None)
            if latest.status in {"pass", "fail"}:
                return latest
        assert latest is not None
        return latest

    @staticmethod
    def _trial_result(jobs_dir: Path, task_id: str) -> Path | None:
        candidates = []
        for path in jobs_dir.glob(f"**/{task_id}__*/result.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("task_name") == task_id:
                    candidates.append(path)
            except (OSError, json.JSONDecodeError):
                pass
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _persist(self, task_id: str, output: Path, payload: dict[str, Any], source: Path | None) -> TaskRunResult:
        timed_out = bool(payload.get("timed_out")) or "timeout" in str(payload.get("exception_info", "")).lower()
        error = "" if not payload.get("exception_info") else str(payload["exception_info"]).lower()
        status = "timeout" if timed_out else ("infra_error" if any(x in error for x in ("docker", "network", "connection", "dns", "pull")) else ("execution_error" if error else ""))
        ratio = _ratio(source.parent, payload) if source else float(payload.get("test_pass_ratio", 0.0) or 0.0)
        if not status:
            status = "pass" if ratio >= 1.0 else "fail"
        agent = payload.get("agent_result", {}) or {}
        record = _redact(payload)
        record.update({"status": status, "test_pass_ratio": ratio, "cost_usd": float(agent.get("cost_usd", payload.get("cost_usd", 0.0)) or 0.0), "duration_sec": _seconds(payload.get("started_at"), payload.get("finished_at"))})
        path = output / f"{task_id}-{int(time.time() * 1000)}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return TaskRunResult(task_id, status, ratio, float(record["cost_usd"]), float(record["duration_sec"]), status if status not in {"pass", "fail"} else "none", path, source)
