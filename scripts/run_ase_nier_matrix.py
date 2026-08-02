#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_paths(repo_root: Path) -> None:
    roots = [repo_root / "src"]
    for path in roots:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output-root", default="experiments/ase_nier_rerun")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--methods", default="no_skill,original_skills,skillmoo")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=0,
        metavar="N",
        help="Run each method N times with RNG seeds 0..N-1 (seed_0..seed_{N-1}). When N>0, overrides --seeds.",
    )
    parser.add_argument("--model-name", default="GLM-5")
    parser.add_argument("--agent-name", default="claude-code")
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=3)
    parser.add_argument("--max-retries-per-trial", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--keep-going-on-infra-error",
        action="store_true",
        help="Record Docker/Harbor infrastructure failures instead of stopping immediately.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _add_paths(repo_root)

    from skillmoo.config import RunConfig
    from skillmoo.config import SEARCH_METHODS
    from skillmoo.io import write_json
    from skillmoo.reporting import build_matrix_report
    from skillmoo.runner import run_experiment

    output_root = (repo_root / args.output_root).resolve()
    methods = _parse_csv(args.methods)
    if int(args.repeat_runs) > 0:
        seeds = list(range(int(args.repeat_runs)))
    else:
        seeds = [int(item) for item in _parse_csv(args.seeds)]
    write_json(
        output_root / "matrix_config.json",
        {
            "methods": methods,
            "seeds": seeds,
            "repeat_runs": int(args.repeat_runs),
            "seeds_cli": args.seeds,
            "population_size": args.population_size,
            "num_generations": args.num_generations,
            "model_name": args.model_name,
            "agent_name": args.agent_name,
            "timeout_sec": args.timeout_sec,
            "strict_formal": bool(args.strict_formal),
            "mock": bool(args.mock),
        },
    )

    for method in methods:
        for seed in seeds:
            run_root = output_root / method / f"seed_{seed}"
            summary_path = run_root / "summary.json"
            if summary_path.is_file() and not args.force:
                print(f"[ase-nier] skip existing {summary_path}")
                continue
            cfg = RunConfig(
                repo_root=repo_root,
                skillsbench_root=repo_root / "skillsbench",
                output_root=run_root,
                model_name=args.model_name,
                agent_name=args.agent_name,
                population_size=args.population_size if method in SEARCH_METHODS else 1,
                num_generations=args.num_generations if method in SEARCH_METHODS else 1,
                max_retries_per_trial=args.max_retries_per_trial,
                timeout_sec=args.timeout_sec,
                strict_formal=bool(args.strict_formal),
                mock=bool(args.mock),
                fail_fast_infra=not bool(args.keep_going_on_infra_error),
                task_ids=_parse_task_ids(args.task_ids),
                method=method,
                seed=seed,
            )
            print(f"[ase-nier] run method={method} seed={seed} out={run_root}")
            run_experiment(cfg)

    report = build_matrix_report(output_root)
    print(
        "[ase-nier] report: "
        f"tasks={report['task_count']} methods={report['method_count']} "
        f"path={output_root / 'ase_nier_report.csv'}"
    )


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _parse_task_ids(raw: str) -> tuple[str, ...] | None:
    items = _parse_csv(raw)
    return tuple(items) if items else None


if __name__ == "__main__":
    main()
