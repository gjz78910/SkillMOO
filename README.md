# SkillMOO Replication Package (ASE 2026 NIER)

This package supports the replication of the ASE 2026 NIER paper:
**"SkillMOO: Multi-Objective Optimization of Agent Skill Bundles for Software Engineering Tasks"**

It supports two use cases:

* Mode A: **Inspect frozen results** (recommended for artifact evaluation).
* Mode B: **Full 16-task rerun** (requires external infrastructure).

Primary model in the frozen package: `GLM-5`.

## 1) Repository Structure

```text
SkillMOO/
├── README.md                                  # This guide: setup, modes, and key outputs
├── pyproject.toml                             # Python package metadata and dependencies
├── tasks_manifest.json                        # Default 16-task SkillsBench pool used by Mode B
├── patches/skillsbench-ase-nier-overlay.tar.gz # Task and verifier changes applied during setup
├── src/skillmoo/                              # Core SkillMOO source code
│   ├── candidates.py                          # Candidate bundle representation and population generation
│   ├── config.py                              # Run configuration, method definitions, task-pool loading
│   ├── feedback.py                            # Verifier-output parsing and retry-family recommendation
│   ├── io.py                                  # JSON/CSV read-write helpers
│   ├── loop.py                                # Solver-optimizer generation loop (evaluate -> select -> edit)
│   ├── llm_provider.py                        # Minimal OpenAI-compatible HTTP client for the skill optimizer
│   ├── metrics.py                             # Per-task pass rate / cost / duration aggregation
│   ├── operators.py                           # Adaptive weighting over prune/substitute/add/reorder families
│   ├── optimizer.py                           # LLM-proposed bundle edits, with heuristic fallback
│   ├── reporting.py                           # Matrix report builder (aggregates a completed Mode B run)
│   ├── runner.py                              # Top-level experiment runner (Mode B)
│   ├── selection.py                           # NSGA-II non-dominated sorting and survivor ordering
│   └── tasks.py                               # Task manifest resolution against a SkillsBench checkout
├── scripts/
│   ├── build_ase_nier_report.py               # Rebuild the RQ1/RQ2/RQ3 aggregate reports from a completed run
│   ├── run_ase_nier_matrix.py                 # Full 16-task matrix runner (Mode B)
│   ├── compute_sk_esd_ranks.py                # Optional: Scott-Knott ESD r_p/r_c ranks (Table 2), needs R
│   └── setup_skillsbench.py                   # Pins and prepares the external SkillsBench checkout
├── reports/                                   # Frozen results supporting paper tables
│   ├── results_summary.csv                    # Per-task summary: 16 tasks × 3 methods (Table 2 / RQ1)
│   ├── results_records.csv                    # Per-run evaluation detail per method/task (10 seeds each)
│   ├── results_full.json                      # Full structured results with cost calibration metadata
│   ├── ase_nier_records_diagnostics.{csv,json} # Per-task/method status/failure histograms
│   ├── rq2_hv.csv                             # Per-task HV values and optimization overhead (Table 3 / RQ2)
│   └── rq3_summary.csv                        # Skill-edit operation counts and outcome ratios (Table 4 / RQ3)
└── tests/
    ├── conftest.py
    ├── unit/                                  # Unit tests for individual modules
    └── regression/                            # Regression tests, including paper-table consistency checks
```

## 2) Quick Setup

From the package root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONUNBUFFERED=1
```

## 3) Mode A: Inspect Frozen Results

This mode does not require external task execution or API credentials.
The frozen reports in `reports/` are the canonical pre-built artifacts directly supporting the paper tables.

### Verify RQ1 (Table 2): pass rate and cost per task/method

`reports/results_summary.csv` contains the per-task, per-method summary (16 tasks × 3 methods):

```bash
python -c "
import pandas as pd
df = pd.read_csv('reports/results_summary.csv')
print(df[['task_id','method','pass_mean','cost_mean','best_pass']].to_string(index=False))
"
```

Table 2's $r_p$/$r_c$ columns are Scott-Knott ESD ranks computed separately from the 10
per-seed runs in `results_records.csv` — see [Section 6a](#6a-optional-reproducing-table-2s-rank-columns).

### Verify RQ2 (Table 3): HV improvement and optimization overhead

`reports/rq2_hv.csv` contains pre-computed hypervolume values for the 12 non-zero-pass tasks
(Tasks 7, 13, 15, 16 are excluded: all methods score zero pass rate).
HV is computed in normalized [0,1] space with reference point (0, 1.1):

```bash
python -c "
import pandas as pd
df = pd.read_csv('reports/rq2_hv.csv')
print(df[['task_id','hv_ori_skill','hv_skillmoo','delta_hv_pct','opt_cost_usd']].to_string(index=False))
"
```

Break-even reuse count (how many repeated task runs it takes for the one-time optimization cost to
be recovered by the per-run cost saving of `skillmoo` over `original_skills`), joining `rq2_hv.csv`
with the per-task mean costs in `results_summary.csv`:

```bash
python -c "
import math
import pandas as pd
summary = pd.read_csv('reports/results_summary.csv')
hv = pd.read_csv('reports/rq2_hv.csv')
skillmoo = summary[summary.method == 'skillmoo'][['task_id', 'cost_mean']].rename(columns={'cost_mean': 'cost_skillmoo'})
original = summary[summary.method == 'original_skills'][['task_id', 'cost_mean']].rename(columns={'cost_mean': 'cost_ori'})
merged = hv.merge(skillmoo, on='task_id').merge(original, on='task_id')
merged['per_run_saving'] = merged['cost_ori'] - merged['cost_skillmoo']
merged['breakeven_reuses'] = (merged['opt_cost_usd'] / merged['per_run_saving']).apply(math.ceil)
print(merged[['task_id', 'opt_cost_usd', 'per_run_saving', 'breakeven_reuses']].to_string(index=False))
print('min:', merged['breakeven_reuses'].min(), 'median:', merged['breakeven_reuses'].median(), 'max:', merged['breakeven_reuses'].max())
"
```

### Verify RQ3 (Table 4): skill-edit operation counts and outcomes

`reports/rq3_summary.csv` contains the 38 analyzed skill-bundle edits broken down by operation
type and subcategory, with outcome ratios (pass improved / cost reduced / runtime reduced):

```bash
python -c "
import pandas as pd
df = pd.read_csv('reports/rq3_summary.csv')
totals = df.groupby('operation')['n_edits'].sum().reset_index()
print(totals.to_string(index=False))
print('Total edits:', df['n_edits'].sum())
"
```

## 4) Algorithm Design Notes

### Operator families and adaptive weighting

`OperatorPolicy` (`src/skillmoo/operators.py`) selects between three edit families each generation:

| Family | Bias |
|--------|------|
| `pass` | Add a skill to improve correctness |
| `cost` | Prune a skill to reduce inference cost |
| `length` | Substitute or shrink bundle to reduce context length |

Weights are updated multiplicatively after each generation (gain × 1.2 on success, decay × 0.8 on failure, floor 0.15). This adaptive weighting is an implementation-level mechanism; the paper describes the three families in §2 as "pass rate-focused, cost-focused, and length-focused edits."

### Population replacement strategy

`_next_population` in `loop.py` applies a mu,lambda strategy: at each generation, NSGA-II selects the best parents from the *current* generation's evaluated candidates, and the next population consists entirely of their offspring. No parent individual is carried forward unchanged. The paper's §2 describes this as "top-ranked survivors breed the next generation's population," which refers to the selection of breeding parents, not parent elitism.

## 5) Mode B: Full 16-Task Rerun (External Dependencies Required)

This mode requires external infrastructure:

- The exact SkillsBench checkout and ASE-NIER task overlay, prepared by:

```bash
python scripts/setup_skillsbench.py
```

  The script clones `benchflow-ai/skillsbench` at commit `593b0c6a3d95e0d4acc813788b12b6c044560b43` and applies the packaged task, skill, and verifier overlay.
- Docker running locally, and the [Harbor](https://github.com/harbor-framework/harbor) CLI that `skillmoo.skillsbench_runner` shells out to:

```bash
pip install harbor
harbor --help  # confirms the CLI is on PATH
```
- API credentials for the **task solver** and **skill optimizer** agents (both use the same model):

| Variable | Purpose | Example |
|----------|---------|---------|
| `SKILLMOO_LLM_API_KEY` | API key for both agents (also accepts `OPENAI_API_KEY` / `GLM_API_KEY`) | `sk-...` |
| `SKILLMOO_LLM_BASE_URL` | OpenAI-compatible base URL (also accepts `OPENAI_BASE_URL` / `GLM_BASE_URL`) | `https://open.bigmodel.cn/api/paas/v4` |
| `SKILLMOO_LLM_MODEL` | Model name for both agents (also accepts `GLM_MODEL`) | `glm-4-0520` |
| `SKILLMOO_LLM_TIMEOUT_SEC` | Per-request timeout in seconds (default: 180) | `300` |

If the LLM optimizer env vars are not set, the skill optimizer falls back to a heuristic rule-based mutator.

### Run the full 16-task matrix

Without `--task-ids`, the runner defaults to the 16-task pool bundled in
[`tasks_manifest.json`](tasks_manifest.json) (the same tasks listed in Section 6's table below).

```bash
python scripts/run_ase_nier_matrix.py \
  --repo-root <path-to-repo-root> \
  --output-root experiments/ase_nier_rerun \
  --methods no_skill,original_skills,skillmoo \
  --repeat-runs 10 \
  --model-name GLM-5 \
  --agent-name terminus-2 \
  --population-size 4 \
  --num-generations 3 \
  --timeout-sec 900
```

### Rebuild the report from a completed run

```bash
python scripts/build_ase_nier_report.py \
  --repo-root <path-to-repo-root> \
  --input-root experiments/ase_nier_rerun
```

Optional: merge `reports/results_records.csv` into the matrix report (auto-detected from this repository). This adds per-task/method **`eligible_for_rq1`** and status histograms so **infrastructure failures** (Docker, verifier incomplete, agent setup) are not confused with solver pass rate.

```bash
python scripts/build_ase_nier_report.py \
  --repo-root <path-to-repo-root> \
  --input-root experiments/ase_nier_rerun \
  --records-csv reports/results_records.csv
```

Diagnostics only (no `summary.json` required):

```bash
python scripts/build_ase_nier_report.py \
  --repo-root <path-to-repo-root> \
  --records-diagnostics-only
```

### Infra errors during matrix runs

By default, SkillMOO **stops on Harbor/Docker infrastructure failures** so you notice broken images early. To **log and continue** (useful when diagnosing flaky tasks), pass `--keep-going-on-infra-error` to [`scripts/run_ase_nier_matrix.py`](scripts/run_ase_nier_matrix.py).

### Rerun only the four previously zero-pass tasks (after fixing task images)

```bash
python scripts/run_ase_nier_matrix.py \
  --repo-root <path-to-repository-root> \
  --output-root experiments/ase_nier_rerun_four \
  --task-ids flink-query,simpo-code-reproduction,taxonomy-tree-merge,trend-anomaly-causal-inference \
  --methods no_skill,original_skills,skillmoo \
  --repeat-runs 10 \
  --model-name GLM-5 \
  --population-size 4 \
  --num-generations 3 \
  --timeout-sec 900
```

Then rebuild aggregate reports (and merge diagnostics if `results_records.csv` is copied alongside the run or passed via `--records-csv`):

```bash
python scripts/build_ase_nier_report.py \
  --repo-root <path-to-repository-root> \
  --input-root experiments/ase_nier_rerun_four \
  --records-csv reports/results_records.csv
```

## 6) Key Outputs and Paper Correspondence

| Frozen file | Paper element |
|-------------|--------------|
| `reports/results_summary.csv` | Table 2 (RQ1): pass rate and cost per task/method |
| `reports/results_records.csv` | Underlying per-record evidence for Table 2 |
| `reports/results_full.json` | Full structured results with cost calibration metadata |
| `reports/rq2_hv.csv` | Table 3 (RQ2): HV improvement and optimization overhead |
| `reports/rq3_summary.csv` | Table 4 (RQ3): skill-edit operation counts and outcomes |
| `reports/ase_nier_records_diagnostics.csv` | Optional: per-task/method outcome classification (`eligible_for_rq1`) from `results_records.csv` |

## 6a) Optional: Reproducing Table 2's Rank Columns

`scripts/compute_sk_esd_ranks.py` computes the $r_p$ (pass rate) and $r_c$ (cost) Scott-Knott
ESD ranks from the 10 per-seed runs in `reports/results_records.csv`, using the official CRAN
`ScottKnottESD` R package via `rpy2`. This is a one-off statistical dependency, not part of the
core `skillmoo` package (see `pyproject.toml`):

```bash
conda install -y -c conda-forge r-base rpy2
Rscript -e "install.packages('ScottKnottESD', repos='https://cloud.r-project.org')"
python scripts/compute_sk_esd_ranks.py
```

### China-friendly builds (SkillsBench task images)

When building Docker images locally, use mirror **build-args** where each task Dockerfile documents them (typically `PIP_INDEX_URL`, `HF_ENDPOINT`, optional Apache/PyTorch index URLs). Example Hugging Face hub mirror: `HF_ENDPOINT=https://hf-mirror.com` (verify reachability from your network). PyPI: `https://pypi.tuna.tsinghua.edu.cn/simple`. Configure Docker registry mirrors in the daemon if pulls from Docker Hub are slow.

### Methods

| Method key | Description |
|------------|-------------|
| `no_skill` | Agent receives no skill guidance |
| `original_skills` | Agent uses the original static skill bundle for each task |
| `skillmoo` | Agent uses the best bundle found by SkillMOO optimization |

### Tasks and Skill Inventories

All 16 SkillsBench SE tasks are prepared by the setup script. Each task is paired with a curated skill pool sourced from the SkillsBench task environment. Single-skill tasks provide one targeted domain reference; multi-skill tasks bundle complementary guidance covering different aspects of the same problem.

| ID | Task | Category | Skills in pool |
|----|------|----------|----------------|
| 1 | citation-check | Info. Retrieval | `citation-management` |
| 2 | data-to-d3 | Data Visualization | `d3-visualization` |
| 3 | dialogue-parser | NLP/Parsing | `dialogue_graph` |
| 4 | enterprise-information-search | Info. Retrieval | `enterprise-artifact-search` |
| 5 | fix-build-agentops | Build Repair | `analyze-ci`, `patch-diff-workflow`, `pyproject-tox-compat`, `pytest-failure-triage`, `python-build-fix-playbook`, `temporal-python-testing`, `testing-python`, `uv-package-manager` |
| 6 | fix-build-google-auto | Build Repair | `maven-build-lifecycle`, `maven-dependency-management`, `maven-plugin-configuration` |
| 7 | flink-query | Data Engineering | `pdf`, `senior-data-engineer` |
| 8 | gh-repo-analytics | DevOps Analytics | `gh-cli` |
| 9 | jax-computing-basics | Sci. Computing | `jax-skills` |
| 10 | parallel-tfidf-search | Perf. Optimization | `memory-optimization`, `python-parallelization`, `workload-balancing` |
| 11 | python-scala-translation | Code Migration | `python-scala-collections`, `python-scala-functional`, `python-scala-idioms`, `python-scala-libraries`, `python-scala-oop`, `python-scala-syntax-mapping` |
| 12 | react-performance-debugging | Perf. Optimization | `browser-testing`, `react-best-practices` |
| 13 | simpo-code-reproduction | ML Reproduction | `nlp-research-repo-package-installment`, `pdf` |
| 14 | spring-boot-jakarta-migration | Code Migration | `hibernate-upgrade`, `jakarta-namespace`, `restclient-migration`, `spring-boot-migration`, `spring-security-6` |
| 15 | taxonomy-tree-merge | Data Engineering | `hierarchical-taxonomy-clustering` |
| 16 | trend-anomaly-causal-inference | Data Analysis | `data_cleaning`, `did_causal_analysis`, `feature_engineering`, `time_series_anomaly_detection` |

The skill folders (each containing a `SKILL.md` with instructions and any supporting assets) are located under `tasks/<task-id>/environment/skills/` in the SkillsBench repository.

### Test Augmentation

The default SkillsBench verifier test suites are sparse for many tasks. To provide denser behavioral coverage and safety assertions, we augmented each task's test suite using GPT-5.4. Augmented tests were:
- generated independently per task with no access to skill content,
- validated manually for correctness before inclusion,
- applied uniformly across all three methods (`skillmoo`, `original_skills`, `no_skill`),
- additive only — original compile and build gates are fully preserved.

The table below shows augmented test item counts as collected by pytest (parametrized cases count as separate items), alongside the test categories present for each task to demonstrate coverage breadth.

| ID | Task | Items | Test categories |
|----|------|------:|-----------------|
| 1 | citation-check | 40 | output format, fake citation detection, real citation non-regression, structural constraints |
| 2 | data-to-d3 | 40 | file existence, HTML/D3 structure, browser rendering, data integrity, interactivity |
| 3 | dialogue-parser | 40 | system basics, narrative content, graph logic, visualization validity, structural integrity |
| 4 | enterprise-information-search | 40 | output schema, per-question answer presence, false-positive checks, key constraints |
| 5 | fix-build-agentops | 40 | build success, env setup, test collection, patch target validation, forbidden path checks |
| 6 | fix-build-google-auto | 40 | build success, diff validity, patch file format, source file modification, run-pass verification |
| 7 | flink-query | 41 | Maven build, Flink job execution, output format/count, Java source structure, API usage |
| 8 | gh-repo-analytics | 40 | report schema, PR/issue field presence, numeric constraints, contributor fields, cross-field consistency |
| 9 | jax-computing-basics | 43 | output file existence, shape correctness, numerical accuracy, task ID coverage, no-NaN checks |
| 10 | parallel-tfidf-search | 40 | parallel implementation, small/large correctness, indexing/search performance, IDF properties, worker scaling |
| 11 | python-scala-translation | 40 | Scala compilation, unit test pass, holdout compile/test, required components/methods, forbidden placeholder absence |
| 12 | react-performance-debugging | 40 | page/API/client performance, bundle optimization, functionality, server availability, content integrity |
| 13 | simpo-code-reproduction | 40 | output file existence, loss shape/value accuracy, numerical closeness to ground truth, Python version, per-index checks |
| 14 | spring-boot-jakarta-migration | 40 | compile success, Maven test pass, namespace migration, security/RestClient markers, legacy marker absence |
| 15 | taxonomy-tree-merge | 40 | output file format, source preservation, hierarchy structure, cluster balance, naming constraints, sibling distinctiveness |
| 16 | trend-anomaly-causal-inference | 41 | data cleaning correctness, anomaly detection, feature engineering, DID causal analysis, cross-file consistency |

The packaged SkillsBench overlay installs the augmented test files into both task variants while retaining the original compile and build gates as the base.
