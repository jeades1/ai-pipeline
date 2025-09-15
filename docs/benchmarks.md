# Benchmarking Harness

This repo includes a simple, reproducible benchmarking script to quantify rediscovery performance with confidence intervals.

- Script: `benchmarks/run_benchmarks.py`
- Inputs:
  - `artifacts/promoted.tsv` — produced by your pipeline run
  - `data/benchmarks/*.tsv` — list(s) of known biomarkers for a given context
- Metrics: precision@k, recall@k with 95% bootstrap CIs
- Output: `artifacts/bench/benchmark_report.json`

## Run

```bash
make bench-run
```
