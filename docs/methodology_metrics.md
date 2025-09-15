# Methodology: Metrics and Assessments

## 1) Platform Precision/Ranking Metrics
- Precision@K (K ∈ {5,10,20,50,100})
- Recall@K
- Mean Reciprocal Rank (MRR)
- NDCG@K
- Hit@K (any anchor in top-K)

Inputs: `artifacts/*/promoted.tsv`, `benchmarks/aki_markers.json`

## 2) Experimental Rigor (Objective Proxies)
- Experimental integration: # of in-vitro assay types integrated end-to-end (ELISA, scRNA, proteomics, MPS, imaging) — scaled 0–10
- Mechanistic understanding: # of causal edges with interventional support in KG / total disease-relevant edges — scaled 0–10
- Clinical translation: # of clinically validated biomarkers or program milestones — scaled 0–10
- Data scale: Total samples/edges with provenance and QC — scaled 0–10
- Validation throughput: # experiments/week achievable given current automation — scaled 0–10

Each metric has a count-based basis and is normalized to 0–10 for comparability.

## 3) Competitive Comparison (Outcome-Based)
- Benchmark Precision@K across public disease benchmarks
- # of diseases validated and time-to-validate
- False discovery rate in prospective validations
- Commercial signals: partnerships, pilots, revenue

## 4) Reporting Artifacts
- `artifacts/bench/benchmark_report.json` — numeric metrics
- `artifacts/pitch/precision_analysis.png` — refreshed from benchmark report
- `artifacts/pitch/experimental_rigor_comparison.png` — refreshed from objective proxies
- `artifacts/pitch/platform/capabilities_matrix.png` — replaced by outcome-based comparison when available
