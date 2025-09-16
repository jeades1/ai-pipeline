# Causal Biomarker Discovery Report

## Summary Statistics
- **Total Biomarkers Analyzed**: 4
- **Biomarkers with Causal Evidence (>50% confidence)**: 3
- **Average Integrated Score**: 0.781

## Evidence Tier Distribution
- **Tier 1**: 1 biomarkers
- **Tier 2**: 2 biomarkers
- **Tier 5**: 1 biomarkers

## Top 10 Biomarkers by Integrated Score

| Rank | Biomarker | Integrated Score | Causal Confidence | Evidence Tier | Layer |
|------|-----------|------------------|-------------------|---------------|-------|
| 1 | CRP | 0.957 | 1.000 | 1 | proteomic |
| 2 | METABOLITE_X | 0.953 | 0.878 | 2 | metabolomic |
| 3 | ENZYME_ACTIVITY | 0.827 | 0.856 | 2 | proteomic |
| 4 | RANDOM_MARKER | 0.386 | 0.288 | 5 | proteomic |

## Causal Discovery Results
- **Total Causal Edges Discovered**: 19
- **High-Confidence Causal Edges (>80%)**: 11
- **Methods Used**: NOTEARS, PC-MCI, Mendelian Randomization

## Strong Causal Biomarkers (Top 5)

### CRP
- **Causal Confidence**: 1.000
- **Mechanism**: multi_method_consensus
- **Evidence Level**: 5/5
- **Upstream Factors**: ENZYME_ACTIVITY, METABOLITE_X, RANDOM_MARKER, disease_outcome
- **Downstream Targets**: ENZYME_ACTIVITY, METABOLITE_X, RANDOM_MARKER, disease_outcome

### METABOLITE_X
- **Causal Confidence**: 0.878
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: ENZYME_ACTIVITY, CRP, disease_outcome
- **Downstream Targets**: ENZYME_ACTIVITY, CRP, RANDOM_MARKER, disease_outcome

### ENZYME_ACTIVITY
- **Causal Confidence**: 0.856
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: METABOLITE_X, CRP, disease_outcome
- **Downstream Targets**: METABOLITE_X, CRP, disease_outcome
