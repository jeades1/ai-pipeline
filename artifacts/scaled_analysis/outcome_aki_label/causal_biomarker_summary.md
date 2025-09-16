# Causal Biomarker Discovery Report

## Summary Statistics
- **Total Biomarkers Analyzed**: 78
- **Biomarkers with Causal Evidence (>50% confidence)**: 38
- **Average Integrated Score**: 0.550

## Evidence Tier Distribution
- **Tier 2**: 14 biomarkers
- **Tier 3**: 9 biomarkers
- **Tier 4**: 12 biomarkers
- **Tier 5**: 43 biomarkers

## Top 10 Biomarkers by Integrated Score

| Rank | Biomarker | Integrated Score | Causal Confidence | Evidence Tier | Layer |
|------|-----------|------------------|-------------------|---------------|-------|
| 1 | creatinine_slope_24h | 0.933 | 0.925 | 2 | clinical |
| 2 | creatinine_std | 0.932 | 0.933 | 2 | clinical |
| 3 | potassium_std | 0.920 | 0.930 | 2 | molecular |
| 4 | potassium_slope_24h | 0.918 | 0.928 | 2 | molecular |
| 5 | sodium_std | 0.917 | 0.928 | 2 | clinical |
| 6 | sodium_slope_24h | 0.916 | 0.933 | 2 | clinical |
| 7 | chloride_std | 0.915 | 0.925 | 2 | molecular |
| 8 | chloride_slope_24h | 0.914 | 0.929 | 2 | molecular |
| 9 | creatinine_max | 0.912 | 0.918 | 2 | clinical |
| 10 | creatinine_mean | 0.912 | 0.914 | 2 | clinical |

## Causal Discovery Results
- **Total Causal Edges Discovered**: 5709
- **High-Confidence Causal Edges (>80%)**: 843
- **Methods Used**: NOTEARS, PC-MCI, Mendelian Randomization

## Strong Causal Biomarkers (Top 5)

### creatinine_std
- **Causal Confidence**: 0.933
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_injury, module_inflammation, gene_HAVCR1, gene_TNF, gene_IL6, gene_SOD2, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_max, lactate_mean, ast_min, ast_mean, alt_min, alt_max, bilirubin_min, bilirubin_mean, module_collecting_duct, module_injury, module_tubular_transport, module_inflammation, module_metabolism, gene_HAVCR1, gene_LCN2, gene_IL18, gene_TNF, gene_COL1A1, outcome

### sodium_slope_24h
- **Causal Confidence**: 0.933
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_max, sodium_mean, sodium_std, sodium_peak_time, chloride_min, chloride_max, chloride_mean, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_max, bilirubin_mean, module_collecting_duct, module_injury, module_repair, module_tubular_transport, module_inflammation, gene_HAVCR1, gene_IL18, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_min, sodium_max, sodium_mean, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, module_repair, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_fibrosis, module_metabolism, gene_HAVCR1, gene_TIMP2, gene_TNF, gene_IL6, outcome

### potassium_std
- **Causal Confidence**: 0.930
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_TAL, module_repair, module_tubular_transport, module_oxidative_stress, module_inflammation, module_metabolism, gene_IL18, gene_TNF, gene_COL1A1, gene_SOD2, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, ast_min, ast_max, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_TAL, module_collecting_duct, module_injury, module_tubular_transport, module_inflammation, module_apoptosis, module_metabolism, gene_HAVCR1, gene_TNF, gene_SOD2, outcome

### chloride_slope_24h
- **Causal Confidence**: 0.929
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_min, sodium_mean, sodium_std, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, hemoglobin_min, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, glucose_min, glucose_max, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_inflammation, gene_HAVCR1, gene_LCN2, gene_TNF, gene_COL1A1, pathway_inflammation_fibrosis, pathway_oxidative_apoptosis, outcome
- **Downstream Targets**: creatinine_max, creatinine_mean, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_peak_time, sodium_mean, sodium_std, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_repair, module_tubular_transport, module_inflammation, gene_LCN2, gene_TIMP2, gene_TNF, gene_IL6, gene_SOD2, outcome

### potassium_slope_24h
- **Causal Confidence**: 0.928
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_mean, potassium_std, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_TAL, module_acute_kidney_injury, module_tubular_transport, module_oxidative_stress, module_apoptosis, gene_LCN2, gene_TNF, gene_COL1A1, pathway_inflammation_fibrosis, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, sodium_max, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_metabolism, gene_HAVCR1, gene_TIMP2, gene_IGFBP7, gene_TNF, gene_SOD2, pathway_oxidative_apoptosis, outcome
