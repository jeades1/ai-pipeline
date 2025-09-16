# Causal Biomarker Discovery Report

## Summary Statistics
- **Total Biomarkers Analyzed**: 78
- **Biomarkers with Causal Evidence (>50% confidence)**: 34
- **Average Integrated Score**: 0.516

## Evidence Tier Distribution
- **Tier 2**: 15 biomarkers
- **Tier 3**: 5 biomarkers
- **Tier 4**: 7 biomarkers
- **Tier 5**: 51 biomarkers

## Top 10 Biomarkers by Integrated Score

| Rank | Biomarker | Integrated Score | Causal Confidence | Evidence Tier | Layer |
|------|-----------|------------------|-------------------|---------------|-------|
| 1 | creatinine_std | 0.904 | 0.885 | 2 | clinical |
| 2 | creatinine_slope_24h | 0.898 | 0.879 | 2 | clinical |
| 3 | creatinine_max | 0.894 | 0.880 | 2 | clinical |
| 4 | creatinine_min | 0.893 | 0.860 | 2 | clinical |
| 5 | chloride_slope_24h | 0.888 | 0.886 | 2 | molecular |
| 6 | urea_slope_24h | 0.888 | 0.850 | 2 | clinical |
| 7 | sodium_slope_24h | 0.887 | 0.894 | 2 | clinical |
| 8 | creatinine_mean | 0.887 | 0.840 | 2 | clinical |
| 9 | sodium_std | 0.887 | 0.877 | 2 | clinical |
| 10 | potassium_slope_24h | 0.882 | 0.869 | 2 | molecular |

## Causal Discovery Results
- **Total Causal Edges Discovered**: 5743
- **High-Confidence Causal Edges (>80%)**: 847
- **Methods Used**: NOTEARS, PC-MCI, Mendelian Randomization

## Strong Causal Biomarkers (Top 5)

### sodium_slope_24h
- **Causal Confidence**: 0.894
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_collecting_duct, module_injury, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_metabolism, gene_HAVCR1, gene_TNF, gene_IL6, gene_TGFB1, gene_SOD2, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_std, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_mean, bilirubin_mean, module_injury, module_acute_kidney_injury, module_inflammation, module_apoptosis, module_metabolism, gene_LCN2, gene_IL18, gene_TNF, gene_SOD2, pathway_inflammation_fibrosis, outcome

### chloride_slope_24h
- **Causal Confidence**: 0.886
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_peak_time, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_mean, bilirubin_max, bilirubin_mean, module_collecting_duct, module_acute_kidney_injury, module_tubular_transport, module_oxidative_stress, module_fibrosis, gene_HAVCR1, gene_TIMP2, gene_TNF, pathway_inflammation_fibrosis, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_max, chloride_mean, chloride_std, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_collecting_duct, module_injury, module_repair, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_apoptosis, gene_LCN2, gene_TNF, gene_IL6, gene_TGFB1, gene_SOD2, outcome

### creatinine_std
- **Causal Confidence**: 0.885
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, creatinine_peak_time, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_TAL, module_injury, module_repair, module_tubular_transport, module_inflammation, module_fibrosis, gene_TIMP2, gene_IGFBP7, gene_TNF, gene_IL6, gene_TGFB1, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_min, sodium_max, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_collecting_duct, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_fibrosis, gene_HAVCR1, gene_IL18, gene_TIMP2, gene_IGFBP7, gene_TNF, gene_TGFB1, gene_SOD2, outcome

### creatinine_max
- **Causal Confidence**: 0.880
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, module_proximal_tubule, module_tubular_transport, module_oxidative_stress, module_inflammation, module_metabolism, gene_HAVCR1, gene_LCN2, gene_TIMP2, gene_TNF, gene_COL1A1, gene_SOD2, outcome
- **Downstream Targets**: creatinine_min, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_collecting_duct, module_injury, module_repair, module_tubular_transport, module_inflammation, gene_LCN2, gene_IGFBP7, gene_TNF, outcome

### chloride_std
- **Causal Confidence**: 0.880
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, bilirubin_max, bilirubin_mean, module_tubular_transport, gene_HAVCR1, gene_IL18, gene_TNF, gene_TGFB1, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, urea_min, urea_mean, urea_std, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_collecting_duct, module_tubular_transport, module_oxidative_stress, module_inflammation, module_apoptosis, module_metabolism, gene_HAVCR1, gene_TIMP2, gene_TNF, gene_SOD2, outcome
