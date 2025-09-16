# Causal Biomarker Discovery Report

## Summary Statistics
- **Total Biomarkers Analyzed**: 78
- **Biomarkers with Causal Evidence (>50% confidence)**: 25
- **Average Integrated Score**: 0.470

## Evidence Tier Distribution
- **Tier 2**: 11 biomarkers
- **Tier 3**: 8 biomarkers
- **Tier 5**: 59 biomarkers

## Top 10 Biomarkers by Integrated Score

| Rank | Biomarker | Integrated Score | Causal Confidence | Evidence Tier | Layer |
|------|-----------|------------------|-------------------|---------------|-------|
| 1 | creatinine_std | 0.701 | 0.766 | 2 | clinical |
| 2 | creatinine_slope_24h | 0.700 | 0.768 | 2 | clinical |
| 3 | chloride_std | 0.687 | 0.755 | 2 | molecular |
| 4 | creatinine_min | 0.684 | 0.750 | 2 | clinical |
| 5 | sodium_std | 0.684 | 0.747 | 2 | clinical |
| 6 | creatinine_max | 0.681 | 0.734 | 2 | clinical |
| 7 | chloride_slope_24h | 0.679 | 0.732 | 2 | molecular |
| 8 | creatinine_mean | 0.679 | 0.735 | 2 | clinical |
| 9 | potassium_slope_24h | 0.678 | 0.768 | 2 | molecular |
| 10 | sodium_slope_24h | 0.674 | 0.699 | 2 | clinical |

## Causal Discovery Results
- **Total Causal Edges Discovered**: 5684
- **High-Confidence Causal Edges (>80%)**: 823
- **Methods Used**: NOTEARS, PC-MCI, Mendelian Randomization

## Strong Causal Biomarkers (Top 5)

### creatinine_slope_24h
- **Causal Confidence**: 0.768
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_repair, module_acute_kidney_injury, module_tubular_transport, module_metabolism, gene_LCN2, gene_TIMP2, gene_IGFBP7, gene_SOD2, pathway_inflammation_fibrosis, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_repair, module_tubular_transport, module_inflammation, module_fibrosis, module_apoptosis, module_metabolism, gene_HAVCR1, gene_TNF, gene_IL6, outcome

### potassium_slope_24h
- **Causal Confidence**: 0.768
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_peak_time, sodium_min, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, wbc_min, wbc_max, wbc_mean, glucose_min, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, bilirubin_min, bilirubin_max, bilirubin_mean, module_inflammation, gene_HAVCR1, gene_TNF, gene_TGFB1, gene_SOD2, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_max, potassium_std, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_max, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_injury, module_tubular_transport, module_inflammation, module_fibrosis, gene_HAVCR1, gene_TNF, gene_IL6, gene_COL1A1, gene_SOD2, outcome

### creatinine_std
- **Causal Confidence**: 0.766
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, urea_min, urea_max, urea_mean, urea_std, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, chloride_peak_time, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_mean, ast_min, ast_max, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_injury, module_tubular_transport, module_inflammation, module_metabolism, gene_HAVCR1, gene_TNF, gene_IL6, gene_COL1A1, gene_SOD2, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_collecting_duct, module_acute_kidney_injury, module_tubular_transport, module_inflammation, gene_LCN2, gene_IL6, pathway_inflammation_fibrosis, pathway_oxidative_apoptosis, outcome

### chloride_std
- **Causal Confidence**: 0.755
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_mean, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_TAL, module_collecting_duct, module_injury, module_tubular_transport, module_oxidative_stress, module_inflammation, module_fibrosis, gene_HAVCR1, gene_IL18, gene_TNF, gene_IL6, gene_TGFB1, pathway_inflammation_fibrosis, pathway_oxidative_apoptosis, outcome
- **Downstream Targets**: creatinine_min, creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_mean, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_slope_24h, chloride_peak_time, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_max, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_max, bilirubin_mean, module_injury, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_fibrosis, gene_TNF, gene_TGFB1, outcome

### creatinine_min
- **Causal Confidence**: 0.750
- **Mechanism**: Direct relationship
- **Evidence Level**: 3/5
- **Upstream Factors**: creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, potassium_min, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_min, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_min, platelets_max, platelets_mean, wbc_min, wbc_max, glucose_min, glucose_max, glucose_mean, lactate_min, lactate_max, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_injury, module_repair, module_acute_kidney_injury, module_tubular_transport, module_inflammation, module_apoptosis, module_metabolism, gene_IL18, gene_IGFBP7, gene_TNF, gene_COL1A1, outcome
- **Downstream Targets**: creatinine_max, creatinine_mean, creatinine_std, creatinine_slope_24h, creatinine_peak_time, urea_min, urea_max, urea_mean, urea_std, urea_slope_24h, urea_peak_time, potassium_min, potassium_max, potassium_mean, potassium_std, potassium_slope_24h, potassium_peak_time, sodium_max, sodium_mean, sodium_std, sodium_slope_24h, sodium_peak_time, chloride_min, chloride_max, chloride_mean, chloride_std, chloride_slope_24h, hemoglobin_min, hemoglobin_max, hemoglobin_mean, platelets_max, platelets_mean, wbc_min, wbc_max, wbc_mean, glucose_min, glucose_mean, lactate_min, lactate_max, lactate_mean, ast_min, ast_max, ast_mean, alt_min, alt_max, alt_mean, bilirubin_min, bilirubin_max, bilirubin_mean, module_proximal_tubule, module_injury, module_repair, module_acute_kidney_injury, module_tubular_transport, module_oxidative_stress, module_inflammation, gene_HAVCR1, gene_TNF, gene_TGFB1, pathway_inflammation_fibrosis, outcome
