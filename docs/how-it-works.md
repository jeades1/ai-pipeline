# How It Works: End-to-End Walkthrough

This walkthrough connects fundamental elements to the assembled system and shows how discoveries are made, validated, and translated.

## 1) Fundamentals (What we observe and enforce)
- Multi-modal inputs: omics, imaging, functional readouts (TEER/MEA/contractility), clinical variables.
- Mesoscale mediators: tissue function mediates molecular → clinical; enforced via mediation and avatars.
- Knowledge Graph priors: typed biological relations constrain search and aggregation.

## 2) Assembly (How components fit together)
- Feature extraction and QC → leakage-safe splits → association screening (E0/E1).
- Causal/mediation analysis to enforce molecular → function → clinical structure.
- Active learning selects perturbations for tissue-chip experiments to disambiguate hypotheses.
- Evidence updates flow into the KG with provenance and uncertainty.

## 3) Operation (Closed-loop discovery)
1. Hypothesize via KG + prior evidence.
2. Run targeted chip experiments; measure functional changes; replicate across donors/models (E2/E3).
3. Update avatars and re-score biomarker candidates.
4. Validate in federated clinical cohorts with calibration and transport checks (E4/E5).
5. Produce ranked dossiers with mechanism paths, effect sizes, and assayability.

## 4) Why current best practices fall short
- Associative focus: unstable across time/sites; confounds and batch effects.
- Downstream tracking: markers capture consequences, not causal mediators; poor intervention utility.
- Limited transportability: site-specific pipelines underperform in new populations.
- Minimal analytical rigor: delayed LoD/LoQ and robustness testing.

## 5) How this improves translation
- Requires interventional functional evidence early (E2) before expensive clinical work.
- Enforces mediation through tissue function, improving mechanistic interpretability.
- Federated, time-aware validation improves generalization and safety.
- Early analytical validation reduces late-stage failure.

## 6) Outputs
- Evidence-backed biomarker panels with dossiers.
- Patient-specific avatars and treatment simulations.
- KG updates that accelerate the next discovery cycle.

See also: ../ENHANCED_VALIDATION_SUMMARY.md, capabilities.md, methodology_metrics.md, benchmarks.md
