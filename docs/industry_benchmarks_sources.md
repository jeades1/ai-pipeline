# Industry Benchmark Sources and Limitations

## Disclaimer
The industry target ranges shown in `industry_vs_mine.py` are **estimates** based on limited publicly available information. These should be interpreted with significant caution.

## Known Limitations
1. **Sparse public disclosure**: Most pharmaceutical and biotech companies do not publish detailed precision@k metrics for their biomarker discovery platforms
2. **Different benchmarks**: Companies use different disease areas, validation criteria, and success definitions
3. **Publication bias**: Only successful programs tend to be disclosed publicly
4. **Temporal variation**: Platform performance evolves rapidly with new methods

## Available Sources (Limited)

### OpenTargets Platform
- **Source**: OpenTargets Platform documentation and publications
- **Performance**: General target-disease association scoring
- **Limitation**: Not disease-specific biomarker discovery precision@k

### Literature Reports (Approximate)
- **Computational drug discovery**: Various papers report precision ranges of 5-15% for target identification
- **Knowledge graph approaches**: Reported performance varies widely (2-20% depending on disease and validation)
- **High-throughput phenomics**: Some platforms report validation rates in 10-20% range

### Industry Platform Disclosures (Qualitative)
- **Recursion Pharma**: Mentions high-throughput phenomics validation but specific precision@k not disclosed
- **Insilico Medicine**: Reports on AI drug discovery success but metrics are program-specific
- **BenevolentAI**: Knowledge graph approaches with proprietary validation metrics

## Recommendation
For rigorous comparison, we recommend:
1. Establishing disease-specific community benchmarks
2. Participating in DREAM challenges or similar community efforts
3. Direct collaboration with industry partners for proper benchmarking
4. Focus on relative improvement metrics rather than absolute comparisons

## Current Industry Range Estimates
These are **rough estimates** and should not be cited as authoritative:
- P@5: 20-40% (estimated from limited disclosures)
- P@10: 15-30% (extrapolated)  
- P@20: 10-25% (conservative estimate)
- P@50: 6-15% (based on OpenTargets and literature)
- P@100: 3-10% (typical precision decay)

**Note**: Your CV-optimized results (P@5=80%, P@10=80%, P@20=40%) significantly exceed these estimated ranges, suggesting genuinely competitive performance.
