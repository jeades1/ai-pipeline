# Ranking Improvement Plan

## Current Problems
- Precision@K = 0 because known AKI markers rank 700-2800 instead of top-100
- Overly simple scoring: just 0.7×effect_size + 0.3×(-log10 p_value)
- No integration of knowledge graph, tissue relevance, or causal information

## Proposed Solutions

### 1. Knowledge Graph Features
- **Path-based scoring**: Shortest path distance to known disease nodes
- **Network centrality**: PageRank, betweenness centrality in disease subgraph
- **Pathway membership**: Boolean flags for disease-relevant pathways (injury, repair, inflammation)

### 2. Multi-Modal Feature Engineering
```python
def enhanced_ranking_score(gene_data):
    # Current simple score
    base_score = 0.7 * effect_size + 0.3 * (-log10(p_value))
    
    # Knowledge graph features
    kg_score = (
        0.3 * pathway_relevance_score +  # injury/repair pathway membership
        0.2 * network_centrality +       # PageRank in disease subgraph  
        0.2 * expression_specificity +   # tissue-specific expression
        0.1 * literature_evidence +      # PubMed co-occurrence
        0.2 * causal_support            # experimental evidence strength
    )
    
    # Adapter-specific weighting
    assay_multiplier = adapter_weights[resolve_assay_type(dataset)]
    
    return base_score * (1 + kg_score) * assay_multiplier
```

### 3. Learning-to-Rank (LTR) Approach
- Train on known benchmark markers as positive examples
- Use pairwise or listwise ranking loss
- Feature engineering: statistical + network + biological
- Cross-validation to prevent overfitting

### 4. Uncertainty Calibration
- Confidence intervals on rankings
- Ensemble methods across multiple feature sets
- Calibrate against held-out validation benchmarks

## Implementation Priority
1. **Week 1**: Add pathway membership and network centrality features
2. **Week 2**: Implement LTR with scikit-learn RankSVM or XGBoost
3. **Week 3**: Cross-validate against AKI benchmark + other disease benchmarks
4. **Week 4**: Uncertainty calibration and confidence intervals

## Success Metrics
- **Primary**: Precision@10 > 0.1, Precision@50 > 0.04 on AKI benchmark
- **Secondary**: Mean reciprocal rank improvement, NDCG@100
- **Validation**: Performance on oncology, cardio-metabolic benchmarks
