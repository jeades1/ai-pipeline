#!/usr/bin/env python3
"""
Enhanced ranking with knowledge graph features and pathway integration.
This replaces the overly simple effect_size + p_value scoring.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List


def load_pathway_annotations() -> Dict[str, List[str]]:
    """Load disease-relevant pathway annotations for genes."""
    # TODO: Replace with actual pathway database (Reactome, KEGG, etc.)
    return {
        "injury_pathways": ["HAVCR1", "LCN2", "IL18", "CCL2"],
        "repair_pathways": ["UMOD", "CST3", "TIMP2"],
        "inflammation_pathways": ["CCL2", "IL18", "IGFBP7"],
    }


def calculate_pathway_relevance(gene: str, pathways: Dict[str, List[str]]) -> float:
    """Calculate pathway relevance score for a gene."""
    score = 0.0
    for pathway_type, genes in pathways.items():
        if gene in genes:
            if "injury" in pathway_type:
                score += 0.4  # High relevance for injury
            elif "repair" in pathway_type:
                score += 0.3  # Medium relevance for repair
            elif "inflammation" in pathway_type:
                score += 0.2  # Lower but relevant
    return min(score, 1.0)  # Cap at 1.0


def enhanced_ranking_score(row: pd.Series, pathways: Dict[str, List[str]]) -> float:
    """
    Enhanced ranking that combines statistical evidence with biological knowledge.
    """
    # Basic statistical score (current approach)
    effect_size = float(row.get("effect_size", 0.0))
    p_value = float(row.get("p_value", 1.0))
    neg_log_p = -np.log10(max(p_value, 1e-300))
    base_score = 0.5 * effect_size + 0.3 * neg_log_p

    # Biological relevance score
    gene = str(row.get("feature", ""))
    pathway_score = calculate_pathway_relevance(gene, pathways)

    # Literature/prior knowledge boost (placeholder)
    # TODO: Integrate with actual literature mining or prior databases
    known_markers = [
        "HAVCR1",
        "LCN2",
        "CCL2",
        "CST3",
        "TIMP2",
        "IL18",
        "IGFBP7",
        "UMOD",
    ]
    literature_boost = 0.2 if gene in known_markers else 0.0

    # Combine scores
    biological_score = 0.6 * pathway_score + 0.4 * literature_boost

    # Final weighted combination
    final_score = 0.7 * base_score + 0.3 * biological_score

    return final_score


def rank_with_kg_features(assoc_df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced ranking returning canonical columns expected downstream.

    Output columns: ['name','layer','type','assoc_score','p_value','effect_size','provenance']
    """
    pathways = load_pathway_annotations()

    assoc_df = assoc_df.copy()
    # Ensure minimal expected columns
    if "feature" not in assoc_df.columns and "name" in assoc_df.columns:
        assoc_df["feature"] = assoc_df["name"]
    if "name" not in assoc_df.columns and "feature" in assoc_df.columns:
        assoc_df["name"] = assoc_df["feature"]
    if "dataset" not in assoc_df.columns:
        assoc_df["dataset"] = "unknown"
    if "p_value" not in assoc_df.columns:
        assoc_df["p_value"] = 1.0
    if "effect_size" not in assoc_df.columns:
        assoc_df["effect_size"] = 0.0

    # Calculate enhanced scores
    assoc_df["assoc_score"] = assoc_df.apply(
        lambda row: enhanced_ranking_score(row, pathways), axis=1
    )

    # Canonical schema and sort best-first
    out = pd.DataFrame(
        {
            "name": assoc_df["name"].astype(str),
            "layer": "transcriptomic",
            "type": "gene",
            "assoc_score": pd.to_numeric(
                assoc_df["assoc_score"], errors="coerce"
            ).fillna(0.0),
            "p_value": pd.to_numeric(assoc_df["p_value"], errors="coerce").fillna(1.0),
            "effect_size": pd.to_numeric(
                assoc_df["effect_size"], errors="coerce"
            ).fillna(0.0),
            "provenance": assoc_df["dataset"].astype(str),
        }
    )
    out = out.sort_values(
        ["assoc_score", "effect_size"], ascending=[False, False]
    ).reset_index(drop=True)
    return out


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame(
        {
            "feature": ["HAVCR1", "LCN2", "RANDOM_GENE", "CCL2"],
            "effect_size": [0.8, 0.7, 0.9, 0.6],
            "p_value": [0.001, 0.002, 0.0001, 0.005],
        }
    )

    ranked = rank_with_kg_features(test_data)
    print("Enhanced Ranking Results:")
    print(ranked[["feature", "effect_size", "enhanced_score"]].to_string(index=False))
