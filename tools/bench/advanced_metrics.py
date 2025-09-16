#!/usr/bin/env python3
"""
Advanced metrics for biomarker discovery evaluation beyond standard precision/recall.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class AdvancedMetrics:
    """Container for advanced biomarker discovery metrics."""

    # Clinical relevance metrics
    clinical_enrichment_score: float
    druggability_score: float
    actionability_index: float

    # Discovery efficiency metrics
    cost_per_discovery: float
    time_to_validation_estimate: float
    hit_rate_confidence: Tuple[float, float]  # Wilson CI

    # Biological coherence metrics
    pathway_coherence_score: float
    mechanism_diversity_index: float
    literature_support_score: float

    # Practical utility metrics
    assay_feasibility_score: float
    biomarker_specificity_index: float
    translational_readiness_score: float


def calculate_clinical_enrichment(
    hits: List[str], reference_db: str = "opentargets"
) -> float:
    """
    Calculate enrichment for clinically validated targets.
    Higher score = more hits have existing clinical evidence.
    """
    # TODO: Query OpenTargets/ChEMBL for clinical trial presence
    # For now, estimate based on known drug targets
    known_drug_targets = {
        "HMGCR",
        "LDLR",
        "PCSK9",
        "APOB",
        "CETP",
        "ABCA1",
        "APOE",  # Lipid targets
    }

    clinical_hits = len([h for h in hits if h in known_drug_targets])
    return clinical_hits / len(hits) if hits else 0.0


def calculate_druggability_score(hits: List[str]) -> float:
    """
    Estimate druggability based on protein class and structure.
    Uses simplified heuristics - could be enhanced with ChEMBL data.
    """
    # Simplified druggability classes
    highly_druggable = {"HMGCR", "PCSK9", "CETP"}  # Enzymes, secreted
    moderately_druggable = {"LDLR", "ABCA1", "APOE"}  # Receptors, transporters

    scores = []
    for hit in hits:
        if hit in highly_druggable:
            scores.append(1.0)
        elif hit in moderately_druggable:
            scores.append(0.7)
        else:
            scores.append(0.3)  # Default for unknown

    return float(np.mean(scores)) if scores else 0.0


def calculate_pathway_coherence(hits: List[str], kg_file: Path | None = None) -> float:
    """
    Measure how coherently hits cluster in biological pathways.
    Higher score = hits are functionally related (not random).
    """
    # Simplified - could use actual KG connectivity
    lipid_metabolism = {
        "HMGCR",
        "LDLR",
        "PCSK9",
        "APOB",
        "CETP",
        "ABCA1",
        "APOE",
        "LPL",
    }

    lipid_hits = len([h for h in hits if h in lipid_metabolism])
    coherence = lipid_hits / len(hits) if hits else 0.0

    # Bonus for having multiple hits in same pathway
    if lipid_hits >= 3:
        coherence *= 1.2

    return min(coherence, 1.0)


def calculate_mechanism_diversity(hits: List[str]) -> float:
    """
    Shannon diversity of mechanism classes among hits.
    Balance between coherence and coverage.
    """
    mechanisms = {
        "HMGCR": "enzyme",
        "PCSK9": "secreted_protein",
        "LDLR": "receptor",
        "APOB": "structural_protein",
        "CETP": "transfer_protein",
        "ABCA1": "transporter",
        "APOE": "lipoprotein",
        "LPL": "enzyme",
    }

    hit_mechanisms = [mechanisms.get(h, "unknown") for h in hits]
    unique_mechanisms = set(hit_mechanisms)

    if len(unique_mechanisms) <= 1:
        return 0.0

    # Shannon diversity
    counts = {m: hit_mechanisms.count(m) for m in unique_mechanisms}
    proportions = [c / len(hits) for c in counts.values()]
    shannon = -sum(p * np.log2(p) for p in proportions if p > 0)

    # Normalize by max possible diversity
    max_shannon = np.log2(len(unique_mechanisms))
    return shannon / max_shannon if max_shannon > 0 else 0.0


def calculate_literature_support(hits: List[str]) -> float:
    """
    Estimate literature support for disease-gene associations.
    Could be enhanced with PubMed/GWAS catalog queries.
    """
    # Simplified based on known cardiovascular literature
    strong_support = {"HMGCR", "LDLR", "PCSK9", "APOB", "APOE"}  # Well-established
    moderate_support = {"CETP", "ABCA1", "LPL"}  # Good evidence

    scores = []
    for hit in hits:
        if hit in strong_support:
            scores.append(1.0)
        elif hit in moderate_support:
            scores.append(0.7)
        else:
            scores.append(0.3)

    return float(np.mean(scores)) if scores else 0.0


def calculate_assay_feasibility(hits: List[str]) -> float:
    """
    Estimate how easily hits can be measured in clinical samples.
    Secreted > membrane > intracellular.
    """
    secreted = {"PCSK9", "APOB", "CETP", "LPL"}
    membrane = {"LDLR", "ABCA1"}
    intracellular = {"HMGCR", "APOE"}

    scores = []
    for hit in hits:
        if hit in secreted:
            scores.append(1.0)  # Easy - blood/urine
        elif hit in membrane:
            scores.append(0.6)  # Moderate - tissue/imaging
        elif hit in intracellular:
            scores.append(0.3)  # Hard - invasive sampling
        else:
            scores.append(0.5)  # Unknown

    return float(np.mean(scores)) if scores else 0.0


def calculate_translational_readiness(hits: List[str], precision_at_20: float) -> float:
    """
    Overall translational readiness combining multiple factors.
    """
    clinical_score = calculate_clinical_enrichment(hits)
    druggability = calculate_druggability_score(hits)
    feasibility = calculate_assay_feasibility(hits)

    # Weight by discovery performance
    performance_weight = min(precision_at_20 * 2, 1.0)  # Cap at 1.0

    readiness = (
        clinical_score * 0.4 + druggability * 0.3 + feasibility * 0.3
    ) * performance_weight
    return readiness


def calculate_cost_per_discovery(
    n_candidates: int, hits: int, cost_per_candidate: float = 1000.0
) -> float:
    """
    Estimate cost efficiency of discovery approach.
    """
    total_cost = n_candidates * cost_per_candidate
    return total_cost / hits if hits > 0 else float("inf")


def wilson_confidence_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Wilson score interval for hit rate confidence."""
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n

    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower_val = (center - margin) / denominator
    upper_val = (center + margin) / denominator
    lower = float(max(0.0, float(lower_val)))
    upper = float(min(1.0, float(upper_val)))

    return (lower, upper)


def compute_advanced_metrics(
    hits: List[str],
    total_candidates: int,
    precision_at_20: float,
    benchmark_size: int = 10,
) -> AdvancedMetrics:
    """Compute all advanced metrics for a biomarker discovery run."""

    n_hits = len(hits)

    return AdvancedMetrics(
        # Clinical relevance
        clinical_enrichment_score=calculate_clinical_enrichment(hits),
        druggability_score=calculate_druggability_score(hits),
        actionability_index=calculate_clinical_enrichment(hits)
        * calculate_druggability_score(hits),
        # Discovery efficiency
        cost_per_discovery=calculate_cost_per_discovery(total_candidates, n_hits),
        time_to_validation_estimate=30.0 / (precision_at_20 + 0.1),  # Days, heuristic
        hit_rate_confidence=wilson_confidence_interval(n_hits, benchmark_size),
        # Biological coherence
        pathway_coherence_score=calculate_pathway_coherence(hits),
        mechanism_diversity_index=calculate_mechanism_diversity(hits),
        literature_support_score=calculate_literature_support(hits),
        # Practical utility
        assay_feasibility_score=calculate_assay_feasibility(hits),
        biomarker_specificity_index=precision_at_20,  # Proxy for specificity
        translational_readiness_score=calculate_translational_readiness(
            hits, precision_at_20
        ),
    )


def generate_advanced_report(
    metrics_file: Path, ranked_file: Path, output_file: Path
) -> None:
    """Generate comprehensive advanced metrics report."""

    # Load existing metrics
    with open(metrics_file) as f:
        base_metrics = json.load(f)

    # Extract hits from top 20
    ranked_df = pd.read_csv(ranked_file, sep="\t")
    top_20_hits = ranked_df.head(20)["name"].tolist()

    # Get benchmark info
    total_candidates = len(ranked_df)
    precision_at_20 = base_metrics["metrics"]["@20"]["precision"]
    benchmark_size = base_metrics["total_relevant"]

    # Calculate advanced metrics
    advanced = compute_advanced_metrics(
        top_20_hits, total_candidates, precision_at_20, benchmark_size
    )

    # Create comprehensive report
    report = {
        "summary": {
            "evaluation_date": "2025-09-12",
            "total_candidates": total_candidates,
            "benchmark_size": benchmark_size,
            "precision_at_20": precision_at_20,
            "top_20_hits": top_20_hits,
        },
        "clinical_relevance": {
            "clinical_enrichment_score": advanced.clinical_enrichment_score,
            "interpretation": "Fraction of hits with existing clinical evidence",
            "druggability_score": advanced.druggability_score,
            "actionability_index": advanced.actionability_index,
            "benchmark": {
                "excellent": ">0.7",
                "good": "0.4-0.7",
                "needs_improvement": "<0.4",
            },
        },
        "discovery_efficiency": {
            "cost_per_discovery": advanced.cost_per_discovery,
            "time_to_validation_days": advanced.time_to_validation_estimate,
            "hit_rate_confidence_95": {
                "lower": advanced.hit_rate_confidence[0],
                "upper": advanced.hit_rate_confidence[1],
            },
            "interpretation": "Lower cost and time indicate more efficient discovery",
        },
        "biological_coherence": {
            "pathway_coherence_score": advanced.pathway_coherence_score,
            "mechanism_diversity_index": advanced.mechanism_diversity_index,
            "literature_support_score": advanced.literature_support_score,
            "interpretation": "High coherence + moderate diversity = biologically meaningful hits",
        },
        "translational_utility": {
            "assay_feasibility_score": advanced.assay_feasibility_score,
            "biomarker_specificity_index": advanced.biomarker_specificity_index,
            "translational_readiness_score": advanced.translational_readiness_score,
            "interpretation": "Readiness for clinical translation and validation",
        },
        "recommendations": _generate_recommendations(advanced, precision_at_20),
    }

    # Save report
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[advanced-metrics] Generated comprehensive report: {output_file}")


def _generate_recommendations(metrics: AdvancedMetrics, precision: float) -> List[str]:
    """Generate actionable recommendations based on metrics."""
    recommendations = []

    if precision > 0.3:
        recommendations.append(
            "Excellent precision - prioritize validation of top hits"
        )

    if metrics.clinical_enrichment_score > 0.6:
        recommendations.append(
            "Strong clinical relevance - consider partnership with clinical teams"
        )

    if metrics.druggability_score > 0.7:
        recommendations.append(
            "High druggability - explore therapeutic development opportunities"
        )

    if metrics.pathway_coherence_score > 0.8:
        recommendations.append(
            "Excellent pathway coherence - consider systems-level interventions"
        )

    if metrics.assay_feasibility_score > 0.7:
        recommendations.append(
            "High assay feasibility - prioritize biomarker validation studies"
        )

    if metrics.cost_per_discovery < 50000:
        recommendations.append(
            "Cost-effective discovery - scale to additional disease areas"
        )

    if not recommendations:
        recommendations.append(
            "Continue optimization - focus on improving precision and biological relevance"
        )

    return recommendations


if __name__ == "__main__":
    # Generate report for CV-optimized results
    generate_advanced_report(
        Path("artifacts/bench/metrics_extended_cv_opt.json"),
        Path("artifacts/ranked_cv_opt.tsv"),
        Path("artifacts/bench/advanced_metrics_report.json"),
    )
