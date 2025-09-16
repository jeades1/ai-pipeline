#!/usr/bin/env python3
"""
Comprehensive metrics dashboard including advanced biomarker discovery metrics.
"""
from __future__ import annotations
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def create_comprehensive_metrics_dashboard(
    base_metrics_file: Path, advanced_metrics_file: Path, output_file: Path
) -> None:
    """Create expanded metrics dashboard with advanced biomarker metrics."""

    # Load data
    with open(base_metrics_file) as f:
        base_metrics = json.load(f)

    with open(advanced_metrics_file) as f:
        advanced_metrics = json.load(f)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Standard Performance Metrics (top left)
    ax1 = plt.subplot(2, 4, 1)
    k_values = [5, 10, 20, 50, 100]
    precision_values = [base_metrics["metrics"][f"@{k}"]["precision"] for k in k_values]
    recall_values = [base_metrics["metrics"][f"@{k}"]["recall"] for k in k_values]

    ax1.plot(
        k_values, precision_values, "o-", label="Precision", linewidth=2, markersize=6
    )
    ax1.plot(k_values, recall_values, "s-", label="Recall", linewidth=2, markersize=6)
    ax1.set_xlabel("K (Top-K)")
    ax1.set_ylabel("Score")
    ax1.set_title("Standard Performance Metrics", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Clinical Relevance Radar (top middle-left)
    ax2 = plt.subplot(2, 4, 2, projection="polar")
    clinical_metrics = [
        advanced_metrics["clinical_relevance"]["clinical_enrichment_score"],
        advanced_metrics["clinical_relevance"]["druggability_score"],
        advanced_metrics["biological_coherence"]["literature_support_score"],
        advanced_metrics["translational_utility"]["assay_feasibility_score"],
        advanced_metrics["translational_utility"]["translational_readiness_score"],
    ]
    labels = [
        "Clinical\nEvidence",
        "Druggability",
        "Literature\nSupport",
        "Assay\nFeasibility",
        "Translational\nReadiness",
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    clinical_metrics += clinical_metrics[:1]  # Complete the circle
    angles += angles[:1]

    ax2.plot(angles, clinical_metrics, "o-", linewidth=2, markersize=6, color="#e74c3c")
    ax2.fill(angles, clinical_metrics, alpha=0.25, color="#e74c3c")
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title("Clinical Relevance Profile", fontweight="bold", pad=20)

    # 3. Discovery Efficiency (top middle-right)
    ax3 = plt.subplot(2, 4, 3)
    efficiency_labels = [
        "Cost per\nDiscovery\n($K)",
        "Time to\nValidation\n(days)",
        "Hit Rate\nConfidence\n(width)",
    ]
    efficiency_values = [
        advanced_metrics["discovery_efficiency"]["cost_per_discovery"]
        / 1000,  # Convert to $K
        advanced_metrics["discovery_efficiency"]["time_to_validation_days"],
        advanced_metrics["discovery_efficiency"]["hit_rate_confidence_95"]["upper"]
        - advanced_metrics["discovery_efficiency"]["hit_rate_confidence_95"]["lower"],
    ]

    # Normalize for comparison (lower is better for all)
    normalized_efficiency = [
        min(val / 200, 1.0) for val in efficiency_values  # Scale appropriately
    ]

    bars = ax3.bar(
        efficiency_labels,
        normalized_efficiency,
        color=["#3498db", "#f39c12", "#2ecc71"],
        alpha=0.7,
    )
    ax3.set_ylabel("Normalized Score (lower=better)")
    ax3.set_title("Discovery Efficiency", fontweight="bold")
    ax3.set_ylim(0, 1)

    # Add value labels
    for bar, val in zip(bars, efficiency_values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Biological Coherence (top right)
    ax4 = plt.subplot(2, 4, 4)
    bio_labels = ["Pathway\nCoherence", "Mechanism\nDiversity", "Literature\nSupport"]
    bio_values = [
        advanced_metrics["biological_coherence"]["pathway_coherence_score"],
        advanced_metrics["biological_coherence"]["mechanism_diversity_index"],
        advanced_metrics["biological_coherence"]["literature_support_score"],
    ]

    colors = ["#e67e22", "#9b59b6", "#1abc9c"]
    bars = ax4.bar(bio_labels, bio_values, color=colors, alpha=0.7)
    ax4.set_ylabel("Score")
    ax4.set_title("Biological Coherence", fontweight="bold")
    ax4.set_ylim(0, 1)

    # Add value labels
    for bar, val in zip(bars, bio_values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 5. Top Biomarkers with Classifications (bottom left, spans 2 columns)
    ax5 = plt.subplot(2, 4, (5, 6))
    top_hits = advanced_metrics["summary"]["top_20_hits"][:10]  # Top 10

    # Classify hits by type (simplified)
    hit_types = []
    for hit in top_hits:
        if hit in ["HMGCR", "LPL", "LCAT"]:
            hit_types.append("Enzyme")
        elif hit in ["PCSK9", "APOB", "ALB"]:
            hit_types.append("Secreted")
        elif hit in ["LDLR", "NPC1L1", "ABCA1"]:
            hit_types.append("Receptor/Transporter")
        elif hit in ["CETP", "APOE", "PLTP"]:
            hit_types.append("Transfer Protein")
        else:
            hit_types.append("Other")

    # Create horizontal bar chart
    y_pos = np.arange(len(top_hits))
    colors_map = {
        "Enzyme": "#e74c3c",
        "Secreted": "#3498db",
        "Receptor/Transporter": "#2ecc71",
        "Transfer Protein": "#f39c12",
        "Other": "#95a5a6",
    }
    colors = [colors_map[t] for t in hit_types]

    bars = ax5.barh(y_pos, [1] * len(top_hits), color=colors, alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(top_hits)
    ax5.set_xlabel("Rank (all normalized to 1.0)")
    ax5.set_title("Top 10 Biomarker Candidates by Type", fontweight="bold")

    # Legend for protein types
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors_map[t], alpha=0.7, label=t)
        for t in colors_map.keys()
    ]
    ax5.legend(handles=legend_elements, loc="lower right")

    # 6. Tissue-Chip Integration Roadmap (bottom middle-right)
    ax6 = plt.subplot(2, 4, 7)
    phases = [
        "Proof of\nConcept",
        "Closed\nLoop",
        "Multi-Scale\nValidation",
        "Clinical\nTranslation",
    ]
    timeline = [1, 3, 6, 12]  # months

    bars = ax6.bar(
        phases, timeline, color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"], alpha=0.7
    )
    ax6.set_ylabel("Timeline (months)")
    ax6.set_title("Tissue-Chip Integration Roadmap", fontweight="bold")

    # Add timeline labels
    for bar, months in zip(bars, timeline):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{months}mo",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 7. Success Metrics Scorecard (bottom right)
    ax7 = plt.subplot(2, 4, 8)
    success_metrics = [
        "Precision@20",
        "Clinical\nEnrichment",
        "Druggability",
        "Pathway\nCoherence",
        "Translational\nReadiness",
    ]
    current_scores = [
        base_metrics["metrics"]["@20"]["precision"],
        advanced_metrics["clinical_relevance"]["clinical_enrichment_score"],
        advanced_metrics["clinical_relevance"]["druggability_score"],
        advanced_metrics["biological_coherence"]["pathway_coherence_score"],
        advanced_metrics["translational_utility"]["translational_readiness_score"],
    ]
    target_scores = [0.3, 0.6, 0.7, 0.8, 0.6]  # Target benchmarks

    x_pos = np.arange(len(success_metrics))
    width = 0.35

    bars1 = ax7.bar(
        x_pos - width / 2,
        current_scores,
        width,
        label="Current",
        color="#2ecc71",
        alpha=0.7,
    )
    bars2 = ax7.bar(
        x_pos + width / 2,
        target_scores,
        width,
        label="Target",
        color="#e74c3c",
        alpha=0.7,
    )

    ax7.set_ylabel("Score")
    ax7.set_title("Success Metrics Scorecard", fontweight="bold")
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(success_metrics, rotation=45, ha="right")
    ax7.legend()
    ax7.set_ylim(0, 1)

    # Overall title
    fig.suptitle(
        "Comprehensive Biomarker Discovery Metrics Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    print(f"[comprehensive-dashboard] Generated dashboard: {output_file}")


if __name__ == "__main__":
    create_comprehensive_metrics_dashboard(
        Path("artifacts/bench/metrics_extended_cv_opt.json"),
        Path("artifacts/bench/advanced_metrics_report.json"),
        Path("artifacts/pitch/comprehensive_metrics_dashboard.png"),
    )
