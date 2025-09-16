#!/usr/bin/env python3
"""
Create visualizations for biomarker benchmarking results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use("default")
sns.set_palette("husl")


def create_benchmarking_visualizations():
    """Create comprehensive benchmarking visualizations"""

    # Read data
    report_df = pd.read_csv("comprehensive_benchmarking_report.csv")
    comparisons_df = pd.read_csv("biomarker_comparisons.csv")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Performance comparison bar chart
    ax1 = plt.subplot(2, 3, 1)
    biomarkers = report_df["biomarker_name"].str.replace(
        "Novel Multi-modal AKI Predictor", "Novel AI\nBiomarker"
    )
    biomarkers = biomarkers.str.replace(
        "Neutrophil Gelatinase-Associated Lipocalin", "NGAL"
    )
    biomarkers = biomarkers.str.replace("RIFLE Criteria Score", "RIFLE")
    biomarkers = biomarkers.str.replace("Serum Creatinine", "Creatinine")

    colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
    bars = ax1.bar(biomarkers, report_df["auc"], color=colors, alpha=0.8)
    ax1.set_ylabel("ROC AUC")
    ax1.set_title("Biomarker Performance Comparison", fontweight="bold")
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for bar, auc in zip(bars, report_df["auc"]):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add literature AUC reference line where available
    for i, lit_auc in enumerate(report_df["literature_auc"]):
        if pd.notna(lit_auc):
            ax1.axhline(
                y=lit_auc,
                xmin=i / len(biomarkers),
                xmax=(i + 1) / len(biomarkers),
                color="red",
                linestyle="--",
                alpha=0.6,
                linewidth=2,
            )

    plt.xticks(rotation=45, ha="right")

    # 2. Cost-effectiveness analysis
    ax2 = plt.subplot(2, 3, 2)
    costs = report_df["implementation_cost"].fillna(0)
    performance_per_cost = report_df["auc"] / (
        costs + 1
    )  # Add 1 to avoid division by zero

    scatter = ax2.scatter(
        costs, report_df["auc"], s=performance_per_cost * 500, c=colors, alpha=0.7
    )
    ax2.set_xlabel("Implementation Cost ($)")
    ax2.set_ylabel("ROC AUC")
    ax2.set_title("Cost vs Performance", fontweight="bold")

    # Add biomarker labels
    for i, (cost, auc, name) in enumerate(zip(costs, report_df["auc"], biomarkers)):
        ax2.annotate(
            name,
            (cost, auc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    # 3. Statistical significance heatmap
    ax3 = plt.subplot(2, 3, 3)

    # Filter for ROC AUC bootstrap comparisons
    roc_comps = comparisons_df[
        (comparisons_df["metric_type"] == "roc_auc")
        & (comparisons_df["statistical_test"] == "bootstrap")
    ].copy()

    # Create significance matrix
    biomarker_ids = report_df["biomarker_id"].tolist()
    n_biomarkers = len(biomarker_ids)
    sig_matrix = np.zeros((n_biomarkers, n_biomarkers))

    for _, row in roc_comps.iterrows():
        i = biomarker_ids.index(row["biomarker_1_id"])
        j = biomarker_ids.index(row["biomarker_2_id"])
        if row["p_value"] < 0.05:
            sig_matrix[i, j] = 1 if row["difference"] > 0 else -1

    # Create short names for heatmap
    short_names = ["Novel AI", "Creatinine", "NGAL", "RIFLE"]

    sns.heatmap(
        sig_matrix,
        annot=True,
        cmap="RdBu_r",
        center=0,
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax3,
        cbar_kws={"label": "Significance"},
    )
    ax3.set_title("Statistical Significance\n(Row vs Column)", fontweight="bold")

    # 4. Cross-validation stability
    ax4 = plt.subplot(2, 3, 4)
    cv_mean = [0.748, 0.723, 0.649, 0.574]  # From demonstration output
    cv_std = [0.077, 0.087, 0.048, 0.056]

    bars = ax4.bar(
        range(len(biomarkers)), cv_mean, yerr=cv_std, color=colors, alpha=0.8, capsize=5
    )
    ax4.set_xticks(range(len(biomarkers)))
    ax4.set_xticklabels(biomarkers, rotation=45, ha="right")
    ax4.set_ylabel("Cross-Validation AUC")
    ax4.set_title("Model Stability (5-Fold CV)", fontweight="bold")
    ax4.set_ylim(0, 1)

    # 5. Clinical utility comparison
    ax5 = plt.subplot(2, 3, 5)

    # Calculate clinical utility metrics
    sensitivity = [0.70, 0.69, 0.58, 0.60]  # Estimated from demonstration
    specificity = [0.66, 0.63, 0.59, 0.52]  # Estimated from demonstration

    x_pos = np.arange(len(biomarkers))
    width = 0.35

    bars1 = ax5.bar(
        x_pos - width / 2,
        sensitivity,
        width,
        label="Sensitivity",
        color="lightcoral",
        alpha=0.8,
    )
    bars2 = ax5.bar(
        x_pos + width / 2,
        specificity,
        width,
        label="Specificity",
        color="lightblue",
        alpha=0.8,
    )

    ax5.set_xlabel("Biomarker")
    ax5.set_ylabel("Performance")
    ax5.set_title("Clinical Utility Metrics", fontweight="bold")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(biomarkers, rotation=45, ha="right")
    ax5.legend()
    ax5.set_ylim(0, 1)

    # 6. Ranking summary
    ax6 = plt.subplot(2, 3, 6)

    # Create ranking visualization
    rankings = [1, 2, 3, 4]  # From demonstration output
    colors_rank = ["gold", "silver", "#CD7F32", "gray"]  # Gold, silver, bronze, gray

    bars = ax6.barh(
        range(len(biomarkers)), [5 - r for r in rankings], color=colors_rank, alpha=0.8
    )
    ax6.set_yticks(range(len(biomarkers)))
    ax6.set_yticklabels(biomarkers)
    ax6.set_xlabel("Ranking Score (Higher = Better)")
    ax6.set_title("Overall Biomarker Ranking", fontweight="bold")

    # Add ranking numbers
    for i, (bar, rank) in enumerate(zip(bars, rankings)):
        width = bar.get_width()
        ax6.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"#{rank}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig("biomarker_benchmarking_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Benchmarking visualizations created!")


if __name__ == "__main__":
    create_benchmarking_visualizations()
