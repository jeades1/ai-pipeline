#!/usr/bin/env python3
"""
Strategic Development Plan: Pathway to Industry Leadership
Generate roadmap visualization showing capability development over time
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def create_capability_evolution_roadmap():
    """Create visualization showing capability development over 7 years"""

    # Timeline data
    years = [0, 1, 2, 3, 4, 5, 6, 7]
    year_labels = [
        "Current",
        "Year 1",
        "Year 2",
        "Year 3",
        "Year 4",
        "Year 5",
        "Year 6",
        "Year 7",
    ]

    # Capability evolution trajectories
    capabilities = {
        "Disease Coverage": [2.1, 3.5, 5.0, 6.5, 7.5, 8.5, 9.0, 9.5],
        "Biomarker Discovery": [7.2, 7.5, 8.0, 8.5, 9.0, 9.2, 9.5, 9.8],
        "Clinical Translation": [0.4, 1.5, 3.0, 5.0, 7.0, 8.0, 8.5, 9.0],
        "Real-world Deployment": [0.6, 1.0, 2.0, 3.5, 5.5, 7.0, 8.0, 8.5],
        "Evidence Generation": [0.7, 2.0, 4.0, 6.0, 7.5, 8.0, 8.5, 9.0],
    }

    # Development phases
    phases = [
        {"name": "Research\nValidation", "start": 0, "end": 2, "color": "#ffcdd2"},
        {"name": "Clinical\nValidation", "start": 1, "end": 4, "color": "#fff3e0"},
        {"name": "Market\nEntry", "start": 3, "end": 6, "color": "#e8f5e8"},
        {"name": "Industry\nLeadership", "start": 5, "end": 7, "color": "#e3f2fd"},
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Top plot: Capability evolution over time
    colors = ["#2e7d32", "#1976d2", "#d32f2f", "#ff9800", "#7b1fa2"]

    for i, (capability, scores) in enumerate(capabilities.items()):
        ax1.plot(
            years,
            scores,
            marker="o",
            linewidth=3,
            markersize=6,
            color=colors[i],
            label=capability,
            alpha=0.9,
        )

        # Add endpoint annotations
        ax1.annotate(
            f"{scores[-1]:.1f}",
            xy=(years[-1], scores[-1]),
            xytext=(years[-1] + 0.1, scores[-1]),
            fontsize=10,
            fontweight="bold",
            color=colors[i],
        )

    # Add phase backgrounds
    for phase in phases:
        ax1.axvspan(phase["start"], phase["end"], alpha=0.2, color=phase["color"])
        ax1.text(
            (phase["start"] + phase["end"]) / 2,
            0.5,
            phase["name"],
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=phase["color"], alpha=0.8),
        )

    ax1.set_xlabel("Development Timeline", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Capability Score (0-10)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Strategic Capability Development Roadmap\nPathway to Industry Leadership",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(years)
    ax1.set_xticklabels(year_labels)
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Bottom plot: Competitive positioning evolution
    # Calculate composite scores
    discovery_scores = []
    clinical_scores = []

    for i in range(len(years)):
        discovery = (
            capabilities["Disease Coverage"][i] * 0.3
            + capabilities["Biomarker Discovery"][i] * 0.5
            + capabilities["Evidence Generation"][i] * 0.2
        )

        clinical = (
            capabilities["Clinical Translation"][i] * 0.6
            + capabilities["Real-world Deployment"][i] * 0.4
        )

        discovery_scores.append(discovery)
        clinical_scores.append(clinical)

    # Plot trajectory
    ax2.plot(
        discovery_scores,
        clinical_scores,
        marker="o",
        linewidth=4,
        markersize=8,
        color="#2e7d32",
        alpha=0.8,
        label="Our Platform Evolution",
    )

    # Add year labels to points
    for i, (x, y) in enumerate(zip(discovery_scores, clinical_scores)):
        if i in [0, 2, 4, 7]:  # Show labels for key years
            ax2.annotate(
                year_labels[i],
                xy=(x, y),
                xytext=(x + 0.2, y + 0.2),
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7),
            )

    # Add current competitor positions for reference
    competitors = {
        "Tempus Labs": (9.5, 10.0),
        "Foundation Medicine": (7.6, 10.0),
        "Guardant Health": (7.5, 10.0),
        "10x Genomics": (6.6, 0.7),
    }

    for company, (x, y) in competitors.items():
        ax2.scatter(x, y, s=150, alpha=0.6, color="gray", marker="s")
        ax2.annotate(
            company, xy=(x, y), xytext=(x - 0.5, y - 0.3), fontsize=8, alpha=0.7
        )

    # Add quadrant lines and labels
    ax2.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=5, color="gray", linestyle="--", alpha=0.5)

    # Quadrant labels
    ax2.text(
        8.5,
        8.5,
        "Industry\nLeaders",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    ax2.text(
        2.5,
        8.5,
        "Clinical\nAdopters",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    ax2.text(
        8.5,
        2.5,
        "Research\nPlatforms",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    ax2.text(
        2.5,
        2.5,
        "Early\nStage",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
    )

    ax2.set_xlabel("Biomarker Discovery Capability", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Clinical Impact Capability", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Competitive Position Evolution Over Time", fontsize=14, fontweight="bold"
    )
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path("presentation/figures")
    plt.savefig(
        output_dir / "industry_leadership_roadmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_development_milestones():
    """Create milestone timeline visualization"""

    fig, ax = plt.subplots(figsize=(16, 10))

    # Milestone data
    milestones = [
        {
            "year": 0.5,
            "category": "Clinical",
            "milestone": "First clinical partnership",
            "impact": "Foundation",
        },
        {
            "year": 1.0,
            "category": "Patients",
            "milestone": "500+ patients enrolled",
            "impact": "Validation",
        },
        {
            "year": 1.5,
            "category": "Research",
            "milestone": "First publication",
            "impact": "Credibility",
        },
        {
            "year": 2.0,
            "category": "Regulatory",
            "milestone": "FDA pre-submission",
            "impact": "Pathway",
        },
        {
            "year": 2.5,
            "category": "Platform",
            "milestone": "Clinical-grade system",
            "impact": "Commercial",
        },
        {
            "year": 3.0,
            "category": "Clinical",
            "milestone": "Clinical utility proven",
            "impact": "Market Entry",
        },
        {
            "year": 3.5,
            "category": "Regulatory",
            "milestone": "Breakthrough designation",
            "impact": "Acceleration",
        },
        {
            "year": 4.0,
            "category": "Commercial",
            "milestone": "First EHR integration",
            "impact": "Adoption",
        },
        {
            "year": 4.5,
            "category": "Regulatory",
            "milestone": "First FDA approval",
            "impact": "Validation",
        },
        {
            "year": 5.0,
            "category": "Market",
            "milestone": "Clinical guidelines",
            "impact": "Standard",
        },
        {
            "year": 6.0,
            "category": "Scale",
            "milestone": "50+ institutions",
            "impact": "Network",
        },
        {
            "year": 7.0,
            "category": "Leadership",
            "milestone": "Industry recognition",
            "impact": "Leadership",
        },
    ]

    # Category colors
    category_colors = {
        "Clinical": "#2e7d32",
        "Patients": "#1976d2",
        "Research": "#7b1fa2",
        "Regulatory": "#d32f2f",
        "Platform": "#ff9800",
        "Commercial": "#795548",
        "Market": "#00695c",
        "Scale": "#5d4037",
        "Leadership": "#1565c0",
    }

    # Plot milestones
    for i, milestone in enumerate(milestones):
        y_pos = i % 6  # Stagger vertically for readability

        # Plot milestone point
        ax.scatter(
            milestone["year"],
            y_pos,
            s=200,
            color=category_colors[milestone["category"]],
            alpha=0.8,
            edgecolors="black",
            linewidth=2,
        )

        # Add milestone text
        ax.annotate(
            f"{milestone['milestone']}\n({milestone['impact']})",
            xy=(milestone["year"], y_pos),
            xytext=(milestone["year"], y_pos + 0.3),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=category_colors[milestone["category"]],
                alpha=0.7,
            ),
        )

    # Add phase backgrounds
    phases = [
        {"name": "Research Validation", "start": 0, "end": 2, "color": "#ffebee"},
        {"name": "Clinical Validation", "start": 2, "end": 4, "color": "#fff3e0"},
        {"name": "Market Entry", "start": 4, "end": 6, "color": "#e8f5e8"},
        {"name": "Industry Leadership", "start": 6, "end": 7.5, "color": "#e3f2fd"},
    ]

    for phase in phases:
        ax.axvspan(phase["start"], phase["end"], alpha=0.1, color=phase["color"])
        ax.text(
            (phase["start"] + phase["end"]) / 2,
            -1,
            phase["name"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=phase["color"], alpha=0.8),
        )

    # Formatting
    ax.set_xlabel("Years from Present", fontsize=14, fontweight="bold")
    ax.set_ylabel("Milestone Categories", fontsize=14, fontweight="bold")
    ax.set_title(
        "Strategic Development Milestones\nCritical Achievements for Industry Leadership",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-2, 6)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)

    # Add legend
    legend_elements = [
        plt.scatter([], [], s=100, color=color, label=category, alpha=0.8)
        for category, color in category_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    output_dir = Path("presentation/figures")
    plt.savefig(output_dir / "development_milestones.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_investment_timeline():
    """Create funding and development timeline"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Investment timeline
    years = [0, 1, 2, 3, 4, 5, 6, 7]

    # Cumulative investment
    cumulative_investment = [0, 3, 8, 15, 25, 40, 55, 75]  # Millions
    annual_investment = [0, 3, 5, 7, 10, 15, 15, 20]

    # Revenue projection
    revenue = [0, 0, 0, 0.5, 2, 8, 20, 50]  # Millions

    # Top plot: Investment vs Revenue
    ax1.bar(
        years,
        annual_investment,
        alpha=0.7,
        color="#d32f2f",
        label="Annual Investment ($M)",
        width=0.4,
    )
    ax1.plot(
        years,
        revenue,
        marker="o",
        linewidth=3,
        color="#2e7d32",
        markersize=8,
        label="Annual Revenue ($M)",
    )
    ax1.fill_between(years, revenue, alpha=0.3, color="#2e7d32")

    ax1.set_xlabel("Years", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Millions USD", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Investment and Revenue Timeline\nPathway to Profitability",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Valuation progression
    valuations = [5, 15, 40, 100, 250, 500, 800, 1200]  # Millions

    ax2.plot(
        years,
        valuations,
        marker="s",
        linewidth=4,
        color="#1976d2",
        markersize=8,
        alpha=0.8,
    )
    ax2.fill_between(years, valuations, alpha=0.2, color="#1976d2")

    # Add valuation annotations
    for i, val in enumerate(valuations):
        if i % 2 == 0:  # Show every other year
            ax2.annotate(
                f"${val}M",
                xy=(years[i], val),
                xytext=(years[i], val + 50),
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            )

    ax2.set_xlabel("Years", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Valuation (Millions USD)", fontsize=12, fontweight="bold")
    ax2.set_title("Platform Valuation Progression", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()

    output_dir = Path("presentation/figures")
    plt.savefig(output_dir / "investment_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate strategic development visualizations"""

    print("🚀 Generating strategic development roadmap...")

    print("  📈 Creating capability evolution roadmap...")
    create_capability_evolution_roadmap()

    print("  🎯 Creating development milestones timeline...")
    create_development_milestones()

    print("  💰 Creating investment and valuation timeline...")
    create_investment_timeline()

    print("✅ Strategic development roadmap complete")
    print("\n📋 Files generated:")
    print("  • industry_leadership_roadmap.png")
    print("  • development_milestones.png")
    print("  • investment_timeline.png")

    print("\n🎯 Strategic Summary:")
    print("  • Current Position: Research platform with high potential")
    print("  • 7-year pathway: Research → Clinical → Market → Leadership")
    print("  • Total investment: $75M over 7 years")
    print("  • Target valuation: $1.2B at industry leadership")
    print("  • Revenue potential: $50M+ annually by year 7")


if __name__ == "__main__":
    main()
