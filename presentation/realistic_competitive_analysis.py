#!/usr/bin/env python3
"""
Realistic Competitive Capabilities Analysis
Conservative and honest assessment of platform capabilities vs established players
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def get_realistic_company_data():
    """Realistic capability data accounting for actual market position"""

    companies = {
        "Our Platform": {
            "data_types": 4,  # We do integrate 4 types
            "integration_depth": "intermediate",  # Honest: not fully advanced yet
            "processing_time_seconds": 30,  # Realistic: 30 seconds for complex analysis
            "privacy_features": {
                "federated_learning": True,  # We have this
                "differential_privacy": False,  # Prototype only
                "homomorphic_encryption": False,
                "secure_multiparty": False,  # Not implemented
                "governance_framework": False,  # Still developing
            },
            "ml_algorithms": [
                "deep_learning",
                "traditional_ml",
            ],  # Honest about current state
            "validation_rigor": 1,  # Proof-of-concept stage, not clinical validation
            "publications": 0,  # No peer-reviewed publications yet
            "api_availability": "basic_api",  # Working but not production-grade
            "ehr_integration": "manual",  # Not native integration yet
            "deployment_options": ["cloud"],  # Limited deployment options
            "architecture_type": "federated",  # This is our advantage
            "current_scale": 1,  # Pilot stage with limited deployment
            "performance_limits": "moderate",  # Scaling challenges exist
            "evidence_notes": "Proof-of-concept platform with federated learning advantage but limited validation",
            "maturity_stage": "prototype",
        },
        "Tempus Labs": {
            "data_types": 3,
            "integration_depth": "advanced",  # They have years of experience
            "processing_time_seconds": 180,
            "privacy_features": {
                "federated_learning": False,
                "differential_privacy": False,
                "homomorphic_encryption": False,
                "secure_multiparty": False,
                "governance_framework": True,  # Established governance
            },
            "ml_algorithms": ["deep_learning", "traditional_ml", "ensemble_methods"],
            "validation_rigor": 3,  # Clinical studies, FDA interactions
            "publications": 50,  # Strong publication record
            "api_availability": "rest_api",  # Production-grade APIs
            "ehr_integration": "native",  # Established EHR partnerships
            "deployment_options": ["cloud", "hybrid"],
            "architecture_type": "centralized_cloud",
            "current_scale": 200,  # Large customer base
            "performance_limits": "minimal",
            "evidence_notes": "Market leader with established clinical validation and partnerships",
            "maturity_stage": "commercial",
        },
        "Foundation Medicine": {
            "data_types": 2,
            "integration_depth": "advanced",  # Deep genomics expertise
            "processing_time_seconds": 432000,  # 5 days - but comprehensive analysis
            "privacy_features": {
                "federated_learning": False,
                "differential_privacy": False,
                "homomorphic_encryption": False,
                "secure_multiparty": False,
                "governance_framework": True,
            },
            "ml_algorithms": ["traditional_ml", "ensemble_methods"],
            "validation_rigor": 3,  # FDA-approved, rigorous
            "publications": 100,  # Extensive clinical evidence
            "api_availability": "rest_api",
            "ehr_integration": "native",  # Established clinical workflows
            "deployment_options": ["centralized"],
            "architecture_type": "centralized_local",
            "current_scale": 500,  # Very large scale
            "performance_limits": "minimal",
            "evidence_notes": "FDA-approved platform with extensive clinical validation",
            "maturity_stage": "commercial",
        },
        "Guardant Health": {
            "data_types": 1,
            "integration_depth": "advanced",  # Deep expertise in their domain
            "processing_time_seconds": 129600,
            "privacy_features": {
                "federated_learning": False,
                "differential_privacy": False,
                "homomorphic_encryption": False,
                "secure_multiparty": False,
                "governance_framework": True,
            },
            "ml_algorithms": ["traditional_ml", "ensemble_methods"],
            "validation_rigor": 3,  # FDA-approved assays
            "publications": 75,
            "api_availability": "rest_api",
            "ehr_integration": "native",
            "deployment_options": ["centralized", "cloud"],
            "architecture_type": "centralized_cloud",
            "current_scale": 300,
            "performance_limits": "minimal",
            "evidence_notes": "FDA-approved liquid biopsy leader with strong clinical evidence",
            "maturity_stage": "commercial",
        },
        "Veracyte": {
            "data_types": 2,
            "integration_depth": "intermediate",
            "processing_time_seconds": 259200,
            "privacy_features": {
                "federated_learning": False,
                "differential_privacy": False,
                "homomorphic_encryption": False,
                "secure_multiparty": False,
                "governance_framework": True,
            },
            "ml_algorithms": ["traditional_ml"],
            "validation_rigor": 2,
            "publications": 30,
            "api_availability": "basic_api",
            "ehr_integration": "plugin",
            "deployment_options": ["centralized"],
            "architecture_type": "centralized_local",
            "current_scale": 100,
            "performance_limits": "moderate",
            "evidence_notes": "Established player with clinical validation in specific areas",
            "maturity_stage": "commercial",
        },
        "10x Genomics": {
            "data_types": 2,
            "integration_depth": "advanced",  # Cutting-edge single-cell tech
            "processing_time_seconds": 86400,
            "privacy_features": {
                "federated_learning": False,
                "differential_privacy": False,
                "homomorphic_encryption": False,
                "secure_multiparty": False,
                "governance_framework": False,
            },
            "ml_algorithms": ["deep_learning", "traditional_ml"],
            "validation_rigor": 1,
            "publications": 150,  # Extensive research impact
            "api_availability": "basic_api",
            "ehr_integration": "manual",
            "deployment_options": ["on_premise"],
            "architecture_type": "centralized_local",
            "current_scale": 50,
            "performance_limits": "moderate",
            "evidence_notes": "Research-focused platform with cutting-edge technology but limited clinical application",
            "maturity_stage": "research",
        },
    }

    return companies


class RealisticCapabilityScorer:
    """More conservative scoring framework"""

    def __init__(self):
        self.capability_weights = {
            "data_integration": 0.20,
            "processing_speed": 0.15,
            "privacy_tech": 0.15,
            "discovery_methods": 0.20,
            "clinical_integration": 0.15,
            "scalability": 0.15,
        }

    def score_data_integration(
        self, company_name, data_types, integration_depth, maturity
    ):
        """More realistic scoring that accounts for implementation maturity"""
        base_score = min(data_types * 1.5, 6)  # Reduced multiplier

        depth_bonus = {"basic": 0, "intermediate": 1.5, "advanced": 3}

        # Maturity penalty for unproven platforms
        maturity_factor = {
            "prototype": 0.6,  # Significant penalty for prototype
            "research": 0.8,
            "commercial": 1.0,
        }

        raw_score = base_score + depth_bonus.get(integration_depth, 0)
        final_score = raw_score * maturity_factor.get(maturity, 1.0)

        return min(final_score, 10)

    def score_processing_speed(self, processing_time_seconds, maturity):
        """Realistic speed scoring with maturity consideration"""
        if processing_time_seconds <= 10:
            base_score = 8  # Very good, but not perfect for prototype
        elif processing_time_seconds <= 60:
            base_score = 7
        elif processing_time_seconds <= 3600:
            base_score = 6
        elif processing_time_seconds <= 86400:
            base_score = 4
        elif processing_time_seconds <= 604800:
            base_score = 2
        else:
            base_score = 1

        # Maturity factor - speed claims need validation
        maturity_factor = {
            "prototype": 0.7,  # Speed in prototype may not scale
            "research": 0.8,
            "commercial": 1.0,
        }

        return base_score * maturity_factor.get(maturity, 1.0)

    def score_privacy_tech(self, privacy_features, maturity):
        """Conservative privacy tech scoring"""
        score = 0

        if privacy_features.get("federated_learning", False):
            score += 3  # This is genuinely innovative
        if privacy_features.get("differential_privacy", False):
            score += 2
        if privacy_features.get("homomorphic_encryption", False):
            score += 2
        if privacy_features.get("secure_multiparty", False):
            score += 1
        if privacy_features.get("governance_framework", False):
            score += 2

        # Prototype penalty for unproven privacy tech
        maturity_factor = {
            "prototype": 0.6,  # Privacy tech needs real-world validation
            "research": 0.8,
            "commercial": 1.0,
        }

        return min(score * maturity_factor.get(maturity, 1.0), 10)

    def score_discovery_methods(
        self, ml_algorithms, validation_rigor, publications, maturity
    ):
        """More realistic ML scoring"""
        algorithm_score = 0

        if "deep_learning" in ml_algorithms:
            algorithm_score += 2  # Reduced from 3
        if "graph_neural_networks" in ml_algorithms:
            algorithm_score += 1  # Not implemented yet
        if "causal_discovery" in ml_algorithms:
            algorithm_score += 1  # Prototype level
        if "ensemble_methods" in ml_algorithms:
            algorithm_score += 1
        if "traditional_ml" in ml_algorithms:
            algorithm_score += 1

        validation_score = min(
            validation_rigor * 1.5, 4.5
        )  # Higher weight on validation
        publication_score = min(publications / 10, 3)  # Publications matter a lot

        # Maturity heavily affects discovery methods credibility
        maturity_factor = {
            "prototype": 0.5,  # Major penalty for unvalidated methods
            "research": 0.7,
            "commercial": 1.0,
        }

        raw_score = algorithm_score + validation_score + publication_score
        return min(raw_score * maturity_factor.get(maturity, 1.0), 10)

    def score_clinical_integration(
        self, api_availability, ehr_integration, deployment_options, maturity
    ):
        """Realistic clinical integration scoring"""
        score = 0

        api_scores = {"rest_api": 3, "basic_api": 2, "limited": 1}
        score += api_scores.get(api_availability, 0)

        ehr_scores = {"native": 4, "plugin": 2, "manual": 1}
        score += ehr_scores.get(ehr_integration, 0)

        deployment_score = len(deployment_options)
        score += min(deployment_score, 3)

        # Clinical integration requires proven track record
        maturity_factor = {
            "prototype": 0.4,  # Very hard to score high without proven integration
            "research": 0.6,
            "commercial": 1.0,
        }

        return min(score * maturity_factor.get(maturity, 1.0), 10)

    def score_scalability(
        self, architecture_type, current_scale, performance_limits, maturity
    ):
        """Realistic scalability assessment"""
        arch_scores = {
            "federated": 3,  # Potential advantage but unproven
            "distributed": 2,
            "centralized_cloud": 2,
            "centralized_local": 1,
        }

        scale_score = min(current_scale / 50, 4)  # More realistic scale expectations

        perf_scores = {"none": 2, "minimal": 1.5, "moderate": 1, "significant": 0}
        perf_score = perf_scores.get(performance_limits, 0)

        # Scalability claims need real-world proof
        maturity_factor = {
            "prototype": 0.5,  # Scalability is largely theoretical
            "research": 0.7,
            "commercial": 1.0,
        }

        raw_score = arch_scores.get(architecture_type, 0) + scale_score + perf_score
        return min(raw_score * maturity_factor.get(maturity, 1.0), 10)


def calculate_realistic_scores():
    """Calculate honest capability scores"""

    scorer = RealisticCapabilityScorer()
    companies = get_realistic_company_data()

    results = {}

    for company, data in companies.items():
        scores = {}
        maturity = data.get("maturity_stage", "commercial")

        scores["data_integration"] = scorer.score_data_integration(
            company, data["data_types"], data["integration_depth"], maturity
        )

        scores["processing_speed"] = scorer.score_processing_speed(
            data["processing_time_seconds"], maturity
        )

        scores["privacy_tech"] = scorer.score_privacy_tech(
            data["privacy_features"], maturity
        )

        scores["discovery_methods"] = scorer.score_discovery_methods(
            data["ml_algorithms"],
            data["validation_rigor"],
            data["publications"],
            maturity,
        )

        scores["clinical_integration"] = scorer.score_clinical_integration(
            data["api_availability"],
            data["ehr_integration"],
            data["deployment_options"],
            maturity,
        )

        scores["scalability"] = scorer.score_scalability(
            data["architecture_type"],
            data["current_scale"],
            data["performance_limits"],
            maturity,
        )

        # Calculate composite scores
        technical_innovation = (
            scores["data_integration"] * 0.35
            + scores["discovery_methods"] * 0.35
            + scores["privacy_tech"] * 0.30
        )

        operational_excellence = (
            scores["processing_speed"] * 0.25
            + scores["clinical_integration"]
            * 0.40  # Higher weight on proven integration
            + scores["scalability"] * 0.35
        )

        results[company] = {
            "individual_scores": scores,
            "technical_innovation": technical_innovation,
            "operational_excellence": operational_excellence,
            "evidence_notes": data["evidence_notes"],
            "maturity_stage": maturity,
        }

    return results, scorer


def create_realistic_positioning():
    """Create honest competitive positioning"""

    results, scorer = calculate_realistic_scores()

    fig, ax = plt.subplots(figsize=(14, 10))

    # More realistic company styling
    company_styles = {
        "Our Platform": {"color": "#2e7d32", "size": 200, "marker": "s", "alpha": 0.8},
        "Tempus Labs": {"color": "#ff9800", "size": 800, "marker": "o", "alpha": 0.9},
        "Foundation Medicine": {
            "color": "#d32f2f",
            "size": 900,
            "marker": "o",
            "alpha": 0.9,
        },
        "Guardant Health": {
            "color": "#1976d2",
            "size": 700,
            "marker": "o",
            "alpha": 0.9,
        },
        "Veracyte": {"color": "#7b1fa2", "size": 400, "marker": "o", "alpha": 0.9},
        "10x Genomics": {"color": "#795548", "size": 500, "marker": "^", "alpha": 0.9},
    }

    # Plot companies with realistic positioning
    for company, data in results.items():
        style = company_styles[company]
        ax.scatter(
            data["technical_innovation"],
            data["operational_excellence"],
            c=style["color"],
            s=style["size"],
            alpha=style["alpha"],
            marker=style["marker"],
            edgecolors="black",
            linewidth=2,
            label=f"{company} ({data['maturity_stage']})",
        )

        # Add labels
        offset_x = 0.2 if company != "Our Platform" else -0.5
        offset_y = 0.15
        ax.annotate(
            company,
            (
                data["technical_innovation"] + offset_x,
                data["operational_excellence"] + offset_y,
            ),
            fontsize=10,
            fontweight="bold",
        )

    # Add realistic quadrant lines
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=5, color="gray", linestyle="--", alpha=0.5)

    # More honest quadrant labels
    ax.text(
        7.5,
        8.5,
        "Market Leaders\n(Proven + Effective)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    ax.text(
        2.5,
        8.5,
        "Operational Excellence\n(Proven but Conservative)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    ax.text(
        7.5,
        2.5,
        "Innovation Potential\n(Promising but Unproven)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    ax.text(
        2.5,
        2.5,
        "Niche Players\n(Limited Scope)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
    )

    # Honest assessment of our position
    our_data = results["Our Platform"]
    ax.annotate(
        "Federated Learning\nAdvantage\n(Needs Validation)",
        xy=(our_data["technical_innovation"], our_data["operational_excellence"]),
        xytext=(
            our_data["technical_innovation"] + 1,
            our_data["operational_excellence"] - 1,
        ),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        fontsize=10,
        fontweight="bold",
        color="green",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    ax.set_xlabel(
        "Technical Innovation Capability\n(Adjusted for Maturity and Validation)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Operational Excellence Capability\n(Proven Track Record Weight)",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_title(
        "Realistic Competitive Analysis\nBiomarker Discovery Platforms (Maturity-Adjusted)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    # Add honest methodology note
    methodology_text = """REALISTIC SCORING:
• Prototype platforms penalized for unproven claims
• Clinical validation heavily weighted
• Publication record and FDA approval considered
• Current scale vs theoretical potential
• Maturity stage affects all capability scores

MATURITY STAGES:
• Prototype: 40-70% penalty (unproven)
• Research: 20-30% penalty (limited validation)  
• Commercial: Full score (market-proven)"""

    ax.text(
        0.02,
        0.98,
        methodology_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"
        ),
    )

    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    output_dir = Path("presentation/figures")
    plt.savefig(
        output_dir / "realistic_competitive_positioning.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_honest_radar():
    """Create honest radar chart showing realistic scores"""

    results, scorer = calculate_realistic_scores()

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

    capabilities = list(scorer.capability_weights.keys())
    capability_labels = [
        "Multi-omics\nIntegration",
        "Processing\nSpeed",
        "Privacy\nTechnology",
        "Discovery\nMethods",
        "Clinical\nIntegration",
        "Scalability",
    ]

    companies = list(results.keys())
    colors = ["#2e7d32", "#ff9800", "#d32f2f", "#1976d2", "#7b1fa2", "#795548"]
    line_styles = ["-", "--", "-.", ":", "--", "-."]
    line_widths = [2.5, 2, 2, 2, 2, 2]  # Less dramatic difference
    alphas = [0.9, 0.8, 0.8, 0.8, 0.8, 0.8]

    angles = np.linspace(0, 2 * np.pi, len(capabilities), endpoint=False)

    for idx, company in enumerate(companies):
        scores = [results[company]["individual_scores"][cap] for cap in capabilities]
        scores_plot = scores + [scores[0]]
        angles_plot = np.concatenate([angles, [angles[0]]])

        ax.plot(
            angles_plot,
            scores_plot,
            marker="o",
            linewidth=line_widths[idx],
            color=colors[idx],
            alpha=alphas[idx],
            linestyle=line_styles[idx],
            markersize=5,
            label=f"{company} ({results[company]['maturity_stage']})",
        )

        # Only fill for top performers to avoid clutter
        if idx < 2:  # Our platform and Tempus
            ax.fill(angles_plot, scores_plot, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles)
    ax.set_xticklabels(capability_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_title(
        "Honest Capability Assessment\nMaturity-Adjusted Competitive Analysis",
        fontsize=16,
        fontweight="bold",
        pad=30,
    )

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Honest assessment text
    assessment_text = """REALISTIC ASSESSMENT:
Our Platform Advantages:
• Federated learning architecture (innovative)
• Multi-omics integration potential
• Faster processing in prototype testing

Our Platform Challenges:
• Limited clinical validation (prototype stage)
• No peer-reviewed publications yet
• Unproven at scale
• Basic API and integration capabilities

Market Leaders' Strengths:
• Extensive clinical validation
• FDA approvals and partnerships
• Proven scalability and reliability
• Strong publication records"""

    ax.text(
        1.4,
        0.2,
        assessment_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
        verticalalignment="center",
    )

    plt.tight_layout()

    output_dir = Path("presentation/figures")
    plt.savefig(
        output_dir / "honest_capability_radar.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def export_realistic_analysis():
    """Export honest analysis results"""

    results, scorer = calculate_realistic_scores()

    # Create honest summary
    summary_data = []
    for company, data in results.items():
        summary_data.append(
            {
                "Company": company,
                "Maturity Stage": data["maturity_stage"],
                "Technical Innovation": f"{data['technical_innovation']:.1f}",
                "Operational Excellence": f"{data['operational_excellence']:.1f}",
                "Key Reality Check": data["evidence_notes"],
            }
        )

    df = pd.DataFrame(summary_data)

    output_dir = Path("presentation")
    df.to_csv(output_dir / "realistic_competitive_assessment.csv", index=False)

    return results


def main():
    """Generate realistic competitive analysis"""

    print("🔍 Generating REALISTIC competitive analysis...")
    print("  (Accounting for prototype stage and market realities)")

    print("  📊 Calculating honest capability scores...")
    results = calculate_realistic_scores()

    print("  🎯 Creating realistic positioning matrix...")
    create_realistic_positioning()

    print("  📡 Creating honest radar chart...")
    create_honest_radar()

    print("  📋 Exporting realistic assessment...")
    export_realistic_analysis()

    print("✅ Realistic competitive analysis complete")
    print("\n📋 Files generated:")
    print("  • realistic_competitive_positioning.png")
    print("  • honest_capability_radar.png")
    print("  • realistic_competitive_assessment.csv")

    # Print honest summary
    our_results = results[0]["Our Platform"]
    print("\n🎯 HONEST Assessment of Our Platform:")
    print(f"  • Technical Innovation: {our_results['technical_innovation']:.1f}/10")
    print(f"  • Operational Excellence: {our_results['operational_excellence']:.1f}/10")
    print("  • Reality: Promising prototype with federated learning advantage")
    print("  • Challenge: Needs clinical validation and proven scalability")
    print("  • Opportunity: First-mover in federated biomarker discovery")


if __name__ == "__main__":
    main()
