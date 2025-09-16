#!/usr/bin/env python3
"""
Interactive 3D Competitive Analysis with Plotly
Creates rotatable, zoomable 3D visualization viewable in any web browser
"""

import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import numpy as np


def create_interactive_3d_analysis():
    """Create interactive 3D competitive analysis using Plotly"""

    # Import our analysis results
    import sys

    sys.path.append(".")
    from presentation.application_competitive_analysis import (
        calculate_application_scores,
    )

    results, scorer = calculate_application_scores()

    # Prepare data for plotting
    plot_data = []
    for company, data in results.items():
        plot_data.append(
            {
                "Company": company,
                "Discovery_Capability": data["discovery_capability"],
                "Clinical_Impact": data["clinical_impact"],
                "Federated_Personalization": data["individual_scores"][
                    "federated_personalization"
                ],
                "Type": (
                    "Target"
                    if "Year 7" in company
                    else ("Our Platform" if "Our Platform" in company else "Competitor")
                ),
            }
        )

    df = pd.DataFrame(plot_data)

    # Define colors and sizes
    color_map = {
        "Our Platform": "#2e7d32",
        "Target": "#4caf50",
        "Competitor": "#757575",
    }

    size_map = {"Our Platform": 15, "Target": 20, "Competitor": 12}

    symbol_map = {"Our Platform": "square", "Target": "diamond", "Competitor": "circle"}

    df["Color"] = df["Type"].map(color_map)
    df["Size"] = df["Type"].map(size_map)
    df["Symbol"] = df["Type"].map(symbol_map)

    # Create the 3D scatter plot
    fig = go.Figure()

    # Add trajectory line first (so it appears behind points)
    our_current = df[df["Company"] == "Our Platform"].iloc[0]
    our_target = df[df["Company"] == "Our Platform (Year 7 Target)"].iloc[0]

    fig.add_trace(
        go.Scatter3d(
            x=[our_current["Discovery_Capability"], our_target["Discovery_Capability"]],
            y=[our_current["Clinical_Impact"], our_target["Clinical_Impact"]],
            z=[
                our_current["Federated_Personalization"],
                our_target["Federated_Personalization"],
            ],
            mode="lines",
            line=dict(color="#2e7d32", width=8),
            name="Development Trajectory",
            showlegend=True,
        )
    )

    # Add points for each company type
    for company_type in df["Type"].unique():
        subset = df[df["Type"] == company_type]

        fig.add_trace(
            go.Scatter3d(
                x=subset["Discovery_Capability"],
                y=subset["Clinical_Impact"],
                z=subset["Federated_Personalization"],
                mode="markers",
                marker=dict(
                    size=subset["Size"],
                    color=subset["Color"],
                    symbol=subset["Symbol"].iloc[0],
                    line=dict(width=2, color="black"),
                    opacity=0.9,
                ),
                text=subset["Company"],
                textposition="top center",
                name=company_type.replace("_", " ").title(),
                hovertemplate="<b>%{text}</b><br>"
                + "Discovery: %{x:.1f}/10<br>"
                + "Clinical Impact: %{y:.1f}/10<br>"
                + "Federated Personalization: %{z:.1f}/10<br>"
                + "<extra></extra>",
            )
        )

    # Add reference planes to show competitive quadrants
    # Discovery-Clinical plane at z=5
    xx, yy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    zz = np.ones_like(xx) * 5

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale=[[0, "rgba(200,200,200,0.1)"], [1, "rgba(200,200,200,0.1)"]],
            showscale=False,
            name="Mid-level Federated Capability (5/10)",
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "Interactive 3D Competitive Analysis<br><sub>Rotate, zoom, and explore the competitive landscape</sub>",
            "x": 0.5,
            "font": {"size": 18},
        },
        scene=dict(
            xaxis_title="Biomarker Discovery Capability",
            yaxis_title="Clinical Impact Capability",
            zaxis_title="Federated Personalization Capability",
            xaxis=dict(range=[0, 10], dtick=2),
            yaxis=dict(range=[0, 10], dtick=2),
            zaxis=dict(range=[0, 10], dtick=2),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Nice initial viewing angle
            aspectratio=dict(x=1, y=1, z=0.8),
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="<b>Key Insights:</b><br>"
                + "• Our platform targets unoccupied space (high Z-axis)<br>"
                + "• No competitor exceeds 4.5/10 in Federated Personalization<br>"
                + "• 55% market opportunity in federated capabilities<br>"
                + "• Network effects create sustainable competitive moat",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10),
            )
        ],
    )

    return fig


def create_interactive_capabilities_radar():
    """Create interactive radar chart comparing all capabilities"""

    import sys

    sys.path.append(".")
    from presentation.application_competitive_analysis import (
        calculate_application_scores,
    )

    results, scorer = calculate_application_scores()

    # Prepare data
    capabilities = list(scorer.capability_weights.keys())
    capability_labels = [
        "Disease Coverage",
        "Biomarker Discovery",
        "Clinical Translation",
        "Real-world Deployment",
        "Evidence Generation",
        "Federated Personalization",
    ]

    fig = go.Figure()

    # Add traces for key companies
    key_companies = [
        "Our Platform",
        "Our Platform (Year 7 Target)",
        "Tempus Labs",
        "Foundation Medicine",
    ]

    colors = ["#2e7d32", "#4caf50", "#ff9800", "#d32f2f"]

    for i, company in enumerate(key_companies):
        if company in results:
            scores = [
                results[company]["individual_scores"][cap] for cap in capabilities
            ]
            scores_plot = scores + [scores[0]]  # Close the polygon

            fig.add_trace(
                go.Scatterpolar(
                    r=scores_plot,
                    theta=capability_labels + [capability_labels[0]],
                    fill="toself",
                    fillcolor=f"rgba{tuple(list(int(colors[i][j:j+2], 16) for j in (1, 3, 5)) + [0.1])}",
                    line=dict(color=colors[i], width=3),
                    name=company,
                    hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}/10<extra></extra>",
                )
            )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                ticktext=["2", "4", "6", "8", "10"],
            )
        ),
        title={
            "text": "Interactive Capability Assessment<br><sub>Hover for details, toggle companies in legend</sub>",
            "x": 0.5,
            "font": {"size": 18},
        },
        width=800,
        height=800,
        showlegend=True,
    )

    return fig


def main():
    """Generate interactive visualizations"""

    print("🌐 Creating interactive 3D competitive analysis...")

    # Create 3D plot
    fig_3d = create_interactive_3d_analysis()

    # Create radar plot
    fig_radar = create_interactive_capabilities_radar()

    # Save as HTML files
    output_dir = Path("presentation/figures")
    output_dir.mkdir(exist_ok=True)

    # Save 3D plot
    fig_3d.write_html(
        output_dir / "interactive_3d_competitive_analysis.html",
        config={"displayModeBar": True, "displaylogo": False},
    )

    # Save radar plot
    fig_radar.write_html(
        output_dir / "interactive_capabilities_radar.html",
        config={"displayModeBar": True, "displaylogo": False},
    )

    print("✅ Interactive visualizations created:")
    print("  📊 interactive_3d_competitive_analysis.html")
    print("  📡 interactive_capabilities_radar.html")
    print("\n🌐 Usage:")
    print("  • Open HTML files in any web browser")
    print("  • Rotate, zoom, and pan the 3D visualization")
    print("  • Hover for detailed information")
    print("  • Toggle companies in legend")
    print("  • Export as PNG/PDF using toolbar")


if __name__ == "__main__":
    main()
