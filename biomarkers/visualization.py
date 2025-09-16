"""
Causal Graph Visualization for Biomarker Discovery
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import seaborn as sns
from biomarkers.causal_scoring import CausalBiomarkerScore
from learn.causal_discovery import CausalGraph, CausalEdge


class CausalGraphVisualizer:
    """
    Visualization utilities for causal biomarker discovery results
    """

    def __init__(self):
        plt.style.use("default")
        sns.set_palette("husl")

    def visualize_causal_graph(
        self,
        causal_graph: CausalGraph,
        networkx_graph: nx.DiGraph,
        min_confidence: float = 0.5,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Visualize the discovered causal graph
        """

        # Filter edges by confidence
        filtered_edges = [
            e for e in causal_graph.edges if e.confidence >= min_confidence
        ]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Network graph
        self._plot_network_graph(networkx_graph, filtered_edges, ax1, min_confidence)

        # Right plot: Edge confidence distribution
        self._plot_confidence_distribution(causal_graph.edges, ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🎨 Causal graph visualization saved to: {save_path}")

        plt.show()

    def _plot_network_graph(
        self,
        networkx_graph: nx.DiGraph,
        edges: List[CausalEdge],
        ax,
        min_confidence: float,
    ):
        """Plot the network graph structure"""

        # Create subgraph with filtered edges
        G = nx.DiGraph()
        for edge in edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.strength,
                confidence=edge.confidence,
                method=edge.method,
            )

        if len(G.nodes()) == 0:
            ax.text(
                0.5,
                0.5,
                f"No edges with confidence ≥ {min_confidence}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Causal Network Graph")
            return

        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Node colors based on type
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if "outcome" in node.lower() or "disease" in node.lower():
                node_colors.append("red")
                node_sizes.append(1500)
            elif any(marker in node.lower() for marker in ["crp", "inflammatory"]):
                node_colors.append("orange")
                node_sizes.append(1200)
            elif any(marker in node.lower() for marker in ["enzyme", "protein"]):
                node_colors.append("green")
                node_sizes.append(1200)
            elif any(marker in node.lower() for marker in ["metabolite", "compound"]):
                node_colors.append("blue")
                node_sizes.append(1200)
            else:
                node_colors.append("gray")
                node_sizes.append(1000)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
        )

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

        # Draw edges with thickness based on confidence
        for edge in G.edges(data=True):
            confidence = edge[2]["confidence"]
            width = max(0.5, confidence * 3)
            alpha = max(0.3, confidence)

            # Color based on method
            method = edge[2]["method"]
            if "Integrated" in method:
                color = "purple"
            elif "NOTEARS" in method:
                color = "blue"
            elif "Mendelian" in method:
                color = "red"
            else:
                color = "gray"

            nx.draw_networkx_edges(
                G,
                pos,
                [(edge[0], edge[1])],
                width=width,
                alpha=alpha,
                edge_color=color,
                arrows=True,
                arrowsize=20,
                ax=ax,
            )

        ax.set_title(f"Causal Network Graph\n(min confidence: {min_confidence})")
        ax.axis("off")

        # Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Clinical Outcome",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=10,
                label="Inflammatory",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="Protein/Enzyme",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Metabolite",
            ),
            Line2D([0], [0], color="purple", linewidth=2, label="Integrated Evidence"),
            Line2D([0], [0], color="blue", linewidth=2, label="NOTEARS"),
            Line2D([0], [0], color="red", linewidth=2, label="Mendelian Randomization"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    def _plot_confidence_distribution(self, edges: List[CausalEdge], ax):
        """Plot distribution of edge confidences"""

        confidences = [edge.confidence for edge in edges]
        methods = [edge.method for edge in edges]

        # Create confidence histogram by method
        method_colors = {
            "NOTEARS": "blue",
            "PC-MCI": "green",
            "Mendelian Randomization": "red",
            "Integrated": "purple",
        }

        # Group by method
        method_confidences = {}
        for conf, method in zip(confidences, methods):
            # Extract base method name
            base_method = "Integrated" if "Integrated" in method else method
            if base_method not in method_confidences:
                method_confidences[base_method] = []
            method_confidences[base_method].append(conf)

        # Plot histograms
        bins = np.linspace(0, 1, 21)
        bottom = np.zeros(len(bins) - 1)

        for method, conf_list in method_confidences.items():
            color = method_colors.get(method, "gray")
            hist, _ = np.histogram(conf_list, bins=bins)
            ax.bar(
                bins[:-1],
                hist,
                width=0.04,
                bottom=bottom,
                label=f"{method} (n={len(conf_list)})",
                color=color,
                alpha=0.7,
            )
            bottom += hist

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Number of Edges")
        ax.set_title("Edge Confidence Distribution\nby Discovery Method")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def visualize_biomarker_scores(
        self,
        scored_biomarkers: List[CausalBiomarkerScore],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> None:
        """
        Visualize biomarker scoring results
        """

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Integrated scores ranking
        self._plot_biomarker_ranking(scored_biomarkers, ax1)

        # Plot 2: Causal vs Traditional evidence
        self._plot_causal_vs_traditional(scored_biomarkers, ax2)

        # Plot 3: Evidence tiers distribution
        self._plot_evidence_tiers(scored_biomarkers, ax3)

        # Plot 4: Score components breakdown
        self._plot_score_components(scored_biomarkers, ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🎨 Biomarker scores visualization saved to: {save_path}")

        plt.show()

    def _plot_biomarker_ranking(
        self, scored_biomarkers: List[CausalBiomarkerScore], ax
    ):
        """Plot biomarker ranking by integrated score"""

        names = [score.name for score in scored_biomarkers[:10]]  # Top 10
        scores = [score.integrated_score for score in scored_biomarkers[:10]]
        tiers = [score.evidence_tier for score in scored_biomarkers[:10]]

        # Color by evidence tier
        tier_colors = {1: "darkgreen", 2: "green", 3: "orange", 4: "red", 5: "darkred"}
        colors = [tier_colors[tier] for tier in tiers]

        bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Integrated Score")
        ax.set_title("Top Biomarkers by Integrated Score")
        ax.grid(True, alpha=0.3)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=8,
            )

        # Legend for evidence tiers
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=tier_colors[i], label=f"Tier {i}")
            for i in sorted(tier_colors.keys())
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    def _plot_causal_vs_traditional(
        self, scored_biomarkers: List[CausalBiomarkerScore], ax
    ):
        """Plot causal confidence vs traditional statistical measures"""

        causal_conf = [score.causal_confidence for score in scored_biomarkers]
        p_values = [-np.log10(score.p_value + 1e-10) for score in scored_biomarkers]
        names = [score.name for score in scored_biomarkers]
        tiers = [score.evidence_tier for score in scored_biomarkers]

        # Color by evidence tier
        tier_colors = {1: "darkgreen", 2: "green", 3: "orange", 4: "red", 5: "darkred"}
        colors = [tier_colors[tier] for tier in tiers]

        scatter = ax.scatter(causal_conf, p_values, c=colors, s=100, alpha=0.8)

        # Add biomarker labels
        for i, name in enumerate(names):
            ax.annotate(
                name,
                (causal_conf[i], p_values[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        ax.set_xlabel("Causal Confidence")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("Causal Evidence vs Statistical Significance")
        ax.grid(True, alpha=0.3)

        # Add quadrant lines
        ax.axhline(
            -np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p=0.05"
        )
        ax.axvline(
            0.5, color="blue", linestyle="--", alpha=0.5, label="Causal threshold"
        )
        ax.legend()

    def _plot_evidence_tiers(self, scored_biomarkers: List[CausalBiomarkerScore], ax):
        """Plot distribution of evidence tiers"""

        tiers = [score.evidence_tier for score in scored_biomarkers]
        tier_counts = {i: tiers.count(i) for i in range(1, 6)}

        tier_labels = [
            "Strongest\n(Causal + Stats)",
            "Strong\n(Good Causal)",
            "Moderate\n(Some Causal)",
            "Weak\n(Stats Only)",
            "Insufficient\n(Poor Evidence)",
        ]

        colors = ["darkgreen", "green", "orange", "red", "darkred"]

        wedges, texts, autotexts = ax.pie(
            [tier_counts.get(i, 0) for i in range(1, 6)],
            labels=tier_labels,
            colors=colors,
            autopct="%1.0f",
            startangle=90,
        )
        ax.set_title("Evidence Tier Distribution")

    def _plot_score_components(self, scored_biomarkers: List[CausalBiomarkerScore], ax):
        """Plot breakdown of score components"""

        # Calculate component contributions for top biomarkers
        top_biomarkers = scored_biomarkers[:5]
        names = [score.name for score in top_biomarkers]

        # Approximate component scores (this would ideally come from the scorer)
        causal_components = [score.causal_confidence * 0.3 for score in top_biomarkers]
        effect_components = [
            min(score.effect_size, 1.0) * 0.15 for score in top_biomarkers
        ]
        sig_components = [
            max(0, 1 + np.log10(score.p_value + 1e-10) / 10) * 0.25
            for score in top_biomarkers
        ]
        pathway_components = [
            score.pathway_centrality * 0.1 for score in top_biomarkers
        ]
        clinical_components = [
            score.clinical_actionability * 0.05 for score in top_biomarkers
        ]
        other_components = [
            score.integrated_score - (causal + effect + sig + pathway + clinical)
            for score, causal, effect, sig, pathway, clinical in zip(
                top_biomarkers,
                causal_components,
                effect_components,
                sig_components,
                pathway_components,
                clinical_components,
            )
        ]

        # Stack the components
        width = 0.6
        x = np.arange(len(names))

        ax.bar(
            x, causal_components, width, label="Causal Evidence (30%)", color="purple"
        )
        ax.bar(
            x,
            sig_components,
            width,
            bottom=causal_components,
            label="Statistical Significance (25%)",
            color="blue",
        )
        ax.bar(
            x,
            effect_components,
            width,
            bottom=np.array(causal_components) + np.array(sig_components),
            label="Effect Size (15%)",
            color="green",
        )
        ax.bar(
            x,
            pathway_components,
            width,
            bottom=np.array(causal_components)
            + np.array(sig_components)
            + np.array(effect_components),
            label="Pathway Centrality (10%)",
            color="orange",
        )
        ax.bar(
            x,
            clinical_components,
            width,
            bottom=np.array(causal_components)
            + np.array(sig_components)
            + np.array(effect_components)
            + np.array(pathway_components),
            label="Clinical Actionability (5%)",
            color="red",
        )
        ax.bar(
            x,
            other_components,
            width,
            bottom=np.array(causal_components)
            + np.array(sig_components)
            + np.array(effect_components)
            + np.array(pathway_components)
            + np.array(clinical_components),
            label="Other (15%)",
            color="gray",
        )

        ax.set_xlabel("Biomarkers")
        ax.set_ylabel("Score Components")
        ax.set_title("Integrated Score Breakdown (Top 5 Biomarkers)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_biomarker_dashboard(
        self,
        scored_biomarkers: List[CausalBiomarkerScore],
        causal_graph: CausalGraph,
        networkx_graph: nx.DiGraph,
        output_dir: Path,
    ) -> None:
        """
        Create a comprehensive dashboard with all visualizations
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        print("🎨 Creating Causal Biomarker Discovery Dashboard...")

        # 1. Causal graph visualization
        self.visualize_causal_graph(
            causal_graph,
            networkx_graph,
            min_confidence=0.5,
            save_path=output_dir / "causal_graph.png",
        )

        # 2. Biomarker scores visualization
        self.visualize_biomarker_scores(
            scored_biomarkers, save_path=output_dir / "biomarker_scores.png"
        )

        # 3. High-confidence edges only
        self.visualize_causal_graph(
            causal_graph,
            networkx_graph,
            min_confidence=0.8,
            save_path=output_dir / "high_confidence_graph.png",
        )

        # 4. Create summary HTML dashboard
        self._create_html_dashboard(scored_biomarkers, causal_graph, output_dir)

        print(f"✅ Dashboard created in: {output_dir}")

    def _create_html_dashboard(
        self,
        scored_biomarkers: List[CausalBiomarkerScore],
        causal_graph: CausalGraph,
        output_dir: Path,
    ) -> None:
        """Create HTML dashboard with embedded images"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Causal Biomarker Discovery Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics {{ display: flex; justify-content: space-around; }}
        .metric {{ text-align: center; padding: 10px; background-color: #ecf0f1; border-radius: 5px; }}
        .top-biomarkers {{ background-color: #e8f5e8; }}
        .causal-edges {{ background-color: #e8f4fd; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Causal Biomarker Discovery Dashboard</h1>
        <p>Comprehensive analysis integrating causal discovery with biomarker scoring</p>
    </div>
    
    <div class="section">
        <h2>📊 Summary Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <h3>{len(scored_biomarkers)}</h3>
                <p>Biomarkers Analyzed</p>
            </div>
            <div class="metric">
                <h3>{len(causal_graph.edges)}</h3>
                <p>Causal Relationships</p>
            </div>
            <div class="metric">
                <h3>{len([e for e in causal_graph.edges if e.confidence > 0.8])}</h3>
                <p>High-Confidence Edges</p>
            </div>
            <div class="metric">
                <h3>{len([s for s in scored_biomarkers if s.evidence_tier <= 2])}</h3>
                <p>Strong Evidence Biomarkers</p>
            </div>
        </div>
    </div>
    
    <div class="section top-biomarkers">
        <h2>🏆 Top Biomarkers</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Biomarker</th>
                <th>Integrated Score</th>
                <th>Causal Confidence</th>
                <th>Evidence Tier</th>
                <th>P-value</th>
            </tr>
"""

        for i, score in enumerate(scored_biomarkers[:10], 1):
            html_content += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{score.name}</strong></td>
                <td>{score.integrated_score:.3f}</td>
                <td>{score.causal_confidence:.3f}</td>
                <td>{score.evidence_tier}</td>
                <td>{score.p_value:.2e}</td>
            </tr>"""

        html_content += """
        </table>
    </div>
    
    <div class="section causal-edges">
        <h2>🔗 Strong Causal Relationships</h2>
        <table>
            <tr>
                <th>Source</th>
                <th>Target</th>
                <th>Strength</th>
                <th>Confidence</th>
                <th>Method</th>
            </tr>
"""

        strong_edges = [e for e in causal_graph.edges if e.confidence > 0.7][:10]
        for edge in strong_edges:
            html_content += f"""
            <tr>
                <td>{edge.source}</td>
                <td>{edge.target}</td>
                <td>{edge.strength:.3f}</td>
                <td>{edge.confidence:.3f}</td>
                <td>{edge.method}</td>
            </tr>"""

        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>🎨 Visualizations</h2>
        
        <h3>Causal Network Graph</h3>
        <img src="causal_graph.png" alt="Causal Graph">
        
        <h3>High-Confidence Relationships</h3>
        <img src="high_confidence_graph.png" alt="High Confidence Graph">
        
        <h3>Biomarker Scoring Analysis</h3>
        <img src="biomarker_scores.png" alt="Biomarker Scores">
    </div>
    
    <div class="section">
        <h2>🔬 Methodology</h2>
        <p><strong>Causal Discovery Methods:</strong></p>
        <ul>
            <li><strong>NOTEARS:</strong> Neural-oriented causal discovery for observational data</li>
            <li><strong>PC-MCI:</strong> Temporal causal discovery with conditional independence testing</li>
            <li><strong>Mendelian Randomization:</strong> Genetic instrument-based causal inference</li>
        </ul>
        
        <p><strong>Integrated Scoring Components:</strong></p>
        <ul>
            <li>Causal Evidence (30%): Confidence in causal relationships</li>
            <li>Statistical Significance (25%): P-value based scoring</li>
            <li>Effect Size (15%): Magnitude of biomarker association</li>
            <li>Pathway Centrality (10%): Position in causal network</li>
            <li>Clinical Actionability (5%): Therapeutic potential</li>
            <li>Other factors (15%): Association strength, temporal stability</li>
        </ul>
    </div>
    
</body>
</html>
"""

        dashboard_path = output_dir / "dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)

        print(f"📄 HTML dashboard created: {dashboard_path}")


def create_visualization_demo():
    """
    Demonstration of causal biomarker visualization capabilities
    """
    print("🎨 CAUSAL BIOMARKER VISUALIZATION DEMONSTRATION")
    print("=" * 60)

    # Import the scoring demo results
    from biomarkers.causal_scoring import create_causal_biomarker_demo

    print("🔄 Running causal biomarker analysis...")
    scored_biomarkers, scorer = create_causal_biomarker_demo()

    # Create visualizer
    visualizer = CausalGraphVisualizer()

    # Create comprehensive dashboard
    output_dir = Path("artifacts/causal_biomarker_dashboard")
    if scorer.causal_graph is not None and scorer.networkx_graph is not None:
        visualizer.create_biomarker_dashboard(
            scored_biomarkers=scored_biomarkers,
            causal_graph=scorer.causal_graph,
            networkx_graph=scorer.networkx_graph,
            output_dir=output_dir,
        )
    else:
        print("⚠️ No causal graph available for visualization")

    print("\n✅ Visualization Demo Complete!")
    print(f"📂 Dashboard available at: {output_dir}/dashboard.html")

    return visualizer, scored_biomarkers, scorer
