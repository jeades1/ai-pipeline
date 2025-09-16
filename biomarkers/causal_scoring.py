"""
Enhanced Biomarker Scoring with Causal Discovery Integration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from learn.causal_discovery import CausalDiscoveryEngine, CausalGraph


@dataclass
class CausalBiomarkerScore:
    """
    Enhanced biomarker scoring with causal evidence integration
    """

    name: str
    layer: str  # genomic, transcriptomic, proteomic, metabolomic, clinical
    type: str  # secreted, enzymatic, structural, regulatory

    # Traditional scoring
    assoc_score: float  # Association strength (correlation, mutual info)
    effect_size: float  # Effect size measure
    p_value: float  # Statistical significance

    # Causal evidence (NEW)
    causal_strength: float  # Strength of causal relationship
    causal_confidence: float  # Confidence in causal relationship
    causal_evidence_level: int  # Evidence hierarchy (1-5)

    # Pathway context
    pathway_centrality: float  # Centrality in causal pathway

    # Clinical relevance
    clinical_actionability: float  # How actionable for treatment
    temporal_stability: float  # Stability over time

    # Integrated score
    integrated_score: float  # Final composite score
    evidence_tier: int  # Evidence tier (1=strongest, 5=weakest)

    # Optional fields with defaults
    causal_mechanism: Optional[str] = None  # Mechanism description
    upstream_biomarkers: Optional[List[str]] = None  # Causal upstream factors
    downstream_targets: Optional[List[str]] = None  # Causal downstream effects
    discovery_methods: Optional[List[str]] = (
        None  # Methods that discovered this relationship
    )
    datasets: Optional[List[str]] = None  # Source datasets

    def __post_init__(self):
        """Initialize optional list fields"""
        if self.upstream_biomarkers is None:
            self.upstream_biomarkers = []
        if self.downstream_targets is None:
            self.downstream_targets = []
        if self.discovery_methods is None:
            self.discovery_methods = []
        if self.datasets is None:
            self.datasets = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "name": self.name,
            "layer": self.layer,
            "type": self.type,
            "assoc_score": self.assoc_score,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "causal_strength": self.causal_strength,
            "causal_confidence": self.causal_confidence,
            "causal_evidence_level": self.causal_evidence_level,
            "causal_mechanism": self.causal_mechanism,
            "pathway_centrality": self.pathway_centrality,
            "upstream_biomarkers": self.upstream_biomarkers,
            "downstream_targets": self.downstream_targets,
            "clinical_actionability": self.clinical_actionability,
            "temporal_stability": self.temporal_stability,
            "integrated_score": self.integrated_score,
            "evidence_tier": self.evidence_tier,
            "discovery_methods": self.discovery_methods,
            "datasets": self.datasets,
        }


class CausalBiomarkerScorer:
    """
    Enhanced biomarker scoring system integrating causal discovery
    """

    def __init__(self, causal_engine: Optional[CausalDiscoveryEngine] = None):
        self.causal_engine = causal_engine or CausalDiscoveryEngine()
        self.causal_graph: Optional[CausalGraph] = None
        self.networkx_graph: Optional[nx.DiGraph] = None

    def discover_and_score_biomarkers(
        self,
        biomarker_data: pd.DataFrame,
        clinical_outcomes: pd.DataFrame,
        temporal_data: Optional[pd.DataFrame] = None,
        genetic_data: Optional[pd.DataFrame] = None,
        biomarker_metadata: Optional[pd.DataFrame] = None,
        outcome_column: str = "outcome",
    ) -> List[CausalBiomarkerScore]:
        """
        Discover causal relationships and score biomarkers comprehensively
        """
        print("🔍 Discovering causal relationships...")

        # Combine biomarker data with outcomes for causal discovery
        combined_data = self._prepare_combined_dataset(
            biomarker_data, clinical_outcomes, outcome_column
        )

        # Discover causal relationships
        self.causal_graph = self.causal_engine.discover_causal_relationships(
            data=combined_data,
            temporal_data=temporal_data,
            genetic_data=genetic_data,
            time_column="time" if temporal_data is not None else None,
        )

        # Convert to NetworkX for analysis
        self.networkx_graph = self.causal_engine.export_to_networkx(
            self.causal_graph, min_confidence=0.3
        )

        print(f"📊 Discovered {len(self.causal_graph.edges)} causal relationships")
        print(f"🎯 Analyzing {len(biomarker_data.columns)} biomarkers")

        # Score each biomarker
        biomarker_scores = []
        for biomarker in biomarker_data.columns:
            if biomarker == outcome_column:
                continue

            score = self._score_biomarker(
                biomarker,
                biomarker_data,
                clinical_outcomes,
                biomarker_metadata,
                outcome_column,
            )
            biomarker_scores.append(score)

        # Sort by integrated score
        biomarker_scores.sort(key=lambda x: x.integrated_score, reverse=True)

        print(f"✅ Scored {len(biomarker_scores)} biomarkers")
        return biomarker_scores

    def _prepare_combined_dataset(
        self,
        biomarker_data: pd.DataFrame,
        clinical_outcomes: pd.DataFrame,
        outcome_column: str,
    ) -> pd.DataFrame:
        """Prepare combined dataset for causal discovery"""
        # Align datasets by index (patient ID or time points)
        common_index = biomarker_data.index.intersection(clinical_outcomes.index)

        combined = biomarker_data.loc[common_index].copy()
        combined[outcome_column] = clinical_outcomes.loc[common_index, outcome_column]

        return combined

    def _score_biomarker(
        self,
        biomarker: str,
        biomarker_data: pd.DataFrame,
        clinical_outcomes: pd.DataFrame,
        biomarker_metadata: Optional[pd.DataFrame],
        outcome_column: str,
    ) -> CausalBiomarkerScore:
        """Score individual biomarker with causal evidence"""

        # Basic association metrics
        biomarker_values = biomarker_data[biomarker].dropna()
        outcome_values = clinical_outcomes.loc[biomarker_values.index, outcome_column]

        # Traditional association measures
        correlation = np.corrcoef(biomarker_values, outcome_values)[0, 1]
        assoc_score = abs(correlation)

        # Effect size (simplified calculation to avoid type issues)
        if outcome_values.nunique() == 2:
            # Simple mean difference as effect size for binary outcomes
            group0_mean = float(biomarker_values[outcome_values == 0].mean())
            group1_mean = float(biomarker_values[outcome_values == 1].mean())
            group0_std = float(biomarker_values[outcome_values == 0].std())
            group1_std = float(biomarker_values[outcome_values == 1].std())
            pooled_std = (group0_std + group1_std) / 2
            effect_size = abs(group1_mean - group0_mean) / (pooled_std + 1e-6)
        else:
            effect_size = abs(correlation)

        # P-value (simplified calculation)
        from scipy import stats

        try:
            t_stat = correlation * np.sqrt(
                (len(biomarker_values) - 2) / (1 - correlation**2 + 1e-6)
            )
            p_value = float(
                2 * (1 - stats.t.cdf(abs(t_stat), len(biomarker_values) - 2))
            )
        except:
            p_value = 0.5  # Default moderate p-value

        # Causal evidence from discovered graph
        causal_evidence = self._extract_causal_evidence(biomarker, outcome_column)

        # Pathway analysis
        pathway_metrics = self._analyze_pathway_context(biomarker, outcome_column)

        # Metadata-based features
        metadata = self._get_biomarker_metadata(biomarker, biomarker_metadata)

        # Clinical actionability (placeholder - should be based on druggability, etc.)
        clinical_actionability = self._assess_clinical_actionability(
            biomarker, metadata
        )

        # Temporal stability (placeholder - should analyze temporal data)
        temporal_stability = 0.8  # Default high stability

        # Integrated scoring
        integrated_score = self._calculate_integrated_score(
            assoc_score,
            effect_size,
            p_value,
            causal_evidence,
            pathway_metrics,
            clinical_actionability,
        )

        # Evidence tier based on causal evidence level and statistical significance
        evidence_tier = self._determine_evidence_tier(
            causal_evidence["evidence_level"], p_value, effect_size
        )

        return CausalBiomarkerScore(
            name=biomarker,
            layer=metadata.get("layer", "unknown"),
            type=metadata.get("type", "unknown"),
            assoc_score=float(assoc_score),
            effect_size=float(effect_size),
            p_value=float(p_value),
            causal_strength=causal_evidence["strength"],
            causal_confidence=causal_evidence["confidence"],
            causal_evidence_level=causal_evidence["evidence_level"],
            causal_mechanism=causal_evidence["mechanism"],
            pathway_centrality=pathway_metrics["centrality"],
            upstream_biomarkers=pathway_metrics["upstream"],
            downstream_targets=pathway_metrics["downstream"],
            clinical_actionability=clinical_actionability,
            temporal_stability=temporal_stability,
            integrated_score=integrated_score,
            evidence_tier=evidence_tier,
            discovery_methods=causal_evidence["methods"],
            datasets=[
                "biomarker_dataset",
                "clinical_dataset",
            ],  # Should track actual datasets
        )

    def _extract_causal_evidence(self, biomarker: str, outcome: str) -> Dict[str, Any]:
        """Extract causal evidence for biomarker from discovered graph"""
        if not self.causal_graph:
            return {
                "strength": 0.0,
                "confidence": 0.0,
                "evidence_level": 1,
                "mechanism": None,
                "methods": [],
            }

        # Find direct causal relationship
        direct_edge = None
        for edge in self.causal_graph.edges:
            if edge.source == biomarker and edge.target == outcome:
                direct_edge = edge
                break

        if direct_edge:
            return {
                "strength": direct_edge.strength,
                "confidence": direct_edge.confidence,
                "evidence_level": direct_edge.evidence_level,
                "mechanism": direct_edge.mechanism,
                "methods": [direct_edge.method],
            }

        # Check for indirect relationships through mediators
        if (
            self.networkx_graph
            and biomarker in self.networkx_graph
            and outcome in self.networkx_graph
        ):
            try:
                path = nx.shortest_path(self.networkx_graph, biomarker, outcome)
                if len(path) <= 3:  # Allow one mediator
                    # Find weakest link in chain
                    path_strengths = []
                    path_confidences = []
                    for i in range(len(path) - 1):
                        edge_data = self.networkx_graph[path[i]][path[i + 1]]
                        path_strengths.append(edge_data.get("weight", 0))
                        path_confidences.append(edge_data.get("confidence", 0))

                    return {
                        "strength": min(path_strengths),
                        "confidence": min(path_confidences),
                        "evidence_level": 2,  # Indirect evidence
                        "mechanism": (
                            f"Indirect via {' → '.join(path[1:-1])}"
                            if len(path) > 2
                            else "Direct"
                        ),
                        "methods": ["pathway_analysis"],
                    }
            except nx.NetworkXNoPath:
                pass

        # No causal relationship found
        return {
            "strength": 0.0,
            "confidence": 0.0,
            "evidence_level": 1,
            "mechanism": None,
            "methods": [],
        }

    def _analyze_pathway_context(self, biomarker: str, outcome: str) -> Dict[str, Any]:
        """Analyze biomarker's position in causal pathway network"""
        if not self.networkx_graph or biomarker not in self.networkx_graph:
            return {"centrality": 0.0, "upstream": [], "downstream": []}

        # Calculate centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.networkx_graph).get(
                biomarker, 0
            )
            # Simple degree count - count neighbors
            in_degree = len(list(self.networkx_graph.predecessors(biomarker)))
            out_degree = len(list(self.networkx_graph.successors(biomarker)))
            degree_count = in_degree + out_degree
            centrality = (
                betweenness + degree_count / max(len(self.networkx_graph), 1)
            ) / 2
        except:
            centrality = 0.0

        # Find upstream and downstream biomarkers
        upstream = list(self.networkx_graph.predecessors(biomarker))
        downstream = list(self.networkx_graph.successors(biomarker))

        return {
            "centrality": centrality,
            "upstream": upstream,
            "downstream": downstream,
        }

    def _get_biomarker_metadata(
        self, biomarker: str, metadata: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Get biomarker metadata (layer, type, etc.)"""
        if metadata is not None and biomarker in metadata.index:
            return dict(metadata.loc[biomarker])

        # Default metadata based on biomarker name patterns
        layer = "proteomic"  # Default assumption
        biomarker_type = "secreted"  # Default assumption

        # Simple pattern matching for layer detection
        if any(x in biomarker.lower() for x in ["mrna", "transcript", "gene"]):
            layer = "transcriptomic"
        elif any(x in biomarker.lower() for x in ["metabolite", "compound"]):
            layer = "metabolomic"
        elif any(x in biomarker.lower() for x in ["snp", "variant", "allele"]):
            layer = "genomic"
        elif any(x in biomarker.lower() for x in ["creatinine", "urea", "glucose"]):
            layer = "clinical"

        return {"layer": layer, "type": biomarker_type}

    def _assess_clinical_actionability(
        self, biomarker: str, metadata: Dict[str, str]
    ) -> float:
        """Assess clinical actionability of biomarker"""
        # Placeholder scoring - should be based on:
        # - Druggability databases
        # - Known therapeutic targets
        # - Biomarker accessibility (secreted vs intracellular)
        # - Regulatory approval status

        actionability = 0.5  # Default moderate actionability

        # Boost for secreted proteins (easier to measure)
        if metadata.get("type") == "secreted":
            actionability += 0.2

        # Boost for known clinical biomarkers
        clinical_markers = ["kim1", "ngal", "lcn2", "havcr1", "creatinine", "urea"]
        if any(marker in biomarker.lower() for marker in clinical_markers):
            actionability += 0.3

        return min(1.0, actionability)

    def _calculate_integrated_score(
        self,
        assoc_score: float,
        effect_size: float,
        p_value: float,
        causal_evidence: Dict[str, Any],
        pathway_metrics: Dict[str, Any],
        clinical_actionability: float,
    ) -> float:
        """Calculate integrated biomarker score"""

        # Statistical significance component (higher for lower p-value)
        sig_component = max(0, 1 - np.log10(p_value + 1e-10) / 10)

        # Effect size component
        effect_component = min(1.0, effect_size)

        # Association component
        assoc_component = min(1.0, assoc_score)

        # Causal evidence component (weighted by evidence level)
        causal_component = causal_evidence["confidence"] * (
            causal_evidence["evidence_level"] / 5.0
        )

        # Pathway importance component
        pathway_component = pathway_metrics["centrality"]

        # Clinical relevance component
        clinical_component = clinical_actionability

        # Weighted combination
        weights = {
            "statistical": 0.25,
            "effect": 0.15,
            "association": 0.15,
            "causal": 0.30,  # Highest weight for causal evidence
            "pathway": 0.10,
            "clinical": 0.05,
        }

        integrated_score = (
            weights["statistical"] * sig_component
            + weights["effect"] * effect_component
            + weights["association"] * assoc_component
            + weights["causal"] * causal_component
            + weights["pathway"] * pathway_component
            + weights["clinical"] * clinical_component
        )

        return float(integrated_score)

    def _determine_evidence_tier(
        self, causal_evidence_level: int, p_value: float, effect_size: float
    ) -> int:
        """Determine evidence tier based on multiple factors"""

        # Start with causal evidence level (1=strongest causal evidence)
        if causal_evidence_level >= 4 and p_value < 0.001 and effect_size > 0.5:
            return 1  # Strongest evidence
        elif causal_evidence_level >= 3 and p_value < 0.01 and effect_size > 0.3:
            return 2  # Strong evidence
        elif causal_evidence_level >= 2 and p_value < 0.05 and effect_size > 0.2:
            return 3  # Moderate evidence
        elif p_value < 0.05:
            return 4  # Weak evidence (statistical only)
        else:
            return 5  # Insufficient evidence

    def export_scored_biomarkers(
        self, scored_biomarkers: List[CausalBiomarkerScore], output_path: Path
    ) -> None:
        """Export scored biomarkers to various formats"""

        # Convert to DataFrame
        df = pd.DataFrame([score.to_dict() for score in scored_biomarkers])

        # Save as CSV
        csv_path = output_path / "causal_biomarker_scores.csv"
        df.to_csv(csv_path, index=False)

        # Save detailed JSON
        import json

        json_path = output_path / "causal_biomarker_scores.json"
        with open(json_path, "w") as f:
            json.dump([score.to_dict() for score in scored_biomarkers], f, indent=2)

        # Save summary report
        self._generate_summary_report(scored_biomarkers, output_path)

        print(
            f"📊 Exported {len(scored_biomarkers)} scored biomarkers to {output_path}"
        )

    def _generate_summary_report(
        self, scored_biomarkers: List[CausalBiomarkerScore], output_path: Path
    ) -> None:
        """Generate summary report of biomarker scoring results"""

        report_path = output_path / "causal_biomarker_summary.md"

        # Tier distribution
        tier_counts = {}
        for score in scored_biomarkers:
            tier_counts[score.evidence_tier] = (
                tier_counts.get(score.evidence_tier, 0) + 1
            )

        # Top biomarkers
        top_10 = scored_biomarkers[:10]

        # Causal relationships summary
        causal_biomarkers = [s for s in scored_biomarkers if s.causal_confidence > 0.5]

        report_content = f"""# Causal Biomarker Discovery Report

## Summary Statistics
- **Total Biomarkers Analyzed**: {len(scored_biomarkers)}
- **Biomarkers with Causal Evidence (>50% confidence)**: {len(causal_biomarkers)}
- **Average Integrated Score**: {np.mean([s.integrated_score for s in scored_biomarkers]):.3f}

## Evidence Tier Distribution
"""
        for tier in sorted(tier_counts.keys()):
            report_content += f"- **Tier {tier}**: {tier_counts[tier]} biomarkers\n"

        report_content += """
## Top 10 Biomarkers by Integrated Score

| Rank | Biomarker | Integrated Score | Causal Confidence | Evidence Tier | Layer |
|------|-----------|------------------|-------------------|---------------|-------|
"""

        for i, score in enumerate(top_10, 1):
            report_content += f"| {i} | {score.name} | {score.integrated_score:.3f} | {score.causal_confidence:.3f} | {score.evidence_tier} | {score.layer} |\n"

        report_content += f"""
## Causal Discovery Results
- **Total Causal Edges Discovered**: {len(self.causal_graph.edges) if self.causal_graph else 0}
- **High-Confidence Causal Edges (>80%)**: {len([e for e in (self.causal_graph.edges if self.causal_graph else []) if e.confidence > 0.8])}
- **Methods Used**: {', '.join(self.causal_graph.method_metadata.get('methods_used', [])) if self.causal_graph else 'None'}

## Strong Causal Biomarkers (Top 5)
"""

        strong_causal = sorted(
            [s for s in scored_biomarkers if s.causal_confidence > 0.7],
            key=lambda x: x.causal_confidence,
            reverse=True,
        )[:5]

        for score in strong_causal:
            report_content += f"""
### {score.name}
- **Causal Confidence**: {score.causal_confidence:.3f}
- **Mechanism**: {score.causal_mechanism or 'Direct relationship'}
- **Evidence Level**: {score.causal_evidence_level}/5
- **Upstream Factors**: {', '.join(score.upstream_biomarkers) if score.upstream_biomarkers else 'None'}
- **Downstream Targets**: {', '.join(score.downstream_targets) if score.downstream_targets else 'None'}
"""

        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"📋 Generated summary report: {report_path}")


def create_causal_biomarker_demo():
    """
    Demonstration of causal biomarker scoring system
    """
    print("🔬 CAUSAL BIOMARKER SCORING DEMONSTRATION")
    print("=" * 60)

    # Generate synthetic biomarker and clinical data
    np.random.seed(42)
    n_patients = 300

    # Synthetic biomarker data with known causal structure
    # Pathway: Genetic → Enzyme → Metabolite → Clinical Outcome
    genetic_factor = np.random.binomial(2, 0.3, n_patients)
    enzyme_activity = 100 + 30 * genetic_factor + np.random.normal(0, 20, n_patients)
    metabolite_level = 50 + 0.5 * enzyme_activity + np.random.normal(0, 10, n_patients)

    # Additional biomarkers with varying causal relationships
    inflammatory_marker = (
        20 + 0.1 * metabolite_level + np.random.normal(0, 5, n_patients)
    )
    confounding_marker = np.random.normal(30, 8, n_patients)  # No causal relationship

    # Clinical outcome influenced by metabolite and inflammation
    outcome_prob = 1 / (
        1 + np.exp(-(0.05 * metabolite_level + 0.1 * inflammatory_marker - 8))
    )
    clinical_outcome = np.random.binomial(1, outcome_prob, n_patients)

    # Create datasets
    biomarker_data = pd.DataFrame(
        {
            "ENZYME_ACTIVITY": enzyme_activity,
            "METABOLITE_X": metabolite_level,
            "CRP": inflammatory_marker,
            "RANDOM_MARKER": confounding_marker,
        },
        index=[f"patient_{i}" for i in range(n_patients)],
    )

    clinical_outcomes = pd.DataFrame(
        {"disease_outcome": clinical_outcome}, index=biomarker_data.index
    )

    # Genetic data for Mendelian randomization
    genetic_data = pd.DataFrame(
        {"rs12345_enzyme_variant": genetic_factor}, index=biomarker_data.index
    )

    # Biomarker metadata
    biomarker_metadata = pd.DataFrame(
        {
            "layer": ["proteomic", "metabolomic", "proteomic", "proteomic"],
            "type": ["enzymatic", "metabolite", "secreted", "secreted"],
        },
        index=biomarker_data.columns,
    )

    print("📊 Generated synthetic data:")
    print(f"   Patients: {n_patients}")
    print(f"   Biomarkers: {list(biomarker_data.columns)}")
    print("   Known pathway: Genetic → ENZYME_ACTIVITY → METABOLITE_X → Outcome")
    print("   Inflammatory mediator: CRP")
    print("   Confounding marker: RANDOM_MARKER (no causal relationship)")

    # Initialize causal biomarker scorer
    scorer = CausalBiomarkerScorer()

    print("\n🔍 Running Causal Biomarker Scoring...")

    # Discover and score biomarkers
    scored_biomarkers = scorer.discover_and_score_biomarkers(
        biomarker_data=biomarker_data,
        clinical_outcomes=clinical_outcomes,
        genetic_data=genetic_data,
        biomarker_metadata=biomarker_metadata,
        outcome_column="disease_outcome",
    )

    print("\n📈 BIOMARKER SCORING RESULTS:")
    print(f"   Total biomarkers scored: {len(scored_biomarkers)}")

    print("\n🏆 TOP BIOMARKERS (by integrated score):")
    for i, score in enumerate(scored_biomarkers[:5], 1):
        print(f"   {i}. {score.name}")
        print(f"      Integrated Score: {score.integrated_score:.3f}")
        print(f"      Causal Confidence: {score.causal_confidence:.3f}")
        print(f"      Evidence Tier: {score.evidence_tier}")
        print(
            f"      Mechanism: {score.causal_mechanism or 'Statistical association only'}"
        )
        print(f"      P-value: {score.p_value:.3e}")
        print()

    print("🎯 CAUSAL RELATIONSHIPS DISCOVERED:")
    if scorer.causal_graph:
        high_conf_edges = [e for e in scorer.causal_graph.edges if e.confidence > 0.6]
        for edge in high_conf_edges:
            print(f"   {edge.source} → {edge.target}")
            print(f"      Strength: {edge.strength:.3f}")
            print(f"      Confidence: {edge.confidence:.3f}")
            print(f"      Method: {edge.method}")
            print()

    # Export results
    output_dir = Path("artifacts/causal_biomarker_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer.export_scored_biomarkers(scored_biomarkers, output_dir)

    print("✅ Causal Biomarker Scoring Complete!")
    print(f"📂 Results exported to: {output_dir}")

    return scored_biomarkers, scorer
