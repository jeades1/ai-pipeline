"""
Mechanism Knowledge Graph Extensions

This module extends kg/build_graph.py to include exposure-mechanism relationships:
- CTD (Comparative Toxicogenomics Database) chemical-gene-disease links
- AOP (Adverse Outcome Pathways) molecular initiating events → key events → adverse outcomes
- LINCS perturbation signatures for mechanism validation
- Chemical ontology integration (CHEBI, DSSTox)

Integrates with existing KG infrastructure while adding mechanistic depth for
exposure-biomarker-outcome causal chains.

Author: AI Pipeline Team  
Date: September 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CTDRelationship:
    """CTD chemical-gene-disease relationship"""

    chemical_id: str
    chemical_name: str
    gene_symbol: str
    organism: str
    interaction_type: str  # increases^activity, decreases^expression, etc.
    pubmed_ids: List[str]
    evidence_count: int
    inference_score: float = 0.0


@dataclass
class AOPRelationship:
    """AOP molecular initiating event to adverse outcome relationship"""

    aop_id: str
    aop_title: str
    molecular_initiating_event: str
    key_events: List[str]
    adverse_outcome: str
    confidence_level: str  # high/moderate/low
    taxonomic_applicability: List[str]
    life_stage_applicability: List[str]
    evidence_support: str


@dataclass
class MechanismEvidence:
    """Evidence for exposure-mechanism relationships"""

    exposure_id: str
    mechanism_id: str
    evidence_type: str  # CTD/AOP/LINCS/literature
    evidence_strength: float  # 0-1
    supporting_studies: List[str]
    confidence_level: str


def add_exposure_mechanism_links(
    kg,
    ctd_relationships: List[CTDRelationship],
    aop_relationships: List[AOPRelationship],
    chemical_mappings: Optional[Dict[str, str]] = None,
) -> None:
    """
    Add exposure-mechanism links to the knowledge graph

    Args:
        kg: Knowledge graph instance with add_edge/ensure_node methods
        ctd_relationships: CTD chemical-gene relationships
        aop_relationships: AOP pathway relationships
        chemical_mappings: Mapping from common names to standard IDs
    """

    logger.info(f"Adding {len(ctd_relationships)} CTD relationships to KG")
    logger.info(f"Adding {len(aop_relationships)} AOP relationships to KG")

    # Add CTD chemical-gene relationships
    for ctd_rel in ctd_relationships:

        # Create chemical node
        chemical_id = _standardize_chemical_id(
            ctd_rel.chemical_id, ctd_rel.chemical_name, chemical_mappings
        )
        kg.ensure_node(
            chemical_id,
            kind="Chemical",
            layer="exposome",
            name=ctd_rel.chemical_name,
            database_id=ctd_rel.chemical_id,
            organism=ctd_rel.organism,
        )

        # Create gene node
        gene_id = ctd_rel.gene_symbol.upper()
        kg.ensure_node(
            gene_id,
            kind="Gene",
            layer="transcriptomic",
            name=ctd_rel.gene_symbol,
            organism=ctd_rel.organism,
        )

        # Parse interaction type for directionality
        interaction_direction, effect_type = _parse_ctd_interaction(
            ctd_rel.interaction_type
        )

        # Add chemical → gene edge
        kg.add_edge(
            chemical_id,
            gene_id,
            etype="perturbs",
            direction=interaction_direction,
            effect_type=effect_type,
            evidence_count=ctd_rel.evidence_count,
            pubmed_ids=ctd_rel.pubmed_ids[:5],  # Limit to top 5 PMIDs
            inference_score=ctd_rel.inference_score,
            provenance="CTD",
            context=_make_context(environment="human", organism=ctd_rel.organism),
            layer="chemical→gene",
            interaction_type=ctd_rel.interaction_type,
        )

    # Add AOP pathway relationships
    for aop_rel in aop_relationships:

        # Create AOP pathway node
        aop_id = f"AOP:{aop_rel.aop_id}"
        kg.ensure_node(
            aop_id,
            kind="AdverseOutcomePathway",
            layer="pathway",
            name=aop_rel.aop_title,
            confidence_level=aop_rel.confidence_level,
            taxonomic_applicability=aop_rel.taxonomic_applicability,
            life_stage_applicability=aop_rel.life_stage_applicability,
        )

        # Create molecular initiating event node
        mie_id = f"MIE:{_sanitize_id(aop_rel.molecular_initiating_event)}"
        kg.ensure_node(
            mie_id,
            kind="MolecularInitiatingEvent",
            layer="molecular",
            name=aop_rel.molecular_initiating_event,
            aop_id=aop_rel.aop_id,
        )

        # Create key event nodes and chain them
        previous_event = mie_id
        for i, key_event in enumerate(aop_rel.key_events):
            ke_id = f"KE:{_sanitize_id(key_event)}"
            kg.ensure_node(
                ke_id,
                kind="KeyEvent",
                layer="intermediate",
                name=key_event,
                aop_id=aop_rel.aop_id,
                event_order=i,
            )

            # Add causal edge from previous event
            kg.add_edge(
                previous_event,
                ke_id,
                etype="leads_to",
                confidence=aop_rel.confidence_level,
                evidence_support=aop_rel.evidence_support,
                provenance="AOP-Wiki",
                context=_make_context(environment="human"),
                layer="aop_progression",
            )

            previous_event = ke_id

        # Create adverse outcome node
        ao_id = f"AO:{_sanitize_id(aop_rel.adverse_outcome)}"
        kg.ensure_node(
            ao_id,
            kind="AdverseOutcome",
            layer="phenotypic",
            name=aop_rel.adverse_outcome,
            aop_id=aop_rel.aop_id,
        )

        # Add final edge to adverse outcome
        kg.add_edge(
            previous_event,
            ao_id,
            etype="leads_to",
            confidence=aop_rel.confidence_level,
            evidence_support=aop_rel.evidence_support,
            provenance="AOP-Wiki",
            context=_make_context(environment="human"),
            layer="aop_progression",
        )

        # Add overall AOP pathway edge
        kg.add_edge(
            mie_id,
            ao_id,
            etype="causally_linked_via_aop",
            aop_id=aop_id,
            confidence=aop_rel.confidence_level,
            provenance="AOP-Wiki",
            context=_make_context(environment="human"),
            layer="aop_overview",
        )


def load_ctd_relationships(ctd_file: Optional[Path] = None) -> List[CTDRelationship]:
    """Load CTD chemical-gene interactions from file or generate synthetic data"""

    if ctd_file and ctd_file.exists():
        logger.info(f"Loading CTD relationships from {ctd_file}")
        return _load_ctd_from_file(ctd_file)
    else:
        logger.info("Generating synthetic CTD relationships for demonstration")
        return _generate_synthetic_ctd_data()


def load_aop_relationships(aop_file: Optional[Path] = None) -> List[AOPRelationship]:
    """Load AOP pathway data from file or generate synthetic data"""

    if aop_file and aop_file.exists():
        logger.info(f"Loading AOP relationships from {aop_file}")
        return _load_aop_from_file(aop_file)
    else:
        logger.info("Generating synthetic AOP relationships for demonstration")
        return _generate_synthetic_aop_data()


def create_chemical_mappings() -> Dict[str, str]:
    """Create mapping from common chemical names to standard identifiers"""

    return {
        # Air pollutants
        "particulate matter": "CHEBI:132076",
        "nitrogen dioxide": "CHEBI:17632",
        "ozone": "CHEBI:25812",
        "sulfur dioxide": "CHEBI:18422",
        "carbon monoxide": "CHEBI:17245",
        # PFAS chemicals
        "perfluorooctanoic acid": "CHEBI:39421",
        "perfluorooctanesulfonic acid": "CHEBI:39422",
        "PFOA": "CHEBI:39421",
        "PFOS": "CHEBI:39422",
        # Metals
        "lead": "CHEBI:25016",
        "mercury": "CHEBI:16170",
        "cadmium": "CHEBI:22977",
        "arsenic": "CHEBI:22632",
        # Pesticides
        "atrazine": "CHEBI:15930",
        "glyphosate": "CHEBI:27744",
        "2,4-dichlorophenoxyacetic acid": "CHEBI:28854",
        # Industrial chemicals
        "bisphenol A": "CHEBI:33216",
        "benzene": "CHEBI:16716",
        "toluene": "CHEBI:17578",
        "formaldehyde": "CHEBI:16842",
    }


def query_mechanism_paths(
    kg, exposure_id: str, outcome_id: str, max_path_length: int = 5
) -> List[Dict[str, Any]]:
    """
    Query mechanism paths between exposure and clinical outcome

    Args:
        kg: Knowledge graph instance
        exposure_id: Chemical/exposure identifier
        outcome_id: Clinical outcome identifier
        max_path_length: Maximum path length to search

    Returns:
        List of mechanism paths with evidence scores
    """

    try:
        import networkx as nx

        # Get the networkx graph from KG
        G = kg.G if hasattr(kg, "G") else kg

        if not isinstance(G, nx.Graph):
            logger.warning("KG does not contain NetworkX graph for path analysis")
            return []

        # Find all simple paths up to max length
        try:
            paths = list(
                nx.all_simple_paths(G, exposure_id, outcome_id, cutoff=max_path_length)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.info(f"No paths found between {exposure_id} and {outcome_id}")
            return []

        mechanism_paths = []

        for path in paths:
            # Calculate path evidence score
            path_score = _calculate_path_evidence_score(G, path)

            # Extract mechanism types along path
            mechanism_types = _extract_mechanism_types(G, path)

            # Count evidence sources
            evidence_sources = _count_evidence_sources(G, path)

            path_info = {
                "path": path,
                "path_length": len(path) - 1,
                "evidence_score": path_score,
                "mechanism_types": mechanism_types,
                "evidence_sources": evidence_sources,
                "pathway_description": _describe_mechanism_path(G, path),
            }

            mechanism_paths.append(path_info)

        # Sort by evidence score
        mechanism_paths.sort(key=lambda x: x["evidence_score"], reverse=True)

        logger.info(
            f"Found {len(mechanism_paths)} mechanism paths from {exposure_id} to {outcome_id}"
        )

        return mechanism_paths

    except Exception as e:
        logger.error(f"Error querying mechanism paths: {e}")
        return []


def validate_mechanism_with_lincs(
    chemical_id: str, gene_targets: List[str], lincs_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Validate mechanism predictions using LINCS perturbation data

    Args:
        chemical_id: Chemical identifier
        gene_targets: Predicted gene targets from CTD/AOP
        lincs_data: LINCS L1000 perturbation data

    Returns:
        Validation results with concordance scores
    """

    if lincs_data is None:
        # Generate synthetic LINCS validation for demo
        return _generate_synthetic_lincs_validation(chemical_id, gene_targets)

    validation_results = {
        "chemical_id": chemical_id,
        "predicted_targets": gene_targets,
        "validated_targets": [],
        "concordance_score": 0.0,
        "perturbation_signatures": {},
        "confidence_level": "low",
    }

    # Look for chemical in LINCS data
    chemical_matches = lincs_data[
        lincs_data["pert_iname"].str.contains(chemical_id, case=False, na=False)
    ]

    if chemical_matches.empty:
        logger.info(f"No LINCS data found for chemical {chemical_id}")
        return validation_results

    # Analyze perturbation signatures for predicted targets
    validated_targets = []
    perturbation_scores = {}

    for gene in gene_targets:
        # Look for differential expression of this gene
        gene_data = chemical_matches[chemical_matches["gene_symbol"] == gene.upper()]

        if not gene_data.empty:
            # Calculate average perturbation score
            avg_score = gene_data["z_score"].mean()
            perturbation_scores[gene] = avg_score

            # Consider significant if |z_score| > 2
            if abs(avg_score) > 2.0:
                validated_targets.append(gene)

    # Calculate concordance
    if gene_targets:
        concordance_score = len(validated_targets) / len(gene_targets)
    else:
        concordance_score = 0.0

    # Determine confidence level
    if concordance_score > 0.7:
        confidence_level = "high"
    elif concordance_score > 0.3:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    validation_results.update(
        {
            "validated_targets": validated_targets,
            "concordance_score": concordance_score,
            "perturbation_signatures": perturbation_scores,
            "confidence_level": confidence_level,
            "lincs_experiments": len(chemical_matches),
        }
    )

    return validation_results


# Helper functions


def _standardize_chemical_id(
    ctd_id: str, chemical_name: str, mappings: Optional[Dict[str, str]] = None
) -> str:
    """Standardize chemical identifier to CHEBI format"""

    if mappings and chemical_name.lower() in mappings:
        return mappings[chemical_name.lower()]

    # Try to extract CHEBI ID from CTD ID
    if "CHEBI:" in ctd_id:
        return ctd_id

    # Create standardized ID
    return f"CHEMICAL:{_sanitize_id(chemical_name)}"


def _parse_ctd_interaction(interaction_type: str) -> Tuple[str, str]:
    """Parse CTD interaction type for directionality and effect"""

    interaction_lower = interaction_type.lower()

    if "increases" in interaction_lower:
        direction = "positive"
        effect = "increase"
    elif "decreases" in interaction_lower:
        direction = "negative"
        effect = "decrease"
    elif "affects" in interaction_lower:
        direction = "neutral"
        effect = "modulates"
    else:
        direction = "unknown"
        effect = "unknown"

    return direction, effect


def _sanitize_id(text: str) -> str:
    """Sanitize text for use as node ID"""
    # Remove special characters and spaces
    sanitized = re.sub(r"[^\w\s-]", "", text)
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized.upper()[:50]  # Limit length


def _make_context(
    environment: str = "human", organism: str = "Homo sapiens"
) -> Dict[str, str]:
    """Create context dictionary for KG edges"""
    return {
        "environment": environment,
        "organism": organism,
        "evidence_type": "literature",
    }


def _load_ctd_from_file(ctd_file: Path) -> List[CTDRelationship]:
    """Load CTD relationships from file (TSV format)"""

    relationships = []

    try:
        df = pd.read_csv(ctd_file, sep="\t", comment="#", low_memory=False)

        # Expected columns: ChemicalID, ChemicalName, GeneSymbol, Organism, Interaction, PubMedIDs
        for _, row in df.iterrows():

            pubmed_ids = (
                str(row.get("PubMedIDs", "")).split("|") if row.get("PubMedIDs") else []
            )

            relationship = CTDRelationship(
                chemical_id=str(row.get("ChemicalID", "")),
                chemical_name=str(row.get("ChemicalName", "")),
                gene_symbol=str(row.get("GeneSymbol", "")),
                organism=str(row.get("Organism", "Homo sapiens")),
                interaction_type=str(row.get("Interaction", "")),
                pubmed_ids=pubmed_ids,
                evidence_count=len(pubmed_ids),
                inference_score=float(row.get("InferenceScore", 0.0)),
            )

            relationships.append(relationship)

    except Exception as e:
        logger.error(f"Error loading CTD file: {e}")

    return relationships


def _load_aop_from_file(aop_file: Path) -> List[AOPRelationship]:
    """Load AOP relationships from file (JSON format)"""

    relationships = []

    try:
        with open(aop_file, "r") as f:
            aop_data = json.load(f)

        for aop in aop_data.get("aops", []):
            relationship = AOPRelationship(
                aop_id=str(aop.get("id", "")),
                aop_title=aop.get("title", ""),
                molecular_initiating_event=aop.get("mie", ""),
                key_events=aop.get("key_events", []),
                adverse_outcome=aop.get("adverse_outcome", ""),
                confidence_level=aop.get("confidence", "moderate"),
                taxonomic_applicability=aop.get("taxonomic_applicability", ["human"]),
                life_stage_applicability=aop.get("life_stage_applicability", ["adult"]),
                evidence_support=aop.get("evidence_support", "moderate"),
            )

            relationships.append(relationship)

    except Exception as e:
        logger.error(f"Error loading AOP file: {e}")

    return relationships


def _generate_synthetic_ctd_data() -> List[CTDRelationship]:
    """Generate synthetic CTD relationships for demonstration"""

    synthetic_relationships = [
        # Air pollution effects
        CTDRelationship(
            chemical_id="CHEBI:132076",
            chemical_name="particulate matter",
            gene_symbol="IL6",
            organism="Homo sapiens",
            interaction_type="increases^expression",
            pubmed_ids=["12345678", "87654321"],
            evidence_count=25,
            inference_score=0.85,
        ),
        CTDRelationship(
            chemical_id="CHEBI:17632",
            chemical_name="nitrogen dioxide",
            gene_symbol="TNF",
            organism="Homo sapiens",
            interaction_type="increases^activity",
            pubmed_ids=["11111111", "22222222"],
            evidence_count=18,
            inference_score=0.78,
        ),
        # PFAS effects
        CTDRelationship(
            chemical_id="CHEBI:39421",
            chemical_name="PFOA",
            gene_symbol="PPARA",
            organism="Homo sapiens",
            interaction_type="increases^activity",
            pubmed_ids=["33333333", "44444444"],
            evidence_count=32,
            inference_score=0.92,
        ),
        CTDRelationship(
            chemical_id="CHEBI:39421",
            chemical_name="PFOA",
            gene_symbol="FABP1",
            organism="Homo sapiens",
            interaction_type="increases^expression",
            pubmed_ids=["55555555"],
            evidence_count=12,
            inference_score=0.71,
        ),
        # Metal effects
        CTDRelationship(
            chemical_id="CHEBI:25016",
            chemical_name="lead",
            gene_symbol="MT1A",
            organism="Homo sapiens",
            interaction_type="increases^expression",
            pubmed_ids=["66666666", "77777777", "88888888"],
            evidence_count=45,
            inference_score=0.96,
        ),
        CTDRelationship(
            chemical_id="CHEBI:16170",
            chemical_name="mercury",
            gene_symbol="GSS",
            organism="Homo sapiens",
            interaction_type="decreases^activity",
            pubmed_ids=["99999999"],
            evidence_count=8,
            inference_score=0.63,
        ),
    ]

    return synthetic_relationships


def _generate_synthetic_aop_data() -> List[AOPRelationship]:
    """Generate synthetic AOP relationships for demonstration"""

    synthetic_aops = [
        AOPRelationship(
            aop_id="AOP001",
            aop_title="Oxidative stress leading to kidney injury",
            molecular_initiating_event="Reactive oxygen species generation",
            key_events=[
                "Mitochondrial dysfunction",
                "Inflammatory response activation",
                "Tubular epithelial cell death",
            ],
            adverse_outcome="Acute kidney injury",
            confidence_level="high",
            taxonomic_applicability=["human", "rat", "mouse"],
            life_stage_applicability=["adult", "elderly"],
            evidence_support="strong",
        ),
        AOPRelationship(
            aop_id="AOP002",
            aop_title="PPAR-alpha activation leading to liver steatosis",
            molecular_initiating_event="PPAR-alpha receptor activation",
            key_events=[
                "Fatty acid oxidation increase",
                "Lipid accumulation",
                "Hepatocyte hypertrophy",
            ],
            adverse_outcome="Hepatic steatosis",
            confidence_level="moderate",
            taxonomic_applicability=["human", "mouse"],
            life_stage_applicability=["adult"],
            evidence_support="moderate",
        ),
        AOPRelationship(
            aop_id="AOP003",
            aop_title="Metal-induced oxidative stress to nephrotoxicity",
            molecular_initiating_event="Heavy metal binding to sulfhydryl groups",
            key_events=[
                "Glutathione depletion",
                "Oxidative stress",
                "DNA damage",
                "Cell cycle arrest",
            ],
            adverse_outcome="Chronic kidney disease",
            confidence_level="high",
            taxonomic_applicability=["human"],
            life_stage_applicability=["adult", "child"],
            evidence_support="strong",
        ),
    ]

    return synthetic_aops


def _calculate_path_evidence_score(G, path: List[str]) -> float:
    """Calculate cumulative evidence score for a mechanism path"""

    total_score = 1.0

    for i in range(len(path) - 1):
        source, target = path[i], path[i + 1]

        # Get edge data
        if G.has_edge(source, target):
            edge_data = G[source][target]

            # Extract evidence metrics (handle MultiGraph)
            if isinstance(edge_data, dict):
                evidence_count = edge_data.get("evidence_count", 1)
                inference_score = edge_data.get("inference_score", 0.5)
            else:
                # MultiGraph - use first edge data
                first_edge = next(iter(edge_data.values()))
                evidence_count = first_edge.get("evidence_count", 1)
                inference_score = first_edge.get("inference_score", 0.5)

            # Calculate edge score (combines evidence count and inference score)
            edge_score = min(1.0, (evidence_count / 10.0) * inference_score)
            total_score *= edge_score
        else:
            total_score *= 0.1  # Penalty for missing edges

    return total_score


def _extract_mechanism_types(G, path: List[str]) -> List[str]:
    """Extract mechanism types along a path"""

    mechanism_types = []

    for node in path:
        if G.has_node(node):
            node_data = G.nodes[node]
            node_kind = node_data.get("kind", "unknown")
            mechanism_types.append(node_kind)

    return mechanism_types


def _count_evidence_sources(G, path: List[str]) -> Dict[str, int]:
    """Count evidence sources along a path"""

    evidence_sources = {}

    for i in range(len(path) - 1):
        source, target = path[i], path[i + 1]

        if G.has_edge(source, target):
            edge_data = G[source][target]

            if isinstance(edge_data, dict):
                provenance = edge_data.get("provenance", "unknown")
            else:
                first_edge = next(iter(edge_data.values()))
                provenance = first_edge.get("provenance", "unknown")

            evidence_sources[provenance] = evidence_sources.get(provenance, 0) + 1

    return evidence_sources


def _describe_mechanism_path(G, path: List[str]) -> str:
    """Generate human-readable description of mechanism path"""

    if len(path) < 2:
        return "Invalid path"

    descriptions = []

    for i in range(len(path) - 1):
        source, target = path[i], path[i + 1]

        # Get node names
        source_name = (
            G.nodes[source].get("name", source) if G.has_node(source) else source
        )
        target_name = (
            G.nodes[target].get("name", target) if G.has_node(target) else target
        )

        # Get edge type
        if G.has_edge(source, target):
            edge_data = G[source][target]
            if isinstance(edge_data, dict):
                etype = edge_data.get("etype", "affects")
            else:
                first_edge = next(iter(edge_data.values()))
                etype = first_edge.get("etype", "affects")
        else:
            etype = "affects"

        descriptions.append(f"{source_name} {etype} {target_name}")

    return " → ".join(descriptions)


def _generate_synthetic_lincs_validation(
    chemical_id: str, gene_targets: List[str]
) -> Dict[str, Any]:
    """Generate synthetic LINCS validation results"""

    # Simulate realistic validation results
    np.random.seed(hash(chemical_id) % 2**32)

    validated_targets = []
    perturbation_scores = {}

    for gene in gene_targets:
        # Simulate perturbation score
        z_score = np.random.normal(0, 3)
        perturbation_scores[gene] = z_score

        # Consider validated if |z_score| > 2
        if abs(z_score) > 2.0:
            validated_targets.append(gene)

    concordance_score = (
        len(validated_targets) / len(gene_targets) if gene_targets else 0.0
    )

    if concordance_score > 0.7:
        confidence_level = "high"
    elif concordance_score > 0.3:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    return {
        "chemical_id": chemical_id,
        "predicted_targets": gene_targets,
        "validated_targets": validated_targets,
        "concordance_score": concordance_score,
        "perturbation_signatures": perturbation_scores,
        "confidence_level": confidence_level,
        "lincs_experiments": np.random.randint(5, 25),
    }


def run_mechanism_kg_demo():
    """Demonstrate mechanism knowledge graph enhancement"""

    print("\n🧬 MECHANISM KNOWLEDGE GRAPH DEMONSTRATION")
    print("=" * 60)

    # Create mock KG object for demonstration
    class MockKG:
        def __init__(self):
            import networkx as nx

            self.G = nx.MultiDiGraph()
            self.nodes_added = []
            self.edges_added = []

        def ensure_node(self, node_id, **kwargs):
            self.G.add_node(node_id, **kwargs)
            self.nodes_added.append((node_id, kwargs))

        def add_edge(self, source, target, **kwargs):
            self.G.add_edge(source, target, **kwargs)
            self.edges_added.append((source, target, kwargs))

    kg = MockKG()

    # Load mechanism data
    print("📊 Loading mechanism relationship data...")
    ctd_relationships = load_ctd_relationships()
    aop_relationships = load_aop_relationships()
    chemical_mappings = create_chemical_mappings()

    print(f"   CTD relationships: {len(ctd_relationships)}")
    print(f"   AOP pathways: {len(aop_relationships)}")
    print(f"   Chemical mappings: {len(chemical_mappings)}")

    # Add to knowledge graph
    print("\n🔗 Adding mechanism links to knowledge graph...")
    add_exposure_mechanism_links(
        kg, ctd_relationships, aop_relationships, chemical_mappings
    )

    print(f"   Nodes added: {len(kg.nodes_added)}")
    print(f"   Edges added: {len(kg.edges_added)}")

    # Show example relationships
    print("\n📋 Example mechanism relationships:")

    if ctd_relationships:
        example_ctd = ctd_relationships[0]
        print("\n   CTD Relationship:")
        print(
            f"      Chemical: {example_ctd.chemical_name} ({example_ctd.chemical_id})"
        )
        print(f"      Target gene: {example_ctd.gene_symbol}")
        print(f"      Interaction: {example_ctd.interaction_type}")
        print(f"      Evidence: {example_ctd.evidence_count} studies")
        print(f"      Confidence: {example_ctd.inference_score:.2f}")

    if aop_relationships:
        example_aop = aop_relationships[0]
        print("\n   AOP Pathway:")
        print(f"      Title: {example_aop.aop_title}")
        print(f"      MIE: {example_aop.molecular_initiating_event}")
        print(f"      Key events: {' → '.join(example_aop.key_events[:2])}...")
        print(f"      Adverse outcome: {example_aop.adverse_outcome}")
        print(f"      Confidence: {example_aop.confidence_level}")

    # Demonstrate mechanism path queries
    print("\n🔍 Demonstrating mechanism path queries...")

    # Add some sample clinical outcome nodes for path demonstration
    kg.ensure_node(
        "CLINICAL:acute_kidney_injury",
        kind="ClinicalOutcome",
        name="Acute Kidney Injury",
    )
    kg.ensure_node(
        "CLINICAL:liver_toxicity", kind="ClinicalOutcome", name="Liver Toxicity"
    )

    # Query paths between exposure and outcome
    exposure_id = "CHEBI:132076"  # PM2.5
    outcome_id = "CLINICAL:acute_kidney_injury"

    mechanism_paths = query_mechanism_paths(kg, exposure_id, outcome_id)

    if mechanism_paths:
        print(f"   Found {len(mechanism_paths)} mechanism paths:")
        for i, path_info in enumerate(mechanism_paths[:3], 1):
            print(f"      Path {i}: {' → '.join(path_info['path'][:3])}...")
            print(f"         Evidence score: {path_info['evidence_score']:.3f}")
            print(f"         Mechanism types: {path_info['mechanism_types']}")
    else:
        print("   No mechanism paths found (demo KG may be incomplete)")

    # Demonstrate LINCS validation
    print("\n🧪 Demonstrating LINCS mechanism validation...")

    chemical_id = "CHEBI:39421"  # PFOA
    predicted_targets = ["PPARA", "FABP1", "CYP4A11"]

    validation_results = validate_mechanism_with_lincs(chemical_id, predicted_targets)

    print(f"   Chemical: {validation_results['chemical_id']}")
    print(f"   Predicted targets: {validation_results['predicted_targets']}")
    print(f"   Validated targets: {validation_results['validated_targets']}")
    print(f"   Concordance score: {validation_results['concordance_score']:.2f}")
    print(f"   Confidence level: {validation_results['confidence_level']}")

    # Show perturbation signatures
    if validation_results["perturbation_signatures"]:
        print("   Perturbation signatures:")
        for gene, score in validation_results["perturbation_signatures"].items():
            print(f"      {gene}: {score:.2f}")

    print("\n✅ Mechanism knowledge graph demonstration complete!")
    print("\nKey capabilities demonstrated:")
    print("  • CTD chemical-gene relationship integration")
    print("  • AOP molecular pathway representation")
    print("  • Mechanism path discovery and scoring")
    print("  • LINCS perturbation validation")
    print("  • Chemical ontology standardization")

    return {
        "kg": kg,
        "ctd_relationships": ctd_relationships,
        "aop_relationships": aop_relationships,
        "mechanism_paths": mechanism_paths,
        "validation_results": validation_results,
    }


if __name__ == "__main__":
    run_mechanism_kg_demo()
