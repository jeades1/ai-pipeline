"""
Cell-cell interaction integration framework.
Builds on existing CellPhoneDB foundation to create cell-type interaction layer.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class InteractionType(Enum):
    """Types of cell-cell interactions"""
    CELL_CONTACT = "cell_cell_contact"
    PARACRINE = "paracrine_signaling"
    MECHANICAL = "mechanical_coupling"
    METABOLIC = "metabolic_cross_feeding"


@dataclass
class CellTypeConfig:
    """Configuration for cell type properties"""
    name: str
    marker_genes: List[str]
    functions: List[str]
    tissue_location: Optional[str] = None
    
    
@dataclass
class InteractionConfig:
    """Configuration for interaction properties"""
    interaction_type: InteractionType
    directionality: str  # "bidirectional", "source_to_target"
    evidence_types: List[str]
    confidence_threshold: float = 0.5


class CCIIntegrator:
    """Main class for cell-cell interaction integration"""
    
    def __init__(self, kg_path: Optional[Path] = None):
        self.kg_path = kg_path
        self.cell_types = self._initialize_cell_types()
        self.interaction_configs = self._initialize_interaction_configs()
        
    def _initialize_cell_types(self) -> Dict[str, CellTypeConfig]:
        """Initialize kidney-relevant cell types"""
        return {
            "ProximalTubular": CellTypeConfig(
                name="ProximalTubular",
                marker_genes=["SLC34A1", "LRP2", "SLC5A12", "SLC22A6"],
                functions=["transport", "reabsorption", "metabolism"],
                tissue_location="proximal_tubule"
            ),
            "DistalTubular": CellTypeConfig(
                name="DistalTubular", 
                marker_genes=["SLC12A3", "SCNN1A", "AQP2", "UMOD"],
                functions=["electrolyte_balance", "water_reabsorption"],
                tissue_location="distal_tubule"
            ),
            "Podocyte": CellTypeConfig(
                name="Podocyte",
                marker_genes=["NPHS1", "NPHS2", "PODXL", "WT1"],
                functions=["filtration", "barrier_maintenance"],
                tissue_location="glomerulus"
            ),
            "Endothelial": CellTypeConfig(
                name="Endothelial",
                marker_genes=["PECAM1", "VWF", "CD34", "ENG"],
                functions=["vascular_barrier", "hemodynamics"],
                tissue_location="vasculature"
            ),
            "Immune": CellTypeConfig(
                name="Immune",
                marker_genes=["CD68", "CD3E", "CD19", "PTPRC"],
                functions=["inflammation", "tissue_repair"],
                tissue_location="interstitium"
            ),
            "Fibroblast": CellTypeConfig(
                name="Fibroblast",
                marker_genes=["COL1A1", "ACTA2", "FN1", "VIM"],
                functions=["ECM_production", "tissue_remodeling"],
                tissue_location="interstitium"
            )
        }
    
    def _initialize_interaction_configs(self) -> Dict[InteractionType, InteractionConfig]:
        """Initialize interaction type configurations"""
        return {
            InteractionType.CELL_CONTACT: InteractionConfig(
                interaction_type=InteractionType.CELL_CONTACT,
                directionality="bidirectional",
                evidence_types=["imaging", "proteomics", "adhesion_assays"],
                confidence_threshold=0.6
            ),
            InteractionType.PARACRINE: InteractionConfig(
                interaction_type=InteractionType.PARACRINE,
                directionality="source_to_target",
                evidence_types=["secretome", "functional", "literature"],
                confidence_threshold=0.5
            ),
            InteractionType.MECHANICAL: InteractionConfig(
                interaction_type=InteractionType.MECHANICAL,
                directionality="bidirectional",
                evidence_types=["TEER", "contractility", "AFM"],
                confidence_threshold=0.7
            ),
            InteractionType.METABOLIC: InteractionConfig(
                interaction_type=InteractionType.METABOLIC,
                directionality="source_to_target",
                evidence_types=["metabolomics", "flux_analysis"],
                confidence_threshold=0.6
            )
        }
    
    def add_cell_type_nodes(self, kg: Any) -> None:
        """Add cell type nodes to knowledge graph"""
        
        for cell_type, config in self.cell_types.items():
            # Add cell type node
            kg.ensure_node(
                cell_type,
                kind="CellType",
                layer="cellular",
                marker_genes=config.marker_genes,
                functions=config.functions,
                tissue_location=config.tissue_location
            )
            
            # Link marker genes to cell type
            for gene in config.marker_genes:
                kg.add_edge(
                    gene, cell_type,
                    etype="marker_for",
                    provenance="literature",
                    confidence=0.8
                )
    
    def add_interaction_edges(self, kg: Any, 
                            expression_data: Optional[pd.DataFrame] = None) -> None:
        """Add cell-cell interaction edges based on literature and expression"""
        
        # Predefined interactions from literature
        literature_interactions = [
            # Tubular-Endothelial crosstalk
            ("ProximalTubular", "Endothelial", InteractionType.PARACRINE, 
             {"mediators": ["VEGF", "NO", "ANGPT1"], "context": "vascular_maintenance"}),
            
            # Glomerular filtration barrier
            ("Podocyte", "Endothelial", InteractionType.MECHANICAL,
             {"mediators": ["integrin", "laminin"], "context": "filtration_barrier"}),
             
            # Immune-Tubular interaction
            ("Immune", "ProximalTubular", InteractionType.CELL_CONTACT,
             {"mediators": ["CD40", "CD40L"], "context": "inflammation"}),
             
            # Fibroblast-Tubular fibrosis
            ("Fibroblast", "ProximalTubular", InteractionType.PARACRINE,
             {"mediators": ["TGFB1", "PDGF", "FGF2"], "context": "fibrosis"}),
             
            # Tubular-Immune signaling
            ("ProximalTubular", "Immune", InteractionType.PARACRINE,
             {"mediators": ["CCL2", "IL6", "TNF"], "context": "injury_response"}),
             
            # Metabolic coupling
            ("ProximalTubular", "DistalTubular", InteractionType.METABOLIC,
             {"mediators": ["lactate", "glutamine"], "context": "energy_metabolism"})
        ]
        
        for source, target, interaction_type, evidence in literature_interactions:
            self._add_interaction_edge(kg, source, target, interaction_type, evidence, "literature")
        
        # Add expression-based interactions if data available
        if expression_data is not None:
            self._add_expression_based_interactions(kg, expression_data)
    
    def _add_interaction_edge(self, kg: Any, source: str, target: str, 
                            interaction_type: InteractionType, evidence: Dict[str, Any], 
                            provenance: str) -> None:
        """Add individual interaction edge with metadata"""
        
        config = self.interaction_configs[interaction_type]
        
        kg.add_edge(
            source, target,
            etype=interaction_type.value,
            provenance=provenance,
            evidence=evidence,
            directionality=config.directionality,
            evidence_types=config.evidence_types,
            confidence=evidence.get("confidence", config.confidence_threshold)
        )
        
        # Add reverse edge if bidirectional
        if config.directionality == "bidirectional":
            kg.add_edge(
                target, source,
                etype=interaction_type.value,
                provenance=provenance,
                evidence=evidence,
                directionality=config.directionality,
                evidence_types=config.evidence_types,
                confidence=evidence.get("confidence", config.confidence_threshold)
            )
    
    def _add_expression_based_interactions(self, kg: Any, expression_data: pd.DataFrame) -> None:
        """Add interactions based on expression data analysis"""
        
        # This would analyze co-expression patterns, ligand-receptor pairs, etc.
        # Simplified implementation for now
        
        # Group expression by cell type and condition
        if 'cell_type' not in expression_data.columns:
            return
        
        cell_type_expression = expression_data.groupby(['cell_type', 'gene'])['value'].mean().reset_index()
        
        # Find highly expressed ligands and receptors
        for cell_type in cell_type_expression['cell_type'].unique():
            if cell_type not in self.cell_types:
                continue
                
            cell_expr = cell_type_expression[cell_type_expression['cell_type'] == cell_type]
            highly_expressed = cell_expr[cell_expr['value'] > cell_expr['value'].quantile(0.8)]
            
            # Add high-expression marker edges
            for _, row in highly_expressed.iterrows():
                kg.add_edge(
                    row['gene'], cell_type,
                    etype="highly_expressed_in",
                    provenance="expression_analysis",
                    expression_level=row['value'],
                    confidence=0.7
                )
    
    def score_interaction_activity(self, kg: Any, 
                                  expression_data: Optional[pd.DataFrame] = None,
                                  functional_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Score interaction activity based on available evidence"""
        
        scores = []
        
        # Get all interaction edges from KG
        interaction_edges = self._get_interaction_edges(kg)
        
        for edge in interaction_edges:
            source, target, interaction_type = edge['source'], edge['target'], edge['etype']
            
            # Calculate activity score
            activity_score = self._calculate_activity_score(
                source, target, interaction_type, expression_data, functional_data
            )
            
            scores.append({
                'source_cell_type': source,
                'target_cell_type': target,
                'interaction_type': interaction_type,
                'activity_score': activity_score['score'],
                'evidence_strength': activity_score['evidence_strength'],
                'expression_support': activity_score['expression_support'],
                'functional_support': activity_score['functional_support']
            })
        
        return pd.DataFrame(scores)
    
    def _get_interaction_edges(self, kg: Any) -> List[Dict[str, Any]]:
        """Extract interaction edges from knowledge graph"""
        # This would query the actual KG structure
        # Simplified implementation
        interaction_types = [it.value for it in InteractionType]
        
        edges = []
        for source in self.cell_types:
            for target in self.cell_types:
                if source != target:
                    for int_type in interaction_types:
                        # Check if edge exists in KG
                        edges.append({
                            'source': source,
                            'target': target,
                            'etype': int_type
                        })
        
        return edges
    
    def _calculate_activity_score(self, source: str, target: str, interaction_type: str,
                                expression_data: Optional[pd.DataFrame] = None,
                                functional_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate interaction activity score"""
        
        # Base score from literature evidence
        base_score = 0.3
        
        # Expression-based evidence
        expression_support = 0.0
        if expression_data is not None:
            expression_support = self._score_expression_evidence(
                source, target, expression_data
            )
        
        # Functional evidence
        functional_support = 0.0
        if functional_data is not None:
            functional_support = self._score_functional_evidence(
                source, target, interaction_type, functional_data
            )
        
        # Combine scores
        evidence_strength = (expression_support + functional_support) / 2
        activity_score = base_score + (0.7 * evidence_strength)
        
        return {
            'score': min(activity_score, 1.0),
            'evidence_strength': evidence_strength,
            'expression_support': expression_support,
            'functional_support': functional_support
        }
    
    def _score_expression_evidence(self, source: str, target: str, 
                                 expression_data: pd.DataFrame) -> float:
        """Score based on expression data"""
        
        if 'cell_type' not in expression_data.columns:
            return 0.0
        
        # Check if source and target cell types are present and active
        source_data = expression_data[expression_data['cell_type'] == source]
        target_data = expression_data[expression_data['cell_type'] == target]
        
        if source_data.empty or target_data.empty:
            return 0.0
        
        # Simple scoring based on expression levels
        source_activity = source_data['value'].mean()
        target_activity = target_data['value'].mean()
        
        # Normalize to 0-1 range
        combined_activity = (source_activity + target_activity) / 2
        max_possible = expression_data['value'].max()
        
        return min(combined_activity / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _score_functional_evidence(self, source: str, target: str, interaction_type: str,
                                 functional_data: pd.DataFrame) -> float:
        """Score based on functional assay data"""
        
        # Map interaction types to relevant functional readouts
        interaction_function_map = {
            "cell_cell_contact": ["barrier_integrity", "junction_integrity"],
            "paracrine_signaling": ["protein_secretion", "signaling_activity"],
            "mechanical_coupling": ["barrier_integrity", "contractility"],
            "metabolic_cross_feeding": ["metabolic_activity", "transport"]
        }
        
        relevant_functions = interaction_function_map.get(interaction_type, [])
        
        if not relevant_functions:
            return 0.0
        
        # Score based on functional effects
        function_scores = []
        for function in relevant_functions:
            if 'function' in functional_data.columns:
                function_data = functional_data[functional_data['function'] == function]
                if not function_data.empty:
                    # Use effect size as proxy for interaction strength
                    avg_effect = np.abs(function_data['effect_size']).mean()
                    function_scores.append(min(avg_effect / 2.0, 1.0))  # Normalize
        
        return float(np.mean(function_scores)) if function_scores else 0.0


def add_cellular_interaction_layer(kg: Any, expression_data: Optional[pd.DataFrame] = None,
                                 functional_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Main function to add cellular interaction layer to KG"""
    
    integrator = CCIIntegrator()
    
    # Add cell type nodes
    integrator.add_cell_type_nodes(kg)
    
    # Add interaction edges
    integrator.add_interaction_edges(kg, expression_data)
    
    # Score interaction activities
    activity_scores = integrator.score_interaction_activity(kg, expression_data, functional_data)
    
    return activity_scores


def link_genes_to_cell_functions(kg: Any, invitro_data: pd.DataFrame) -> None:
    """Link gene expression to cellular function outcomes (updated from docs)"""
    
    if invitro_data.empty:
        return
    
    # Group by gene and compute functional consistency across assays
    for gene, gene_data in invitro_data.groupby("gene"):
        functional_scores = {}
        
        # TEER/permeability → barrier function
        teer_data = gene_data[gene_data["assay_type"] == "teer"]
        if not teer_data.empty:
            functional_scores["barrier_integrity"] = teer_data["value"].mean()
        
        # Secretome → signaling function  
        secretome_data = gene_data[gene_data["assay_type"] == "secretome"]
        if not secretome_data.empty:
            functional_scores["signaling_activity"] = secretome_data["value"].mean()
        
        # Add gene→function edges
        for function, score in functional_scores.items():
            kg.add_edge(
                gene, function, 
                etype="regulates_function", 
                evidence={"score": score}, 
                provenance="invitro_integrated"
            )
