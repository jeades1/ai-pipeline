"""
Activity scoring for cell-cell interactions.
Integrates multiple evidence types to quantify interaction strength.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvidenceWeight:
    """Weights for different evidence types"""
    literature: float = 0.3
    expression: float = 0.3
    functional: float = 0.3
    structural: float = 0.1


@dataclass 
class ActivityScore:
    """Comprehensive activity score for an interaction"""
    interaction_id: str
    source_cell: str
    target_cell: str
    interaction_type: str
    overall_score: float
    evidence_breakdown: Dict[str, float]
    confidence: float
    supporting_data: Dict[str, Any]


class ActivityScorer:
    """Scores cell-cell interaction activity from multiple evidence sources"""
    
    def __init__(self, evidence_weights: Optional[EvidenceWeight] = None):
        self.weights = evidence_weights or EvidenceWeight()
        self.ligand_receptor_db = self._initialize_lr_database()
        
    def _initialize_lr_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize ligand-receptor interaction database"""
        
        # Key kidney-relevant L-R pairs with functional context
        return {
            "VEGFA_VEGFR2": {
                "ligand": "VEGFA",
                "receptor": "KDR",  # VEGFR2
                "function": "angiogenesis",
                "cell_types": ["Endothelial", "ProximalTubular"],
                "confidence": 0.9,
                "pathway": "VEGF_signaling"
            },
            "TGFB1_TGFBR1": {
                "ligand": "TGFB1", 
                "receptor": "TGFBR1",
                "function": "fibrosis",
                "cell_types": ["Fibroblast", "ProximalTubular"],
                "confidence": 0.95,
                "pathway": "TGF_beta_signaling"
            },
            "CCL2_CCR2": {
                "ligand": "CCL2",
                "receptor": "CCR2", 
                "function": "inflammation",
                "cell_types": ["ProximalTubular", "Immune"],
                "confidence": 0.85,
                "pathway": "chemokine_signaling"
            },
            "PDGFB_PDGFRB": {
                "ligand": "PDGFB",
                "receptor": "PDGFRB",
                "function": "proliferation",
                "cell_types": ["Endothelial", "Fibroblast"],
                "confidence": 0.8,
                "pathway": "PDGF_signaling"
            },
            "FN1_ITGB1": {
                "ligand": "FN1",  # Fibronectin
                "receptor": "ITGB1",  # Integrin beta-1
                "function": "adhesion",
                "cell_types": ["Fibroblast", "ProximalTubular"],
                "confidence": 0.9,
                "pathway": "ECM_interaction"
            },
            "ANGPT1_TEK": {
                "ligand": "ANGPT1",
                "receptor": "TEK",  # Tie2
                "function": "vascular_stability",
                "cell_types": ["ProximalTubular", "Endothelial"],
                "confidence": 0.85,
                "pathway": "angiopoietin_signaling"
            }
        }
    
    def score_interaction_activity(self, source_cell: str, target_cell: str, 
                                 interaction_type: str,
                                 expression_data: Optional[pd.DataFrame] = None,
                                 functional_data: Optional[pd.DataFrame] = None,
                                 secretome_data: Optional[pd.DataFrame] = None) -> ActivityScore:
        """Score activity for a specific cell-cell interaction"""
        
        interaction_id = f"{source_cell}_{target_cell}_{interaction_type}"
        
        # Score different evidence types
        lit_score = self._score_literature_evidence(source_cell, target_cell, interaction_type)
        expr_score = self._score_expression_evidence(source_cell, target_cell, expression_data)
        func_score = self._score_functional_evidence(source_cell, target_cell, functional_data)
        struct_score = self._score_structural_evidence(source_cell, target_cell, secretome_data)
        
        # Calculate weighted overall score
        overall_score = (
            lit_score * self.weights.literature +
            expr_score * self.weights.expression + 
            func_score * self.weights.functional +
            struct_score * self.weights.structural
        )
        
        # Calculate confidence based on evidence diversity
        evidence_types_present = sum([
            lit_score > 0, expr_score > 0, func_score > 0, struct_score > 0
        ])
        confidence = min(evidence_types_present / 4.0 * 1.2, 1.0)
        
        evidence_breakdown = {
            "literature": lit_score,
            "expression": expr_score,
            "functional": func_score,
            "structural": struct_score
        }
        
        supporting_data = self._collect_supporting_data(
            source_cell, target_cell, expression_data, functional_data, secretome_data
        )
        
        return ActivityScore(
            interaction_id=interaction_id,
            source_cell=source_cell,
            target_cell=target_cell,
            interaction_type=interaction_type,
            overall_score=overall_score,
            evidence_breakdown=evidence_breakdown,
            confidence=confidence,
            supporting_data=supporting_data
        )
    
    def _score_literature_evidence(self, source_cell: str, target_cell: str, 
                                  interaction_type: str) -> float:
        """Score based on literature-derived L-R pairs"""
        
        # Find relevant L-R pairs for these cell types
        relevant_pairs = []
        for pair_id, pair_data in self.ligand_receptor_db.items():
            if (source_cell in pair_data["cell_types"] and 
                target_cell in pair_data["cell_types"]):
                relevant_pairs.append(pair_data)
        
        if not relevant_pairs:
            return 0.0
        
        # Weight by confidence and interaction type relevance
        type_relevance = {
            "paracrine_signaling": 1.0,
            "cell_cell_contact": 0.7,
            "mechanical_coupling": 0.5,
            "metabolic_cross_feeding": 0.3
        }
        
        relevance_weight = type_relevance.get(interaction_type, 0.5)
        avg_confidence = np.mean([pair["confidence"] for pair in relevant_pairs])
        
        return float(avg_confidence * relevance_weight)
    
    def _score_expression_evidence(self, source_cell: str, target_cell: str,
                                  expression_data: Optional[pd.DataFrame]) -> float:
        """Score based on ligand-receptor expression correlation"""
        
        if expression_data is None or expression_data.empty:
            return 0.0
        
        if 'cell_type' not in expression_data.columns:
            return 0.0
        
        # Get expression for source and target cells
        source_expr = expression_data[expression_data['cell_type'] == source_cell]
        target_expr = expression_data[expression_data['cell_type'] == target_cell]
        
        if source_expr.empty or target_expr.empty:
            return 0.0
        
        # Score ligand-receptor pairs
        lr_scores = []
        for pair_id, pair_data in self.ligand_receptor_db.items():
            if (source_cell in pair_data["cell_types"] and 
                target_cell in pair_data["cell_types"]):
                
                ligand = pair_data["ligand"]
                receptor = pair_data["receptor"]
                
                # Check ligand expression in source
                ligand_expr = source_expr[source_expr['gene'] == ligand]
                receptor_expr = target_expr[target_expr['gene'] == receptor]
                
                if not ligand_expr.empty and not receptor_expr.empty:
                    ligand_level = ligand_expr['value'].mean()
                    receptor_level = receptor_expr['value'].mean()
                    
                    # Score based on both being expressed
                    max_expr = expression_data['value'].max()
                    if max_expr > 0:
                        norm_ligand = ligand_level / max_expr
                        norm_receptor = receptor_level / max_expr
                        lr_scores.append(np.sqrt(norm_ligand * norm_receptor))
        
        return float(np.mean(lr_scores)) if lr_scores else 0.0
    
    def _score_functional_evidence(self, source_cell: str, target_cell: str,
                                  functional_data: Optional[pd.DataFrame]) -> float:
        """Score based on functional assay evidence"""
        
        if functional_data is None or functional_data.empty:
            return 0.0
        
        # Map cell types to functional outcomes they influence
        cell_function_map = {
            "ProximalTubular": ["transport", "barrier_integrity", "metabolism"],
            "DistalTubular": ["electrolyte_balance", "transport"],
            "Podocyte": ["filtration", "barrier_integrity"],
            "Endothelial": ["barrier_integrity", "vascular_function"],
            "Immune": ["inflammation", "tissue_repair"],
            "Fibroblast": ["ECM_production", "tissue_remodeling"]
        }
        
        source_functions = cell_function_map.get(source_cell, [])
        target_functions = cell_function_map.get(target_cell, [])
        
        # Look for functional effects that could indicate interaction
        shared_functions = set(source_functions) & set(target_functions)
        
        if 'function' not in functional_data.columns:
            return 0.0
        
        function_scores = []
        for function in shared_functions:
            func_data = functional_data[functional_data['function'] == function]
            if not func_data.empty:
                # Use absolute effect size as interaction strength proxy
                avg_effect = np.abs(func_data['effect_size']).mean()
                # Normalize by assuming max effect size of 3.0
                function_scores.append(min(avg_effect / 3.0, 1.0))
        
        return float(np.mean(function_scores)) if function_scores else 0.0
    
    def _score_structural_evidence(self, source_cell: str, target_cell: str,
                                  secretome_data: Optional[pd.DataFrame]) -> float:
        """Score based on secretome/protein interaction evidence"""
        
        if secretome_data is None or secretome_data.empty:
            return 0.0
        
        # Look for secreted factors that could mediate interaction
        interaction_scores = []
        
        for pair_id, pair_data in self.ligand_receptor_db.items():
            if (source_cell in pair_data["cell_types"] and 
                target_cell in pair_data["cell_types"]):
                
                ligand = pair_data["ligand"]
                
                # Check if ligand is found in secretome
                if 'protein' in secretome_data.columns:
                    ligand_secretion = secretome_data[secretome_data['protein'] == ligand]
                    
                    if not ligand_secretion.empty:
                        # Score based on secretion level
                        secretion_level = ligand_secretion['value'].mean()
                        max_secretion = secretome_data['value'].max()
                        
                        if max_secretion > 0:
                            norm_secretion = secretion_level / max_secretion
                            interaction_scores.append(norm_secretion)
        
        return float(np.mean(interaction_scores)) if interaction_scores else 0.0
    
    def _collect_supporting_data(self, source_cell: str, target_cell: str,
                               expression_data: Optional[pd.DataFrame],
                               functional_data: Optional[pd.DataFrame],
                               secretome_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Collect supporting data for the interaction score"""
        
        supporting_data = {
            "relevant_lr_pairs": [],
            "expression_evidence": {},
            "functional_evidence": {},
            "secretome_evidence": {}
        }
        
        # Relevant L-R pairs
        for pair_id, pair_data in self.ligand_receptor_db.items():
            if (source_cell in pair_data["cell_types"] and 
                target_cell in pair_data["cell_types"]):
                supporting_data["relevant_lr_pairs"].append({
                    "pair_id": pair_id,
                    "ligand": pair_data["ligand"],
                    "receptor": pair_data["receptor"],
                    "function": pair_data["function"],
                    "confidence": pair_data["confidence"]
                })
        
        # Expression evidence
        if expression_data is not None and not expression_data.empty:
            if 'cell_type' in expression_data.columns:
                source_genes = expression_data[expression_data['cell_type'] == source_cell]['gene'].unique()
                target_genes = expression_data[expression_data['cell_type'] == target_cell]['gene'].unique()
                
                supporting_data["expression_evidence"] = {
                    "source_genes_detected": len(source_genes),
                    "target_genes_detected": len(target_genes),
                    "source_top_genes": list(source_genes[:10]) if len(source_genes) > 0 else [],
                    "target_top_genes": list(target_genes[:10]) if len(target_genes) > 0 else []
                }
        
        return supporting_data
    
    def score_all_interactions(self, cell_types: List[str],
                             interaction_types: List[str],
                             expression_data: Optional[pd.DataFrame] = None,
                             functional_data: Optional[pd.DataFrame] = None,
                             secretome_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Score activity for all possible interactions"""
        
        scores = []
        
        for source in cell_types:
            for target in cell_types:
                if source != target:  # No self-interactions
                    for interaction_type in interaction_types:
                        score = self.score_interaction_activity(
                            source, target, interaction_type,
                            expression_data, functional_data, secretome_data
                        )
                        
                        scores.append({
                            'source_cell': score.source_cell,
                            'target_cell': score.target_cell, 
                            'interaction_type': score.interaction_type,
                            'overall_score': score.overall_score,
                            'confidence': score.confidence,
                            'literature_score': score.evidence_breakdown['literature'],
                            'expression_score': score.evidence_breakdown['expression'],
                            'functional_score': score.evidence_breakdown['functional'],
                            'structural_score': score.evidence_breakdown['structural'],
                            'num_lr_pairs': len(score.supporting_data['relevant_lr_pairs'])
                        })
        
        return pd.DataFrame(scores).sort_values('overall_score', ascending=False)


def create_interaction_activity_matrix(activity_scores: pd.DataFrame) -> pd.DataFrame:
    """Create interaction activity matrix for visualization"""
    
    # Pivot to create cell type x cell type matrix
    pivot_data = activity_scores.groupby(['source_cell', 'target_cell'])['overall_score'].max().reset_index()
    
    matrix = pivot_data.pivot(index='source_cell', columns='target_cell', values='overall_score')
    matrix = matrix.fillna(0)
    
    return matrix
