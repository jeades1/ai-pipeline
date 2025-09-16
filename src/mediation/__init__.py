"""
Mediation Pipeline Integration Module
Connects invitro functional outputs to clinical outcome predictions through causal mediation analysis.
Implements the core molecular→functional→clinical bridge for biomarker validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediationPathway(Enum):
    """Types of mediation pathways"""
    MOLECULAR_TO_FUNCTIONAL = "molecular_to_functional"
    FUNCTIONAL_TO_CLINICAL = "functional_to_clinical"
    FULL_MEDIATION = "molecular_to_functional_to_clinical"
    DIRECT_MOLECULAR_CLINICAL = "molecular_to_clinical_direct"


@dataclass
class MediationEvidence:
    """Evidence for a mediation relationship"""
    mediation_id: str
    pathway_type: MediationPathway
    
    # Entities in mediation path
    molecular_entity: str  # Gene, protein, metabolite
    functional_mediator: Optional[str]  # TEER, secretome, etc.
    clinical_outcome: str  # AKI, mortality, etc.
    
    # Statistical evidence
    direct_effect: float  # X → Y without mediator
    indirect_effect: float  # X → M → Y through mediator
    total_effect: float  # X → Y total
    mediation_proportion: float  # Indirect / Total effect
    
    # Significance testing
    p_value_direct: float
    p_value_indirect: float
    p_value_mediation: float
    confidence_interval: Tuple[float, float]
    
    # Effect sizes and directions
    molecular_to_functional_effect: Optional[float] = None
    functional_to_clinical_effect: Optional[float] = None
    effect_direction: str = "unknown"  # "positive", "negative", "bidirectional"
    
    # Supporting data
    sample_size: int = 0
    r_squared: float = 0.0
    bootstrap_replicates: int = 1000
    
    # Metadata
    assay_types: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    evidence_strength: str = "weak"  # "weak", "moderate", "strong"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MediationResult:
    """Results from mediation analysis"""
    mediation_evidence: MediationEvidence
    supporting_data: Dict[str, Any]
    model_diagnostics: Dict[str, float]
    validation_metrics: Dict[str, float]
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    stability_score: float = 0.0
    
    # Biological interpretation
    mechanism_hypothesis: str = ""
    intervention_potential: str = ""  # "high", "medium", "low"
    translational_readiness: str = ""  # "ready", "needs_validation", "early"


class MediationAnalyzer:
    """Core mediation analysis engine"""
    
    def __init__(self, kg_path: Optional[Path] = None):
        self.kg_path = kg_path
        self.mediation_results = {}
        self.functional_assay_mapping = self._initialize_assay_mapping()
        
    def _initialize_assay_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mapping between functional assays and clinical outcomes"""
        return {
            "teer": {
                "description": "Trans-epithelial electrical resistance",
                "biological_meaning": "Barrier integrity",
                "clinical_correlates": ["acute_kidney_injury", "barrier_dysfunction", "inflammation"],
                "measurement_units": "ohm_cm2",
                "expected_direction": "negative",  # Lower TEER indicates dysfunction
                "sensitivity": "high",
                "specificity": "moderate"
            },
            "permeability": {
                "description": "Barrier permeability",
                "biological_meaning": "Transport barrier function",
                "clinical_correlates": ["proteinuria", "kidney_dysfunction", "filtration_barrier_damage"],
                "measurement_units": "cm_s",
                "expected_direction": "positive",  # Higher permeability indicates dysfunction
                "sensitivity": "high",
                "specificity": "high"
            },
            "secretome": {
                "description": "Protein secretion",
                "biological_meaning": "Cellular stress response and signaling",
                "clinical_correlates": ["inflammation", "kidney_injury", "biomarker_release"],
                "measurement_units": "pg_ml",
                "expected_direction": "positive",  # Higher secretion indicates stress
                "sensitivity": "moderate",
                "specificity": "moderate"
            },
            "imaging": {
                "description": "Morphological changes",
                "biological_meaning": "Structural integrity and organization",
                "clinical_correlates": ["tissue_damage", "fibrosis", "apoptosis"],
                "measurement_units": "normalized_intensity",
                "expected_direction": "variable",
                "sensitivity": "moderate",
                "specificity": "high"
            },
            "scrna": {
                "description": "Single-cell gene expression",
                "biological_meaning": "Cellular state and pathway activity",
                "clinical_correlates": ["molecular_signatures", "pathway_dysregulation", "cell_death"],
                "measurement_units": "log2_tpm",
                "expected_direction": "variable",
                "sensitivity": "high",
                "specificity": "variable"
            }
        }
    
    def analyze_mediation_pathway(self, 
                                molecular_data: pd.DataFrame,
                                functional_data: pd.DataFrame,
                                clinical_data: pd.DataFrame,
                                target_outcome: str,
                                molecular_entity: str,
                                functional_mediator: Optional[str] = None) -> MediationResult:
        """
        Analyze mediation pathway: molecular → functional → clinical
        
        Args:
            molecular_data: Gene expression, protein levels, etc.
            functional_data: TEER, permeability, secretome, etc.
            clinical_data: Clinical outcomes and patient metadata
            target_outcome: Target clinical outcome (e.g., 'aki', 'mortality')
            molecular_entity: Gene/protein of interest
            functional_mediator: Functional assay type (auto-detect if None)
        
        Returns:
            MediationResult with evidence and statistics
        """
        
        # Prepare data for analysis
        merged_data = self._prepare_mediation_data(
            molecular_data, functional_data, clinical_data, 
            molecular_entity, target_outcome, functional_mediator
        )
        
        if merged_data.empty:
            logger.warning(f"No data available for mediation analysis: {molecular_entity} → {target_outcome}")
            return self._create_null_result(molecular_entity, target_outcome)
        
        # Determine best functional mediator if not specified
        if functional_mediator is None:
            functional_mediator = self._select_best_mediator(merged_data, target_outcome)
        
        # Perform mediation analysis
        mediation_stats = self._calculate_mediation_statistics(
            merged_data, molecular_entity, functional_mediator, target_outcome
        )
        
        # Create mediation evidence
        evidence = self._create_mediation_evidence(
            molecular_entity, functional_mediator, target_outcome, mediation_stats, merged_data
        )
        
        # Model diagnostics and validation
        diagnostics = self._calculate_model_diagnostics(merged_data, mediation_stats)
        validation_metrics = self._cross_validate_mediation(merged_data, molecular_entity, functional_mediator, target_outcome)
        
        # Biological interpretation
        mechanism_hypothesis = self._generate_mechanism_hypothesis(evidence)
        intervention_potential = self._assess_intervention_potential(evidence)
        translational_readiness = self._assess_translational_readiness(evidence, validation_metrics)
        
        result = MediationResult(
            mediation_evidence=evidence,
            supporting_data=self._extract_supporting_data(merged_data, mediation_stats),
            model_diagnostics=diagnostics,
            validation_metrics=validation_metrics,
            mechanism_hypothesis=mechanism_hypothesis,
            intervention_potential=intervention_potential,
            translational_readiness=translational_readiness
        )
        
        # Store result
        mediation_id = evidence.mediation_id
        self.mediation_results[mediation_id] = result
        
        logger.info(f"Completed mediation analysis: {mediation_id}")
        logger.info(f"Mediation proportion: {evidence.mediation_proportion:.3f}")
        logger.info(f"Evidence strength: {evidence.evidence_strength}")
        
        return result
    
    def _prepare_mediation_data(self, molecular_data: pd.DataFrame, functional_data: pd.DataFrame,
                               clinical_data: pd.DataFrame, molecular_entity: str, 
                               target_outcome: str, functional_mediator: Optional[str]) -> pd.DataFrame:
        """Prepare and merge data for mediation analysis"""
        
        # Extract molecular entity data
        if 'gene' in molecular_data.columns:
            mol_data = molecular_data[molecular_data['gene'] == molecular_entity].copy()
        else:
            logger.warning(f"No 'gene' column found in molecular data")
            return pd.DataFrame()
        
        if mol_data.empty:
            logger.warning(f"No data found for molecular entity: {molecular_entity}")
            return pd.DataFrame()
        
        # Extract functional data
        if functional_mediator:
            func_data = functional_data[functional_data['assay_type'] == functional_mediator].copy()
        else:
            func_data = functional_data.copy()
        
        # Extract clinical outcome data
        if target_outcome in clinical_data.columns:
            clin_data = clinical_data[['patient_id', target_outcome]].copy()
        else:
            logger.warning(f"Target outcome '{target_outcome}' not found in clinical data")
            return pd.DataFrame()
        
        # Merge datasets on patient_id
        # Start with molecular data
        merged = mol_data.copy()
        
        # Add functional data
        if not func_data.empty:
            if 'patient_id' in func_data.columns:
                merged = merged.merge(func_data, on='patient_id', how='inner', suffixes=('_mol', '_func'))
            else:
                logger.warning("No 'patient_id' column in functional data")
                return pd.DataFrame()
        
        # Add clinical data
        if 'patient_id' in merged.columns and 'patient_id' in clin_data.columns:
            merged = merged.merge(clin_data, on='patient_id', how='inner')
        else:
            logger.warning("Cannot merge clinical data - missing patient_id")
            return pd.DataFrame()
        
        # Rename columns for clarity
        if 'value_mol' in merged.columns:
            merged = merged.rename(columns={'value_mol': 'molecular_value'})
        if 'value_func' in merged.columns:
            merged = merged.rename(columns={'value_func': 'functional_value'})
        if 'value' in merged.columns and 'molecular_value' not in merged.columns:
            merged = merged.rename(columns={'value': 'molecular_value'})
        
        return merged
    
    def _select_best_mediator(self, merged_data: pd.DataFrame, target_outcome: str) -> str:
        """Select the best functional mediator for the analysis"""
        
        if 'assay_type' not in merged_data.columns:
            return "unknown"
        
        # Calculate correlation between each assay type and clinical outcome
        assay_correlations = {}
        
        for assay_type in merged_data['assay_type'].unique():
            assay_data = merged_data[merged_data['assay_type'] == assay_type]
            if len(assay_data) > 5 and 'functional_value' in assay_data.columns:
                corr = np.corrcoef(assay_data['functional_value'], assay_data[target_outcome])[0, 1]
                if not np.isnan(corr):
                    assay_correlations[assay_type] = abs(corr)
        
        if assay_correlations:
            best_mediator = max(assay_correlations.keys(), key=lambda x: assay_correlations[x])
            logger.info(f"Selected best mediator: {best_mediator} (correlation: {assay_correlations[best_mediator]:.3f})")
            return best_mediator
        else:
            return "unknown"
    
    def _calculate_mediation_statistics(self, data: pd.DataFrame, molecular_entity: str,
                                      functional_mediator: str, target_outcome: str) -> Dict[str, float]:
        """Calculate mediation statistics using Baron & Kenny approach"""
        
        try:
            # Extract variables
            X = np.asarray(data['molecular_value'].values, dtype=float)  # Molecular entity
            M = np.asarray(data.get('functional_value', pd.Series(np.zeros(len(data)))).values, dtype=float)  # Functional mediator
            Y = np.asarray(data[target_outcome].values, dtype=float)  # Clinical outcome
            
            # Remove NaN values
            mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
            X, M, Y = X[mask], M[mask], Y[mask]
            
            if len(X) < 10:
                logger.warning(f"Insufficient data for mediation analysis: {len(X)} samples")
                return self._create_null_statistics()
            
            # Standardize variables
            X = (X - np.mean(X)) / np.std(X) if np.std(X) > 0 else X
            M = (M - np.mean(M)) / np.std(M) if np.std(M) > 0 else M
            Y = (Y - np.mean(Y)) / np.std(Y) if np.std(Y) > 0 else Y
            
            # Calculate regression coefficients
            # Path c: X → Y (total effect)
            c = np.corrcoef(X, Y)[0, 1] if np.std(Y) > 0 else 0.0
            
            # Path a: X → M
            a = np.corrcoef(X, M)[0, 1] if np.std(M) > 0 else 0.0
            
            # Path b: M → Y (controlling for X)
            # Use partial correlation: corr(M,Y) - corr(M,X)*corr(X,Y) / sqrt((1-corr(M,X)^2)*(1-corr(X,Y)^2))
            if np.std(M) > 0 and np.std(Y) > 0:
                r_my = np.corrcoef(M, Y)[0, 1]
                r_mx = np.corrcoef(M, X)[0, 1]
                r_xy = np.corrcoef(X, Y)[0, 1]
                
                if abs(r_mx) < 0.99 and abs(r_xy) < 0.99:
                    b = (r_my - r_mx * r_xy) / np.sqrt((1 - r_mx**2) * (1 - r_xy**2))
                else:
                    b = 0.0
            else:
                b = 0.0
            
            # Path c': X → Y (direct effect, controlling for M)
            # Direct effect = total effect - indirect effect
            indirect_effect = a * b
            direct_effect = c - indirect_effect
            
            # Mediation proportion
            mediation_proportion = indirect_effect / c if abs(c) > 1e-6 else 0.0
            
            # Statistical significance (simplified t-tests)
            n = len(X)
            se_c = 1 / np.sqrt(n - 3) if n > 3 else 1.0
            se_a = 1 / np.sqrt(n - 3) if n > 3 else 1.0
            se_b = 1 / np.sqrt(n - 3) if n > 3 else 1.0
            se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2) if se_a > 0 and se_b > 0 else 1.0
            
            # Calculate p-values (approximate)
            from scipy.stats import t
            df = max(n - 3, 1)
            
            t_c = c / se_c if se_c > 0 else 0.0
            t_indirect = indirect_effect / se_indirect if se_indirect > 0 else 0.0
            
            p_direct = 2 * (1 - t.cdf(abs(t_c), df))
            p_indirect = 2 * (1 - t.cdf(abs(t_indirect), df))
            p_mediation = p_indirect  # Simplified
            
            # R-squared approximation
            r_squared = c**2 if not np.isnan(c) else 0.0
            
            return {
                'direct_effect': float(direct_effect),
                'indirect_effect': float(indirect_effect),
                'total_effect': float(c),
                'mediation_proportion': float(mediation_proportion),
                'path_a': float(a),
                'path_b': float(b),
                'path_c': float(c),
                'path_c_prime': float(direct_effect),
                'p_value_direct': float(p_direct),
                'p_value_indirect': float(p_indirect),
                'p_value_mediation': float(p_mediation),
                'r_squared': float(r_squared),
                'sample_size': int(n),
                'se_direct': float(se_c),
                'se_indirect': float(se_indirect)
            }
            
        except Exception as e:
            logger.error(f"Error in mediation calculation: {e}")
            return self._create_null_statistics()
    
    def _create_null_statistics(self) -> Dict[str, float]:
        """Create null statistics when analysis fails"""
        return {
            'direct_effect': 0.0,
            'indirect_effect': 0.0,
            'total_effect': 0.0,
            'mediation_proportion': 0.0,
            'path_a': 0.0,
            'path_b': 0.0,
            'path_c': 0.0,
            'path_c_prime': 0.0,
            'p_value_direct': 1.0,
            'p_value_indirect': 1.0,
            'p_value_mediation': 1.0,
            'r_squared': 0.0,
            'sample_size': 0,
            'se_direct': 1.0,
            'se_indirect': 1.0
        }
    
    def _create_mediation_evidence(self, molecular_entity: str, functional_mediator: str,
                                 target_outcome: str, stats: Dict[str, float], 
                                 data: pd.DataFrame) -> MediationEvidence:
        """Create MediationEvidence object from statistics"""
        
        mediation_id = f"{molecular_entity}_{functional_mediator}_{target_outcome}"
        
        # Determine pathway type
        if stats['mediation_proportion'] > 0.3:
            pathway_type = MediationPathway.FULL_MEDIATION
        elif abs(stats['indirect_effect']) > abs(stats['direct_effect']):
            pathway_type = MediationPathway.MOLECULAR_TO_FUNCTIONAL
        else:
            pathway_type = MediationPathway.DIRECT_MOLECULAR_CLINICAL
        
        # Effect direction
        if stats['total_effect'] > 0.1:
            effect_direction = "positive"
        elif stats['total_effect'] < -0.1:
            effect_direction = "negative"
        else:
            effect_direction = "minimal"
        
        # Evidence strength
        if (stats['p_value_mediation'] < 0.05 and 
            abs(stats['mediation_proportion']) > 0.3 and 
            stats['sample_size'] > 30):
            evidence_strength = "strong"
        elif (stats['p_value_mediation'] < 0.1 and 
              abs(stats['mediation_proportion']) > 0.1 and 
              stats['sample_size'] > 10):
            evidence_strength = "moderate"
        else:
            evidence_strength = "weak"
        
        # Confidence interval (approximate)
        margin_error = 1.96 * stats['se_indirect']
        ci_lower = stats['indirect_effect'] - margin_error
        ci_upper = stats['indirect_effect'] + margin_error
        
        # Extract metadata
        assay_types = data['assay_type'].unique().tolist() if 'assay_type' in data.columns else []
        cell_types = data['cell_type'].unique().tolist() if 'cell_type' in data.columns else []
        conditions = data['condition'].unique().tolist() if 'condition' in data.columns else []
        
        return MediationEvidence(
            mediation_id=mediation_id,
            pathway_type=pathway_type,
            molecular_entity=molecular_entity,
            functional_mediator=functional_mediator,
            clinical_outcome=target_outcome,
            direct_effect=stats['direct_effect'],
            indirect_effect=stats['indirect_effect'],
            total_effect=stats['total_effect'],
            mediation_proportion=stats['mediation_proportion'],
            p_value_direct=stats['p_value_direct'],
            p_value_indirect=stats['p_value_indirect'],
            p_value_mediation=stats['p_value_mediation'],
            confidence_interval=(ci_lower, ci_upper),
            molecular_to_functional_effect=stats['path_a'],
            functional_to_clinical_effect=stats['path_b'],
            effect_direction=effect_direction,
            sample_size=int(stats['sample_size']),
            r_squared=stats['r_squared'],
            assay_types=assay_types,
            cell_types=cell_types,
            conditions=conditions,
            evidence_strength=evidence_strength
        )
    
    def _create_null_result(self, molecular_entity: str, target_outcome: str) -> MediationResult:
        """Create null result when analysis cannot be performed"""
        
        null_stats = self._create_null_statistics()
        null_evidence = MediationEvidence(
            mediation_id=f"{molecular_entity}_null_{target_outcome}",
            pathway_type=MediationPathway.DIRECT_MOLECULAR_CLINICAL,
            molecular_entity=molecular_entity,
            functional_mediator=None,
            clinical_outcome=target_outcome,
            direct_effect=0.0,
            indirect_effect=0.0,
            total_effect=0.0,
            mediation_proportion=0.0,
            p_value_direct=1.0,
            p_value_indirect=1.0,
            p_value_mediation=1.0,
            confidence_interval=(0.0, 0.0),
            evidence_strength="none"
        )
        
        return MediationResult(
            mediation_evidence=null_evidence,
            supporting_data={},
            model_diagnostics={'sample_size': 0.0, 'r_squared': 0.0},
            validation_metrics={'cv_score': 0.0},
            mechanism_hypothesis="Insufficient data for analysis",
            intervention_potential="unknown",
            translational_readiness="needs_data"
        )
    
    def _calculate_model_diagnostics(self, data: pd.DataFrame, stats: Dict[str, float]) -> Dict[str, float]:
        """Calculate model diagnostic metrics"""
        
        return {
            'sample_size': stats['sample_size'],
            'r_squared': stats['r_squared'],
            'effect_size': abs(stats['total_effect']),
            'mediation_strength': abs(stats['mediation_proportion']),
            'path_a_strength': abs(stats['path_a']),
            'path_b_strength': abs(stats['path_b']),
            'data_completeness': len(data) / max(len(data), 1),
            'variable_correlation': stats.get('max_correlation', 0.0)
        }
    
    def _cross_validate_mediation(self, data: pd.DataFrame, molecular_entity: str,
                                functional_mediator: str, target_outcome: str) -> Dict[str, float]:
        """Perform cross-validation of mediation analysis"""
        
        if len(data) < 20:
            return {'cv_score': 0.0, 'stability_score': 0.0}
        
        # Simple bootstrap validation
        n_bootstrap = 100
        mediation_props = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data), size=len(data), replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            # Calculate mediation statistics
            bootstrap_stats = self._calculate_mediation_statistics(
                bootstrap_data, molecular_entity, functional_mediator, target_outcome
            )
            
            mediation_props.append(bootstrap_stats['mediation_proportion'])
        
        # Calculate stability metrics
        cv_score = np.mean(mediation_props)
        stability_score = 1.0 - (np.std(mediation_props) / (abs(cv_score) + 1e-6))
        
        return {
            'cv_score': float(cv_score),
            'stability_score': float(max(0.0, float(stability_score))),
            'bootstrap_std': float(np.std(mediation_props)),
            'bootstrap_samples': n_bootstrap
        }
    
    def _extract_supporting_data(self, data: pd.DataFrame, stats: Dict[str, float]) -> Dict[str, Any]:
        """Extract supporting data for the mediation analysis"""
        
        supporting_data = {
            'raw_statistics': stats,
            'data_summary': {
                'n_samples': len(data),
                'n_patients': data['patient_id'].nunique() if 'patient_id' in data.columns else 0,
                'molecular_mean': data['molecular_value'].mean() if 'molecular_value' in data.columns else 0,
                'molecular_std': data['molecular_value'].std() if 'molecular_value' in data.columns else 0,
                'functional_mean': data['functional_value'].mean() if 'functional_value' in data.columns else 0,
                'functional_std': data['functional_value'].std() if 'functional_value' in data.columns else 0
            }
        }
        
        # Add pathway coefficients
        supporting_data['pathway_coefficients'] = {
            'molecular_to_functional': stats['path_a'],
            'functional_to_clinical': stats['path_b'],
            'molecular_to_clinical_total': stats['path_c'],
            'molecular_to_clinical_direct': stats['path_c_prime']
        }
        
        return supporting_data
    
    def _generate_mechanism_hypothesis(self, evidence: MediationEvidence) -> str:
        """Generate biological mechanism hypothesis"""
        
        mol_entity = evidence.molecular_entity
        func_mediator = evidence.functional_mediator or "unknown"
        outcome = evidence.clinical_outcome
        direction = evidence.effect_direction
        
        # Get functional assay information
        assay_info = self.functional_assay_mapping.get(func_mediator, {})
        bio_meaning = assay_info.get('biological_meaning', 'tissue function')
        
        if evidence.mediation_proportion > 0.3:
            return (f"{mol_entity} {direction}ly affects {bio_meaning} ({func_mediator}), "
                   f"which mediates {direction} effects on {outcome}. "
                   f"Mediation proportion: {evidence.mediation_proportion:.1%}")
        elif abs(evidence.direct_effect) > abs(evidence.indirect_effect):
            return (f"{mol_entity} has primarily direct {direction} effects on {outcome}, "
                   f"with minimal mediation through {bio_meaning}")
        else:
            return (f"Weak evidence for {mol_entity} affecting {outcome} through {bio_meaning}")
    
    def _assess_intervention_potential(self, evidence: MediationEvidence) -> str:
        """Assess potential for therapeutic intervention"""
        
        if (evidence.evidence_strength == "strong" and 
            abs(evidence.mediation_proportion) > 0.3 and
            evidence.p_value_mediation < 0.05):
            return "high"
        elif (evidence.evidence_strength == "moderate" and
              abs(evidence.mediation_proportion) > 0.1):
            return "medium"
        else:
            return "low"
    
    def _assess_translational_readiness(self, evidence: MediationEvidence, 
                                      validation_metrics: Dict[str, float]) -> str:
        """Assess readiness for clinical translation"""
        
        stability = validation_metrics.get('stability_score', 0.0)
        
        if (evidence.evidence_strength == "strong" and
            evidence.sample_size > 50 and
            stability > 0.7):
            return "ready"
        elif (evidence.evidence_strength in ["moderate", "strong"] and
              evidence.sample_size > 20 and
              stability > 0.5):
            return "needs_validation"
        else:
            return "early"


class MediationPipeline:
    """High-level pipeline for systematic mediation analysis"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/mediation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = MediationAnalyzer()
        self.results = {}
        
    def run_systematic_analysis(self, 
                              molecular_data: pd.DataFrame,
                              functional_data: pd.DataFrame, 
                              clinical_data: pd.DataFrame,
                              target_outcomes: List[str],
                              molecular_entities: Optional[List[str]] = None) -> Dict[str, MediationResult]:
        """
        Run systematic mediation analysis across multiple entities and outcomes
        
        Args:
            molecular_data: Gene/protein expression data
            functional_data: Invitro assay results  
            clinical_data: Clinical outcomes and metadata
            target_outcomes: List of clinical outcomes to analyze
            molecular_entities: List of genes/proteins (auto-detect if None)
            
        Returns:
            Dictionary of mediation results
        """
        
        logger.info("Starting systematic mediation analysis")
        
        # Auto-detect molecular entities if not provided
        if molecular_entities is None:
            if 'gene' in molecular_data.columns:
                molecular_entities = molecular_data['gene'].unique().tolist()[:20]  # Limit for demo
            else:
                logger.error("No 'gene' column found and no molecular entities specified")
                return {}
        
        # Ensure we have molecular entities
        if not molecular_entities:
            logger.error("No molecular entities available for analysis")
            return {}
        
        # Run analysis for each combination
        results = {}
        total_analyses = len(molecular_entities) * len(target_outcomes)
        completed = 0
        
        for molecular_entity in molecular_entities:
            for target_outcome in target_outcomes:
                try:
                    logger.info(f"Analyzing {molecular_entity} → {target_outcome} ({completed+1}/{total_analyses})")
                    
                    result = self.analyzer.analyze_mediation_pathway(
                        molecular_data=molecular_data,
                        functional_data=functional_data,
                        clinical_data=clinical_data,
                        target_outcome=target_outcome,
                        molecular_entity=molecular_entity
                    )
                    
                    analysis_id = f"{molecular_entity}_{target_outcome}"
                    results[analysis_id] = result
                    
                    completed += 1
                    
                except Exception as e:
                    logger.error(f"Error analyzing {molecular_entity} → {target_outcome}: {e}")
                    continue
        
        logger.info(f"Completed {completed}/{total_analyses} mediation analyses")
        
        # Store results
        self.results.update(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, MediationResult]) -> None:
        """Save mediation results to files"""
        
        # Save detailed results as JSON
        results_data = {}
        for analysis_id, result in results.items():
            results_data[analysis_id] = {
                'mediation_evidence': {
                    'mediation_id': result.mediation_evidence.mediation_id,
                    'pathway_type': result.mediation_evidence.pathway_type.value,
                    'molecular_entity': result.mediation_evidence.molecular_entity,
                    'functional_mediator': result.mediation_evidence.functional_mediator,
                    'clinical_outcome': result.mediation_evidence.clinical_outcome,
                    'direct_effect': result.mediation_evidence.direct_effect,
                    'indirect_effect': result.mediation_evidence.indirect_effect,
                    'total_effect': result.mediation_evidence.total_effect,
                    'mediation_proportion': result.mediation_evidence.mediation_proportion,
                    'p_value_mediation': result.mediation_evidence.p_value_mediation,
                    'evidence_strength': result.mediation_evidence.evidence_strength,
                    'sample_size': result.mediation_evidence.sample_size
                },
                'validation_metrics': result.validation_metrics,
                'mechanism_hypothesis': result.mechanism_hypothesis,
                'intervention_potential': result.intervention_potential,
                'translational_readiness': result.translational_readiness
            }
        
        results_file = self.output_dir / "mediation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary as CSV
        summary_data = []
        for analysis_id, result in results.items():
            evidence = result.mediation_evidence
            summary_data.append({
                'analysis_id': analysis_id,
                'molecular_entity': evidence.molecular_entity,
                'functional_mediator': evidence.functional_mediator,
                'clinical_outcome': evidence.clinical_outcome,
                'mediation_proportion': evidence.mediation_proportion,
                'p_value': evidence.p_value_mediation,
                'evidence_strength': evidence.evidence_strength,
                'intervention_potential': result.intervention_potential,
                'translational_readiness': result.translational_readiness,
                'sample_size': evidence.sample_size
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "mediation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved summary to {summary_file}")
    
    def generate_mediation_report(self, min_evidence_strength: str = "moderate") -> pd.DataFrame:
        """Generate comprehensive mediation analysis report"""
        
        if not self.results:
            logger.warning("No mediation results available for report")
            return pd.DataFrame()
        
        # Filter by evidence strength
        strength_order = {"weak": 0, "moderate": 1, "strong": 2, "none": -1}
        min_strength_value = strength_order.get(min_evidence_strength, 1)
        
        filtered_results = {
            analysis_id: result for analysis_id, result in self.results.items()
            if strength_order.get(result.mediation_evidence.evidence_strength, -1) >= min_strength_value
        }
        
        if not filtered_results:
            logger.warning(f"No results meet minimum evidence strength: {min_evidence_strength}")
            return pd.DataFrame()
        
        # Create comprehensive report
        report_data = []
        for analysis_id, result in filtered_results.items():
            evidence = result.mediation_evidence
            
            report_data.append({
                'molecular_entity': evidence.molecular_entity,
                'functional_mediator': evidence.functional_mediator,
                'clinical_outcome': evidence.clinical_outcome,
                'pathway_type': evidence.pathway_type.value,
                'mediation_proportion': evidence.mediation_proportion,
                'total_effect': evidence.total_effect,
                'direct_effect': evidence.direct_effect,
                'indirect_effect': evidence.indirect_effect,
                'p_value_mediation': evidence.p_value_mediation,
                'evidence_strength': evidence.evidence_strength,
                'sample_size': evidence.sample_size,
                'cv_stability': result.validation_metrics.get('stability_score', 0.0),
                'mechanism_hypothesis': result.mechanism_hypothesis,
                'intervention_potential': result.intervention_potential,
                'translational_readiness': result.translational_readiness,
                'effect_direction': evidence.effect_direction
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by evidence strength and mediation proportion
        strength_map = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}
        report_df['strength_numeric'] = report_df['evidence_strength'].map(strength_map)
        report_df = report_df.sort_values(['strength_numeric', 'mediation_proportion'], 
                                        ascending=[False, False])
        
        # Save report
        report_file = self.output_dir / f"mediation_report_{min_evidence_strength}.csv"
        report_df.to_csv(report_file, index=False)
        
        logger.info(f"Generated mediation report: {report_file}")
        logger.info(f"Found {len(report_df)} mediation pathways with {min_evidence_strength}+ evidence")
        
        return report_df


def create_demo_mediation_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create demonstration data for mediation analysis"""
    
    np.random.seed(42)
    n_patients = 100
    
    # Generate patient IDs
    patient_ids = [f"DEMO_{i:03d}" for i in range(n_patients)]
    
    # Molecular data (gene expression)
    molecular_data = []
    genes = ["VEGFA", "TGFB1", "CCL2", "PDGFB", "FN1", "ANGPT1"]
    
    for patient_id in patient_ids:
        for gene in genes:
            # Simulate gene expression with some biological correlation
            base_expression = np.random.normal(5, 1)  # Log2 scale
            molecular_data.append({
                'patient_id': patient_id,
                'gene': gene,
                'value': base_expression
            })
    
    molecular_df = pd.DataFrame(molecular_data)
    
    # Functional data (invitro assays)
    functional_data = []
    assay_types = ["teer", "permeability", "secretome"]
    
    for patient_id in patient_ids:
        for assay_type in assay_types:
            # Simulate functional readouts correlated with gene expression
            patient_mol_data = molecular_df[molecular_df['patient_id'] == patient_id]
            
            if assay_type == "teer":
                # TEER inversely correlated with inflammatory genes
                vegfa_expr = patient_mol_data[patient_mol_data['gene'] == 'VEGFA']['value'].values[0]
                tgfb1_expr = patient_mol_data[patient_mol_data['gene'] == 'TGFB1']['value'].values[0]
                functional_value = 1000 - (vegfa_expr + tgfb1_expr) * 50 + np.random.normal(0, 100)
                
            elif assay_type == "permeability":
                # Permeability positively correlated with barrier disruption genes
                ccl2_expr = patient_mol_data[patient_mol_data['gene'] == 'CCL2']['value'].values[0]
                functional_value = 0.001 + ccl2_expr * 0.0002 + np.random.normal(0, 0.0005)
                
            else:  # secretome
                # Protein secretion correlated with stress response genes
                tgfb1_expr = patient_mol_data[patient_mol_data['gene'] == 'TGFB1']['value'].values[0]
                functional_value = 10 + tgfb1_expr * 20 + np.random.normal(0, 30)
            
            functional_data.append({
                'patient_id': patient_id,
                'assay_type': assay_type,
                'value': max(0, functional_value)  # Ensure positive values
            })
    
    functional_df = pd.DataFrame(functional_data)
    
    # Clinical data (outcomes)
    clinical_data = []
    
    for patient_id in patient_ids:
        # Get functional values for this patient
        patient_func_data = functional_df[functional_df['patient_id'] == patient_id]
        
        # AKI outcome correlated with functional dysfunction
        teer_value = patient_func_data[patient_func_data['assay_type'] == 'teer']['value'].values[0]
        perm_value = patient_func_data[patient_func_data['assay_type'] == 'permeability']['value'].values[0]
        
        # Lower TEER and higher permeability increase AKI risk
        aki_risk = 0.1 + (1000 - teer_value) / 5000 + perm_value * 200
        aki_outcome = 1 if np.random.random() < aki_risk else 0
        
        # Mortality correlated with AKI and secretome
        secr_value = patient_func_data[patient_func_data['assay_type'] == 'secretome']['value'].values[0]
        mort_risk = 0.05 + aki_outcome * 0.2 + secr_value / 1000
        mortality_outcome = 1 if np.random.random() < mort_risk else 0
        
        clinical_data.append({
            'patient_id': patient_id,
            'aki': aki_outcome,
            'mortality': mortality_outcome,
            'age': np.random.randint(40, 80),
            'sex': np.random.choice(['male', 'female'])
        })
    
    clinical_df = pd.DataFrame(clinical_data)
    
    return molecular_df, functional_df, clinical_df


def run_mediation_demo():
    """Run demonstration of mediation pipeline"""
    
    print("\n🔗 MEDIATION PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Create demo data
    print("📊 Generating demonstration data...")
    molecular_data, functional_data, clinical_data = create_demo_mediation_data()
    
    print(f"   Molecular data: {len(molecular_data)} measurements")
    print(f"   Functional data: {len(functional_data)} assay results")
    print(f"   Clinical data: {len(clinical_data)} patients")
    
    # Initialize pipeline
    pipeline = MediationPipeline(output_dir=Path("demo_outputs/mediation"))
    
    # Run systematic analysis
    print("\n🔬 Running mediation analysis...")
    results = pipeline.run_systematic_analysis(
        molecular_data=molecular_data,
        functional_data=functional_data,
        clinical_data=clinical_data,
        target_outcomes=['aki', 'mortality'],
        molecular_entities=['VEGFA', 'TGFB1', 'CCL2']
    )
    
    print(f"\n📈 Analysis completed: {len(results)} mediation pathways analyzed")
    
    # Generate report
    print("\n📋 Generating mediation report...")
    report = pipeline.generate_mediation_report(min_evidence_strength="weak")
    
    if not report.empty:
        print(f"\n🎯 Top mediation pathways:")
        top_results = report.head(5)
        
        for _, row in top_results.iterrows():
            print(f"\n   {row['molecular_entity']} → {row['functional_mediator']} → {row['clinical_outcome']}")
            print(f"      Mediation: {row['mediation_proportion']:.2%}")
            print(f"      Evidence: {row['evidence_strength']}")
            print(f"      Intervention potential: {row['intervention_potential']}")
            print(f"      Mechanism: {row['mechanism_hypothesis'][:100]}...")
    else:
        print("\n❌ No significant mediation pathways found")
    
    return results, report


if __name__ == "__main__":
    run_mediation_demo()
