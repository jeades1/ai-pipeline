"""
Tissue-Chip Integration Demo

Demonstration system showing how personalized biomarker predictions can guide 
tissue-chip experiment design and parameter selection for lab validation readiness.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import warnings

warnings.filterwarnings("ignore")

# Import our personalized biomarker components
from .avatar_integration import PatientProfile, PersonalizedBiomarkerEngine
from .trajectory_prediction import TrajectoryPredictor
from .clinical_decision_support import ClinicalDecisionSupportAPI
from .multi_outcome_prediction import MultiOutcomePredictionEngine

logger = logging.getLogger(__name__)


class TissueChipType(Enum):
    """Types of tissue-chip platforms"""

    HEART_ON_CHIP = "heart_on_chip"
    LIVER_ON_CHIP = "liver_on_chip"
    KIDNEY_ON_CHIP = "kidney_on_chip"
    LUNG_ON_CHIP = "lung_on_chip"
    VASCULATURE_ON_CHIP = "vasculature_on_chip"
    MULTI_ORGAN_CHIP = "multi_organ_chip"


class ExperimentObjective(Enum):
    """Experimental objectives for tissue-chip studies"""

    BIOMARKER_VALIDATION = "biomarker_validation"
    DRUG_SCREENING = "drug_screening"
    TOXICITY_TESTING = "toxicity_testing"
    DISEASE_MODELING = "disease_modeling"
    PERSONALIZED_MEDICINE = "personalized_medicine"
    MECHANISM_STUDY = "mechanism_study"


class ExperimentalCondition(Enum):
    """Experimental conditions to test"""

    CONTROL = "control"
    DISEASE_STATE = "disease_state"
    DRUG_TREATMENT = "drug_treatment"
    BIOMARKER_MODULATION = "biomarker_modulation"
    STRESS_CONDITIONS = "stress_conditions"
    GENETIC_VARIANTS = "genetic_variants"


@dataclass
class TissueChipSpecification:
    """Specifications for tissue-chip experiment design"""

    chip_type: TissueChipType
    experiment_objective: ExperimentObjective

    # Cell sources and conditions
    cell_sources: List[str] = field(default_factory=list)
    patient_derived_cells: bool = False
    genetic_background: Optional[str] = None

    # Culture conditions
    culture_media_composition: Dict[str, float] = field(default_factory=dict)
    flow_rates: Dict[str, float] = field(default_factory=dict)  # μL/min
    oxygen_levels: float = 21.0  # % O2

    # Biomarker monitoring
    target_biomarkers: List[str] = field(default_factory=list)
    sampling_timepoints: List[float] = field(default_factory=list)  # hours
    detection_methods: Dict[str, str] = field(default_factory=dict)

    # Experimental variables
    conditions_to_test: List[ExperimentalCondition] = field(default_factory=list)
    dose_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    duration: float = 72.0  # hours

    # Validation targets
    expected_biomarker_changes: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    clinical_correlation_targets: List[str] = field(default_factory=list)

    # Quality metrics
    viability_thresholds: Dict[str, float] = field(default_factory=dict)
    reproducibility_targets: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentRecommendation:
    """Recommendations for tissue-chip experiments"""

    recommendation_id: str
    patient_profile: PatientProfile
    chip_specification: TissueChipSpecification

    # Prioritization
    priority_score: float  # 0-1, higher = more important
    clinical_relevance: float  # 0-1, higher = more clinically relevant
    feasibility_score: float  # 0-1, higher = more feasible

    # Predictions
    predicted_outcomes: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Resource requirements
    estimated_cost: float = 0.0  # USD
    estimated_duration: float = 0.0  # days
    required_expertise: List[str] = field(default_factory=list)

    # Success criteria
    success_metrics: Dict[str, float] = field(default_factory=dict)
    validation_endpoints: List[str] = field(default_factory=list)

    # Risk assessment
    technical_risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

    # Clinical translation potential
    translation_pathway: str = ""
    regulatory_considerations: List[str] = field(default_factory=list)

    # Metadata
    generated_date: datetime = field(default_factory=datetime.now)
    rationale: str = ""


@dataclass
class ExperimentalProtocol:
    """Detailed experimental protocol for tissue-chip studies"""

    protocol_id: str
    title: str

    # Protocol steps
    preparation_steps: List[str] = field(default_factory=list)
    execution_steps: List[str] = field(default_factory=list)
    analysis_steps: List[str] = field(default_factory=list)

    # Timeline
    total_duration: float = 0.0  # hours
    critical_timepoints: List[Tuple[float, str]] = field(default_factory=list)

    # Materials and reagents
    required_materials: Dict[str, str] = field(default_factory=dict)
    reagent_concentrations: Dict[str, float] = field(default_factory=dict)

    # Equipment specifications
    required_equipment: List[str] = field(default_factory=list)
    instrument_settings: Dict[str, Any] = field(default_factory=dict)

    # Quality control
    qc_checkpoints: List[str] = field(default_factory=list)
    acceptance_criteria: Dict[str, str] = field(default_factory=dict)

    # Data collection
    measurement_schedule: List[Tuple[float, List[str]]] = field(default_factory=list)
    data_analysis_methods: List[str] = field(default_factory=list)

    # Safety considerations
    safety_precautions: List[str] = field(default_factory=list)
    waste_disposal: List[str] = field(default_factory=list)


class TissueChipDesigner:
    """Designs tissue-chip experiments based on personalized biomarker predictions"""

    def __init__(self):
        # Initialize personalized biomarker components
        self.biomarker_engine = PersonalizedBiomarkerEngine()
        self.trajectory_predictor = TrajectoryPredictor()
        self.clinical_support = ClinicalDecisionSupportAPI()
        self.outcome_engine = MultiOutcomePredictionEngine()

        # Tissue-chip capabilities database
        self.chip_capabilities = self._initialize_chip_capabilities()
        self.biomarker_detection_methods = self._initialize_detection_methods()
        self.cost_models = self._initialize_cost_models()

    def _initialize_chip_capabilities(self) -> Dict[TissueChipType, Dict[str, Any]]:
        """Initialize tissue-chip platform capabilities"""
        return {
            TissueChipType.HEART_ON_CHIP: {
                "primary_biomarkers": ["CRP", "Troponin", "BNP", "PCSK9"],
                "secondary_biomarkers": ["APOB", "LPA"],
                "cell_types": ["cardiomyocytes", "endothelial_cells", "fibroblasts"],
                "disease_models": [
                    "myocardial_infarction",
                    "heart_failure",
                    "arrhythmia",
                ],
                "drug_targets": ["ACE_inhibitors", "beta_blockers", "statins"],
                "culture_duration": (24, 168),  # hours
                "throughput": "medium",
                "cost_per_chip": 150.0,
                "expertise_required": ["cardiac_biology", "microfluidics"],
            },
            TissueChipType.LIVER_ON_CHIP: {
                "primary_biomarkers": ["ALT", "AST", "HMGCR", "APOB"],
                "secondary_biomarkers": ["CRP", "albumin"],
                "cell_types": ["hepatocytes", "stellate_cells", "kupffer_cells"],
                "disease_models": ["hepatotoxicity", "NASH", "drug_metabolism"],
                "drug_targets": ["statins", "metformin", "hepatotoxins"],
                "culture_duration": (48, 336),  # hours
                "throughput": "high",
                "cost_per_chip": 120.0,
                "expertise_required": ["hepatology", "toxicology"],
            },
            TissueChipType.KIDNEY_ON_CHIP: {
                "primary_biomarkers": ["creatinine", "BUN", "albumin"],
                "secondary_biomarkers": ["CRP", "electrolytes"],
                "cell_types": [
                    "tubular_epithelial",
                    "glomerular_endothelial",
                    "podocytes",
                ],
                "disease_models": ["nephrotoxicity", "CKD", "AKI"],
                "drug_targets": ["ACE_inhibitors", "diuretics", "nephrotoxins"],
                "culture_duration": (72, 168),  # hours
                "throughput": "medium",
                "cost_per_chip": 180.0,
                "expertise_required": ["nephrology", "renal_physiology"],
            },
            TissueChipType.VASCULATURE_ON_CHIP: {
                "primary_biomarkers": ["CRP", "APOB", "LPA", "PCSK9"],
                "secondary_biomarkers": ["NO", "endothelin", "VCAM"],
                "cell_types": ["endothelial_cells", "smooth_muscle_cells", "pericytes"],
                "disease_models": ["atherosclerosis", "inflammation", "thrombosis"],
                "drug_targets": ["statins", "anti_inflammatories", "anticoagulants"],
                "culture_duration": (24, 120),  # hours
                "throughput": "high",
                "cost_per_chip": 100.0,
                "expertise_required": ["vascular_biology", "endothelial_function"],
            },
            TissueChipType.MULTI_ORGAN_CHIP: {
                "primary_biomarkers": ["CRP", "APOB", "PCSK9", "LPA", "HMGCR"],
                "secondary_biomarkers": ["cytokines", "metabolites"],
                "cell_types": ["multiple_organ_specific"],
                "disease_models": ["systemic_disease", "multi_organ_toxicity"],
                "drug_targets": ["systemic_therapeutics"],
                "culture_duration": (168, 504),  # hours
                "throughput": "low",
                "cost_per_chip": 500.0,
                "expertise_required": ["systems_biology", "multi_organ_physiology"],
            },
        }

    def _initialize_detection_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize biomarker detection methods for tissue chips"""
        return {
            "CRP": {
                "primary_method": "ELISA",
                "sensitivity": 0.1,  # mg/L
                "dynamic_range": (0.1, 100),  # mg/L
                "sample_volume": 50,  # μL
                "assay_time": 2.0,  # hours
                "cost_per_sample": 15.0,
            },
            "APOB": {
                "primary_method": "immunoturbidimetry",
                "sensitivity": 1.0,  # mg/dL
                "dynamic_range": (10, 200),  # mg/dL
                "sample_volume": 25,  # μL
                "assay_time": 1.0,  # hours
                "cost_per_sample": 12.0,
            },
            "PCSK9": {
                "primary_method": "ELISA",
                "sensitivity": 10,  # ng/mL
                "dynamic_range": (10, 1000),  # ng/mL
                "sample_volume": 100,  # μL
                "assay_time": 3.0,  # hours
                "cost_per_sample": 25.0,
            },
            "LPA": {
                "primary_method": "immunonephelometry",
                "sensitivity": 1.0,  # mg/dL
                "dynamic_range": (1, 100),  # mg/dL
                "sample_volume": 50,  # μL
                "assay_time": 1.5,  # hours
                "cost_per_sample": 20.0,
            },
            "HMGCR": {
                "primary_method": "enzymatic_assay",
                "sensitivity": 5,  # U/L
                "dynamic_range": (5, 500),  # U/L
                "sample_volume": 75,  # μL
                "assay_time": 2.5,  # hours
                "cost_per_sample": 18.0,
            },
        }

    def _initialize_cost_models(self) -> Dict[str, float]:
        """Initialize cost models for different experiment components"""
        return {
            "personnel_per_hour": 75.0,  # USD/hour
            "equipment_overhead": 0.2,  # 20% of direct costs
            "consumables_markup": 1.3,  # 30% markup
            "facility_cost_per_day": 200.0,  # USD/day
            "analysis_per_sample": 50.0,  # USD/sample
            "protocol_development": 2000.0,  # USD one-time
            "validation_studies": 5000.0,  # USD
        }

    def recommend_experiments(
        self,
        patient: PatientProfile,
        research_objectives: List[ExperimentObjective],
        budget_limit: Optional[float] = None,
    ) -> List[ExperimentRecommendation]:
        """Generate experiment recommendations for a patient"""

        logger.info(
            f"Generating tissue-chip experiment recommendations for patient {patient.patient_id}"
        )

        # Get personalized biomarker insights
        biomarker_scores = self.biomarker_engine.generate_personalized_scores(patient)

        # Get current biomarker values (simulate from patient history)
        current_biomarkers = self._extract_current_biomarkers(patient)

        # Predict outcomes and trajectories
        outcome_profile = self.outcome_engine.predict_comprehensive_outcomes(
            patient, current_biomarkers, prediction_horizon_days=365
        )

        recommendations = []

        for objective in research_objectives:
            # Select appropriate tissue chip types
            relevant_chip_types = self._select_chip_types_for_objective(
                objective, biomarker_scores
            )

            for chip_type in relevant_chip_types:
                # Design experiment specification
                chip_spec = self._design_chip_specification(
                    chip_type, objective, patient, biomarker_scores, outcome_profile
                )

                # Calculate priority and feasibility
                priority_score = self._calculate_priority_score(
                    patient, chip_spec, outcome_profile
                )
                feasibility_score = self._calculate_feasibility_score(
                    chip_spec, budget_limit
                )
                clinical_relevance = self._calculate_clinical_relevance(
                    patient, chip_spec
                )

                # Predict experimental outcomes
                predicted_outcomes = self._predict_experimental_outcomes(
                    patient, chip_spec, current_biomarkers
                )

                # Estimate costs and timeline
                cost_estimate = self._estimate_experiment_cost(chip_spec)
                duration_estimate = self._estimate_experiment_duration(chip_spec)

                # Skip if over budget
                if budget_limit and cost_estimate > budget_limit:
                    continue

                # Create recommendation
                recommendation = ExperimentRecommendation(
                    recommendation_id=f"{patient.patient_id}_{chip_type.value}_{objective.value}",
                    patient_profile=patient,
                    chip_specification=chip_spec,
                    priority_score=priority_score,
                    clinical_relevance=clinical_relevance,
                    feasibility_score=feasibility_score,
                    predicted_outcomes=predicted_outcomes,
                    estimated_cost=cost_estimate,
                    estimated_duration=duration_estimate,
                    required_expertise=self.chip_capabilities[chip_type][
                        "expertise_required"
                    ],
                    success_metrics=self._define_success_metrics(chip_spec),
                    validation_endpoints=self._define_validation_endpoints(
                        chip_spec, outcome_profile
                    ),
                    technical_risks=self._assess_technical_risks(chip_spec),
                    mitigation_strategies=self._recommend_mitigation_strategies(
                        chip_spec
                    ),
                    translation_pathway=self._define_translation_pathway(
                        objective, chip_spec
                    ),
                    regulatory_considerations=self._assess_regulatory_needs(
                        objective, chip_spec
                    ),
                    rationale=self._generate_rationale(patient, chip_spec, objective),
                )

                recommendations.append(recommendation)

        # Sort by priority score
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        logger.info(f"Generated {len(recommendations)} experiment recommendations")
        return recommendations

    def _extract_current_biomarkers(self, patient: PatientProfile) -> Dict[str, float]:
        """Extract current biomarker values from patient"""
        current_biomarkers = {}

        if not patient.lab_history.empty:
            # Get most recent values
            for biomarker in patient.lab_history["biomarker"].unique():
                biomarker_data = patient.lab_history[
                    patient.lab_history["biomarker"] == biomarker
                ]
                if not biomarker_data.empty:
                    most_recent = biomarker_data.loc[biomarker_data["date"].idxmax()]
                    current_biomarkers[biomarker] = most_recent["value"]
        else:
            # Generate synthetic values based on patient characteristics
            current_biomarkers = self._generate_synthetic_biomarkers(patient)

        return current_biomarkers

    def _generate_synthetic_biomarkers(
        self, patient: PatientProfile
    ) -> Dict[str, float]:
        """Generate synthetic biomarker values based on patient characteristics"""

        # Base values
        biomarkers = {
            "CRP": np.random.lognormal(0.5, 0.8),
            "APOB": np.random.normal(90, 25),
            "PCSK9": np.random.normal(150, 50),
            "LPA": np.random.lognormal(2.5, 1.0),
            "HMGCR": np.random.normal(75, 30),
        }

        # Adjust for patient characteristics
        if "diabetes" in patient.comorbidities:
            biomarkers["CRP"] *= 1.5
            biomarkers["APOB"] *= 1.3

        if "hypertension" in patient.comorbidities:
            biomarkers["CRP"] *= 1.2
            biomarkers["PCSK9"] *= 1.1

        if patient.age > 65:
            biomarkers["CRP"] *= 1.2
            biomarkers["APOB"] *= 1.1

        return {k: max(0.1, v) for k, v in biomarkers.items()}

    def _select_chip_types_for_objective(
        self, objective: ExperimentObjective, biomarker_scores: List
    ) -> List[TissueChipType]:
        """Select appropriate chip types for research objective"""

        if objective == ExperimentObjective.BIOMARKER_VALIDATION:
            # Focus on chips that can measure target biomarkers
            return [
                TissueChipType.VASCULATURE_ON_CHIP,
                TissueChipType.LIVER_ON_CHIP,
                TissueChipType.HEART_ON_CHIP,
            ]

        elif objective == ExperimentObjective.DRUG_SCREENING:
            # Use high-throughput platforms
            return [TissueChipType.LIVER_ON_CHIP, TissueChipType.VASCULATURE_ON_CHIP]

        elif objective == ExperimentObjective.DISEASE_MODELING:
            # Use disease-specific chips
            return [
                TissueChipType.HEART_ON_CHIP,
                TissueChipType.KIDNEY_ON_CHIP,
                TissueChipType.MULTI_ORGAN_CHIP,
            ]

        elif objective == ExperimentObjective.PERSONALIZED_MEDICINE:
            # Use comprehensive platforms
            return [TissueChipType.MULTI_ORGAN_CHIP, TissueChipType.HEART_ON_CHIP]

        else:
            # Default selection
            return [TissueChipType.VASCULATURE_ON_CHIP, TissueChipType.LIVER_ON_CHIP]

    def _design_chip_specification(
        self,
        chip_type: TissueChipType,
        objective: ExperimentObjective,
        patient: PatientProfile,
        biomarker_scores: List,
        outcome_profile,
    ) -> TissueChipSpecification:
        """Design detailed chip specification"""

        capabilities = self.chip_capabilities[chip_type]

        # Select target biomarkers based on patient profile
        target_biomarkers = []
        for score in biomarker_scores:
            if score.biomarker in capabilities["primary_biomarkers"]:
                target_biomarkers.append(score.biomarker)

        # Add secondary biomarkers if relevant
        if len(target_biomarkers) < 3:
            for biomarker in capabilities["secondary_biomarkers"]:
                if biomarker not in target_biomarkers:
                    target_biomarkers.append(biomarker)
                    if len(target_biomarkers) >= 5:
                        break

        # Design sampling schedule
        sampling_timepoints = self._design_sampling_schedule(chip_type, objective)

        # Select experimental conditions
        conditions = self._select_experimental_conditions(patient, objective)

        # Design culture conditions
        culture_media = self._design_culture_media(chip_type, patient)
        flow_rates = self._design_flow_rates(chip_type)

        # Set detection methods
        detection_methods = {}
        for biomarker in target_biomarkers:
            if biomarker in self.biomarker_detection_methods:
                detection_methods[biomarker] = self.biomarker_detection_methods[
                    biomarker
                ]["primary_method"]

        return TissueChipSpecification(
            chip_type=chip_type,
            experiment_objective=objective,
            cell_sources=capabilities["cell_types"],
            patient_derived_cells=(
                objective == ExperimentObjective.PERSONALIZED_MEDICINE
            ),
            genetic_background=self._determine_genetic_background(patient),
            culture_media_composition=culture_media,
            flow_rates=flow_rates,
            target_biomarkers=target_biomarkers,
            sampling_timepoints=sampling_timepoints,
            detection_methods=detection_methods,
            conditions_to_test=conditions,
            duration=max(capabilities["culture_duration"]),
            expected_biomarker_changes=self._predict_biomarker_changes(
                patient, target_biomarkers
            ),
            viability_thresholds={"general": 0.8, "endpoint": 0.7},
            reproducibility_targets={"CV": 0.15, "correlation": 0.85},
        )

    def _design_sampling_schedule(
        self, chip_type: TissueChipType, objective: ExperimentObjective
    ) -> List[float]:
        """Design biomarker sampling schedule"""

        if objective == ExperimentObjective.DRUG_SCREENING:
            # Frequent early sampling for pharmacodynamics
            return [0, 2, 6, 12, 24, 48, 72]
        elif objective == ExperimentObjective.BIOMARKER_VALIDATION:
            # Regular intervals for biomarker kinetics
            return [0, 6, 12, 24, 48, 72, 96, 120]
        elif objective == ExperimentObjective.DISEASE_MODELING:
            # Extended timeline for disease progression
            return [0, 12, 24, 48, 96, 168, 240]
        else:
            # Standard schedule
            return [0, 12, 24, 48, 72]

    def _select_experimental_conditions(
        self, patient: PatientProfile, objective: ExperimentObjective
    ) -> List[ExperimentalCondition]:
        """Select experimental conditions based on patient and objective"""

        conditions = [ExperimentalCondition.CONTROL]

        # Add disease-relevant conditions
        if "diabetes" in patient.comorbidities:
            conditions.append(ExperimentalCondition.DISEASE_STATE)

        if objective in [
            ExperimentObjective.DRUG_SCREENING,
            ExperimentObjective.PERSONALIZED_MEDICINE,
        ]:
            conditions.append(ExperimentalCondition.DRUG_TREATMENT)

        if objective == ExperimentObjective.BIOMARKER_VALIDATION:
            conditions.append(ExperimentalCondition.BIOMARKER_MODULATION)

        # Add stress conditions for robustness testing
        conditions.append(ExperimentalCondition.STRESS_CONDITIONS)

        return conditions

    def _design_culture_media(
        self, chip_type: TissueChipType, patient: PatientProfile
    ) -> Dict[str, float]:
        """Design culture media composition"""

        base_media = {
            "glucose": 5.5,  # mM
            "albumin": 40.0,  # g/L
            "pH": 7.4,
            "osmolality": 290,  # mOsm/kg
        }

        # Adjust for patient characteristics
        if "diabetes" in patient.comorbidities:
            base_media["glucose"] = 11.0  # Diabetic glucose levels

        if chip_type == TissueChipType.LIVER_ON_CHIP:
            base_media["insulin"] = 100  # nM
            base_media["glucagon"] = 10  # nM

        return base_media

    def _design_flow_rates(self, chip_type: TissueChipType) -> Dict[str, float]:
        """Design flow rates for tissue chip"""

        if chip_type == TissueChipType.HEART_ON_CHIP:
            return {"perfusion": 50.0, "waste": 25.0}  # μL/min
        elif chip_type == TissueChipType.LIVER_ON_CHIP:
            return {"perfusion": 100.0, "waste": 50.0}
        elif chip_type == TissueChipType.VASCULATURE_ON_CHIP:
            return {"arterial": 75.0, "venous": 25.0}
        else:
            return {"perfusion": 50.0}

    def _determine_genetic_background(self, patient: PatientProfile) -> Optional[str]:
        """Determine relevant genetic background for experiments"""

        if patient.race in ["African_American", "Hispanic"]:
            return f"{patient.race}_genetic_background"

        # Check for high genetic risk
        if patient.genetic_risk_scores:
            max_risk = max(patient.genetic_risk_scores.values())
            if max_risk > 0.8:
                return "high_genetic_risk_background"

        return "standard_genetic_background"

    def _predict_biomarker_changes(
        self, patient: PatientProfile, biomarkers: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Predict expected biomarker changes in experiments"""

        changes = {}

        for biomarker in biomarkers:
            # Predict changes under different conditions
            biomarker_changes = {
                "control": 0.0,  # No change
                "disease_state": self._predict_disease_change(biomarker, patient),
                "drug_treatment": self._predict_drug_response(biomarker, patient),
                "stress_conditions": self._predict_stress_response(biomarker),
            }
            changes[biomarker] = biomarker_changes

        return changes

    def _predict_disease_change(self, biomarker: str, patient: PatientProfile) -> float:
        """Predict biomarker change under disease conditions"""

        disease_effects = {
            "CRP": 2.5 if "diabetes" in patient.comorbidities else 1.5,
            "APOB": 1.3 if "coronary_artery_disease" in patient.comorbidities else 1.1,
            "PCSK9": 1.2,
            "LPA": 1.0,  # Genetic, less affected by disease state
            "HMGCR": 1.4 if "diabetes" in patient.comorbidities else 1.2,
        }

        return disease_effects.get(biomarker, 1.2) - 1.0  # Return fold-change - 1

    def _predict_drug_response(self, biomarker: str, patient: PatientProfile) -> float:
        """Predict biomarker response to drug treatment"""

        # Simulate statin treatment effects
        statin_effects = {
            "CRP": -0.3,  # 30% reduction
            "APOB": -0.4,  # 40% reduction
            "PCSK9": 0.2,  # 20% increase (compensatory)
            "LPA": -0.1,  # 10% reduction
            "HMGCR": -0.6,  # 60% reduction (direct target)
        }

        return statin_effects.get(biomarker, -0.1)

    def _predict_stress_response(self, biomarker: str) -> float:
        """Predict biomarker response to stress conditions"""

        stress_effects = {
            "CRP": 1.8,  # 80% increase
            "APOB": 1.2,  # 20% increase
            "PCSK9": 1.3,  # 30% increase
            "LPA": 1.1,  # 10% increase
            "HMGCR": 1.5,  # 50% increase
        }

        return stress_effects.get(biomarker, 1.2) - 1.0

    def _calculate_priority_score(
        self,
        patient: PatientProfile,
        chip_spec: TissueChipSpecification,
        outcome_profile,
    ) -> float:
        """Calculate experiment priority score"""

        # Factors influencing priority
        clinical_urgency = self._assess_clinical_urgency(patient, outcome_profile)
        biomarker_relevance = self._assess_biomarker_relevance(
            chip_spec.target_biomarkers, patient
        )
        novelty_score = self._assess_experimental_novelty(chip_spec)
        translational_potential = self._assess_translational_potential(chip_spec)

        # Weighted combination
        priority = (
            clinical_urgency * 0.3
            + biomarker_relevance * 0.25
            + novelty_score * 0.2
            + translational_potential * 0.25
        )

        return min(1.0, priority)

    def _assess_clinical_urgency(
        self, patient: PatientProfile, outcome_profile
    ) -> float:
        """Assess clinical urgency based on patient risk"""

        if hasattr(outcome_profile, "overall_risk_score"):
            risk_score = outcome_profile.overall_risk_score
            return risk_score / 100.0  # Convert to 0-1 scale

        # Fallback based on comorbidities
        urgency = len(patient.comorbidities) * 0.2
        if patient.age > 70:
            urgency += 0.3

        return min(1.0, urgency)

    def _assess_biomarker_relevance(
        self, biomarkers: List[str], patient: PatientProfile
    ) -> float:
        """Assess relevance of biomarkers to patient"""

        # Score based on biomarker clinical significance
        relevance_scores = {
            "CRP": 0.9,  # Highly relevant inflammatory marker
            "APOB": 0.8,  # Important lipid marker
            "PCSK9": 0.7,  # Emerging therapeutic target
            "LPA": 0.6,  # Genetic risk factor
            "HMGCR": 0.5,  # Enzyme target
        }

        if not biomarkers:
            return 0.0

        avg_relevance = np.mean([relevance_scores.get(b, 0.3) for b in biomarkers])

        # Boost for high-risk patients
        if len(patient.comorbidities) > 2:
            avg_relevance *= 1.2

        return min(1.0, float(avg_relevance))

    def _assess_experimental_novelty(self, chip_spec: TissueChipSpecification) -> float:
        """Assess experimental novelty and innovation"""

        novelty = 0.5  # Base novelty

        # Higher novelty for personalized medicine
        if chip_spec.experiment_objective == ExperimentObjective.PERSONALIZED_MEDICINE:
            novelty += 0.3

        # Higher novelty for multi-organ chips
        if chip_spec.chip_type == TissueChipType.MULTI_ORGAN_CHIP:
            novelty += 0.2

        # Higher novelty for patient-derived cells
        if chip_spec.patient_derived_cells:
            novelty += 0.2

        return min(1.0, novelty)

    def _assess_translational_potential(
        self, chip_spec: TissueChipSpecification
    ) -> float:
        """Assess translational potential to clinical practice"""

        potential = 0.4  # Base potential

        # Higher potential for validated biomarkers
        validated_biomarkers = ["CRP", "APOB", "PCSK9"]
        validation_score = len(
            [b for b in chip_spec.target_biomarkers if b in validated_biomarkers]
        )
        potential += validation_score * 0.1

        # Higher potential for drug screening
        if chip_spec.experiment_objective == ExperimentObjective.DRUG_SCREENING:
            potential += 0.3

        return min(1.0, potential)

    def _calculate_feasibility_score(
        self, chip_spec: TissueChipSpecification, budget_limit: Optional[float]
    ) -> float:
        """Calculate experimental feasibility"""

        feasibility = 1.0

        # Reduce feasibility for complex chips
        if chip_spec.chip_type == TissueChipType.MULTI_ORGAN_CHIP:
            feasibility *= 0.7

        # Reduce feasibility for patient-derived cells
        if chip_spec.patient_derived_cells:
            feasibility *= 0.8

        # Reduce feasibility for many conditions
        if len(chip_spec.conditions_to_test) > 4:
            feasibility *= 0.9

        # Budget constraint
        if budget_limit:
            estimated_cost = self._estimate_experiment_cost(chip_spec)
            if estimated_cost > budget_limit * 0.8:
                feasibility *= 0.6

        return feasibility

    def _calculate_clinical_relevance(
        self, patient: PatientProfile, chip_spec: TissueChipSpecification
    ) -> float:
        """Calculate clinical relevance score"""

        relevance = 0.5

        # Higher relevance for disease modeling
        if chip_spec.experiment_objective == ExperimentObjective.DISEASE_MODELING:
            relevance += 0.2

        # Higher relevance for personalized medicine
        if chip_spec.experiment_objective == ExperimentObjective.PERSONALIZED_MEDICINE:
            relevance += 0.3

        # Higher relevance for patients with multiple comorbidities
        relevance += len(patient.comorbidities) * 0.05

        return min(1.0, relevance)

    def _predict_experimental_outcomes(
        self,
        patient: PatientProfile,
        chip_spec: TissueChipSpecification,
        current_biomarkers: Dict[str, float],
    ) -> Dict[str, float]:
        """Predict experimental outcomes"""

        outcomes = {}

        for biomarker in chip_spec.target_biomarkers:
            if biomarker in current_biomarkers:
                current_value = current_biomarkers[biomarker]

                # Predict outcomes for different conditions
                for condition in chip_spec.conditions_to_test:
                    if condition == ExperimentalCondition.CONTROL:
                        outcomes[f"{biomarker}_control"] = current_value
                    elif condition == ExperimentalCondition.DISEASE_STATE:
                        change = self._predict_disease_change(biomarker, patient)
                        outcomes[f"{biomarker}_disease"] = current_value * (1 + change)
                    elif condition == ExperimentalCondition.DRUG_TREATMENT:
                        change = self._predict_drug_response(biomarker, patient)
                        outcomes[f"{biomarker}_drug"] = current_value * (1 + change)

        return outcomes

    def _estimate_experiment_cost(self, chip_spec: TissueChipSpecification) -> float:
        """Estimate total experiment cost"""

        # Base chip costs
        chip_cost = self.chip_capabilities[chip_spec.chip_type]["cost_per_chip"]
        num_chips = len(chip_spec.conditions_to_test) * 3  # Triplicate
        total_chip_cost = chip_cost * num_chips

        # Biomarker analysis costs
        analysis_cost = 0
        for biomarker in chip_spec.target_biomarkers:
            if biomarker in self.biomarker_detection_methods:
                cost_per_sample = self.biomarker_detection_methods[biomarker][
                    "cost_per_sample"
                ]
                num_samples = len(chip_spec.sampling_timepoints) * num_chips
                analysis_cost += cost_per_sample * num_samples

        # Personnel costs
        duration_days = chip_spec.duration / 24
        personnel_cost = (
            duration_days * 8 * self.cost_models["personnel_per_hour"]
        )  # 8 hours/day

        # Overhead and other costs
        facility_cost = duration_days * self.cost_models["facility_cost_per_day"]
        overhead = (total_chip_cost + analysis_cost) * self.cost_models[
            "equipment_overhead"
        ]

        total_cost = (
            total_chip_cost + analysis_cost + personnel_cost + facility_cost + overhead
        )

        # Add development costs for novel experiments
        if chip_spec.patient_derived_cells:
            total_cost += self.cost_models["protocol_development"]

        return total_cost

    def _estimate_experiment_duration(
        self, chip_spec: TissueChipSpecification
    ) -> float:
        """Estimate experiment duration in days"""

        # Preparation time
        prep_time = 3.0  # days
        if chip_spec.patient_derived_cells:
            prep_time += 7.0  # Additional time for cell derivation

        # Experiment duration
        experiment_time = chip_spec.duration / 24  # Convert hours to days

        # Analysis time
        analysis_time = len(chip_spec.target_biomarkers) * 0.5  # 0.5 days per biomarker

        return prep_time + experiment_time + analysis_time

    def _define_success_metrics(
        self, chip_spec: TissueChipSpecification
    ) -> Dict[str, float]:
        """Define success metrics for experiment"""

        metrics = {
            "cell_viability": 0.8,  # >80% viability
            "biomarker_detection": 0.9,  # >90% successful measurements
            "reproducibility": 0.85,  # >85% correlation between replicates
            "signal_to_noise": 3.0,  # >3:1 signal to noise ratio
        }

        # Add biomarker-specific metrics
        for biomarker in chip_spec.target_biomarkers:
            if biomarker in chip_spec.expected_biomarker_changes:
                expected_changes = chip_spec.expected_biomarker_changes[biomarker]
                for condition, change in expected_changes.items():
                    if abs(change) > 0.1:  # Significant change expected
                        metrics[f"{biomarker}_{condition}_detection"] = 0.8

        return metrics

    def _define_validation_endpoints(
        self, chip_spec: TissueChipSpecification, outcome_profile
    ) -> List[str]:
        """Define validation endpoints"""

        endpoints = [
            "biomarker_measurement_accuracy",
            "dose_response_relationship",
            "temporal_kinetics_validation",
        ]

        if chip_spec.experiment_objective == ExperimentObjective.DRUG_SCREENING:
            endpoints.extend(
                [
                    "therapeutic_window_identification",
                    "toxicity_threshold_determination",
                ]
            )

        if chip_spec.experiment_objective == ExperimentObjective.PERSONALIZED_MEDICINE:
            endpoints.extend(
                [
                    "patient_specific_response_prediction",
                    "clinical_correlation_validation",
                ]
            )

        return endpoints

    def _assess_technical_risks(self, chip_spec: TissueChipSpecification) -> List[str]:
        """Assess technical risks"""

        risks = []

        # Chip-specific risks
        if chip_spec.chip_type == TissueChipType.MULTI_ORGAN_CHIP:
            risks.append("Complex multi-organ interactions may confound results")

        if chip_spec.patient_derived_cells:
            risks.extend(
                [
                    "Patient cell availability and quality variability",
                    "Extended culture time requirements",
                ]
            )

        # Biomarker-specific risks
        if "PCSK9" in chip_spec.target_biomarkers:
            risks.append("PCSK9 assay sensitivity may be limiting")

        # General risks
        risks.extend(
            [
                "Cell viability decline over extended culture",
                "Biomarker stability in culture media",
                "Reproducibility across experimental replicates",
            ]
        )

        return risks

    def _recommend_mitigation_strategies(
        self, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Recommend risk mitigation strategies"""

        strategies = [
            "Implement robust quality control checkpoints",
            "Use multiple biomarker detection methods for validation",
            "Include positive and negative controls in all experiments",
            "Monitor cell viability throughout experiment duration",
        ]

        if chip_spec.patient_derived_cells:
            strategies.extend(
                [
                    "Establish cell banking protocols for patient samples",
                    "Validate cell phenotype before experimentation",
                ]
            )

        if chip_spec.chip_type == TissueChipType.MULTI_ORGAN_CHIP:
            strategies.append("Include single-organ controls for comparison")

        return strategies

    def _define_translation_pathway(
        self, objective: ExperimentObjective, chip_spec: TissueChipSpecification
    ) -> str:
        """Define clinical translation pathway"""

        if objective == ExperimentObjective.BIOMARKER_VALIDATION:
            return "Biomarker validation → Clinical correlation studies → Diagnostic development"
        elif objective == ExperimentObjective.DRUG_SCREENING:
            return "Target identification → Lead optimization → Preclinical studies → Clinical trials"
        elif objective == ExperimentObjective.PERSONALIZED_MEDICINE:
            return "Patient stratification → Precision dosing → Personalized treatment protocols"
        else:
            return "Research findings → Validation studies → Clinical application"

    def _assess_regulatory_needs(
        self, objective: ExperimentObjective, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Assess regulatory considerations"""

        considerations = ["Good Laboratory Practice (GLP) compliance"]

        if objective == ExperimentObjective.DRUG_SCREENING:
            considerations.extend(
                [
                    "FDA guidance on organ-on-chip models",
                    "ICH guidelines for drug development",
                ]
            )

        if chip_spec.patient_derived_cells:
            considerations.extend(
                [
                    "IRB approval for patient sample use",
                    "HIPAA compliance for patient data",
                    "Informed consent documentation",
                ]
            )

        if objective == ExperimentObjective.PERSONALIZED_MEDICINE:
            considerations.append("FDA guidance on precision medicine")

        return considerations

    def _generate_rationale(
        self,
        patient: PatientProfile,
        chip_spec: TissueChipSpecification,
        objective: ExperimentObjective,
    ) -> str:
        """Generate experiment rationale"""

        rationale = f"""
        This {chip_spec.chip_type.value} experiment is designed to {objective.value} for patient {patient.patient_id}.
        
        Patient characteristics:
        - Age: {patient.age}, Sex: {patient.sex}
        - Comorbidities: {', '.join(patient.comorbidities) if patient.comorbidities else 'None'}
        - High-priority biomarkers: {', '.join(chip_spec.target_biomarkers[:3])}
        
        Experimental approach:
        - Target biomarkers will be monitored over {chip_spec.duration} hours
        - Multiple conditions will test disease relevance and therapeutic potential
        - Patient-specific factors will inform experimental parameters
        
        Expected outcomes:
        - Validate biomarker responses in controlled environment
        - Identify patient-specific therapeutic targets
        - Generate data for clinical translation
        """

        return rationale.strip()

    def generate_experimental_protocol(
        self, recommendation: ExperimentRecommendation
    ) -> ExperimentalProtocol:
        """Generate detailed experimental protocol"""

        chip_spec = recommendation.chip_specification

        # Generate protocol steps
        prep_steps = self._generate_preparation_steps(chip_spec)
        execution_steps = self._generate_execution_steps(chip_spec)
        analysis_steps = self._generate_analysis_steps(chip_spec)

        # Generate timeline
        total_duration = recommendation.estimated_duration * 24  # Convert days to hours
        critical_timepoints = self._generate_critical_timepoints(chip_spec)

        # Generate materials list
        materials = self._generate_materials_list(chip_spec)
        reagents = self._generate_reagent_concentrations(chip_spec)

        # Generate equipment and settings
        equipment = self._generate_equipment_list(chip_spec)
        settings = self._generate_instrument_settings(chip_spec)

        protocol = ExperimentalProtocol(
            protocol_id=f"PROTOCOL_{recommendation.recommendation_id}",
            title=f"{chip_spec.chip_type.value} Protocol for {chip_spec.experiment_objective.value}",
            preparation_steps=prep_steps,
            execution_steps=execution_steps,
            analysis_steps=analysis_steps,
            total_duration=total_duration,
            critical_timepoints=critical_timepoints,
            required_materials=materials,
            reagent_concentrations=reagents,
            required_equipment=equipment,
            instrument_settings=settings,
            qc_checkpoints=self._generate_qc_checkpoints(chip_spec),
            acceptance_criteria=self._generate_acceptance_criteria(chip_spec),
            measurement_schedule=self._generate_measurement_schedule(chip_spec),
            data_analysis_methods=self._generate_analysis_methods(chip_spec),
            safety_precautions=self._generate_safety_precautions(chip_spec),
            waste_disposal=self._generate_waste_disposal_procedures(chip_spec),
        )

        return protocol

    def _generate_preparation_steps(
        self, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Generate preparation steps"""

        steps = [
            "1. Sterilize all equipment and workspace",
            "2. Prepare culture media according to composition specifications",
            "3. Pre-condition tissue chips at 37°C, 5% CO2 for 2 hours",
            f"4. Prepare cell suspensions for {', '.join(chip_spec.cell_sources)}",
            "5. Calibrate pumps for specified flow rates",
            "6. Prepare biomarker detection reagents",
        ]

        if chip_spec.patient_derived_cells:
            steps.insert(3, "3a. Isolate and characterize patient-derived cells")
            steps.insert(4, "3b. Expand cells to required numbers")

        return steps

    def _generate_execution_steps(
        self, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Generate execution steps"""

        steps = [
            "1. Seed cells onto tissue chip platforms",
            "2. Allow cell attachment and initial culture (4-6 hours)",
            "3. Establish perfusion flow at specified rates",
            "4. Monitor cell viability and confluence",
            "5. Apply experimental conditions according to study design",
        ]

        # Add condition-specific steps
        for condition in chip_spec.conditions_to_test:
            if condition == ExperimentalCondition.DRUG_TREATMENT:
                steps.append("6. Apply drug treatments at specified concentrations")
            elif condition == ExperimentalCondition.STRESS_CONDITIONS:
                steps.append("7. Apply stress conditions (hypoxia, inflammation)")

        steps.extend(
            [
                f"8. Collect samples at timepoints: {chip_spec.sampling_timepoints}",
                "9. Monitor biomarker levels throughout experiment",
                "10. Document all observations and measurements",
            ]
        )

        return steps

    def _generate_analysis_steps(self, chip_spec: TissueChipSpecification) -> List[str]:
        """Generate analysis steps"""

        steps = [
            "1. Process samples according to biomarker-specific protocols",
            "2. Perform quality control on all measurements",
            "3. Calculate biomarker concentrations from standard curves",
            "4. Assess cell viability and morphology",
            "5. Perform statistical analysis of results",
            "6. Compare results to expected outcomes",
            "7. Generate comprehensive data report",
        ]

        # Add biomarker-specific analysis
        for biomarker in chip_spec.target_biomarkers:
            if biomarker in chip_spec.detection_methods:
                method = chip_spec.detection_methods[biomarker]
                steps.append(f"8. Analyze {biomarker} using {method}")

        return steps

    def _generate_critical_timepoints(
        self, chip_spec: TissueChipSpecification
    ) -> List[Tuple[float, str]]:
        """Generate critical timepoints"""

        timepoints: List[Tuple[float, str]] = [
            (0.0, "Experiment start - baseline measurements"),
            (4.0, "Cell attachment check"),
            (24.0, "First major sampling timepoint"),
        ]

        # Add sampling timepoints
        for tp in chip_spec.sampling_timepoints[1:]:  # Skip t=0
            timepoints.append((float(tp), f"Biomarker sampling at {tp}h"))

        # Add endpoint
        timepoints.append((float(chip_spec.duration), "Experiment termination"))

        return sorted(timepoints)

    def _generate_materials_list(
        self, chip_spec: TissueChipSpecification
    ) -> Dict[str, str]:
        """Generate materials list"""

        materials = {
            f"{chip_spec.chip_type.value}_platform": "Primary experimental platform",
            "cell_culture_media": "Custom formulated media",
            "perfusion_pumps": "For maintaining flow rates",
            "incubator": "37°C, 5% CO2 environment",
            "sterile_pipettes": "For sample collection",
            "sample_tubes": "For biomarker storage",
        }

        # Add biomarker-specific materials
        for biomarker in chip_spec.target_biomarkers:
            materials[f"{biomarker}_assay_kit"] = f"For {biomarker} detection"

        return materials

    def _generate_reagent_concentrations(
        self, chip_spec: TissueChipSpecification
    ) -> Dict[str, float]:
        """Generate reagent concentrations"""

        concentrations = {}

        # Add culture media components
        for component, conc in chip_spec.culture_media_composition.items():
            concentrations[component] = conc

        # Add biomarker-specific reagents
        for biomarker in chip_spec.target_biomarkers:
            if biomarker in self.biomarker_detection_methods:
                concentrations[f"{biomarker}_antibody"] = 1.0  # μg/mL
                concentrations[f"{biomarker}_standard"] = 100.0  # ng/mL

        return concentrations

    def _generate_equipment_list(self, chip_spec: TissueChipSpecification) -> List[str]:
        """Generate equipment list"""

        equipment = [
            "Tissue chip perfusion system",
            "Cell culture incubator",
            "Microscope for cell monitoring",
            "Precision pumps",
            "Sample collection system",
        ]

        # Add detection equipment
        detection_methods = set(chip_spec.detection_methods.values())
        for method in detection_methods:
            if method == "ELISA":
                equipment.append("Plate reader for ELISA")
            elif method == "immunoturbidimetry":
                equipment.append("Turbidimetric analyzer")

        return equipment

    def _generate_instrument_settings(
        self, chip_spec: TissueChipSpecification
    ) -> Dict[str, Any]:
        """Generate instrument settings"""

        settings = {
            "incubator_temperature": 37.0,
            "co2_concentration": 5.0,
            "humidity": 95.0,
        }

        # Add flow rate settings
        settings.update(chip_spec.flow_rates)

        # Add detection settings
        for biomarker in chip_spec.target_biomarkers:
            if biomarker in self.biomarker_detection_methods:
                method_info = self.biomarker_detection_methods[biomarker]
                settings[f"{biomarker}_detection_range"] = method_info["dynamic_range"]
                settings[f"{biomarker}_sample_volume"] = method_info["sample_volume"]

        return settings

    def _generate_qc_checkpoints(self, chip_spec: TissueChipSpecification) -> List[str]:
        """Generate quality control checkpoints"""

        return [
            "Cell viability >80% at seeding",
            "Proper chip perfusion confirmed",
            "Media composition verified",
            "Flow rates within ±5% of target",
            "Biomarker standards perform within expected range",
            "Negative controls show no signal",
            "Positive controls show expected response",
        ]

    def _generate_acceptance_criteria(
        self, chip_spec: TissueChipSpecification
    ) -> Dict[str, str]:
        """Generate acceptance criteria"""

        criteria = {
            "cell_viability": ">80% throughout experiment",
            "flow_stability": "±5% of target flow rates",
            "biomarker_detection": "CV <15% between replicates",
            "signal_to_noise": ">3:1 for all biomarkers",
        }

        # Add biomarker-specific criteria
        for biomarker in chip_spec.target_biomarkers:
            criteria[f"{biomarker}_sensitivity"] = "Detect >10% change from baseline"

        return criteria

    def _generate_measurement_schedule(
        self, chip_spec: TissueChipSpecification
    ) -> List[Tuple[float, List[str]]]:
        """Generate measurement schedule"""

        schedule = []

        for timepoint in chip_spec.sampling_timepoints:
            measurements = ["cell_viability"] + chip_spec.target_biomarkers
            schedule.append((timepoint, measurements))

        return schedule

    def _generate_analysis_methods(
        self, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Generate data analysis methods"""

        methods = [
            "Descriptive statistics for all measurements",
            "ANOVA for condition comparisons",
            "Time series analysis for kinetic data",
            "Dose-response curve fitting",
            "Quality control analysis",
            "Outlier detection and handling",
        ]

        if len(chip_spec.target_biomarkers) > 1:
            methods.append("Multi-biomarker correlation analysis")

        return methods

    def _generate_safety_precautions(
        self, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Generate safety precautions"""

        return [
            "Work in biosafety cabinet for all cell culture procedures",
            "Wear appropriate PPE (gloves, lab coat, safety glasses)",
            "Follow institutional biosafety guidelines",
            "Handle all biological materials as potentially infectious",
            "Maintain aseptic technique throughout",
            "Monitor for equipment malfunctions",
        ]

    def _generate_waste_disposal_procedures(
        self, chip_spec: TissueChipSpecification
    ) -> List[str]:
        """Generate waste disposal procedures"""

        return [
            "Dispose of cell culture waste in biohazard containers",
            "Autoclave all contaminated materials",
            "Follow institutional chemical waste disposal guidelines",
            "Properly dispose of sharps in designated containers",
            "Document all waste disposal activities",
        ]
