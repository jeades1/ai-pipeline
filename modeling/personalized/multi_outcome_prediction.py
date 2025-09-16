"""
Multi-Outcome Prediction Integration

Integrates cardiovascular, metabolic, and inflammatory outcome models with biomarker
trajectories for comprehensive risk assessment and multi-domain health predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import our personalized biomarker components
from .avatar_integration import PatientProfile
from .trajectory_prediction import TrajectoryPredictor, BiomarkerPrediction

logger = logging.getLogger(__name__)


class OutcomeCategory(Enum):
    """Categories of health outcomes"""

    CARDIOVASCULAR = "cardiovascular"
    METABOLIC = "metabolic"
    INFLAMMATORY = "inflammatory"
    RENAL = "renal"
    HEPATIC = "hepatic"
    NEUROLOGIC = "neurologic"
    ONCOLOGIC = "oncologic"


class RiskTimeframe(Enum):
    """Risk prediction timeframes"""

    SHORT_TERM = "1_month"  # 30 days
    MEDIUM_TERM = "6_months"  # 180 days
    LONG_TERM = "1_year"  # 365 days
    LIFETIME = "lifetime"  # 10+ years


class OutcomeSeverity(Enum):
    """Severity levels for predicted outcomes"""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class OutcomePrediction:
    """Individual outcome prediction"""

    outcome_name: str
    category: OutcomeCategory
    patient_id: str
    prediction_date: datetime

    # Risk assessment
    risk_probability: float  # 0-1 scale
    severity_prediction: OutcomeSeverity
    time_to_event: Optional[float] = None  # Days until predicted event

    # Timeframe-specific risks
    risk_1_month: float = 0.0
    risk_6_months: float = 0.0
    risk_1_year: float = 0.0
    risk_lifetime: float = 0.0

    # Contributing factors
    primary_biomarkers: List[str] = field(default_factory=list)
    biomarker_contributions: Dict[str, float] = field(default_factory=dict)
    demographic_factors: Dict[str, float] = field(default_factory=dict)
    genetic_factors: Dict[str, float] = field(default_factory=dict)

    # Model metadata
    model_confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    similar_patient_outcomes: List[str] = field(default_factory=list)

    # Clinical context
    preventable_risk: float = 0.0  # Portion of risk that's modifiable
    intervention_recommendations: List[str] = field(default_factory=list)
    monitoring_priorities: List[str] = field(default_factory=list)


@dataclass
class MultiOutcomeRiskProfile:
    """Comprehensive multi-outcome risk profile for a patient"""

    patient_id: str
    assessment_date: datetime

    # Individual outcome predictions
    outcome_predictions: Dict[str, OutcomePrediction] = field(default_factory=dict)

    # Aggregate risk scores
    overall_risk_score: float = 0.0  # 0-100 scale
    category_risk_scores: Dict[OutcomeCategory, float] = field(default_factory=dict)

    # Risk interactions
    synergistic_risks: List[Tuple[str, str, float]] = field(
        default_factory=list
    )  # (outcome1, outcome2, interaction_strength)
    protective_interactions: List[Tuple[str, str, float]] = field(default_factory=list)

    # Time-based risk trajectory
    risk_trajectory_30d: List[Tuple[datetime, float]] = field(default_factory=list)
    risk_trajectory_1y: List[Tuple[datetime, float]] = field(default_factory=list)

    # Intervention opportunities
    high_impact_interventions: List[Dict[str, Any]] = field(default_factory=list)
    optimal_intervention_timing: Optional[datetime] = None

    # Model performance
    prediction_confidence: float = 0.0
    uncertainty_bounds: Tuple[float, float] = (0.0, 0.0)


class CardiovascularOutcomeModel:
    """Cardiovascular outcome prediction model"""

    def __init__(self):
        # Evidence-based cardiovascular risk factors and weights
        self.biomarker_weights = {
            "APOB": {
                "mi_risk": 0.25,  # Myocardial infarction
                "stroke_risk": 0.15,
                "cad_risk": 0.30,  # Coronary artery disease
                "heart_failure_risk": 0.10,
            },
            "LPA": {
                "mi_risk": 0.20,
                "stroke_risk": 0.25,
                "cad_risk": 0.25,
                "aortic_stenosis_risk": 0.35,
            },
            "PCSK9": {
                "mi_risk": 0.15,
                "stroke_risk": 0.10,
                "cad_risk": 0.20,
                "atherosclerosis_risk": 0.25,
            },
            "CRP": {
                "mi_risk": 0.20,
                "stroke_risk": 0.15,
                "heart_failure_risk": 0.15,
                "arrhythmia_risk": 0.10,
            },
        }

        # Age-specific risk multipliers
        self.age_multipliers = {
            (18, 30): 0.1,
            (30, 40): 0.3,
            (40, 50): 0.6,
            (50, 60): 1.0,
            (60, 70): 1.5,
            (70, 80): 2.2,
            (80, 100): 3.0,
        }

        # Sex-specific adjustments
        self.sex_adjustments = {
            "male": {"mi_risk": 1.3, "stroke_risk": 1.1, "cad_risk": 1.4},
            "female": {"mi_risk": 0.8, "stroke_risk": 1.0, "cad_risk": 0.7},
        }

        # Comorbidity risk multipliers
        self.comorbidity_multipliers = {
            "diabetes": 2.0,
            "hypertension": 1.6,
            "chronic_kidney_disease": 1.8,
            "family_history_cad": 1.4,
            "smoking": 2.5,
            "obesity": 1.3,
        }

    def predict_cardiovascular_outcomes(
        self,
        patient_profile: PatientProfile,
        biomarker_predictions: Dict[str, BiomarkerPrediction],
    ) -> List[OutcomePrediction]:
        """Predict cardiovascular outcomes based on biomarker trajectories"""
        outcomes = []

        # Define cardiovascular outcomes to predict
        cv_outcomes = [
            "myocardial_infarction",
            "stroke",
            "coronary_artery_disease",
            "heart_failure",
            "sudden_cardiac_death",
            "peripheral_artery_disease",
        ]

        for outcome in cv_outcomes:
            risk_prob = self._calculate_outcome_risk(
                outcome, patient_profile, biomarker_predictions
            )

            # Determine severity based on risk probability
            if risk_prob >= 0.7:
                severity = OutcomeSeverity.CRITICAL
            elif risk_prob >= 0.5:
                severity = OutcomeSeverity.SEVERE
            elif risk_prob >= 0.3:
                severity = OutcomeSeverity.MODERATE
            else:
                severity = OutcomeSeverity.MILD

            # Calculate timeframe-specific risks
            risk_1m = risk_prob * 0.1  # 10% of annual risk in 1 month
            risk_6m = risk_prob * 0.5  # 50% of annual risk in 6 months
            risk_1y = risk_prob
            risk_lifetime = min(1.0, risk_prob * 5)  # Rough lifetime estimate

            # Identify contributing biomarkers
            contributing_biomarkers = []
            biomarker_contributions = {}

            for biomarker, prediction in biomarker_predictions.items():
                if biomarker in self.biomarker_weights:
                    outcome_weights = self.biomarker_weights[biomarker]
                    if outcome.replace("_", "_") in outcome_weights or any(
                        key in outcome for key in outcome_weights.keys()
                    ):
                        contributing_biomarkers.append(biomarker)
                        # Use prediction trend and values to estimate contribution
                        trend_factor = (
                            1.2
                            if prediction.trend_direction.value
                            in ["increasing", "volatile"]
                            else 0.8
                        )
                        avg_predicted = (
                            np.mean(prediction.predicted_values)
                            if prediction.predicted_values
                            else 1.0
                        )
                        biomarker_contributions[biomarker] = (
                            avg_predicted * trend_factor * 0.1
                        )

            # Calculate model confidence
            confidence = self._calculate_model_confidence(
                patient_profile, biomarker_predictions, outcome
            )

            outcome_pred = OutcomePrediction(
                outcome_name=outcome,
                category=OutcomeCategory.CARDIOVASCULAR,
                patient_id=patient_profile.patient_id,
                prediction_date=datetime.now(),
                risk_probability=risk_prob,
                severity_prediction=severity,
                time_to_event=self._estimate_time_to_event(risk_prob),
                risk_1_month=risk_1m,
                risk_6_months=risk_6m,
                risk_1_year=risk_1y,
                risk_lifetime=risk_lifetime,
                primary_biomarkers=contributing_biomarkers,
                biomarker_contributions=biomarker_contributions,
                model_confidence=confidence,
                preventable_risk=self._calculate_preventable_risk(
                    outcome, patient_profile
                ),
                intervention_recommendations=self._generate_cv_interventions(
                    outcome, risk_prob, patient_profile
                ),
            )

            outcomes.append(outcome_pred)

        return outcomes

    def _calculate_outcome_risk(
        self,
        outcome: str,
        patient_profile: PatientProfile,
        biomarker_predictions: Dict[str, BiomarkerPrediction],
    ) -> float:
        """Calculate risk probability for specific cardiovascular outcome"""
        base_risk = 0.05  # 5% baseline annual risk

        # Age adjustment
        age_multiplier = 1.0
        for age_range, multiplier in self.age_multipliers.items():
            if age_range[0] <= patient_profile.age < age_range[1]:
                age_multiplier = multiplier
                break

        # Sex adjustment
        sex_adjustment = 1.0
        if outcome.replace("_", "_") in self.sex_adjustments.get(
            patient_profile.sex, {}
        ):
            sex_adjustment = self.sex_adjustments[patient_profile.sex][
                outcome.replace("_", "_")
            ]

        # Comorbidity adjustment
        comorbidity_multiplier = 1.0
        for condition in patient_profile.comorbidities:
            if condition in self.comorbidity_multipliers:
                comorbidity_multiplier *= self.comorbidity_multipliers[condition]

        # Biomarker contribution
        biomarker_risk = 0.0
        for biomarker, prediction in biomarker_predictions.items():
            if biomarker in self.biomarker_weights:
                weights = self.biomarker_weights[biomarker]

                # Find relevant weight for this outcome
                outcome_weight = 0.0
                for weight_key, weight_value in weights.items():
                    if weight_key.replace("_risk", "") in outcome:
                        outcome_weight = weight_value
                        break

                if outcome_weight > 0:
                    # Use predicted biomarker trajectory
                    if prediction.predicted_values:
                        avg_value = np.mean(prediction.predicted_values)
                        trend_factor = (
                            1.3
                            if prediction.trend_direction.value == "increasing"
                            else 1.0
                        )
                        biomarker_risk += (
                            outcome_weight * avg_value * trend_factor * 0.01
                        )

        # Combine all factors
        total_risk = (
            base_risk * age_multiplier * sex_adjustment * comorbidity_multiplier
            + biomarker_risk
        )

        # Cap at 95% maximum risk
        return min(0.95, float(total_risk))

    def _calculate_model_confidence(
        self,
        patient_profile: PatientProfile,
        biomarker_predictions: Dict[str, BiomarkerPrediction],
        outcome: str,
    ) -> float:
        """Calculate confidence in outcome prediction"""
        confidence_factors = []

        # Biomarker prediction confidence
        biomarker_confidences = [
            pred.model_confidence for pred in biomarker_predictions.values()
        ]
        if biomarker_confidences:
            confidence_factors.append(np.mean(biomarker_confidences))

        # Data completeness
        data_completeness = 0.8  # Default good completeness
        if len(patient_profile.comorbidities) > 0:
            data_completeness += 0.1
        if patient_profile.genetic_risk_scores:
            data_completeness += 0.1
        confidence_factors.append(min(1.0, data_completeness))

        # Model validation strength (simulated)
        model_validation = 0.85  # High validation for CV outcomes
        confidence_factors.append(model_validation)

        return float(np.mean(confidence_factors))

    def _estimate_time_to_event(self, risk_probability: float) -> Optional[float]:
        """Estimate days until predicted event"""
        if risk_probability < 0.1:
            return None  # Too low risk to estimate

        # Inverse relationship: higher risk = shorter time to event
        # Rough estimate based on survival analysis principles
        base_time = 365 * 5  # 5 years baseline
        adjusted_time = base_time * (1 - risk_probability)

        return max(30, adjusted_time)  # Minimum 30 days

    def _calculate_preventable_risk(
        self, outcome: str, patient_profile: PatientProfile
    ) -> float:
        """Calculate what portion of risk is modifiable through intervention"""
        base_preventable = 0.6  # 60% of CV risk is typically modifiable

        # Younger patients have more preventable risk
        if patient_profile.age < 50:
            base_preventable += 0.2
        elif patient_profile.age > 70:
            base_preventable -= 0.2

        # Modifiable risk factors increase preventable portion
        modifiable_conditions = ["diabetes", "hypertension", "obesity", "smoking"]
        modifiable_count = sum(
            1 for cond in patient_profile.comorbidities if cond in modifiable_conditions
        )
        base_preventable += modifiable_count * 0.1

        return min(0.9, max(0.2, base_preventable))

    def _generate_cv_interventions(
        self, outcome: str, risk_prob: float, patient_profile: PatientProfile
    ) -> List[str]:
        """Generate cardiovascular intervention recommendations"""
        interventions = []

        # High-risk interventions
        if risk_prob > 0.5:
            interventions.extend(
                [
                    "Intensive statin therapy",
                    "Antiplatelet therapy consideration",
                    "ACE inhibitor optimization",
                    "Cardiac imaging evaluation",
                ]
            )

        # Moderate-risk interventions
        if risk_prob > 0.3:
            interventions.extend(
                [
                    "Lifestyle modification counseling",
                    "Exercise stress testing",
                    "Nutritional consultation",
                ]
            )

        # Outcome-specific interventions
        if "stroke" in outcome:
            interventions.append("Carotid artery screening")
        elif "heart_failure" in outcome:
            interventions.append("Echocardiogram evaluation")
        elif "arrhythmia" in outcome:
            interventions.append("Holter monitor consideration")

        return interventions


class MetabolicOutcomeModel:
    """Metabolic outcome prediction model"""

    def __init__(self):
        self.biomarker_weights = {
            "CRP": {
                "diabetes_risk": 0.20,
                "metabolic_syndrome_risk": 0.25,
                "insulin_resistance_risk": 0.30,
            },
            "APOB": {
                "diabetes_risk": 0.15,
                "metabolic_syndrome_risk": 0.20,
                "fatty_liver_risk": 0.25,
            },
        }

        self.metabolic_outcomes = [
            "type_2_diabetes",
            "metabolic_syndrome",
            "insulin_resistance",
            "fatty_liver_disease",
            "diabetic_complications",
        ]

    def predict_metabolic_outcomes(
        self,
        patient_profile: PatientProfile,
        biomarker_predictions: Dict[str, BiomarkerPrediction],
    ) -> List[OutcomePrediction]:
        """Predict metabolic outcomes"""
        outcomes = []

        for outcome in self.metabolic_outcomes:
            # Simplified risk calculation
            base_risk = 0.1 if "diabetes" not in patient_profile.comorbidities else 0.3

            # BMI contribution
            bmi_risk = 0.0
            if patient_profile.bmi > 30:
                bmi_risk = 0.2
            elif patient_profile.bmi > 25:
                bmi_risk = 0.1

            total_risk = min(0.9, base_risk + bmi_risk)

            outcome_pred = OutcomePrediction(
                outcome_name=outcome,
                category=OutcomeCategory.METABOLIC,
                patient_id=patient_profile.patient_id,
                prediction_date=datetime.now(),
                risk_probability=total_risk,
                severity_prediction=OutcomeSeverity.MODERATE,
                risk_1_year=total_risk,
                model_confidence=0.75,
                intervention_recommendations=[
                    "Weight management program",
                    "Glucose monitoring",
                    "Dietary consultation",
                ],
            )
            outcomes.append(outcome_pred)

        return outcomes


class InflammatoryOutcomeModel:
    """Inflammatory outcome prediction model"""

    def __init__(self):
        self.inflammatory_outcomes = [
            "chronic_inflammation",
            "autoimmune_flare",
            "inflammatory_bowel_disease",
            "rheumatoid_arthritis",
        ]

    def predict_inflammatory_outcomes(
        self,
        patient_profile: PatientProfile,
        biomarker_predictions: Dict[str, BiomarkerPrediction],
    ) -> List[OutcomePrediction]:
        """Predict inflammatory outcomes"""
        outcomes = []

        # CRP-based inflammatory risk
        crp_prediction = biomarker_predictions.get("CRP")
        if crp_prediction:
            avg_crp = (
                np.mean(crp_prediction.predicted_values)
                if crp_prediction.predicted_values
                else 2.0
            )
            inflammatory_risk = min(0.8, float(avg_crp / 10.0))  # Normalize CRP to risk

            for outcome in self.inflammatory_outcomes:
                outcome_pred = OutcomePrediction(
                    outcome_name=outcome,
                    category=OutcomeCategory.INFLAMMATORY,
                    patient_id=patient_profile.patient_id,
                    prediction_date=datetime.now(),
                    risk_probability=inflammatory_risk,
                    severity_prediction=OutcomeSeverity.MODERATE,
                    risk_1_year=inflammatory_risk,
                    model_confidence=0.70,
                    intervention_recommendations=[
                        "Anti-inflammatory medication review",
                        "Inflammatory marker monitoring",
                        "Rheumatology consultation",
                    ],
                )
                outcomes.append(outcome_pred)

        return outcomes


class MultiOutcomePredictionEngine:
    """Main engine for multi-outcome prediction integration"""

    def __init__(self):
        self.trajectory_predictor = TrajectoryPredictor()
        self.cv_model = CardiovascularOutcomeModel()
        self.metabolic_model = MetabolicOutcomeModel()
        self.inflammatory_model = InflammatoryOutcomeModel()

    def predict_comprehensive_outcomes(
        self,
        patient_profile: PatientProfile,
        current_biomarkers: Dict[str, float],
        prediction_horizon_days: int = 365,
    ) -> MultiOutcomeRiskProfile:
        """
        Generate comprehensive multi-outcome risk profile

        Args:
            patient_profile: Patient characteristics
            current_biomarkers: Current biomarker values
            prediction_horizon_days: Prediction timeframe

        Returns:
            Comprehensive risk profile across all outcome categories
        """
        logger.info(
            f"Starting comprehensive outcome prediction for {patient_profile.patient_id}"
        )

        # Step 1: Generate biomarker trajectory predictions
        biomarker_predictions = {}
        for biomarker, current_value in current_biomarkers.items():
            try:
                prediction = self.trajectory_predictor.predict_biomarker_trajectory(
                    biomarker=biomarker,
                    patient_profile=patient_profile,
                    current_value=current_value,
                    prediction_horizon_days=prediction_horizon_days,
                )
                biomarker_predictions[biomarker] = prediction
            except Exception as e:
                logger.warning(f"Failed to predict trajectory for {biomarker}: {e}")
                continue

        # Step 2: Generate outcome predictions by category
        all_outcome_predictions = []

        # Cardiovascular outcomes
        cv_outcomes = self.cv_model.predict_cardiovascular_outcomes(
            patient_profile, biomarker_predictions
        )
        all_outcome_predictions.extend(cv_outcomes)

        # Metabolic outcomes
        metabolic_outcomes = self.metabolic_model.predict_metabolic_outcomes(
            patient_profile, biomarker_predictions
        )
        all_outcome_predictions.extend(metabolic_outcomes)

        # Inflammatory outcomes
        inflammatory_outcomes = self.inflammatory_model.predict_inflammatory_outcomes(
            patient_profile, biomarker_predictions
        )
        all_outcome_predictions.extend(inflammatory_outcomes)

        # Step 3: Create comprehensive risk profile
        risk_profile = self._create_risk_profile(
            patient_profile, all_outcome_predictions, biomarker_predictions
        )

        logger.info(
            f"Completed outcome prediction for {patient_profile.patient_id}: "
            f"{len(all_outcome_predictions)} outcomes predicted"
        )

        return risk_profile

    def _create_risk_profile(
        self,
        patient_profile: PatientProfile,
        outcome_predictions: List[OutcomePrediction],
        biomarker_predictions: Dict[str, BiomarkerPrediction],
    ) -> MultiOutcomeRiskProfile:
        """Create comprehensive multi-outcome risk profile"""

        # Organize predictions by outcome name
        outcome_dict = {pred.outcome_name: pred for pred in outcome_predictions}

        # Calculate category-specific risk scores
        category_risks = {}
        for category in OutcomeCategory:
            category_outcomes = [
                p for p in outcome_predictions if p.category == category
            ]
            if category_outcomes:
                # Weighted average of risks in category
                weights = [p.model_confidence for p in category_outcomes]
                risks = [p.risk_probability for p in category_outcomes]
                if weights and risks:
                    weighted_risk = np.average(risks, weights=weights)
                    category_risks[category] = (
                        weighted_risk * 100
                    )  # Convert to 0-100 scale

        # Calculate overall risk score
        if category_risks:
            overall_risk = np.mean(list(category_risks.values()))
        else:
            overall_risk = 0.0

        # Identify synergistic risks (outcomes that compound each other)
        synergistic_risks = self._identify_synergistic_risks(outcome_predictions)

        # Generate high-impact interventions
        high_impact_interventions = self._identify_high_impact_interventions(
            outcome_predictions, patient_profile
        )

        # Calculate prediction confidence
        confidences = [p.model_confidence for p in outcome_predictions]
        prediction_confidence = np.mean(confidences) if confidences else 0.0

        # Generate risk trajectory (simplified)
        risk_trajectory_30d = self._generate_risk_trajectory(outcome_predictions, 30)
        risk_trajectory_1y = self._generate_risk_trajectory(outcome_predictions, 365)

        return MultiOutcomeRiskProfile(
            patient_id=patient_profile.patient_id,
            assessment_date=datetime.now(),
            outcome_predictions=outcome_dict,
            overall_risk_score=float(overall_risk),
            category_risk_scores=category_risks,
            synergistic_risks=synergistic_risks,
            risk_trajectory_30d=risk_trajectory_30d,
            risk_trajectory_1y=risk_trajectory_1y,
            high_impact_interventions=high_impact_interventions,
            prediction_confidence=float(prediction_confidence),
            uncertainty_bounds=(
                max(0.0, float(prediction_confidence) - 0.15),
                min(1.0, float(prediction_confidence) + 0.15),
            ),
        )

    def _identify_synergistic_risks(
        self, outcome_predictions: List[OutcomePrediction]
    ) -> List[Tuple[str, str, float]]:
        """Identify outcome pairs with synergistic risks"""
        synergistic_pairs = []

        # Known synergistic combinations
        known_synergies = {
            ("diabetes", "cardiovascular"): 0.8,
            ("inflammation", "cardiovascular"): 0.6,
            ("metabolic_syndrome", "fatty_liver"): 0.7,
            ("diabetes", "chronic_kidney_disease"): 0.9,
        }

        for (category1, category2), strength in known_synergies.items():
            outcomes1 = [
                p for p in outcome_predictions if category1 in p.outcome_name.lower()
            ]
            outcomes2 = [
                p for p in outcome_predictions if category2 in p.outcome_name.lower()
            ]

            if outcomes1 and outcomes2:
                # Use highest risk from each category
                risk1 = max(p.risk_probability for p in outcomes1)
                risk2 = max(p.risk_probability for p in outcomes2)

                if risk1 > 0.3 and risk2 > 0.3:  # Both significant risks
                    synergistic_pairs.append(
                        (
                            outcomes1[0].outcome_name,
                            outcomes2[0].outcome_name,
                            strength * risk1 * risk2,
                        )
                    )

        return synergistic_pairs

    def _identify_high_impact_interventions(
        self,
        outcome_predictions: List[OutcomePrediction],
        patient_profile: PatientProfile,
    ) -> List[Dict[str, Any]]:
        """Identify interventions with highest impact across multiple outcomes"""
        intervention_impacts = {}

        # Aggregate intervention recommendations across outcomes
        for prediction in outcome_predictions:
            for intervention in prediction.intervention_recommendations:
                if intervention not in intervention_impacts:
                    intervention_impacts[intervention] = {
                        "total_risk_reduction": 0.0,
                        "outcomes_affected": [],
                        "avg_preventable_risk": 0.0,
                    }

                # Weight by risk probability and preventable risk
                impact = prediction.risk_probability * prediction.preventable_risk
                intervention_impacts[intervention]["total_risk_reduction"] += impact
                intervention_impacts[intervention]["outcomes_affected"].append(
                    prediction.outcome_name
                )
                intervention_impacts[intervention][
                    "avg_preventable_risk"
                ] += prediction.preventable_risk

        # Calculate average preventable risk
        for intervention_data in intervention_impacts.values():
            if intervention_data["outcomes_affected"]:
                intervention_data["avg_preventable_risk"] /= len(
                    intervention_data["outcomes_affected"]
                )

        # Sort by total risk reduction
        sorted_interventions = sorted(
            intervention_impacts.items(),
            key=lambda x: x[1]["total_risk_reduction"],
            reverse=True,
        )

        # Format as high-impact interventions
        high_impact = []
        for intervention, data in sorted_interventions[:5]:  # Top 5
            high_impact.append(
                {
                    "intervention": intervention,
                    "total_risk_reduction": data["total_risk_reduction"],
                    "outcomes_affected": data["outcomes_affected"],
                    "impact_score": data["total_risk_reduction"]
                    * len(data["outcomes_affected"]),
                    "priority": (
                        "high" if data["total_risk_reduction"] > 0.5 else "moderate"
                    ),
                }
            )

        return high_impact

    def _generate_risk_trajectory(
        self, outcome_predictions: List[OutcomePrediction], days: int
    ) -> List[Tuple[datetime, float]]:
        """Generate risk trajectory over time"""
        trajectory = []

        # Simple linear progression model
        start_date = datetime.now()
        for day in range(0, days, max(1, days // 10)):  # 10 points max
            date = start_date + timedelta(days=day)

            # Calculate risk at this timepoint
            total_risk = 0.0
            for prediction in outcome_predictions:
                # Linear interpolation between current and future risk
                if days <= 30:
                    future_risk = prediction.risk_1_month
                elif days <= 180:
                    future_risk = prediction.risk_6_months
                else:
                    future_risk = prediction.risk_1_year

                progress = day / days
                interpolated_risk = (
                    prediction.risk_probability * (1 - progress)
                    + future_risk * progress
                )
                total_risk += interpolated_risk

            # Average across outcomes
            avg_risk = (
                total_risk / len(outcome_predictions) if outcome_predictions else 0.0
            )
            trajectory.append((date, avg_risk))

        return trajectory

    def get_intervention_optimization(
        self, risk_profile: MultiOutcomeRiskProfile
    ) -> Dict[str, Any]:
        """Optimize intervention strategies across multiple outcomes"""

        # Analyze intervention synergies
        intervention_synergies = {}
        all_interventions = set()

        for intervention_data in risk_profile.high_impact_interventions:
            intervention = intervention_data["intervention"]
            all_interventions.add(intervention)

        # Calculate optimal intervention timing
        optimal_timing = datetime.now() + timedelta(days=7)  # Default 1 week

        # Find highest risk period from trajectories
        if risk_profile.risk_trajectory_30d:
            max_risk_point = max(risk_profile.risk_trajectory_30d, key=lambda x: x[1])
            optimal_timing = max_risk_point[0] - timedelta(
                days=14
            )  # Intervene 2 weeks before peak

        return {
            "optimal_timing": optimal_timing,
            "intervention_synergies": intervention_synergies,
            "prioritized_interventions": risk_profile.high_impact_interventions[:3],
            "expected_risk_reduction": sum(
                i["total_risk_reduction"]
                for i in risk_profile.high_impact_interventions[:3]
            ),
            "monitoring_adjustments": {
                "frequency_increase_biomarkers": [
                    pred.primary_biomarkers[0]
                    for pred in risk_profile.outcome_predictions.values()
                    if pred.risk_probability > 0.5 and pred.primary_biomarkers
                ],
                "new_biomarkers_to_add": [
                    (
                        "hs_troponin"
                        if any(
                            "cardiac" in outcome
                            for outcome in risk_profile.outcome_predictions.keys()
                        )
                        else None
                    ),
                    (
                        "hba1c"
                        if any(
                            "diabetes" in outcome
                            for outcome in risk_profile.outcome_predictions.keys()
                        )
                        else None
                    ),
                ],
            },
        }
