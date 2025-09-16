"""
Temporal Trajectory Prediction System

Predicts biomarker kinetics for individual patients with uncertainty quantification
and risk period identification for proactive clinical intervention.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings

warnings.filterwarnings("ignore")

from modeling.personalized.avatar_integration import PatientProfile

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Biomarker trend directions"""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class RiskPeriod(Enum):
    """Risk period classifications"""

    CRITICAL = "critical"  # Immediate intervention needed
    HIGH_RISK = "high_risk"  # Close monitoring required
    ELEVATED = "elevated"  # Increased vigilance
    NORMAL = "normal"  # Routine monitoring


@dataclass
class BiomarkerPrediction:
    """Individual biomarker trajectory prediction"""

    biomarker: str
    patient_id: str
    prediction_date: datetime

    # Prediction horizon
    time_points: List[datetime]
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]

    # Trend analysis
    trend_direction: TrendDirection
    trend_slope: float
    trend_confidence: float

    # Risk assessment
    risk_periods: List[Tuple[datetime, datetime, RiskPeriod]]
    threshold_crossings: List[
        Tuple[datetime, str, float]
    ]  # (date, threshold_type, value)

    # Clinical context
    intervention_windows: List[
        Tuple[datetime, datetime, str]
    ]  # (start, end, intervention_type)
    mechanism_drivers: List[str]  # Factors driving the trajectory

    # Model metadata
    model_type: str = "ensemble"
    model_confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class PatientTrajectoryForecast:
    """Complete trajectory forecast for a patient"""

    patient_id: str
    forecast_date: datetime
    forecast_horizon_days: int

    # Biomarker predictions
    biomarker_predictions: Dict[str, BiomarkerPrediction] = field(default_factory=dict)

    # Aggregate risk assessment
    overall_risk_score: float = 0.0
    peak_risk_period: Optional[Tuple[datetime, datetime]] = None

    # Clinical recommendations
    recommended_actions: List[str] = field(default_factory=list)
    optimal_intervention_timing: Optional[datetime] = None

    # Model performance
    forecast_confidence: float = 0.0
    similar_patient_outcomes: List[str] = field(default_factory=list)


class BiomarkerKineticsModel:
    """
    Models biomarker kinetics based on physiological half-lives and patient factors
    """

    def __init__(self):
        # Pharmacokinetic parameters for biomarkers
        self.kinetic_parameters = {
            "APOB": {
                "half_life_hours": 72,
                "clearance_pathway": "hepatic",
                "volume_distribution": 0.1,  # L/kg
                "protein_bound_fraction": 0.95,
                "renal_elimination": 0.05,
            },
            "CRP": {
                "half_life_hours": 19,
                "clearance_pathway": "hepatic",
                "volume_distribution": 0.05,
                "protein_bound_fraction": 0.1,
                "renal_elimination": 0.1,
            },
            "PCSK9": {
                "half_life_hours": 168,  # ~7 days
                "clearance_pathway": "hepatic_ldlr",
                "volume_distribution": 0.08,
                "protein_bound_fraction": 0.2,
                "renal_elimination": 0.05,
            },
            "LPA": {
                "half_life_hours": 26280,  # ~3 years - genetic
                "clearance_pathway": "minimal",
                "volume_distribution": 0.05,
                "protein_bound_fraction": 0.99,
                "renal_elimination": 0.01,
            },
            "HMGCR": {
                "half_life_hours": 24,
                "clearance_pathway": "hepatic_proteasomal",
                "volume_distribution": 0.02,  # Intracellular
                "protein_bound_fraction": 0.0,
                "renal_elimination": 0.0,
            },
        }

    def predict_concentration_profile(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        initial_concentration: float,
        time_hours: np.ndarray,
        interventions: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """
        Predict biomarker concentration over time using PK modeling

        Args:
            biomarker: Biomarker name
            patient_profile: Patient characteristics
            initial_concentration: Starting concentration
            time_hours: Time points for prediction (hours)
            interventions: List of interventions with timing and effects

        Returns:
            Predicted concentrations at each time point
        """
        params = self.kinetic_parameters.get(biomarker, {})

        # Adjust parameters for patient characteristics
        adjusted_params = self._adjust_kinetic_parameters(params, patient_profile)

        # Calculate clearance rate
        half_life = adjusted_params.get("half_life_hours", 24)
        elimination_rate = np.log(2) / half_life  # k = 0.693 / t1/2

        # Base exponential decay
        concentrations = initial_concentration * np.exp(-elimination_rate * time_hours)

        # Apply production rate (steady-state assumption)
        production_rate = self._estimate_production_rate(
            biomarker, patient_profile, initial_concentration
        )
        steady_state = production_rate / elimination_rate

        # Move toward steady state
        concentrations = steady_state + (initial_concentration - steady_state) * np.exp(
            -elimination_rate * time_hours
        )

        # Apply interventions
        if interventions:
            concentrations = self._apply_interventions(
                concentrations, time_hours, interventions, adjusted_params
            )

        # Add biological noise
        concentrations = self._add_biological_variability(concentrations, biomarker)

        return concentrations

    def _adjust_kinetic_parameters(
        self, base_params: Dict, patient_profile: PatientProfile
    ) -> Dict:
        """Adjust kinetic parameters based on patient characteristics"""
        adjusted = base_params.copy()

        # Age effects on clearance
        if patient_profile.age > 65:
            # Reduced clearance in elderly
            adjusted["half_life_hours"] = base_params.get("half_life_hours", 24) * 1.3

        # Kidney function effects
        if "ckd" in patient_profile.comorbidities:
            renal_fraction = base_params.get("renal_elimination", 0.1)
            # Reduced renal clearance increases half-life
            half_life_increase = 1 + (renal_fraction * 2)  # Up to 2x increase
            adjusted["half_life_hours"] = (
                base_params.get("half_life_hours", 24) * half_life_increase
            )

        # Liver function effects
        if "liver_disease" in patient_profile.comorbidities:
            if base_params.get("clearance_pathway", "").startswith("hepatic"):
                adjusted["half_life_hours"] = (
                    base_params.get("half_life_hours", 24) * 2.0
                )

        # BMI effects on volume of distribution
        if patient_profile.bmi > 30:
            adjusted["volume_distribution"] = (
                base_params.get("volume_distribution", 0.1) * 1.2
            )

        return adjusted

    def _estimate_production_rate(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        initial_concentration: float,
    ) -> float:
        """Estimate endogenous production rate"""

        # Production rate modifiers based on patient characteristics
        production_modifiers = {
            "APOB": {
                "diabetes": 1.5,  # Increased hepatic production
                "hyperlipidemia": 1.3,
                "obesity": 1.2,
            },
            "CRP": {
                "inflammation": 5.0,  # Massive increase during inflammation
                "diabetes": 2.0,
                "ckd": 1.8,
                "obesity": 1.5,
            },
            "PCSK9": {
                "diabetes": 1.3,
                "ckd": 1.6,  # Reduced LDLR, increased PCSK9
                "inflammation": 1.4,
            },
        }

        base_production = initial_concentration * 0.5  # Assume 50% turnover rate

        modifiers = production_modifiers.get(biomarker, {})
        production_factor = 1.0

        for condition, factor in modifiers.items():
            if condition in patient_profile.comorbidities:
                production_factor *= factor

        return base_production * production_factor

    def _apply_interventions(
        self,
        concentrations: np.ndarray,
        time_hours: np.ndarray,
        interventions: List[Dict],
        params: Dict,
    ) -> np.ndarray:
        """Apply intervention effects to concentration profile"""

        modified_concentrations = concentrations.copy()

        for intervention in interventions:
            start_time = intervention.get("start_hour", 0)
            duration = intervention.get("duration_hours", 24)
            effect_magnitude = intervention.get(
                "effect_magnitude", 0.5
            )  # Fraction reduction
            intervention_type = intervention.get("type", "inhibitor")

            # Find time points within intervention window
            intervention_mask = (time_hours >= start_time) & (
                time_hours <= start_time + duration
            )

            if intervention_type == "inhibitor":
                # Reduce production or increase clearance
                modified_concentrations[intervention_mask] *= 1 - effect_magnitude
            elif intervention_type == "stimulator":
                # Increase production or reduce clearance
                modified_concentrations[intervention_mask] *= 1 + effect_magnitude

        return modified_concentrations

    def _add_biological_variability(
        self, concentrations: np.ndarray, biomarker: str
    ) -> np.ndarray:
        """Add realistic biological noise to predictions"""

        # Coefficient of variation by biomarker
        cv_values = {
            "APOB": 0.08,  # 8% CV
            "CRP": 0.35,  # 35% CV - highly variable
            "PCSK9": 0.12,  # 12% CV
            "LPA": 0.03,  # 3% CV - very stable
            "HMGCR": 0.15,  # 15% CV
        }

        cv = cv_values.get(biomarker, 0.15)

        # Add correlated noise (biological rhythms)
        np.random.seed(42)  # Reproducible for demonstration
        noise = np.random.normal(1.0, cv, len(concentrations))

        return concentrations * noise


class TrajectoryPredictor:
    """
    Main trajectory prediction engine
    """

    def __init__(self):
        self.kinetics_model = BiomarkerKineticsModel()

    def predict_biomarker_trajectory(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        current_value: float,
        prediction_horizon_days: int = 365,
        interventions: Optional[List[Dict]] = None,
    ) -> BiomarkerPrediction:
        """
        Predict biomarker trajectory for individual patient

        Args:
            biomarker: Biomarker name
            patient_profile: Patient characteristics
            current_value: Current biomarker level
            prediction_horizon_days: Days to predict forward
            interventions: Planned interventions

        Returns:
            BiomarkerPrediction with trajectory and risk analysis
        """
        # Create time points
        time_points = [
            datetime.now() + timedelta(days=d)
            for d in range(0, prediction_horizon_days + 1, 7)  # Weekly predictions
        ]
        time_hours = np.array(
            [d * 24 for d in range(0, prediction_horizon_days + 1, 7)]
        )

        # Predict concentrations
        predicted_concentrations = self.kinetics_model.predict_concentration_profile(
            biomarker, patient_profile, current_value, time_hours, interventions
        )

        # Calculate confidence intervals using Monte Carlo
        confidence_intervals = self._calculate_confidence_intervals(
            biomarker, patient_profile, current_value, time_hours, interventions
        )

        # Analyze trends
        trend_direction, trend_slope, trend_confidence = self._analyze_trend(
            time_hours, predicted_concentrations
        )

        # Identify risk periods
        risk_periods = self._identify_risk_periods(
            time_points, predicted_concentrations, biomarker, patient_profile
        )

        # Find threshold crossings
        threshold_crossings = self._find_threshold_crossings(
            time_points, predicted_concentrations, biomarker, patient_profile
        )

        # Recommend intervention windows
        intervention_windows = self._recommend_intervention_windows(
            time_points, predicted_concentrations, risk_periods
        )

        # Identify mechanism drivers
        mechanism_drivers = self._identify_mechanism_drivers(biomarker, patient_profile)

        # Calculate model confidence
        model_confidence = self._calculate_model_confidence(patient_profile, biomarker)

        return BiomarkerPrediction(
            biomarker=biomarker,
            patient_id=patient_profile.patient_id,
            prediction_date=datetime.now(),
            time_points=time_points,
            predicted_values=predicted_concentrations.tolist(),
            confidence_intervals=confidence_intervals,
            trend_direction=trend_direction,
            trend_slope=trend_slope,
            trend_confidence=trend_confidence,
            risk_periods=risk_periods,
            threshold_crossings=threshold_crossings,
            intervention_windows=intervention_windows,
            mechanism_drivers=mechanism_drivers,
            model_confidence=model_confidence,
        )

    def _calculate_confidence_intervals(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        current_value: float,
        time_hours: np.ndarray,
        interventions: Optional[List[Dict]] = None,
        n_simulations: int = 100,
    ) -> List[Tuple[float, float]]:
        """Calculate confidence intervals using Monte Carlo simulation"""

        if interventions is None:
            interventions = []

        simulations = []

        for _ in range(n_simulations):
            # Add parameter uncertainty
            noisy_current_value = current_value * np.random.normal(
                1.0, 0.1
            )  # 10% uncertainty

            # Predict with noise
            prediction = self.kinetics_model.predict_concentration_profile(
                biomarker,
                patient_profile,
                noisy_current_value,
                time_hours,
                interventions,
            )
            simulations.append(prediction)

        simulations = np.array(simulations)

        # Calculate 95% confidence intervals
        confidence_intervals = []
        for i in range(len(time_hours)):
            values = simulations[:, i]
            ci_low = np.percentile(values, 2.5)
            ci_high = np.percentile(values, 97.5)
            confidence_intervals.append((ci_low, ci_high))

        return confidence_intervals

    def _analyze_trend(
        self, time_hours: np.ndarray, concentrations: np.ndarray
    ) -> Tuple[TrendDirection, float, float]:
        """Analyze biomarker trend"""

        # Simple linear trend calculation to avoid scipy issues
        if len(time_hours) < 2:
            return TrendDirection.STABLE, 0.0, 0.0

        # Calculate slope manually
        n = len(time_hours)
        mean_time = np.mean(time_hours)
        mean_conc = np.mean(concentrations)

        numerator = np.sum((time_hours - mean_time) * (concentrations - mean_conc))
        denominator = np.sum((time_hours - mean_time) ** 2)

        if denominator == 0:
            slope = 0.0
            r_value = 0.0
        else:
            slope = numerator / denominator

            # Calculate correlation coefficient
            ss_tot = np.sum((concentrations - mean_conc) ** 2)
            ss_res = np.sum(
                (
                    concentrations
                    - (slope * time_hours + (mean_conc - slope * mean_time))
                )
                ** 2
            )
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_value = np.sqrt(max(0.0, float(r_squared)))

        # Classify trend direction
        if abs(slope) < 0.001:  # Essentially flat
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check for volatility
        cv = np.std(concentrations) / (np.mean(concentrations) + 1e-6)
        if cv > 0.3:  # High coefficient of variation
            direction = TrendDirection.VOLATILE

        trend_confidence = max(0.0, min(1.0, r_value**2))  # R-squared as confidence

        return direction, float(slope), float(trend_confidence)

    def _identify_risk_periods(
        self,
        time_points: List[datetime],
        concentrations: List[float],
        biomarker: str,
        patient_profile: PatientProfile,
    ) -> List[Tuple[datetime, datetime, RiskPeriod]]:
        """Identify periods of elevated risk"""

        # Define risk thresholds (these would come from clinical guidelines)
        risk_thresholds = {
            "APOB": {"elevated": 120, "high": 160, "critical": 200},
            "CRP": {"elevated": 3.0, "high": 10.0, "critical": 50.0},
            "PCSK9": {"elevated": 400, "high": 600, "critical": 800},
            "LPA": {"elevated": 30, "high": 50, "critical": 75},
        }

        thresholds = risk_thresholds.get(
            biomarker, {"elevated": 100, "high": 150, "critical": 200}
        )

        risk_periods = []
        current_risk = RiskPeriod.NORMAL
        period_start = None

        for i, (time_point, concentration) in enumerate(
            zip(time_points, concentrations)
        ):
            # Determine risk level
            if concentration >= thresholds["critical"]:
                new_risk = RiskPeriod.CRITICAL
            elif concentration >= thresholds["high"]:
                new_risk = RiskPeriod.HIGH_RISK
            elif concentration >= thresholds["elevated"]:
                new_risk = RiskPeriod.ELEVATED
            else:
                new_risk = RiskPeriod.NORMAL

            # Check for risk level changes
            if new_risk != current_risk:
                # End previous period
                if period_start is not None and current_risk != RiskPeriod.NORMAL:
                    risk_periods.append((period_start, time_point, current_risk))

                # Start new period
                if new_risk != RiskPeriod.NORMAL:
                    period_start = time_point
                else:
                    period_start = None

                current_risk = new_risk

        # Close final period if needed
        if period_start is not None and current_risk != RiskPeriod.NORMAL:
            risk_periods.append((period_start, time_points[-1], current_risk))

        return risk_periods

    def _find_threshold_crossings(
        self,
        time_points: List[datetime],
        concentrations: List[float],
        biomarker: str,
        patient_profile: PatientProfile,
    ) -> List[Tuple[datetime, str, float]]:
        """Find when biomarker crosses important thresholds"""

        # Clinical decision thresholds
        decision_thresholds = {
            "APOB": [("treatment_initiation", 120), ("intensify_therapy", 160)],
            "CRP": [("mild_inflammation", 3.0), ("severe_inflammation", 10.0)],
            "PCSK9": [("therapeutic_target", 400), ("max_recommended", 600)],
        }

        thresholds = decision_thresholds.get(biomarker, [("upper_normal", 100)])

        crossings = []

        for threshold_name, threshold_value in thresholds:
            # Find crossings
            for i in range(1, len(concentrations)):
                prev_val = concentrations[i - 1]
                curr_val = concentrations[i]

                # Upward crossing
                if prev_val <= threshold_value < curr_val:
                    crossings.append(
                        (time_points[i], f"{threshold_name}_exceeded", curr_val)
                    )
                # Downward crossing
                elif prev_val >= threshold_value > curr_val:
                    crossings.append(
                        (time_points[i], f"{threshold_name}_restored", curr_val)
                    )

        return crossings

    def _recommend_intervention_windows(
        self,
        time_points: List[datetime],
        concentrations: List[float],
        risk_periods: List[Tuple[datetime, datetime, RiskPeriod]],
    ) -> List[Tuple[datetime, datetime, str]]:
        """Recommend optimal intervention timing"""

        intervention_windows = []

        for start_time, end_time, risk_level in risk_periods:
            if risk_level == RiskPeriod.CRITICAL:
                # Immediate intervention
                intervention_windows.append(
                    (
                        start_time,
                        start_time + timedelta(days=1),
                        "immediate_intervention_required",
                    )
                )
            elif risk_level == RiskPeriod.HIGH_RISK:
                # Intervention within 1-2 weeks
                intervention_windows.append(
                    (
                        start_time,
                        start_time + timedelta(days=14),
                        "urgent_intervention_recommended",
                    )
                )
            elif risk_level == RiskPeriod.ELEVATED:
                # Consider intervention within a month
                intervention_windows.append(
                    (
                        start_time,
                        start_time + timedelta(days=30),
                        "intervention_consideration_window",
                    )
                )

        return intervention_windows

    def _identify_mechanism_drivers(
        self, biomarker: str, patient_profile: PatientProfile
    ) -> List[str]:
        """Identify factors driving biomarker trajectory"""

        drivers = []

        # Age-related drivers
        if patient_profile.age > 65:
            drivers.append("age_related_decline")

        # Comorbidity drivers
        comorbidity_drivers = {
            "diabetes": [
                "insulin_resistance",
                "glycation_effects",
                "inflammatory_state",
            ],
            "ckd": ["reduced_clearance", "uremic_toxins", "mineral_bone_disorder"],
            "inflammation": ["cytokine_cascade", "acute_phase_response"],
            "hyperlipidemia": ["lipid_metabolism_dysfunction", "oxidative_stress"],
        }

        for condition in patient_profile.comorbidities:
            drivers.extend(comorbidity_drivers.get(condition, [condition]))

        # Biomarker-specific drivers
        biomarker_drivers = {
            "APOB": ["hepatic_production", "lipoprotein_assembly"],
            "CRP": ["hepatic_synthesis", "il6_stimulation"],
            "PCSK9": ["ldlr_regulation", "cholesterol_homeostasis"],
            "LPA": ["genetic_polymorphisms", "kringle_structure"],
        }

        drivers.extend(biomarker_drivers.get(biomarker, []))

        return list(set(drivers))  # Remove duplicates

    def _calculate_model_confidence(
        self, patient_profile: PatientProfile, biomarker: str
    ) -> float:
        """Calculate confidence in trajectory prediction"""

        confidence = 0.5  # Base confidence

        # Boost confidence for better-characterized biomarkers
        well_characterized = ["APOB", "CRP", "PCSK9"]
        if biomarker in well_characterized:
            confidence += 0.2

        # Reduce confidence for complex patients
        if len(patient_profile.comorbidities) > 3:
            confidence -= 0.1

        # Age effects
        if 30 <= patient_profile.age <= 70:
            confidence += 0.1  # Middle-aged patients more predictable
        else:
            confidence -= 0.05

        # Data completeness
        if not patient_profile.lab_history.empty:
            confidence += 0.15

        return max(0.1, min(0.95, confidence))


def create_trajectory_demo():
    """Demonstrate temporal trajectory prediction system"""

    print("\n📈 TEMPORAL TRAJECTORY PREDICTION DEMONSTRATION")
    print("=" * 60)

    # Import patient profiles
    from modeling.personalized.avatar_integration import create_demo_patients

    # Initialize predictor
    predictor = TrajectoryPredictor()

    # Create demo patients
    patients = create_demo_patients()

    for patient in patients:
        print(f"\n👤 Patient: {patient.patient_id}")
        print(f"   Profile: Age {patient.age}, {patient.sex}")
        print(f"   Comorbidities: {', '.join(patient.comorbidities)}")

        # Predict trajectories for top biomarkers
        biomarkers = ["CRP", "APOB", "PCSK9"]
        current_values = {"CRP": 5.0, "APOB": 110.0, "PCSK9": 350.0}  # Example values

        for biomarker in biomarkers:
            print(f"\n   📊 {biomarker} Trajectory Prediction:")

            # Create prediction
            prediction = predictor.predict_biomarker_trajectory(
                biomarker,
                patient,
                current_values[biomarker],
                prediction_horizon_days=180,
            )

            print(f"     Current: {current_values[biomarker]:.1f}")
            print(
                f"     6-month prediction: {prediction.predicted_values[-1]:.1f} "
                f"(CI: {prediction.confidence_intervals[-1][0]:.1f}-{prediction.confidence_intervals[-1][1]:.1f})"
            )
            print(
                f"     Trend: {prediction.trend_direction.value} (slope: {prediction.trend_slope:.4f})"
            )
            print(f"     Model confidence: {prediction.model_confidence:.2f}")

            # Risk periods
            if prediction.risk_periods:
                print(f"     Risk periods: {len(prediction.risk_periods)}")
                for start, end, risk_level in prediction.risk_periods[
                    :2
                ]:  # Show first 2
                    duration = (end - start).days
                    print(
                        f"       {risk_level.value}: {duration} days starting {start.strftime('%Y-%m-%d')}"
                    )

            # Threshold crossings
            if prediction.threshold_crossings:
                print(
                    f"     Threshold crossings: {len(prediction.threshold_crossings)}"
                )
                for date, crossing_type, value in prediction.threshold_crossings[:2]:
                    print(
                        f"       {crossing_type}: {value:.1f} on {date.strftime('%Y-%m-%d')}"
                    )

            # Mechanism drivers
            print(f"     Key drivers: {', '.join(prediction.mechanism_drivers[:3])}")


if __name__ == "__main__":
    create_trajectory_demo()
