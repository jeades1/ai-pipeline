"""
Risk-Stratified Monitoring System

Creates dynamic monitoring schedules based on patient risk profiles,
biomarker volatility analysis, and personalized testing frequencies.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from modeling.personalized.avatar_integration import PatientProfile, BiomarkerScore

logger = logging.getLogger(__name__)


class MonitoringPriority(Enum):
    """Monitoring priority levels"""

    CRITICAL = "critical"  # Daily to weekly
    HIGH = "high"  # Weekly to bi-weekly
    STANDARD = "standard"  # Monthly
    LOW = "low"  # Quarterly
    MAINTENANCE = "maintenance"  # Semi-annually


class RiskLevel(Enum):
    """Patient risk stratification levels"""

    VERY_HIGH = "very_high"  # Multiple comorbidities, elderly, ICU
    HIGH = "high"  # Significant comorbidities, age >65
    MODERATE = "moderate"  # Some risk factors
    LOW = "low"  # Young, healthy, minimal risk


@dataclass
class MonitoringSchedule:
    """Individual biomarker monitoring schedule"""

    biomarker: str
    patient_id: str

    # Schedule parameters
    base_frequency_days: int
    adjusted_frequency_days: int
    priority: MonitoringPriority
    next_test_date: datetime

    # Modifying factors
    risk_adjustment: float = 1.0
    volatility_adjustment: float = 1.0
    clinical_urgency_adjustment: float = 1.0

    # Performance tracking
    adherence_score: float = 1.0
    missed_tests: int = 0
    early_detections: int = 0

    # Clinical context
    indication: str = ""
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientMonitoringPlan:
    """Complete monitoring plan for a patient"""

    patient_id: str
    patient_risk_level: RiskLevel

    # Biomarker schedules
    biomarker_schedules: Dict[str, MonitoringSchedule] = field(default_factory=dict)

    # Plan metadata
    plan_created: datetime = field(default_factory=datetime.now)
    plan_updated: datetime = field(default_factory=datetime.now)
    next_review_date: datetime = field(default_factory=datetime.now)

    # Clinical oversight
    ordering_physician: str = ""
    care_team_notes: str = ""
    cost_estimate: float = 0.0


class BiomarkerVolatilityAnalyzer:
    """
    Analyzes biomarker volatility patterns to optimize monitoring frequency
    """

    def __init__(self):
        # Volatility profiles from literature and clinical experience
        self.volatility_profiles = {
            "APOB": {
                "half_life_days": 3,
                "cv_percent": 8,  # Coefficient of variation
                "seasonal_variation": False,
                "meal_dependent": True,
                "diurnal_variation": False,
                "stress_responsive": False,
            },
            "HMGCR": {
                "half_life_days": 1,
                "cv_percent": 15,
                "seasonal_variation": False,
                "meal_dependent": False,
                "diurnal_variation": True,  # Cholesterol synthesis peaks at night
                "stress_responsive": True,
            },
            "PCSK9": {
                "half_life_days": 7,
                "cv_percent": 12,
                "seasonal_variation": False,
                "meal_dependent": False,
                "diurnal_variation": True,
                "stress_responsive": True,
            },
            "CRP": {
                "half_life_days": 0.8,
                "cv_percent": 35,  # Highly variable
                "seasonal_variation": True,  # Infections more common in winter
                "meal_dependent": False,
                "diurnal_variation": False,
                "stress_responsive": True,
            },
            "LPA": {
                "half_life_days": 1095,  # ~3 years - genetic
                "cv_percent": 3,  # Very stable
                "seasonal_variation": False,
                "meal_dependent": False,
                "diurnal_variation": False,
                "stress_responsive": False,
            },
            "LDLR": {
                "half_life_days": 2,
                "cv_percent": 10,
                "seasonal_variation": False,
                "meal_dependent": True,
                "diurnal_variation": True,
                "stress_responsive": True,
            },
            "LPL": {
                "half_life_days": 0.5,
                "cv_percent": 20,
                "seasonal_variation": False,
                "meal_dependent": True,
                "diurnal_variation": False,
                "stress_responsive": True,
            },
            "ADIPOQ": {
                "half_life_days": 6,
                "cv_percent": 18,
                "seasonal_variation": True,
                "meal_dependent": False,
                "diurnal_variation": True,
                "stress_responsive": True,
            },
            "CETP": {
                "half_life_days": 2,
                "cv_percent": 12,
                "seasonal_variation": False,
                "meal_dependent": False,
                "diurnal_variation": False,
                "stress_responsive": False,
            },
            "APOA1": {
                "half_life_days": 4,
                "cv_percent": 8,
                "seasonal_variation": False,
                "meal_dependent": True,
                "diurnal_variation": False,
                "stress_responsive": False,
            },
        }

    def get_volatility_score(self, biomarker: str) -> float:
        """Calculate normalized volatility score (0-1, higher = more volatile)"""
        profile = self.volatility_profiles.get(biomarker, {})

        # Base volatility from coefficient of variation
        cv = profile.get("cv_percent", 15) / 100
        base_volatility = min(1.0, cv / 0.5)  # Normalize to 50% CV = 1.0

        # Adjust for additional factors
        volatility_modifiers = 0
        if profile.get("seasonal_variation", False):
            volatility_modifiers += 0.1
        if profile.get("meal_dependent", False):
            volatility_modifiers += 0.1
        if profile.get("diurnal_variation", False):
            volatility_modifiers += 0.1
        if profile.get("stress_responsive", False):
            volatility_modifiers += 0.2

        final_volatility = min(1.0, base_volatility + volatility_modifiers)
        return final_volatility

    def get_optimal_sampling_interval(
        self, biomarker: str, detection_threshold: float = 0.2
    ) -> int:
        """
        Calculate optimal sampling interval to detect meaningful changes

        Args:
            biomarker: Biomarker name
            detection_threshold: Minimum % change to detect (0.2 = 20%)

        Returns:
            Recommended sampling interval in days
        """
        profile = self.volatility_profiles.get(biomarker, {})

        # Half-life based calculation
        half_life = profile.get("half_life_days", 7)
        cv = profile.get("cv_percent", 15) / 100

        # For detecting changes above biological noise
        # Use 2-3 times the time needed for signal to exceed noise
        noise_threshold = cv * 2  # 2x CV for 95% confidence

        if detection_threshold < noise_threshold:
            # Need multiple measurements to detect small changes
            optimal_interval = max(1, half_life // 3)
        else:
            # Can use longer intervals for large changes
            optimal_interval = min(half_life * 2, 30)  # Cap at monthly

        return int(optimal_interval)


class PatientRiskStratifier:
    """
    Stratifies patients into risk categories for monitoring optimization
    """

    def calculate_patient_risk_score(
        self, patient_profile: PatientProfile
    ) -> Tuple[float, RiskLevel]:
        """
        Calculate comprehensive patient risk score

        Args:
            patient_profile: Patient data

        Returns:
            Tuple of (risk_score, risk_level)
        """
        risk_score = 0.0

        # Age factor (0-30 points)
        age_score = min(30, max(0, (patient_profile.age - 30) / 2))
        risk_score += age_score

        # Comorbidity factors (0-40 points)
        comorbidity_weights = {
            "diabetes": 8,
            "ckd": 10,
            "heart_disease": 12,
            "hypertension": 5,
            "hyperlipidemia": 3,
            "inflammation": 6,
            "family_history": 4,
        }

        comorbidity_score = sum(
            comorbidity_weights.get(condition, 2)
            for condition in patient_profile.comorbidities
        )
        comorbidity_score = min(40, comorbidity_score)
        risk_score += comorbidity_score

        # Laboratory abnormalities (0-20 points)
        lab_score = self._calculate_lab_risk_score(patient_profile.lab_history)
        risk_score += lab_score

        # Genetic risk factors (0-10 points)
        genetic_score = sum(patient_profile.genetic_risk_scores.values()) * 10
        genetic_score = min(10, genetic_score)
        risk_score += genetic_score

        # Normalize to 0-100 scale
        risk_score = min(100, risk_score)

        # Convert to risk level
        if risk_score >= 80:
            risk_level = RiskLevel.VERY_HIGH
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 40:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW

        return risk_score, risk_level

    def _calculate_lab_risk_score(self, lab_history: pd.DataFrame) -> float:
        """Calculate risk score from laboratory abnormalities"""
        if lab_history.empty:
            return 0.0

        risk_score = 0.0

        # Check for abnormal values
        abnormal_thresholds = {
            "creatinine": (0.6, 1.3),  # Normal range
            "sodium": (135, 145),
            "potassium": (3.5, 5.0),
            "urine_output": (800, 2000),
        }

        for lab, (low, high) in abnormal_thresholds.items():
            if lab in lab_history.columns:
                values = lab_history[lab].dropna()
                if not values.empty:
                    # Calculate percentage of abnormal values
                    abnormal_count = ((values < low) | (values > high)).sum()
                    abnormal_pct = abnormal_count / len(values)
                    risk_score += abnormal_pct * 5  # Max 5 points per lab

        return min(20, risk_score)


class MonitoringScheduleOptimizer:
    """
    Optimizes monitoring schedules based on risk, volatility, and clinical priorities
    """

    def __init__(self):
        self.volatility_analyzer = BiomarkerVolatilityAnalyzer()
        self.risk_stratifier = PatientRiskStratifier()

        # Base monitoring frequencies by priority (days)
        self.base_frequencies = {
            MonitoringPriority.CRITICAL: 3,
            MonitoringPriority.HIGH: 14,
            MonitoringPriority.STANDARD: 30,
            MonitoringPriority.LOW: 90,
            MonitoringPriority.MAINTENANCE: 180,
        }

    def create_monitoring_plan(
        self, patient_profile: PatientProfile, biomarker_scores: List[BiomarkerScore]
    ) -> PatientMonitoringPlan:
        """
        Create comprehensive monitoring plan for a patient

        Args:
            patient_profile: Patient data
            biomarker_scores: Personalized biomarker rankings

        Returns:
            Complete monitoring plan
        """
        # Calculate patient risk
        risk_score, risk_level = self.risk_stratifier.calculate_patient_risk_score(
            patient_profile
        )

        # Create monitoring schedules for top biomarkers
        biomarker_schedules = {}

        for i, biomarker_score in enumerate(biomarker_scores[:10]):  # Top 10 biomarkers
            schedule = self._create_biomarker_schedule(
                biomarker_score, patient_profile, risk_level, i + 1
            )
            biomarker_schedules[biomarker_score.biomarker] = schedule

        # Calculate next review date (based on highest priority biomarker)
        next_review = self._calculate_next_review_date(biomarker_schedules)

        # Estimate costs
        cost_estimate = self._estimate_monitoring_costs(biomarker_schedules)

        return PatientMonitoringPlan(
            patient_id=patient_profile.patient_id,
            patient_risk_level=risk_level,
            biomarker_schedules=biomarker_schedules,
            next_review_date=next_review,
            cost_estimate=cost_estimate,
        )

    def _create_biomarker_schedule(
        self,
        biomarker_score: BiomarkerScore,
        patient_profile: PatientProfile,
        patient_risk_level: RiskLevel,
        rank: int,
    ) -> MonitoringSchedule:
        """Create individual biomarker monitoring schedule"""

        # Determine base priority from ranking and personalized score
        if rank <= 2 and biomarker_score.personalized_score > 1.0:
            base_priority = MonitoringPriority.HIGH
        elif rank <= 5:
            base_priority = MonitoringPriority.STANDARD
        elif rank <= 8:
            base_priority = MonitoringPriority.LOW
        else:
            base_priority = MonitoringPriority.MAINTENANCE

        # Adjust priority based on patient risk
        if patient_risk_level == RiskLevel.VERY_HIGH:
            if base_priority == MonitoringPriority.STANDARD:
                priority = MonitoringPriority.HIGH
            elif base_priority == MonitoringPriority.LOW:
                priority = MonitoringPriority.STANDARD
            else:
                priority = base_priority
        else:
            priority = base_priority

        # Get base frequency
        base_frequency = self.base_frequencies[priority]

        # Apply risk adjustment
        risk_adjustments = {
            RiskLevel.VERY_HIGH: 0.5,  # 2x more frequent
            RiskLevel.HIGH: 0.7,  # 1.4x more frequent
            RiskLevel.MODERATE: 1.0,  # No change
            RiskLevel.LOW: 1.5,  # Less frequent
        }
        risk_adjustment = risk_adjustments[patient_risk_level]

        # Apply volatility adjustment
        volatility_score = self.volatility_analyzer.get_volatility_score(
            biomarker_score.biomarker
        )
        volatility_adjustment = 0.5 + (1 - volatility_score) * 0.5  # Range: 0.5-1.0

        # Apply clinical urgency (based on biomarker score and confidence)
        urgency_factor = min(
            2.0, biomarker_score.personalized_score / biomarker_score.population_score
        )
        confidence_factor = biomarker_score.confidence
        clinical_urgency_adjustment = 0.8 + (urgency_factor * confidence_factor * 0.4)

        # Calculate final frequency
        adjusted_frequency = int(
            base_frequency
            * risk_adjustment
            * volatility_adjustment
            * clinical_urgency_adjustment
        )

        # Constraints
        adjusted_frequency = max(1, min(365, adjusted_frequency))  # 1 day to 1 year

        # Calculate next test date
        next_test_date = datetime.now() + timedelta(days=adjusted_frequency)

        # Create alert thresholds
        alert_thresholds = self._create_alert_thresholds(
            biomarker_score.biomarker, patient_profile
        )

        return MonitoringSchedule(
            biomarker=biomarker_score.biomarker,
            patient_id=patient_profile.patient_id,
            base_frequency_days=base_frequency,
            adjusted_frequency_days=adjusted_frequency,
            priority=priority,
            next_test_date=next_test_date,
            risk_adjustment=risk_adjustment,
            volatility_adjustment=volatility_adjustment,
            clinical_urgency_adjustment=clinical_urgency_adjustment,
            indication=biomarker_score.mechanism_relevance,
            alert_thresholds=alert_thresholds,
        )

    def _create_alert_thresholds(
        self, biomarker: str, patient_profile: PatientProfile
    ) -> Dict[str, float]:
        """Create personalized alert thresholds for biomarker"""

        # Base reference ranges (these would come from lab databases)
        base_ranges = {
            "APOB": {"low": 60, "high": 120, "critical_high": 200},  # mg/dL
            "CRP": {"low": 0.1, "high": 3.0, "critical_high": 10.0},  # mg/L
            "PCSK9": {"low": 150, "high": 400, "critical_high": 800},  # ng/mL
            "LPA": {"low": 0, "high": 30, "critical_high": 50},  # mg/dL
            "HMGCR": {"low": 1.0, "high": 5.0, "critical_high": 10.0},  # Activity units
        }

        base_range = base_ranges.get(
            biomarker, {"low": 0, "high": 100, "critical_high": 200}
        )

        # Personalize based on patient characteristics
        if "diabetes" in patient_profile.comorbidities:
            # Tighter control for diabetics
            base_range["high"] *= 0.8
            base_range["critical_high"] *= 0.9

        if patient_profile.age > 70:
            # Slightly relaxed for elderly
            base_range["high"] *= 1.1

        return base_range

    def _calculate_next_review_date(
        self, schedules: Dict[str, MonitoringSchedule]
    ) -> datetime:
        """Calculate when to review overall monitoring plan"""
        if not schedules:
            return datetime.now() + timedelta(days=90)

        # Review when highest priority biomarker is due
        earliest_test = min(schedule.next_test_date for schedule in schedules.values())

        # But not more than 6 months out
        max_review = datetime.now() + timedelta(days=180)

        return min(earliest_test, max_review)

    def _estimate_monitoring_costs(
        self, schedules: Dict[str, MonitoringSchedule]
    ) -> float:
        """Estimate annual monitoring costs"""

        # Estimated costs per test (USD)
        test_costs = {
            "APOB": 25,
            "CRP": 15,
            "PCSK9": 45,
            "LPA": 35,
            "HMGCR": 30,
            "LDLR": 40,
            "LPL": 35,
            "ADIPOQ": 25,
            "CETP": 30,
            "APOA1": 20,
        }

        annual_cost = 0.0

        for biomarker, schedule in schedules.items():
            test_cost = test_costs.get(biomarker, 25)
            tests_per_year = 365 / schedule.adjusted_frequency_days
            annual_cost += test_cost * tests_per_year

        return annual_cost


def create_monitoring_demo():
    """Demonstrate risk-stratified monitoring system"""

    print("\n⏰ RISK-STRATIFIED MONITORING SYSTEM DEMONSTRATION")
    print("=" * 65)

    # Import patient profiles from avatar integration
    from modeling.personalized.avatar_integration import (
        create_demo_patients,
        PersonalizedBiomarkerEngine,
    )

    # Initialize systems
    engine = PersonalizedBiomarkerEngine()
    optimizer = MonitoringScheduleOptimizer()

    # Create demo patients
    patients = create_demo_patients()

    for patient in patients:
        print(f"\n👤 Patient: {patient.patient_id}")
        print(f"   Profile: Age {patient.age}, {patient.sex}, BMI {patient.bmi}")
        print(f"   Comorbidities: {', '.join(patient.comorbidities)}")

        # Generate personalized biomarker scores
        biomarker_scores = engine.generate_personalized_scores(patient)

        # Create monitoring plan
        monitoring_plan = optimizer.create_monitoring_plan(patient, biomarker_scores)

        print(f"   Risk Level: {monitoring_plan.patient_risk_level.value}")
        print(f"   Annual Cost Estimate: ${monitoring_plan.cost_estimate:.0f}")

        print("\n   📋 Monitoring Schedule:")
        for biomarker, schedule in list(monitoring_plan.biomarker_schedules.items())[
            :5
        ]:
            print(f"     {biomarker}:")
            print(
                f"       Frequency: Every {schedule.adjusted_frequency_days} days ({schedule.priority.value})"
            )
            print(f"       Next Test: {schedule.next_test_date.strftime('%Y-%m-%d')}")
            print(
                f"       Adjustments: Risk={schedule.risk_adjustment:.2f}, "
                f"Volatility={schedule.volatility_adjustment:.2f}"
            )

        print("\n   🔍 Volatility Analysis (Top 3):")
        for biomarker_score in biomarker_scores[:3]:
            biomarker = biomarker_score.biomarker
            volatility = optimizer.volatility_analyzer.get_volatility_score(biomarker)
            optimal_interval = (
                optimizer.volatility_analyzer.get_optimal_sampling_interval(biomarker)
            )

            print(
                f"     {biomarker}: Volatility={volatility:.2f}, "
                f"Optimal interval={optimal_interval} days"
            )


if __name__ == "__main__":
    create_monitoring_demo()
