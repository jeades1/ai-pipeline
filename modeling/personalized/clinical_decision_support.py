"""
Clinical Decision Support API

Real-time clinical decision support system with threshold crossing detection,
intervention recommendations, and personalized alert systems for biomarker-based care.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import our personalized biomarker components
from .avatar_integration import PatientProfile, PersonalizedBiomarkerEngine
from .monitoring_system import MonitoringScheduleOptimizer
from .trajectory_prediction import (
    TrajectoryPredictor,
    BiomarkerPrediction,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels for clinical notifications"""

    CRITICAL = "critical"  # Immediate intervention required
    HIGH = "high"  # Action needed within 24 hours
    MODERATE = "moderate"  # Review within 3 days
    LOW = "low"  # Routine follow-up
    INFORMATIONAL = "info"  # FYI only


class InterventionType(Enum):
    """Types of clinical interventions"""

    MEDICATION_ADJUSTMENT = "medication_adjustment"
    LIFESTYLE_MODIFICATION = "lifestyle_modification"
    ADDITIONAL_TESTING = "additional_testing"
    SPECIALIST_REFERRAL = "specialist_referral"
    EMERGENCY_CARE = "emergency_care"
    MONITORING_FREQUENCY = "monitoring_frequency"
    PREVENTION_STRATEGY = "prevention_strategy"


class AlertStatus(Enum):
    """Status of clinical alerts"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class ThresholdCrossing:
    """Represents a biomarker threshold crossing event"""

    biomarker: str
    patient_id: str
    crossing_time: datetime
    threshold_type: (
        str  # 'upper_normal', 'lower_normal', 'critical_high', 'critical_low'
    )
    threshold_value: float
    actual_value: float
    severity_score: float  # 0-1 scale
    trend_direction: TrendDirection
    time_to_crossing: Optional[float] = None  # Hours until crossing (for predictions)
    confidence: float = 1.0


@dataclass
class ClinicalRecommendation:
    """Clinical intervention recommendation"""

    recommendation_id: str
    patient_id: str
    biomarker: str
    intervention_type: InterventionType
    priority: AlertPriority

    # Recommendation details
    title: str
    description: str
    rationale: str
    expected_outcome: str

    # Timing and urgency
    recommended_action_time: datetime
    urgency_hours: float

    # Supporting evidence
    evidence_strength: float  # 0-1 scale
    guidelines_reference: List[str] = field(default_factory=list)
    similar_cases: int = 0

    # Implementation
    implementation_complexity: str = "moderate"  # "low", "moderate", "high"
    estimated_cost: Optional[float] = None
    contraindications: List[str] = field(default_factory=list)

    # Follow-up
    monitoring_plan: str = ""
    expected_response_time: Optional[float] = None  # Hours


@dataclass
class ClinicalAlert:
    """Clinical alert for healthcare providers"""

    alert_id: str
    patient_id: str
    creation_time: datetime
    priority: AlertPriority
    title: str
    message: str
    status: AlertStatus = AlertStatus.ACTIVE

    # Alert content
    biomarkers_involved: List[str] = field(default_factory=list)

    # Clinical context
    threshold_crossings: List[ThresholdCrossing] = field(default_factory=list)
    recommendations: List[ClinicalRecommendation] = field(default_factory=list)

    # Metadata
    alert_source: str = "biomarker_engine"
    requires_acknowledgment: bool = True
    auto_dismiss_hours: Optional[float] = None

    # Workflow integration
    assigned_provider: Optional[str] = None
    department: Optional[str] = None
    care_team_notifications: List[str] = field(default_factory=list)


@dataclass
class PatientClinicalContext:
    """Current clinical context for a patient"""

    patient_id: str
    current_admission: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    care_team: List[str] = field(default_factory=list)

    # Current status
    acuity_level: str = "stable"  # "stable", "acute", "critical"
    location: str = "outpatient"  # "outpatient", "inpatient", "icu", "emergency"

    # Treatment context
    current_medications: List[str] = field(default_factory=list)
    recent_procedures: List[Dict] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)

    # Goals of care
    treatment_goals: List[str] = field(default_factory=list)
    patient_preferences: Dict[str, Any] = field(default_factory=dict)


class ThresholdManager:
    """Manages biomarker thresholds and crossing detection"""

    def __init__(self):
        # Standard clinical thresholds by biomarker
        self.clinical_thresholds = {
            "CRP": {
                "lower_normal": 0.0,
                "upper_normal": 3.0,
                "critical_high": 10.0,
                "units": "mg/L",
            },
            "APOB": {
                "lower_normal": 40.0,
                "upper_normal": 120.0,
                "critical_high": 150.0,
                "units": "mg/dL",
            },
            "PCSK9": {
                "lower_normal": 50.0,
                "upper_normal": 200.0,
                "critical_high": 400.0,
                "units": "ng/mL",
            },
            "LPA": {
                "lower_normal": 0.0,
                "upper_normal": 30.0,
                "critical_high": 50.0,
                "units": "mg/dL",
            },
            "HMGCR": {
                "lower_normal": 0.0,
                "upper_normal": 100.0,
                "critical_high": 200.0,
                "units": "U/L",
            },
        }

    def get_personalized_thresholds(
        self, biomarker: str, patient_profile: PatientProfile
    ) -> Dict[str, float]:
        """Calculate personalized thresholds based on patient characteristics"""
        base_thresholds = self.clinical_thresholds.get(biomarker, {})
        if not base_thresholds:
            return {}

        # Age adjustments
        age_factor = 1.0
        if patient_profile.age >= 65:
            age_factor = 1.2  # More permissive thresholds for elderly
        elif patient_profile.age <= 30:
            age_factor = 0.9  # Stricter thresholds for young adults

        # Sex adjustments
        sex_factor = 1.0
        if biomarker == "CRP" and patient_profile.sex == "female":
            sex_factor = 1.1  # Slightly higher baseline for females
        elif biomarker in ["APOB", "LPA"] and patient_profile.sex == "male":
            sex_factor = 1.05  # Slightly higher risk thresholds for males

        # Comorbidity adjustments
        comorbidity_factor = 1.0
        cardiovascular_conditions = [
            "coronary_artery_disease",
            "hypertension",
            "diabetes",
        ]
        if any(
            cond in patient_profile.comorbidities for cond in cardiovascular_conditions
        ):
            comorbidity_factor = 0.8  # Stricter thresholds for high-risk patients

        # Apply adjustments
        personalized_thresholds = {}
        for threshold_type, base_value in base_thresholds.items():
            if threshold_type != "units":
                adjusted_value = (
                    base_value * age_factor * sex_factor * comorbidity_factor
                )
                personalized_thresholds[threshold_type] = adjusted_value
            else:
                personalized_thresholds[threshold_type] = base_value

        return personalized_thresholds

    def detect_threshold_crossings(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        current_value: float,
        prediction: BiomarkerPrediction,
    ) -> List[ThresholdCrossing]:
        """Detect current and predicted threshold crossings"""
        thresholds = self.get_personalized_thresholds(biomarker, patient_profile)
        crossings = []

        # Check current value crossings
        for threshold_type, threshold_value in thresholds.items():
            if threshold_type == "units":
                continue

            # Determine if crossing occurred
            crossed = False
            severity = 0.0

            if (
                threshold_type in ["upper_normal", "critical_high"]
                and current_value > threshold_value
            ):
                crossed = True
                severity = min(1.0, (current_value - threshold_value) / threshold_value)
            elif (
                threshold_type in ["lower_normal", "critical_low"]
                and current_value < threshold_value
            ):
                crossed = True
                severity = min(1.0, (threshold_value - current_value) / threshold_value)

            if crossed:
                crossing = ThresholdCrossing(
                    biomarker=biomarker,
                    patient_id=patient_profile.patient_id,
                    crossing_time=datetime.now(),
                    threshold_type=threshold_type,
                    threshold_value=threshold_value,
                    actual_value=current_value,
                    severity_score=severity,
                    trend_direction=prediction.trend_direction,
                    confidence=1.0,
                )
                crossings.append(crossing)

        # Check predicted future crossings
        if prediction.threshold_crossings:
            for (
                crossing_time,
                threshold_type,
                predicted_value,
            ) in prediction.threshold_crossings:
                threshold_value = thresholds.get(threshold_type, 0)
                if threshold_value > 0:
                    severity = min(
                        1.0, abs(predicted_value - threshold_value) / threshold_value
                    )
                    time_to_crossing = (
                        crossing_time - datetime.now()
                    ).total_seconds() / 3600

                    crossing = ThresholdCrossing(
                        biomarker=biomarker,
                        patient_id=patient_profile.patient_id,
                        crossing_time=crossing_time,
                        threshold_type=threshold_type,
                        threshold_value=threshold_value,
                        actual_value=predicted_value,
                        severity_score=severity,
                        trend_direction=prediction.trend_direction,
                        time_to_crossing=time_to_crossing,
                        confidence=prediction.model_confidence,
                    )
                    crossings.append(crossing)

        return crossings


class RecommendationEngine:
    """Generates evidence-based clinical recommendations"""

    def __init__(self):
        # Clinical decision rules by biomarker and condition
        self.decision_rules = {
            "CRP": {
                "elevated_inflammatory": {
                    "conditions": lambda v, t: v > t.get("upper_normal", 3.0),
                    "interventions": [
                        {
                            "type": InterventionType.ADDITIONAL_TESTING,
                            "title": "Infection Workup",
                            "description": "Order blood cultures, urinalysis, and imaging as indicated",
                            "rationale": "Elevated CRP suggests acute inflammatory process",
                            "urgency_hours": 24,
                        },
                        {
                            "type": InterventionType.MEDICATION_ADJUSTMENT,
                            "title": "Anti-inflammatory Therapy",
                            "description": "Consider NSAIDs or corticosteroids based on clinical context",
                            "rationale": "Reduce inflammatory burden",
                            "urgency_hours": 48,
                        },
                    ],
                },
                "critical_elevation": {
                    "conditions": lambda v, t: v > t.get("critical_high", 10.0),
                    "interventions": [
                        {
                            "type": InterventionType.EMERGENCY_CARE,
                            "title": "Immediate Evaluation",
                            "description": "Emergency department evaluation for sepsis or severe inflammation",
                            "rationale": "Critical CRP elevation requires immediate assessment",
                            "urgency_hours": 2,
                        }
                    ],
                },
            },
            "APOB": {
                "cardiovascular_risk": {
                    "conditions": lambda v, t: v > t.get("upper_normal", 120.0),
                    "interventions": [
                        {
                            "type": InterventionType.MEDICATION_ADJUSTMENT,
                            "title": "Statin Therapy Optimization",
                            "description": "Increase statin dose or add ezetimibe",
                            "rationale": "Elevated ApoB indicates increased cardiovascular risk",
                            "urgency_hours": 168,  # 1 week
                        },
                        {
                            "type": InterventionType.LIFESTYLE_MODIFICATION,
                            "title": "Intensive Lifestyle Counseling",
                            "description": "Diet modification and exercise program",
                            "rationale": "Lifestyle changes can significantly reduce ApoB levels",
                            "urgency_hours": 336,  # 2 weeks
                        },
                    ],
                }
            },
            "PCSK9": {
                "high_levels": {
                    "conditions": lambda v, t: v > t.get("upper_normal", 200.0),
                    "interventions": [
                        {
                            "type": InterventionType.MEDICATION_ADJUSTMENT,
                            "title": "PCSK9 Inhibitor Consideration",
                            "description": "Evaluate for alirocumab or evolocumab therapy",
                            "rationale": "High PCSK9 levels may respond to targeted inhibition",
                            "urgency_hours": 336,  # 2 weeks
                        }
                    ],
                }
            },
        }

    def generate_recommendations(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        current_value: float,
        thresholds: Dict[str, float],
        crossings: List[ThresholdCrossing],
    ) -> List[ClinicalRecommendation]:
        """Generate clinical recommendations based on biomarker values and patient context"""
        recommendations = []

        rules = self.decision_rules.get(biomarker, {})

        for rule_name, rule_config in rules.items():
            condition_func = rule_config["conditions"]

            # Check if condition is met
            if condition_func(current_value, thresholds):
                for intervention_template in rule_config["interventions"]:
                    # Customize recommendation based on patient context
                    rec_id = f"{patient_profile.patient_id}_{biomarker}_{rule_name}_{datetime.now().isoformat()}"

                    # Adjust urgency based on patient acuity and comorbidities
                    urgency_adjustment = self._calculate_urgency_adjustment(
                        patient_profile, crossings
                    )
                    adjusted_urgency = (
                        intervention_template["urgency_hours"] * urgency_adjustment
                    )

                    recommendation = ClinicalRecommendation(
                        recommendation_id=rec_id,
                        patient_id=patient_profile.patient_id,
                        biomarker=biomarker,
                        intervention_type=intervention_template["type"],
                        priority=self._determine_priority(adjusted_urgency, crossings),
                        title=intervention_template["title"],
                        description=intervention_template["description"],
                        rationale=intervention_template["rationale"],
                        expected_outcome=f"Improvement in {biomarker} levels within 2-4 weeks",
                        recommended_action_time=datetime.now()
                        + timedelta(hours=adjusted_urgency),
                        urgency_hours=adjusted_urgency,
                        evidence_strength=0.8,  # Default high evidence strength
                        guidelines_reference=["AHA/ACC Guidelines", "ESC Guidelines"],
                        implementation_complexity="moderate",
                        monitoring_plan=f"Recheck {biomarker} in 4-6 weeks",
                    )

                    recommendations.append(recommendation)

        return recommendations

    def _calculate_urgency_adjustment(
        self, patient_profile: PatientProfile, crossings: List[ThresholdCrossing]
    ) -> float:
        """Calculate urgency adjustment factor based on patient risk"""
        adjustment = 1.0

        # Age adjustment
        if patient_profile.age >= 75:
            adjustment *= 0.8  # More urgent for elderly
        elif patient_profile.age <= 40:
            adjustment *= 1.2  # Less urgent for young adults

        # Comorbidity adjustment
        high_risk_conditions = [
            "coronary_artery_disease",
            "diabetes",
            "chronic_kidney_disease",
        ]
        if any(cond in patient_profile.comorbidities for cond in high_risk_conditions):
            adjustment *= 0.7  # More urgent for high-risk patients

        # Crossing severity adjustment
        max_severity = max([c.severity_score for c in crossings], default=0)
        if max_severity > 0.5:
            adjustment *= 0.6  # More urgent for severe crossings

        return max(0.1, adjustment)  # Minimum 10% of original urgency

    def _determine_priority(
        self, urgency_hours: float, crossings: List[ThresholdCrossing]
    ) -> AlertPriority:
        """Determine alert priority based on urgency and crossing severity"""
        max_severity = max([c.severity_score for c in crossings], default=0)

        if urgency_hours <= 4 or max_severity > 0.8:
            return AlertPriority.CRITICAL
        elif urgency_hours <= 24 or max_severity > 0.6:
            return AlertPriority.HIGH
        elif urgency_hours <= 72 or max_severity > 0.3:
            return AlertPriority.MODERATE
        else:
            return AlertPriority.LOW


class ClinicalDecisionSupportAPI:
    """Main clinical decision support system"""

    def __init__(self):
        self.personalized_engine = PersonalizedBiomarkerEngine()
        self.trajectory_predictor = TrajectoryPredictor()
        self.monitoring_optimizer = MonitoringScheduleOptimizer()
        self.threshold_manager = ThresholdManager()
        self.recommendation_engine = RecommendationEngine()

        # Active alerts storage (in production, this would be a database)
        self.active_alerts: Dict[str, List[ClinicalAlert]] = {}

    def evaluate_patient(
        self,
        patient_profile: PatientProfile,
        current_biomarkers: Dict[str, float],
        clinical_context: Optional[PatientClinicalContext] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive patient evaluation with clinical decision support

        Returns:
            Dictionary containing alerts, recommendations, and monitoring plans
        """
        logger.info(
            f"Starting clinical evaluation for patient {patient_profile.patient_id}"
        )

        evaluation_results = {
            "patient_id": patient_profile.patient_id,
            "evaluation_time": datetime.now(),
            "biomarkers_evaluated": list(current_biomarkers.keys()),
            "alerts": [],
            "recommendations": [],
            "monitoring_plan": {},
            "risk_assessment": {},
            "summary": {},
        }

        all_crossings = []
        all_recommendations = []

        # Evaluate each biomarker
        for biomarker, current_value in current_biomarkers.items():
            logger.debug(f"Evaluating {biomarker} = {current_value}")

            try:
                # Get trajectory prediction
                prediction = self.trajectory_predictor.predict_biomarker_trajectory(
                    biomarker=biomarker,
                    patient_profile=patient_profile,
                    current_value=current_value,
                    prediction_horizon_days=30,
                )

                # Detect threshold crossings
                thresholds = self.threshold_manager.get_personalized_thresholds(
                    biomarker, patient_profile
                )
                crossings = self.threshold_manager.detect_threshold_crossings(
                    biomarker, patient_profile, current_value, prediction
                )

                # Generate recommendations
                recommendations = self.recommendation_engine.generate_recommendations(
                    biomarker, patient_profile, current_value, thresholds, crossings
                )

                all_crossings.extend(crossings)
                all_recommendations.extend(recommendations)

                # Store biomarker-specific results
                evaluation_results["risk_assessment"][biomarker] = {
                    "current_value": current_value,
                    "personalized_thresholds": thresholds,
                    "prediction_summary": {
                        "trend": prediction.trend_direction.value,
                        "confidence": prediction.model_confidence,
                        "risk_periods": len(prediction.risk_periods),
                    },
                    "crossings": len(crossings),
                    "recommendations": len(recommendations),
                }

            except Exception as e:
                logger.error(f"Error evaluating {biomarker}: {e}")
                continue

        # Generate clinical alerts
        alerts = self._generate_clinical_alerts(
            patient_profile, all_crossings, all_recommendations, clinical_context
        )

        # Create monitoring plan
        monitoring_plan = self._create_monitoring_plan(
            patient_profile, current_biomarkers, all_crossings
        )

        # Compile results
        evaluation_results.update(
            {
                "alerts": [alert.__dict__ for alert in alerts],
                "recommendations": [rec.__dict__ for rec in all_recommendations],
                "monitoring_plan": monitoring_plan,
                "summary": {
                    "total_crossings": len(all_crossings),
                    "critical_alerts": len(
                        [a for a in alerts if a.priority == AlertPriority.CRITICAL]
                    ),
                    "high_priority_alerts": len(
                        [a for a in alerts if a.priority == AlertPriority.HIGH]
                    ),
                    "total_recommendations": len(all_recommendations),
                    "immediate_actions_required": len(
                        [r for r in all_recommendations if r.urgency_hours <= 24]
                    ),
                },
            }
        )

        # Store active alerts
        self.active_alerts[patient_profile.patient_id] = alerts

        logger.info(
            f"Clinical evaluation complete for {patient_profile.patient_id}: "
            f"{len(alerts)} alerts, {len(all_recommendations)} recommendations"
        )

        return evaluation_results

    def _generate_clinical_alerts(
        self,
        patient_profile: PatientProfile,
        crossings: List[ThresholdCrossing],
        recommendations: List[ClinicalRecommendation],
        clinical_context: Optional[PatientClinicalContext],
    ) -> List[ClinicalAlert]:
        """Generate clinical alerts based on crossings and recommendations"""
        alerts = []

        # Group crossings by priority
        critical_crossings = [c for c in crossings if c.severity_score > 0.8]
        high_crossings = [c for c in crossings if 0.5 < c.severity_score <= 0.8]
        moderate_crossings = [c for c in crossings if 0.2 < c.severity_score <= 0.5]

        # Critical alerts
        if critical_crossings:
            alert = ClinicalAlert(
                alert_id=f"CRIT_{patient_profile.patient_id}_{datetime.now().isoformat()}",
                patient_id=patient_profile.patient_id,
                creation_time=datetime.now(),
                priority=AlertPriority.CRITICAL,
                title="Critical Biomarker Abnormality",
                message=f"Critical threshold crossed for {len(critical_crossings)} biomarker(s). Immediate evaluation required.",
                biomarkers_involved=[c.biomarker for c in critical_crossings],
                threshold_crossings=critical_crossings,
                recommendations=[
                    r for r in recommendations if r.priority == AlertPriority.CRITICAL
                ],
                requires_acknowledgment=True,
                auto_dismiss_hours=None,  # Never auto-dismiss critical alerts
            )
            alerts.append(alert)

        # High priority alerts
        if high_crossings:
            alert = ClinicalAlert(
                alert_id=f"HIGH_{patient_profile.patient_id}_{datetime.now().isoformat()}",
                patient_id=patient_profile.patient_id,
                creation_time=datetime.now(),
                priority=AlertPriority.HIGH,
                title="Significant Biomarker Changes",
                message="Significant biomarker changes detected requiring attention within 24 hours.",
                biomarkers_involved=[c.biomarker for c in high_crossings],
                threshold_crossings=high_crossings,
                recommendations=[
                    r for r in recommendations if r.priority == AlertPriority.HIGH
                ],
                requires_acknowledgment=True,
                auto_dismiss_hours=48,
            )
            alerts.append(alert)

        # Moderate alerts
        if moderate_crossings:
            alert = ClinicalAlert(
                alert_id=f"MOD_{patient_profile.patient_id}_{datetime.now().isoformat()}",
                patient_id=patient_profile.patient_id,
                creation_time=datetime.now(),
                priority=AlertPriority.MODERATE,
                title="Biomarker Monitoring Alert",
                message="Biomarker trends require review and possible intervention.",
                biomarkers_involved=[c.biomarker for c in moderate_crossings],
                threshold_crossings=moderate_crossings,
                recommendations=[
                    r for r in recommendations if r.priority == AlertPriority.MODERATE
                ],
                requires_acknowledgment=False,
                auto_dismiss_hours=168,  # 1 week
            )
            alerts.append(alert)

        return alerts

    def _create_monitoring_plan(
        self,
        patient_profile: PatientProfile,
        current_biomarkers: Dict[str, float],
        crossings: List[ThresholdCrossing],
    ) -> Dict[str, Any]:
        """Create personalized monitoring plan"""

        # Create biomarker scores for monitoring optimizer
        biomarker_scores = []
        for biomarker in current_biomarkers:
            # Create a simple BiomarkerScore for each biomarker
            from modeling.personalized.avatar_integration import BiomarkerScore

            score = BiomarkerScore(
                biomarker=biomarker,
                population_rank=50,  # Default middle rank
                population_score=0.5,  # Default score
                personalized_score=0.5,
                confidence=0.8,
            )
            biomarker_scores.append(score)

        # Use existing monitoring optimizer
        monitoring_schedule = self.monitoring_optimizer.create_monitoring_plan(
            patient_profile, biomarker_scores
        )

        # Adjust frequencies based on threshold crossings
        biomarker_frequencies = {}
        for biomarker in current_biomarkers:
            # Get base frequency from monitoring schedule
            base_frequency = 90  # Default quarterly
            biomarker_schedule = monitoring_schedule.biomarker_schedules.get(biomarker)
            if biomarker_schedule:
                base_frequency = biomarker_schedule.adjusted_frequency_days

            # Check if this biomarker has crossings
            biomarker_crossings = [c for c in crossings if c.biomarker == biomarker]
            if biomarker_crossings:
                max_severity = max(c.severity_score for c in biomarker_crossings)
                if max_severity > 0.7:
                    adjusted_frequency = min(
                        3, base_frequency
                    )  # Every 3 days for severe
                elif max_severity > 0.5:
                    adjusted_frequency = min(7, base_frequency)  # Weekly for moderate
                else:
                    adjusted_frequency = min(14, base_frequency)  # Bi-weekly for mild
            else:
                adjusted_frequency = base_frequency

            biomarker_frequencies[biomarker] = adjusted_frequency

        return {
            "biomarker_frequencies": biomarker_frequencies,
            "next_monitoring_date": datetime.now()
            + timedelta(days=min(biomarker_frequencies.values())),
            "monitoring_rationale": "Frequencies adjusted based on threshold crossings and patient risk",
            "estimated_annual_cost": monitoring_schedule.cost_estimate,
            "preferred_collection_method": "venous_draw",
            "special_instructions": [],
        }

    def get_patient_alerts(
        self, patient_id: str, status_filter: Optional[AlertStatus] = None
    ) -> List[ClinicalAlert]:
        """Retrieve active alerts for a patient"""
        patient_alerts = self.active_alerts.get(patient_id, [])

        if status_filter:
            return [alert for alert in patient_alerts if alert.status == status_filter]

        return patient_alerts

    def acknowledge_alert(self, alert_id: str, provider_id: str) -> bool:
        """Acknowledge a clinical alert"""
        for patient_id, alerts in self.active_alerts.items():
            for alert in alerts:
                if alert.alert_id == alert_id:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.assigned_provider = provider_id
                    logger.info(f"Alert {alert_id} acknowledged by {provider_id}")
                    return True
        return False

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        total_alerts = sum(len(alerts) for alerts in self.active_alerts.values())
        active_alerts = sum(
            len([a for a in alerts if a.status == AlertStatus.ACTIVE])
            for alerts in self.active_alerts.values()
        )

        return {
            "total_patients_monitored": len(self.active_alerts),
            "total_alerts_generated": total_alerts,
            "active_alerts": active_alerts,
            "alert_distribution": {
                "critical": sum(
                    len([a for a in alerts if a.priority == AlertPriority.CRITICAL])
                    for alerts in self.active_alerts.values()
                ),
                "high": sum(
                    len([a for a in alerts if a.priority == AlertPriority.HIGH])
                    for alerts in self.active_alerts.values()
                ),
                "moderate": sum(
                    len([a for a in alerts if a.priority == AlertPriority.MODERATE])
                    for alerts in self.active_alerts.values()
                ),
            },
            "system_health": "operational",
        }
