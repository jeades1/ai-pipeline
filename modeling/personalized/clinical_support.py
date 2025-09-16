"""
Clinical Decision Support System for Personalized Biomarkers

Week 3-4: Clinical Integration and Multi-Endpoint Prediction
Extends the core engine with clinical decision support capabilities
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from modeling.personalized.engine import (
    PersonalizedBiomarkerEngine,
    PatientProfile,
    PersonalizedBiomarkerPanel,
)

logger = logging.getLogger(__name__)


@dataclass
class ClinicalAlert:
    """Clinical alert for biomarker values requiring attention"""

    patient_id: str
    biomarker: str
    current_value: float
    reference_range: Tuple[float, float]
    severity: str  # 'routine', 'urgent', 'immediate'
    recommendation: str
    confidence: float
    timestamp: datetime


@dataclass
class ClinicalOutcome:
    """Clinical outcome prediction"""

    outcome_type: str  # 'mortality_30d', 'aki_progression', 'cv_event', etc.
    probability: float
    confidence_interval: Tuple[float, float]
    risk_factors: List[str]
    biomarker_contributions: Dict[str, float]
    timeline_estimate: Optional[int] = None  # days


class ClinicalDecisionSupport:
    """
    Clinical decision support system for personalized biomarker monitoring

    Provides real-time recommendations, outcome predictions, and alerts
    based on patient-specific biomarker panels and current values.
    """

    def __init__(self, biomarker_engine: PersonalizedBiomarkerEngine):
        self.biomarker_engine = biomarker_engine

        # Clinical outcome models (simplified for demo)
        self.outcome_models = self._initialize_outcome_models()

        # Alert thresholds
        self.alert_config = self._initialize_alert_config()

    def _initialize_outcome_models(self) -> Dict:
        """Initialize clinical outcome prediction models"""
        return {
            "mortality_30d": {
                "biomarker_weights": {
                    "CRP": 0.25,
                    "BNP": 0.30,
                    "TROPONIN_I": 0.20,
                    "NGAL": 0.15,
                    "PCSK9": 0.10,
                },
                "comorbidity_weights": {
                    "age": 0.20,
                    "diabetes": 0.15,
                    "ckd": 0.25,
                    "cv_history": 0.40,
                },
                "baseline_risk": 0.05,
            },
            "aki_progression": {
                "biomarker_weights": {
                    "NGAL": 0.35,
                    "CRP": 0.20,
                    "PCSK9": 0.15,
                    "IL6": 0.15,
                    "TNFR1": 0.15,
                },
                "comorbidity_weights": {
                    "ckd": 0.40,
                    "diabetes": 0.30,
                    "hypertension": 0.20,
                    "age": 0.10,
                },
                "baseline_risk": 0.10,
            },
            "cv_event_1yr": {
                "biomarker_weights": {
                    "APOB": 0.25,
                    "LPA": 0.20,
                    "CRP": 0.15,
                    "HMGCR": 0.15,
                    "PCSK9": 0.15,
                    "BNP": 0.10,
                },
                "comorbidity_weights": {
                    "family_history_cv": 0.25,
                    "diabetes": 0.20,
                    "hypertension": 0.15,
                    "hyperlipidemia": 0.15,
                    "smoking": 0.15,
                    "age": 0.10,
                },
                "baseline_risk": 0.08,
            },
            "ckd_progression": {
                "biomarker_weights": {
                    "NGAL": 0.30,
                    "CRP": 0.25,
                    "IL6": 0.20,
                    "TNFR1": 0.15,
                    "PCSK9": 0.10,
                },
                "comorbidity_weights": {
                    "ckd": 0.35,
                    "diabetes": 0.30,
                    "hypertension": 0.25,
                    "age": 0.10,
                },
                "baseline_risk": 0.15,
            },
        }

    def _initialize_alert_config(self) -> Dict:
        """Initialize alert configuration"""
        return {
            "severity_thresholds": {
                "immediate": 2.5,  # > 2.5x upper reference range
                "urgent": 1.5,  # > 1.5x upper reference range
                "routine": 1.1,  # > 1.1x upper reference range
            },
            "trending_windows": {
                "short_term": 7,  # days
                "medium_term": 30,
                "long_term": 90,
            },
        }

    def analyze_current_biomarkers(
        self,
        patient_profile: PatientProfile,
        current_values: Dict[str, float],
        panel: PersonalizedBiomarkerPanel,
    ) -> Dict:
        """
        Analyze current biomarker values and generate clinical recommendations

        Args:
            patient_profile: Patient data
            current_values: Current biomarker measurements
            panel: Personalized biomarker panel for this patient

        Returns:
            Dictionary with alerts, recommendations, and outcome predictions
        """
        alerts = []
        recommendations = []

        # Check each biomarker against personalized reference ranges
        for biomarker, value in current_values.items():
            if biomarker in panel.reference_ranges:
                alert = self._check_biomarker_alert(
                    biomarker, value, patient_profile, panel
                )
                if alert:
                    alerts.append(alert)
                    recommendations.extend(
                        self._generate_biomarker_recommendations(alert, patient_profile)
                    )

        # Predict clinical outcomes
        outcome_predictions = self._predict_clinical_outcomes(
            patient_profile, current_values
        )

        # Generate next actions
        next_actions = self._generate_next_actions(alerts, outcome_predictions)

        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(current_values, patient_profile)

        return {
            "patient_id": patient_profile.patient_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "alerts": [alert.__dict__ for alert in alerts],
            "recommendations": recommendations,
            "outcome_predictions": [pred.__dict__ for pred in outcome_predictions],
            "next_actions": next_actions,
            "overall_risk_score": overall_risk,
            "summary": self._generate_summary(
                alerts, outcome_predictions, overall_risk
            ),
        }

    def _check_biomarker_alert(
        self,
        biomarker: str,
        value: float,
        patient_profile: PatientProfile,
        panel: PersonalizedBiomarkerPanel,
    ) -> Optional[ClinicalAlert]:
        """Check if biomarker value triggers an alert"""

        ref_range = panel.reference_ranges.get(biomarker)
        if not ref_range:
            return None

        ref_low, ref_high = ref_range["low"], ref_range["high"]

        # Check if value is outside reference range
        if ref_low <= value <= ref_high:
            return None  # Normal range, no alert

        # Determine severity
        if value > ref_high:
            excess_ratio = value / ref_high
            if excess_ratio >= self.alert_config["severity_thresholds"]["immediate"]:
                severity = "immediate"
            elif excess_ratio >= self.alert_config["severity_thresholds"]["urgent"]:
                severity = "urgent"
            else:
                severity = "routine"
        else:  # value < ref_low
            severity = "routine"  # Low values generally less concerning

        # Generate recommendation
        recommendation = self._generate_alert_recommendation(
            biomarker, value, ref_range, severity, patient_profile
        )

        # Calculate confidence
        confidence = panel.confidence_scores.get(biomarker, 0.5)

        return ClinicalAlert(
            patient_id=patient_profile.patient_id,
            biomarker=biomarker,
            current_value=value,
            reference_range=(ref_low, ref_high),
            severity=severity,
            recommendation=recommendation,
            confidence=confidence,
            timestamp=datetime.now(),
        )

    def _generate_alert_recommendation(
        self,
        biomarker: str,
        value: float,
        ref_range: Dict,
        severity: str,
        patient_profile: PatientProfile,
    ) -> str:
        """Generate specific recommendation for biomarker alert"""

        biomarker_actions = {
            "APOB": {
                "immediate": "Consider immediate cardiology consultation and aggressive lipid management",
                "urgent": "Increase statin dose, consider PCSK9 inhibitor, recheck in 2 weeks",
                "routine": "Lifestyle counseling, consider medication adjustment, recheck in 4 weeks",
            },
            "CRP": {
                "immediate": "Evaluate for acute infection or inflammatory condition, consider hospitalization",
                "urgent": "Anti-inflammatory therapy, search for underlying cause, recheck in 1 week",
                "routine": "Consider low-dose aspirin, lifestyle modification, recheck in 2 weeks",
            },
            "PCSK9": {
                "immediate": "Urgent lipidology consultation, consider PCSK9 inhibitor",
                "urgent": "PCSK9 inhibitor therapy, aggressive statin management",
                "routine": "Monitor lipid response to current therapy, consider dose adjustment",
            },
            "NGAL": {
                "immediate": "Urgent nephrology consultation, assess for acute kidney injury",
                "urgent": "Nephrotoxin review, IV hydration if appropriate, daily monitoring",
                "routine": "Monitor kidney function, review medications, follow-up in 1 week",
            },
        }

        default_actions = {
            "immediate": f"Urgent clinical evaluation for {biomarker} elevation",
            "urgent": f"Close monitoring and intervention for {biomarker}",
            "routine": f"Follow-up and reassessment of {biomarker} in 2-4 weeks",
        }

        actions = biomarker_actions.get(biomarker, default_actions)
        return actions.get(severity, default_actions[severity])

    def _generate_biomarker_recommendations(
        self, alert: ClinicalAlert, patient_profile: PatientProfile
    ) -> List[str]:
        """Generate comprehensive recommendations for biomarker alert"""
        recommendations = []

        # Add the primary recommendation
        recommendations.append(alert.recommendation)

        # Add patient-specific considerations
        if patient_profile.ckd and alert.biomarker in ["NGAL", "CRP"]:
            recommendations.append("Adjust dosing for CKD - avoid nephrotoxic agents")

        if patient_profile.diabetes and alert.biomarker in ["CRP", "IL6"]:
            recommendations.append("Consider diabetes management optimization")

        if (
            alert.biomarker in ["APOB", "LDLR", "PCSK9"]
            and patient_profile.hyperlipidemia
        ):
            recommendations.append("Reassess lipid management strategy and adherence")

        # Add monitoring recommendations
        if alert.severity == "immediate":
            recommendations.append("Monitor closely - consider inpatient management")
        elif alert.severity == "urgent":
            recommendations.append("Schedule follow-up within 1 week")
        else:
            recommendations.append("Schedule routine follow-up in 2-4 weeks")

        return recommendations

    def _predict_clinical_outcomes(
        self, patient_profile: PatientProfile, current_values: Dict[str, float]
    ) -> List[ClinicalOutcome]:
        """Predict clinical outcomes based on current biomarker values"""

        outcomes = []

        for outcome_type, model in self.outcome_models.items():

            # Calculate biomarker contribution
            biomarker_score = 0.0
            biomarker_contributions = {}

            for biomarker, weight in model["biomarker_weights"].items():
                if biomarker in current_values:
                    # Normalize biomarker value (simplified)
                    normalized_value = min(current_values[biomarker] / 100.0, 3.0)
                    contribution = normalized_value * weight
                    biomarker_score += contribution
                    biomarker_contributions[biomarker] = contribution

            # Calculate comorbidity contribution
            comorbidity_score = 0.0
            risk_factors = []

            for factor, weight in model["comorbidity_weights"].items():
                if factor == "age":
                    age_contribution = min(patient_profile.age / 80.0, 1.0) * weight
                    comorbidity_score += age_contribution
                    if patient_profile.age > 65:
                        risk_factors.append("advanced_age")
                elif factor == "diabetes" and patient_profile.diabetes:
                    comorbidity_score += weight
                    risk_factors.append("diabetes")
                elif factor == "ckd" and patient_profile.ckd:
                    comorbidity_score += weight
                    risk_factors.append("chronic_kidney_disease")
                elif factor == "hypertension" and patient_profile.hypertension:
                    comorbidity_score += weight
                    risk_factors.append("hypertension")
                elif factor == "hyperlipidemia" and patient_profile.hyperlipidemia:
                    comorbidity_score += weight
                    risk_factors.append("hyperlipidemia")
                elif (
                    factor == "family_history_cv" and patient_profile.family_history_cv
                ):
                    comorbidity_score += weight
                    risk_factors.append("family_history")
                elif factor == "smoking" and patient_profile.smoking:
                    comorbidity_score += weight
                    risk_factors.append("smoking")

            # Calculate final probability
            total_score = biomarker_score + comorbidity_score
            probability = model["baseline_risk"] + (total_score * 0.1)  # Scale factor
            probability = min(max(probability, 0.0), 1.0)  # Bound between 0 and 1

            # Simple confidence interval (±15%)
            ci_width = 0.15
            ci_lower = max(probability - ci_width, 0.0)
            ci_upper = min(probability + ci_width, 1.0)

            # Estimate timeline based on outcome type
            timeline_map = {
                "mortality_30d": 30,
                "aki_progression": 60,
                "cv_event_1yr": 365,
                "ckd_progression": 180,
            }

            outcomes.append(
                ClinicalOutcome(
                    outcome_type=outcome_type,
                    probability=probability,
                    confidence_interval=(ci_lower, ci_upper),
                    risk_factors=risk_factors,
                    biomarker_contributions=biomarker_contributions,
                    timeline_estimate=timeline_map.get(outcome_type),
                )
            )

        return outcomes

    def _generate_next_actions(
        self, alerts: List[ClinicalAlert], outcomes: List[ClinicalOutcome]
    ) -> List[str]:
        """Generate prioritized next actions"""
        actions = []

        # Immediate actions for critical alerts
        immediate_alerts = [a for a in alerts if a.severity == "immediate"]
        if immediate_alerts:
            actions.append("🚨 IMMEDIATE: Address critical biomarker elevations")
            for alert in immediate_alerts:
                actions.append(f"   • {alert.biomarker}: {alert.recommendation}")

        # High-risk outcome actions
        high_risk_outcomes = [o for o in outcomes if o.probability > 0.3]
        if high_risk_outcomes:
            actions.append("⚠️ HIGH RISK: Consider preventive interventions")
            for outcome in high_risk_outcomes:
                outcome_name = outcome.outcome_type.replace("_", " ").title()
                actions.append(f"   • {outcome_name}: {outcome.probability:.1%} risk")

        # Routine monitoring actions
        urgent_alerts = [a for a in alerts if a.severity == "urgent"]
        if urgent_alerts:
            actions.append("📋 URGENT: Schedule follow-up within 1 week")

        routine_alerts = [a for a in alerts if a.severity == "routine"]
        if routine_alerts:
            actions.append("📅 ROUTINE: Schedule follow-up in 2-4 weeks")

        if not actions:
            actions.append("✅ Continue current monitoring schedule")

        return actions

    def _calculate_overall_risk(
        self, current_values: Dict[str, float], patient_profile: PatientProfile
    ) -> float:
        """Calculate overall patient risk score"""

        # Weight different risk components
        biomarker_risk = 0.0
        comorbidity_risk = 0.0

        # Biomarker component (simplified)
        high_risk_biomarkers = ["CRP", "BNP", "TROPONIN_I", "NGAL"]
        for biomarker in high_risk_biomarkers:
            if biomarker in current_values:
                # Normalize and add to risk
                normalized = min(current_values[biomarker] / 100.0, 2.0)
                biomarker_risk += normalized * 0.1

        # Comorbidity component
        if patient_profile.age > 75:
            comorbidity_risk += 0.2
        elif patient_profile.age > 65:
            comorbidity_risk += 0.1

        if patient_profile.diabetes:
            comorbidity_risk += 0.15
        if patient_profile.ckd:
            comorbidity_risk += 0.2
        if patient_profile.family_history_cv:
            comorbidity_risk += 0.1

        total_risk = min(biomarker_risk + comorbidity_risk, 1.0)
        return total_risk

    def _generate_summary(
        self,
        alerts: List[ClinicalAlert],
        outcomes: List[ClinicalOutcome],
        overall_risk: float,
    ) -> str:
        """Generate clinical summary"""

        alert_count = len(alerts)
        immediate_count = len([a for a in alerts if a.severity == "immediate"])
        urgent_count = len([a for a in alerts if a.severity == "urgent"])

        high_risk_outcomes = [o for o in outcomes if o.probability > 0.2]

        summary_parts = []

        # Risk level
        if overall_risk > 0.7:
            summary_parts.append("🔴 HIGH RISK patient")
        elif overall_risk > 0.4:
            summary_parts.append("🟡 MODERATE RISK patient")
        else:
            summary_parts.append("🟢 LOW RISK patient")

        # Alert summary
        if immediate_count > 0:
            summary_parts.append(f"{immediate_count} IMMEDIATE alert(s)")
        if urgent_count > 0:
            summary_parts.append(f"{urgent_count} urgent alert(s)")
        if alert_count == 0:
            summary_parts.append("No biomarker alerts")

        # Outcome summary
        if high_risk_outcomes:
            outcome_names = [
                o.outcome_type.replace("_", " ") for o in high_risk_outcomes[:2]
            ]
            summary_parts.append(f"Elevated risk for: {', '.join(outcome_names)}")

        return " | ".join(summary_parts)

    def generate_monitoring_recommendations(
        self,
        patient_profile: PatientProfile,
        panel: PersonalizedBiomarkerPanel,
        current_analysis: Dict,
    ) -> Dict:
        """Generate updated monitoring recommendations based on current analysis"""

        recommendations = {
            "patient_id": patient_profile.patient_id,
            "updated_schedule": {},
            "additional_tests": [],
            "medication_considerations": [],
            "lifestyle_recommendations": [],
        }

        # Adjust monitoring frequencies based on alerts
        alerts = [
            ClinicalAlert(**alert_data) for alert_data in current_analysis["alerts"]
        ]

        for biomarker in panel.primary_biomarkers + panel.secondary_biomarkers:
            current_freq = panel.monitoring_schedule.get(biomarker, {}).get(
                "frequency_days", 60
            )

            # Check if this biomarker has alerts
            biomarker_alerts = [a for a in alerts if a.biomarker == biomarker]

            if biomarker_alerts:
                # Increase monitoring frequency for alerted biomarkers
                max_severity = max(
                    [a.severity for a in biomarker_alerts],
                    key=lambda x: ["routine", "urgent", "immediate"].index(x),
                )

                if max_severity == "immediate":
                    new_freq = 3  # Every 3 days
                elif max_severity == "urgent":
                    new_freq = 7  # Weekly
                else:
                    new_freq = max(
                        current_freq // 2, 14
                    )  # Twice as frequent, min 2 weeks
            else:
                new_freq = current_freq

            recommendations["updated_schedule"][biomarker] = {
                "frequency_days": new_freq,
                "reason": f"Adjusted based on {'alert severity' if biomarker_alerts else 'stable values'}",
            }

        # Additional test recommendations based on risk profile
        overall_risk = current_analysis["overall_risk_score"]
        if overall_risk > 0.6:
            recommendations["additional_tests"].extend(
                [
                    "Complete metabolic panel",
                    "Inflammatory markers panel",
                    "Cardiac biomarkers if indicated",
                ]
            )

        # Medication considerations
        if any(
            alert["biomarker"] in ["APOB", "LDLR", "PCSK9"]
            for alert in current_analysis["alerts"]
        ):
            recommendations["medication_considerations"].append(
                "Consider lipid-lowering therapy optimization"
            )

        if any(
            alert["biomarker"] in ["CRP", "IL6"] for alert in current_analysis["alerts"]
        ):
            recommendations["medication_considerations"].append(
                "Consider anti-inflammatory therapy evaluation"
            )

        # Lifestyle recommendations
        if patient_profile.bmi > 25:
            recommendations["lifestyle_recommendations"].append(
                "Weight management program"
            )

        if patient_profile.exercise_level == "low":
            recommendations["lifestyle_recommendations"].append(
                "Structured exercise program"
            )

        if patient_profile.smoking:
            recommendations["lifestyle_recommendations"].append(
                "Smoking cessation program"
            )

        return recommendations


# Demo and testing functions


def run_clinical_decision_support_demo():
    """Comprehensive demo of clinical decision support system"""

    print("🏥 CLINICAL DECISION SUPPORT SYSTEM DEMO")
    print("=" * 60)

    # Initialize system
    biomarker_engine = PersonalizedBiomarkerEngine()
    clinical_system = ClinicalDecisionSupport(biomarker_engine)

    # Create demo patient
    patient = PatientProfile(
        patient_id="CLINICAL_DEMO_001",
        age=72,
        sex="female",
        bmi=31.2,
        diabetes=True,
        ckd=True,
        hyperlipidemia=True,
        family_history_cv=True,
        medications=["metformin", "lisinopril", "atorvastatin", "furosemide"],
        genetic_risk_cv=0.8,
        framingham_risk=0.7,
        smoking=False,
        exercise_level="low",
    )

    # Generate personalized panel
    panel = biomarker_engine.generate_personalized_panel(patient, "cardiovascular")

    print(f"Patient: {patient.patient_id} (Age {patient.age}, {patient.sex})")
    print("Conditions: Diabetes, CKD, Hyperlipidemia, Family History")
    print(f"Primary Biomarkers: {', '.join(panel.primary_biomarkers)}")

    # Simulate current biomarker values (some elevated)
    current_values = {
        "APOB": 140,  # Elevated (normal ~100)
        "PCSK9": 520,  # Elevated (normal ~300)
        "CRP": 12.5,  # Significantly elevated (normal <3)
        "NGAL": 180,  # Elevated (normal <100)
        "LPA": 65,  # Elevated (normal <30)
        "HMGCR": 2.8,  # Elevated (normal <2.0)
        "BNP": 450,  # Elevated (normal <100)
    }

    print("\nCurrent Biomarker Values:")
    for biomarker, value in current_values.items():
        print(f"  {biomarker}: {value}")

    # Analyze current biomarkers
    analysis = clinical_system.analyze_current_biomarkers(
        patient, current_values, panel
    )

    print("\n🔍 CLINICAL ANALYSIS")
    print(f"Overall Risk Score: {analysis['overall_risk_score']:.2f}")
    print(f"Summary: {analysis['summary']}")

    print(f"\n🚨 ALERTS ({len(analysis['alerts'])} total):")
    for alert in analysis["alerts"]:
        print(
            f"  {alert['severity'].upper()}: {alert['biomarker']} = {alert['current_value']}"
        )
        print(
            f"    Reference: {alert['reference_range'][0]:.1f} - {alert['reference_range'][1]:.1f}"
        )
        print(f"    Recommendation: {alert['recommendation']}")

    print("\n📊 OUTCOME PREDICTIONS:")
    for outcome in analysis["outcome_predictions"]:
        print(
            f"  {outcome['outcome_type'].replace('_', ' ').title()}: {outcome['probability']:.1%}"
        )
        if outcome["risk_factors"]:
            print(f"    Risk factors: {', '.join(outcome['risk_factors'])}")

    print("\n📋 NEXT ACTIONS:")
    for action in analysis["next_actions"]:
        print(f"  {action}")

    # Generate monitoring recommendations
    monitoring_recs = clinical_system.generate_monitoring_recommendations(
        patient, panel, analysis
    )

    print("\n📅 UPDATED MONITORING SCHEDULE:")
    for biomarker, schedule in monitoring_recs["updated_schedule"].items():
        print(
            f"  {biomarker}: Every {schedule['frequency_days']} days ({schedule['reason']})"
        )

    if monitoring_recs["additional_tests"]:
        print("\n🔬 ADDITIONAL TESTS RECOMMENDED:")
        for test in monitoring_recs["additional_tests"]:
            print(f"  • {test}")

    if monitoring_recs["medication_considerations"]:
        print("\n💊 MEDICATION CONSIDERATIONS:")
        for med in monitoring_recs["medication_considerations"]:
            print(f"  • {med}")

    print("\n✅ Clinical decision support system successfully demonstrated!")


if __name__ == "__main__":
    run_clinical_decision_support_demo()
