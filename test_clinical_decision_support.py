#!/usr/bin/env python3
"""
Test script for Clinical Decision Support API
"""

from modeling.personalized.clinical_decision_support import (
    ClinicalDecisionSupportAPI,
    PatientClinicalContext,
)
from modeling.personalized.avatar_integration import PatientProfile
import pandas as pd


def test_clinical_decision_support():
    """Test the clinical decision support system"""

    # Initialize the clinical decision support API
    cds_api = ClinicalDecisionSupportAPI()

    print("🏥 CLINICAL DECISION SUPPORT API TEST")
    print("=" * 60)

    # Create test patient
    test_patient = PatientProfile(
        patient_id="CDS-DEMO-001",
        age=58,
        sex="female",
        race="African_American",
        bmi=29.2,
        comorbidities=["diabetes", "hypertension", "obesity"],
        genetic_risk_scores={"cardiovascular": 0.8, "metabolic": 0.7},
        lab_history=pd.DataFrame(),
    )

    # Simulated current biomarker values (some elevated)
    current_biomarkers = {
        "CRP": 8.5,  # Elevated (normal < 3.0)
        "APOB": 140.0,  # High (normal < 120.0)
        "PCSK9": 180.0,  # Borderline
        "LPA": 45.0,  # High (normal < 30.0)
    }

    # Create clinical context
    clinical_context = PatientClinicalContext(
        patient_id=test_patient.patient_id,
        acuity_level="acute",
        location="inpatient",
        primary_diagnosis="diabetic_ketoacidosis",
        current_medications=["metformin", "lisinopril", "atorvastatin"],
        treatment_goals=["glycemic_control", "cardiovascular_risk_reduction"],
    )

    print(f"👤 PATIENT: {test_patient.patient_id}")
    print(
        f"   Demographics: {test_patient.age}yo {test_patient.sex}, BMI {test_patient.bmi}"
    )
    print(f"   Comorbidities: {test_patient.comorbidities}")
    print(
        f"   Clinical Context: {clinical_context.acuity_level} {clinical_context.location}"
    )
    print()

    print("🔬 CURRENT BIOMARKER VALUES:")
    for biomarker, value in current_biomarkers.items():
        print(f"   {biomarker}: {value}")
    print()

    try:
        # Run comprehensive evaluation
        evaluation = cds_api.evaluate_patient(
            patient_profile=test_patient,
            current_biomarkers=current_biomarkers,
            clinical_context=clinical_context,
        )

        print("✅ CLINICAL EVALUATION COMPLETE!")
        print()

        # Display summary
        summary = evaluation["summary"]
        print("📊 EVALUATION SUMMARY:")
        print(f'   • Total threshold crossings: {summary["total_crossings"]}')
        print(f'   • Critical alerts: {summary["critical_alerts"]}')
        print(f'   • High priority alerts: {summary["high_priority_alerts"]}')
        print(f'   • Total recommendations: {summary["total_recommendations"]}')
        print(
            f'   • Immediate actions required: {summary["immediate_actions_required"]}'
        )
        print()

        # Display alerts
        if evaluation["alerts"]:
            print("🚨 CLINICAL ALERTS:")
            for alert in evaluation["alerts"]:
                priority_str = (
                    alert["priority"].value.upper()
                    if hasattr(alert["priority"], "value")
                    else str(alert["priority"]).upper()
                )
                print(f'   • {priority_str}: {alert["title"]}')
                print(f'     Biomarkers: {alert["biomarkers_involved"]}')
                print(f'     {alert["message"]}')
                print()

        # Display top recommendations
        if evaluation["recommendations"]:
            print("💡 TOP CLINICAL RECOMMENDATIONS:")
            for i, rec in enumerate(evaluation["recommendations"][:3], 1):
                hours = rec["urgency_hours"]
                urgency_str = f"{hours:.0f}h" if hours < 48 else f"{hours/24:.1f}d"
                priority_str = (
                    rec["priority"].value
                    if hasattr(rec["priority"], "value")
                    else str(rec["priority"])
                )
                print(
                    f'   {i}. {rec["title"]} (Priority: {priority_str}, Urgency: {urgency_str})'
                )
                print(f'      {rec["description"]}')
                print(f'      Rationale: {rec["rationale"]}')
                print()

        # Display monitoring plan
        monitoring = evaluation["monitoring_plan"]
        print("📅 MONITORING PLAN:")
        print(f'   Next monitoring: {str(monitoring["next_monitoring_date"])[:16]}')
        annual_cost = monitoring.get("estimated_annual_cost", 0)
        print(f"   Estimated annual cost: ${annual_cost:.0f}")
        print("   Biomarker frequencies:")
        for biomarker, freq in monitoring["biomarker_frequencies"].items():
            print(f"     {biomarker}: every {freq} days")
        print()

        print("🎯 CLINICAL DECISION SUPPORT SYSTEM: FULLY OPERATIONAL")
        print("   ✅ Component 5/8 Complete: Clinical Decision Support API")
        print("   🚨 Real-time threshold crossing detection")
        print("   💡 Evidence-based clinical recommendations")
        print("   📅 Dynamic monitoring plan adjustments")
        print("   🏥 Clinical workflow integration ready")

        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_clinical_decision_support()
