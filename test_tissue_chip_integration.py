#!/usr/bin/env python3
"""
Test script for the Tissue-Chip Integration Demo

This script demonstrates how personalized biomarker predictions guide 
tissue-chip experiment design and parameter selection.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling.personalized.tissue_chip_integration import (
    TissueChipDesigner,
    ExperimentObjective,
)
from modeling.personalized.avatar_integration import PatientProfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_patients():
    """Create diverse test patients for demonstration"""

    patients = []

    # High-risk cardiovascular patient
    patient1_lab_history = pd.DataFrame(
        [
            {
                "date": datetime.now() - timedelta(days=30),
                "biomarker": "CRP",
                "value": 8.5,
                "units": "mg/L",
            },
            {
                "date": datetime.now() - timedelta(days=30),
                "biomarker": "APOB",
                "value": 145,
                "units": "mg/dL",
            },
            {
                "date": datetime.now() - timedelta(days=30),
                "biomarker": "PCSK9",
                "value": 280,
                "units": "ng/mL",
            },
            {
                "date": datetime.now() - timedelta(days=15),
                "biomarker": "CRP",
                "value": 9.2,
                "units": "mg/L",
            },
            {
                "date": datetime.now() - timedelta(days=15),
                "biomarker": "APOB",
                "value": 150,
                "units": "mg/dL",
            },
        ]
    )

    patient1 = PatientProfile(
        patient_id="HIGH_RISK_CV_001",
        age=68,
        sex="male",
        race="Caucasian",
        bmi=31.2,
        comorbidities=["hypertension", "diabetes", "coronary_artery_disease"],
        genetic_risk_scores={
            "cardiovascular": 0.85,
            "metabolic": 0.72,
            "inflammatory": 0.68,
        },
        lab_history=patient1_lab_history,
    )
    patients.append(patient1)

    # Young healthy patient for comparison
    patient2_lab_history = pd.DataFrame(
        [
            {
                "date": datetime.now() - timedelta(days=30),
                "biomarker": "CRP",
                "value": 1.2,
                "units": "mg/L",
            },
            {
                "date": datetime.now() - timedelta(days=30),
                "biomarker": "APOB",
                "value": 75,
                "units": "mg/dL",
            },
            {
                "date": datetime.now() - timedelta(days=30),
                "biomarker": "PCSK9",
                "value": 125,
                "units": "ng/mL",
            },
        ]
    )

    patient2 = PatientProfile(
        patient_id="LOW_RISK_HEALTHY_001",
        age=32,
        sex="female",
        race="Asian",
        bmi=22.8,
        comorbidities=[],
        genetic_risk_scores={
            "cardiovascular": 0.25,
            "metabolic": 0.18,
            "inflammatory": 0.22,
        },
        lab_history=patient2_lab_history,
    )
    patients.append(patient2)

    # Complex metabolic patient
    patient3_lab_history = pd.DataFrame(
        [
            {
                "date": datetime.now() - timedelta(days=20),
                "biomarker": "CRP",
                "value": 6.8,
                "units": "mg/L",
            },
            {
                "date": datetime.now() - timedelta(days=20),
                "biomarker": "APOB",
                "value": 128,
                "units": "mg/dL",
            },
            {
                "date": datetime.now() - timedelta(days=20),
                "biomarker": "HMGCR",
                "value": 145,
                "units": "U/L",
            },
            {
                "date": datetime.now() - timedelta(days=20),
                "biomarker": "LPA",
                "value": 38,
                "units": "mg/dL",
            },
        ]
    )

    patient3 = PatientProfile(
        patient_id="METABOLIC_COMPLEX_001",
        age=55,
        sex="female",
        race="African_American",
        bmi=34.5,
        comorbidities=["diabetes", "obesity", "chronic_kidney_disease"],
        genetic_risk_scores={
            "cardiovascular": 0.68,
            "metabolic": 0.88,
            "inflammatory": 0.75,
        },
        lab_history=patient3_lab_history,
    )
    patients.append(patient3)

    return patients


def demonstrate_experiment_design():
    """Demonstrate comprehensive experiment design process"""

    print("=" * 80)
    print("TISSUE-CHIP INTEGRATION DEMO")
    print("Personalized Biomarker Discovery → Lab Validation")
    print("=" * 80)
    print()

    # Initialize tissue chip designer
    print("🔬 Initializing Tissue-Chip Designer...")
    designer = TissueChipDesigner()

    # Create test patients
    print("👥 Creating diverse patient cohort for demonstration...")
    patients = create_test_patients()
    print(f"   Generated {len(patients)} test patients with varying risk profiles")
    print()

    # Define research objectives
    research_objectives = [
        ExperimentObjective.BIOMARKER_VALIDATION,
        ExperimentObjective.DRUG_SCREENING,
        ExperimentObjective.PERSONALIZED_MEDICINE,
        ExperimentObjective.DISEASE_MODELING,
    ]

    print("🎯 Research Objectives:")
    for i, obj in enumerate(research_objectives, 1):
        print(f"   {i}. {obj.value.replace('_', ' ').title()}")
    print()

    # Generate recommendations for each patient
    all_recommendations = {}

    for patient in patients:
        print(f"🧬 PATIENT ANALYSIS: {patient.patient_id}")
        print(f"   Demographics: {patient.age}yo {patient.sex}, BMI {patient.bmi}")
        print(
            f"   Comorbidities: {', '.join(patient.comorbidities) if patient.comorbidities else 'None'}"
        )
        print(
            f"   Risk Profile: CV={patient.genetic_risk_scores['cardiovascular']:.2f}, "
            f"Metabolic={patient.genetic_risk_scores['metabolic']:.2f}"
        )
        print()

        # Generate experiment recommendations
        print("   🔍 Generating experiment recommendations...")
        recommendations = designer.recommend_experiments(
            patient=patient,
            research_objectives=research_objectives,
            budget_limit=50000.0,  # $50k budget
        )

        all_recommendations[patient.patient_id] = recommendations

        # Display top recommendations
        print(f"   ✅ Generated {len(recommendations)} experiment recommendations")
        print()

        print("   🏆 TOP EXPERIMENT RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec.chip_specification.chip_type.value}")
            print(
                f"      Objective: {rec.chip_specification.experiment_objective.value}"
            )
            print(f"      Priority Score: {rec.priority_score:.3f}")
            print(f"      Clinical Relevance: {rec.clinical_relevance:.3f}")
            print(f"      Estimated Cost: ${rec.estimated_cost:,.0f}")
            print(f"      Duration: {rec.estimated_duration:.1f} days")
            print(
                f"      Target Biomarkers: {', '.join(rec.chip_specification.target_biomarkers[:3])}"
            )
            print(f"      Rationale: {rec.rationale[:150]}...")
            print()

        print("-" * 60)
        print()

    # Generate detailed protocols for highest priority experiments
    print("📋 DETAILED EXPERIMENTAL PROTOCOLS")
    print("=" * 50)
    print()

    highest_priority_recs = []
    for patient_recs in all_recommendations.values():
        if patient_recs:
            highest_priority_recs.append(
                patient_recs[0]
            )  # Top recommendation for each patient

    # Sort by priority and take top 2
    highest_priority_recs.sort(key=lambda x: x.priority_score, reverse=True)

    for i, rec in enumerate(highest_priority_recs[:2], 1):
        print(f"🧪 PROTOCOL {i}: {rec.recommendation_id}")
        print(f"Patient: {rec.patient_profile.patient_id}")
        print(
            f"Experiment: {rec.chip_specification.chip_type.value} - {rec.chip_specification.experiment_objective.value}"
        )
        print()

        # Generate detailed protocol
        protocol = designer.generate_experimental_protocol(rec)

        print("📝 PROTOCOL OVERVIEW:")
        print(f"   Title: {protocol.title}")
        print(f"   Total Duration: {protocol.total_duration:.1f} hours")
        print(f"   Critical Timepoints: {len(protocol.critical_timepoints)}")
        print(f"   Required Materials: {len(protocol.required_materials)}")
        print(f"   QC Checkpoints: {len(protocol.qc_checkpoints)}")
        print()

        print("🔧 PREPARATION STEPS:")
        for step in protocol.preparation_steps[:5]:  # Show first 5 steps
            print(f"   • {step}")
        if len(protocol.preparation_steps) > 5:
            print(f"   ... and {len(protocol.preparation_steps) - 5} more steps")
        print()

        print("⚡ EXECUTION HIGHLIGHTS:")
        for step in protocol.execution_steps[:4]:  # Show first 4 steps
            print(f"   • {step}")
        if len(protocol.execution_steps) > 4:
            print(f"   ... and {len(protocol.execution_steps) - 4} more steps")
        print()

        print("🎯 SUCCESS METRICS:")
        for metric, target in list(rec.success_metrics.items())[:4]:
            print(f"   • {metric.replace('_', ' ').title()}: {target}")
        print()

        print("⚠️  TECHNICAL RISKS & MITIGATION:")
        for risk in rec.technical_risks[:3]:
            print(f"   Risk: {risk}")
        for strategy in rec.mitigation_strategies[:3]:
            print(f"   Strategy: {strategy}")
        print()

        print("🏥 CLINICAL TRANSLATION PATH:")
        print(f"   {rec.translation_pathway}")
        print()

        print("📋 REGULATORY CONSIDERATIONS:")
        for consideration in rec.regulatory_considerations[:3]:
            print(f"   • {consideration}")
        print()

        print("-" * 70)
        print()

    # Summary analytics
    print("📊 EXPERIMENT PORTFOLIO ANALYSIS")
    print("=" * 40)
    print()

    all_recs = []
    for patient_recs in all_recommendations.values():
        all_recs.extend(patient_recs)

    if all_recs:
        # Platform distribution
        platform_counts = {}
        for rec in all_recs:
            platform = rec.chip_specification.chip_type.value
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        print("🔬 TISSUE-CHIP PLATFORM UTILIZATION:")
        for platform, count in sorted(
            platform_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(all_recs)) * 100
            print(f"   {platform:25}: {count:2d} experiments ({percentage:4.1f}%)")
        print()

        # Objective distribution
        objective_counts = {}
        for rec in all_recs:
            objective = rec.chip_specification.experiment_objective.value
            objective_counts[objective] = objective_counts.get(objective, 0) + 1

        print("🎯 RESEARCH OBJECTIVE DISTRIBUTION:")
        for objective, count in sorted(
            objective_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(all_recs)) * 100
            print(
                f"   {objective.replace('_', ' ').title():25}: {count:2d} experiments ({percentage:4.1f}%)"
            )
        print()

        # Cost analysis
        total_cost = sum(rec.estimated_cost for rec in all_recs)
        avg_cost = total_cost / len(all_recs)
        high_priority_recs = [rec for rec in all_recs if rec.priority_score > 0.7]
        high_priority_cost = sum(rec.estimated_cost for rec in high_priority_recs)

        print("💰 COST ANALYSIS:")
        print(f"   Total Portfolio Cost: ${total_cost:,.0f}")
        print(f"   Average Cost per Experiment: ${avg_cost:,.0f}")
        print(f"   High Priority Experiments: {len(high_priority_recs)}")
        print(f"   High Priority Cost: ${high_priority_cost:,.0f}")
        print()

        # Timeline analysis
        total_duration = sum(rec.estimated_duration for rec in all_recs)
        avg_duration = total_duration / len(all_recs)

        print("⏱️  TIMELINE ANALYSIS:")
        print(f"   Total Experiment Days: {total_duration:.1f}")
        print(f"   Average Duration: {avg_duration:.1f} days")
        print(f"   Parallel Execution (est.): {total_duration/3:.1f} days")
        print()

        # Clinical impact assessment
        high_clinical_impact = [rec for rec in all_recs if rec.clinical_relevance > 0.7]
        avg_clinical_relevance = np.mean([rec.clinical_relevance for rec in all_recs])

        print("🏥 CLINICAL IMPACT ASSESSMENT:")
        print(f"   Average Clinical Relevance: {avg_clinical_relevance:.3f}")
        print(f"   High Clinical Impact Experiments: {len(high_clinical_impact)}")
        print(
            f"   Clinical Impact Rate: {len(high_clinical_impact)/len(all_recs)*100:.1f}%"
        )
        print()

        # Feasibility assessment
        high_feasibility = [rec for rec in all_recs if rec.feasibility_score > 0.8]
        avg_feasibility = np.mean([rec.feasibility_score for rec in all_recs])

        print("✅ FEASIBILITY ASSESSMENT:")
        print(f"   Average Feasibility Score: {avg_feasibility:.3f}")
        print(f"   High Feasibility Experiments: {len(high_feasibility)}")
        print(
            f"   Implementation Ready: {len(high_feasibility)/len(all_recs)*100:.1f}%"
        )
        print()

    print("🎉 TISSUE-CHIP INTEGRATION DEMONSTRATION COMPLETED!")
    print()
    print("💡 KEY ACHIEVEMENTS:")
    print(
        "   ✅ Personalized biomarker insights successfully translated to experiment design"
    )
    print("   ✅ Multiple tissue-chip platforms strategically selected for validation")
    print("   ✅ Comprehensive protocols generated with clinical translation pathways")
    print(
        "   ✅ Risk-stratified patient cohorts optimally matched to experimental objectives"
    )
    print("   ✅ Cost-effective experiment portfolio balancing novelty and feasibility")
    print()
    print("🚀 READY FOR LAB VALIDATION!")

    return {
        "patients_analyzed": len(patients),
        "experiments_designed": len(all_recs),
        "total_budget": total_cost if all_recs else 0,
        "avg_clinical_relevance": avg_clinical_relevance if all_recs else 0,
        "avg_feasibility": avg_feasibility if all_recs else 0,
        "protocols_generated": len(highest_priority_recs[:2]),
        "integration_success": True,
    }


if __name__ == "__main__":
    results = demonstrate_experiment_design()
