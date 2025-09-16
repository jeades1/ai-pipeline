#!/usr/bin/env python3
"""
Test script for the Validation Framework

This script demonstrates comprehensive validation of the Personalized Biomarker Discovery Engine.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling.personalized.validation_framework import (
    ValidationFramework,
)
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive validation demonstration"""
    print("=" * 70)
    print("PERSONALIZED BIOMARKER DISCOVERY ENGINE - VALIDATION FRAMEWORK")
    print("=" * 70)
    print()

    # Initialize validation framework
    print("🔬 Initializing Validation Framework...")
    validation_framework = ValidationFramework()

    # Run comprehensive validation
    print("🧪 Running comprehensive validation with synthetic cohort...")
    print("   - Generating 500 synthetic patients")
    print("   - Simulating clinical outcomes")
    print("   - Validating all engine components")
    print()

    try:
        # Run validation with moderate cohort size for demonstration
        validation_report = validation_framework.run_comprehensive_validation(
            n_patients=500
        )

        print("✅ Validation completed successfully!")
        print()

        # Display validation summary
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)

        summary = validation_framework.generate_validation_summary(validation_report)
        print(summary)

        # Save detailed report
        report_filename = (
            f"validation_report_{validation_report.report_id.split('_')[-1]}.json"
        )
        validation_framework.save_validation_report(validation_report, report_filename)

        print(f"\n💾 Detailed validation report saved to: {report_filename}")

        # Component-specific insights
        print("\n🔍 COMPONENT INSIGHTS")
        print("=" * 30)

        for component, results in validation_report.component_results.items():
            if results:
                result = results[0]  # Take first result
                status_emoji = (
                    "✅"
                    if result.primary_score >= 0.7
                    else "⚠️" if result.primary_score >= 0.5 else "❌"
                )

                print(f"{status_emoji} {component}")
                print(f"   Score: {result.primary_score:.3f}")
                print(f"   Clinical Impact: {result.clinical_impact_score:.3f}")
                print(f"   Significance: {result.clinical_significance}")

                # Show key metrics
                if result.all_metrics:
                    print("   Key Metrics:")
                    for metric, value in result.all_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"     • {metric}: {value:.3f}")
                        else:
                            print(f"     • {metric}: {value}")
                print()
            else:
                print(f"❌ {component}: FAILED VALIDATION")
                print()

        # Clinical readiness assessment
        print("🏥 CLINICAL READINESS ASSESSMENT")
        print("=" * 35)

        readiness_score = validation_report.clinical_readiness_score
        if readiness_score >= 0.8:
            readiness_status = "🟢 READY"
            readiness_desc = "System is ready for clinical pilot studies"
        elif readiness_score >= 0.6:
            readiness_status = "🟡 APPROACHING"
            readiness_desc = "Minor improvements needed before clinical deployment"
        else:
            readiness_status = "🔴 DEVELOPING"
            readiness_desc = "Significant development required for clinical use"

        print(f"Status: {readiness_status}")
        print(f"Score: {readiness_score:.2f}/1.00")
        print(f"Assessment: {readiness_desc}")
        print()

        # Regulatory readiness assessment
        print("📋 REGULATORY READINESS ASSESSMENT")
        print("=" * 36)

        regulatory_score = validation_report.regulatory_readiness_score
        if regulatory_score >= 0.8:
            regulatory_status = "🟢 READY"
            regulatory_desc = "Ready for regulatory submission pathway"
        elif regulatory_score >= 0.6:
            regulatory_status = "🟡 APPROACHING"
            regulatory_desc = "Additional validation needed for regulatory approval"
        else:
            regulatory_status = "🔴 DEVELOPING"
            regulatory_desc = "Substantial validation required for regulatory approval"

        print(f"Status: {regulatory_status}")
        print(f"Score: {regulatory_score:.2f}/1.00")
        print(f"Assessment: {regulatory_desc}")
        print()

        # Next steps
        print("🎯 NEXT STEPS")
        print("=" * 15)

        if validation_report.improvement_recommendations:
            print("Priority improvements:")
            for i, rec in enumerate(
                validation_report.improvement_recommendations[:5], 1
            ):
                print(f"{i}. {rec}")

        print()
        print("🔬 Validation framework demonstration completed!")

        # Return summary metrics for any automated testing
        return {
            "overall_performance": validation_report.overall_performance_score,
            "clinical_readiness": validation_report.clinical_readiness_score,
            "regulatory_readiness": validation_report.regulatory_readiness_score,
            "components_validated": len(
                [r for r in validation_report.component_results.values() if r]
            ),
            "validation_success": True,
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"❌ Validation failed: {e}")
        return {"validation_success": False, "error": str(e)}


if __name__ == "__main__":
    results = main()
