# Advanced Tissue Architecture Validation Metrics

"""
Comprehensive validation metrics for multicellular architecture performance,
vascularization efficiency, and kinetic analysis accuracy.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


class ValidationLevel(Enum):
    """Validation confidence levels"""

    EXCELLENT = "excellent"  # >90% confidence
    GOOD = "good"  # 80-90% confidence
    MODERATE = "moderate"  # 60-80% confidence
    POOR = "poor"  # <60% confidence


class ArchitectureMetric(Enum):
    """Multicellular architecture validation metrics"""

    CELL_TYPE_COMPOSITION = "cell_type_composition"
    BARRIER_INTEGRITY = "barrier_integrity"
    TISSUE_ORGANIZATION = "tissue_organization"
    CELL_CELL_SIGNALING = "cell_cell_signaling"
    TUBULAR_GEOMETRY = "tubular_geometry"
    PHYSIOLOGICAL_FUNCTION = "physiological_function"


class VascularizationMetric(Enum):
    """PDO vascularization validation metrics"""

    NETWORK_DENSITY = "network_density"
    MOLECULAR_DELIVERY = "molecular_delivery"
    PERFUSION_EFFICIENCY = "perfusion_efficiency"
    ENDOTHELIAL_INTEGRATION = "endothelial_integration"
    LARGE_TISSUE_PENETRATION = "large_tissue_penetration"
    CULTURE_LONGEVITY = "culture_longevity"


class KineticMetric(Enum):
    """Kinetic analysis validation metrics"""

    TEMPORAL_RESOLUTION = "temporal_resolution"
    SECRETION_KINETICS = "secretion_kinetics"
    CLEARANCE_ANALYSIS = "clearance_analysis"
    PHARMACOKINETIC_MODELING = "pharmacokinetic_modeling"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    RECIRCULATION_EFFICIENCY = "recirculation_efficiency"


@dataclass
class ValidationResult:
    """Validation result for a specific metric"""

    metric_name: str
    measured_value: float
    reference_range: Tuple[float, float]
    validation_level: ValidationLevel
    confidence_score: float
    notes: Optional[str] = None

    def is_within_range(self) -> bool:
        """Check if measured value is within reference range"""
        return self.reference_range[0] <= self.measured_value <= self.reference_range[1]

    def calculate_deviation(self) -> float:
        """Calculate relative deviation from reference range"""
        if self.is_within_range():
            return 0.0

        range_center = (self.reference_range[0] + self.reference_range[1]) / 2
        return abs(self.measured_value - range_center) / range_center


class MulticellularArchitectureValidator:
    """Validator for multicellular architecture performance"""

    # Reference ranges for multicellular architecture metrics
    REFERENCE_RANGES = {
        ArchitectureMetric.CELL_TYPE_COMPOSITION: {
            "epithelial_percent": (55.0, 65.0),
            "endothelial_percent": (15.0, 25.0),
            "fibroblast_percent": (10.0, 20.0),
            "immune_percent": (2.0, 8.0),
        },
        ArchitectureMetric.BARRIER_INTEGRITY: {
            "TEER_ohm_cm2": (1000.0, 3000.0),
            "permeability_cm_s": (1e-7, 1e-5),
            "tight_junction_score": (0.8, 1.0),
        },
        ArchitectureMetric.TISSUE_ORGANIZATION: {
            "3D_organization_score": (0.8, 1.0),
            "cell_polarity_index": (0.7, 1.0),
            "tubular_structure_integrity": (0.85, 1.0),
        },
        ArchitectureMetric.CELL_CELL_SIGNALING: {
            "paracrine_signaling_index": (0.7, 1.0),
            "gap_junction_connectivity": (0.6, 1.0),
            "cytokine_gradient_formation": (0.8, 1.0),
        },
        ArchitectureMetric.TUBULAR_GEOMETRY: {
            "diameter_um": (50.0, 200.0),
            "wall_thickness_um": (10.0, 50.0),
            "aspect_ratio": (2.0, 10.0),
        },
        ArchitectureMetric.PHYSIOLOGICAL_FUNCTION: {
            "metabolic_activity_fold": (1.5, 3.0),
            "transport_function_score": (0.8, 1.0),
            "stress_response_index": (0.7, 1.0),
        },
    }

    def validate_architecture(
        self, architecture_data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """Validate multicellular architecture metrics"""
        results = {}

        for metric in ArchitectureMetric:
            metric_results = self._validate_metric(metric, architecture_data)
            results.update(metric_results)

        return results

    def _validate_metric(
        self, metric: ArchitectureMetric, data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """Validate a specific architecture metric"""
        results = {}
        reference_ranges = self.REFERENCE_RANGES[metric]

        for param_name, ref_range in reference_ranges.items():
            measured_value = data.get(param_name, 0.0)

            # Calculate validation level
            if ref_range[0] <= measured_value <= ref_range[1]:
                validation_level = ValidationLevel.EXCELLENT
                confidence = 0.95
            else:
                deviation = abs(measured_value - (ref_range[0] + ref_range[1]) / 2) / (
                    (ref_range[1] - ref_range[0]) / 2
                )
                if deviation <= 0.2:
                    validation_level = ValidationLevel.GOOD
                    confidence = 0.85
                elif deviation <= 0.5:
                    validation_level = ValidationLevel.MODERATE
                    confidence = 0.70
                else:
                    validation_level = ValidationLevel.POOR
                    confidence = 0.50

            results[f"{metric.value}_{param_name}"] = ValidationResult(
                metric_name=f"{metric.value}_{param_name}",
                measured_value=measured_value,
                reference_range=ref_range,
                validation_level=validation_level,
                confidence_score=confidence,
            )

        return results


class VascularizationValidator:
    """Validator for PDO vascularization system performance"""

    REFERENCE_RANGES = {
        VascularizationMetric.NETWORK_DENSITY: {
            "vascular_coverage_percent": (70.0, 95.0),
            "network_connectivity_index": (0.8, 1.0),
            "branch_point_density": (10.0, 50.0),  # per mm²
        },
        VascularizationMetric.MOLECULAR_DELIVERY: {
            "delivery_enhancement_fold": (10.0, 100.0),
            "penetration_depth_um": (200.0, 1000.0),
            "distribution_uniformity": (0.8, 1.0),
        },
        VascularizationMetric.PERFUSION_EFFICIENCY: {
            "flow_distribution_cv": (0.1, 0.3),  # Lower is better
            "pressure_uniformity": (0.8, 1.0),
            "shear_stress_physiological": (0.1, 10.0),  # dyn/cm²
        },
        VascularizationMetric.ENDOTHELIAL_INTEGRATION: {
            "CD31_expression_score": (0.8, 1.0),
            "VE_cadherin_score": (0.7, 1.0),
            "barrier_function_score": (0.8, 1.0),
        },
        VascularizationMetric.LARGE_TISSUE_PENETRATION: {
            "penetration_efficiency": (0.7, 1.0),
            "molecular_weight_cutoff_kDa": (50.0, 150.0),
            "delivery_time_hours": (0.5, 4.0),
        },
        VascularizationMetric.CULTURE_LONGEVITY: {
            "viability_days": (14.0, 28.0),
            "metabolic_activity_retention": (0.8, 1.0),
            "structural_integrity_retention": (0.9, 1.0),
        },
    }

    def validate_vascularization(
        self, vascularization_data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """Validate PDO vascularization metrics"""
        results = {}

        for metric in VascularizationMetric:
            metric_results = self._validate_metric(metric, vascularization_data)
            results.update(metric_results)

        return results

    def _validate_metric(
        self, metric: VascularizationMetric, data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """Validate a specific vascularization metric"""
        results = {}
        reference_ranges = self.REFERENCE_RANGES[metric]

        for param_name, ref_range in reference_ranges.items():
            measured_value = data.get(param_name, 0.0)

            # Special handling for CV (lower is better)
            if "cv" in param_name.lower():
                in_range = measured_value <= ref_range[1]
                deviation = max(0, measured_value - ref_range[1]) / ref_range[1]
            else:
                in_range = ref_range[0] <= measured_value <= ref_range[1]
                if in_range:
                    deviation = 0
                else:
                    range_center = (ref_range[0] + ref_range[1]) / 2
                    deviation = abs(measured_value - range_center) / range_center

            # Calculate validation level
            if in_range:
                validation_level = ValidationLevel.EXCELLENT
                confidence = 0.95
            elif deviation <= 0.2:
                validation_level = ValidationLevel.GOOD
                confidence = 0.85
            elif deviation <= 0.5:
                validation_level = ValidationLevel.MODERATE
                confidence = 0.70
            else:
                validation_level = ValidationLevel.POOR
                confidence = 0.50

            results[f"{metric.value}_{param_name}"] = ValidationResult(
                metric_name=f"{metric.value}_{param_name}",
                measured_value=measured_value,
                reference_range=ref_range,
                validation_level=validation_level,
                confidence_score=confidence,
            )

        return results


class KineticAnalysisValidator:
    """Validator for real-time kinetic analysis accuracy"""

    REFERENCE_RANGES = {
        KineticMetric.TEMPORAL_RESOLUTION: {
            "sampling_frequency_minutes": (0.5, 2.0),  # Sub-minute to 2-minute
            "data_density_points_hour": (30.0, 120.0),
            "temporal_accuracy_percent": (95.0, 100.0),
        },
        KineticMetric.SECRETION_KINETICS: {
            "secretion_rate_accuracy": (0.9, 1.0),
            "baseline_stability_cv": (0.05, 0.15),
            "dynamic_range_logs": (3.0, 5.0),
        },
        KineticMetric.CLEARANCE_ANALYSIS: {
            "elimination_rate_accuracy": (0.85, 1.0),
            "half_life_precision_percent": (90.0, 100.0),
            "clearance_model_r2": (0.9, 1.0),
        },
        KineticMetric.PHARMACOKINETIC_MODELING: {
            "pk_parameter_accuracy": (0.9, 1.0),
            "model_fit_quality": (0.85, 1.0),
            "prediction_accuracy": (0.8, 1.0),
        },
        KineticMetric.CONTINUOUS_MONITORING: {
            "monitoring_duration_hours": (24.0, 168.0),  # 1-7 days
            "data_completeness_percent": (95.0, 100.0),
            "system_uptime_percent": (98.0, 100.0),
        },
        KineticMetric.RECIRCULATION_EFFICIENCY: {
            "flow_consistency_cv": (0.05, 0.15),
            "mixing_efficiency": (0.9, 1.0),
            "sample_recovery_percent": (95.0, 100.0),
        },
    }

    def validate_kinetics(
        self, kinetic_data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """Validate kinetic analysis metrics"""
        results = {}

        for metric in KineticMetric:
            metric_results = self._validate_metric(metric, kinetic_data)
            results.update(metric_results)

        return results

    def _validate_metric(
        self, metric: KineticMetric, data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """Validate a specific kinetic metric"""
        results = {}
        reference_ranges = self.REFERENCE_RANGES[metric]

        for param_name, ref_range in reference_ranges.items():
            measured_value = data.get(param_name, 0.0)

            # Calculate validation level
            if ref_range[0] <= measured_value <= ref_range[1]:
                validation_level = ValidationLevel.EXCELLENT
                confidence = 0.95
            else:
                range_center = (ref_range[0] + ref_range[1]) / 2
                deviation = abs(measured_value - range_center) / range_center

                if deviation <= 0.15:
                    validation_level = ValidationLevel.GOOD
                    confidence = 0.85
                elif deviation <= 0.35:
                    validation_level = ValidationLevel.MODERATE
                    confidence = 0.70
                else:
                    validation_level = ValidationLevel.POOR
                    confidence = 0.50

            results[f"{metric.value}_{param_name}"] = ValidationResult(
                metric_name=f"{metric.value}_{param_name}",
                measured_value=measured_value,
                reference_range=ref_range,
                validation_level=validation_level,
                confidence_score=confidence,
            )

        return results


class ComprehensiveValidator:
    """Comprehensive validation system for all tissue-chip capabilities"""

    def __init__(self):
        self.architecture_validator = MulticellularArchitectureValidator()
        self.vascularization_validator = VascularizationValidator()
        self.kinetic_validator = KineticAnalysisValidator()

    def validate_complete_system(
        self,
        architecture_data: Dict[str, Any],
        vascularization_data: Dict[str, Any],
        kinetic_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform comprehensive validation of entire tissue-chip system"""

        # Individual component validations
        architecture_results = self.architecture_validator.validate_architecture(
            architecture_data
        )
        vascularization_results = (
            self.vascularization_validator.validate_vascularization(
                vascularization_data
            )
        )
        kinetic_results = self.kinetic_validator.validate_kinetics(kinetic_data)

        # Combine all results
        all_results = {
            **architecture_results,
            **vascularization_results,
            **kinetic_results,
        }

        # Calculate system-level metrics
        system_metrics = self._calculate_system_metrics(all_results)

        # Generate comprehensive report
        validation_report = {
            "timestamp": str(datetime.now()),
            "system_metrics": system_metrics,
            "architecture_validation": self._summarize_results(architecture_results),
            "vascularization_validation": self._summarize_results(
                vascularization_results
            ),
            "kinetic_validation": self._summarize_results(kinetic_results),
            "detailed_results": {
                name: result.__dict__ for name, result in all_results.items()
            },
            "recommendations": self._generate_recommendations(all_results),
            "clinical_translation_readiness": self._assess_translation_readiness(
                system_metrics
            ),
        }

        return validation_report

    def _calculate_system_metrics(
        self, all_results: Dict[str, ValidationResult]
    ) -> Dict[str, float]:
        """Calculate overall system performance metrics"""

        # Overall confidence score
        confidence_scores = [result.confidence_score for result in all_results.values()]
        overall_confidence = np.mean(confidence_scores)

        # Validation level distribution
        level_counts = {}
        for result in all_results.values():
            level = result.validation_level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        total_metrics = len(all_results)
        level_percentages = {
            level: count / total_metrics for level, count in level_counts.items()
        }

        # Component-specific scores
        architecture_scores = [
            r.confidence_score
            for name, r in all_results.items()
            if "architecture" in name
        ]
        vascularization_scores = [
            r.confidence_score
            for name, r in all_results.items()
            if "vascularization" in name or "network" in name or "delivery" in name
        ]
        kinetic_scores = [
            r.confidence_score
            for name, r in all_results.items()
            if "kinetic" in name or "temporal" in name or "recirculation" in name
        ]

        return {
            "overall_confidence_score": overall_confidence,
            "excellent_percentage": level_percentages.get("excellent", 0.0),
            "good_percentage": level_percentages.get("good", 0.0),
            "moderate_percentage": level_percentages.get("moderate", 0.0),
            "poor_percentage": level_percentages.get("poor", 0.0),
            "architecture_score": (
                np.mean(architecture_scores) if architecture_scores else 0.0
            ),
            "vascularization_score": (
                np.mean(vascularization_scores) if vascularization_scores else 0.0
            ),
            "kinetic_analysis_score": (
                np.mean(kinetic_scores) if kinetic_scores else 0.0
            ),
            "system_readiness_score": overall_confidence,
        }

    def _summarize_results(
        self, results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """Summarize validation results for a component"""
        excellent_count = sum(
            1
            for r in results.values()
            if r.validation_level == ValidationLevel.EXCELLENT
        )
        good_count = sum(
            1 for r in results.values() if r.validation_level == ValidationLevel.GOOD
        )
        moderate_count = sum(
            1
            for r in results.values()
            if r.validation_level == ValidationLevel.MODERATE
        )
        poor_count = sum(
            1 for r in results.values() if r.validation_level == ValidationLevel.POOR
        )

        total = len(results)
        avg_confidence = np.mean([r.confidence_score for r in results.values()])

        return {
            "total_metrics": total,
            "excellent_count": excellent_count,
            "good_count": good_count,
            "moderate_count": moderate_count,
            "poor_count": poor_count,
            "average_confidence": avg_confidence,
            "pass_rate": (excellent_count + good_count) / total if total > 0 else 0.0,
        }

    def _generate_recommendations(
        self, all_results: Dict[str, ValidationResult]
    ) -> List[str]:
        """Generate improvement recommendations based on validation results"""
        recommendations = []

        # Find poorly performing metrics
        poor_metrics = [
            name
            for name, result in all_results.items()
            if result.validation_level
            in [ValidationLevel.POOR, ValidationLevel.MODERATE]
        ]

        for metric_name in poor_metrics[:5]:  # Top 5 recommendations
            if "barrier" in metric_name:
                recommendations.append(
                    "Optimize culture conditions to improve barrier integrity (TEER)"
                )
            elif "delivery" in metric_name:
                recommendations.append(
                    "Enhance vascularization density for improved molecular delivery"
                )
            elif "temporal" in metric_name:
                recommendations.append(
                    "Increase sampling frequency for better kinetic resolution"
                )
            elif "cell_type" in metric_name:
                recommendations.append(
                    "Adjust cell seeding ratios for optimal multicellular composition"
                )
            elif "perfusion" in metric_name:
                recommendations.append("Optimize flow rates and perfusion parameters")
            else:
                recommendations.append(
                    f"Investigate and optimize {metric_name} performance"
                )

        return recommendations

    def _assess_translation_readiness(
        self, system_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess clinical translation readiness"""
        overall_score = system_metrics["overall_confidence_score"]
        excellent_rate = system_metrics["excellent_percentage"]

        if overall_score >= 0.9 and excellent_rate >= 0.8:
            readiness_level = "High"
            translation_probability = 0.85
        elif overall_score >= 0.8 and excellent_rate >= 0.6:
            readiness_level = "Moderate-High"
            translation_probability = 0.70
        elif overall_score >= 0.7 and excellent_rate >= 0.4:
            readiness_level = "Moderate"
            translation_probability = 0.55
        else:
            readiness_level = "Low"
            translation_probability = 0.30

        return {
            "readiness_level": readiness_level,
            "translation_probability": translation_probability,
            "confidence_score": overall_score,
            "excellence_rate": excellent_rate,
            "estimated_success_multiplier": min(translation_probability * 3, 3.0),
        }

    def save_validation_report(self, report: Dict[str, Any], output_path: Path):
        """Save comprehensive validation report"""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)


# Usage example and validation metrics summary
TISSUE_CHIP_VALIDATION_SUMMARY = {
    "multicellular_architecture": {
        "metrics_count": 18,
        "key_parameters": [
            "cell_composition",
            "barrier_integrity",
            "tissue_organization",
            "signaling_networks",
        ],
        "excellence_threshold": 0.9,
        "clinical_relevance": "authentic_tissue_responses",
    },
    "pdo_vascularization": {
        "metrics_count": 18,
        "key_parameters": [
            "network_density",
            "molecular_delivery",
            "perfusion_efficiency",
            "culture_longevity",
        ],
        "excellence_threshold": 0.9,
        "clinical_relevance": "enhanced_drug_delivery_accuracy",
    },
    "kinetic_analysis": {
        "metrics_count": 18,
        "key_parameters": [
            "temporal_resolution",
            "secretion_kinetics",
            "clearance_analysis",
            "continuous_monitoring",
        ],
        "excellence_threshold": 0.9,
        "clinical_relevance": "real_time_biomarker_characterization",
    },
    "overall_system": {
        "total_metrics": 54,
        "validation_confidence_target": 0.85,
        "clinical_translation_enhancement": "3x_improved_success_rate",
        "competitive_advantage": "proven_multicellular_vascularized_kinetic_platform",
    },
}
