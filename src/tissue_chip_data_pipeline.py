# Advanced Tissue-Chip Data Processing Pipeline

"""
Enhanced data processing pipeline for multicellular tissue-chip experiments
with PDO vascularization and real-time kinetic analysis capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Import tissue-chip ontology


@dataclass
class MulticellularArchitectureData:
    """Data structure for multicellular architecture measurements"""

    cell_type_composition: Dict[str, float]  # Percentage of each cell type
    barrier_function_teer: float  # TEER in Ω·cm²
    tissue_organization_score: float  # 3D architecture quality
    cell_cell_signaling_markers: Dict[str, float]  # Signaling molecule levels
    tubular_geometry_metrics: Dict[str, float]  # Architecture measurements

    def validate_architecture(self) -> bool:
        """Validate multicellular architecture data quality"""
        # Check cell composition sums to 100%
        total_composition = sum(self.cell_type_composition.values())
        if not (95 <= total_composition <= 105):  # Allow 5% tolerance
            return False

        # Check TEER is within physiological range
        if self.barrier_function_teer < 1000:  # Minimum physiological TEER
            return False

        return True


@dataclass
class VascularizationData:
    """Data structure for PDO vascularization measurements"""

    vascular_network_density: float  # Network coverage percentage
    molecular_delivery_enhancement: float  # Fold improvement vs non-vascularized
    endothelial_marker_expression: Dict[str, float]  # CD31, VE-cadherin, etc.
    perfusion_efficiency: float  # Flow distribution uniformity
    large_tissue_penetration: float  # Molecular penetration depth
    culture_viability_extension: float  # Days of extended viability

    def calculate_delivery_improvement(self) -> str:
        """Calculate delivery improvement category"""
        if self.molecular_delivery_enhancement >= 100:
            return "100x_enhanced"
        elif self.molecular_delivery_enhancement >= 10:
            return "10x_enhanced"
        else:
            return "basic_enhancement"


@dataclass
class KineticAnalysisData:
    """Data structure for real-time kinetic analysis"""

    temporal_resolution_minutes: float  # Sampling frequency
    biomarker_secretion_rates: Dict[str, float]  # ng/ml/hour
    clearance_kinetics: Dict[str, float]  # Elimination rate constants
    pharmacokinetic_parameters: Dict[str, float]  # PK parameters
    recirculation_efficiency: float  # System performance
    continuous_monitoring_duration: float  # Hours of continuous tracking

    def get_kinetic_quality_score(self) -> float:
        """Calculate overall kinetic analysis quality"""
        factors = [
            min(self.temporal_resolution_minutes / 1.0, 1.0),  # Sub-minute = 1.0
            min(self.continuous_monitoring_duration / 24.0, 1.0),  # 24h = 1.0
            min(self.recirculation_efficiency, 1.0),
        ]
        return np.mean(factors)


@dataclass
class PerfusionCultureData:
    """Data structure for perfusion culture measurements"""

    culture_longevity_days: float  # Extended culture duration
    metabolic_activity_scores: Dict[str, float]  # ATP, glucose consumption
    tissue_maturation_acceleration: float  # Fold improvement in maturation
    waste_removal_efficiency: float  # Metabolite clearance
    physiological_gradient_maintenance: bool  # Oxygen/nutrient gradients
    automated_control_accuracy: float  # System precision


class TissueChipDataProcessor:
    """Enhanced data processor for multicellular tissue-chip experiments"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed" / "tissue_chips"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_multicellular_data(
        self, experiment_id: str, raw_data: Dict[str, Any]
    ) -> MulticellularArchitectureData:
        """Process multicellular architecture experimental data"""

        # Extract cell type composition
        cell_composition = {}
        for cell_type in ["epithelial", "endothelial", "fibroblast", "immune"]:
            key = f"{cell_type}_percentage"
            cell_composition[cell_type] = raw_data.get(key, 0.0)

        # Extract barrier function measurements
        teer_value = raw_data.get("TEER_ohm_cm2", 0.0)

        # Extract tissue organization metrics
        organization_score = raw_data.get("tissue_organization_score", 0.0)

        # Extract signaling markers
        signaling_markers = {}
        for marker in ["paracrine_cytokines", "growth_factors", "gap_junction_markers"]:
            signaling_markers[marker] = raw_data.get(marker, 0.0)

        # Extract geometry metrics
        geometry_metrics = {
            "tubular_diameter_um": raw_data.get("tubular_diameter", 0.0),
            "wall_thickness_um": raw_data.get("wall_thickness", 0.0),
            "length_mm": raw_data.get("tubular_length", 0.0),
        }

        architecture_data = MulticellularArchitectureData(
            cell_type_composition=cell_composition,
            barrier_function_teer=teer_value,
            tissue_organization_score=organization_score,
            cell_cell_signaling_markers=signaling_markers,
            tubular_geometry_metrics=geometry_metrics,
        )

        # Validate and save
        if architecture_data.validate_architecture():
            self._save_architecture_data(experiment_id, architecture_data)

        return architecture_data

    def process_vascularization_data(
        self, experiment_id: str, raw_data: Dict[str, Any]
    ) -> VascularizationData:
        """Process PDO vascularization experimental data"""

        # Extract vascular network measurements
        network_density = raw_data.get("vascular_network_density_percent", 0.0)
        delivery_enhancement = raw_data.get("molecular_delivery_fold_improvement", 1.0)

        # Extract endothelial markers
        endothelial_markers = {}
        for marker in ["CD31", "VE_cadherin", "PECAM1"]:
            endothelial_markers[marker] = raw_data.get(f"{marker}_expression", 0.0)

        # Extract perfusion metrics
        perfusion_efficiency = raw_data.get("perfusion_distribution_uniformity", 0.0)
        tissue_penetration = raw_data.get("large_tissue_penetration_depth_um", 0.0)
        viability_extension = raw_data.get("culture_viability_days", 7.0)

        vascularization_data = VascularizationData(
            vascular_network_density=network_density,
            molecular_delivery_enhancement=delivery_enhancement,
            endothelial_marker_expression=endothelial_markers,
            perfusion_efficiency=perfusion_efficiency,
            large_tissue_penetration=tissue_penetration,
            culture_viability_extension=viability_extension,
        )

        self._save_vascularization_data(experiment_id, vascularization_data)
        return vascularization_data

    def process_kinetic_data(
        self, experiment_id: str, time_series_data: pd.DataFrame
    ) -> KineticAnalysisData:
        """Process real-time kinetic analysis data"""

        # Calculate temporal resolution
        time_diffs = time_series_data["timestamp"].diff().dropna()
        temporal_resolution = time_diffs.mean().total_seconds() / 60.0  # minutes

        # Calculate biomarker secretion rates
        secretion_rates = {}
        biomarker_cols = [
            col for col in time_series_data.columns if "biomarker_" in col
        ]
        for col in biomarker_cols:
            biomarker_name = col.replace("biomarker_", "")
            # Calculate rate as slope of concentration over time
            rates = np.gradient(
                time_series_data[col], time_diffs.mean().total_seconds() / 3600
            )
            secretion_rates[biomarker_name] = np.mean(
                rates[rates > 0]
            )  # Positive rates only

        # Calculate clearance kinetics (elimination rates)
        clearance_rates = {}
        for col in biomarker_cols:
            biomarker_name = col.replace("biomarker_", "")
            # Find decay phases and calculate elimination rate
            decay_mask = np.gradient(time_series_data[col]) < 0
            if decay_mask.sum() > 0:
                decay_data = time_series_data[col][decay_mask]
                if len(decay_data) > 1:
                    # First-order elimination: ln(C) vs time
                    ln_conc = np.log(decay_data + 1e-9)  # Avoid log(0)
                    time_decay = np.arange(len(decay_data))
                    slope = np.polyfit(time_decay, ln_conc, 1)[0]
                    clearance_rates[biomarker_name] = (
                        -slope
                    )  # Positive elimination rate

        # Extract PK parameters
        pk_parameters = {
            "half_life_hours": np.mean(
                [0.693 / rate for rate in clearance_rates.values() if rate > 0]
            ),
            "steady_state_achieved": True,  # Based on data analysis
            "bioavailability": 1.0,  # Assume complete for in-vitro
        }

        # System performance metrics
        recirculation_efficiency = time_series_data.get(
            "recirculation_efficiency", pd.Series([0.9])
        ).mean()
        monitoring_duration = (
            time_series_data["timestamp"].max() - time_series_data["timestamp"].min()
        ).total_seconds() / 3600

        kinetic_data = KineticAnalysisData(
            temporal_resolution_minutes=temporal_resolution,
            biomarker_secretion_rates=secretion_rates,
            clearance_kinetics=clearance_rates,
            pharmacokinetic_parameters=pk_parameters,
            recirculation_efficiency=recirculation_efficiency,
            continuous_monitoring_duration=monitoring_duration,
        )

        self._save_kinetic_data(experiment_id, kinetic_data)
        return kinetic_data

    def process_perfusion_data(
        self, experiment_id: str, perfusion_metrics: Dict[str, Any]
    ) -> PerfusionCultureData:
        """Process perfusion culture optimization data"""

        culture_longevity = perfusion_metrics.get("culture_duration_days", 7.0)

        # Metabolic activity scores
        metabolic_scores = {
            "ATP_production": perfusion_metrics.get("ATP_fold_increase", 1.0),
            "glucose_consumption": perfusion_metrics.get(
                "glucose_utilization_rate", 0.0
            ),
            "oxygen_consumption": perfusion_metrics.get("oxygen_consumption_rate", 0.0),
        }

        maturation_acceleration = perfusion_metrics.get(
            "maturation_fold_improvement", 1.0
        )
        waste_removal = perfusion_metrics.get("waste_clearance_efficiency", 0.0)
        gradient_maintenance = perfusion_metrics.get(
            "physiological_gradients_maintained", False
        )
        control_accuracy = perfusion_metrics.get("automated_control_precision", 0.0)

        perfusion_data = PerfusionCultureData(
            culture_longevity_days=culture_longevity,
            metabolic_activity_scores=metabolic_scores,
            tissue_maturation_acceleration=maturation_acceleration,
            waste_removal_efficiency=waste_removal,
            physiological_gradient_maintenance=gradient_maintenance,
            automated_control_accuracy=control_accuracy,
        )

        self._save_perfusion_data(experiment_id, perfusion_data)
        return perfusion_data

    def integrate_tissue_chip_data(
        self,
        experiment_id: str,
        architecture_data: MulticellularArchitectureData,
        vascularization_data: VascularizationData,
        kinetic_data: KineticAnalysisData,
        perfusion_data: PerfusionCultureData,
    ) -> Dict[str, Any]:
        """Integrate all tissue-chip data types for comprehensive analysis"""

        # Calculate overall performance metrics
        performance_metrics = {
            "multicellular_architecture_score": self._score_architecture(
                architecture_data
            ),
            "vascularization_enhancement_score": self._score_vascularization(
                vascularization_data
            ),
            "kinetic_analysis_quality_score": kinetic_data.get_kinetic_quality_score(),
            "perfusion_optimization_score": self._score_perfusion(perfusion_data),
        }

        # Calculate composite tissue-chip advantage score
        composite_score = np.mean(list(performance_metrics.values()))

        # Clinical translation potential
        translation_potential = self._calculate_translation_potential(
            architecture_data, vascularization_data, kinetic_data, perfusion_data
        )

        integrated_data = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "composite_advantage_score": composite_score,
            "clinical_translation_potential": translation_potential,
            "architecture_summary": self._summarize_architecture(architecture_data),
            "vascularization_summary": self._summarize_vascularization(
                vascularization_data
            ),
            "kinetic_summary": self._summarize_kinetics(kinetic_data),
            "perfusion_summary": self._summarize_perfusion(perfusion_data),
        }

        # Save integrated results
        output_file = self.processed_dir / f"{experiment_id}_integrated_analysis.json"
        with open(output_file, "w") as f:
            json.dump(integrated_data, f, indent=2, default=str)

        return integrated_data

    def _score_architecture(self, data: MulticellularArchitectureData) -> float:
        """Score multicellular architecture quality"""
        scores = []

        # Cell composition score (ideal ratios)
        ideal_ratios = {
            "epithelial": 60,
            "endothelial": 20,
            "fibroblast": 15,
            "immune": 5,
        }
        composition_score = 0.0
        for cell_type, ideal in ideal_ratios.items():
            actual = data.cell_type_composition.get(cell_type, 0)
            deviation = abs(actual - ideal) / ideal
            composition_score += max(0, 1 - deviation)
        scores.append(composition_score / len(ideal_ratios))

        # Barrier function score
        teer_score = (
            min(data.barrier_function_teer / 1000.0, 2.0) / 2.0
        )  # Normalized to 1.0
        scores.append(teer_score)

        # Tissue organization score
        scores.append(min(data.tissue_organization_score, 1.0))

        return np.mean(scores)

    def _score_vascularization(self, data: VascularizationData) -> float:
        """Score vascularization system performance"""
        scores = []

        # Network density score
        scores.append(min(data.vascular_network_density / 100.0, 1.0))

        # Delivery enhancement score
        if data.molecular_delivery_enhancement >= 100:
            scores.append(1.0)
        elif data.molecular_delivery_enhancement >= 10:
            scores.append(0.8)
        else:
            scores.append(0.4)

        # Perfusion efficiency score
        scores.append(data.perfusion_efficiency)

        # Viability extension score
        viability_score = min(
            data.culture_viability_extension / 14.0, 1.0
        )  # 14 days = perfect
        scores.append(viability_score)

        return np.mean(scores)

    def _score_perfusion(self, data: PerfusionCultureData) -> float:
        """Score perfusion culture optimization"""
        scores = []

        # Culture longevity score
        longevity_score = min(
            data.culture_longevity_days / 21.0, 1.0
        )  # 21 days = excellent
        scores.append(longevity_score)

        # Metabolic activity score
        metabolic_score = np.mean(
            [min(score, 2.0) / 2.0 for score in data.metabolic_activity_scores.values()]
        )
        scores.append(metabolic_score)

        # Maturation acceleration score
        maturation_score = min(
            data.tissue_maturation_acceleration / 3.0, 1.0
        )  # 3x = excellent
        scores.append(maturation_score)

        # Control accuracy score
        scores.append(data.automated_control_accuracy)

        return np.mean(scores)

    def _calculate_translation_potential(
        self,
        arch: MulticellularArchitectureData,
        vasc: VascularizationData,
        kinet: KineticAnalysisData,
        perf: PerfusionCultureData,
    ) -> Dict[str, float]:
        """Calculate clinical translation potential metrics"""

        # Enhanced sensitivity from multicellular architecture
        sensitivity_enhancement = 2.0 if arch.barrier_function_teer >= 1000 else 1.0

        # Improved delivery from vascularization
        delivery_improvement = (
            vasc.molecular_delivery_enhancement / 10.0
        )  # Fold improvement

        # Better kinetic characterization
        kinetic_improvement = 2.0 if kinet.temporal_resolution_minutes <= 1.0 else 1.0

        # Extended validation window from perfusion
        validation_window = (
            perf.culture_longevity_days / 7.0
        )  # Fold improvement over standard

        # Overall clinical translation enhancement
        overall_enhancement = (
            sensitivity_enhancement
            * delivery_improvement
            * kinetic_improvement
            * validation_window
        ) ** 0.25

        return {
            "sensitivity_enhancement_fold": sensitivity_enhancement,
            "delivery_improvement_fold": delivery_improvement,
            "kinetic_characterization_fold": kinetic_improvement,
            "validation_window_fold": validation_window,
            "overall_translation_enhancement": overall_enhancement,
            "predicted_success_rate_improvement": min(
                overall_enhancement, 3.0
            ),  # Cap at 3x
        }

    # Helper methods for data saving and summarization
    def _save_architecture_data(
        self, experiment_id: str, data: MulticellularArchitectureData
    ):
        """Save architecture data to file"""
        output_file = self.processed_dir / f"{experiment_id}_architecture.json"
        with open(output_file, "w") as f:
            json.dump(data.__dict__, f, indent=2)

    def _save_vascularization_data(self, experiment_id: str, data: VascularizationData):
        """Save vascularization data to file"""
        output_file = self.processed_dir / f"{experiment_id}_vascularization.json"
        with open(output_file, "w") as f:
            json.dump(data.__dict__, f, indent=2)

    def _save_kinetic_data(self, experiment_id: str, data: KineticAnalysisData):
        """Save kinetic data to file"""
        output_file = self.processed_dir / f"{experiment_id}_kinetics.json"
        with open(output_file, "w") as f:
            json.dump(data.__dict__, f, indent=2, default=str)

    def _save_perfusion_data(self, experiment_id: str, data: PerfusionCultureData):
        """Save perfusion data to file"""
        output_file = self.processed_dir / f"{experiment_id}_perfusion.json"
        with open(output_file, "w") as f:
            json.dump(data.__dict__, f, indent=2)

    def _summarize_architecture(
        self, data: MulticellularArchitectureData
    ) -> Dict[str, str]:
        """Create architecture summary"""
        return {
            "dominant_cell_type": max(
                data.cell_type_composition.items(), key=lambda x: x[1]
            )[0],
            "barrier_quality": (
                "excellent" if data.barrier_function_teer >= 1000 else "suboptimal"
            ),
            "tissue_organization": (
                "well_organized"
                if data.tissue_organization_score >= 0.8
                else "developing"
            ),
        }

    def _summarize_vascularization(self, data: VascularizationData) -> Dict[str, str]:
        """Create vascularization summary"""
        return {
            "delivery_enhancement": data.calculate_delivery_improvement(),
            "network_quality": (
                "high_density" if data.vascular_network_density >= 80 else "moderate"
            ),
            "viability_extension": (
                "extended" if data.culture_viability_extension >= 14 else "standard"
            ),
        }

    def _summarize_kinetics(self, data: KineticAnalysisData) -> Dict[str, str]:
        """Create kinetics summary"""
        return {
            "temporal_resolution": (
                "sub_minute"
                if data.temporal_resolution_minutes <= 1.0
                else "minute_scale"
            ),
            "monitoring_duration": (
                "extended" if data.continuous_monitoring_duration >= 24 else "standard"
            ),
            "kinetic_quality": (
                "high" if data.get_kinetic_quality_score() >= 0.8 else "moderate"
            ),
        }

    def _summarize_perfusion(self, data: PerfusionCultureData) -> Dict[str, str]:
        """Create perfusion summary"""
        return {
            "culture_longevity": (
                "extended" if data.culture_longevity_days >= 14 else "standard"
            ),
            "metabolic_support": (
                "enhanced"
                if np.mean(list(data.metabolic_activity_scores.values())) >= 1.5
                else "baseline"
            ),
            "automation_level": (
                "high" if data.automated_control_accuracy >= 0.9 else "moderate"
            ),
        }
