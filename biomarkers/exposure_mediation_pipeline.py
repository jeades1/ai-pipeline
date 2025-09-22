"""
Exposure-Mediation Pipeline

This module extends the existing mediation framework to handle exposure → mediator → outcome pathways:
- Temporal alignment of exposure and biomarker data
- Environment-epigenetic-clinical mediation analysis  
- Multi-exposure mixture modeling
- Mechanism-informed mediation with KG pathway support
- Bootstrap validation with exposure uncertainty propagation

Builds on src/mediation framework with exposure-specific enhancements.

Author: AI Pipeline Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import base mediation framework
try:
    from ..src.mediation import (
        MediationResult,
        MediationPathway,
    )

    # Try to import base classes
    try:
        from ..src.mediation import MediationEvidence, MediationAnalyzer

        BASE_MEDIATION_AVAILABLE = True
    except ImportError:
        BASE_MEDIATION_AVAILABLE = False

        # Define minimal base classes
        class MediationEvidence:
            pass

        class MediationAnalyzer:
            pass

except ImportError:
    # Define fallback classes if mediation module not available
    BASE_MEDIATION_AVAILABLE = False

    class MediationResult:
        pass

    class MediationPathway:
        pass

    class MediationEvidence:
        pass

    class MediationAnalyzer:
        pass


# Import exposure standards
from .exposure_standards import ExposureRecord, ExposureDataset, TemporalAligner
from .mechanism_kg_extensions import query_mechanism_paths

logger = logging.getLogger(__name__)


@dataclass
class ExposureMediationEvidence(MediationEvidence):
    """Extended mediation evidence for exposure pathways"""

    # Exposure-specific fields
    exposure_window: timedelta = timedelta(days=30)
    exposure_measurement_count: int = 0
    exposure_temporal_stability: float = 0.0

    # Mixture effects
    co_exposures: List[str] = field(default_factory=list)
    interaction_effects: Dict[str, float] = field(default_factory=dict)
    mixture_analysis_performed: bool = False

    # Mechanism support
    mechanism_pathways: List[str] = field(default_factory=list)
    aop_pathway_support: Optional[str] = None
    ctd_evidence_count: int = 0
    pathway_concordance_score: float = 0.0

    # Environmental context
    seasonal_variation: Optional[float] = None
    geographic_heterogeneity: Optional[float] = None
    exposure_source_reliability: str = "unknown"


@dataclass
class ExposureMediationResult(MediationResult):
    """Extended mediation result with exposure validation"""

    exposure_mediation_evidence: ExposureMediationEvidence
    temporal_alignment_quality: float = 0.0
    exposure_uncertainty_propagation: Dict[str, float] = field(default_factory=dict)
    mixture_model_results: Optional[Dict[str, Any]] = None
    mechanism_pathway_analysis: Optional[Dict[str, Any]] = None


class ExposureMediationAnalyzer(MediationAnalyzer):
    """Extended mediation analyzer for exposure pathways"""

    def __init__(self, kg_path: Optional[Path] = None, mechanism_kg=None):
        super().__init__(kg_path)
        self.mechanism_kg = mechanism_kg
        self.temporal_aligner = TemporalAligner()

        # Exposure-specific parameters
        self.exposure_window_defaults = {
            "air_quality": timedelta(days=7),
            "chemical_biomarker": timedelta(days=1),
            "lifestyle_behavioral": timedelta(days=14),
            "occupational": timedelta(days=30),
            "dietary": timedelta(days=7),
        }

    def analyze_exposure_mediation_pathway(
        self,
        exposure_data: ExposureDataset,
        molecular_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        exposure_analyte: str,
        molecular_mediator: str,
        clinical_outcome: str,
        exposure_window: Optional[timedelta] = None,
    ) -> ExposureMediationResult:
        """
        Analyze exposure → molecular mediator → clinical outcome pathway

        Args:
            exposure_data: Standardized exposure dataset
            molecular_data: Molecular biomarker data (epigenetics, transcriptomics, etc.)
            clinical_data: Clinical outcomes and patient metadata
            exposure_analyte: Specific exposure of interest
            molecular_mediator: Molecular mediator (gene, protein, metabolite)
            clinical_outcome: Target clinical outcome
            exposure_window: Exposure averaging window (auto-determined if None)

        Returns:
            ExposureMediationResult with exposure-specific validation
        """

        logger.info(
            f"Analyzing exposure mediation: {exposure_analyte} → {molecular_mediator} → {clinical_outcome}"
        )

        # Determine exposure window
        if exposure_window is None:
            exposure_window = self._determine_exposure_window(
                exposure_data, exposure_analyte
            )

        # Prepare temporally aligned data
        aligned_data = self._prepare_exposure_mediation_data(
            exposure_data,
            molecular_data,
            clinical_data,
            exposure_analyte,
            molecular_mediator,
            clinical_outcome,
            exposure_window,
        )

        if aligned_data.empty:
            logger.warning(
                f"No aligned data for exposure mediation: {exposure_analyte}"
            )
            return self._create_null_exposure_result(
                exposure_analyte, molecular_mediator, clinical_outcome
            )

        # Assess temporal alignment quality
        alignment_quality = self._assess_temporal_alignment_quality(
            aligned_data, exposure_window
        )

        # Run core mediation analysis
        mediation_stats = self._calculate_exposure_mediation_statistics(
            aligned_data, exposure_analyte, molecular_mediator, clinical_outcome
        )

        # Analyze co-exposures and mixture effects
        mixture_results = self._analyze_exposure_mixtures(
            aligned_data, exposure_analyte, molecular_mediator, clinical_outcome
        )

        # Query mechanism pathways if KG available
        mechanism_analysis = None
        if self.mechanism_kg:
            mechanism_analysis = self._analyze_mechanism_pathways(
                exposure_analyte, molecular_mediator, clinical_outcome
            )

        # Create exposure-specific evidence
        exposure_evidence = self._create_exposure_mediation_evidence(
            exposure_analyte,
            molecular_mediator,
            clinical_outcome,
            mediation_stats,
            aligned_data,
            exposure_window,
            mixture_results,
            mechanism_analysis,
        )

        # Propagate exposure uncertainty
        uncertainty_propagation = self._propagate_exposure_uncertainty(
            aligned_data, mediation_stats, exposure_analyte
        )

        # Validation with bootstrap accounting for exposure correlation
        validation_metrics = self._cross_validate_exposure_mediation(
            aligned_data, exposure_analyte, molecular_mediator, clinical_outcome
        )

        # Generate biological interpretation with mechanism context
        mechanism_hypothesis = self._generate_exposure_mechanism_hypothesis(
            exposure_evidence, mechanism_analysis
        )

        result = ExposureMediationResult(
            mediation_evidence=exposure_evidence,
            exposure_mediation_evidence=exposure_evidence,
            supporting_data=self._extract_exposure_supporting_data(
                aligned_data, mediation_stats
            ),
            model_diagnostics=self._calculate_exposure_model_diagnostics(
                aligned_data, mediation_stats
            ),
            validation_metrics=validation_metrics,
            temporal_alignment_quality=alignment_quality,
            exposure_uncertainty_propagation=uncertainty_propagation,
            mixture_model_results=mixture_results,
            mechanism_pathway_analysis=mechanism_analysis,
            mechanism_hypothesis=mechanism_hypothesis,
            intervention_potential=self._assess_exposure_intervention_potential(
                exposure_evidence
            ),
            translational_readiness=self._assess_exposure_translational_readiness(
                exposure_evidence, validation_metrics
            ),
        )

        logger.info(
            f"Exposure mediation analysis complete: {exposure_evidence.mediation_id}"
        )

        return result

    def _determine_exposure_window(
        self, exposure_data: ExposureDataset, exposure_analyte: str
    ) -> timedelta:
        """Determine appropriate exposure averaging window"""

        # Get exposure type for this analyte
        exposure_records = [
            r
            for r in exposure_data.records
            if r.analyte_id == exposure_analyte or r.analyte_name == exposure_analyte
        ]

        if not exposure_records:
            return timedelta(days=30)  # Default

        exposure_type = exposure_records[0].exposure_type.value
        return self.exposure_window_defaults.get(exposure_type, timedelta(days=30))

    def _prepare_exposure_mediation_data(
        self,
        exposure_data: ExposureDataset,
        molecular_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        exposure_analyte: str,
        molecular_mediator: str,
        clinical_outcome: str,
        exposure_window: timedelta,
    ) -> pd.DataFrame:
        """Prepare temporally aligned exposure-molecular-clinical data"""

        # Extract relevant exposure records
        exposure_records = [
            r
            for r in exposure_data.records
            if (r.analyte_id == exposure_analyte or r.analyte_name == exposure_analyte)
        ]

        if not exposure_records:
            logger.warning(f"No exposure records found for {exposure_analyte}")
            return pd.DataFrame()

        # Convert exposure records to DataFrame
        exposure_df = pd.DataFrame(
            [
                {
                    "subject_id": r.subject_id,
                    "measured_at": r.measured_at,
                    "exposure_value": r.value,
                    "exposure_unit": r.unit,
                    "exposure_quality": r.measurement_quality,
                    "exposure_uncertainty": r.uncertainty or 0.0,
                }
                for r in exposure_records
            ]
        )

        # Extract molecular mediator data
        if "gene" in molecular_data.columns:
            mol_data = molecular_data[
                molecular_data["gene"] == molecular_mediator
            ].copy()
        elif "biomarker_id" in molecular_data.columns:
            mol_data = molecular_data[
                molecular_data["biomarker_id"] == molecular_mediator
            ].copy()
        else:
            logger.warning("No gene/biomarker_id column found in molecular data")
            return pd.DataFrame()

        if mol_data.empty:
            logger.warning(f"No molecular data found for {molecular_mediator}")
            return pd.DataFrame()

        # Ensure we have required columns
        required_cols = ["subject_id", "sample_time", "value"]
        if not all(col in mol_data.columns for col in required_cols):
            logger.warning(
                f"Missing required columns in molecular data: {required_cols}"
            )
            return pd.DataFrame()

        # Extract clinical outcome data
        if clinical_outcome not in clinical_data.columns:
            logger.warning(
                f"Clinical outcome '{clinical_outcome}' not found in clinical data"
            )
            return pd.DataFrame()

        clin_data = clinical_data[["subject_id", clinical_outcome]].copy()

        # Temporal alignment: for each molecular sample, find exposure values within window
        aligned_records = []

        for _, mol_row in mol_data.iterrows():
            subject_id = mol_row["subject_id"]
            sample_time = pd.to_datetime(mol_row["sample_time"])
            molecular_value = mol_row["value"]

            # Find subject's exposures within window before sampling
            subject_exposures = exposure_df[
                exposure_df["subject_id"] == subject_id
            ].copy()

            if subject_exposures.empty:
                continue

            # Filter exposures within window
            window_start = sample_time - exposure_window
            window_exposures = subject_exposures[
                (subject_exposures["measured_at"] >= window_start)
                & (subject_exposures["measured_at"] <= sample_time)
            ]

            if window_exposures.empty:
                continue

            # Calculate exposure summary statistics (weighted by recency)
            exposure_values = window_exposures["exposure_value"].values
            exposure_times = pd.to_datetime(window_exposures["measured_at"]).values

            # Weight by recency (more recent exposures weighted higher)
            time_diffs = [
                (sample_time - pd.to_datetime(t)).total_seconds() / 86400
                for t in exposure_times
            ]  # Days
            weights = [1.0 / (1.0 + td) for td in time_diffs]  # Inverse time weighting

            # Weighted exposure metrics
            weighted_mean = np.average(exposure_values, weights=weights)
            exposure_max = np.max(exposure_values)
            exposure_count = len(exposure_values)
            exposure_std = np.std(exposure_values) if len(exposure_values) > 1 else 0.0

            # Get clinical outcome for this subject
            subject_clinical = clin_data[clin_data["subject_id"] == subject_id]
            if subject_clinical.empty:
                continue

            clinical_value = subject_clinical[clinical_outcome].iloc[0]

            aligned_record = {
                "subject_id": subject_id,
                "sample_time": sample_time,
                "exposure_mean": weighted_mean,
                "exposure_max": exposure_max,
                "exposure_count": exposure_count,
                "exposure_stability": 1.0 / (1.0 + exposure_std),  # Stability score
                "molecular_value": molecular_value,
                "clinical_outcome": clinical_value,
                "exposure_window_days": exposure_window.days,
            }

            aligned_records.append(aligned_record)

        aligned_df = pd.DataFrame(aligned_records)

        logger.info(
            f"Aligned {len(aligned_df)} samples for exposure mediation analysis"
        )

        return aligned_df

    def _assess_temporal_alignment_quality(
        self, aligned_data: pd.DataFrame, exposure_window: timedelta
    ) -> float:
        """Assess quality of temporal alignment between exposure and biomarker data"""

        if aligned_data.empty:
            return 0.0

        # Quality metrics

        # 1. Exposure measurement density (measurements per window)
        avg_exposure_count = aligned_data["exposure_count"].mean()
        density_score = min(
            1.0, avg_exposure_count / 5.0
        )  # Normalize to 5 measurements per window

        # 2. Temporal stability of exposures
        avg_stability = aligned_data["exposure_stability"].mean()

        # 3. Window appropriateness (prefer shorter windows for more precise alignment)
        window_score = max(
            0.1, 1.0 - (exposure_window.days / 365.0)
        )  # Penalty for very long windows

        # 4. Data completeness
        completeness_score = 1.0 - aligned_data.isnull().sum().sum() / (
            len(aligned_data) * len(aligned_data.columns)
        )

        # Combined quality score
        quality_score = (
            0.3 * density_score
            + 0.3 * avg_stability
            + 0.2 * window_score
            + 0.2 * completeness_score
        )

        return min(1.0, quality_score)

    def _calculate_exposure_mediation_statistics(
        self,
        aligned_data: pd.DataFrame,
        exposure_analyte: str,
        molecular_mediator: str,
        clinical_outcome: str,
    ) -> Dict[str, float]:
        """Calculate mediation statistics for exposure pathway"""

        try:
            # Extract variables (use weighted exposure mean)
            X = aligned_data["exposure_mean"].values  # Exposure
            M = aligned_data["molecular_value"].values  # Molecular mediator
            Y = aligned_data["clinical_outcome"].values  # Clinical outcome

            # Remove missing values
            mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
            X, M, Y = X[mask], M[mask], Y[mask]

            if len(X) < 10:
                logger.warning(
                    f"Insufficient data for exposure mediation: {len(X)} samples"
                )
                return self._create_null_statistics()

            # Standardize variables for stability
            X = (X - np.mean(X)) / np.std(X) if np.std(X) > 0 else X
            M = (M - np.mean(M)) / np.std(M) if np.std(M) > 0 else M
            Y = (Y - np.mean(Y)) / np.std(Y) if np.std(Y) > 0 else Y

            # Mediation analysis (Baron & Kenny approach)
            # Path c: X → Y (total effect)
            c = np.corrcoef(X, Y)[0, 1] if np.std(Y) > 0 else 0.0

            # Path a: X → M (exposure → mediator)
            a = np.corrcoef(X, M)[0, 1] if np.std(M) > 0 else 0.0

            # Path b: M → Y (controlling for X)
            if np.std(M) > 0 and np.std(Y) > 0:
                r_my = np.corrcoef(M, Y)[0, 1]
                r_mx = np.corrcoef(M, X)[0, 1]
                r_xy = np.corrcoef(X, Y)[0, 1]

                # Partial correlation: b = corr(M,Y|X)
                if abs(r_mx) < 0.99 and abs(r_xy) < 0.99:
                    denominator = np.sqrt((1 - r_mx**2) * (1 - r_xy**2))
                    b = (
                        (r_my - r_mx * r_xy) / denominator
                        if denominator > 1e-6
                        else 0.0
                    )
                else:
                    b = 0.0
            else:
                b = 0.0

            # Calculate effects
            indirect_effect = a * b
            direct_effect = c - indirect_effect
            total_effect = c

            # Mediation proportion
            mediation_proportion = (
                indirect_effect / total_effect if abs(total_effect) > 1e-6 else 0.0
            )

            # Statistical significance (approximate)
            n = len(X)
            se_c = 1 / np.sqrt(max(1, n - 3))
            se_a = 1 / np.sqrt(max(1, n - 3))
            se_b = 1 / np.sqrt(max(1, n - 3))
            se_indirect = (
                np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
                if se_a > 0 and se_b > 0
                else 1.0
            )

            # Calculate p-values (approximate using t-distribution)
            from scipy.stats import t

            df = max(1, n - 3)

            t_c = c / se_c if se_c > 0 else 0.0
            t_indirect = indirect_effect / se_indirect if se_indirect > 0 else 0.0

            p_direct = 2 * (1 - t.cdf(abs(t_c), df))
            p_indirect = 2 * (1 - t.cdf(abs(t_indirect), df))
            p_mediation = p_indirect

            # R-squared for exposure → mediator relationship (path a strength)
            r_squared_a = a**2 if not np.isnan(a) else 0.0

            # Account for exposure uncertainty in effect estimates
            exposure_uncertainty = aligned_data["exposure_stability"].mean()
            uncertainty_adjusted_effects = {
                "total_effect": total_effect * exposure_uncertainty,
                "direct_effect": direct_effect * exposure_uncertainty,
                "indirect_effect": indirect_effect * exposure_uncertainty,
            }

            return {
                "direct_effect": float(direct_effect),
                "indirect_effect": float(indirect_effect),
                "total_effect": float(total_effect),
                "mediation_proportion": float(mediation_proportion),
                "path_a": float(a),
                "path_b": float(b),
                "path_c": float(c),
                "path_c_prime": float(direct_effect),
                "p_value_direct": float(p_direct),
                "p_value_indirect": float(p_indirect),
                "p_value_mediation": float(p_mediation),
                "r_squared": float(r_squared_a),
                "sample_size": int(n),
                "se_direct": float(se_c),
                "se_indirect": float(se_indirect),
                "exposure_uncertainty_score": float(exposure_uncertainty),
                "uncertainty_adjusted_total": float(
                    uncertainty_adjusted_effects["total_effect"]
                ),
                "uncertainty_adjusted_indirect": float(
                    uncertainty_adjusted_effects["indirect_effect"]
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating exposure mediation statistics: {e}")
            return self._create_null_statistics()

    def _analyze_exposure_mixtures(
        self,
        aligned_data: pd.DataFrame,
        primary_exposure: str,
        molecular_mediator: str,
        clinical_outcome: str,
    ) -> Optional[Dict[str, Any]]:
        """Analyze mixture effects and co-exposures"""

        # Check if we have multiple exposure variables in the data
        exposure_cols = [
            col
            for col in aligned_data.columns
            if col.startswith("exposure_") and "mean" in col
        ]

        if len(exposure_cols) <= 1:
            logger.info("No co-exposures detected for mixture analysis")
            return None

        logger.info(f"Analyzing mixture effects with {len(exposure_cols)} exposures")

        # Simple mixture analysis: assess correlation between co-exposures
        mixture_analysis = {
            "primary_exposure": primary_exposure,
            "co_exposures": [
                col.replace("exposure_", "").replace("_mean", "")
                for col in exposure_cols
                if "mean" in col
            ],
            "correlation_matrix": {},
            "interaction_effects": {},
            "mixture_model_r2": 0.0,
        }

        # Calculate exposure correlations
        for i, exp1 in enumerate(exposure_cols):
            for j, exp2 in enumerate(exposure_cols):
                if i != j:
                    corr = np.corrcoef(aligned_data[exp1], aligned_data[exp2])[0, 1]
                    mixture_analysis["correlation_matrix"][f"{exp1}_vs_{exp2}"] = float(
                        corr
                    )

        # Simple interaction assessment: product terms
        Y = aligned_data["clinical_outcome"].values

        for exp_col in exposure_cols[1:]:  # Skip primary exposure
            if exp_col != "exposure_mean":
                # Test interaction between primary exposure and co-exposure
                X1 = aligned_data["exposure_mean"].values
                X2 = aligned_data[exp_col].values
                interaction_term = X1 * X2

                # Simple correlation with outcome
                if np.std(interaction_term) > 0:
                    interaction_corr = np.corrcoef(interaction_term, Y)[0, 1]
                    mixture_analysis["interaction_effects"][exp_col] = float(
                        interaction_corr
                    )

        return mixture_analysis

    def _analyze_mechanism_pathways(
        self, exposure_analyte: str, molecular_mediator: str, clinical_outcome: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze mechanism pathways using knowledge graph"""

        if not self.mechanism_kg:
            return None

        logger.info(
            f"Analyzing mechanism pathways for {exposure_analyte} → {molecular_mediator} → {clinical_outcome}"
        )

        # Query exposure → mediator paths
        exposure_mediator_paths = query_mechanism_paths(
            self.mechanism_kg, exposure_analyte, molecular_mediator, max_path_length=3
        )

        # Query mediator → outcome paths
        mediator_outcome_paths = query_mechanism_paths(
            self.mechanism_kg, molecular_mediator, clinical_outcome, max_path_length=3
        )

        # Query direct exposure → outcome paths
        direct_paths = query_mechanism_paths(
            self.mechanism_kg, exposure_analyte, clinical_outcome, max_path_length=5
        )

        pathway_analysis = {
            "exposure_to_mediator_paths": exposure_mediator_paths[:5],  # Top 5 paths
            "mediator_to_outcome_paths": mediator_outcome_paths[:5],
            "direct_exposure_outcome_paths": direct_paths[:3],
            "pathway_support_score": 0.0,
            "aop_pathways": [],
            "ctd_relationships": 0,
        }

        # Calculate pathway support score
        total_paths = (
            len(exposure_mediator_paths)
            + len(mediator_outcome_paths)
            + len(direct_paths)
        )
        if total_paths > 0:
            avg_evidence_score = np.mean(
                [
                    p["evidence_score"]
                    for p in exposure_mediator_paths[:3]
                    + mediator_outcome_paths[:3]
                    + direct_paths[:3]
                ]
            )
            pathway_analysis["pathway_support_score"] = float(avg_evidence_score)

        # Extract AOP pathways and CTD relationships
        for path_group in [
            exposure_mediator_paths,
            mediator_outcome_paths,
            direct_paths,
        ]:
            for path in path_group:
                # Count AOP pathways
                aop_evidence = [
                    node for node in path["path"] if node.startswith("AOP:")
                ]
                pathway_analysis["aop_pathways"].extend(aop_evidence)

                # Count CTD relationships
                ctd_count = path["evidence_sources"].get("CTD", 0)
                pathway_analysis["ctd_relationships"] += ctd_count

        # Remove duplicates
        pathway_analysis["aop_pathways"] = list(set(pathway_analysis["aop_pathways"]))

        return pathway_analysis

    def _create_exposure_mediation_evidence(
        self,
        exposure_analyte: str,
        molecular_mediator: str,
        clinical_outcome: str,
        stats: Dict[str, float],
        aligned_data: pd.DataFrame,
        exposure_window: timedelta,
        mixture_results: Optional[Dict[str, Any]] = None,
        mechanism_analysis: Optional[Dict[str, Any]] = None,
    ) -> ExposureMediationEvidence:
        """Create comprehensive exposure mediation evidence"""

        mediation_id = (
            f"EXPOSURE_{exposure_analyte}_{molecular_mediator}_{clinical_outcome}"
        )

        # Determine pathway type
        if stats["mediation_proportion"] > 0.3:
            pathway_type = MediationPathway.FULL_MEDIATION
        elif abs(stats["indirect_effect"]) > abs(stats["direct_effect"]):
            pathway_type = MediationPathway.MOLECULAR_TO_FUNCTIONAL
        else:
            pathway_type = MediationPathway.DIRECT_MOLECULAR_CLINICAL

        # Effect direction
        if stats["total_effect"] > 0.1:
            effect_direction = "positive"
        elif stats["total_effect"] < -0.1:
            effect_direction = "negative"
        else:
            effect_direction = "minimal"

        # Evidence strength (considering exposure uncertainty)
        uncertainty_score = stats.get("exposure_uncertainty_score", 0.5)
        adjusted_p_value = (
            stats["p_value_mediation"] / uncertainty_score
        )  # Penalize for uncertainty

        if (
            adjusted_p_value < 0.05
            and abs(stats["mediation_proportion"]) > 0.3
            and stats["sample_size"] > 30
            and uncertainty_score > 0.7
        ):
            evidence_strength = "strong"
        elif (
            adjusted_p_value < 0.1
            and abs(stats["mediation_proportion"]) > 0.1
            and stats["sample_size"] > 10
            and uncertainty_score > 0.5
        ):
            evidence_strength = "moderate"
        else:
            evidence_strength = "weak"

        # Confidence interval (exposure uncertainty-adjusted)
        margin_error = 1.96 * stats["se_indirect"] / uncertainty_score
        ci_lower = stats["indirect_effect"] - margin_error
        ci_upper = stats["indirect_effect"] + margin_error

        # Extract mixture and mechanism information
        co_exposures = mixture_results["co_exposures"] if mixture_results else []
        interaction_effects = (
            mixture_results["interaction_effects"] if mixture_results else {}
        )

        mechanism_pathways = []
        aop_pathway_support = None
        ctd_evidence_count = 0
        pathway_concordance_score = 0.0

        if mechanism_analysis:
            mechanism_pathways = mechanism_analysis.get("aop_pathways", [])
            aop_pathway_support = mechanism_pathways[0] if mechanism_pathways else None
            ctd_evidence_count = mechanism_analysis.get("ctd_relationships", 0)
            pathway_concordance_score = mechanism_analysis.get(
                "pathway_support_score", 0.0
            )

        # Environmental context
        seasonal_variation = None
        if "sample_time" in aligned_data.columns:
            # Simple seasonal analysis based on month
            sample_times = pd.to_datetime(aligned_data["sample_time"])
            months = sample_times.dt.month
            exposure_by_month = aligned_data.groupby(months)["exposure_mean"].std()
            seasonal_variation = (
                float(exposure_by_month.mean()) if len(exposure_by_month) > 1 else None
            )

        return ExposureMediationEvidence(
            mediation_id=mediation_id,
            pathway_type=pathway_type,
            molecular_entity=molecular_mediator,
            functional_mediator=None,  # Not applicable for exposure mediation
            clinical_outcome=clinical_outcome,
            direct_effect=stats["direct_effect"],
            indirect_effect=stats["indirect_effect"],
            total_effect=stats["total_effect"],
            mediation_proportion=stats["mediation_proportion"],
            p_value_direct=stats["p_value_direct"],
            p_value_indirect=stats["p_value_indirect"],
            p_value_mediation=stats["p_value_mediation"],
            confidence_interval=(ci_lower, ci_upper),
            molecular_to_functional_effect=stats["path_a"],
            functional_to_clinical_effect=stats["path_b"],
            effect_direction=effect_direction,
            sample_size=int(stats["sample_size"]),
            r_squared=stats["r_squared"],
            evidence_strength=evidence_strength,
            # Exposure-specific fields
            exposure_window=exposure_window,
            exposure_measurement_count=int(aligned_data["exposure_count"].mean()),
            exposure_temporal_stability=float(
                aligned_data["exposure_stability"].mean()
            ),
            co_exposures=co_exposures,
            interaction_effects=interaction_effects,
            mixture_analysis_performed=mixture_results is not None,
            mechanism_pathways=mechanism_pathways,
            aop_pathway_support=aop_pathway_support,
            ctd_evidence_count=ctd_evidence_count,
            pathway_concordance_score=pathway_concordance_score,
            seasonal_variation=seasonal_variation,
            exposure_source_reliability=self._assess_exposure_source_reliability(
                aligned_data
            ),
        )

    def _propagate_exposure_uncertainty(
        self,
        aligned_data: pd.DataFrame,
        mediation_stats: Dict[str, float],
        exposure_analyte: str,
    ) -> Dict[str, float]:
        """Propagate exposure measurement uncertainty through mediation analysis"""

        # Calculate uncertainty metrics
        exposure_cv = (
            aligned_data["exposure_stability"].std()
            / aligned_data["exposure_stability"].mean()
        )
        measurement_density = aligned_data["exposure_count"].mean()

        # Uncertainty propagation factors
        temporal_uncertainty = 1.0 / (
            1.0 + measurement_density
        )  # Lower with more measurements
        measurement_uncertainty = exposure_cv  # Higher with more variable exposures

        # Propagate uncertainty to effect estimates
        uncertainty_factor = np.sqrt(
            temporal_uncertainty**2 + measurement_uncertainty**2
        )

        return {
            "temporal_uncertainty": float(temporal_uncertainty),
            "measurement_uncertainty": float(measurement_uncertainty),
            "combined_uncertainty_factor": float(uncertainty_factor),
            "uncertainty_adjusted_mediation_proportion": float(
                mediation_stats["mediation_proportion"] * (1.0 - uncertainty_factor)
            ),
            "uncertainty_confidence_penalty": float(uncertainty_factor * 0.5),
        }

    def _cross_validate_exposure_mediation(
        self,
        aligned_data: pd.DataFrame,
        exposure_analyte: str,
        molecular_mediator: str,
        clinical_outcome: str,
    ) -> Dict[str, float]:
        """Cross-validate exposure mediation with bootstrap accounting for exposure correlation"""

        if len(aligned_data) < 20:
            return {"cv_score": 0.0, "stability_score": 0.0}

        # Bootstrap validation accounting for within-subject exposure correlation
        n_bootstrap = 100
        mediation_props = []

        # Group data by subject to preserve correlation structure
        subjects = aligned_data["subject_id"].unique()

        for _ in range(n_bootstrap):
            # Bootstrap subjects (not individual measurements)
            bootstrap_subjects = np.random.choice(
                subjects, size=len(subjects), replace=True
            )

            # Reconstruct bootstrap dataset
            bootstrap_data = []
            for subject in bootstrap_subjects:
                subject_data = aligned_data[aligned_data["subject_id"] == subject]
                bootstrap_data.append(subject_data)

            if bootstrap_data:
                bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)

                # Calculate mediation statistics
                bootstrap_stats = self._calculate_exposure_mediation_statistics(
                    bootstrap_df, exposure_analyte, molecular_mediator, clinical_outcome
                )

                mediation_props.append(bootstrap_stats["mediation_proportion"])

        # Calculate validation metrics
        cv_score = np.mean(mediation_props)
        stability_score = 1.0 - (np.std(mediation_props) / (abs(cv_score) + 1e-6))

        return {
            "cv_score": float(cv_score),
            "stability_score": float(max(0.0, stability_score)),
            "bootstrap_std": float(np.std(mediation_props)),
            "bootstrap_samples": n_bootstrap,
        }

    def _assess_exposure_source_reliability(self, aligned_data: pd.DataFrame) -> str:
        """Assess reliability of exposure data sources"""

        # Simple assessment based on measurement density and stability
        avg_count = aligned_data["exposure_count"].mean()
        avg_stability = aligned_data["exposure_stability"].mean()

        if avg_count >= 5 and avg_stability >= 0.8:
            return "high"
        elif avg_count >= 2 and avg_stability >= 0.6:
            return "moderate"
        else:
            return "low"

    def _generate_exposure_mechanism_hypothesis(
        self,
        evidence: ExposureMediationEvidence,
        mechanism_analysis: Optional[Dict[str, Any]],
    ) -> str:
        """Generate mechanism hypothesis incorporating pathway information"""

        base_hypothesis = f"{evidence.molecular_entity} mediates the {evidence.effect_direction} effect of {evidence.mediation_id.split('_')[1]} exposure on {evidence.clinical_outcome}"

        if (
            mechanism_analysis
            and mechanism_analysis.get("pathway_support_score", 0) > 0.5
        ):
            aop_support = f" AOP pathway support: {len(mechanism_analysis.get('aop_pathways', []))} pathways identified."
            ctd_support = f" CTD evidence: {mechanism_analysis.get('ctd_relationships', 0)} literature relationships."
            mechanism_hypothesis = base_hypothesis + aop_support + ctd_support
        else:
            mechanism_hypothesis = (
                base_hypothesis + " Limited mechanistic pathway support available."
            )

        if evidence.mixture_analysis_performed and evidence.co_exposures:
            mixture_note = f" Co-exposure analysis identified {len(evidence.co_exposures)} concurrent exposures."
            mechanism_hypothesis += mixture_note

        return mechanism_hypothesis

    def _assess_exposure_intervention_potential(
        self, evidence: ExposureMediationEvidence
    ) -> str:
        """Assess intervention potential considering exposure modifiability"""

        # Base assessment from mediation strength
        if (
            evidence.evidence_strength == "strong"
            and abs(evidence.mediation_proportion) > 0.3
            and evidence.p_value_mediation < 0.05
        ):
            base_potential = "high"
        elif (
            evidence.evidence_strength == "moderate"
            and abs(evidence.mediation_proportion) > 0.1
        ):
            base_potential = "medium"
        else:
            base_potential = "low"

        # Adjust based on exposure modifiability (simplified heuristic)
        exposure_name = evidence.mediation_id.split("_")[1].lower()

        # More modifiable exposures
        if any(
            term in exposure_name
            for term in ["diet", "lifestyle", "physical", "behavioral"]
        ):
            modifiability_bonus = 1
        # Moderately modifiable exposures
        elif any(
            term in exposure_name for term in ["indoor", "occupational", "residential"]
        ):
            modifiability_bonus = 0
        # Less modifiable exposures
        else:
            modifiability_bonus = -1

        # Adjust potential
        potential_levels = ["low", "medium", "high"]
        current_level = potential_levels.index(base_potential)
        adjusted_level = max(0, min(2, current_level + modifiability_bonus))

        return potential_levels[adjusted_level]

    def _assess_exposure_translational_readiness(
        self, evidence: ExposureMediationEvidence, validation_metrics: Dict[str, float]
    ) -> str:
        """Assess translational readiness for exposure interventions"""

        stability = validation_metrics.get("stability_score", 0.0)

        # Stricter criteria for exposure studies due to complexity
        if (
            evidence.evidence_strength == "strong"
            and evidence.sample_size > 100  # Larger sample needed
            and stability > 0.8
            and evidence.exposure_temporal_stability > 0.7
            and evidence.pathway_concordance_score > 0.6
        ):
            return "ready"
        elif (
            evidence.evidence_strength in ["moderate", "strong"]
            and evidence.sample_size > 50
            and stability > 0.6
            and evidence.exposure_temporal_stability > 0.5
        ):
            return "needs_validation"
        else:
            return "early"

    def _extract_exposure_supporting_data(
        self, aligned_data: pd.DataFrame, mediation_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Extract exposure-specific supporting data"""

        supporting_data = {
            "raw_statistics": mediation_stats,
            "exposure_data_summary": {
                "n_samples": len(aligned_data),
                "n_subjects": aligned_data["subject_id"].nunique(),
                "exposure_mean": aligned_data["exposure_mean"].mean(),
                "exposure_std": aligned_data["exposure_mean"].std(),
                "molecular_mean": aligned_data["molecular_value"].mean(),
                "molecular_std": aligned_data["molecular_value"].std(),
                "avg_measurement_count": aligned_data["exposure_count"].mean(),
                "avg_exposure_stability": aligned_data["exposure_stability"].mean(),
            },
            "temporal_alignment": {
                "avg_window_days": (
                    aligned_data["exposure_window_days"].iloc[0]
                    if len(aligned_data) > 0
                    else 0
                ),
                "measurement_density": aligned_data["exposure_count"].mean(),
                "temporal_stability": aligned_data["exposure_stability"].mean(),
            },
        }

        return supporting_data

    def _calculate_exposure_model_diagnostics(
        self, aligned_data: pd.DataFrame, stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate exposure-specific model diagnostics"""

        return {
            "sample_size": stats["sample_size"],
            "r_squared": stats["r_squared"],
            "effect_size": abs(stats["total_effect"]),
            "mediation_strength": abs(stats["mediation_proportion"]),
            "path_a_strength": abs(stats["path_a"]),
            "path_b_strength": abs(stats["path_b"]),
            "data_completeness": 1.0
            - aligned_data.isnull().sum().sum()
            / (len(aligned_data) * len(aligned_data.columns)),
            "exposure_uncertainty_score": stats.get("exposure_uncertainty_score", 0.5),
            "temporal_alignment_quality": aligned_data["exposure_stability"].mean(),
        }

    def _create_null_exposure_result(
        self, exposure_analyte: str, molecular_mediator: str, clinical_outcome: str
    ) -> ExposureMediationResult:
        """Create null result for failed exposure mediation analysis"""

        null_evidence = ExposureMediationEvidence(
            mediation_id=f"EXPOSURE_{exposure_analyte}_{molecular_mediator}_{clinical_outcome}",
            pathway_type=MediationPathway.DIRECT_MOLECULAR_CLINICAL,
            molecular_entity=molecular_mediator,
            functional_mediator=None,
            clinical_outcome=clinical_outcome,
            direct_effect=0.0,
            indirect_effect=0.0,
            total_effect=0.0,
            mediation_proportion=0.0,
            p_value_direct=1.0,
            p_value_indirect=1.0,
            p_value_mediation=1.0,
            confidence_interval=(0.0, 0.0),
            evidence_strength="none",
            exposure_window=timedelta(days=30),
            exposure_source_reliability="unknown",
        )

        return ExposureMediationResult(
            mediation_evidence=null_evidence,
            exposure_mediation_evidence=null_evidence,
            supporting_data={},
            model_diagnostics={"sample_size": 0.0, "r_squared": 0.0},
            validation_metrics={"cv_score": 0.0},
            temporal_alignment_quality=0.0,
            mechanism_hypothesis="Insufficient data for exposure mediation analysis",
            intervention_potential="unknown",
            translational_readiness="needs_data",
        )


def create_exposure_mediation_demo_data() -> (
    Tuple[ExposureDataset, pd.DataFrame, pd.DataFrame]
):
    """Create demonstration data for exposure mediation analysis"""

    from .exposure_standards import (
        ExposureDataset,
        ExposureType,
        TemporalResolution,
        SpatialResolution,
    )

    np.random.seed(42)
    n_subjects = 100

    # Generate subject IDs
    subject_ids = [f"SUBJ_{i:03d}" for i in range(n_subjects)]

    # Create exposure records (air pollution exposure)
    exposure_records = []
    base_date = datetime(2023, 1, 1)

    for subject_id in subject_ids:
        # Generate 30 days of exposure data per subject
        for day in range(30):
            exposure_date = base_date + timedelta(days=day)

            # Simulate PM2.5 exposure with some subject-specific variation
            subject_baseline = np.random.normal(15, 5)  # Subject-specific baseline
            daily_pm25 = max(
                0, subject_baseline + np.random.normal(0, 3)
            )  # Daily variation

            record = ExposureRecord(
                subject_id=subject_id,
                exposure_id=f"PM25_{subject_id}_{exposure_date.strftime('%Y%m%d')}",
                analyte_id="CHEBI:132076",
                analyte_name="PM2.5",
                measured_at=exposure_date,
                measurement_window=timedelta(days=1),
                value=daily_pm25,
                unit="ug/m3",
                latitude=40.7 + np.random.normal(0, 0.1),
                longitude=-74.0 + np.random.normal(0, 0.1),
                location_type="residential",
                temporal_resolution=TemporalResolution.DAILY,
                data_source="EPA_AQS",
                exposure_type=ExposureType.AIR_QUALITY,
                measurement_quality="good",
            )
            exposure_records.append(record)

    # Create exposure dataset
    exposure_dataset = ExposureDataset(
        records=exposure_records,
        dataset_id="PM25_DEMO_20230101",
        dataset_name="PM2.5 Exposure Demo",
        exposure_types=[ExposureType.AIR_QUALITY],
        start_date=base_date,
        end_date=base_date + timedelta(days=29),
        temporal_resolution=TemporalResolution.DAILY,
        spatial_extent={
            "min_lat": 40.5,
            "max_lat": 40.9,
            "min_lon": -74.3,
            "max_lon": -73.7,
        },
        spatial_resolution=SpatialResolution.POINT,
        n_subjects=n_subjects,
        completeness_score=0.95,
    )

    # Create molecular mediator data (inflammatory gene expression)
    molecular_data = []
    sample_date = base_date + timedelta(days=30)  # Sampled after exposure period

    for subject_id in subject_ids:
        # Calculate subject's cumulative PM2.5 exposure
        subject_exposures = [r for r in exposure_records if r.subject_id == subject_id]
        avg_pm25 = np.mean([r.value for r in subject_exposures])

        # Simulate IL6 expression correlated with PM2.5 exposure
        # Higher PM2.5 → higher IL6 expression
        il6_expression = 5.0 + 0.1 * avg_pm25 + np.random.normal(0, 0.5)

        molecular_data.append(
            {
                "subject_id": subject_id,
                "gene": "IL6",
                "sample_time": sample_date
                + timedelta(hours=np.random.randint(-12, 12)),
                "value": il6_expression,
                "sample_type": "blood",
                "platform": "qPCR",
            }
        )

    molecular_df = pd.DataFrame(molecular_data)

    # Create clinical outcome data
    clinical_data = []

    for subject_id in subject_ids:
        # Get subject's IL6 expression
        subject_il6 = molecular_df[molecular_df["subject_id"] == subject_id][
            "value"
        ].iloc[0]

        # Simulate acute kidney injury risk
        # Higher IL6 → higher AKI risk (mediation pathway)
        aki_risk = 0.05 + 0.02 * (subject_il6 - 5.0)  # Baseline 5% risk + IL6 effect
        aki_outcome = 1 if np.random.random() < aki_risk else 0

        # Also add direct PM2.5 effect (partial mediation)
        subject_exposures = [r for r in exposure_records if r.subject_id == subject_id]
        avg_pm25 = np.mean([r.value for r in subject_exposures])
        direct_pm25_risk = 0.01 * (avg_pm25 - 15) / 10  # Direct effect

        if np.random.random() < direct_pm25_risk:
            aki_outcome = 1

        clinical_data.append(
            {
                "subject_id": subject_id,
                "aki": aki_outcome,
                "age": np.random.randint(40, 80),
                "sex": np.random.choice(["male", "female"]),
                "comorbidities": np.random.choice(["none", "diabetes", "hypertension"]),
            }
        )

    clinical_df = pd.DataFrame(clinical_data)

    return exposure_dataset, molecular_df, clinical_df


def run_exposure_mediation_demo():
    """Demonstrate exposure-mediation pipeline"""

    print("\n🌍➡️🧬➡️🏥 EXPOSURE-MEDIATION PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Create demo data
    print("📊 Generating demonstration data...")
    exposure_dataset, molecular_data, clinical_data = (
        create_exposure_mediation_demo_data()
    )

    print(f"   Exposure records: {len(exposure_dataset.records)}")
    print(f"   Molecular samples: {len(molecular_data)}")
    print(f"   Clinical subjects: {len(clinical_data)}")
    print(f"   Exposure type: {exposure_dataset.exposure_types[0].value}")
    print(
        f"   Temporal coverage: {exposure_dataset.start_date.date()} to {exposure_dataset.end_date.date()}"
    )

    # Initialize exposure mediation analyzer
    print("\n🔬 Initializing exposure mediation analyzer...")
    analyzer = ExposureMediationAnalyzer()

    # Run exposure mediation analysis
    print("\n🔍 Running exposure mediation analysis:")
    print("   Pathway: PM2.5 exposure → IL6 expression → acute kidney injury")

    result = analyzer.analyze_exposure_mediation_pathway(
        exposure_data=exposure_dataset,
        molecular_data=molecular_data,
        clinical_data=clinical_data,
        exposure_analyte="PM2.5",
        molecular_mediator="IL6",
        clinical_outcome="aki",
    )

    evidence = result.exposure_mediation_evidence

    print("\n📈 EXPOSURE MEDIATION RESULTS:")
    print(f"   Mediation ID: {evidence.mediation_id}")
    print(f"   Pathway type: {evidence.pathway_type.value}")
    print(f"   Mediation proportion: {evidence.mediation_proportion:.3f}")
    print(f"   Total effect: {evidence.total_effect:.3f}")
    print(f"   Direct effect: {evidence.direct_effect:.3f}")
    print(f"   Indirect effect: {evidence.indirect_effect:.3f}")
    print(f"   P-value (mediation): {evidence.p_value_mediation:.3f}")
    print(f"   Evidence strength: {evidence.evidence_strength}")

    print("\n⏰ TEMPORAL ALIGNMENT:")
    print(f"   Exposure window: {evidence.exposure_window.days} days")
    print(f"   Measurement count: {evidence.exposure_measurement_count}")
    print(f"   Temporal stability: {evidence.exposure_temporal_stability:.3f}")
    print(f"   Alignment quality: {result.temporal_alignment_quality:.3f}")

    print("\n🎯 VALIDATION METRICS:")
    print(f"   Sample size: {evidence.sample_size}")
    print(f"   Cross-validation score: {result.validation_metrics['cv_score']:.3f}")
    print(f"   Stability score: {result.validation_metrics['stability_score']:.3f}")
    print(f"   Bootstrap std: {result.validation_metrics['bootstrap_std']:.3f}")

    print("\n🔗 PATHWAY INFORMATION:")
    print(f"   Effect direction: {evidence.effect_direction}")
    print(
        f"   Exposure → Mediator effect: {evidence.molecular_to_functional_effect:.3f}"
    )
    print(f"   Mediator → Outcome effect: {evidence.functional_to_clinical_effect:.3f}")

    if result.exposure_uncertainty_propagation:
        print("\n📊 UNCERTAINTY ANALYSIS:")
        uncertainty = result.exposure_uncertainty_propagation
        print(f"   Temporal uncertainty: {uncertainty['temporal_uncertainty']:.3f}")
        print(
            f"   Measurement uncertainty: {uncertainty['measurement_uncertainty']:.3f}"
        )
        print(
            f"   Combined uncertainty factor: {uncertainty['combined_uncertainty_factor']:.3f}"
        )
        print(
            f"   Uncertainty-adjusted mediation: {uncertainty['uncertainty_adjusted_mediation_proportion']:.3f}"
        )

    print("\n🧬 MECHANISM HYPOTHESIS:")
    print(f"   {result.mechanism_hypothesis}")

    print("\n🎯 CLINICAL IMPLICATIONS:")
    print(f"   Intervention potential: {result.intervention_potential}")
    print(f"   Translational readiness: {result.translational_readiness}")
    print(f"   Source reliability: {evidence.exposure_source_reliability}")

    # Show model diagnostics
    print("\n📋 MODEL DIAGNOSTICS:")
    diagnostics = result.model_diagnostics
    for metric, value in diagnostics.items():
        print(f"   {metric}: {value:.3f}")

    print("\n✅ Exposure-mediation analysis demonstration complete!")
    print("\nKey capabilities demonstrated:")
    print("  • Temporal alignment of exposure and biomarker data")
    print("  • Uncertainty propagation from exposure measurements")
    print("  • Bootstrap validation with correlation structure preservation")
    print("  • Exposure-specific evidence assessment")
    print("  • Intervention potential evaluation")

    return result


if __name__ == "__main__":
    if not BASE_MEDIATION_AVAILABLE:
        print(
            "Warning: Base mediation framework not available. Run demo with simplified implementation."
        )

    run_exposure_mediation_demo()
