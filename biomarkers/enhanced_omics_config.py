"""
Enhanced Multi-Omics Configuration for 6-Omics Integration

This module provides enhanced configuration classes supporting:
- Epigenomics data (methylation, histone modifications, chromatin accessibility)
- Exposomics data (environmental exposures, lifestyle factors)
- Temporal resolution and environmental context
- Extended biological hierarchy validation

Author: AI Pipeline Team
Date: September 2025
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OmicsType(Enum):
    """Enhanced omics data types including epigenomics and exposomics"""

    GENOMICS = "genomics"
    EPIGENOMICS = "epigenomics"
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    EXPOSOMICS = "exposomics"
    CLINICAL = "clinical"


class EpigenomicsSubType(Enum):
    """Sub-types of epigenomics data"""

    DNA_METHYLATION = "dna_methylation"
    HISTONE_MODIFICATIONS = "histone_modifications"
    CHROMATIN_ACCESSIBILITY = "chromatin_accessibility"
    CHROMATIN_3D_STRUCTURE = "chromatin_3d_structure"
    NON_CODING_RNA = "non_coding_rna"


class ExposomicsSubType(Enum):
    """Sub-types of exposomics data"""

    AIR_QUALITY = "air_quality"
    CHEMICAL_EXPOSURES = "chemical_exposures"
    BUILT_ENVIRONMENT = "built_environment"
    LIFESTYLE_EXPOSURES = "lifestyle_exposures"
    OCCUPATIONAL_EXPOSURES = "occupational_exposures"
    CLIMATE_FACTORS = "climate_factors"


class TemporalResolution(Enum):
    """Temporal resolution options for longitudinal data"""

    STATIC = "static"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class NormalizationMethod(Enum):
    """Normalization methods for different omics types"""

    STANDARD = "standard"
    ROBUST = "robust"
    QUANTILE = "quantile"
    LOG_TRANSFORM = "log_transform"
    BETA_VALUE = "beta_value"  # For methylation data
    M_VALUE = "m_value"  # For methylation data
    TPM = "tpm"  # For transcriptomics
    FPKM = "fpkm"  # For transcriptomics
    VSN = "vsn"  # For proteomics


@dataclass
class EnhancedOmicsDataConfig:
    """Enhanced configuration for omics data integration supporting 6-omics types"""

    # Basic configuration
    data_type: OmicsType
    feature_prefix: str

    # Sub-type specification for complex omics
    sub_types: List[Union[EpigenomicsSubType, ExposomicsSubType]] = field(
        default_factory=list
    )

    # Data processing parameters
    normalization_method: NormalizationMethod = NormalizationMethod.STANDARD
    missing_threshold: float = 0.3
    variance_threshold: float = 0.01

    # Temporal and environmental context
    temporal_resolution: TemporalResolution = TemporalResolution.STATIC
    environmental_context: bool = False
    geographic_resolution: Optional[str] = (
        None  # "zip_code", "county", "state", "coordinates"
    )

    # Biological constraints
    pathway_informed: bool = True
    tissue_specificity: List[str] = field(default_factory=list)

    # Data quality parameters
    min_sample_size: int = 10
    max_missing_rate: float = 0.5
    batch_effect_correction: bool = True

    # Validation parameters
    cross_validation_folds: int = 5
    validation_strategy: str = "temporal"  # "random", "temporal", "spatial"

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""

        # Validate sub-types match data type
        if self.data_type == OmicsType.EPIGENOMICS:
            if not all(isinstance(st, EpigenomicsSubType) for st in self.sub_types):
                raise ValueError(
                    "Epigenomics data type requires EpigenomicsSubType sub-types"
                )

        elif self.data_type == OmicsType.EXPOSOMICS:
            if not all(isinstance(st, ExposomicsSubType) for st in self.sub_types):
                raise ValueError(
                    "Exposomics data type requires ExposomicsSubType sub-types"
                )

        # Validate thresholds
        if not 0 <= self.missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")

        if not 0 <= self.max_missing_rate <= 1:
            raise ValueError("max_missing_rate must be between 0 and 1")

        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")

        # Validate temporal resolution for static data types
        if (
            self.data_type == OmicsType.GENOMICS
            and self.temporal_resolution != TemporalResolution.STATIC
        ):
            logger.warning(
                "Genomics data is typically static - temporal resolution may not be meaningful"
            )

        # Validate environmental context
        if self.environmental_context and self.geographic_resolution is None:
            logger.warning(
                "Environmental context enabled but no geographic resolution specified"
            )

    def get_biological_hierarchy_level(self) -> int:
        """Get biological hierarchy level for causal discovery constraints"""
        hierarchy = {
            OmicsType.GENOMICS: 0,
            OmicsType.EPIGENOMICS: 1,
            OmicsType.TRANSCRIPTOMICS: 2,
            OmicsType.PROTEOMICS: 3,
            OmicsType.METABOLOMICS: 4,
            OmicsType.CLINICAL: 5,
            OmicsType.EXPOSOMICS: -1,  # Special: can influence any level
        }
        return hierarchy[self.data_type]

    def is_temporal_data(self) -> bool:
        """Check if this data type supports temporal resolution"""
        return self.temporal_resolution != TemporalResolution.STATIC

    def get_recommended_normalization(self) -> NormalizationMethod:
        """Get recommended normalization method for data type"""
        recommendations = {
            OmicsType.GENOMICS: NormalizationMethod.STANDARD,
            OmicsType.EPIGENOMICS: NormalizationMethod.BETA_VALUE,
            OmicsType.TRANSCRIPTOMICS: NormalizationMethod.TPM,
            OmicsType.PROTEOMICS: NormalizationMethod.VSN,
            OmicsType.METABOLOMICS: NormalizationMethod.LOG_TRANSFORM,
            OmicsType.EXPOSOMICS: NormalizationMethod.ROBUST,
            OmicsType.CLINICAL: NormalizationMethod.STANDARD,
        }
        return recommendations.get(self.data_type, NormalizationMethod.STANDARD)


@dataclass
class EpigenomicsDataConfig(EnhancedOmicsDataConfig):
    """Specialized configuration for epigenomics data"""

    # Epigenomics-specific parameters
    methylation_platform: Optional[str] = None  # "450K", "EPIC", "WGBS"
    chip_seq_antibodies: List[str] = field(
        default_factory=list
    )  # ["H3K4me3", "H3K27me3"]
    peak_calling_method: str = "MACS2"

    # Analysis parameters
    dmr_detection: bool = True  # Differentially methylated regions
    chromatin_states: bool = True  # ChromHMM states
    transcription_factor_binding: bool = True  # TFBS prediction

    def __post_init__(self):
        # Set defaults for epigenomics
        if not hasattr(self, "data_type") or self.data_type != OmicsType.EPIGENOMICS:
            self.data_type = OmicsType.EPIGENOMICS

        if not self.sub_types:
            self.sub_types = [EpigenomicsSubType.DNA_METHYLATION]

        if self.normalization_method == NormalizationMethod.STANDARD:
            self.normalization_method = NormalizationMethod.BETA_VALUE

        super().__post_init__()


@dataclass
class ExposomicsDataConfig(EnhancedOmicsDataConfig):
    """Specialized configuration for exposomics data"""

    # Exposomics-specific parameters
    data_sources: List[str] = field(
        default_factory=list
    )  # ["EPA_AQS", "NHANES", "GIS"]
    exposure_windows: List[str] = field(
        default_factory=list
    )  # ["acute", "chronic", "lifetime"]
    spatial_resolution: str = "zip_code"  # "address", "zip_code", "county", "state"

    # Environmental parameters
    pollutant_types: List[str] = field(default_factory=list)  # ["PM2.5", "NO2", "O3"]
    chemical_classes: List[str] = field(
        default_factory=list
    )  # ["PFAS", "metals", "pesticides"]
    lifestyle_factors: List[str] = field(
        default_factory=list
    )  # ["diet", "exercise", "sleep"]

    # Quality control
    measurement_uncertainty: bool = True
    spatial_interpolation: str = "kriging"  # "kriging", "idw", "nearest"
    temporal_aggregation: str = "daily_mean"  # "hourly", "daily_mean", "monthly_mean"

    def __post_init__(self):
        # Set defaults for exposomics
        if not hasattr(self, "data_type") or self.data_type != OmicsType.EXPOSOMICS:
            self.data_type = OmicsType.EXPOSOMICS

        if not self.sub_types:
            self.sub_types = [ExposomicsSubType.AIR_QUALITY]

        self.environmental_context = True  # Always true for exposomics

        if self.geographic_resolution is None:
            self.geographic_resolution = self.spatial_resolution

        if self.normalization_method == NormalizationMethod.STANDARD:
            self.normalization_method = NormalizationMethod.ROBUST

        super().__post_init__()


class Enhanced6OmicsConfigManager:
    """Manager class for handling 6-omics configuration sets"""

    def __init__(self):
        self.configs: Dict[OmicsType, EnhancedOmicsDataConfig] = {}
        self.temporal_alignment = True
        self.spatial_alignment = True

    def add_config(self, config: EnhancedOmicsDataConfig):
        """Add omics configuration"""
        self.configs[config.data_type] = config
        logger.info(f"Added {config.data_type.value} configuration")

    def get_config(self, data_type: OmicsType) -> Optional[EnhancedOmicsDataConfig]:
        """Get configuration for specific omics type"""
        return self.configs.get(data_type)

    def get_temporal_configs(self) -> List[EnhancedOmicsDataConfig]:
        """Get configurations that support temporal resolution"""
        return [config for config in self.configs.values() if config.is_temporal_data()]

    def get_environmental_configs(self) -> List[EnhancedOmicsDataConfig]:
        """Get configurations with environmental context"""
        return [
            config for config in self.configs.values() if config.environmental_context
        ]

    def validate_compatibility(self) -> Dict[str, Any]:
        """Validate compatibility across all configurations"""

        validation_results = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        # Check temporal alignment
        temporal_configs = self.get_temporal_configs()
        if len(temporal_configs) > 1:
            resolutions = [config.temporal_resolution for config in temporal_configs]
            if len(set(resolutions)) > 1:
                validation_results["warnings"].append(
                    "Multiple temporal resolutions detected - data alignment may be needed"
                )

        # Check geographic alignment
        env_configs = self.get_environmental_configs()
        if len(env_configs) > 1:
            geo_resolutions = [config.geographic_resolution for config in env_configs]
            if len(set(geo_resolutions)) > 1:
                validation_results["warnings"].append(
                    "Multiple geographic resolutions detected - spatial harmonization may be needed"
                )

        # Check normalization compatibility
        norm_methods = [config.normalization_method for config in self.configs.values()]
        if len(set(norm_methods)) == len(norm_methods):
            validation_results["recommendations"].append(
                "Consider standardizing normalization methods for better integration"
            )

        # Check sample size requirements
        min_samples = [config.min_sample_size for config in self.configs.values()]
        max_min_sample = max(min_samples) if min_samples else 0
        if max_min_sample > 50:
            validation_results["warnings"].append(
                f"High minimum sample size requirement: {max_min_sample}"
            )

        return validation_results

    def get_biological_hierarchy(self) -> Dict[OmicsType, int]:
        """Get biological hierarchy for all configured omics types"""
        return {
            omics_type: config.get_biological_hierarchy_level()
            for omics_type, config in self.configs.items()
        }

    def create_default_6omics_config(self) -> Dict[OmicsType, EnhancedOmicsDataConfig]:
        """Create default configuration for all 6 omics types"""

        configs = {
            OmicsType.GENOMICS: EnhancedOmicsDataConfig(
                data_type=OmicsType.GENOMICS,
                feature_prefix="genetic_",
                normalization_method=NormalizationMethod.STANDARD,
                missing_threshold=0.1,
                variance_threshold=0.01,
                pathway_informed=True,
            ),
            OmicsType.EPIGENOMICS: EpigenomicsDataConfig(
                feature_prefix="epigenetic_",
                sub_types=[EpigenomicsSubType.DNA_METHYLATION],
                methylation_platform="EPIC",
                missing_threshold=0.2,
                dmr_detection=True,
            ),
            OmicsType.TRANSCRIPTOMICS: EnhancedOmicsDataConfig(
                data_type=OmicsType.TRANSCRIPTOMICS,
                feature_prefix="transcript_",
                normalization_method=NormalizationMethod.TPM,
                missing_threshold=0.1,
                variance_threshold=0.01,
                pathway_informed=True,
            ),
            OmicsType.PROTEOMICS: EnhancedOmicsDataConfig(
                data_type=OmicsType.PROTEOMICS,
                feature_prefix="protein_",
                normalization_method=NormalizationMethod.VSN,
                missing_threshold=0.3,
                variance_threshold=0.01,
                pathway_informed=True,
            ),
            OmicsType.METABOLOMICS: EnhancedOmicsDataConfig(
                data_type=OmicsType.METABOLOMICS,
                feature_prefix="metabolite_",
                normalization_method=NormalizationMethod.LOG_TRANSFORM,
                missing_threshold=0.3,
                variance_threshold=0.01,
                pathway_informed=True,
            ),
            OmicsType.EXPOSOMICS: ExposomicsDataConfig(
                feature_prefix="exposure_",
                sub_types=[
                    ExposomicsSubType.AIR_QUALITY,
                    ExposomicsSubType.LIFESTYLE_EXPOSURES,
                ],
                temporal_resolution=TemporalResolution.DAILY,
                data_sources=["EPA_AQS", "wearables"],
                spatial_resolution="zip_code",
            ),
            OmicsType.CLINICAL: EnhancedOmicsDataConfig(
                data_type=OmicsType.CLINICAL,
                feature_prefix="clinical_",
                normalization_method=NormalizationMethod.STANDARD,
                missing_threshold=0.2,
                variance_threshold=0.01,
                pathway_informed=False,
            ),
        }

        # Add all configs to manager
        for config in configs.values():
            self.add_config(config)

        return configs


def create_kidney_disease_6omics_config() -> Enhanced6OmicsConfigManager:
    """Create specialized 6-omics configuration for kidney disease research"""

    manager = Enhanced6OmicsConfigManager()

    # Genomics - kidney disease specific
    genomics_config = EnhancedOmicsDataConfig(
        data_type=OmicsType.GENOMICS,
        feature_prefix="genetic_",
        normalization_method=NormalizationMethod.STANDARD,
        missing_threshold=0.05,  # Stricter for genomics
        tissue_specificity=["kidney", "urinary_system"],
        pathway_informed=True,
    )

    # Epigenomics - kidney-specific methylation patterns
    epigenomics_config = EpigenomicsDataConfig(
        feature_prefix="epigenetic_",
        sub_types=[
            EpigenomicsSubType.DNA_METHYLATION,
            EpigenomicsSubType.HISTONE_MODIFICATIONS,
        ],
        methylation_platform="EPIC",
        chip_seq_antibodies=["H3K4me3", "H3K27me3", "H3K9me3"],
        tissue_specificity=["kidney", "podocyte", "tubular_epithelial"],
        dmr_detection=True,
        chromatin_states=True,
    )

    # Transcriptomics - kidney gene expression
    transcriptomics_config = EnhancedOmicsDataConfig(
        data_type=OmicsType.TRANSCRIPTOMICS,
        feature_prefix="transcript_",
        normalization_method=NormalizationMethod.TPM,
        tissue_specificity=["kidney"],
        pathway_informed=True,
    )

    # Proteomics - kidney biomarkers
    proteomics_config = EnhancedOmicsDataConfig(
        data_type=OmicsType.PROTEOMICS,
        feature_prefix="protein_",
        normalization_method=NormalizationMethod.VSN,
        missing_threshold=0.3,
        tissue_specificity=["kidney", "urine", "serum"],
        pathway_informed=True,
    )

    # Metabolomics - kidney metabolism
    metabolomics_config = EnhancedOmicsDataConfig(
        data_type=OmicsType.METABOLOMICS,
        feature_prefix="metabolite_",
        normalization_method=NormalizationMethod.LOG_TRANSFORM,
        missing_threshold=0.3,
        tissue_specificity=["kidney", "urine", "plasma"],
        pathway_informed=True,
    )

    # Exposomics - environmental kidney disease risk factors
    exposomics_config = ExposomicsDataConfig(
        feature_prefix="exposure_",
        sub_types=[
            ExposomicsSubType.AIR_QUALITY,
            ExposomicsSubType.CHEMICAL_EXPOSURES,
            ExposomicsSubType.LIFESTYLE_EXPOSURES,
        ],
        temporal_resolution=TemporalResolution.DAILY,
        data_sources=["EPA_AQS", "NHANES", "wearables"],
        pollutant_types=["PM2.5", "NO2", "O3"],
        chemical_classes=["heavy_metals", "PFAS"],
        lifestyle_factors=["physical_activity", "diet_quality", "sleep"],
        spatial_resolution="zip_code",
    )

    # Clinical - kidney function measures
    clinical_config = EnhancedOmicsDataConfig(
        data_type=OmicsType.CLINICAL,
        feature_prefix="clinical_",
        normalization_method=NormalizationMethod.STANDARD,
        missing_threshold=0.2,
        temporal_resolution=TemporalResolution.MONTHLY,  # Clinical follow-up
        pathway_informed=False,
    )

    # Add all configurations
    for config in [
        genomics_config,
        epigenomics_config,
        transcriptomics_config,
        proteomics_config,
        metabolomics_config,
        exposomics_config,
        clinical_config,
    ]:
        manager.add_config(config)

    # Validate compatibility
    validation = manager.validate_compatibility()
    if not validation["compatible"]:
        logger.warning("Configuration compatibility issues detected")
        for error in validation["errors"]:
            logger.error(f"Configuration error: {error}")
        for warning in validation["warnings"]:
            logger.warning(f"Configuration warning: {warning}")

    return manager


if __name__ == "__main__":
    # Demonstrate enhanced 6-omics configuration
    logger.info("=== Enhanced 6-Omics Configuration Demo ===")

    # Create kidney disease specific configuration
    manager = create_kidney_disease_6omics_config()

    # Display configurations
    print(f"\nConfigured omics types: {list(manager.configs.keys())}")
    print(
        f"Temporal configs: {[c.data_type.value for c in manager.get_temporal_configs()]}"
    )
    print(
        f"Environmental configs: {[c.data_type.value for c in manager.get_environmental_configs()]}"
    )

    # Show biological hierarchy
    hierarchy = manager.get_biological_hierarchy()
    print("\nBiological hierarchy:")
    for omics_type, level in sorted(hierarchy.items(), key=lambda x: x[1]):
        print(f"  {omics_type.value}: level {level}")

    # Validation results
    validation = manager.validate_compatibility()
    print("\nConfiguration validation:")
    print(f"  Compatible: {validation['compatible']}")
    print(f"  Warnings: {len(validation['warnings'])}")
    print(f"  Recommendations: {len(validation['recommendations'])}")

    print("\n✅ Enhanced 6-omics configuration system ready for implementation")
