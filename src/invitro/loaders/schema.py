"""
Standardized assay schema definitions and validation utilities.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Raised when assay data fails validation"""
    pass


class AssayType(Enum):
    """Supported assay types"""
    TEER = "teer"
    PERMEABILITY = "permeability"
    SECRETOME = "secretome"
    IMAGING = "imaging"
    SCRNA = "scrna"


class Direction(Enum):
    """Effect direction interpretation"""
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class AssayMetadata:
    """Metadata for assay configuration"""
    assay_type: AssayType
    units: str
    direction: Direction
    cell_types: List[str]
    dynamic_range: tuple[float, float]
    detection_limit: Optional[float] = None
    temporal_resolution: Optional[str] = None  # e.g., "5min", "1hour"


class AssaySchema:
    """Standard schema definitions for all assay types"""
    
    REQUIRED_COLUMNS = {
        "sample_id", "gene", "condition", "timepoint", 
        "value", "error", "cell_type", "assay_type"
    }
    
    ASSAY_CONFIGS = {
        AssayType.TEER: AssayMetadata(
            assay_type=AssayType.TEER,
            units="ohm*cm2",
            direction=Direction.HIGHER_IS_BETTER,
            cell_types=["epithelial", "endothelial"],
            dynamic_range=(10.0, 10000.0),
            detection_limit=5.0,
            temporal_resolution="5min"
        ),
        AssayType.PERMEABILITY: AssayMetadata(
            assay_type=AssayType.PERMEABILITY,
            units="cm/s",
            direction=Direction.LOWER_IS_BETTER,
            cell_types=["epithelial", "endothelial"],
            dynamic_range=(1e-8, 1e-4),
            detection_limit=1e-9,
            temporal_resolution="30min"
        ),
        AssayType.SECRETOME: AssayMetadata(
            assay_type=AssayType.SECRETOME,
            units="pg/ml",
            direction=Direction.BIDIRECTIONAL,
            cell_types=["all"],
            dynamic_range=(0.1, 10000.0),
            detection_limit=0.05,
            temporal_resolution="1hour"
        ),
        AssayType.IMAGING: AssayMetadata(
            assay_type=AssayType.IMAGING,
            units="normalized_intensity",
            direction=Direction.BIDIRECTIONAL,
            cell_types=["all"],
            dynamic_range=(0.0, 1.0),
            detection_limit=0.01,
            temporal_resolution="10min"
        ),
        AssayType.SCRNA: AssayMetadata(
            assay_type=AssayType.SCRNA,
            units="log2_tpm",
            direction=Direction.BIDIRECTIONAL,
            cell_types=["all"],
            dynamic_range=(0.0, 15.0),
            detection_limit=0.1,
            temporal_resolution="endpoint"
        ),
    }
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, assay_type: AssayType) -> None:
        """Validate dataframe against schema requirements"""
        
        # Check required columns
        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
        
        # Get assay config
        config = cls.ASSAY_CONFIGS[assay_type]
        
        # Validate values within dynamic range
        min_val, max_val = config.dynamic_range
        out_of_range = df[(df['value'] < min_val) | (df['value'] > max_val)]
        if not out_of_range.empty:
            raise ValidationError(
                f"Values outside dynamic range [{min_val}, {max_val}]: "
                f"{len(out_of_range)} rows"
            )
        
        # Check for required data types
        if not pd.api.types.is_numeric_dtype(df['value']):
            raise ValidationError("'value' column must be numeric")
        
        if 'timepoint' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['timepoint']):
                raise ValidationError("'timepoint' column must be numeric")
    
    @classmethod
    def normalize_units(cls, df: pd.DataFrame, assay_type: AssayType, 
                       source_units: str) -> pd.DataFrame:
        """Normalize units to standard schema"""
        config = cls.ASSAY_CONFIGS[assay_type]
        target_units = config.units
        
        if source_units == target_units:
            return df
        
        # Unit conversion factors
        conversion_factors = {
            # TEER conversions
            ("ohm*cm2", "ohm*cm2"): 1.0,
            ("ohm", "ohm*cm2"): 1.0,  # Assume 1 cm2 if not specified
            
            # Permeability conversions  
            ("cm/s", "cm/s"): 1.0,
            ("mm/s", "cm/s"): 0.1,
            ("um/s", "cm/s"): 1e-4,
            
            # Secretome conversions
            ("pg/ml", "pg/ml"): 1.0,
            ("ng/ml", "pg/ml"): 1000.0,
            ("ug/ml", "pg/ml"): 1e6,
            
            # Imaging conversions
            ("normalized_intensity", "normalized_intensity"): 1.0,
            ("raw_intensity", "normalized_intensity"): 1.0,  # Will need normalization
            
            # scRNA conversions
            ("log2_tpm", "log2_tpm"): 1.0,
            ("tpm", "log2_tpm"): lambda x: np.log2(x + 1),
            ("counts", "log2_tpm"): lambda x: np.log2(x + 1),  # Simplified
        }
        
        key = (source_units, target_units)
        if key not in conversion_factors:
            raise ValidationError(f"No conversion from {source_units} to {target_units}")
        
        factor = conversion_factors[key]
        df_normalized = df.copy()
        
        if callable(factor):
            df_normalized['value'] = factor(df_normalized['value'])
        else:
            df_normalized['value'] = df_normalized['value'] * factor
        
        return df_normalized
    
    @classmethod
    def apply_quality_filters(cls, df: pd.DataFrame, assay_type: AssayType) -> pd.DataFrame:
        """Apply quality control filters"""
        config = cls.ASSAY_CONFIGS[assay_type]
        
        # Remove below detection limit
        if config.detection_limit is not None:
            df = df[df['value'] >= config.detection_limit]
        
        # Remove outliers (beyond 3 sigma from mean)
        mean_val = df['value'].mean()
        std_val = df['value'].std()
        df = df[np.abs(df['value'] - mean_val) <= 3 * std_val]
        
        return df
