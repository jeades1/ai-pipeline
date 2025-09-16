"""
TEER (Trans-Epithelial Electrical Resistance) assay loader.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from .schema import AssaySchema, AssayType, ValidationError


class TEERLoader:
    """Loader for TEER measurement data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/processed/invitro/teer")
        self.schema = AssaySchema()
    
    def load_raw_data(self, file_path: Path, 
                     metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load raw TEER data from various formats"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"TEER data file not found: {file_path}")
        
        # Support multiple file formats
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self._standardize_columns(df, metadata)
    
    def _standardize_columns(self, df: pd.DataFrame, 
                           metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Standardize column names and add required metadata"""
        
        # Common column mappings for TEER data
        column_mappings = {
            # Standard variations
            'resistance': 'value',
            'teer': 'value', 
            'ohms': 'value',
            'resistance_ohm_cm2': 'value',
            
            # Sample identifiers
            'well': 'sample_id',
            'well_id': 'sample_id',
            'sample': 'sample_id',
            
            # Conditions
            'treatment': 'condition',
            'compound': 'condition',
            'stimulus': 'condition',
            
            # Time
            'time': 'timepoint',
            'time_hours': 'timepoint',
            'hours': 'timepoint',
            
            # Error/uncertainty
            'std': 'error',
            'stderr': 'error',
            'sem': 'error',
            'sd': 'error',
        }
        
        # Apply mappings
        df_mapped = df.rename(columns=column_mappings)
        
        # Add required columns if missing
        if 'assay_type' not in df_mapped.columns:
            df_mapped['assay_type'] = 'teer'
        
        if 'cell_type' not in df_mapped.columns:
            # Try to infer from metadata or use default
            default_cell_type = metadata.get('cell_type', 'epithelial') if metadata else 'epithelial'
            df_mapped['cell_type'] = default_cell_type
        
        if 'gene' not in df_mapped.columns:
            # For functional assays, gene might be perturbation target
            df_mapped['gene'] = metadata.get('target_gene', 'control') if metadata else 'control'
        
        if 'condition' not in df_mapped.columns:
            df_mapped['condition'] = 'control'
        
        if 'timepoint' not in df_mapped.columns:
            df_mapped['timepoint'] = 24.0  # Default 24h endpoint
        
        if 'error' not in df_mapped.columns:
            # Estimate error as fraction of value if not provided
            df_mapped['error'] = df_mapped['value'] * 0.1
        
        return df_mapped
    
    def process_data(self, df: pd.DataFrame, 
                    source_units: str = "ohm*cm2",
                    apply_qc: bool = True) -> pd.DataFrame:
        """Process and validate TEER data"""
        
        # Normalize units
        df_processed = self.schema.normalize_units(df, AssayType.TEER, source_units)
        
        # Apply quality control filters
        if apply_qc:
            df_processed = self.schema.apply_quality_filters(df_processed, AssayType.TEER)
        
        # Validate against schema
        self.schema.validate_dataframe(df_processed, AssayType.TEER)
        
        return df_processed
    
    def calculate_functional_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gene-level functional impact scores"""
        
        # Group by gene and condition to calculate effects
        baseline = df[df['condition'] == 'control'].groupby('gene')['value'].mean()
        
        scores = []
        for gene in df['gene'].unique():
            gene_data = df[df['gene'] == gene]
            
            for condition in gene_data['condition'].unique():
                if condition == 'control':
                    continue
                
                condition_data = gene_data[gene_data['condition'] == condition]
                mean_val = condition_data['value'].mean()
                baseline_val = baseline.get(gene, mean_val)
                
                # Calculate fold change (higher TEER = better barrier function)
                fold_change = mean_val / baseline_val if baseline_val > 0 else 1.0
                effect_size = np.log2(fold_change)
                
                # Calculate significance (t-test approximation)
                n = len(condition_data)
                sem = condition_data['value'].std() / np.sqrt(n) if n > 1 else 0.1
                t_stat = (mean_val - baseline_val) / sem if sem > 0 else 0
                p_value = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + np.sqrt(n - 1))) if n > 1 else 0.5
                
                scores.append({
                    'gene': gene,
                    'condition': condition,
                    'function': 'barrier_integrity',
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'n_replicates': n,
                    'mean_value': mean_val,
                    'baseline_value': baseline_val
                })
        
        return pd.DataFrame(scores)
    
    def load_and_process(self, file_path: Path, 
                        metadata: Optional[Dict[str, Any]] = None,
                        source_units: str = "ohm*cm2") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Complete pipeline: load, process, and calculate scores"""
        
        # Load raw data
        df_raw = self.load_raw_data(file_path, metadata)
        
        # Process and validate
        df_processed = self.process_data(df_raw, source_units)
        
        # Calculate functional scores
        df_scores = self.calculate_functional_scores(df_processed)
        
        return df_processed, df_scores
    
    def save_processed_data(self, df_processed: pd.DataFrame, 
                           df_scores: pd.DataFrame,
                           output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save processed data to standard locations"""
        
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed measurements
        processed_path = output_dir / "teer_processed.parquet"
        df_processed.to_parquet(processed_path, index=False)
        
        # Save functional scores
        scores_path = output_dir / "teer_functional_scores.parquet"
        df_scores.to_parquet(scores_path, index=False)
        
        return {
            'processed': processed_path,
            'scores': scores_path
        }
