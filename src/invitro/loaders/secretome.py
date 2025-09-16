"""
Secretome assay loader for protein secretion measurements.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from .schema import AssaySchema, AssayType, ValidationError


class SecretomeLoader:
    """Loader for secretome (protein secretion) measurement data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/processed/invitro/secretome")
        self.schema = AssaySchema()
    
    def load_raw_data(self, file_path: Path, 
                     metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load raw secretome data from various formats"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"Secretome data file not found: {file_path}")
        
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
        
        # Common column mappings for secretome data
        column_mappings = {
            # Protein concentration measurements
            'concentration': 'value',
            'conc': 'value',
            'pg_ml': 'value',
            'ng_ml': 'value',
            'protein_conc': 'value',
            'secreted_protein': 'value',
            
            # Protein identifiers
            'protein': 'gene',
            'protein_name': 'gene',
            'analyte': 'gene',
            'biomarker': 'gene',
            'target': 'gene',
            
            # Sample identifiers
            'well': 'sample_id',
            'well_id': 'sample_id',
            'sample': 'sample_id',
            'supernatant': 'sample_id',
            
            # Conditions
            'treatment': 'condition',
            'compound': 'condition',
            'stimulus': 'condition',
            'drug': 'condition',
            'intervention': 'condition',
            
            # Time
            'time': 'timepoint',
            'time_hours': 'timepoint',
            'hours': 'timepoint',
            'collection_time': 'timepoint',
            
            # Error/uncertainty
            'std': 'error',
            'stderr': 'error',
            'sem': 'error',
            'sd': 'error',
            'cv': 'error',
        }
        
        # Apply mappings
        df_mapped = df.rename(columns=column_mappings)
        
        # Add required columns if missing
        if 'assay_type' not in df_mapped.columns:
            df_mapped['assay_type'] = 'secretome'
        
        if 'cell_type' not in df_mapped.columns:
            # Default based on metadata or common secretome cell types
            default_cell_type = metadata.get('cell_type', 'mixed') if metadata else 'mixed'
            df_mapped['cell_type'] = default_cell_type
        
        if 'condition' not in df_mapped.columns:
            df_mapped['condition'] = 'control'
        
        if 'timepoint' not in df_mapped.columns:
            df_mapped['timepoint'] = 24.0  # Default 24h collection
        
        if 'error' not in df_mapped.columns:
            # Estimate error as fraction of value if not provided
            df_mapped['error'] = df_mapped['value'] * 0.2  # Secretome has high variability
        
        return df_mapped
    
    def process_data(self, df: pd.DataFrame, 
                    source_units: str = "pg/ml",
                    apply_qc: bool = True) -> pd.DataFrame:
        """Process and validate secretome data"""
        
        # Normalize units
        df_processed = self.schema.normalize_units(df, AssayType.SECRETOME, source_units)
        
        # Apply quality control filters
        if apply_qc:
            df_processed = self.schema.apply_quality_filters(df_processed, AssayType.SECRETOME)
        
        # Additional secretome-specific QC
        df_processed = self._apply_secretome_qc(df_processed)
        
        # Validate against schema
        self.schema.validate_dataframe(df_processed, AssayType.SECRETOME)
        
        return df_processed
    
    def _apply_secretome_qc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply secretome-specific quality control"""
        
        # Remove below detection limit values (typical ELISA sensitivity)
        detection_limit = 0.1  # pg/ml
        df_filtered = df[df['value'] >= detection_limit].copy()
        
        # Flag very high concentrations that might be outliers
        for protein in df_filtered['gene'].unique():
            protein_data = df_filtered[df_filtered['gene'] == protein]
            q95 = protein_data['value'].quantile(0.95)
            q05 = protein_data['value'].quantile(0.05)
            
            # Flag extreme outliers
            df_filtered.loc[
                (df_filtered['gene'] == protein) & 
                ((df_filtered['value'] > q95 * 10) | (df_filtered['value'] < q05 * 0.1)),
                'outlier_flag'
            ] = True
        
        if 'outlier_flag' not in df_filtered.columns:
            df_filtered['outlier_flag'] = False
        
        return df_filtered
    
    def calculate_functional_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gene-level functional impact scores"""
        
        scores = []
        
        for protein in df['gene'].unique():
            protein_data = df[df['gene'] == protein]
            
            # Get baseline (control) values
            baseline_data = protein_data[protein_data['condition'] == 'control']
            baseline_mean = baseline_data['value'].mean() if not baseline_data.empty else 0
            
            for condition in protein_data['condition'].unique():
                if condition == 'control':
                    continue
                
                condition_data = protein_data[protein_data['condition'] == condition]
                mean_val = condition_data['value'].mean()
                
                # Calculate fold change
                fold_change = mean_val / baseline_mean if baseline_mean > 0 else 1.0
                effect_size = np.log2(fold_change) if fold_change > 0 else 0
                
                # Calculate significance (t-test approximation)
                n = len(condition_data)
                sem = condition_data['value'].std() / np.sqrt(n) if n > 1 else 0.1
                t_stat = (mean_val - baseline_mean) / sem if sem > 0 else 0
                p_value = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + np.sqrt(n - 1))) if n > 1 else 0.5
                
                # Classify secretion pattern
                if fold_change > 2:
                    secretion_pattern = 'upregulated'
                elif fold_change < 0.5:
                    secretion_pattern = 'downregulated'
                else:
                    secretion_pattern = 'stable'
                
                # Additional metrics
                outliers = condition_data['outlier_flag'].sum() if 'outlier_flag' in condition_data.columns else 0
                
                scores.append({
                    'gene': protein,
                    'condition': condition,
                    'function': 'protein_secretion',
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'n_replicates': n,
                    'mean_value': mean_val,
                    'baseline_value': baseline_mean,
                    'fold_change': fold_change,
                    'secretion_pattern': secretion_pattern,
                    'outlier_count': outliers
                })
        
        return pd.DataFrame(scores)
    
    def analyze_secretion_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze multi-protein secretion profiles"""
        
        profiles = []
        
        # Group by sample and condition
        for (sample_id, condition), group in df.groupby(['sample_id', 'condition']):
            if len(group) < 2:  # Need multiple proteins
                continue
            
            # Calculate profile metrics
            total_secretion = group['value'].sum()
            protein_count = len(group)
            dominant_protein = group.loc[group['value'].idxmax(), 'gene']
            dominant_fraction = group['value'].max() / total_secretion if total_secretion > 0 else 0
            
            # Calculate diversity (Shannon entropy)
            proportions = group['value'] / total_secretion
            shannon_entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            
            profiles.append({
                'sample_id': sample_id,
                'condition': condition,
                'total_secretion': total_secretion,
                'protein_count': protein_count,
                'dominant_protein': dominant_protein,
                'dominant_fraction': dominant_fraction,
                'shannon_entropy': shannon_entropy,
                'diversity_index': shannon_entropy / np.log2(protein_count) if protein_count > 1 else 0
            })
        
        return pd.DataFrame(profiles)
    
    def identify_biomarker_panels(self, df_scores: pd.DataFrame, 
                                 min_effect_size: float = 1.0,
                                 max_p_value: float = 0.05) -> Dict[str, List[str]]:
        """Identify condition-specific biomarker panels"""
        
        panels = {}
        
        for condition in df_scores['condition'].unique():
            condition_scores = df_scores[df_scores['condition'] == condition]
            
            # Filter by significance and effect size
            significant = condition_scores[
                (np.abs(condition_scores['effect_size']) >= min_effect_size) &
                (condition_scores['p_value'] <= max_p_value)
            ]
            
            # Rank by effect size magnitude
            ranked = significant.reindex(
                significant['effect_size'].abs().sort_values(ascending=False).index
            )
            
            panels[condition] = ranked['gene'].tolist()
        
        return panels
    
    def load_and_process(self, file_path: Path, 
                        metadata: Optional[Dict[str, Any]] = None,
                        source_units: str = "pg/ml") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """Complete pipeline: load, process, and analyze"""
        
        # Load raw data
        df_raw = self.load_raw_data(file_path, metadata)
        
        # Process and validate
        df_processed = self.process_data(df_raw, source_units)
        
        # Calculate functional scores
        df_scores = self.calculate_functional_scores(df_processed)
        
        # Analyze secretion profiles
        df_profiles = self.analyze_secretion_profiles(df_processed)
        
        # Identify biomarker panels
        biomarker_panels = self.identify_biomarker_panels(df_scores)
        
        return df_processed, df_scores, df_profiles, biomarker_panels
    
    def save_processed_data(self, df_processed: pd.DataFrame, 
                           df_scores: pd.DataFrame,
                           df_profiles: pd.DataFrame,
                           biomarker_panels: Dict[str, List[str]],
                           output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save processed data to standard locations"""
        
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed measurements
        processed_path = output_dir / "secretome_processed.parquet"
        df_processed.to_parquet(processed_path, index=False)
        
        # Save functional scores
        scores_path = output_dir / "secretome_functional_scores.parquet"
        df_scores.to_parquet(scores_path, index=False)
        
        # Save secretion profiles
        profiles_path = output_dir / "secretome_profiles.parquet"
        if not df_profiles.empty:
            df_profiles.to_parquet(profiles_path, index=False)
        
        # Save biomarker panels as JSON
        import json
        panels_path = output_dir / "biomarker_panels.json"
        with open(panels_path, 'w') as f:
            json.dump(biomarker_panels, f, indent=2)
        
        return {
            'processed': processed_path,
            'scores': scores_path,
            'profiles': profiles_path,
            'panels': panels_path
        }
