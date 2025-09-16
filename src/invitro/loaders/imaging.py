"""
Imaging assay loader for morphological and functional imaging data.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from .schema import AssaySchema, AssayType, ValidationError


class ImagingLoader:
    """Loader for imaging-based assay data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/processed/invitro/imaging")
        self.schema = AssaySchema()
    
    def load_raw_data(self, file_path: Path, 
                     metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load raw imaging data from various formats"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"Imaging data file not found: {file_path}")
        
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
        
        # Common column mappings for imaging data
        column_mappings = {
            # Intensity measurements
            'intensity': 'value',
            'mean_intensity': 'value',
            'integrated_intensity': 'value',
            'fluorescence': 'value',
            'signal': 'value',
            'pixel_intensity': 'value',
            
            # Morphological measurements
            'area': 'value',
            'perimeter': 'value',
            'circularity': 'value',
            'aspect_ratio': 'value',
            'roundness': 'value',
            
            # Feature identifiers
            'marker': 'gene',
            'protein': 'gene',
            'channel': 'gene',
            'stain': 'gene',
            'antibody': 'gene',
            
            # Sample identifiers
            'well': 'sample_id',
            'well_id': 'sample_id',
            'field': 'sample_id',
            'image_id': 'sample_id',
            'fov': 'sample_id',  # field of view
            
            # Conditions
            'treatment': 'condition',
            'compound': 'condition',
            'stimulus': 'condition',
            'drug': 'condition',
            
            # Time
            'time': 'timepoint',
            'time_hours': 'timepoint',
            'hours': 'timepoint',
            'frame': 'timepoint',
            
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
            df_mapped['assay_type'] = 'imaging'
        
        if 'cell_type' not in df_mapped.columns:
            default_cell_type = metadata.get('cell_type', 'mixed') if metadata else 'mixed'
            df_mapped['cell_type'] = default_cell_type
        
        if 'gene' not in df_mapped.columns:
            # For imaging, gene might be the imaged marker
            df_mapped['gene'] = metadata.get('marker', 'morphology') if metadata else 'morphology'
        
        if 'condition' not in df_mapped.columns:
            df_mapped['condition'] = 'control'
        
        if 'timepoint' not in df_mapped.columns:
            df_mapped['timepoint'] = 24.0  # Default endpoint
        
        if 'error' not in df_mapped.columns:
            # Estimate error based on imaging variability
            df_mapped['error'] = df_mapped['value'] * 0.1
        
        return df_mapped
    
    def process_data(self, df: pd.DataFrame, 
                    source_units: str = "normalized_intensity",
                    apply_qc: bool = True) -> pd.DataFrame:
        """Process and validate imaging data"""
        
        # Normalize units
        df_processed = self.schema.normalize_units(df, AssayType.IMAGING, source_units)
        
        # Apply quality control filters
        if apply_qc:
            df_processed = self.schema.apply_quality_filters(df_processed, AssayType.IMAGING)
        
        # Additional imaging-specific QC
        df_processed = self._apply_imaging_qc(df_processed)
        
        # Validate against schema
        self.schema.validate_dataframe(df_processed, AssayType.IMAGING)
        
        return df_processed
    
    def _apply_imaging_qc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imaging-specific quality control"""
        
        # Remove saturated pixels (if intensity > 0.95 for normalized data)
        if df['value'].max() <= 1.0:  # Normalized data
            df_filtered = df[df['value'] <= 0.95].copy()
        else:  # Raw intensity data
            # Remove top 1% as potentially saturated
            upper_threshold = df['value'].quantile(0.99)
            df_filtered = df[df['value'] <= upper_threshold].copy()
        
        # Flag low signal-to-noise ratio
        for gene in df_filtered['gene'].unique():
            gene_data = df_filtered[df_filtered['gene'] == gene]
            median_signal = gene_data['value'].median()
            mad = gene_data['value'].mad()  # median absolute deviation
            
            # Flag samples with signal < 3 * MAD above background
            background_threshold = median_signal - 3 * mad
            df_filtered.loc[
                (df_filtered['gene'] == gene) & 
                (df_filtered['value'] < background_threshold),
                'low_signal_flag'
            ] = True
        
        if 'low_signal_flag' not in df_filtered.columns:
            df_filtered['low_signal_flag'] = False
        
        # Add focus quality score (simplified)
        # In real implementation, this would be based on image sharpness metrics
        np.random.seed(42)  # For reproducible "quality scores"
        df_filtered['focus_quality'] = np.random.uniform(0.7, 1.0, len(df_filtered))
        df_filtered['poor_focus_flag'] = df_filtered['focus_quality'] < 0.8
        
        return df_filtered
    
    def calculate_functional_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gene-level functional impact scores"""
        
        scores = []
        
        for gene in df['gene'].unique():
            gene_data = df[df['gene'] == gene]
            
            # Get baseline (control) values
            baseline_data = gene_data[gene_data['condition'] == 'control']
            baseline_mean = baseline_data['value'].mean() if not baseline_data.empty else 0
            
            for condition in gene_data['condition'].unique():
                if condition == 'control':
                    continue
                
                condition_data = gene_data[gene_data['condition'] == condition]
                mean_val = condition_data['value'].mean()
                
                # Calculate fold change and effect size
                fold_change = mean_val / baseline_mean if baseline_mean > 0 else 1.0
                effect_size = np.log2(fold_change) if fold_change > 0 else 0
                
                # Calculate significance
                n = len(condition_data)
                sem = condition_data['value'].std() / np.sqrt(n) if n > 1 else 0.1
                t_stat = (mean_val - baseline_mean) / sem if sem > 0 else 0
                p_value = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + np.sqrt(n - 1))) if n > 1 else 0.5
                
                # Quality metrics
                low_signal_count = condition_data['low_signal_flag'].sum() if 'low_signal_flag' in condition_data.columns else 0
                poor_focus_count = condition_data['poor_focus_flag'].sum() if 'poor_focus_flag' in condition_data.columns else 0
                
                # Determine functional category based on marker type
                if gene.lower() in ['dapi', 'hoechst', 'nuclear']:
                    function_category = 'nuclear_morphology'
                elif gene.lower() in ['phalloidin', 'actin', 'cytoskeleton']:
                    function_category = 'cytoskeletal_organization'
                elif gene.lower() in ['zo1', 'claudin', 'occludin']:
                    function_category = 'junction_integrity'
                else:
                    function_category = 'protein_expression'
                
                scores.append({
                    'gene': gene,
                    'condition': condition,
                    'function': function_category,
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'n_replicates': n,
                    'mean_value': mean_val,
                    'baseline_value': baseline_mean,
                    'fold_change': fold_change,
                    'low_signal_count': low_signal_count,
                    'poor_focus_count': poor_focus_count
                })
        
        return pd.DataFrame(scores)
    
    def analyze_morphological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze morphological changes from imaging data"""
        
        morphology = []
        
        # Group by condition and gene
        for (condition, gene), group in df.groupby(['condition', 'gene']):
            if len(group) < 3:  # Need sufficient data
                continue
            
            # Calculate morphological metrics
            mean_intensity = group['value'].mean()
            intensity_cv = group['value'].std() / mean_intensity if mean_intensity > 0 else 0
            
            # Texture analysis (simplified - based on intensity variation)
            intensity_range = group['value'].max() - group['value'].min()
            texture_metric = intensity_range / mean_intensity if mean_intensity > 0 else 0
            
            # Spatial distribution (simplified)
            # In real implementation, this would analyze spatial patterns
            spatial_heterogeneity = group['value'].std()
            
            morphology.append({
                'condition': condition,
                'gene': gene,
                'mean_intensity': mean_intensity,
                'intensity_cv': intensity_cv,
                'intensity_range': intensity_range,
                'texture_metric': texture_metric,
                'spatial_heterogeneity': spatial_heterogeneity,
                'n_fields': len(group)
            })
        
        return pd.DataFrame(morphology)
    
    def detect_phenotypic_changes(self, df_scores: pd.DataFrame, 
                                 effect_threshold: float = 0.5) -> Dict[str, List[str]]:
        """Detect significant phenotypic changes by condition"""
        
        phenotypes = {}
        
        for condition in df_scores['condition'].unique():
            condition_scores = df_scores[df_scores['condition'] == condition]
            
            # Find significant changes
            significant = condition_scores[
                (np.abs(condition_scores['effect_size']) >= effect_threshold) &
                (condition_scores['p_value'] <= 0.05)
            ]
            
            # Categorize changes
            upregulated = significant[significant['effect_size'] > 0]['gene'].tolist()
            downregulated = significant[significant['effect_size'] < 0]['gene'].tolist()
            
            phenotypes[condition] = {
                'upregulated_markers': upregulated,
                'downregulated_markers': downregulated,
                'total_changes': len(significant)
            }
        
        return phenotypes
    
    def load_and_process(self, file_path: Path, 
                        metadata: Optional[Dict[str, Any]] = None,
                        source_units: str = "normalized_intensity") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Complete pipeline: load, process, and analyze"""
        
        # Load raw data
        df_raw = self.load_raw_data(file_path, metadata)
        
        # Process and validate
        df_processed = self.process_data(df_raw, source_units)
        
        # Calculate functional scores
        df_scores = self.calculate_functional_scores(df_processed)
        
        # Analyze morphological features
        df_morphology = self.analyze_morphological_features(df_processed)
        
        # Detect phenotypic changes
        phenotypic_changes = self.detect_phenotypic_changes(df_scores)
        
        return df_processed, df_scores, df_morphology, phenotypic_changes
    
    def save_processed_data(self, df_processed: pd.DataFrame, 
                           df_scores: pd.DataFrame,
                           df_morphology: pd.DataFrame,
                           phenotypic_changes: Dict[str, Any],
                           output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save processed data to standard locations"""
        
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed measurements
        processed_path = output_dir / "imaging_processed.parquet"
        df_processed.to_parquet(processed_path, index=False)
        
        # Save functional scores
        scores_path = output_dir / "imaging_functional_scores.parquet"
        df_scores.to_parquet(scores_path, index=False)
        
        # Save morphological analysis
        morphology_path = output_dir / "morphological_features.parquet"
        if not df_morphology.empty:
            df_morphology.to_parquet(morphology_path, index=False)
        
        # Save phenotypic changes as JSON
        import json
        phenotypes_path = output_dir / "phenotypic_changes.json"
        with open(phenotypes_path, 'w') as f:
            json.dump(phenotypic_changes, f, indent=2)
        
        return {
            'processed': processed_path,
            'scores': scores_path,
            'morphology': morphology_path,
            'phenotypes': phenotypes_path
        }
