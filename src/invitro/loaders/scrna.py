"""
Single-cell RNA sequencing (scRNA-seq) assay loader.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from .schema import AssaySchema, AssayType, ValidationError


class scRNALoader:
    """Loader for single-cell RNA sequencing data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/processed/invitro/scrna")
        self.schema = AssaySchema()
    
    def load_raw_data(self, file_path: Path, 
                     metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load raw scRNA-seq data from various formats"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"scRNA-seq data file not found: {file_path}")
        
        # Support multiple file formats
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.h5':
            # For scanpy/h5ad files (simplified)
            raise NotImplementedError("H5AD format support requires scanpy integration")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self._standardize_columns(df, metadata)
    
    def _standardize_columns(self, df: pd.DataFrame, 
                           metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Standardize column names and add required metadata"""
        
        # Common column mappings for scRNA-seq data
        column_mappings = {
            # Expression measurements
            'expression': 'value',
            'log2_tpm': 'value',
            'log_expression': 'value',
            'normalized_counts': 'value',
            'tpm': 'value',
            'fpkm': 'value',
            'counts': 'value',
            
            # Gene identifiers
            'gene_symbol': 'gene',
            'gene_name': 'gene',
            'symbol': 'gene',
            'ensembl_id': 'gene',
            
            # Cell identifiers
            'cell_id': 'sample_id',
            'barcode': 'sample_id',
            'cell_barcode': 'sample_id',
            'sample': 'sample_id',
            
            # Conditions
            'treatment': 'condition',
            'compound': 'condition',
            'stimulus': 'condition',
            'perturbation': 'condition',
            'timepoint_condition': 'condition',
            
            # Time
            'time': 'timepoint',
            'time_hours': 'timepoint',
            'hours': 'timepoint',
            'collection_time': 'timepoint',
            
            # Cell type annotations
            'cell_type_annotation': 'cell_type',
            'cluster': 'cell_type',
            'leiden': 'cell_type',
            'seurat_clusters': 'cell_type',
        }
        
        # Apply mappings
        df_mapped = df.rename(columns=column_mappings)
        
        # Add required columns if missing
        if 'assay_type' not in df_mapped.columns:
            df_mapped['assay_type'] = 'scrna'
        
        if 'cell_type' not in df_mapped.columns:
            # Try to infer from metadata or use default
            default_cell_type = metadata.get('cell_type', 'unknown') if metadata else 'unknown'
            df_mapped['cell_type'] = default_cell_type
        
        if 'condition' not in df_mapped.columns:
            df_mapped['condition'] = metadata.get('condition', 'control') if metadata else 'control'
        
        if 'timepoint' not in df_mapped.columns:
            df_mapped['timepoint'] = metadata.get('timepoint', 24.0) if metadata else 24.0
        
        if 'error' not in df_mapped.columns:
            # For scRNA-seq, error represents technical noise
            df_mapped['error'] = 0.1  # Placeholder for dropout/noise model
        
        return df_mapped
    
    def process_data(self, df: pd.DataFrame, 
                    source_units: str = "log2_tpm",
                    apply_qc: bool = True) -> pd.DataFrame:
        """Process and validate scRNA-seq data"""
        
        # Normalize units
        df_processed = self.schema.normalize_units(df, AssayType.SCRNA, source_units)
        
        # Apply quality control filters
        if apply_qc:
            df_processed = self.schema.apply_quality_filters(df_processed, AssayType.SCRNA)
        
        # Additional scRNA-seq specific QC
        df_processed = self._apply_scrna_qc(df_processed)
        
        # Validate against schema
        self.schema.validate_dataframe(df_processed, AssayType.SCRNA)
        
        return df_processed
    
    def _apply_scrna_qc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scRNA-seq specific quality control"""
        
        # Remove lowly expressed genes (present in < 3 cells)
        gene_counts = df.groupby('gene')['sample_id'].nunique()
        expressed_genes = gene_counts[gene_counts >= 3].index
        df_filtered = df[df['gene'].isin(expressed_genes)].copy()
        
        # Remove low-quality cells (expressing < 200 genes)
        cell_gene_counts = df_filtered.groupby('sample_id')['gene'].nunique()
        quality_cells = cell_gene_counts[cell_gene_counts >= 200].index
        df_filtered = df_filtered[df_filtered['sample_id'].isin(quality_cells)].copy()
        
        # Flag potential doublets (cells with very high gene counts)
        high_gene_threshold = float(cell_gene_counts.quantile(0.98))
        doublet_cells = cell_gene_counts[cell_gene_counts > high_gene_threshold].index
        df_filtered['doublet_flag'] = df_filtered['sample_id'].isin(doublet_cells)
        
        # Calculate mitochondrial gene percentage (simplified)
        mito_genes = df_filtered[df_filtered['gene'].str.startswith('MT-')]['gene'].unique()
        if len(mito_genes) > 0:
            mito_expression = df_filtered[df_filtered['gene'].isin(mito_genes)].groupby('sample_id')['value'].sum()
            total_expression = df_filtered.groupby('sample_id')['value'].sum()
            mito_pct = (mito_expression / total_expression * 100).fillna(0)
            
            # Flag cells with high mitochondrial content (> 20%)
            high_mito_cells = mito_pct[mito_pct > 20].index
            df_filtered['high_mito_flag'] = df_filtered['sample_id'].isin(high_mito_cells)
        else:
            df_filtered['high_mito_flag'] = False
        
        return df_filtered
    
    def calculate_functional_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gene-level functional impact scores from scRNA-seq"""
        
        scores = []
        
        for gene in df['gene'].unique():
            gene_data = df[df['gene'] == gene]
            
            # Get baseline (control) expression
            baseline_data = gene_data[gene_data['condition'] == 'control']
            baseline_mean = baseline_data['value'].mean() if not baseline_data.empty else 0
            
            for condition in gene_data['condition'].unique():
                if condition == 'control':
                    continue
                
                condition_data = gene_data[gene_data['condition'] == condition]
                mean_val = condition_data['value'].mean()
                
                # Calculate fold change and effect size
                fold_change = 2**(mean_val - baseline_mean) if baseline_mean > 0 else 1.0
                effect_size = mean_val - baseline_mean  # Log2 fold change
                
                # Calculate significance using t-test approximation
                n_treatment = len(condition_data)
                n_control = len(baseline_data)
                
                if n_treatment > 1 and n_control > 1:
                    pooled_std = np.sqrt(
                        ((n_treatment - 1) * condition_data['value'].var() + 
                         (n_control - 1) * baseline_data['value'].var()) / 
                        (n_treatment + n_control - 2)
                    )
                    se_diff = pooled_std * np.sqrt(1/n_treatment + 1/n_control)
                    t_stat = (mean_val - baseline_mean) / se_diff if se_diff > 0 else 0
                    df_free = n_treatment + n_control - 2
                    p_value = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + np.sqrt(df_free))) if df_free > 0 else 0.5
                else:
                    p_value = 0.5
                
                # Calculate expression frequency (fraction of cells expressing gene)
                expressing_fraction = (condition_data['value'] > 0.1).mean()
                baseline_expressing = (baseline_data['value'] > 0.1).mean() if not baseline_data.empty else 0
                
                # Classify expression pattern
                if effect_size > 1 and expressing_fraction > 0.1:
                    expression_pattern = 'upregulated'
                elif effect_size < -1 and expressing_fraction < baseline_expressing * 0.5:
                    expression_pattern = 'downregulated'
                else:
                    expression_pattern = 'stable'
                
                scores.append({
                    'gene': gene,
                    'condition': condition,
                    'function': 'gene_expression',
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'n_cells_treatment': n_treatment,
                    'n_cells_control': n_control,
                    'mean_expression': mean_val,
                    'baseline_expression': baseline_mean,
                    'fold_change': fold_change,
                    'expressing_fraction': expressing_fraction,
                    'baseline_expressing_fraction': baseline_expressing,
                    'expression_pattern': expression_pattern
                })
        
        return pd.DataFrame(scores)
    
    def identify_cell_type_markers(self, df: pd.DataFrame, 
                                  min_log_fc: float = 0.5,
                                  min_expressing_fraction: float = 0.25) -> Dict[str, List[str]]:
        """Identify cell type-specific marker genes"""
        
        markers = {}
        
        for cell_type in df['cell_type'].unique():
            if cell_type == 'unknown':
                continue
            
            cell_type_data = df[df['cell_type'] == cell_type]
            other_data = df[df['cell_type'] != cell_type]
            
            type_markers = []
            
            for gene in df['gene'].unique():
                type_expr = cell_type_data[cell_type_data['gene'] == gene]['value']
                other_expr = other_data[other_data['gene'] == gene]['value']
                
                if len(type_expr) == 0 or len(other_expr) == 0:
                    continue
                
                # Calculate differential expression
                log_fc = type_expr.mean() - other_expr.mean()
                expressing_frac = (type_expr > 0.1).mean()
                
                # Check criteria for marker genes
                if (log_fc >= min_log_fc and 
                    expressing_frac >= min_expressing_fraction):
                    type_markers.append(gene)
            
            markers[cell_type] = type_markers[:50]  # Top 50 markers
        
        return markers
    
    def analyze_pathway_activity(self, df: pd.DataFrame, 
                                pathway_genes: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Analyze pathway activity from gene expression"""
        
        if pathway_genes is None:
            # Default kidney-relevant pathways
            pathway_genes = {
                'tubular_transport': ['SLC34A1', 'SLC12A3', 'AQP2', 'UMOD'],
                'injury_response': ['HAVCR1', 'LCN2', 'IL18', 'CCL2'],
                'fibrosis': ['COL1A1', 'ACTA2', 'FN1', 'TGFB1'],
                'inflammation': ['IL1B', 'TNF', 'IL6', 'CXCL8']
            }
        
        pathway_scores = []
        
        for (sample_id, condition), cell_data in df.groupby(['sample_id', 'condition']):
            for pathway, genes in pathway_genes.items():
                # Calculate pathway score as mean expression of pathway genes
                pathway_expr = cell_data[cell_data['gene'].isin(genes)]
                
                if not pathway_expr.empty:
                    pathway_score = pathway_expr['value'].mean()
                    n_genes_detected = len(pathway_expr['gene'].unique())
                else:
                    pathway_score = 0
                    n_genes_detected = 0
                
                pathway_scores.append({
                    'sample_id': sample_id,
                    'condition': condition,
                    'pathway': pathway,
                    'pathway_score': pathway_score,
                    'n_genes_detected': n_genes_detected,
                    'total_pathway_genes': len(genes)
                })
        
        return pd.DataFrame(pathway_scores)
    
    def load_and_process(self, file_path: Path, 
                        metadata: Optional[Dict[str, Any]] = None,
                        source_units: str = "log2_tpm") -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
        """Complete pipeline: load, process, and analyze"""
        
        # Load raw data
        df_raw = self.load_raw_data(file_path, metadata)
        
        # Process and validate
        df_processed = self.process_data(df_raw, source_units)
        
        # Calculate functional scores
        df_scores = self.calculate_functional_scores(df_processed)
        
        # Identify cell type markers
        cell_type_markers = self.identify_cell_type_markers(df_processed)
        
        # Analyze pathway activity
        df_pathways = self.analyze_pathway_activity(df_processed)
        
        return df_processed, df_scores, cell_type_markers, df_pathways
    
    def save_processed_data(self, df_processed: pd.DataFrame, 
                           df_scores: pd.DataFrame,
                           cell_type_markers: Dict[str, List[str]],
                           df_pathways: pd.DataFrame,
                           output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save processed data to standard locations"""
        
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed expression data
        processed_path = output_dir / "scrna_processed.parquet"
        df_processed.to_parquet(processed_path, index=False)
        
        # Save functional scores
        scores_path = output_dir / "scrna_functional_scores.parquet"
        df_scores.to_parquet(scores_path, index=False)
        
        # Save pathway analysis
        pathways_path = output_dir / "pathway_activity.parquet"
        if not df_pathways.empty:
            df_pathways.to_parquet(pathways_path, index=False)
        
        # Save cell type markers as JSON
        import json
        markers_path = output_dir / "cell_type_markers.json"
        with open(markers_path, 'w') as f:
            json.dump(cell_type_markers, f, indent=2)
        
        return {
            'processed': processed_path,
            'scores': scores_path,
            'pathways': pathways_path,
            'markers': markers_path
        }
