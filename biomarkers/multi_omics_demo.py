"""
Multi-omics Integration for Causal Biomarker Discovery - Demonstration

This module integrates proteomics, metabolomics, genomics, and clinical data
for comprehensive causal biomarker analysis using Graph Neural Networks.

Author: AI Pipeline Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List
from dataclasses import dataclass
import logging

# Scientific computing
from sklearn.preprocessing import StandardScaler

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OmicsDataConfig:
    """Configuration for omics data integration"""

    data_type: str  # 'proteomics', 'metabolomics', 'genomics', 'clinical'
    feature_prefix: str  # Prefix for feature names
    normalization_method: str = "standard"  # 'standard', 'robust', 'quantile'
    missing_threshold: float = 0.3  # Maximum fraction of missing values
    variance_threshold: float = 0.01  # Minimum variance for feature selection


class MultiOmicsDataGenerator:
    """Generate realistic multi-omics demonstration data"""

    def __init__(self, sample_size: int = 500):
        self.sample_size = sample_size
        np.random.seed(42)

    def generate_proteomics_data(self) -> pd.DataFrame:
        """Generate realistic proteomics data"""
        logger.info("Generating proteomics data...")

        # AKI-related proteins
        proteins = [
            "NGAL_LCN2",
            "KIM1_HAVCR1",
            "CYSTC_CST3",
            "UMOD_UMOD",
            "CLUSTERIN_CLU",
            "OSTEOPONTIN_SPP1",
            "VEGF_VEGFA",
            "PDGF_PDGFA",
            "IL6_IL6",
            "TNF_TNF",
            "CRP_CRP",
            "ALBUMIN_ALB",
            "TRANSFERRIN_TF",
            "HAPTOGLOBIN_HP",
            "FIBRINOGEN_FGA",
        ]

        # Generate correlated protein expression data
        n_proteins = len(proteins)
        correlation_structure = np.random.exponential(0.3, (n_proteins, n_proteins))
        correlation_structure = (correlation_structure + correlation_structure.T) / 2
        np.fill_diagonal(correlation_structure, 1.0)

        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_structure)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_structure = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Generate data with realistic protein expression patterns
        data = np.random.multivariate_normal(
            mean=np.random.uniform(2, 8, n_proteins),  # Log2 expression range
            cov=correlation_structure,
            size=self.sample_size,
        )

        # Add some biological realism - AKI biomarkers higher in some subjects
        for i, protein in enumerate(proteins):
            if "NGAL" in protein or "KIM1" in protein:
                aki_subjects = np.random.choice(
                    self.sample_size, size=self.sample_size // 4, replace=False
                )
                data[aki_subjects, i] += np.random.exponential(2, len(aki_subjects))

        df = pd.DataFrame(data, columns=[f"protein_{p}" for p in proteins])
        df = df.set_axis([f"subject_{i:04d}" for i in range(self.sample_size)], axis=0)

        logger.info(
            f"Generated proteomics data: {df.shape[0]} subjects, {df.shape[1]} proteins"
        )
        return df

    def generate_metabolomics_data(self) -> pd.DataFrame:
        """Generate realistic metabolomics data"""
        logger.info("Generating metabolomics data...")

        # Kidney-related metabolites
        metabolites = [
            "creatinine",
            "urea",
            "cystatin_c",
            "indoxyl_sulfate",
            "p_cresyl_sulfate",
            "hippuric_acid",
            "xanthine",
            "hypoxanthine",
            "uric_acid",
            "allantoin",
            "trimethylamine_oxide",
            "betaine",
            "carnitine",
            "acetylcarnitine",
            "glucose",
            "lactate",
        ]

        n_metabolites = len(metabolites)

        # Create pathway-based correlation structure
        correlation_matrix = np.eye(n_metabolites) * 0.1 + np.random.exponential(
            0.1, (n_metabolites, n_metabolites)
        )
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

        # Make positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Generate data with realistic concentration ranges
        mean_concentrations = np.random.lognormal(0, 1, n_metabolites)
        data = np.random.multivariate_normal(
            mean=mean_concentrations, cov=correlation_matrix, size=self.sample_size
        )

        # Ensure positive concentrations
        data = np.abs(data)

        df = pd.DataFrame(data, columns=[f"metabolite_{m}" for m in metabolites])
        df = df.set_axis([f"subject_{i:04d}" for i in range(self.sample_size)], axis=0)

        logger.info(
            f"Generated metabolomics data: {df.shape[0]} subjects, {df.shape[1]} metabolites"
        )
        return df

    def generate_genomics_data(self) -> pd.DataFrame:
        """Generate realistic genomics data (SNPs and gene expression)"""
        logger.info("Generating genomics data...")

        # AKI-related genes and SNPs
        genetic_variants = [
            "APOL1_G1",
            "APOL1_G2",
            "UMOD_rs4293393",
            "UMOD_rs13333226",
            "SHROOM3_rs17319721",
            "DAB2_rs11959928",
            "SLC34A1_rs3812036",
            "expr_NGAL",
            "expr_KIM1",
            "expr_CYSTC",
            "expr_UMOD",
        ]

        n_features = len(genetic_variants)
        data = np.zeros((self.sample_size, n_features))

        # Generate SNP data (0, 1, 2 for genotypes)
        for i in range(7):  # First 7 are SNPs
            maf = np.random.uniform(0.05, 0.45)  # Minor allele frequency
            genotype_probs = [(1 - maf) ** 2, 2 * maf * (1 - maf), maf**2]
            data[:, i] = np.random.choice(
                [0, 1, 2], size=self.sample_size, p=genotype_probs
            )

        # Generate gene expression data (log2 normalized)
        for i in range(7, n_features):
            base_expression = np.random.normal(
                5, 2, self.sample_size
            )  # Log2 expression
            # Add some correlation with SNPs
            snp_effect = data[:, i % 7] * np.random.normal(0, 0.5, self.sample_size)
            data[:, i] = base_expression + snp_effect

        df = pd.DataFrame(data, columns=[f"genetic_{v}" for v in genetic_variants])
        df = df.set_axis([f"subject_{i:04d}" for i in range(self.sample_size)], axis=0)

        logger.info(
            f"Generated genomics data: {df.shape[0]} subjects, {df.shape[1]} genetic features"
        )
        return df

    def generate_clinical_data(self) -> pd.DataFrame:
        """Generate clinical biomarker data"""
        logger.info("Generating clinical data...")

        # Clinical biomarkers
        clinical_features = [
            "creatinine_admission",
            "creatinine_peak",
            "urea_peak",
            "potassium_max",
            "sodium_min",
            "chloride_max",
            "hemoglobin_min",
            "platelet_min",
            "wbc_max",
        ]

        n_features = len(clinical_features)

        # Generate realistic clinical ranges
        feature_ranges = {
            "creatinine": (0.5, 5.0),
            "urea": (10, 100),
            "potassium": (3.0, 6.0),
            "sodium": (130, 150),
            "chloride": (95, 110),
            "hemoglobin": (8, 16),
            "platelet": (100, 400),
            "wbc": (4, 20),
        }

        data = np.zeros((self.sample_size, n_features))

        for i, feature in enumerate(clinical_features):
            base_name = feature.split("_")[0]
            if base_name in feature_ranges:
                min_val, max_val = feature_ranges[base_name]
                data[:, i] = np.random.uniform(min_val, max_val, self.sample_size)

        # Add some AKI patients with elevated creatinine
        aki_subjects = np.random.choice(
            self.sample_size, size=self.sample_size // 4, replace=False
        )
        creat_indices = [
            i for i, f in enumerate(clinical_features) if "creatinine" in f
        ]
        for idx in creat_indices:
            data[aki_subjects, idx] *= np.random.uniform(2, 5, len(aki_subjects))

        df = pd.DataFrame(data, columns=[f"clinical_{f}" for f in clinical_features])
        df = df.set_axis([f"subject_{i:04d}" for i in range(self.sample_size)], axis=0)

        logger.info(
            f"Generated clinical data: {df.shape[0]} subjects, {df.shape[1]} clinical features"
        )
        return df


class MultiOmicsIntegrator:
    """Integrate and analyze multi-omics data"""

    def __init__(self, configs: List[OmicsDataConfig]):
        self.configs = {config.data_type: config for config in configs}
        self.datasets = {}
        self.harmonized_data = None
        self.causal_graph = None
        self.feature_metadata = None

    def load_datasets(self, sample_size: int = 500):
        """Load all omics datasets"""
        logger.info("Loading multi-omics datasets...")

        generator = MultiOmicsDataGenerator(sample_size)

        # Generate each omics type
        for config in self.configs.values():
            if config.data_type == "proteomics":
                self.datasets[config.data_type] = generator.generate_proteomics_data()
            elif config.data_type == "metabolomics":
                self.datasets[config.data_type] = generator.generate_metabolomics_data()
            elif config.data_type == "genomics":
                self.datasets[config.data_type] = generator.generate_genomics_data()
            elif config.data_type == "clinical":
                self.datasets[config.data_type] = generator.generate_clinical_data()

        logger.info(f"Loaded {len(self.datasets)} omics datasets")
        return self.datasets

    def harmonize_data(self):
        """Harmonize multi-omics datasets"""
        logger.info("Harmonizing multi-omics datasets...")

        if not self.datasets:
            raise ValueError("Must load datasets first")

        # Find common subjects
        common_subjects = None
        for data_type, df in self.datasets.items():
            if common_subjects is None:
                common_subjects = set(df.index)
            else:
                common_subjects = common_subjects.intersection(set(df.index))

        common_subjects = sorted(list(common_subjects))
        logger.info(f"Found {len(common_subjects)} common subjects")

        # Harmonize and concatenate datasets
        harmonized_datasets = []
        feature_metadata = []

        for data_type, df in self.datasets.items():
            # Subset to common subjects
            df_subset = df.loc[common_subjects]

            # Handle missing values
            df_subset = df_subset.fillna(df_subset.median())

            # Normalize
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(df_subset)
            df_normalized = pd.DataFrame(
                normalized_data, index=df_subset.index, columns=df_subset.columns
            )

            harmonized_datasets.append(df_normalized)

            # Track feature metadata
            for feature in df_normalized.columns:
                feature_metadata.append(
                    {
                        "feature_name": feature,
                        "data_type": data_type,
                        "original_name": feature.replace(f"{data_type}_", ""),
                    }
                )

        # Concatenate all datasets
        self.harmonized_data = pd.concat(harmonized_datasets, axis=1)
        self.feature_metadata = pd.DataFrame(feature_metadata)

        logger.info(f"Harmonized data shape: {self.harmonized_data.shape}")
        return self.harmonized_data

    def discover_causal_relationships(self):
        """Discover causal relationships across omics"""
        logger.info("Discovering cross-omics causal relationships...")

        if self.harmonized_data is None:
            raise ValueError("Must harmonize data first")

        # Create causal graph using correlation-based approach
        self.causal_graph = nx.DiGraph()
        features = list(self.harmonized_data.columns)
        self.causal_graph.add_nodes_from(features)

        # Add edges based on correlation and biological hierarchy
        correlation_matrix = self.harmonized_data.corr()

        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i != j:
                    correlation = abs(correlation_matrix.iloc[i, j])
                    if correlation > 0.3:  # Correlation threshold

                        # Check biological plausibility
                        source_omics = self._get_omics_type(feature1)
                        target_omics = self._get_omics_type(feature2)

                        if self._is_biologically_plausible(source_omics, target_omics):
                            self.causal_graph.add_edge(
                                feature1,
                                feature2,
                                weight=float(correlation),
                                confidence=float(correlation),
                            )

        logger.info(
            f"Discovered causal graph: {self.causal_graph.number_of_nodes()} nodes, {self.causal_graph.number_of_edges()} edges"
        )
        return self.causal_graph

    def _get_omics_type(self, feature_name: str) -> str:
        """Get omics type for a feature"""
        for prefix in ["genetic_", "protein_", "metabolite_", "clinical_"]:
            if feature_name.startswith(prefix):
                return prefix.rstrip("_")
        return "unknown"

    def _is_biologically_plausible(self, source_omics: str, target_omics: str) -> bool:
        """Check if causal relationship is biologically plausible"""
        # Hierarchy: genetic -> protein -> metabolite -> clinical
        hierarchy = {"genetic": 0, "protein": 1, "metabolite": 2, "clinical": 3}

        source_level = hierarchy.get(source_omics, -1)
        target_level = hierarchy.get(target_omics, -1)

        # Allow forward flow in hierarchy or within same level
        return source_level >= 0 and target_level >= 0 and source_level <= target_level


class MultiOmicsGNN(nn.Module):
    """Graph Neural Network for multi-omics analysis"""

    def __init__(self, num_subjects: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()

        self.num_subjects = num_subjects
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_subjects, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Output projection
        x = self.output(x)
        return x


class MultiOmicsAnalyzer:
    """Complete multi-omics causal biomarker analysis"""

    def __init__(self, configs: List[OmicsDataConfig]):
        self.integrator = MultiOmicsIntegrator(configs)
        self.gnn = None
        self.embeddings = None

    def run_complete_analysis(self, sample_size: int = 500):
        """Run complete multi-omics analysis"""
        logger.info("=== Multi-Omics Causal Biomarker Analysis ===")

        # Load and harmonize data
        datasets = self.integrator.load_datasets(sample_size)
        harmonized_data = self.integrator.harmonize_data()

        # Discover causal relationships
        causal_graph = self.integrator.discover_causal_relationships()

        # Build and train GNN
        self._build_gnn(harmonized_data, causal_graph)
        self._train_gnn(harmonized_data, causal_graph)

        # Analyze results
        results = self._analyze_results()

        return results

    def _build_gnn(self, data: pd.DataFrame, graph: nx.DiGraph):
        """Build GNN for multi-omics analysis"""
        logger.info("Building multi-omics GNN...")

        num_subjects = data.shape[0]
        self.gnn = MultiOmicsGNN(num_subjects, hidden_dim=64, num_layers=3)

    def _train_gnn(self, data: pd.DataFrame, graph: nx.DiGraph, epochs: int = 50):
        """Train GNN on multi-omics data"""
        logger.info(f"Training GNN for {epochs} epochs...")

        # Prepare data - features as rows (nodes), subjects as feature dimensions
        x = torch.FloatTensor(
            data.values
        )  # [num_subjects x num_features] -> transpose for [num_features x num_subjects]
        x = x.T  # Now [num_features x num_subjects]

        # Convert graph to edge index
        edges = list(graph.edges())
        if edges:
            # Create mapping from feature names to indices
            feature_to_idx = {feature: idx for idx, feature in enumerate(data.columns)}

            # Convert edge names to indices
            edge_indices = []
            for source, target in edges:
                if source in feature_to_idx and target in feature_to_idx:
                    edge_indices.append(
                        [feature_to_idx[source], feature_to_idx[target]]
                    )

            if edge_indices:
                edge_index = torch.LongTensor(edge_indices).t().contiguous()
            else:
                # Create a simple chain if no valid edges
                edge_index = torch.LongTensor(
                    [[i, i + 1] for i in range(data.shape[1] - 1)]
                ).t()
        else:
            # Create a simple chain if no edges
            edge_index = torch.LongTensor(
                [[i, i + 1] for i in range(data.shape[1] - 1)]
            ).t()

        # Training setup
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.01)

        self.gnn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            embeddings = self.gnn(x, edge_index)

            # Simple reconstruction loss
            reconstructed = torch.matmul(embeddings, embeddings.t())
            target = torch.matmul(x, x.t())
            loss = F.mse_loss(reconstructed, target)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Store final embeddings
        self.gnn.eval()
        with torch.no_grad():
            self.embeddings = self.gnn(x, edge_index)

        logger.info("GNN training completed")

    def _analyze_results(self) -> Dict:
        """Analyze multi-omics results"""
        logger.info("Analyzing multi-omics results...")

        results = {
            "data_summary": self._summarize_data(),
            "causal_graph_analysis": self._analyze_causal_graph(),
            "embedding_analysis": self._analyze_embeddings(),
            "top_biomarkers": self._identify_top_biomarkers(),
        }

        return results

    def _summarize_data(self) -> Dict:
        """Summarize loaded data"""
        summary = {
            "total_subjects": self.integrator.harmonized_data.shape[0],
            "total_features": self.integrator.harmonized_data.shape[1],
        }

        # Features by omics type
        feature_counts = (
            self.integrator.feature_metadata["data_type"].value_counts().to_dict()
        )
        summary["features_by_omics"] = feature_counts

        return summary

    def _analyze_causal_graph(self) -> Dict:
        """Analyze causal graph structure"""
        graph = self.integrator.causal_graph

        analysis = {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "density": nx.density(graph),
        }

        # Cross-omics edges
        cross_omics_edges = 0
        for edge in graph.edges():
            source_omics = self.integrator._get_omics_type(edge[0])
            target_omics = self.integrator._get_omics_type(edge[1])
            if source_omics != target_omics:
                cross_omics_edges += 1

        analysis["cross_omics_edges"] = cross_omics_edges

        return analysis

    def _analyze_embeddings(self) -> Dict:
        """Analyze learned embeddings"""
        if self.embeddings is None:
            return {}

        embed_np = self.embeddings.detach().numpy()

        analysis = {
            "embedding_shape": embed_np.shape,
            "mean_norm": float(np.mean(np.linalg.norm(embed_np, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embed_np, axis=1))),
        }

        # Pairwise similarities
        similarities = np.corrcoef(embed_np)
        analysis["mean_similarity"] = float(
            np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        )

        return analysis

    def _identify_top_biomarkers(self) -> List[Dict]:
        """Identify top biomarkers based on graph centrality"""
        graph = self.integrator.causal_graph

        # Calculate centrality measures
        centrality = nx.degree_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)

        biomarker_scores = []
        for node in graph.nodes():
            score = centrality.get(node, 0) * 0.5 + betweenness.get(node, 0) * 0.5

            biomarker_scores.append(
                {
                    "biomarker": node,
                    "omics_type": self.integrator._get_omics_type(node),
                    "centrality_score": score,
                    "degree": graph.degree(node),
                }
            )

        # Sort by centrality score
        biomarker_scores.sort(key=lambda x: x["centrality_score"], reverse=True)

        return biomarker_scores[:10]


def run_multi_omics_demonstration():
    """Run complete multi-omics demonstration"""

    # Configure omics data types
    configs = [
        OmicsDataConfig("proteomics", "protein_", "standard", 0.3, 0.01),
        OmicsDataConfig("metabolomics", "metabolite_", "standard", 0.3, 0.01),
        OmicsDataConfig("genomics", "genetic_", "standard", 0.1, 0.01),
        OmicsDataConfig("clinical", "clinical_", "standard", 0.2, 0.01),
    ]

    # Run analysis
    analyzer = MultiOmicsAnalyzer(configs)
    results = analyzer.run_complete_analysis(sample_size=500)

    # Display results
    print("\n" + "=" * 60)
    print("MULTI-OMICS CAUSAL BIOMARKER ANALYSIS RESULTS")
    print("=" * 60)

    # Data summary
    print("\nData Summary:")
    print(f"  Total subjects: {results['data_summary']['total_subjects']}")
    print(f"  Total features: {results['data_summary']['total_features']}")

    for omics_type, count in results["data_summary"]["features_by_omics"].items():
        print(f"  {omics_type.capitalize()}: {count} features")

    # Causal graph analysis
    graph_stats = results["causal_graph_analysis"]
    print("\nCausal Graph Analysis:")
    print(f"  Nodes: {graph_stats['total_nodes']}")
    print(f"  Edges: {graph_stats['total_edges']}")
    print(f"  Cross-omics edges: {graph_stats['cross_omics_edges']}")
    print(f"  Density: {graph_stats['density']:.4f}")

    # Embedding analysis
    if results["embedding_analysis"]:
        embed_stats = results["embedding_analysis"]
        print("\nGNN Embedding Analysis:")
        print(f"  Embedding shape: {embed_stats['embedding_shape']}")
        print(f"  Mean similarity: {embed_stats['mean_similarity']:.4f}")

    # Top biomarkers
    print("\nTop Multi-Omics Biomarkers:")
    for i, biomarker in enumerate(results["top_biomarkers"][:5], 1):
        print(f"  {i}. {biomarker['biomarker']} ({biomarker['omics_type']})")
        print(
            f"     Centrality: {biomarker['centrality_score']:.3f}, Degree: {biomarker['degree']}"
        )

    print("\n" + "=" * 60)
    print("MULTI-OMICS ANALYSIS COMPLETE")
    print("=" * 60)
    print("✅ Successfully integrated 4 omics types")
    print("✅ Discovered cross-omics causal relationships")
    print("✅ Trained Graph Neural Networks")
    print("✅ Identified prioritized biomarkers")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = run_multi_omics_demonstration()
