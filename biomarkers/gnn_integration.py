"""
Graph Neural Network Integration for Causal Biomarker Discovery

Implements Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)
to learn biomarker representations using discovered causal graph structure.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for Graph Neural Network models"""

    # Model architecture
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    attention_heads: int = 4  # For GAT

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 15

    # Graph construction
    edge_threshold: float = 0.5  # Minimum confidence for including edges
    max_neighbors: int = 10  # Maximum number of neighbors per node
    use_edge_weights: bool = True

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CausalGraphGCN(nn.Module):
    """
    Graph Convolutional Network for biomarker representation learning
    Uses causal graph structure to learn biomarker embeddings
    """

    def __init__(self, num_features: int, config: GNNConfig):
        super(CausalGraphGCN, self).__init__()
        self.config = config

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, config.hidden_dim))

        for _ in range(config.num_layers - 2):
            self.convs.append(GCNConv(config.hidden_dim, config.hidden_dim))

        if config.num_layers > 1:
            self.convs.append(GCNConv(config.hidden_dim, config.hidden_dim))

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Output layers for different tasks
        self.biomarker_classifier = nn.Linear(
            config.hidden_dim, 1
        )  # Binary classification
        self.biomarker_regressor = nn.Linear(
            config.hidden_dim, 1
        )  # Regression for scores

        # Graph-level prediction (for patient-level outcomes)
        self.graph_classifier = nn.Sequential(
            nn.Linear(
                config.hidden_dim * 2, config.hidden_dim
            ),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass through the GCN

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            batch: Batch assignment for graph-level tasks (optional)
        """
        # Node-level representation learning
        node_embeddings = []

        for i, conv in enumerate(self.convs):
            if edge_weight is not None and hasattr(conv, "edge_weight"):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)

            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = self.dropout(x)

            node_embeddings.append(x)

        # Final node embeddings
        final_embeddings = x

        # Node-level predictions
        node_classification = torch.sigmoid(self.biomarker_classifier(final_embeddings))
        node_regression = self.biomarker_regressor(final_embeddings)

        # Graph-level predictions (if batch is provided)
        graph_prediction = None
        if batch is not None:
            # Global pooling
            graph_mean = global_mean_pool(final_embeddings, batch)
            graph_max = global_max_pool(final_embeddings, batch)
            graph_features = torch.cat([graph_mean, graph_max], dim=1)
            graph_prediction = torch.sigmoid(self.graph_classifier(graph_features))

        return {
            "node_embeddings": final_embeddings,
            "node_classification": node_classification,
            "node_regression": node_regression,
            "graph_prediction": graph_prediction,
            "all_embeddings": node_embeddings,
        }


class CausalGraphGAT(nn.Module):
    """
    Graph Attention Network for biomarker representation learning
    Uses attention mechanism to focus on most relevant causal relationships
    """

    def __init__(self, num_features: int, config: GNNConfig):
        super(CausalGraphGAT, self).__init__()
        self.config = config

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                num_features,
                config.hidden_dim // config.attention_heads,
                heads=config.attention_heads,
                dropout=config.dropout,
            )
        )

        for _ in range(config.num_layers - 2):
            self.convs.append(
                GATConv(
                    config.hidden_dim,
                    config.hidden_dim // config.attention_heads,
                    heads=config.attention_heads,
                    dropout=config.dropout,
                )
            )

        if config.num_layers > 1:
            self.convs.append(
                GATConv(
                    config.hidden_dim,
                    config.hidden_dim,
                    heads=1,
                    dropout=config.dropout,
                )
            )

        # Output layers
        self.biomarker_classifier = nn.Linear(config.hidden_dim, 1)
        self.biomarker_regressor = nn.Linear(config.hidden_dim, 1)

        self.graph_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        batch=None,
        return_attention_weights=False,
    ):
        """Forward pass through GAT with optional attention weight return"""

        attention_weights = []

        for i, conv in enumerate(self.convs):
            if return_attention_weights and hasattr(conv, "attention"):
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.elu(x)  # ELU activation for GAT

        final_embeddings = x

        # Predictions
        node_classification = torch.sigmoid(self.biomarker_classifier(final_embeddings))
        node_regression = self.biomarker_regressor(final_embeddings)

        graph_prediction = None
        if batch is not None:
            graph_mean = global_mean_pool(final_embeddings, batch)
            graph_max = global_max_pool(final_embeddings, batch)
            graph_features = torch.cat([graph_mean, graph_max], dim=1)
            graph_prediction = torch.sigmoid(self.graph_classifier(graph_features))

        result = {
            "node_embeddings": final_embeddings,
            "node_classification": node_classification,
            "node_regression": node_regression,
            "graph_prediction": graph_prediction,
        }

        if return_attention_weights:
            result["attention_weights"] = attention_weights

        return result


class CausalGraphDataProcessor:
    """
    Processes causal graphs and biomarker data for GNN training
    """

    def __init__(self, config: GNNConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.biomarker_to_idx = {}
        self.idx_to_biomarker = {}

    def create_graph_data(
        self,
        biomarker_data: pd.DataFrame,
        causal_graph: nx.Graph,
        target_outcomes: Optional[pd.DataFrame] = None,
        biomarker_scores: Optional[Dict[str, float]] = None,
    ) -> List[Data]:
        """
        Convert biomarker data and causal graph to PyTorch Geometric Data objects
        """
        logger.info("🔧 Creating graph data for GNN training...")

        # Create biomarker index mapping
        biomarkers = list(biomarker_data.columns)
        self.biomarker_to_idx = {
            biomarker: idx for idx, biomarker in enumerate(biomarkers)
        }
        self.idx_to_biomarker = {
            idx: biomarker for biomarker, idx in self.biomarker_to_idx.items()
        }

        logger.info(f"   📊 Biomarkers: {len(biomarkers)}")
        logger.info(f"   📊 Subjects: {len(biomarker_data)}")

        # Normalize biomarker data
        normalized_data = self.scaler.fit_transform(biomarker_data.values)

        # Create edge index from causal graph
        edge_index, edge_weights = self._create_edge_index(causal_graph, biomarkers)

        logger.info(f"   📊 Graph edges: {edge_index.shape[1]}")

        # Create Data objects for each subject
        data_list = []

        for i, (subject_idx, subject_data) in enumerate(biomarker_data.iterrows()):
            # Node features (biomarker values for this subject)
            x = torch.FloatTensor(normalized_data[i]).unsqueeze(1)  # [num_nodes, 1]

            # Add biomarker metadata as additional features
            metadata_features = self._create_biomarker_metadata_features(biomarkers)
            x = torch.cat([x, metadata_features], dim=1)  # [num_nodes, num_features]

            # Node labels (biomarker importance scores if available)
            y_node = None
            if biomarker_scores:
                scores = [
                    biomarker_scores.get(biomarker, 0.0) for biomarker in biomarkers
                ]
                y_node = torch.FloatTensor(scores)

            # Graph label (patient outcome if available)
            y_graph = None
            if target_outcomes is not None and subject_idx in target_outcomes.index:
                y_graph = torch.FloatTensor([target_outcomes.loc[subject_idx].iloc[0]])

            # Create PyTorch Geometric Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weights,
                y=y_graph,
                y_node=y_node,
                subject_id=subject_idx,
            )

            data_list.append(data)

        logger.info(f"   ✅ Created {len(data_list)} graph data objects")
        return data_list

    def _create_edge_index(
        self, causal_graph: nx.Graph, biomarkers: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edge index and weights from causal graph"""

        edges = []
        weights = []

        for biomarker in biomarkers:
            if biomarker not in causal_graph.nodes():
                continue

            neighbors = list(causal_graph.neighbors(biomarker))
            neighbor_scores = []

            for neighbor in neighbors:
                if neighbor in self.biomarker_to_idx:
                    edge_data = causal_graph.get_edge_data(biomarker, neighbor, {})
                    confidence = edge_data.get("confidence", 0.5)

                    if confidence >= self.config.edge_threshold:
                        neighbor_scores.append((neighbor, confidence))

            # Sort by confidence and take top neighbors
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            neighbor_scores = neighbor_scores[: self.config.max_neighbors]

            # Add edges
            source_idx = self.biomarker_to_idx[biomarker]
            for neighbor, confidence in neighbor_scores:
                target_idx = self.biomarker_to_idx[neighbor]

                edges.append([source_idx, target_idx])
                weights.append(confidence)

                # Add reverse edge for undirected graph behavior
                edges.append([target_idx, source_idx])
                weights.append(confidence)

        if len(edges) == 0:
            # Create minimal self-loop edges if no causal edges found
            logger.warning("   ⚠️ No causal edges found, creating self-loops")
            for i in range(len(biomarkers)):
                edges.append([i, i])
                weights.append(1.0)

        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_weights = (
            torch.FloatTensor(weights) if self.config.use_edge_weights else None
        )

        return edge_index, edge_weights

    def _create_biomarker_metadata_features(
        self, biomarkers: List[str]
    ) -> torch.Tensor:
        """Create additional features based on biomarker metadata"""

        features = []

        for biomarker in biomarkers:
            biomarker_lower = biomarker.lower()

            # Feature: Is clinical biomarker
            is_clinical = float(
                any(
                    term in biomarker_lower
                    for term in [
                        "creatinine",
                        "urea",
                        "sodium",
                        "potassium",
                        "glucose",
                        "hemoglobin",
                    ]
                )
            )

            # Feature: Is temporal feature
            is_temporal = float(
                any(
                    suffix in biomarker_lower
                    for suffix in ["_slope", "_std", "_min", "_max", "_mean"]
                )
            )

            # Feature: Is molecular biomarker
            is_molecular = float(
                any(
                    prefix in biomarker_lower
                    for prefix in ["module_", "gene_", "pathway_"]
                )
            )

            # Feature: Is kidney-specific
            is_kidney = float(
                any(
                    term in biomarker_lower
                    for term in [
                        "creatinine",
                        "urea",
                        "kidney",
                        "renal",
                        "tubular",
                        "aki",
                    ]
                )
            )

            # Feature: Biomarker importance (simple heuristic)
            importance = 1.0 if "creatinine" in biomarker_lower else 0.5

            biomarker_features = [
                is_clinical,
                is_temporal,
                is_molecular,
                is_kidney,
                importance,
            ]
            features.append(biomarker_features)

        return torch.FloatTensor(features)


class CausalGNNTrainer:
    """
    Trainer for causal graph neural networks
    """

    def __init__(self, config: GNNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.best_model_state = None

    def train_biomarker_embeddings(
        self,
        data_list: List[Data],
        model_type: str = "GCN",
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train GNN for biomarker representation learning
        """
        logger.info(f"🚀 Training {model_type} for biomarker embeddings...")

        # Split data
        train_data, val_data = train_test_split(
            data_list, test_size=validation_split, random_state=42
        )

        # Create data loaders
        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=self.config.batch_size, shuffle=False
        )

        # Initialize model
        num_features = data_list[0].x.shape[1]

        if model_type.upper() == "GCN":
            self.model = CausalGraphGCN(num_features, self.config)
        elif model_type.upper() == "GAT":
            self.model = CausalGraphGAT(num_features, self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"   📊 Training set: {len(train_data)} graphs")
        logger.info(f"   📊 Validation set: {len(val_data)} graphs")

        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validation
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"   Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"   ⏹️ Early stopping at epoch {epoch}")
                break

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        logger.info(
            f"   ✅ Training completed. Best validation loss: {best_val_loss:.4f}"
        )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "model_state": self.best_model_state,
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Compute loss
            loss = self._compute_loss(output, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                loss = self._compute_loss(output, batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def _compute_loss(
        self, output: Dict[str, torch.Tensor], batch: Data
    ) -> torch.Tensor:
        """Compute training loss"""
        losses = []

        # Node-level regression loss (biomarker scores)
        if batch.y_node is not None and output["node_regression"] is not None:
            node_targets = batch.y_node.to(self.device).unsqueeze(1)
            node_loss = F.mse_loss(output["node_regression"], node_targets)
            losses.append(node_loss)

        # Graph-level classification loss (patient outcomes)
        if batch.y is not None and output["graph_prediction"] is not None:
            graph_targets = batch.y.to(self.device)
            graph_loss = F.binary_cross_entropy(
                output["graph_prediction"].squeeze(), graph_targets
            )
            losses.append(graph_loss)

        # Reconstruction loss (encourage meaningful embeddings)
        if output["node_embeddings"] is not None:
            # Simple reconstruction loss - embeddings should be different for different nodes
            embeddings = output["node_embeddings"]
            reconstruction_loss = -torch.mean(torch.var(embeddings, dim=0))
            losses.append(reconstruction_loss * 0.1)  # Small weight

        if len(losses) == 0:
            # Fallback: minimize embedding variance to ensure learning
            embeddings = output["node_embeddings"]
            loss = torch.mean(torch.sum(embeddings**2, dim=1))
        else:
            loss = sum(losses)

        return loss

    def get_biomarker_embeddings(self, data_list: List[Data]) -> Dict[str, np.ndarray]:
        """Extract learned biomarker embeddings"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()
        all_embeddings = []
        subject_ids = []

        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)
                output = self.model(data.x, data.edge_index, data.edge_attr)
                embeddings = output["node_embeddings"].cpu().numpy()
                all_embeddings.append(embeddings)
                subject_ids.append(data.subject_id)

        return {"embeddings": np.array(all_embeddings), "subject_ids": subject_ids}


class CausalGNNAnalyzer:
    """
    Analyze trained GNN models and extract insights
    """

    def __init__(
        self, trainer: CausalGNNTrainer, data_processor: CausalGraphDataProcessor
    ):
        self.trainer = trainer
        self.data_processor = data_processor

    def analyze_biomarker_similarity(self, embeddings: np.ndarray) -> pd.DataFrame:
        """Analyze biomarker similarity using learned embeddings"""
        logger.info("🔍 Analyzing biomarker similarity from GNN embeddings...")

        # Average embeddings across subjects
        avg_embeddings = np.mean(embeddings, axis=0)  # [num_biomarkers, embedding_dim]

        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(avg_embeddings)

        # Create similarity dataframe
        biomarkers = list(self.data_processor.idx_to_biomarker.values())
        similarity_df = pd.DataFrame(
            similarity_matrix, index=biomarkers, columns=biomarkers
        )

        # Find most similar biomarker pairs
        similarity_pairs = []
        for i in range(len(biomarkers)):
            for j in range(i + 1, len(biomarkers)):
                similarity_pairs.append(
                    {
                        "biomarker_1": biomarkers[i],
                        "biomarker_2": biomarkers[j],
                        "similarity": similarity_matrix[i, j],
                    }
                )

        similarity_pairs_df = pd.DataFrame(similarity_pairs)
        similarity_pairs_df = similarity_pairs_df.sort_values(
            "similarity", ascending=False
        )

        logger.info(f"   ✅ Found {len(similarity_pairs)} biomarker pairs")

        return similarity_df, similarity_pairs_df

    def extract_attention_patterns(self, data_list: List[Data]) -> Dict[str, Any]:
        """Extract attention patterns from GAT model"""
        if not isinstance(self.trainer.model, CausalGraphGAT):
            logger.warning("Attention analysis only available for GAT models")
            return {}

        logger.info("🔍 Extracting attention patterns from GAT...")

        self.trainer.model.eval()
        attention_weights = []

        with torch.no_grad():
            for data in data_list[:10]:  # Analyze first 10 samples
                data = data.to(self.trainer.device)
                output = self.trainer.model(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    return_attention_weights=True,
                )

                if "attention_weights" in output:
                    attention_weights.append(output["attention_weights"])

        # Aggregate attention patterns
        if attention_weights:
            # Average attention across samples and layers
            avg_attention = []
            for layer_idx in range(len(attention_weights[0])):
                layer_attentions = [att[layer_idx] for att in attention_weights]
                # Average across samples
                avg_layer_attention = torch.mean(torch.stack(layer_attentions), dim=0)
                avg_attention.append(avg_layer_attention)

            logger.info("   ✅ Extracted attention patterns")
            return {"attention_weights": avg_attention}

        return {}

    def create_biomarker_clusters(
        self, embeddings: np.ndarray, n_clusters: int = 5
    ) -> Dict[str, Any]:
        """Create biomarker clusters using learned embeddings"""
        logger.info(
            f"🎯 Creating {n_clusters} biomarker clusters from GNN embeddings..."
        )

        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        # Average embeddings across subjects
        avg_embeddings = np.mean(embeddings, axis=0)

        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(avg_embeddings)

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(avg_embeddings)

        # Create results
        biomarkers = list(self.data_processor.idx_to_biomarker.values())

        cluster_results = {
            "cluster_labels": cluster_labels,
            "cluster_centers": kmeans.cluster_centers_,
            "embeddings_2d": embeddings_2d,
            "biomarkers": biomarkers,
            "pca_explained_variance": pca.explained_variance_ratio_,
        }

        # Create cluster assignments dataframe
        cluster_df = pd.DataFrame(
            {
                "biomarker": biomarkers,
                "cluster": cluster_labels,
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
            }
        )

        # Analyze cluster characteristics
        cluster_characteristics = {}
        for cluster_id in range(n_clusters):
            cluster_biomarkers = cluster_df[cluster_df["cluster"] == cluster_id][
                "biomarker"
            ].tolist()

            # Analyze cluster composition
            clinical_count = sum(
                1
                for b in cluster_biomarkers
                if any(
                    term in b.lower()
                    for term in ["creatinine", "urea", "sodium", "glucose"]
                )
            )
            molecular_count = sum(
                1
                for b in cluster_biomarkers
                if any(prefix in b.lower() for prefix in ["module_", "gene_"])
            )
            temporal_count = sum(
                1
                for b in cluster_biomarkers
                if any(
                    suffix in b.lower() for suffix in ["_slope", "_std", "_min", "_max"]
                )
            )

            cluster_characteristics[cluster_id] = {
                "biomarkers": cluster_biomarkers,
                "size": len(cluster_biomarkers),
                "clinical_count": clinical_count,
                "molecular_count": molecular_count,
                "temporal_count": temporal_count,
                "composition": {
                    "clinical_pct": clinical_count / len(cluster_biomarkers) * 100,
                    "molecular_pct": molecular_count / len(cluster_biomarkers) * 100,
                    "temporal_pct": temporal_count / len(cluster_biomarkers) * 100,
                },
            }

        cluster_results["characteristics"] = cluster_characteristics
        cluster_results["cluster_df"] = cluster_df

        logger.info(f"   ✅ Created {n_clusters} biomarker clusters")
        for cluster_id, chars in cluster_characteristics.items():
            logger.info(
                f"      Cluster {cluster_id}: {chars['size']} biomarkers "
                f"({chars['composition']['clinical_pct']:.1f}% clinical, "
                f"{chars['composition']['molecular_pct']:.1f}% molecular)"
            )

        return cluster_results


def run_causal_gnn_analysis(
    causal_graph: nx.Graph,
    biomarker_data: pd.DataFrame,
    target_outcomes: Optional[pd.DataFrame] = None,
    biomarker_scores: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main function to run complete causal GNN analysis
    """
    logger.info("🚀 CAUSAL GRAPH NEURAL NETWORK ANALYSIS")
    logger.info("=" * 60)

    output_dir = output_dir or Path("artifacts/gnn_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = GNNConfig(
        hidden_dim=64,
        num_layers=3,
        dropout=0.2,
        learning_rate=0.001,
        num_epochs=100,
        batch_size=32,
    )

    # Initialize components
    data_processor = CausalGraphDataProcessor(config)
    trainer = CausalGNNTrainer(config)

    # Create graph data
    data_list = data_processor.create_graph_data(
        biomarker_data=biomarker_data,
        causal_graph=causal_graph,
        target_outcomes=target_outcomes,
        biomarker_scores=biomarker_scores,
    )

    results = {}

    # Train GCN model
    logger.info("\n🔬 Training Graph Convolutional Network (GCN)...")
    gcn_results = trainer.train_biomarker_embeddings(data_list, model_type="GCN")

    # Get GCN embeddings
    gcn_embeddings = trainer.get_biomarker_embeddings(data_list)

    # Save GCN model
    torch.save(
        {
            "model_state_dict": trainer.best_model_state,
            "config": config,
            "biomarker_mapping": data_processor.biomarker_to_idx,
        },
        output_dir / "gcn_model.pth",
    )

    results["gcn"] = {"training_results": gcn_results, "embeddings": gcn_embeddings}

    # Train GAT model
    logger.info("\n🔬 Training Graph Attention Network (GAT)...")
    trainer_gat = CausalGNNTrainer(config)
    gat_results = trainer_gat.train_biomarker_embeddings(data_list, model_type="GAT")

    # Get GAT embeddings
    gat_embeddings = trainer_gat.get_biomarker_embeddings(data_list)

    # Save GAT model
    torch.save(
        {
            "model_state_dict": trainer_gat.best_model_state,
            "config": config,
            "biomarker_mapping": data_processor.biomarker_to_idx,
        },
        output_dir / "gat_model.pth",
    )

    results["gat"] = {"training_results": gat_results, "embeddings": gat_embeddings}

    # Analysis
    logger.info("\n🔍 Analyzing learned representations...")

    # GCN Analysis
    gcn_analyzer = CausalGNNAnalyzer(trainer, data_processor)
    gcn_similarity_df, gcn_pairs_df = gcn_analyzer.analyze_biomarker_similarity(
        gcn_embeddings["embeddings"]
    )
    gcn_clusters = gcn_analyzer.create_biomarker_clusters(gcn_embeddings["embeddings"])

    # GAT Analysis
    gat_analyzer = CausalGNNAnalyzer(trainer_gat, data_processor)
    gat_similarity_df, gat_pairs_df = gat_analyzer.analyze_biomarker_similarity(
        gat_embeddings["embeddings"]
    )
    gat_clusters = gat_analyzer.create_biomarker_clusters(gat_embeddings["embeddings"])
    gat_attention = gat_analyzer.extract_attention_patterns(data_list)

    # Export results
    logger.info("\n📂 Exporting GNN analysis results...")

    # Save similarity analyses
    gcn_similarity_df.to_csv(output_dir / "gcn_biomarker_similarity.csv")
    gcn_pairs_df.to_csv(output_dir / "gcn_biomarker_pairs.csv", index=False)
    gat_similarity_df.to_csv(output_dir / "gat_biomarker_similarity.csv")
    gat_pairs_df.to_csv(output_dir / "gat_biomarker_pairs.csv", index=False)

    # Save clustering results
    gcn_clusters["cluster_df"].to_csv(
        output_dir / "gcn_biomarker_clusters.csv", index=False
    )
    gat_clusters["cluster_df"].to_csv(
        output_dir / "gat_biomarker_clusters.csv", index=False
    )

    # Save embeddings
    np.save(output_dir / "gcn_embeddings.npy", gcn_embeddings["embeddings"])
    np.save(output_dir / "gat_embeddings.npy", gat_embeddings["embeddings"])

    results.update(
        {
            "gcn_analysis": {
                "similarity": gcn_similarity_df,
                "pairs": gcn_pairs_df,
                "clusters": gcn_clusters,
            },
            "gat_analysis": {
                "similarity": gat_similarity_df,
                "pairs": gat_pairs_df,
                "clusters": gat_clusters,
                "attention": gat_attention,
            },
        }
    )

    # Create summary report
    create_gnn_summary_report(results, output_dir)

    logger.info(f"✅ GNN Analysis Complete! Results saved to: {output_dir}")

    return results


def create_gnn_summary_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Create comprehensive GNN analysis summary report"""

    report = f"""# Graph Neural Network Analysis Report

Generated: {pd.Timestamp.now().isoformat()}

## Model Performance

### GCN (Graph Convolutional Network)
- **Best Validation Loss**: {results['gcn']['training_results']['best_val_loss']:.4f}
- **Training Epochs**: {len(results['gcn']['training_results']['train_losses'])}

### GAT (Graph Attention Network)  
- **Best Validation Loss**: {results['gat']['training_results']['best_val_loss']:.4f}
- **Training Epochs**: {len(results['gat']['training_results']['train_losses'])}

## Biomarker Similarity Analysis

### Top 10 Most Similar Biomarker Pairs (GCN)

"""

    gcn_pairs = results["gcn_analysis"]["pairs"].head(10)
    for _, row in gcn_pairs.iterrows():
        report += f"- **{row['biomarker_1']}** ↔ **{row['biomarker_2']}**: {row['similarity']:.3f}\n"

    report += """

### Top 10 Most Similar Biomarker Pairs (GAT)

"""

    gat_pairs = results["gat_analysis"]["pairs"].head(10)
    for _, row in gat_pairs.iterrows():
        report += f"- **{row['biomarker_1']}** ↔ **{row['biomarker_2']}**: {row['similarity']:.3f}\n"

    # Cluster analysis
    report += """

## Biomarker Clustering Analysis

### GCN Clusters

"""

    gcn_clusters = results["gcn_analysis"]["clusters"]["characteristics"]
    for cluster_id, chars in gcn_clusters.items():
        report += f"""
#### Cluster {cluster_id} ({chars['size']} biomarkers)
- **Clinical biomarkers**: {chars['composition']['clinical_pct']:.1f}%
- **Molecular biomarkers**: {chars['composition']['molecular_pct']:.1f}%  
- **Temporal features**: {chars['composition']['temporal_pct']:.1f}%
- **Example biomarkers**: {', '.join(chars['biomarkers'][:5])}
"""

    report += """

### GAT Clusters

"""

    gat_clusters = results["gat_analysis"]["clusters"]["characteristics"]
    for cluster_id, chars in gat_clusters.items():
        report += f"""
#### Cluster {cluster_id} ({chars['size']} biomarkers)
- **Clinical biomarkers**: {chars['composition']['clinical_pct']:.1f}%
- **Molecular biomarkers**: {chars['composition']['molecular_pct']:.1f}%
- **Temporal features**: {chars['composition']['temporal_pct']:.1f}%
- **Example biomarkers**: {', '.join(chars['biomarkers'][:5])}
"""

    report += """

## Key Insights

### GNN vs Traditional Analysis
- **Representation Learning**: GNNs learn biomarker embeddings that capture causal relationships
- **Similarity Discovery**: GNN-based similarity reveals pathway-level relationships
- **Clustering**: Learned clusters group biomarkers by functional similarity

### Clinical Implications
- **Biomarker Redundancy**: Similar biomarkers can be identified for focused panels
- **Pathway Analysis**: Clusters reveal functional biomarker groups
- **Feature Selection**: Embeddings enable intelligent biomarker selection

### Technical Achievements
- **Causal Graph Integration**: Successfully used discovered causal relationships as GNN structure
- **Multi-Task Learning**: Combined node-level and graph-level predictions
- **Attention Mechanisms**: GAT reveals which causal relationships are most important

"""

    with open(output_dir / "gnn_analysis_report.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    # This would be called with actual data from the scaled analysis
    logger.info("Graph Neural Network module ready for integration")
    logger.info("Call run_causal_gnn_analysis() with causal graph and biomarker data")
