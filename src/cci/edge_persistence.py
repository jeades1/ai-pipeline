"""
Edge persistence for cell-cell interaction knowledge graph.
Manages storage and retrieval of interaction edges with metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class InteractionEdge:
    """Represents a cell-cell interaction edge"""
    edge_id: str
    source_cell: str
    target_cell: str
    interaction_type: str
    activity_score: float
    confidence: float
    evidence_types: List[str]
    mediators: List[str]
    provenance: str
    created_at: str
    metadata: Dict[str, Any]


class EdgePersistence:
    """Manages persistence of cell-cell interaction edges"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "cci_edges.db"
        self.json_path = self.storage_path / "cci_edges.json"
        
        # Create directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for edge storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_cell TEXT NOT NULL,
                    target_cell TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    activity_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_types TEXT NOT NULL,  -- JSON array
                    mediators TEXT,  -- JSON array
                    provenance TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT  -- JSON object
                );
            """)
            
            # Create indices for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_target 
                ON interaction_edges(source_cell, target_cell);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interaction_type 
                ON interaction_edges(interaction_type);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activity_score 
                ON interaction_edges(activity_score DESC);
            """)
            
            conn.commit()
    
    def save_edge(self, edge: InteractionEdge) -> None:
        """Save a single interaction edge"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO interaction_edges 
                (edge_id, source_cell, target_cell, interaction_type, activity_score,
                 confidence, evidence_types, mediators, provenance, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.edge_id,
                edge.source_cell,
                edge.target_cell,
                edge.interaction_type,
                edge.activity_score,
                edge.confidence,
                json.dumps(edge.evidence_types),
                json.dumps(edge.mediators),
                edge.provenance,
                edge.created_at,
                json.dumps(edge.metadata)
            ))
            
            conn.commit()
    
    def save_edges_batch(self, edges: List[InteractionEdge]) -> None:
        """Save multiple edges in a batch"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            edge_data = []
            for edge in edges:
                edge_data.append((
                    edge.edge_id,
                    edge.source_cell,
                    edge.target_cell,
                    edge.interaction_type,
                    edge.activity_score,
                    edge.confidence,
                    json.dumps(edge.evidence_types),
                    json.dumps(edge.mediators),
                    edge.provenance,
                    edge.created_at,
                    json.dumps(edge.metadata)
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO interaction_edges 
                (edge_id, source_cell, target_cell, interaction_type, activity_score,
                 confidence, evidence_types, mediators, provenance, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, edge_data)
            
            conn.commit()
    
    def load_edge(self, edge_id: str) -> Optional[InteractionEdge]:
        """Load a specific edge by ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interaction_edges WHERE edge_id = ?
            """, (edge_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_edge(row)
            return None
    
    def load_edges_by_cells(self, source_cell: Optional[str] = None,
                           target_cell: Optional[str] = None) -> List[InteractionEdge]:
        """Load edges by source and/or target cell types"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if source_cell and target_cell:
                cursor.execute("""
                    SELECT * FROM interaction_edges 
                    WHERE source_cell = ? AND target_cell = ?
                    ORDER BY activity_score DESC
                """, (source_cell, target_cell))
            elif source_cell:
                cursor.execute("""
                    SELECT * FROM interaction_edges 
                    WHERE source_cell = ?
                    ORDER BY activity_score DESC
                """, (source_cell,))
            elif target_cell:
                cursor.execute("""
                    SELECT * FROM interaction_edges 
                    WHERE target_cell = ?
                    ORDER BY activity_score DESC
                """, (target_cell,))
            else:
                cursor.execute("""
                    SELECT * FROM interaction_edges 
                    ORDER BY activity_score DESC
                """)
            
            rows = cursor.fetchall()
            return [self._row_to_edge(row) for row in rows]
    
    def load_edges_by_type(self, interaction_type: str,
                          min_score: float = 0.0) -> List[InteractionEdge]:
        """Load edges by interaction type with minimum score threshold"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interaction_edges 
                WHERE interaction_type = ? AND activity_score >= ?
                ORDER BY activity_score DESC
            """, (interaction_type, min_score))
            
            rows = cursor.fetchall()
            return [self._row_to_edge(row) for row in rows]
    
    def load_high_confidence_edges(self, min_confidence: float = 0.7,
                                  min_score: float = 0.5) -> List[InteractionEdge]:
        """Load high-confidence edges above thresholds"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interaction_edges 
                WHERE confidence >= ? AND activity_score >= ?
                ORDER BY activity_score DESC
            """, (min_confidence, min_score))
            
            rows = cursor.fetchall()
            return [self._row_to_edge(row) for row in rows]
    
    def _row_to_edge(self, row: tuple) -> InteractionEdge:
        """Convert database row to InteractionEdge object"""
        
        return InteractionEdge(
            edge_id=row[0],
            source_cell=row[1],
            target_cell=row[2],
            interaction_type=row[3],
            activity_score=row[4],
            confidence=row[5],
            evidence_types=json.loads(row[6]),
            mediators=json.loads(row[7]) if row[7] else [],
            provenance=row[8],
            created_at=row[9],
            metadata=json.loads(row[10]) if row[10] else {}
        )
    
    def export_to_dataframe(self, min_score: float = 0.0) -> pd.DataFrame:
        """Export edges to pandas DataFrame for analysis"""
        
        edges = self.load_high_confidence_edges(min_confidence=0.0, min_score=min_score)
        
        if not edges:
            return pd.DataFrame()
        
        # Convert to DataFrame
        edge_dicts = []
        for edge in edges:
            edge_dict = asdict(edge)
            # Flatten some fields for easier analysis
            edge_dict['evidence_types_str'] = ', '.join(edge.evidence_types)
            edge_dict['mediators_str'] = ', '.join(edge.mediators)
            edge_dict['num_evidence_types'] = len(edge.evidence_types)
            edge_dict['num_mediators'] = len(edge.mediators)
            edge_dicts.append(edge_dict)
        
        return pd.DataFrame(edge_dicts)
    
    def export_to_json(self, output_path: Optional[Path] = None) -> None:
        """Export all edges to JSON file"""
        
        output_path = output_path or self.json_path
        
        edges = self.load_edges_by_cells()
        edge_dicts = [asdict(edge) for edge in edges]
        
        with open(output_path, 'w') as f:
            json.dump(edge_dicts, f, indent=2)
    
    def import_from_json(self, input_path: Path) -> None:
        """Import edges from JSON file"""
        
        with open(input_path, 'r') as f:
            edge_dicts = json.load(f)
        
        edges = []
        for edge_dict in edge_dicts:
            edges.append(InteractionEdge(**edge_dict))
        
        self.save_edges_batch(edges)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the interaction network"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total edges
            cursor.execute("SELECT COUNT(*) FROM interaction_edges")
            total_edges = cursor.fetchone()[0]
            
            # Unique cell types
            cursor.execute("""
                SELECT COUNT(DISTINCT cell_type) FROM (
                    SELECT source_cell as cell_type FROM interaction_edges
                    UNION
                    SELECT target_cell as cell_type FROM interaction_edges
                )
            """)
            unique_cell_types = cursor.fetchone()[0]
            
            # Average score and confidence
            cursor.execute("""
                SELECT AVG(activity_score), AVG(confidence) 
                FROM interaction_edges
            """)
            avg_score, avg_confidence = cursor.fetchone()
            
            # Interaction type distribution
            cursor.execute("""
                SELECT interaction_type, COUNT(*) 
                FROM interaction_edges
                GROUP BY interaction_type
                ORDER BY COUNT(*) DESC
            """)
            type_distribution = dict(cursor.fetchall())
            
            # Top cell type pairs by activity
            cursor.execute("""
                SELECT source_cell, target_cell, AVG(activity_score), COUNT(*)
                FROM interaction_edges
                GROUP BY source_cell, target_cell
                ORDER BY AVG(activity_score) DESC
                LIMIT 10
            """)
            top_pairs = cursor.fetchall()
            
            return {
                "total_edges": total_edges,
                "unique_cell_types": unique_cell_types,
                "average_activity_score": avg_score,
                "average_confidence": avg_confidence,
                "interaction_type_distribution": type_distribution,
                "top_cell_pairs": [
                    {"source": pair[0], "target": pair[1], 
                     "avg_score": pair[2], "num_interactions": pair[3]}
                    for pair in top_pairs
                ]
            }


def create_edge_from_activity_score(activity_score: Any, provenance: str = "automated") -> InteractionEdge:
    """Create InteractionEdge from ActivityScore object"""
    
    edge_id = f"{activity_score.source_cell}_{activity_score.target_cell}_{activity_score.interaction_type}"
    
    # Extract mediators from supporting data
    mediators = []
    if 'relevant_lr_pairs' in activity_score.supporting_data:
        for lr_pair in activity_score.supporting_data['relevant_lr_pairs']:
            mediators.extend([lr_pair['ligand'], lr_pair['receptor']])
    
    # Determine evidence types based on non-zero scores
    evidence_types = []
    for evidence_type, score in activity_score.evidence_breakdown.items():
        if score > 0:
            evidence_types.append(evidence_type)
    
    return InteractionEdge(
        edge_id=edge_id,
        source_cell=activity_score.source_cell,
        target_cell=activity_score.target_cell,
        interaction_type=activity_score.interaction_type,
        activity_score=activity_score.overall_score,
        confidence=activity_score.confidence,
        evidence_types=evidence_types,
        mediators=list(set(mediators)),  # Remove duplicates
        provenance=provenance,
        created_at=datetime.now().isoformat(),
        metadata=activity_score.supporting_data
    )
