"""
Cell-cell interaction framework.
Provides tools for modeling interactions between different cell types in tissue models.
"""

from .integration import (
    CCIIntegrator, 
    add_cellular_interaction_layer,
    link_genes_to_cell_functions,
    InteractionType,
    CellTypeConfig,
    InteractionConfig
)

from .activity_scorer import (
    ActivityScorer,
    ActivityScore,
    EvidenceWeight,
    create_interaction_activity_matrix
)

from .edge_persistence import (
    EdgePersistence,
    InteractionEdge,
    create_edge_from_activity_score
)

__all__ = [
    # Core integration
    'CCIIntegrator',
    'add_cellular_interaction_layer', 
    'link_genes_to_cell_functions',
    'InteractionType',
    'CellTypeConfig',
    'InteractionConfig',
    
    # Activity scoring
    'ActivityScorer',
    'ActivityScore', 
    'EvidenceWeight',
    'create_interaction_activity_matrix',
    
    # Persistence
    'EdgePersistence',
    'InteractionEdge',
    'create_edge_from_activity_score'
]
