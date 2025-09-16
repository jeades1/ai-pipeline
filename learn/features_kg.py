from __future__ import annotations
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Set
import pandas as pd
import networkx as nx
import hashlib

CACHE_DIR = Path("data/processed/cache")


def _hash_assoc(assoc: pd.DataFrame) -> str:
    # Stable hash from relevant columns for cache key
    cols = [
        c
        for c in ["feature", "p_value", "effect_size", "dataset"]
        if c in assoc.columns
    ]
    data = assoc[cols].copy()
    data = data.sort_values(cols).reset_index(drop=True)
    payload = data.to_csv(index=False).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"kg_features_{key}.parquet"


def gene_kg_features_cached(
    kg: Any, assoc: pd.DataFrame, use_cache: bool = True
) -> pd.DataFrame:
    """Cached wrapper for gene_kg_features; falls back to computing if cache miss."""
    if not use_cache:
        from .features_kg import gene_kg_features as _compute  # type: ignore

        return _compute(kg, assoc)
    key = _hash_assoc(assoc)
    p = _cache_path(key)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            p.unlink(missing_ok=True)
    from .features_kg import gene_kg_features as _compute  # type: ignore

    df = _compute(kg, assoc)
    try:
        df.to_parquet(p, index=True)
    except Exception:
        pass
    return df


def _gene_like(nid: str, attrs: Dict[str, Any]) -> bool:
    k = str(attrs.get("kind", "")).lower()
    name = str(attrs.get("name", nid))
    return k in {"gene", "protein"} and not name.startswith("UNIPROT:")


def _subgraph_gene_interactome(G: nx.MultiDiGraph) -> nx.DiGraph:
    # Build a directed simple graph of gene/protein interactions (exclude associative edges)
    H = nx.DiGraph()
    for nid, attrs in G.nodes(data=True):
        if _gene_like(nid, attrs):
            H.add_node(nid)
    for u, v, key, eattrs in G.edges(keys=True, data=True):
        if key in {"associative", "has_condition"}:
            continue
        if u in H and v in H:
            H.add_edge(u, v)
    return H


def _degree_dict(H: nx.DiGraph) -> Tuple[Dict[str, int], Dict[str, int]]:
    indeg = {n: int(d) for n, d in H.in_degree()}
    outdeg = {n: int(d) for n, d in H.out_degree()}
    return indeg, outdeg


def _pagerank_gene(H: nx.DiGraph) -> Dict[str, float]:
    if H.number_of_nodes() == 0:
        return {}
    try:
        pr = nx.pagerank(H.to_undirected(as_view=True))
        return {str(k): float(v) for k, v in pr.items()}
    except Exception:
        return {n: 0.0 for n in H.nodes}


def _cpdb_binding_degree(G: nx.MultiDiGraph) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for u, v, key, _ in G.edges(keys=True, data=True):
        if key != "binds":
            continue
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def _tf_out_degree(G: nx.MultiDiGraph) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for u, v, key, _ in G.edges(keys=True, data=True):
        if key != "causes":
            continue
        deg[u] = deg.get(u, 0) + 1
    return deg


def _disease_distance_features(
    G: nx.MultiDiGraph, disease_terms: Set[str]
) -> Dict[str, float]:
    """Compute shortest path distance from genes to disease nodes."""
    # Convert to undirected simple graph for distance computation
    H = nx.Graph()
    for nid, attrs in G.nodes(data=True):
        H.add_node(nid)
    for u, v, key, _ in G.edges(keys=True, data=True):
        if key not in {"associative"}:  # Include causal and other edges
            H.add_edge(u, v)

    distances = {}
    for gene in H.nodes():
        if not _gene_like(gene, G.nodes.get(gene, {})):
            continue
        min_dist = float("inf")
        for disease in disease_terms:
            if disease in H:
                try:
                    dist = nx.shortest_path_length(H, source=gene, target=disease)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue
        # Convert to similarity (closer = higher score)
        distances[gene] = 1.0 / (1.0 + min_dist) if min_dist != float("inf") else 0.0
    return distances


def _pathway_membership_features(G: nx.MultiDiGraph) -> Dict[str, Dict[str, float]]:
    """Extract pathway membership features for genes."""
    pathway_features = {}

    # Define pathway keywords for different functional categories
    pathway_keywords = {
        "injury": ["injury", "damage", "apoptosis", "necrosis", "stress", "death"],
        "repair": ["repair", "healing", "regeneration", "recovery", "restoration"],
        "inflammation": ["inflammation", "immune", "cytokine", "interleukin", "nfkb"],
        "metabolism": ["metabolism", "metabolic", "energy", "glucose", "fatty_acid"],
        "transport": ["transport", "ion", "sodium", "potassium", "channel"],
        "signaling": ["signaling", "signal", "cascade", "pathway", "transduction"],
    }

    # Get pathway nodes
    pathway_nodes = {}
    for nid, attrs in G.nodes(data=True):
        kind = str(attrs.get("kind", "")).lower()
        name = str(attrs.get("name", nid)).lower()
        if kind in {"pathway", "process"}:
            for category, keywords in pathway_keywords.items():
                if any(kw in name for kw in keywords):
                    pathway_nodes.setdefault(category, set()).add(nid)

    # Count gene participation in each pathway category
    for gene in G.nodes():
        if not _gene_like(gene, G.nodes.get(gene, {})):
            continue
        gene_pathways = {cat: 0.0 for cat in pathway_keywords.keys()}

        # Count participates_in edges to categorized pathways
        for _, target, key, _ in G.edges(gene, keys=True, data=True):
            if key == "participates_in":
                for category, nodes in pathway_nodes.items():
                    if target in nodes:
                        gene_pathways[category] += 1.0

        pathway_features[gene] = gene_pathways

    return pathway_features


def _literature_evidence_features(
    G: nx.MultiDiGraph, disease_terms: Set[str]
) -> Dict[str, float]:
    """Compute literature co-occurrence strength with disease terms."""
    lit_scores = {}

    for gene in G.nodes():
        if not _gene_like(gene, G.nodes.get(gene, {})):
            continue

        score = 0.0
        # Count edges with literature provenance mentioning disease terms
        for _, _, key, eattrs in G.edges(gene, keys=True, data=True):
            prov = str(eattrs.get("provenance", "")).lower()
            if "pubmed" in prov or "literature" in prov:
                # Simple heuristic: check if edge context mentions disease
                context = str(eattrs.get("evidence", {})).lower()
                if any(term.lower() in context for term in disease_terms):
                    score += 1.0

        lit_scores[gene] = score

    return lit_scores


def _tissue_specificity_features(
    G: nx.MultiDiGraph, target_tissues: Set[str]
) -> Dict[str, float]:
    """Compute tissue specificity scores for target tissues."""
    tissue_scores = {}

    for gene in G.nodes():
        if not _gene_like(gene, G.nodes.get(gene, {})):
            continue

        specificity = 0.0
        total_tissues = 0

        # Look for tissue-specific expression edges
        for _, target, key, eattrs in G.edges(gene, keys=True, data=True):
            if key == "expressed_in":
                total_tissues += 1
                tissue_name = str(eattrs.get("tissue", target)).lower()
                if any(tt.lower() in tissue_name for tt in target_tissues):
                    specificity += 1.0

        # Specificity = fraction of expression in target tissues
        tissue_scores[gene] = specificity / max(1, total_tissues)

    return tissue_scores


def gene_kg_features(kg: Any, assoc: pd.DataFrame) -> pd.DataFrame:
    """Compute KG-derived features for genes present in assoc.

    Returns a DataFrame with index=gene name and columns:
      ['deg_in','deg_out','pr','cpdb_deg','tf_out','disease_dist','injury_pathways',
       'repair_pathways','inflammation_pathways','transport_pathways','literature_score','tissue_specificity']
    """
    G = kg.G  # type: ignore[attr-defined]
    H = _subgraph_gene_interactome(G)
    indeg, outdeg = _degree_dict(H)
    pr = _pagerank_gene(H)
    cpdb = _cpdb_binding_degree(G)
    tf_out = _tf_out_degree(G)

    # Enhanced features
    disease_terms = {"AKI", "kidney_injury", "renal_injury", "acute_kidney_injury"}
    target_tissues = {"kidney", "renal", "tubular", "glomerular"}

    disease_dist = _disease_distance_features(G, disease_terms)
    pathway_features = _pathway_membership_features(G)
    literature_scores = _literature_evidence_features(G, disease_terms)
    tissue_scores = _tissue_specificity_features(G, target_tissues)

    genes = assoc["feature"].astype(str).unique().tolist()
    rows = []
    for g in genes:
        g_upper = g.upper()
        pathway_data = pathway_features.get(g, {})

        rows.append(
            {
                "gene": g_upper,
                "deg_in": float(indeg.get(g, 0)),
                "deg_out": float(outdeg.get(g, 0)),
                "pr": float(pr.get(g, 0.0)),
                "cpdb_deg": float(cpdb.get(g, 0)),
                "tf_out": float(tf_out.get(g, 0)),
                "disease_dist": float(disease_dist.get(g, 0.0)),
                "injury_pathways": float(pathway_data.get("injury", 0.0)),
                "repair_pathways": float(pathway_data.get("repair", 0.0)),
                "inflammation_pathways": float(pathway_data.get("inflammation", 0.0)),
                "transport_pathways": float(pathway_data.get("transport", 0.0)),
                "literature_score": float(literature_scores.get(g, 0.0)),
                "tissue_specificity": float(tissue_scores.get(g, 0.0)),
            }
        )
    df = pd.DataFrame(rows).set_index("gene")
    return df
