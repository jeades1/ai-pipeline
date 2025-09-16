from __future__ import annotations

"""
Re-rank promoted genes using a simple blend of:
  - assoc_score: proxy from initial order in promoted.tsv
  - prior_score: normalized from available priors (OpenTargets/GTEx/HPA)
  - path_score: 1.0 if a path to any Pathway node exists within <=3 hops in the KG

Writes:
  - artifacts/promoted_full.tsv (with scores)
  - artifacts/ranked.tsv (sorted by total_score desc)
"""
import argparse
from pathlib import Path
from typing import Dict
import re
import json
import pandas as pd
import numpy as np
import networkx as nx


def load_promoted(promoted_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(promoted_tsv, sep="\t")
    if not {"name", "layer", "type"}.issubset(df.columns):
        raise SystemExit("promoted.tsv needs columns: name, layer, type")
    genes = df[df["type"].str.lower() == "gene"].copy()
    # assoc proxy: earlier rows are higher ranked
    n = max(len(genes), 1)
    genes = genes.reset_index(drop=True)
    genes["assoc_score"] = np.linspace(1.0, 0.0, num=n, endpoint=False)
    return genes


def load_priors(manifest_path: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if not manifest_path.exists():
        return scores
    # Parse manifest JSON for robust access
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except Exception:
        return scores

    # Prefer unified priors for context-aware weighting
    unified_p = manifest.get("outputs", {}).get("unified")
    context = (manifest.get("context") or "").strip()

    def read_any(p: Path) -> pd.DataFrame | None:
        if not p.exists():
            return None
        try:
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, sep="\t")
        except Exception:
            return None

    if unified_p:
        df = read_any(Path(unified_p))
        if df is not None and not df.empty:
            # Expected columns: gene_symbol, context, score, source, context_type
            # Filter by tissue context if provided
            if "context_type" in df.columns:
                # Keep tissue rows matching context if specified
                if context:
                    tis = df[
                        (df["context_type"] == "tissue")
                        & (
                            df["context"]
                            .astype(str)
                            .str.contains(context, case=False, na=False)
                        )
                    ].copy()
                else:
                    tis = df[df["context_type"] == "tissue"].copy()
                # Keep disease rows too but weight them lower
                dis = (
                    df[df["context_type"] == "disease"].copy()
                    if "context_type" in df.columns
                    else pd.DataFrame(columns=df.columns)
                )
            else:
                # Fallback: treat all as usable
                tis = df.copy()
                dis = pd.DataFrame(columns=df.columns)

            # Source-aware weights (renormalized over available sources)
            base_weights = {
                "GTEx": 0.4,
                "HPA": 0.4,
                "OpenTargets": 0.2,
                "ENCODE": 0.1,
                "PRIDE": 0.1,
            }
            if not context:
                # If no tissue context, bump OT a bit
                base_weights.update({"OpenTargets": 0.4, "GTEx": 0.3, "HPA": 0.3})

            def agg_weighted(dd: pd.DataFrame) -> pd.Series:
                if dd.empty:
                    return pd.Series(dtype=float)
                dd = dd.copy()
                # Ensure required columns exist and are strings
                if "gene_symbol" not in dd.columns:
                    if "name" in dd.columns:
                        dd["gene_symbol"] = dd["name"].astype(str)
                    else:
                        dd["gene_symbol"] = ""
                else:
                    dd["gene_symbol"] = dd["gene_symbol"].astype(str)
                if "source" in dd.columns:
                    dd["source"] = dd["source"].astype(str)
                else:
                    dd["source"] = ""
                # Normalize per source defensively
                dd["score"] = dd["score"].astype(float)
                dd["score"] = dd.groupby("source")["score"].transform(
                    lambda s: (
                        (s - s.min()) / (s.max() - s.min())
                        if (s.max() - s.min()) > 0
                        else 1.0
                    )
                )
                # Apply weights present
                dd["w"] = dd["source"].map(base_weights).fillna(0.1)
                # Renormalize weights over present sources
                w_sum = dd.groupby("source")["w"].first().sum() or 1.0
                dd["w"] = dd["w"] / w_sum
                s = dd.groupby("gene_symbol").apply(
                    lambda g: float((g["score"] * g["w"]).sum())
                )
                s.index = s.index.astype(str).str.upper()
                return s

            s_tis = agg_weighted(tis)
            # Diseases weighted lower overall
            s_dis = agg_weighted(dis) * 0.5 if not dis.empty else pd.Series(dtype=float)
            combined = s_tis.add(s_dis, fill_value=0.0)
            return {k: float(v) for k, v in combined.to_dict().items()}

    # Fallback: aggregate per-source simple mean like before
    src_paths = []
    for key in ["opentargets", "gtex", "hpa", "encode", "pride"]:
        p = manifest.get("outputs", {}).get(key)
        if p:
            src_paths.append(Path(p))
    per_source = []
    for p in src_paths:
        df = read_any(Path(p))
        if df is None or df.empty:
            continue
        col_name = next(
            (
                c
                for c in ["gene", "name", "symbol", "Gene", "gene_symbol"]
                if c in df.columns
            ),
            None,
        )
        if not col_name:
            continue
        col_score = next(
            (
                c
                for c in ["score", "association_score", "expr", "z", "weight", "prior"]
                if c in df.columns
            ),
            None,
        )
        t = (
            df[[col_name]].assign(score=1.0).rename(columns={col_name: "name"})
            if not col_score
            else df[[col_name, col_score]]
            .rename(columns={col_name: "name", col_score: "score"})
            .dropna()
        )
        if t.empty:
            continue
        s = t["score"].astype(float)
        rng = s.max() - s.min()
        t["score"] = (s - s.min()) / rng if rng > 0 else 1.0
        per_source.append(t[["name", "score"]])

    if not per_source:
        return scores
    merged = pd.concat(per_source, ignore_index=True)
    merged["name"] = merged["name"].astype(str).str.upper()
    agg = merged.groupby("name")["score"].mean().reset_index()
    return {r["name"].upper(): float(r["score"]) for _, r in agg.iterrows()}


def build_graph(
    nodes_tsv: Path, edges_tsv: Path
) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    nodes = pd.read_csv(nodes_tsv, sep="\t", low_memory=False)
    edges = pd.read_csv(edges_tsv, sep="\t", low_memory=False)
    G = nx.MultiDiGraph()
    for _, r in nodes.iterrows():
        nid = str(r["id"])
        G.add_node(nid, kind=r.get("kind", ""), name=r.get("name", nid))
    for _, r in edges.iterrows():
        s = str(r["s"])
        o = str(r["o"])
        p = str(r.get("predicate", ""))
        G.add_edge(s, o, key=p)
    return G, nodes


def find_id_by_name(nodes_df: pd.DataFrame, name: str) -> str | None:
    m = nodes_df[nodes_df["name"].astype(str).str.upper() == str(name).upper()]
    if not m.empty:
        return str(m.iloc[0]["id"])
    m2 = nodes_df[nodes_df["id"].astype(str).str.upper() == str(name).upper()]
    if not m2.empty:
        return str(m2.iloc[0]["id"])
    return None


def _pathway_targets(nodes_df: pd.DataFrame, terms: list[str] | None) -> set[str]:
    if nodes_df is None or nodes_df.empty:
        return set()
    pathways = nodes_df[nodes_df["kind"].astype(str).str.lower() == "pathway"].copy()
    if terms:
        t = [x.lower() for x in terms]
        mask = (
            pathways["name"]
            .astype(str)
            .str.lower()
            .apply(lambda s: any(term in s for term in t))
        )
        pathways = pathways[mask]
    return set(pathways["id"].astype(str).tolist())


def has_path_to_pathway(
    G: nx.MultiDiGraph,
    nodes_df: pd.DataFrame,
    gene: str,
    max_hops: int = 3,
    target_pathways: set[str] | None = None,
) -> float:
    src = find_id_by_name(nodes_df, gene)
    if not src or src not in G:
        return 0.0
    from collections import deque

    dq = deque([src])
    seen = {src}
    hops = {src: 0}
    targets = set(target_pathways or [])
    while dq:
        u = dq.popleft()
        if hops[u] >= max_hops:
            continue
        for _, v, _ in G.out_edges(u, keys=True):
            if v in seen:
                continue
            seen.add(v)
            hops[v] = hops[u] + 1
            kind = str(G.nodes[v].get("kind", ""))
            is_pathway = kind.lower() == "pathway"
            if is_pathway and (not targets or v in targets):
                # reward shorter paths higher
                return max(0.0, 1.0 - 0.25 * (hops[v] - 1))
            dq.append(v)
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="Causal+prior re-ranker")
    ap.add_argument("--promoted", type=Path, default=Path("artifacts/promoted.tsv"))
    ap.add_argument(
        "--kg-nodes", type=Path, default=Path("artifacts/kg_dump/kg_nodes.tsv")
    )
    ap.add_argument(
        "--kg-edges", type=Path, default=Path("artifacts/kg_dump/kg_edges.tsv")
    )
    ap.add_argument(
        "--priors-manifest",
        type=Path,
        default=Path("data/processed/priors/manifest.json"),
    )
    ap.add_argument(
        "--out-full", type=Path, default=Path("artifacts/promoted_full.tsv")
    )
    ap.add_argument("--out-ranked", type=Path, default=Path("artifacts/ranked.tsv"))
    ap.add_argument(
        "--pathway-terms",
        nargs="*",
        default=None,
        help="Optional substrings to select relevant pathways for path score (e.g., lipid cholesterol LDL)",
    )
    ap.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[0.35, 0.45, 0.20],
        metavar=("W_ASSOC", "W_PRIOR", "W_PATH"),
        help="Weights for assoc, prior, path (sum not required; will be normalized)",
    )
    ap.add_argument(
        "--penalize-immune",
        action="store_true",
        help="Apply penalties to immunoglobulin/TCR genes to reduce immune artifacts in non-immune contexts",
    )
    ap.add_argument(
        "--immune-penalty",
        type=float,
        default=0.15,
        help="Penalty subtracted from total score for immune-pattern genes (0-1)",
    )
    args = ap.parse_args()

    genes = load_promoted(args.promoted)
    pri = load_priors(args.priors_manifest)
    # Build KG if present; else fall back to zero path scores
    G = None
    nodes_df = None
    target_pw = set()
    try:
        if args.kg_nodes.exists() and args.kg_edges.exists():
            G, nodes_df = build_graph(args.kg_nodes, args.kg_edges)
            target_pw = _pathway_targets(nodes_df, args.pathway_terms)
    except Exception:
        G = None
        nodes_df = None
        target_pw = set()

    # scores
    prior_score = []
    path_score = []
    for g in genes["name"].astype(str):
        prior_score.append(float(pri.get(g.upper(), 0.0)))
        if G is None or nodes_df is None:
            path_score.append(0.0)
        else:
            path_score.append(
                float(
                    has_path_to_pathway(
                        G, nodes_df, g, max_hops=3, target_pathways=target_pw
                    )
                )
            )
    genes["prior_score"] = prior_score
    genes["path_score"] = path_score

    # blend
    # Blend with provided weights (normalized)
    w_assoc, w_prior, w_path = args.weights
    s = max(1e-9, float(w_assoc + w_prior + w_path))
    wa, wp, wc = w_assoc / s, w_prior / s, w_path / s
    genes["total_score"] = (
        wa * genes["assoc_score"] + wp * genes["prior_score"] + wc * genes["path_score"]
    )

    # Optional: penalize immunoglobulins / TCRs (realistic in liver/cardiometabolic contexts)
    if args.penalize_immune:
        patt = re.compile(r"^(IG[HKL]|TR[ABDG])", re.IGNORECASE)
        penalties = []
        for g in genes["name"].astype(str):
            penalties.append(float(args.immune_penalty) if patt.match(g) else 0.0)
        genes["total_score"] = genes["total_score"] - pd.Series(
            penalties, index=genes.index
        )

    # write promoted_full and ranked
    out_full = genes[
        [
            "name",
            "layer",
            "type",
            "assoc_score",
            "prior_score",
            "path_score",
            "total_score",
        ]
    ].copy()
    args.out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.to_csv(args.out_full, sep="\t", index=False)

    ranked = out_full.sort_values("total_score", ascending=False)
    args.out_ranked.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out_ranked, sep="\t", index=False)
    print(f"[rerank] Wrote {args.out_full} and {args.out_ranked}")


if __name__ == "__main__":
    main()
