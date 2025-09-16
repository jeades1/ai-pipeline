from __future__ import annotations

"""
Multi-omics audit for the Evidence Graph and rankings.

Outputs a compact Markdown report with:
 - Node counts by layer and kind
 - Edge counts by predicate and layer
 - Cross-level linkage checks (e.g., paths from genes to pathways)
 - Causal vs associative edge presence
 - Coverage on top-N ranked genes

Usage:
    python -m tools.maint.audit_multiomics \
        --kg-nodes artifacts/kg_dump/kg_nodes.tsv \
        --kg-edges artifacts/kg_dump/kg_edges.tsv \
        --ranked   artifacts/promoted_full.tsv \
        --out      artifacts/audit_multiomics.md
"""

from pathlib import Path
from typing import Optional, Tuple
import argparse
import pandas as pd
import networkx as nx


def _load_graph(
    nodes_path: Path, edges_path: Path
) -> Tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_path, sep="\t", low_memory=False)
    edges = pd.read_csv(edges_path, sep="\t", low_memory=False)

    G = nx.MultiDiGraph()
    for _, r in nodes.iterrows():
        nid = str(r["id"])
        G.add_node(
            nid,
            kind=r.get("kind", ""),
            layer=r.get("layer", ""),
            name=r.get("name", nid),
        )

    for _, r in edges.iterrows():
        s = str(r["s"])
        o = str(r["o"])
        p = str(r.get("predicate", ""))
        G.add_edge(
            s,
            o,
            key=p,
            predicate=p,
            context=r.get("context", ""),
            direction=r.get("direction", ""),
            evidence=r.get("evidence", ""),
            provenance=r.get("provenance", ""),
            sign=r.get("sign", ""),
        )
    return G, nodes, edges


def _counts(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    g = df.groupby(list(cols)).size().reset_index(name="count")
    return g.sort_values(by="count", ascending=False)


def _path_to_pathway(G: nx.MultiDiGraph, nid: Optional[str], max_hops: int = 3) -> bool:
    if not nid or nid not in G:
        return False
    visited = {nid}
    frontier = {nid}
    for _ in range(max_hops):
        next_frontier = set()
        for u in frontier:
            for _, v, _ in G.out_edges(u, keys=True):
                if v in visited:
                    continue
                visited.add(v)
                if str(G.nodes[v].get("kind", "")).lower() == "pathway":
                    return True
                next_frontier.add(v)
        if not next_frontier:
            break
        frontier = next_frontier
    return False


def _find_node_by_name(nodes: pd.DataFrame, name: str) -> Optional[str]:
    m = nodes[nodes["name"].astype(str).str.upper() == str(name).upper()]
    if not m.empty:
        return str(m.iloc[0]["id"])
    # fallback: id exact match
    m2 = nodes[nodes["id"].astype(str).str.upper() == str(name).upper()]
    if not m2.empty:
        return str(m2.iloc[0]["id"])
    return None


def audit(
    nodes_path: Path,
    edges_path: Path,
    ranked_path: Optional[Path],
    out_path: Path,
    top_n: int = 100,
) -> Path:
    G, nodes, edges = _load_graph(nodes_path, edges_path)

    # Node/edge summaries
    nodes_by_layer = _counts(nodes, ("layer", "kind"))
    edges_by_pred = _counts(edges, ("predicate", "provenance"))
    edges_by_layer = _counts(edges, ("predicate", "provenance"))

    # Causal vs associative presence
    has_causal = any(
        edges["predicate"].astype(str).str.contains("caus", case=False, na=False)
    )
    n_causal = int(
        (
            edges["predicate"].astype(str).str.contains("caus", case=False, na=False)
        ).sum()
    )
    n_assoc = int(
        (
            edges["predicate"].astype(str).str.contains("assoc", case=False, na=False)
        ).sum()
    )

    # Ranked coverage to pathways
    top_rows = []
    coverage = {"total": 0, "with_pathway": 0}
    ranked = None
    if ranked_path and ranked_path.exists():
        ranked = (
            pd.read_csv(ranked_path, sep="\t")
            if ranked_path.suffix == ".tsv"
            else pd.read_parquet(ranked_path)
        )
        keep_cols = [
            c for c in ("name", "assoc_score", "total_score") if c in ranked.columns
        ]
        ranked = ranked[keep_cols].drop_duplicates()
        ranked = ranked.head(top_n)
        coverage["total"] = len(ranked)
        for _, r in ranked.iterrows():
            nm = str(r["name"]) if "name" in r else ""
            nid = _find_node_by_name(nodes, nm) if nm else None
            reachable = _path_to_pathway(G, nid, max_hops=3) if nid else False
            top_rows.append(
                {
                    "name": nm,
                    "assoc_score": float(r.get("assoc_score", 0.0)),
                    "total_score": float(r.get("total_score", 0.0)),
                    "to_pathway_<=3": bool(reachable),
                }
            )
            if reachable:
                coverage["with_pathway"] += 1

    # Build markdown
    lines = []
    lines.append("# Multi-Omics Audit\n")
    lines.append(f"Nodes: {len(nodes):,} | Edges: {len(edges):,}\n")
    lines.append("")
    lines.append("## Node counts by layer/kind\n")
    lines.append(nodes_by_layer.to_markdown(index=False))
    lines.append("")
    lines.append("## Edge counts by predicate/provenance\n")
    lines.append(edges_by_pred.to_markdown(index=False))
    lines.append("")
    lines.append(
        f"## Causality signals\n\n- causal edges present: {has_causal} (n={n_causal})\n- associative edges (n={n_assoc})\n"
    )
    lines.append("")
    if ranked is not None and len(top_rows) > 0:
        lines.append(f"## Top-{top_n} ranked pathway reachability (<=3 hops)\n")
        df_top = pd.DataFrame(top_rows)
        lines.append(df_top.to_markdown(index=False))
        lines.append("")
        pct = 100.0 * coverage["with_pathway"] / max(coverage["total"], 1)
        lines.append(
            f"Pathway reachability: {coverage['with_pathway']}/{coverage['total']} ({pct:.1f}%)\n"
        )

    # Heuristic findings
    findings = []
    # If very low pathway reachability, likely missing gene→protein/pathway bridge
    if ranked is not None and coverage["total"] > 0 and coverage["with_pathway"] == 0:
        findings.append(
            "No ranked genes connect to pathway nodes within 3 hops — likely missing gene→protein (UniProt) bridging."
        )
    if n_causal == 0:
        findings.append(
            "No causal edges detected. Consider integrating curated causal sources or deriving causal signatures."
        )
    if len(findings) == 0:
        findings.append("No obvious structural issues detected.")

    lines.append("## Findings\n")
    for f in findings:
        lines.append(f"- {f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser("audit_multiomics")
    ap.add_argument("--kg-nodes", default="artifacts/kg_dump/kg_nodes.tsv")
    ap.add_argument("--kg-edges", default="artifacts/kg_dump/kg_edges.tsv")
    ap.add_argument("--ranked", default="artifacts/promoted_full.tsv")
    ap.add_argument("--out", default="artifacts/audit_multiomics.md")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    nodes_p = Path(args.kg_nodes)
    edges_p = Path(args.kg_edges)
    ranked_p = Path(args.ranked)
    out_p = Path(args.out)

    if not nodes_p.exists() or not edges_p.exists():
        raise SystemExit(f"Missing KG dump: {nodes_p} or {edges_p}")

    audit(
        nodes_p,
        edges_p,
        ranked_p if ranked_p.exists() else None,
        out_p,
        top_n=int(args.top_n),
    )
    print(f"[audit] Wrote {out_p}")


if __name__ == "__main__":
    main()
