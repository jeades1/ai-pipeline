#!/usr/bin/env python3
from __future__ import annotations

"""
Build and export a lightweight KG dump (nodes/edges TSVs) from local releases.

Sources used (if available locally):
  - OmniPath directed/signed interactions (gene->gene)
  - CellPhoneDB ligand-receptor and receptor->TF
  - Reactome protein->pathway + pathway hierarchy
  - HGNC gene->protein bridge when available; otherwise Reactome GMT gene->pathway

Outputs under <outdir>/kg_dump/:
  - kg_nodes.tsv
  - kg_edges.tsv

This enables path-based reranking and plotting without requiring a database.
"""

import argparse
from pathlib import Path

from kg.schema import KGEvidenceGraph
from kg.build_graph import add_causal_hints, add_gene_protein_bridge


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build and export a small KG dump from local data"
    )
    ap.add_argument("--outdir", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--context", type=str, default="human", help="Context tag to stamp on edges"
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    dump_dir = outdir / "kg_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)

    G = KGEvidenceGraph(context=args.context)

    # Populate graph from available sources; each helper is resilient to missing files
    add_causal_hints(G)
    add_gene_protein_bridge(G)

    # Export
    G.export_tsv(dump_dir)
    print(f"[kg-dump] Wrote KG dump to {dump_dir}")


if __name__ == "__main__":
    main()
