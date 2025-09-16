from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd


def main(dump_dir: Path) -> None:
    edges = pd.read_csv(dump_dir / "kg_edges.tsv", sep="\t")
    nodes = pd.read_csv(dump_dir / "kg_nodes.tsv", sep="\t")

    print(f"nodes: {len(nodes):,}")
    print(f"edges: {len(edges):,}")

    miss_pred = int((edges["predicate"] == "").sum()) if "predicate" in edges else -1
    miss_ctx = int((edges["context"] == "").sum()) if "context" in edges else -1
    miss_prov = int((edges["provenance"] == "").sum()) if "provenance" in edges else -1

    print("missing predicate:", miss_pred)
    print("missing context:", miss_ctx)
    print("missing provenance:", miss_prov)

    ids = set(nodes["id"])
    orph = (set(edges["s"]) | set(edges["o"])) - ids
    print("orphan refs:", len(orph))

    deg = pd.concat(
        [
            edges.groupby("s").size().rename("out"),
            edges.groupby("o").size().rename("in"),
        ],
        axis=1,
    ).fillna(0)
    isolates = set(nodes["id"]) - set(deg.index)
    print("isolated nodes:", len(isolates))

    if {"s", "predicate", "o", "context"}.issubset(edges.columns):
        dups = edges.duplicated(
            subset=["s", "predicate", "o", "context"], keep=False
        ).sum()
        print("duplicate edges:", int(dups))
    else:
        print("duplicate edges: (columns missing)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/bench/kg_smoke.py <dump_dir>")
        sys.exit(2)
    main(Path(sys.argv[1]))
