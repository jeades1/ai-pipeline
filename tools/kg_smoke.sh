#!/usr/bin/env bash
# Usage: tools/kg_smoke.sh [dump_dir]
# Example: tools/kg_smoke.sh artifacts/kg_dump
set -euo pipefail

DUMP_DIR="${1:-artifacts/kg_dump}"
NODES="${DUMP_DIR}/kg_nodes.tsv"
EDGES="${DUMP_DIR}/kg_edges.tsv"

if [[ ! -f "$NODES" || ! -f "$EDGES" ]]; then
  echo "[err] KG dump not found in ${DUMP_DIR}"
  exit 1
fi

echo
echo "=== KG SMOKE (${DUMP_DIR}) ==="

# Basic TSV counts
awk -F'\t' 'END{print "nodes:", NR-1}' "$NODES"
awk -F'\t' 'END{print "edges:", NR-1}' "$EDGES"

# Required fields
# edges.tsv header: s  predicate  o  context  direction  evidence  provenance  sign
awk -F'\t' 'NR>1 && $2==""{m++} END{print "missing predicate:", (m+0)}' "$EDGES"
awk -F'\t' 'NR>1 && $4==""{m++} END{print "missing context:", (m+0)}' "$EDGES"
awk -F'\t' 'NR>1 && $7==""{m++} END{print "missing provenance:", (m+0)}' "$EDGES"
awk -F'\t' 'NR>1 && $2=="associative" && $5==""{m++} END{print "assoc missing direction:", (m+0)}' "$EDGES"
awk -F'\t' 'NR>1 && $2=="associative" && $8==""{m++} END{print "assoc missing sign:", (m+0)}' "$EDGES"

# Python checks (pass DUMP_DIR via env to avoid shell interpolation in the heredoc)
DUMP_DIR="$DUMP_DIR" python3 - <<'PY'
import os, pandas as pd

dump = os.environ["DUMP_DIR"]
edges = pd.read_csv(f"{dump}/kg_edges.tsv", sep="\t")
nodes = pd.read_csv(f"{dump}/kg_nodes.tsv", sep="\t")

# Orphan references
ids = set(nodes["id"])
orph = (set(edges["s"]) | set(edges["o"])) - ids
print("orphan refs:", len(orph))

# Isolated nodes
deg = pd.concat(
    [edges.groupby("s").size().rename("out"),
     edges.groupby("o").size().rename("in")],
    axis=1
).fillna(0)
isolates = set(nodes["id"]) - set(deg.index)
print("isolated nodes:", len(isolates))

# Duplicate (s,p,o,context)
if {"s","predicate","o","context"}.issubset(edges.columns):
    dups = edges.duplicated(subset=["s","predicate","o","context"], keep=False).sum()
    print("duplicate edges:", int(dups))
else:
    print("duplicate edges: (columns missing)")
PY