#!/usr/bin/env bash
set -euo pipefail

have_rg=1
command -v rg >/dev/null 2>&1 || have_rg=0

echo "[validate] scanning add_node / add_edge calls for required attrs"
if [ "$have_rg" -eq 1 ]; then
  rg -n "add_node\(" kg | rg -v "kind\s*=" && echo ">> Missing 'kind' on some nodes" || true
  rg -n "add_edge\(" kg | rg -v "predicate\s*=" && echo ">> Missing 'predicate' on some edges" || true
  rg -n "add_edge\(" kg | rg -v "provenance\s*=" && echo ">> Missing 'provenance' on some edges" || true
  rg -n "add_edge\(" kg | rg -v "evidence\s*=" && echo ">> Missing 'evidence' on some edges" || true
else
  echo "skip: ripgrep (rg) not found; skipping source scans"
fi

echo "[validate] ontology prefixes present?"
rg -n "HGNC:|ENSEMBL:|UNIPROT:|MONDO:|DOID:|HPO:|UBERON:|CL:|GO:|REACT:|CHEBI:" -n . >/dev/null \
  && echo "ok: saw standard prefixes" || echo "warn: no standard prefixes found"

echo "[validate] cellphonedb header sanity"
CPDB="data/external/kg/cellphonedb/cellphonedb_extracted/receptor_to_transcription_factor.csv"
if [ -f "$CPDB" ]; then
  head -n1 "$CPDB"
  python - <<'PY'
import pandas as pd, sys, os
p="data/external/kg/cellphonedb/cellphonedb_extracted/receptor_to_transcription_factor.csv"
if os.path.exists(p):
  df=pd.read_csv(p)
  cols={c.lower():c for c in df.columns}
  need=["receptor","tf"]
  missing=[c for c in need if c not in cols]
  if missing: 
    print("MISSING CPDB COLUMNS:",missing,"have:",list(df.columns))
  else:
    print("OK CPDB columns:",list(df.columns))
PY
else
  echo "skip: CPDB file not found"
fi

echo "[validate] duplicate edges (source, predicate, target, context) [requires a dump]"
# If you have an edge dump CSV/JSON, add a check here. Otherwise, add a small KG export later.
echo "todo: add export + dedupe check."