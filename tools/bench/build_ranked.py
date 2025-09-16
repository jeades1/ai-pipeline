#!/usr/bin/env python3
# Build artifacts/ranked.tsv from artifacts/assoc.tsv (p_value asc, |effect_size| desc)
import sys
from pathlib import Path
import pandas as pd

ART = Path("artifacts")
ASSOC = ART / "assoc.tsv"
OUT = ART / "ranked.tsv"

if not ASSOC.exists():
    sys.exit("[build_ranked] artifacts/assoc.tsv not found. Run `make demo` first.")

df = pd.read_csv(ASSOC, sep="\t")
# normalize columns and keep gene rows
ren = {"feature": "name"}
for k, v in ren.items():
    if k in df.columns and v not in df.columns:
        df[v] = df[k]
df["layer"] = "transcriptomic"
df["type"] = "gene"
df["assoc_score"] = (
    -df["p_value"].clip(lower=1e-300).map(float).rpow(10).map(lambda x: -x)
)  # monotonic to p
# sort: p_value asc, then |effect_size| desc
df = df.sort_values(["p_value", df["effect_size"].abs().name], ascending=[True, False])
out = df[
    ["name", "layer", "type", "assoc_score", "p_value", "effect_size", "dataset"]
].dropna()
out.to_csv(OUT, sep="\t", index=False)
print("[build_ranked] Wrote", OUT, "rows:", len(out))
