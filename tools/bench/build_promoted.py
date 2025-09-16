#!/usr/bin/env python3
"""
Build artifacts/promoted.tsv from whatever the demo produced.

Search order (first match wins):
  artifacts/promoted_full.tsv
  artifacts/ranked.tsv
  artifacts/assoc.tsv
  artifacts/features.tsv
As a final fallback, scan all *.tsv under artifacts/ for a file that
contains (name|feature|gene) and (layer) and (type) columns.

Output: artifacts/promoted.tsv with columns: name,layer,type
Exit code: 0 on success, 2 on failure
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")
OUT = ART / "promoted.tsv"

preferred = [
    ART / "promoted_full.tsv",
    ART / "ranked.tsv",
    ART / "assoc.tsv",
    ART / "features.tsv",
]


def try_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def normalize(df: pd.DataFrame, src: Path) -> pd.DataFrame | None:
    # Candidate name columns in priority order
    name_cols = ["name", "feature", "gene"]
    layer_cols = ["layer"]
    type_cols = ["type", "kind"]

    def pick(colopts):
        for c in colopts:
            if c in df.columns:
                return c
        return None

    name_c = pick(name_cols)
    layer_c = pick(layer_cols)
    type_c = pick(type_cols)
    if not (name_c and layer_c and type_c):
        return None

    out = (
        df[[name_c, layer_c, type_c]]
        .rename(columns={name_c: "name", type_c: "type"})
        .dropna(subset=["name", "layer", "type"])
        .drop_duplicates()
        .sort_values(["layer", "name"])
    )
    if len(out) == 0:
        return None

    out.to_csv(OUT, sep="\t", index=False)
    print(f"[promoted] Wrote {OUT} ({len(out)} rows) from {src}")
    return out


def main():
    # Try preferred known outputs first
    for p in preferred:
        df = try_read(p)
        if df is not None:
            if normalize(df, p) is not None:
                return 0

    # Fallback: scan all TSVs under artifacts/
    candidates = sorted(ART.rglob("*.tsv"))
    for p in candidates:
        if p.name == "promoted.tsv":
            continue
        df = try_read(p)
        if df is not None:
            if normalize(df, p) is not None:
                return 0

    print(
        "[promoted] Could not construct promoted.tsv. Tried (in order):\n  - "
        + "\n  - ".join(str(p) for p in preferred)
        + "\n  ...and scanned *.tsv under artifacts/ but found no table with name/layer/type.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
