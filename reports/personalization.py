from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def write_stratified_rankings(
    ranked: List[Dict[str, Any]] | pd.DataFrame,
    outdir: Path,
    strata: Dict[str, List[str]] | None = None,
) -> Path:
    """
    Minimal patient/stratum-specific export. By default, creates two strata
    (early vs late stage) using simple name heuristics as placeholders.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = (
        ranked
        if isinstance(ranked, pd.DataFrame)
        else pd.DataFrame.from_records(ranked)
    )
    if strata is None:
        strata = {
            "early": ["NGAL", "KIM-1", "TIMP-2·IGFBP7"],
            "late": ["IL-18", "C5A", "SERPINA3"],
        }
    for name, features in strata.items():
        mask = (
            df["name"].str.upper().isin([f.upper() for f in features])
            if "name" in df.columns
            else pd.Series([], dtype=bool)
        )
        part = df[mask].copy() if mask.any() else df.head(50).copy()
        part.to_csv(outdir / f"ranked_{name}.tsv", sep="\t", index=False)
    return outdir
