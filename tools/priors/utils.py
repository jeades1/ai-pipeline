from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def prior_scores_for_genes(
    unified_path: Path | str = Path("data/processed/priors/unified_priors.parquet"),
    sources: Optional[Iterable[str]] = None,
    context_types: Optional[Iterable[str]] = None,
    agg: str = "max",
) -> pd.DataFrame:
    """
    Load unified priors and compute a per-gene prior score.

    Returns a DataFrame with columns:
      - name: UPPERCASED gene symbol
      - prior_score: aggregated score across selected contexts/sources

    Defaults to max aggregation across all available contexts, making it a
    conservative presence prior usable as a lightweight boost.
    """
    p = Path(unified_path)
    if not p.exists():
        return pd.DataFrame(columns=["name", "prior_score"])  # empty

    df = pd.read_parquet(p)
    if "gene_symbol" not in df.columns or "score" not in df.columns:
        return pd.DataFrame(columns=["name", "prior_score"])  # unexpected schema

    sub = df.copy()
    if sources:
        sr = {s.lower() for s in sources}
        sub = sub[sub["source"].str.lower().isin(sr)]
    if context_types:
        ct = {c.lower() for c in context_types}
        sub = sub[sub["context_type"].str.lower().isin(ct)]

    if sub.empty:
        return pd.DataFrame(columns=["name", "prior_score"])  # nothing to aggregate

    sub["name"] = sub["gene_symbol"].astype(str).str.upper()

    # Aggregation function mapping
    if agg == "mean":
        func = "mean"
    elif agg == "sum":
        func = "sum"
    else:
        func = "max"

    agg_df = sub.groupby("name", as_index=False).agg(score=("score", func))

    # Normalize to [0,1] for safe mixing with assoc scores
    denom = max(float(agg_df["score"].max()), 1e-9)
    agg_df["prior_score"] = agg_df["score"].astype(float) / denom
    return agg_df[["name", "prior_score"]]


def apply_prior_boost(
    ranked: pd.DataFrame, priors: pd.DataFrame, weight: float = 0.2
) -> pd.DataFrame:
    """
    Join prior scores onto a ranked candidates table and apply a small boost.

    Inputs:
      - ranked: DataFrame with at least columns ['name','assoc_score']
      - priors: DataFrame from prior_scores_for_genes with ['name','prior_score']
      - weight: scale applied to prior_score and added to assoc_score

    Returns a copy of ranked with columns 'prior_score' and updated 'assoc_score'.
    """
    if ranked is None or len(ranked) == 0 or priors is None or len(priors) == 0:
        return ranked.copy() if ranked is not None else pd.DataFrame()

    r = ranked.copy()
    r["name"] = r["name"].astype(str).str.upper()
    p = priors.copy()
    p["name"] = p["name"].astype(str).str.upper()

    merged = r.merge(p, on="name", how="left")
    if "prior_score" not in merged.columns:
        merged["prior_score"] = 0.0
    if "assoc_score" not in merged.columns:
        merged["assoc_score"] = 0.0
    merged["prior_score"] = (
        pd.to_numeric(merged["prior_score"], errors="coerce").fillna(0.0).astype(float)
    )
    merged["assoc_score"] = (
        pd.to_numeric(merged["assoc_score"], errors="coerce").fillna(0.0).astype(float)
        + weight * merged["prior_score"]
    )
    return merged
