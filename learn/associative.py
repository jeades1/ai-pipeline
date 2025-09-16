# learn/associative.py
from __future__ import annotations

import math
from typing import Dict, Any, Union

import pandas as pd


def _safe_neg_log10(p: float) -> float:
    try:
        # Clamp p to avoid -inf and numeric issues
        p = float(p)
    except Exception:
        return 0.0
    p = max(min(p, 1.0), 1e-300)
    return -math.log10(p)


def _assoc_row_score(row: pd.Series) -> float:
    # Keep sign of effect_size to allow up/down ranking; if you prefer
    # purely strength (direction-agnostic), use abs(row["effect_size"])
    eff = float(row.get("effect_size", 0.0))
    neglogp = _safe_neg_log10(row.get("p_value", 1.0))
    # Tunable weights
    return 0.7 * eff + 0.3 * neglogp


def _ensure_assoc_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize feature/name columns
    if "feature" not in df.columns and "name" in df.columns:
        df["feature"] = df["name"]
    if "name" not in df.columns and "feature" in df.columns:
        df["name"] = df["feature"]

    # Expected minimal columns
    for col, default in [
        ("effect_size", 0.0),
        ("p_value", 1.0),
        ("dataset", "unknown"),
        ("direction", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    # Cast types
    df["effect_size"] = pd.to_numeric(df["effect_size"], errors="coerce").fillna(0.0)
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce").fillna(1.0)
    df["dataset"] = df["dataset"].astype(str)
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str)

    return df[["feature", "name", "effect_size", "p_value", "dataset", "direction"]]


def _ensure_features_df(
    assoc: pd.DataFrame, features: pd.DataFrame | None
) -> pd.DataFrame:
    if features is not None and len(features) > 0:
        f = features.copy()
        # Standardize schema
        if "feature" not in f.columns and "name" in f.columns:
            f["feature"] = f["name"]
        if "layer" not in f.columns:
            f["layer"] = "transcriptomic"
        if "type" not in f.columns:
            f["type"] = "gene"
        return f[["feature", "layer", "type"]].drop_duplicates()

    # Fallback: derive from assoc
    base = assoc[["feature"]].drop_duplicates()
    base["layer"] = "transcriptomic"
    base["type"] = "gene"
    return base


def rank_candidates(
    datasets: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    kg: Any = None,
) -> pd.DataFrame:
    """
    Rank candidate biomarkers by an association score.
    Returns a DataFrame with columns:
      ['name','layer','type','assoc_score','p_value','effect_size','provenance']
    """
    # Normalize inputs
    if isinstance(datasets, pd.DataFrame):
        assoc_df = _ensure_assoc_df(datasets)
        feats_df = _ensure_features_df(assoc_df, None)
    elif isinstance(datasets, dict):
        assoc_df = _ensure_assoc_df(datasets.get("assoc", pd.DataFrame()))
        feats_df = _ensure_features_df(assoc_df, datasets.get("features"))
    else:
        raise TypeError(
            "datasets must be a pandas DataFrame or dict with keys 'assoc' and 'features'"
        )

    # Compute association score
    assoc_df = assoc_df.copy()
    assoc_df["assoc_score"] = assoc_df.apply(_assoc_row_score, axis=1)

    # Merge features metadata
    feats_df = feats_df.drop_duplicates(subset=["feature"])
    merged = assoc_df.merge(feats_df, on="feature", how="left")

    # Canonical output columns
    out = pd.DataFrame(
        {
            "name": merged["name"].astype(str),
            "layer": merged["layer"].fillna("transcriptomic").astype(str),
            "type": merged["type"].fillna("gene").astype(str),
            "assoc_score": pd.to_numeric(merged["assoc_score"], errors="coerce").fillna(
                0.0
            ),
            "p_value": pd.to_numeric(merged["p_value"], errors="coerce").fillna(1.0),
            "effect_size": pd.to_numeric(merged["effect_size"], errors="coerce").fillna(
                0.0
            ),
            "provenance": merged["dataset"].astype(str),
        }
    )

    # Sort best-first
    out = out.sort_values(
        ["assoc_score", "effect_size"], ascending=[False, False], kind="mergesort"
    )
    out = out.reset_index(drop=True)
    return out
