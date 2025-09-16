from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


def preprocess_labs_to_features(
    labs_parquet: str = "data/working/labs.parquet",
    labels_df: pd.DataFrame | None = None,
    out_parquet: str = "data/processed/features.parquet",
) -> str:
    """
    Aggregate per-subject features from the labs table and join labels.
    - Works even if only a subset of columns is present.
    """
    labs_p = Path(labs_parquet)
    out_p = Path(out_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not labs_p.exists():
        raise FileNotFoundError(f"Missing labs parquet: {labs_p}")

    df = pd.read_parquet(labs_p)

    if "subject_id" not in df.columns:
        raise ValueError("labs parquet missing `subject_id`.")

    # Identify numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove label-like numeric columns from aggregation (will be joined later)
    for drop_col in ("aki_flag", "aki_label", "label"):
        if drop_col in numeric_cols:
            numeric_cols.remove(drop_col)

    # Use explicit mapping to aggregation functions (Mapping[str, list[str]])
    if numeric_cols:
        g = df.groupby("subject_id")[numeric_cols].agg(["min", "max", "mean"])
        # flatten MultiIndex columns like ('creatinine','min') -> 'creatinine_min'
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        features = g.reset_index()
    else:
        # If no numeric columns, just unique subject_id
        features = df[["subject_id"]].drop_duplicates().copy()

    # Attach labels if provided
    if labels_df is not None and not labels_df.empty:
        # Prefer generic 'label', fallback to 'aki_label'
        label_cols = [c for c in ("label", "aki_label") if c in labels_df.columns]
        if "subject_id" in labels_df.columns and label_cols:
            lc = label_cols[0]
            tmp = (
                labels_df[["subject_id", lc]]
                .drop_duplicates()
                .rename(columns={lc: "label"})
            )
            features = features.merge(tmp, on="subject_id", how="left")

    # Ensure a label column exists
    if "label" not in features.columns and "aki_label" in features.columns:
        features = features.rename(columns={"aki_label": "label"})
    if "label" in features.columns:
        features["label"] = (
            pd.to_numeric(features["label"], errors="coerce").fillna(0).astype(int)
        )
    else:
        features["label"] = 0

    features.to_parquet(out_p, index=False)
    return str(out_p)
