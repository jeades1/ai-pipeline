from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd


def _per_subject_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy()
    g["cr_slope_24h"] = (
        g.set_index("timestamp")["creatinine_mg_dL"]
        .rolling("24H")
        .apply(
            lambda s: (s.iloc[-1] - s.iloc[0])
            / max((s.index[-1] - s.index[0]).total_seconds() / 3600, 1e-9),
            raw=False,
        )
        .reset_index(drop=True)
    )
    g["cr_min_7d"] = (
        g.set_index("timestamp")["creatinine_mg_dL"].rolling("7D").min().values
    )
    g["cr_max_7d"] = (
        g.set_index("timestamp")["creatinine_mg_dL"].rolling("7D").max().values
    )
    g["cr_mean_48h"] = (
        g.set_index("timestamp")["creatinine_mg_dL"].rolling("48H").mean().values
    )
    # Last-known summary per subject at end of series
    last = g.iloc[[-1]].copy()
    last["n_obs"] = len(g)
    return last[
        (
            [
                "subject_id",
                "timestamp",
                "age",
                "creatinine_mg_dL",
                "cr_slope_24h",
                "cr_min_7d",
                "cr_max_7d",
                "cr_mean_48h",
                "aki_flag",
                "n_obs",
            ]
            if "age" in g.columns
            else [
                "subject_id",
                "timestamp",
                "creatinine_mg_dL",
                "cr_slope_24h",
                "cr_min_7d",
                "cr_max_7d",
                "cr_mean_48h",
                "aki_flag",
                "n_obs",
            ]
        )
    ]


def build_features(
    clean_df: pd.DataFrame,
    out_parquet: Optional[Path] = None,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Collapse a subject's series to a compact row of features.
    """
    cols = ["subject_id", "timestamp", "creatinine_mg_dL"]
    if not set(cols).issubset(clean_df.columns):
        raise ValueError(
            "clean_df must include subject_id, timestamp, creatinine_mg_dL"
        )

    features = (
        clean_df.groupby("subject_id", group_keys=False)
        .apply(_per_subject_features)
        .reset_index(drop=True)
    )

    if out_parquet:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out_parquet, index=False)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_csv, index=False)
    return features
