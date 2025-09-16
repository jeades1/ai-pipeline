from __future__ import annotations
from pathlib import Path
import pandas as pd


def ingest_labs(
    in_csv: str = "data/raw/labs.csv",
    out_parquet: str = "data/working/labs.parquet",
) -> str:
    """
    Ingest open-access labs CSV -> Parquet.
    Expects at least: subject_id, charttime, item, value (names flexible).
    """
    in_p = Path(in_csv)
    out_p = Path(out_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_p}")

    df = pd.read_csv(in_p)
    # Normalize a few common patterns
    # Ensure subject_id exists
    if "subject_id" not in df.columns:
        # try alternate common names
        for alt in ("SUBJECT_ID", "pid", "patient_id", "id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "subject_id"})
                break
        if "subject_id" not in df.columns:
            raise ValueError("No `subject_id` column found in input CSV.")

    # Ensure time column if present is parsed to datetime
    for c in ("charttime", "time", "timestamp", "event_time", "datetime"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.rename(columns={c: "charttime"})
            break

    df.to_parquet(out_p, index=False)
    return str(out_p)


def ensure_labels_from_csv(
    labels_csv: str = "data/raw/sample.csv",
    labs_parquet: str = "data/working/labs.parquet",
) -> pd.DataFrame:
    """
    Ensure we have a labels dataframe keyed by `subject_id` with a binary label column.
    Priority:
    1) If labels_csv exists and has subject_id + label (or aki_label), use it.
    2) Else if labs has `label` (or `aki_flag`), derive labels from it (any positive -> 1).
      3) Else return empty with unique subjects (all 0).
    Returns a DataFrame with at least columns: subject_id, label.
    """
    labs_p = Path(labs_parquet)
    labels_p = Path(labels_csv)

    if not labs_p.exists():
        raise FileNotFoundError(f"Missing labs parquet: {labs_p}")

    df_labs = pd.read_parquet(labs_p)

    # 1) Direct labels file
    if labels_p.exists():
        df_lab = pd.read_csv(labels_p)
        # Flexible renaming
        colmap = {}
        if "SUBJECT_ID" in df_lab.columns and "subject_id" not in df_lab.columns:
            colmap["SUBJECT_ID"] = "subject_id"
        # normalize label column name
        if "aki_label" in df_lab.columns:
            colmap["aki_label"] = "label"
        elif "label" in df_lab.columns:
            colmap["label"] = "label"
        df_lab = df_lab.rename(columns=colmap)

        if "subject_id" in df_lab.columns and "label" in df_lab.columns:
            df_lab["label"] = df_lab["label"].astype(int).clip(0, 1)
            return df_lab[["subject_id", "label"]].drop_duplicates()

    # 2) Derive labels from labs if an AKI flag exists
    if "label" in df_labs.columns:
        df_flag = df_labs.groupby("subject_id", as_index=False)["label"].max(
            numeric_only=True
        )
        df_flag = (
            df_flag.to_frame() if not isinstance(df_flag, pd.DataFrame) else df_flag
        )
        df_flag["label"] = df_flag["label"].fillna(0).astype(int).clip(0, 1)
        return df_flag
    if "aki_flag" in df_labs.columns:
        df_flag = df_labs.groupby("subject_id", as_index=False)["aki_flag"].max(
            numeric_only=True
        )
        df_flag = (
            df_flag.to_frame() if not isinstance(df_flag, pd.DataFrame) else df_flag
        )
        # Rename column safely
        if "aki_flag" in df_flag.columns:
            df_flag = df_flag.assign(label=df_flag.pop("aki_flag"))
        df_flag["label"] = df_flag["label"].fillna(0).astype(int).clip(0, 1)
        return df_flag

    # 3) Fallback: zero labels for all subjects present
    df_zero = df_labs[["subject_id"]].drop_duplicates().assign(label=0)
    return df_zero
