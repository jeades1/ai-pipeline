# ingest/loaders.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

DEMO_DIR = None  # removed; demo content purged


def _load_path(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _discover_real_defaults():
    """
    Heuristics based on your repo:
    - Prefer processed parquet if present: data/processed/features.parquet
    - Associations: try data/processed/labs_clean.parquet or outputs/predictions/assoc.parquet
    """
    feat = Path("data/processed/features.parquet")
    assoc_candidates = [
        Path("data/processed/labs_clean.parquet"),
        Path("outputs/predictions/assoc.parquet"),
        Path("data/processed/assoc.parquet"),
    ]
    assoc = next((p for p in assoc_candidates if p.exists()), None)
    return feat if feat.exists() else None, assoc


def load_or_create_demo_data(*args, **kwargs):
    raise RuntimeError(
        "Demo data generation has been removed. Provide real inputs under data/external or data/processed."
    )
