from __future__ import annotations
from pathlib import Path
import pandas as pd


def mine_hard_negatives(
    ranked: pd.DataFrame, top_k: int = 200, label_file: Path | None = None
) -> pd.DataFrame:
    """Select hard negatives from lower-ranked tail while excluding any known positives.

    If a label_file (tsv with column 'name' for positives) is provided, it is used to exclude known positives.
    Returns a DataFrame with columns: name, hard_negative (1), source.
    """
    df = ranked.copy()
    df["rank"] = range(1, len(df) + 1)
    tail = (
        df.sort_values(["assoc_score", "effect_size"], ascending=[False, False])
        .tail(max(1000, top_k))
        .copy()
    )
    if label_file and Path(label_file).exists():
        try:
            labels = pd.read_csv(label_file, sep="\t")
            pos = set(labels["name"].astype(str).str.upper())
            tail = tail[~tail["name"].astype(str).str.upper().isin(pos)]
        except Exception:
            pass
    out = tail[["name"]].drop_duplicates().copy()
    out["hard_negative"] = 1
    out["source"] = "auto_tail"
    return out
