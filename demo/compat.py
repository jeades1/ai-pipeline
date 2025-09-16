# demo/compat.py
# Deprecated: demo compatibility shims removed.
# These functions now no-op and inform callers to migrate.
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _as_dataframe(
    records_or_df: Union[pd.DataFrame, List[Dict[str, Any]]]
) -> pd.DataFrame:
    return (
        records_or_df
        if isinstance(records_or_df, pd.DataFrame)
        else pd.DataFrame.from_records(records_or_df or [])
    )


def safe_write_biomarker_cards(*args: Any, **kwargs: Any) -> Path:
    outdir = _ensure_dir(
        kwargs.get("outdir")
        or (args[1] if len(args) > 1 else "artifacts/biomarker_cards")
    )
    (_as_dataframe(args[0]) if args else pd.DataFrame()).to_csv(
        Path(outdir) / "biomarkers.csv", index=False
    )
    (Path(outdir) / "README.md").write_text(
        "Deprecated demo path; migrate to writers/cards API.\n"
    )
    return Path(outdir)


def safe_write_experiment_cards(*args: Any, **kwargs: Any) -> Path:
    outdir = _ensure_dir(
        kwargs.get("outdir")
        or (args[1] if len(args) > 1 else "artifacts/experiment_cards")
    )
    (_as_dataframe(args[0]) if args else pd.DataFrame()).to_csv(
        Path(outdir) / "experiments.csv", index=False
    )
    (Path(outdir) / "README.md").write_text(
        "Deprecated demo path; migrate to writers/cards API.\n"
    )
    return Path(outdir)


def safe_compare_to_benchmark(*args: Any, **kwargs: Any) -> Path:
    out_path = Path(
        kwargs.get("out_path")
        or (args[1] if len(args) > 1 else "artifacts/benchmark.tsv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (_as_dataframe(args[0]) if args else pd.DataFrame()).to_csv(
        out_path, sep="\t", index=False
    )
    return out_path
