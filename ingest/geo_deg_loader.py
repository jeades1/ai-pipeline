# ingest/geo_deg_loader.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import List
import numpy as np

DEG_DIRS = [
    Path("data/external/geo/deg"),
    Path("data/external/geo") / "deg",  # tolerant
]


def _find_deg_tables() -> List[Path]:
    paths = []
    for root in DEG_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("DE_*vs*.csv"):
            paths.append(p)
        for p in root.rglob("DE_*.csv"):
            if p not in paths:
                paths.append(p)
    return paths


def load_deg_union(min_abs_logfc: float = 0.5, max_adj_p: float = 0.1) -> pd.DataFrame:
    """
    Returns a union of DEG tables with standardized columns:
    columns: ['gene','effect_size','p_value','dataset','direction']
    - effect_size: uses logFC if available else coerces from FC
    - p_value: adj.P.Val or p_adj or pvalue, coerced
    """
    tables = _find_deg_tables()
    records = []
    for p in tables:
        try:
            df = pd.read_csv(p)
        except Exception:
            try:
                df = pd.read_csv(p, sep="\t")
            except Exception:
                continue
        cols = {c.lower(): c for c in df.columns}
        # Heuristic column resolution
        gene_col = (
            cols.get("gene")
            or cols.get("symbol")
            or cols.get("genesymbol")
            or cols.get("hgnc_symbol")
        )
        if not gene_col:
            continue
        # effect (prefer logFC)
        logfc_col = (
            cols.get("logfc")
            or cols.get("log2fc")
            or cols.get("log2_fold_change")
            or cols.get("log2foldchange")
        )
        fc_col = cols.get("fc") or cols.get("foldchange") or cols.get("fold_change")
        p_col = (
            cols.get("adj.p.val")
            or cols.get("padj")
            or cols.get("p_adj")
            or cols.get("fdr")
            or cols.get("pvalue")
        )
        if not p_col:
            # fall back to unadjusted p value if present
            p_col = cols.get("p") or cols.get("p_val") or cols.get("p.value")

        tmp = pd.DataFrame()
        tmp["gene"] = df[gene_col].astype(str).str.strip().str.upper()

        if logfc_col and logfc_col in df:
            tmp["effect_size"] = df[logfc_col]
        elif fc_col and fc_col in df:
            # convert to log2FC if FC present and >0
            tmp["effect_size"] = (
                df[fc_col]
                .replace(0, pd.NA)
                .apply(lambda x: None if pd.isna(x) or x <= 0 else float(np.log2(x)))
            )
        else:
            continue

        if p_col and p_col in df:
            tmp["p_value"] = pd.to_numeric(df[p_col], errors="coerce")
        else:
            # if we truly have no p-value, skip (we need uncertainty)
            continue

        tmp = tmp.dropna(subset=["effect_size", "p_value"])
        tmp["dataset"] = p.parent.name  # e.g., GSE133288 / GSE200818
        tmp["direction"] = tmp["effect_size"].apply(
            lambda z: "up" if z >= 0 else "down"
        )

        # basic filtering
        tmp = tmp[tmp["p_value"] <= max_adj_p]
        tmp = tmp[tmp["effect_size"].abs() >= min_abs_logfc]

        records.append(tmp)

    if not records:
        return pd.DataFrame(
            columns=["gene", "effect_size", "p_value", "dataset", "direction"]
        )

    out = pd.concat(records, ignore_index=True).drop_duplicates(["gene", "dataset"])
    return out
