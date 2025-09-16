from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import zipfile
import io
import pandas as pd


def _read_hpa_tissue(path: Path) -> pd.DataFrame:
    """
    Read HPA rna_tissue (zip or plain .tsv) into a tidy dataframe:
      columns commonly include: Gene, Tissue, TPM, NX, etc.
    We'll use Gene (symbol), Tissue, and TPM if present (else NX as fallback).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if str(p).endswith(".tsv"):
        df = pd.read_csv(p, sep="\t")
    else:
        with zipfile.ZipFile(p, "r") as zf:
            # Find the TSV inside (usually rna_tissue.tsv)
            tsv_name = next((n for n in zf.namelist() if n.endswith(".tsv")), None)
            if not tsv_name:
                raise ValueError("No .tsv in HPA zip")
            with zf.open(tsv_name) as f:
                buf = io.TextIOWrapper(f, encoding="utf-8")
                df = pd.read_csv(buf, sep="\t")

    return df


def build_hpa_prior(
    path: Path,
    keep_tissues: Optional[Iterable[str]] = None,
    min_value: float = 0.0,
    out_parquet: Optional[Path] = None,
    out_tsv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build tissue expression priors from HPA RNA tissue data.

    Output schema:
      - gene_symbol, tissue, score, source
    Score uses TPM if present, else NX. Scores are normalized per tissue to [0,1].
    """
    df = _read_hpa_tissue(Path(path))

    # HPA has multiple TSV schemas. Try RNA tissue (Gene|Tissue|TPM|NX)
    cols = {c.lower(): c for c in df.columns}
    # Prefer human-readable symbol when available
    gene_name_col = (
        cols.get("gene name") or cols.get("gene_symbol") or cols.get("symbol")
    )
    gene_col = cols.get("gene") or cols.get("genes")
    tissue_col = cols.get("tissue") or cols.get("tissues")
    tpm_col = cols.get("tpm") or cols.get("ntpm")
    nx_col = cols.get("nx")
    if not gene_col or not tissue_col:
        # If this is the big catalog TSV (no Tissue column), provide a targeted hint
        raise ValueError(
            "HPA tissue TSV missing Gene/Tissue columns — expected rna_tissue.tsv; got a catalog TSV with metadata only"
        )

    # Choose value column
    val_col = tpm_col or nx_col
    if not val_col:
        raise ValueError("HPA tissue TSV missing TPM and NX columns")

    # Choose gene symbol column: prefer Gene name (symbol) else fallback to Ensembl Gene
    if gene_name_col and gene_name_col in df.columns:
        tidy = df[[gene_name_col, tissue_col, val_col]].rename(
            columns={
                gene_name_col: "gene_symbol",
                tissue_col: "tissue",
                val_col: "value",
            }
        )
    else:
        tidy = df[[gene_col, tissue_col, val_col]].rename(
            columns={gene_col: "gene_symbol", tissue_col: "tissue", val_col: "value"}
        )

    if keep_tissues:
        terms = [t.lower() for t in keep_tissues]
        mask = tidy["tissue"].str.lower().apply(lambda x: any(t in x for t in terms))
        tidy = tidy[mask]

    if min_value > 0:
        tidy = tidy[tidy["value"] >= float(min_value)]

    # Normalize per tissue
    tidy["score"] = tidy.groupby("tissue")["value"].transform(
        lambda s: s / max(float(s.max()), 1e-9)
    )
    out = tidy[["gene_symbol", "tissue", "score"]].dropna(subset=["gene_symbol"]).copy()
    out["source"] = "HPA"

    if out_parquet:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_parquet, index=False)
    if out_tsv:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_tsv, sep="\t", index=False)

    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build HPA tissue priors")
    ap.add_argument(
        "--zip",
        type=Path,
        default=Path("data/external/hpa/rna_tissue.tsv.zip"),
        help="Path to HPA rna_tissue.tsv.zip (or .tsv)",
    )
    ap.add_argument(
        "--tissues",
        nargs="*",
        default=["kidney", "cortex", "medulla"],
        help="Case-insensitive substrings to keep tissues",
    )
    ap.add_argument("--min", type=float, default=0.0)
    ap.add_argument(
        "--out", type=Path, default=Path("data/processed/priors/hpa_prior.parquet")
    )
    ap.add_argument(
        "--out-tsv", type=Path, default=Path("data/processed/priors/hpa_prior.tsv")
    )
    args = ap.parse_args()

    df = build_hpa_prior(
        args.zip,
        keep_tissues=args.tissues,
        min_value=args.min,
        out_parquet=args.out,
        out_tsv=args.out_tsv,
    )
    print(f"[priors] HPA -> {len(df):,} rows; wrote {args.out} and {args.out_tsv}")


if __name__ == "__main__":
    main()
