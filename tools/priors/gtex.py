from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

import gzip
import pandas as pd


def _read_gct(
    gct_path: Path, keep_tissues: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Read a GTEx-style GCT (gz or plain) with two header lines into long form:
      columns: gene_id, gene_symbol, tissue, tpm.
    Note: If the file is not medians-by-tissue but has many columns, each column
    is treated as a "tissue/column" label; optional filtering still applies by substring.
    """
    p = Path(gct_path)
    if not p.exists():
        raise FileNotFoundError(p)

    # GCT format has first two header lines as counts; then header row with Name, Description, tissues/columns...
    if str(p).endswith(".gz"):
        opener = lambda path: gzip.open(path, "rt")
    else:
        opener = lambda path: open(path, "r")

    with opener(p) as fh:
        _ = fh.readline()
        _ = fh.readline()
        df = pd.read_csv(fh, sep="\t")

    # Expect Name, Description, then tissues
    if not {"Name", "Description"}.issubset(df.columns):
        raise ValueError("Unexpected GCT format: missing Name/Description")

    id_col = "Name"
    sym_col = "Description"
    tissue_cols = [c for c in df.columns if c not in (id_col, sym_col)]
    # If keep_tissues provided, reduce columns early to avoid an expensive melt
    if keep_tissues:
        terms = [t.lower() for t in keep_tissues]
        filtered = [c for c in tissue_cols if any(t in c.lower() for t in terms)]
        # If no columns match (common for sample IDs like GTEX-xxxxx), keep all and defer filtering to later stages
        if filtered:
            tissue_cols = filtered

    long = df.melt(
        id_vars=[id_col, sym_col],
        value_vars=tissue_cols,
        var_name="tissue",
        value_name="tpm",
    )
    long = long.rename(columns={id_col: "gene_id", sym_col: "gene_symbol"})
    return long


def _read_gct_wide_subset(
    gct_path: Path,
    sample_cols: List[str] | None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read a GCT (gz or plain) returning a wide dataframe with only selected sample columns.
    Returns (df, all_sample_columns_in_file).
    """
    p = Path(gct_path)
    if not p.exists():
        raise FileNotFoundError(p)

    if str(p).endswith(".gz"):
        opener = lambda path: gzip.open(path, "rt")
    else:
        opener = lambda path: open(path, "r")

    with opener(p) as fh:
        _ = fh.readline()
        _ = fh.readline()
        header = fh.readline().rstrip("\n")
        cols = header.split("\t")

        id_col = "Name"
        sym_col = "Description"
        if id_col not in cols or sym_col not in cols:
            raise ValueError("Unexpected GCT header: missing Name/Description")

        all_sample_cols = [c for c in cols if c not in (id_col, sym_col)]
        # Decide which columns to read
        if sample_cols:
            selected = [id_col, sym_col] + [
                c for c in all_sample_cols if c in sample_cols
            ]
        else:
            selected = cols

        df = pd.read_csv(
            fh,
            sep="\t",
            header=None,
            names=cols,
            usecols=selected,
        )
    return df, all_sample_cols


def build_gtex_prior(
    gct_gz: Path,
    keep_tissues: Optional[Iterable[str]] = None,
    min_tpm: float = 0.0,
    out_parquet: Optional[Path] = None,
    out_tsv: Optional[Path] = None,
    column_map: Optional[Dict[str, str]] = None,
    aggregate: str = "median",
) -> pd.DataFrame:
    """
    Build tissue expression priors from GTEx v8 median TPM.

    Output schema:
      - gene_id, gene_symbol, tissue, score, source
    Where score is normalized TPM in [0,1] per tissue (TPM / (TPM max across genes for that tissue)).
    """
    # Quick header check to avoid OOM: if this is sample-level (thousands of columns) and we don't have a mapping,
    # require a mapping so we can aggregate by tissue without melting.
    try:
        _, all_samples = _read_gct_wide_subset(Path(gct_gz), sample_cols=[])
    except Exception:
        all_samples = []

    if (not column_map) and len(all_samples) > 100:
        raise ValueError(
            "GTEx GCT appears to be sample-level (many columns). Provide --sample-attrs or --col-map to map samples → tissues, "
            "so we can aggregate without exploding memory."
        )

    # If we have a mapping, read only mapped columns (optionally restricted by keep_tissues) and aggregate in wide form
    if column_map:
        # Determine which samples belong to requested tissues
        target_terms = [t.lower() for t in (keep_tissues or [])]

        def is_target_tissue(label: str) -> bool:
            return (not target_terms) or any(t in label.lower() for t in target_terms)

        # Select samples whose mapped tissue matches targets
        selected_samples = [
            s for s, tis in column_map.items() if is_target_tissue(str(tis))
        ]
        # Fallback: if filtering removed all, use all mapped samples
        if not selected_samples:
            selected_samples = list(column_map.keys())
        wide, all_samples = _read_gct_wide_subset(
            Path(gct_gz), sample_cols=selected_samples
        )

        id_col, sym_col = "Name", "Description"
        # Build groups: tissue -> list of sample columns present in df
        present = set(wide.columns)
        groups: Dict[str, List[str]] = {}
        for s in selected_samples:
            if s in present:
                tis = str(column_map.get(s))
                groups.setdefault(tis, []).append(s)

        agg_choice = aggregate.lower() if isinstance(aggregate, str) else "median"
        if agg_choice not in {"median", "mean", "max"}:
            agg_choice = "median"

        agg_df = pd.DataFrame(
            {
                "gene_id": wide[id_col].astype(str),
                "gene_symbol": wide[sym_col].astype(str),
            }
        )
        for tis, cols in groups.items():
            if not cols:
                continue
            if agg_choice == "median":
                agg_series = wide[cols].median(axis=1, numeric_only=True)
            elif agg_choice == "mean":
                agg_series = wide[cols].mean(axis=1, numeric_only=True)
            else:
                agg_series = wide[cols].max(axis=1, numeric_only=True)
            agg_df[tis] = agg_series

        # Melt to long
        value_cols = [c for c in agg_df.columns if c not in ("gene_id", "gene_symbol")]
        df = agg_df.melt(
            id_vars=["gene_id", "gene_symbol"],
            value_vars=value_cols,
            var_name="tissue",
            value_name="tpm",
        )
    else:
        # Legacy path for pre-aggregated (median) matrices
        df = _read_gct(Path(gct_gz), keep_tissues=keep_tissues)

    # Optional remap for non-median matrices: map sample/column names to tissue labels then aggregate
    if column_map:
        # Remap only if current 'tissue' values look like sample IDs (keys of column_map)
        # When we already aggregated in wide form above, df['tissue'] holds tissue labels (values), so skip remap.
        if df["tissue"].isin(set(column_map.keys())).any():
            df["tissue"] = df["tissue"].map(lambda c: column_map.get(str(c), None))
            df = df.dropna(subset=["tissue"])  # keep only mapped tissues
            agg_func = "median"
            agg_choice = aggregate.lower() if isinstance(aggregate, str) else "median"
            if agg_choice in {"mean", "max"}:
                agg_func = agg_choice
            df = df.groupby(["gene_id", "gene_symbol", "tissue"], as_index=False).agg(
                tpm=("tpm", agg_func)
            )

    # keep_tissues already applied in _read_gct to minimize memory/time; keep here for safety in case of column_map
    if keep_tissues:
        terms = [t.lower() for t in keep_tissues]
        mask = df["tissue"].str.lower().apply(lambda x: any(t in x for t in terms))
        df = df[mask]

    if min_tpm > 0:
        df = df[df["tpm"] >= float(min_tpm)]

    # Normalize per tissue to [0,1] to form a comparable prior score
    df["score"] = df.groupby("tissue")["tpm"].transform(
        lambda s: s / max(float(s.max()), 1e-9)
    )
    out = df[["gene_id", "gene_symbol", "tissue", "score"]].copy()
    out = out[out["gene_id"].notna()].copy()
    out["source"] = "GTEx"

    if out_parquet:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_parquet, index=False)
    if out_tsv:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_tsv, sep="\t", index=False)

    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build GTEx priors")
    ap.add_argument(
        "--gct",
        type=Path,
        default=Path("data/external/gtex/GTEx_v8_gene_median_tpm.gct.gz"),
        help="Path to GTEx median TPM .gct[.gz] (median-by-tissue or sample-by-tissue)",
    )
    ap.add_argument(
        "--tissues",
        nargs="*",
        default=["kidney", "cortex", "medulla"],
        help="Case-insensitive substrings to keep tissues",
    )
    ap.add_argument("--min-tpm", type=float, default=0.0)
    ap.add_argument(
        "--col-map",
        type=Path,
        default=None,
        help="Optional TSV/CSV: columns 'column' and 'tissue' to map GCT columns → tissue labels",
    )
    ap.add_argument(
        "--aggregate",
        type=str,
        default="median",
        choices=["median", "mean", "max"],
        help="Aggregation when multiple columns map to same tissue label",
    )
    ap.add_argument(
        "--sample-attrs",
        type=Path,
        default=None,
        help="Optional GTEx SampleAttributes (TSV) to auto-derive column→tissue mapping (uses SMTSD)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("data/processed/priors/gtex_prior.parquet")
    )
    ap.add_argument(
        "--out-tsv", type=Path, default=Path("data/processed/priors/gtex_prior.tsv")
    )
    args = ap.parse_args()

    # Load optional column map
    col_map = None
    if args.col_map and args.col_map.exists():
        m = pd.read_csv(args.col_map, sep="\t|,", engine="python")
        # Flexible column headers: column/name and tissue/label
        col_col = next((c for c in m.columns if c.lower() in {"column", "name"}), None)
        tis_col = next((c for c in m.columns if c.lower() in {"tissue", "label"}), None)
        if col_col and tis_col:
            col_map = {
                str(r[col_col]): str(r[tis_col])
                for _, r in m.iterrows()
                if pd.notna(r[col_col]) and pd.notna(r[tis_col])
            }
    elif args.sample_attrs and args.sample_attrs.exists():
        # Build mapping from GTEx SampleAttributes file (expects columns SAMPID and SMTSD or SMTS)
        attrs = pd.read_csv(args.sample_attrs, sep="\t")
        samp_col = next((c for c in attrs.columns if c.upper() == "SAMPID"), None)
        tis_col = next(
            (c for c in attrs.columns if c.upper() in {"SMTSD", "SMTS"}), None
        )
        if samp_col and tis_col:
            col_map = {
                str(r[samp_col]): str(r[tis_col])
                for _, r in attrs[[samp_col, tis_col]].dropna().iterrows()
            }

    df = build_gtex_prior(
        args.gct,
        keep_tissues=args.tissues,
        min_tpm=args.min_tpm,
        out_parquet=args.out,
        out_tsv=args.out_tsv,
        column_map=col_map,
        aggregate=args.aggregate,
    )
    print(f"[priors] GTEx -> {len(df):,} rows; wrote {args.out} and {args.out_tsv}")


if __name__ == "__main__":
    main()
