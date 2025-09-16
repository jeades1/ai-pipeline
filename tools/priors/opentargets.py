from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _find_parquet_parts(base_dir: Path) -> list[Path]:
    if base_dir.is_file() and base_dir.suffix == ".parquet":
        return [base_dir]
    if base_dir.is_dir():
        parts = sorted(p for p in base_dir.glob("**/*.parquet") if p.is_file())
        return parts
    return []


def build_opentargets_prior(
    source_dir: Path,
    out_parquet: Optional[Path] = None,
    out_tsv: Optional[Path] = None,
    disease_terms: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """
    Build gene-disease association priors from Open Targets association_overall_direct parquet files.

    Output schema:
      - gene_id: Ensembl gene ID (e.g., ENSG00000141510)
      - gene_symbol: optional (if available in dataset; else NA)
      - disease_id: ontology ID (EFO/Orphanet/etc.)
      - disease_label: human-readable disease label if available
      - score: overall association score (float)
      - source: constant 'OpenTargets'

    Notes:
      - We filter to disease terms containing any of the provided disease_terms in a case-insensitive manner,
        matching on disease_label and disease_id. If disease_terms is None, we keep all.
      - We don't attempt online mapping for gene symbols; if the dataset provides fields, we'll pass them through.
    """
    base = Path(source_dir)
    parts = _find_parquet_parts(base)
    if not parts:
        raise FileNotFoundError(f"No parquet files found under {base}")

    # Read all parts with a forgiving set of expected columns
    cols_candidates = [
        "targetId",
        "diseaseId",
        "score",
        "evidenceCount",
        # optional in other dumps
        "targetFromSourceId",
        "targetSymbol",
        "diseaseFromSourceMappedId",
        "diseaseLabel",
        "diseaseName",
        "diseaseFromSource",
        "overallScore",
    ]

    frames: list[pd.DataFrame] = []
    for p in parts:
        df = pd.read_parquet(p)
        keep = [c for c in cols_candidates if c in df.columns]
        if not keep:
            continue
        df = df[keep].copy()
        frames.append(df)

    if not frames:
        raise ValueError("OpenTargets parquet did not contain expected columns")

    df = pd.concat(frames, ignore_index=True)
    # Normalize columns and create a label field for filtering
    df["gene_id"] = df.get("targetId", pd.NA)
    df["gene_symbol"] = df.get("targetSymbol", pd.NA)
    df["disease_id"] = df.get("diseaseId", df.get("diseaseFromSourceMappedId", pd.NA))
    if "score" not in df.columns:
        df["score"] = df["overallScore"] if "overallScore" in df.columns else pd.NA
    # Combine potential disease label fields for robust substring filtering
    df["disease_label_text"] = (
        df.get("diseaseLabel", pd.Series([pd.NA] * len(df)))
        .fillna(df.get("diseaseName", pd.NA))
        .fillna(df.get("diseaseFromSource", pd.NA))
        .astype(str)
    )

    # Optional filter by disease terms across labels and IDs before narrowing columns
    if disease_terms:
        terms = [t.lower() for t in disease_terms]

        def _has_term(x: str) -> bool:
            x = str(x).lower()
            return any(t in x for t in terms)

        mask = df["disease_label_text"].apply(_has_term) | df["disease_id"].astype(
            str
        ).apply(_has_term)
        filtered = df[mask]
        # Fallback: if filtering collapses to zero, keep unfiltered to avoid empty priors
        if len(filtered) > 0:
            df = filtered

    out = df[["gene_id", "gene_symbol", "disease_id", "score"]].copy()
    out = out.dropna(subset=["gene_id", "disease_id", "score"], how="any")

    # Score threshold
    if min_score > 0:
        out = out[out["score"] >= float(min_score)]

    out = out.drop_duplicates().reset_index(drop=True)
    out["source"] = "OpenTargets"

    if out_parquet:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_parquet, index=False)
    if out_tsv:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_tsv, sep="\t", index=False)

    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build Open Targets priors")
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("data/external/opentargets/25.06/association_overall_direct"),
        help="Directory containing Open Targets association_overall_direct parquet parts",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/priors/opentargets_prior.parquet"),
    )
    ap.add_argument(
        "--out-tsv",
        type=Path,
        default=Path("data/processed/priors/opentargets_prior.tsv"),
    )
    ap.add_argument(
        "--terms", nargs="+", default=[], help="Disease label/id substrings to keep"
    )
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    df = build_opentargets_prior(
        args.dir,
        out_parquet=args.out,
        out_tsv=args.out_tsv,
        disease_terms=args.terms,
        min_score=args.min_score,
    )
    print(
        f"[priors] OpenTargets -> {len(df):,} rows; wrote {args.out} and {args.out_tsv}"
    )


if __name__ == "__main__":
    main()
