from __future__ import annotations

from pathlib import Path
import json

# Import via absolute module path to work when run as a script with PYTHONPATH=.
from tools.priors.opentargets import build_opentargets_prior
from tools.priors.gtex import build_gtex_prior
from tools.priors.hpa import build_hpa_prior
from tools.priors.encode import build_encode_priors
from tools.priors.pride import build_pride_priors


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build all available priors")
    ap.add_argument("--outdir", type=Path, default=Path("data/processed/priors"))
    ap.add_argument(
        "--context",
        type=str,
        default="",
        help="Optional tissue/disease context for filters (e.g., 'kidney')",
    )
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument(
        "--ot-terms",
        nargs="*",
        default=None,
        help="Optional disease label/id substrings to filter OpenTargets (e.g., 'cardiovascular', 'cholesterol')",
    )
    # GTEx non-median support
    ap.add_argument(
        "--gtex-col-map",
        type=Path,
        default=None,
        help="Optional TSV/CSV mapping GCT column→tissue labels",
    )
    ap.add_argument(
        "--gtex-aggregate",
        type=str,
        default="median",
        choices=["median", "mean", "max"],
        help="Aggregation when multiple columns map to one tissue label",
    )
    ap.add_argument(
        "--gtex-sample-attrs",
        type=Path,
        default=None,
        help="Optional GTEx SampleAttributes (TSV) to auto-derive column→tissue mapping (uses SMTSD)",
    )
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional filters (context-specific); leave empty for disease-agnostic build
    # OpenTargets: allow user-provided terms; GTEx/HPA: filter tissues by provided context if any
    disease_terms = args.ot_terms if args.ot_terms else None
    tissue_terms = [args.context] if args.context else []

    # 1) Open Targets
    ot_src = Path("data/external/opentargets/25.06/association_overall_direct")
    ot_parquet = outdir / "opentargets_prior.parquet"
    ot_tsv = outdir / "opentargets_prior.tsv"
    ot_df = None
    try:
        ot_df = build_opentargets_prior(
            ot_src,
            out_parquet=ot_parquet,
            out_tsv=ot_tsv,
            disease_terms=disease_terms,
            min_score=args.min_score,
        )
    except Exception as e:
        print(f"[priors] OpenTargets skipped: {e}")

    # 2) GTEx
    gtex_gct = Path("data/external/gtex/GTEx_v8_gene_median_tpm.gct.gz")
    if not gtex_gct.exists():
        alt = Path("data/external/gtex/GTEx_v8_gene_median_tpm.gct")
        if alt.exists():
            gtex_gct = alt
    gtex_parquet = outdir / "gtex_prior.parquet"
    gtex_tsv = outdir / "gtex_prior.tsv"
    gtex_df = None
    try:
        col_map = None
        if args.gtex_col_map and Path(args.gtex_col_map).exists():
            import pandas as pd

            m = pd.read_csv(args.gtex_col_map, sep="\t|,", engine="python")
            col_col = next(
                (c for c in m.columns if c.lower() in {"column", "name"}), None
            )
            tis_col = next(
                (c for c in m.columns if c.lower() in {"tissue", "label"}), None
            )
            if col_col and tis_col:
                col_map = {
                    str(r[col_col]): str(r[tis_col])
                    for _, r in m.iterrows()
                    if pd.notna(r[col_col]) and pd.notna(r[tis_col])
                }
        elif args.gtex_sample_attrs and Path(args.gtex_sample_attrs).exists():
            import pandas as pd

            attrs = pd.read_csv(args.gtex_sample_attrs, sep="\t")
            samp_col = next((c for c in attrs.columns if c.upper() == "SAMPID"), None)
            tis_col = next(
                (c for c in attrs.columns if c.upper() in {"SMTSD", "SMTS"}), None
            )
            if samp_col and tis_col:
                col_map = {
                    str(r[samp_col]): str(r[tis_col])
                    for _, r in attrs[[samp_col, tis_col]].dropna().iterrows()
                }

        gtex_df = build_gtex_prior(
            gtex_gct,
            keep_tissues=tissue_terms,
            out_parquet=gtex_parquet,
            out_tsv=gtex_tsv,
            column_map=col_map,
            aggregate=args.gtex_aggregate,
        )
    except Exception as e:
        print(f"[priors] GTEx skipped: {e}")

    # 3) HPA
    hpa_zip = Path("data/external/hpa/rna_tissue.tsv.zip")
    if not hpa_zip.exists():
        alt = Path("data/external/hpa/rna_tissue.tsv")
        if alt.exists():
            hpa_zip = alt
    hpa_parquet = outdir / "hpa_prior.parquet"
    hpa_tsv = outdir / "hpa_prior.tsv"
    hpa_df = None
    try:
        hpa_df = build_hpa_prior(
            hpa_zip, keep_tissues=tissue_terms, out_parquet=hpa_parquet, out_tsv=hpa_tsv
        )
    except Exception as e:
        print(f"[priors] HPA skipped: {e}")

    # 4) ENCODE ATAC-seq
    encode_peaks = Path("data/external/encode/kidney_atac")
    encode_tsv = outdir / "encode_prior.tsv"
    encode_df = None
    try:
        if encode_peaks.exists() and any(encode_peaks.glob("*.narrowPeak")):
            build_encode_priors(encode_peaks, out_path=encode_tsv)
            import pandas as pd

            encode_df = pd.read_csv(encode_tsv, sep="\t")
        else:
            print("[priors] ENCODE skipped: no narrowPeak files found")
    except Exception as e:
        print(f"[priors] ENCODE skipped: {e}")

    # 5) PRIDE proteomics
    pride_quant = Path("data/external/pride/kidney_proteome")
    pride_tsv = outdir / "pride_prior.tsv"
    pride_df = None
    try:
        if pride_quant.exists() and any(pride_quant.glob("*.tsv")):
            build_pride_priors(
                pride_quant, tissue_context=args.context, out_path=pride_tsv
            )
            import pandas as pd

            pride_df = pd.read_csv(pride_tsv, sep="\t")
        else:
            print("[priors] PRIDE skipped: no quantification TSV files found")
    except Exception as e:
        print(f"[priors] PRIDE skipped: {e}")

    # 6) Unified view across sources with a consistent schema
    # Columns: gene_id, gene_symbol, context, context_type ("disease"|"tissue"), source, score
    unified = None
    parts = []
    if ot_df is not None and len(ot_df):
        # Use disease_id as context (labels not present in this dataset)
        p = ot_df.copy()
        p["context"] = p["disease_id"].astype(str)
        # Ensure gene_symbol is populated (fallback to gene_id)
        if "gene_symbol" in p.columns:
            p["gene_symbol"] = p["gene_symbol"].fillna(p["gene_id"].astype(str))
        else:
            p["gene_symbol"] = p["gene_id"].astype(str)
        p = p[["gene_id", "gene_symbol", "context", "score", "source"]].copy()
        p["context_type"] = "disease"
        parts.append(p)
    if gtex_df is not None and len(gtex_df):
        p = gtex_df.rename(columns={"tissue": "context"})
        # gtex_df may not have gene_id
        if "gene_id" not in p.columns:
            p["gene_id"] = None
        p = p[["gene_id", "gene_symbol", "context", "score", "source"]].copy()
        p["context_type"] = "tissue"
        parts.append(p)
    if hpa_df is not None and len(hpa_df):
        p = hpa_df.rename(columns={"tissue": "context"})
        if "gene_id" not in p.columns:
            p["gene_id"] = None
        p = p[["gene_id", "gene_symbol", "context", "score", "source"]].copy()
        p["context_type"] = "tissue"
        parts.append(p)
    if encode_df is not None and len(encode_df):
        p = encode_df.copy()
        if "gene_id" not in p.columns:
            p["gene_id"] = None
        p = p[["gene_id", "gene_symbol", "context", "score", "source"]].copy()
        p["context_type"] = "tissue"
        parts.append(p)
    if pride_df is not None and len(pride_df):
        p = pride_df.copy()
        if "gene_id" not in p.columns:
            p["gene_id"] = None
        p = p[["gene_id", "gene_symbol", "context", "score", "source"]].copy()
        p["context_type"] = "tissue"
        parts.append(p)

    if parts:
        import pandas as pd

        unified = pd.concat(parts, ignore_index=True).dropna(
            subset=["context", "score"], how="any"
        )
        unified_parquet = outdir / "unified_priors.parquet"
        unified_tsv = outdir / "unified_priors.tsv"
        unified.to_parquet(unified_parquet, index=False)
        unified.to_csv(unified_tsv, sep="\t", index=False)
        print(f"[priors] Unified priors -> {unified_parquet} ({len(unified):,} rows)")

    # Manifest
    manifest = {
        "context": args.context,
        "outputs": {
            "opentargets": str(ot_parquet) if ot_df is not None else None,
            "gtex": str(gtex_parquet) if gtex_df is not None else None,
            "hpa": str(hpa_parquet) if hpa_df is not None else None,
            "encode": str(encode_tsv) if encode_df is not None else None,
            "pride": str(pride_tsv) if pride_df is not None else None,
            "unified": str(outdir / "unified_priors.parquet") if parts else None,
        },
        "counts": {
            "opentargets": 0 if ot_df is None else len(ot_df),
            "gtex": 0 if gtex_df is None else len(gtex_df),
            "hpa": 0 if hpa_df is None else len(hpa_df),
            "encode": 0 if encode_df is None else len(encode_df),
            "pride": 0 if pride_df is None else len(pride_df),
            "unified": 0 if not parts else (0 if unified is None else len(unified)),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[priors] Wrote manifest -> {outdir/'manifest.json'}")


if __name__ == "__main__":
    main()
