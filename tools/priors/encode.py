"""
ENCODE ATAC-seq prior builder.

Reads narrowPeak files from ENCODE and maps peaks to genes via TSS proximity.
Outputs tissue-specific accessibility priors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_gene_annotations(gtf_path: Path) -> pd.DataFrame:
    """Load gene annotations from GTF file (TSS positions)."""
    # Mock annotations for demo - in practice, parse a real GTF
    mock_genes = [
        {
            "gene_id": "ENSG00000142168",
            "gene_symbol": "SOD2",
            "chr": "chr6",
            "tss": 160113611,
        },
        {
            "gene_id": "ENSG00000115414",
            "gene_symbol": "FN1",
            "chr": "chr2",
            "tss": 216225950,
        },
        {
            "gene_id": "ENSG00000171862",
            "gene_symbol": "PTEN",
            "chr": "chr10",
            "tss": 87864470,
        },
        {
            "gene_id": "ENSG00000141510",
            "gene_symbol": "TP53",
            "chr": "chr17",
            "tss": 7668402,
        },
        {
            "gene_id": "ENSG00000134057",
            "gene_symbol": "CCNB1",
            "chr": "chr5",
            "tss": 68462324,
        },
        {
            "gene_id": "ENSG00000105221",
            "gene_symbol": "AKT2",
            "chr": "chr19",
            "tss": 40736376,
        },
        {
            "gene_id": "ENSG00000149925",
            "gene_symbol": "ALDOA",
            "chr": "chr16",
            "tss": 30064191,
        },
        {
            "gene_id": "ENSG00000111640",
            "gene_symbol": "GAPDH",
            "chr": "chr12",
            "tss": 6534405,
        },
        {
            "gene_id": "ENSG00000158710",
            "gene_symbol": "TAGLN2",
            "chr": "chr1",
            "tss": 156844621,
        },
        {
            "gene_id": "ENSG00000166012",
            "gene_symbol": "NANOG",
            "chr": "chr12",
            "tss": 7940646,
        },
        {
            "gene_id": "ENSG00000164362",
            "gene_symbol": "TERT",
            "chr": "chr5",
            "tss": 1253287,
        },
        {
            "gene_id": "ENSG00000188976",
            "gene_symbol": "NOX4",
            "chr": "chr11",
            "tss": 89320978,
        },
        {
            "gene_id": "ENSG00000139618",
            "gene_symbol": "BRCA2",
            "chr": "chr13",
            "tss": 32315508,
        },
        {
            "gene_id": "ENSG00000012048",
            "gene_symbol": "BRCA1",
            "chr": "chr17",
            "tss": 43044295,
        },
        {
            "gene_id": "ENSG00000073756",
            "gene_symbol": "PTGS2",
            "chr": "chr1",
            "tss": 186681189,
        },
        {
            "gene_id": "ENSG00000164305",
            "gene_symbol": "CASP3",
            "chr": "chr4",
            "tss": 184627696,
        },
        {
            "gene_id": "ENSG00000136997",
            "gene_symbol": "MYC",
            "chr": "chr8",
            "tss": 127735434,
        },
        {
            "gene_id": "ENSG00000165949",
            "gene_symbol": "IFI27",
            "chr": "chr14",
            "tss": 94682513,
        },
        {
            "gene_id": "ENSG00000166913",
            "gene_symbol": "YWHAB",
            "chr": "chr20",
            "tss": 43515521,
        },
        {
            "gene_id": "ENSG00000213281",
            "gene_symbol": "NFKB1",
            "chr": "chr4",
            "tss": 102501918,
        },
    ]
    return pd.DataFrame(mock_genes)


def _parse_narrowpeak(peak_file: Path) -> pd.DataFrame:
    """Parse narrowPeak format file."""
    columns = [
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signal",
        "pvalue",
        "qvalue",
        "peak",
    ]
    df = pd.read_csv(peak_file, sep="\t", header=None, names=columns)
    return df


def _map_peaks_to_genes(
    peaks: pd.DataFrame, genes: pd.DataFrame, window: int = 5000
) -> pd.DataFrame:
    """Map ATAC peaks to genes via TSS proximity."""
    results = []

    for _, peak in peaks.iterrows():
        peak_center = (peak["start"] + peak["end"]) // 2

        # Find genes within window
        chr_genes = genes[genes["chr"] == peak["chr"]]
        for _, gene in chr_genes.iterrows():
            distance = abs(gene["tss"] - peak_center)
            if distance <= window:
                # Accessibility score based on signal and distance
                accessibility = peak["signal"] * (1 - distance / window)
                results.append(
                    {
                        "gene_id": gene["gene_id"],
                        "gene_symbol": gene["gene_symbol"],
                        "context": "kidney",  # tissue context
                        "score": accessibility,
                        "source": "ENCODE",
                        "context_type": "tissue",
                    }
                )

    return pd.DataFrame(results)


def build_encode_priors(
    peaks_dir: Path,
    gtf_path: Path | None = None,
    out_path: Path = Path("data/processed/priors/encode_prior.tsv"),
) -> None:
    """Build ENCODE ATAC-seq priors."""
    print(f"[encode] Building priors from {peaks_dir}")

    # Load gene annotations
    if gtf_path and gtf_path.exists():
        genes = _load_gene_annotations(gtf_path)
    else:
        print("[encode] Using mock gene annotations (no GTF provided)")
        genes = _load_gene_annotations(Path(""))

    all_priors = []

    # Process all narrowPeak files
    for peak_file in peaks_dir.glob("*.narrowPeak"):
        print(f"[encode] Processing {peak_file.name}")
        peaks = _parse_narrowpeak(peak_file)
        priors = _map_peaks_to_genes(peaks, genes)
        all_priors.append(priors)

    if not all_priors:
        print("[encode] No narrowPeak files found")
        return

    # Combine and aggregate
    combined = pd.concat(all_priors, ignore_index=True)

    if combined.empty:
        print("[encode] No priors generated")
        return

    # Aggregate by gene/context (max accessibility)
    group_cols = ["gene_symbol", "context", "source", "context_type"]
    if "gene_id" in combined.columns:
        group_cols = ["gene_id"] + group_cols

    aggregated = combined.groupby(group_cols, as_index=False).agg({"score": "max"})

    # Normalize scores per context
    for context in aggregated["context"].unique():
        mask = aggregated["context"] == context
        subset = aggregated[mask]
        if len(subset) > 0:
            max_score = subset["score"].max()
            if max_score > 0:
                aggregated.loc[mask, "score"] = subset["score"] / max_score

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_path, sep="\t", index=False)
    print(f"[encode] Wrote {len(aggregated)} priors to {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build ENCODE ATAC-seq priors")
    ap.add_argument(
        "--peaks",
        type=Path,
        default=Path("data/external/encode/kidney_atac"),
        help="Directory containing narrowPeak files",
    )
    ap.add_argument("--gtf", type=Path, help="GTF annotation file")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/priors/encode_prior.tsv"),
        help="Output TSV file",
    )

    args = ap.parse_args()
    build_encode_priors(args.peaks, args.gtf, args.out)
