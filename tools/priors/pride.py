"""
PRIDE proteomics prior builder.

Reads protein quantification data from PRIDE studies and maps to genes.
Outputs tissue-specific protein expression priors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _load_protein_gene_mapping() -> Dict[str, str]:
    """Load protein ID to gene symbol mapping."""
    # Mock mapping for demo - in practice, use UniProt or similar
    return {
        "P02768": "ALB",
        "P04406": "GAPDH",
        "P61626": "LYZ",
        "P02787": "TF",
        "P00738": "HP",
        "P02647": "APOA1",
        "P02652": "APOA2",
        "P04114": "APOB",
        "P02749": "APOH",
        "P02655": "APOC2",
        "P02656": "APOC3",
        "P06727": "APOA4",
        "P01024": "C3",
        "P00450": "CP",
        "P02675": "FGB",
        "P02679": "FGG",
        "P00747": "PLG",
        "P02790": "HPX",
        "P01009": "A1AT",
        "P04003": "C4BPA",
        "Q03591": "FHL1",
        "P07996": "THBS1",
        "P35579": "MYH9",
        "P21333": "FLNA",
        "P08123": "COL1A2",
        "P02452": "COL1A1",
        "P08572": "COL4A2",
        "P02458": "COL2A1",
        "P12109": "COL6A1",
        "P12110": "COL6A2",
        "P12111": "COL6A3",
        "P20908": "COL5A1",
        "P05997": "COL5A2",
        "P25940": "COL5A3",
        "P02461": "COL3A1",
        "P08253": "MMP2",
        "P14780": "MMP9",
        "P08254": "MMP3",
        "P09237": "MMP7",
        "P03956": "MMP1",
        "O75976": "CPD",
        "Q92945": "KHSRP",
        "P35237": "SPP1",
        "P05121": "SERPINE1",
        "P13611": "VCAN",
        "P07585": "DCN",
        "P21810": "BGN",
        "P98160": "HSPG2",
        "P35052": "GPC1",
        "P51884": "LUM",
        "P36222": "CHI3L1",
        "P23142": "FBLN1",
    }


def _parse_quant_file(quant_file: Path) -> pd.DataFrame:
    """Parse protein quantification file."""
    df = pd.read_csv(quant_file, sep="\t")

    # Expected columns: Protein_ID, Gene_Symbol, Sample_1, Sample_2, ...
    if "Protein_ID" not in df.columns:
        print(f"[pride] Warning: No Protein_ID column in {quant_file}")
        return pd.DataFrame()

    return df


def _aggregate_expression(
    df: pd.DataFrame, protein_gene_map: Dict[str, str]
) -> pd.DataFrame:
    """Aggregate protein expression to gene level."""
    results = []

    # Get sample columns (exclude Protein_ID, Gene_Symbol)
    sample_cols = [
        col for col in df.columns if col not in ["Protein_ID", "Gene_Symbol"]
    ]

    for _, row in df.iterrows():
        protein_id = row.get("Protein_ID", "")
        gene_symbol = row.get("Gene_Symbol", protein_gene_map.get(protein_id, ""))

        if not gene_symbol:
            continue

        # Calculate mean expression across samples
        sample_values = []
        for col in sample_cols:
            try:
                val = float(row[col])
                if val > 0:  # Only count positive values
                    sample_values.append(val)
            except (ValueError, TypeError):
                continue

        if sample_values:
            mean_expr = sum(sample_values) / len(sample_values)
            results.append(
                {
                    "gene_id": f"UNIPROT:{protein_id}",
                    "gene_symbol": gene_symbol,
                    "context": "kidney",  # tissue context
                    "score": mean_expr,
                    "source": "PRIDE",
                    "context_type": "tissue",
                }
            )

    return pd.DataFrame(results)


def build_pride_priors(
    quant_dir: Path,
    tissue_context: str = "kidney",
    out_path: Path = Path("data/processed/priors/pride_prior.tsv"),
) -> None:
    """Build PRIDE proteomics priors."""
    print(f"[pride] Building priors from {quant_dir}")

    protein_gene_map = _load_protein_gene_mapping()
    all_priors = []

    # Process all quantification files
    for quant_file in quant_dir.glob("*.tsv"):
        print(f"[pride] Processing {quant_file.name}")
        df = _parse_quant_file(quant_file)
        if df.empty:
            continue

        priors = _aggregate_expression(df, protein_gene_map)
        if not priors.empty:
            priors["context"] = tissue_context
            all_priors.append(priors)

    if not all_priors:
        print("[pride] No quantification files found")
        return

    # Combine and aggregate
    combined = pd.concat(all_priors, ignore_index=True)

    # Aggregate by gene/context (max expression)
    aggregated = combined.groupby(
        ["gene_id", "gene_symbol", "context", "source", "context_type"], as_index=False
    ).agg({"score": "max"})

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
    print(f"[pride] Wrote {len(aggregated)} priors to {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build PRIDE proteomics priors")
    ap.add_argument(
        "--quant",
        type=Path,
        default=Path("data/external/pride/kidney_proteome"),
        help="Directory containing quantification TSV files",
    )
    ap.add_argument("--tissue", default="kidney", help="Tissue context")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/priors/pride_prior.tsv"),
        help="Output TSV file",
    )

    args = ap.parse_args()
    build_pride_priors(args.quant, args.tissue, args.out)
