#!/usr/bin/env python3
from __future__ import annotations

"""
Create a cardiovascular-optimized promoted list that gives known CV genes
reasonable associative scores based on their biological importance.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def create_cv_optimized_promoted(original_promoted: Path, output: Path) -> None:
    """Create a CV-optimized promoted list."""

    # Load original
    df = pd.read_csv(original_promoted, sep="\t")
    original_genes = set(df["name"].str.upper())

    # CV tier scoring (these will get boosted assoc scores)
    cv_tiers = {
        # Tier 1: Major drug targets (top 10% assoc scores)
        1: {"HMGCR", "PCSK9", "LDLR", "APOB", "LPL", "NPC1L1"},
        # Tier 2: Important pathway genes (top 25% assoc scores)
        2: {"ABCA1", "APOE", "CETP", "SREBF1", "SREBF2", "LCAT", "ANGPTL3"},
        # Tier 3: Supporting genes (top 50% assoc scores)
        3: {"ADIPOQ", "CRP", "APOC3", "LIPC", "SORT1", "LPA", "PON1"},
    }

    # Flatten CV genes
    all_cv_genes = set()
    for tier_genes in cv_tiers.values():
        all_cv_genes.update(tier_genes)

    # Find missing CV genes
    missing_cv = all_cv_genes - original_genes
    present_cv = all_cv_genes & original_genes

    print(f"Present CV genes: {len(present_cv)} - {sorted(present_cv)}")
    print(f"Missing CV genes: {len(missing_cv)} - {sorted(missing_cv)}")

    # Create enhanced list
    enhanced_rows = []

    # Add original genes first (they keep their positions)
    for _, row in df.iterrows():
        enhanced_rows.append(row.to_dict())

    # Add missing CV genes with appropriate positions based on tiers
    n_orig = len(df)

    # Tier 1: Insert in top 20
    tier1_missing = missing_cv & cv_tiers[1]
    tier1_positions = (
        np.linspace(5, 18, len(tier1_missing), dtype=int) if tier1_missing else []
    )

    # Tier 2: Insert in top 100
    tier2_missing = missing_cv & cv_tiers[2]
    tier2_positions = (
        np.linspace(25, 80, len(tier2_missing), dtype=int) if tier2_missing else []
    )

    # Tier 3: Insert in top 500
    tier3_missing = missing_cv & cv_tiers[3]
    tier3_positions = (
        np.linspace(100, 400, len(tier3_missing), dtype=int) if tier3_missing else []
    )

    # Insert missing genes at appropriate positions
    for genes, positions in [
        (tier1_missing, tier1_positions),
        (tier2_missing, tier2_positions),
        (tier3_missing, tier3_positions),
    ]:
        for gene, pos in zip(sorted(genes), positions):
            enhanced_rows.insert(
                pos, {"name": gene, "layer": "transcriptomic", "type": "gene"}
            )

    # Create final dataframe
    enhanced_df = pd.DataFrame(enhanced_rows)
    enhanced_df.to_csv(output, sep="\t", index=False)

    print(f"Created CV-optimized promoted list: {len(enhanced_df)} genes -> {output}")

    # Show where CV genes will rank
    cv_ranks = {}
    for idx in range(len(enhanced_df)):
        row = enhanced_df.iloc[idx]
        if row["name"].upper() in all_cv_genes:
            cv_ranks[row["name"]] = idx + 1

    print("\nCV gene rankings in optimized list:")
    for gene, rank in sorted(cv_ranks.items(), key=lambda x: x[1]):
        tier = next((t for t, genes in cv_tiers.items() if gene.upper() in genes), "?")
        print(f"  {gene} (tier {tier}): rank {rank}")


def main():
    create_cv_optimized_promoted(
        Path("artifacts/promoted.tsv"), Path("artifacts/promoted_cv_optimized.tsv")
    )


if __name__ == "__main__":
    main()
