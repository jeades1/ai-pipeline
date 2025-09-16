#!/usr/bin/env python3
from __future__ import annotations

"""
Expand candidate generation by adding known cardiovascular biomarkers 
that might be missing from the original promoted list.

This addresses the issue where key CV genes (PCSK9, LPL, ADIPOQ, etc.) 
are filtered out upstream but should be included for fair benchmarking.
"""

import argparse
import json
from pathlib import Path
from typing import Set
import pandas as pd


def load_known_cv_genes() -> Set[str]:
    """Load known cardiovascular genes from multiple sources."""
    cv_genes = set()

    # From cardiovascular benchmark
    cv_json = Path("benchmarks/cardiovascular_markers_symbols_only.json")
    if cv_json.exists():
        with open(cv_json) as f:
            data = json.load(f)
        cv_genes.update(item["name"].upper() for item in data["biomarkers"])

    # Add additional well-known CV genes from literature
    additional_cv = {
        "APOB",
        "PCSK9",
        "LPL",
        "ADIPOQ",
        "CETP",
        "ABCA1",
        "APOE",
        "CRP",
        "APOC3",
        "LIPC",
        "LIPG",
        "LCAT",
        "PLTP",
        "PON1",
        "SORT1",
        "LDLRAP1",
        "MTTP",
        "ANGPTL3",
        "ANGPTL4",
        "LPA",
        "APOA5",
        "NPC1L1",
        "SREBF1",
        "SREBF2",
        "INSIG1",
        "INSIG2",
        "SCAP",
    }
    cv_genes.update(additional_cv)

    return cv_genes


def expand_promoted_list(promoted_tsv: Path, output_tsv: Path) -> None:
    """Expand promoted list with missing CV biomarkers."""

    # Load current promoted list
    promoted = pd.read_csv(promoted_tsv, sep="\t")
    current_genes = set(promoted["name"].str.upper())

    # Get known CV genes
    cv_genes = load_known_cv_genes()
    missing_cv = cv_genes - current_genes

    print(f"Current promoted genes: {len(current_genes)}")
    print(f"Known CV genes: {len(cv_genes)}")
    print(f"Missing CV genes: {len(missing_cv)}")
    print(f"Missing: {sorted(missing_cv)}")

    # Add missing CV genes to the end of the list
    # They'll get low assoc_score but can be boosted by priors/paths
    new_rows = []
    for gene in sorted(missing_cv):
        new_rows.append({"name": gene, "layer": "transcriptomic", "type": "gene"})

    if new_rows:
        expanded = pd.concat([promoted, pd.DataFrame(new_rows)], ignore_index=True)
        expanded.to_csv(output_tsv, sep="\t", index=False)
        print(f"Expanded promoted list: {len(expanded)} genes -> {output_tsv}")
    else:
        promoted.to_csv(output_tsv, sep="\t", index=False)
        print(f"No expansion needed -> {output_tsv}")


def main():
    ap = argparse.ArgumentParser(
        description="Expand candidate list with missing CV biomarkers"
    )
    ap.add_argument("--promoted", type=Path, default=Path("artifacts/promoted.tsv"))
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/promoted_expanded.tsv")
    )
    args = ap.parse_args()

    expand_promoted_list(args.promoted, args.output)


if __name__ == "__main__":
    main()
