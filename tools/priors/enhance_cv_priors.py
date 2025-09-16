#!/usr/bin/env python3
from __future__ import annotations

"""
Enhance priors with cardiovascular-specific evidence by assigning
high prior scores to known cardiovascular genes based on literature
and pathway knowledge.
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import pandas as pd


def create_cv_priors() -> Dict[str, float]:
    """Create cardiovascular-specific prior scores."""

    # Tier 1: Established drug targets and major CV genes
    tier1 = {
        "HMGCR": 1.0,  # Statin target
        "PCSK9": 1.0,  # PCSK9 inhibitors
        "LDLR": 1.0,  # LDL receptor
        "APOB": 0.95,  # Major lipoprotein component
        "LPL": 0.95,  # Lipoprotein lipase
        "CETP": 0.9,  # CETP inhibitors
        "NPC1L1": 0.9,  # Ezetimibe target
    }

    # Tier 2: Important pathway components
    tier2 = {
        "ABCA1": 0.8,
        "APOE": 0.8,
        "APOA1": 0.8,
        "APOC3": 0.8,
        "SREBF1": 0.75,
        "SREBF2": 0.75,
        "LCAT": 0.75,
        "LIPC": 0.75,
        "ANGPTL3": 0.7,
        "ANGPTL4": 0.7,
        "SORT1": 0.7,
        "LPA": 0.7,
    }

    # Tier 3: Supporting genes
    tier3 = {
        "ADIPOQ": 0.6,
        "CRP": 0.6,
        "PON1": 0.6,
        "PLTP": 0.6,
        "INSIG1": 0.5,
        "INSIG2": 0.5,
        "SCAP": 0.5,
        "MTTP": 0.5,
        "LDLRAP1": 0.5,
        "APOA5": 0.5,
        "LIPG": 0.5,
    }

    # Combine all tiers
    cv_priors = {}
    cv_priors.update(tier1)
    cv_priors.update(tier2)
    cv_priors.update(tier3)

    return cv_priors


def enhance_priors_manifest(manifest_path: Path) -> None:
    """Add cardiovascular priors to the manifest."""

    # Load existing manifest
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"context": "cardiovascular", "outputs": {}}

    # Create CV priors file
    cv_priors = create_cv_priors()
    cv_df = pd.DataFrame(
        [
            {
                "gene_symbol": gene,
                "score": score,
                "source": "Literature_CV",
                "context": "cardiovascular",
            }
            for gene, score in cv_priors.items()
        ]
    )

    cv_path = Path("data/processed/priors/cardiovascular_literature.tsv")
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(cv_path, sep="\t", index=False)

    # Add to manifest
    manifest["outputs"]["cardiovascular_literature"] = str(cv_path)

    # Also create a unified version that includes the CV priors
    outputs = manifest.get("outputs", {})
    all_dfs = [cv_df]

    # Load and append existing priors
    for source, path in outputs.items():
        if source == "cardiovascular_literature":
            continue
        if path and Path(path).exists():
            try:
                df = pd.read_csv(path, sep="\t")
                if "gene_symbol" in df.columns and "score" in df.columns:
                    # Normalize existing scores to [0,1] and reduce weight slightly
                    if len(df) > 0:
                        df["score"] = df["score"] / df["score"].max() * 0.8
                    all_dfs.append(df[["gene_symbol", "score", "source"]])
            except Exception:
                continue

    if len(all_dfs) > 1:
        # Combine and aggregate
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["gene_symbol"] = combined["gene_symbol"].astype(str).str.upper()

        # Weight CV literature highly
        combined.loc[combined["source"] == "Literature_CV", "score"] *= 2.0

        # Aggregate by gene (max score across sources)
        agg = (
            combined.groupby("gene_symbol")
            .agg({"score": "max", "source": lambda x: ";".join(x)})
            .reset_index()
        )

        unified_path = Path("data/processed/priors/unified_cv_enhanced.tsv")
        agg.to_csv(unified_path, sep="\t", index=False)
        manifest["outputs"]["unified"] = str(unified_path)

    # Update manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Enhanced priors manifest with {len(cv_priors)} CV genes")
    print(f"CV priors saved to: {cv_path}")
    print(f"Updated manifest: {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description="Enhance priors with CV-specific evidence")
    ap.add_argument(
        "--manifest", type=Path, default=Path("data/processed/priors/manifest.json")
    )
    args = ap.parse_args()

    enhance_priors_manifest(args.manifest)


if __name__ == "__main__":
    main()
