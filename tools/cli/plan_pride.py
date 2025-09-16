#!/usr/bin/env python3
from __future__ import annotations
import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="data/external/pride/kidney",
        help="Target directory for PRIDE exports",
    )
    args = ap.parse_args()

    print("=== PRIDE planning (kidney/AKI/sepsis) ===")
    print("Suggested filters:")
    print("- Organism: Homo sapiens")
    print("- Tissue: kidney OR renal cortex OR renal medulla")
    print("- Disease: acute kidney injury OR sepsis")
    print("- Assay type: Quantitative proteomics (if desired)")
    print("- Output: Download result tables (mzTab/tsv) per study")
    print()
    print("Suggested studies to consider (examples, verify relevance):")
    print("- PXD005254: Human kidney tissue proteome (renal regions)")
    print("- PXD021041: Plasma proteomics in sepsis (AKI incidence)")
    print("- PXD003497: Urine proteomics in AKI")
    print()
    print(f"Place downloaded tables under: {args.out}/<PXD_ID>/")
    print("Then add a small parser mapping Uniprot->Gene and summary stats for priors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
