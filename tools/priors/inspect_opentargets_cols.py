from __future__ import annotations

from pathlib import Path
import pandas as pd


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("data/external/opentargets/25.06/association_overall_direct"),
    )
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    parts = sorted(p for p in Path(args.dir).glob("**/*.parquet"))
    if not parts:
        print("No parquet parts found")
        return
    p0 = parts[0]
    df = pd.read_parquet(p0)
    print(f"File: {p0}")
    print("Columns (", len(df.columns), "):")
    for c in df.columns:
        print(" -", c)
    candidates = [
        "targetId",
        "targetFromSourceId",
        "targetSymbol",
        "diseaseId",
        "diseaseFromSourceMappedId",
        "diseaseLabel",
        "diseaseName",
        "diseaseFromSource",
        "score",
        "overallScore",
    ]
    present = [c for c in candidates if c in df.columns]
    print("Present of interest:", present)
    print(df[present].head(args.n))


if __name__ == "__main__":
    main()
