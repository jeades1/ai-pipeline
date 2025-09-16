from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def summarize_unified(
    unified_path: Path, out_json: Path, out_tsv: Path, top_n: int = 50
):
    df = pd.read_parquet(unified_path)
    if df.empty:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"note": "no priors"}, indent=2))
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
        print(f"[priors] No rows in {unified_path}")
        return

    # Basic counts per source and context_type
    counts = {
        "rows": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "by_context_type": df["context_type"].value_counts().to_dict(),
    }

    # Aggregate per gene_symbol per source: max score as a simple strength proxy
    agg = (
        df.dropna(subset=["gene_symbol"])
        .groupby(["source", "gene_symbol"], as_index=False)["score"]
        .max()
    )
    # Normalize score per source for comparability (guard against zero division)
    denom = agg.groupby("source")["score"].transform(lambda s: max(s.max(), 1e-9))
    agg["score_norm"] = agg["score"] / denom
    # Use group-wise nlargest to avoid sort typing issues
    top = (
        agg.groupby("source", group_keys=True)
        .apply(lambda g: g.nlargest(top_n, columns=["score_norm"]))
        .reset_index(drop=True)
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(counts, indent=2))
    top.to_csv(out_tsv, sep="\t", index=False)
    print(f"[priors] Summary -> {out_json}; Top-{top_n} table -> {out_tsv}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Summarize unified priors")
    ap.add_argument(
        "--unified",
        type=Path,
        default=Path("data/processed/priors/unified_priors.parquet"),
    )
    ap.add_argument(
        "--out-json", type=Path, default=Path("artifacts/priors_summary.json")
    )
    ap.add_argument("--out-tsv", type=Path, default=Path("artifacts/priors_top.tsv"))
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    summarize_unified(args.unified, args.out_json, args.out_tsv, top_n=args.top)


if __name__ == "__main__":
    main()
