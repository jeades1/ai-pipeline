#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

PREFERRED = ["promoted_full.tsv", "ranked.tsv", "assoc.tsv", "features.tsv"]


def load_first_available(art: Path) -> tuple[pd.DataFrame, str]:
    tried = []
    for fname in PREFERRED:
        f = art / fname
        if f.exists():
            if fname in ("promoted_full.tsv", "ranked.tsv"):
                df = pd.read_csv(f, sep="\t")
                # normalize likely column names
                df = df.rename(columns={"feature": "name", "kind": "type"})
                need = ["name", "layer", "type"]
                have = [c for c in need if c in df.columns]
                if len(have) == 3:
                    return df[need].copy(), fname
            elif fname == "assoc.tsv":
                df = pd.read_csv(f, sep="\t")
                # expect feature + (optional) layer/type
                df = df.rename(columns={"feature": "name", "kind": "type"})
                if "name" in df.columns:
                    if "layer" not in df.columns:
                        df["layer"] = "transcriptomic"
                    if "type" not in df.columns:
                        df["type"] = "gene"
                    return df[["name", "layer", "type"]].copy(), fname
            elif fname == "features.tsv":
                df = pd.read_csv(f, sep="\t")
                df = df.rename(columns={"feature": "name", "kind": "type"})
                need = ["name", "layer", "type"]
                have = [c for c in need if c in df.columns]
                if len(have) == 3:
                    return df[need].copy(), fname
        tried.append(str(f))
    raise SystemExit(
        "[promoted] Could not construct promoted.tsv. Tried (in order):\n  - "
        + "\n  - ".join(tried)
        + "\nRun `make demo` and ensure at least one of those files exists."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="artifacts directory")
    args = ap.parse_args()
    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)

    df, source = load_first_available(art)
    out = df.drop_duplicates().sort_values(["layer", "name"])
    out_path = art / "promoted.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[promoted] Wrote {out_path} rows: {len(out)} (source: {source})")


if __name__ == "__main__":
    main()
