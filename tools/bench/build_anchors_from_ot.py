#!/usr/bin/env python3
from __future__ import annotations

"""
Build an expanded benchmark anchor list from Open Targets association parquet.

Inputs:
  --dir data/external/opentargets/25.06/association_overall_direct
  --terms cardio cholesterol ldl lipid
  --top-n 100

Outputs:
  - artifacts/bench/<disease>_expanded.tsv (columns: gene)
  - benchmarks/<disease>_markers.json (optional overwrite/update if --write-json)
"""
import json
from pathlib import Path
from typing import Iterable, List
import pandas as pd


def load_ot_parts(base: Path) -> pd.DataFrame:
    parts: List[Path] = []
    if base.is_file() and base.suffix == ".parquet":
        parts = [base]
    elif base.is_dir():
        parts = sorted(p for p in base.glob("**/*.parquet") if p.is_file())
    if not parts:
        raise FileNotFoundError(f"No parquet files under {base}")
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        raise ValueError("OpenTargets parquet did not contain rows")
    return pd.concat(frames, ignore_index=True)


def build_anchors(ot_dir: Path, terms: Iterable[str], top_n: int = 100) -> pd.DataFrame:
    df = load_ot_parts(ot_dir)
    # Minimal schema observed: diseaseId, targetId, score
    cols = {c.lower(): c for c in df.columns}
    t_id = cols.get("targetid")
    did = cols.get("diseaseid")
    score_col = cols.get("score") or cols.get("overallscore")
    if not (t_id and did and score_col):
        raise ValueError(
            "OpenTargets parquet missing required columns: targetId/diseaseId/score"
        )
    x = df[[t_id, did, score_col]].copy()
    x.rename(
        columns={t_id: "targetId", did: "diseaseId", score_col: "score"}, inplace=True
    )
    # Filter by disease ID substrings (since labels not present here)
    trms = [str(t).lower() for t in terms]
    x = x[
        x["diseaseId"]
        .astype(str)
        .str.lower()
        .apply(lambda s: any(t in s for t in trms))
    ]
    # Map Ensembl → symbol if we have priors/opentargets_prior.tsv; else keep Ensembl as name
    sym_map = None
    pri_tsv = Path("data/processed/priors/opentargets_prior.tsv")
    if pri_tsv.exists():
        try:
            pri = pd.read_csv(pri_tsv, sep="\t")
            if {"gene_id", "gene_symbol"}.issubset(pri.columns):
                sym_map = (
                    pri[["gene_id", "gene_symbol"]]
                    .dropna()
                    .drop_duplicates()
                    .set_index("gene_id")["gene_symbol"]
                    .to_dict()
                )
        except Exception:
            sym_map = None
    if sym_map:
        x["gene"] = x["targetId"].map(sym_map).fillna(x["targetId"])
    else:
        x["gene"] = x["targetId"]
    x["gene"] = x["gene"].astype(str).str.upper()
    # Rank by score and take top-N unique symbols
    x = x.sort_values("score", ascending=False)
    x = x.drop_duplicates("gene").head(top_n)
    out = pd.DataFrame({"gene": x["gene"].tolist()})
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("data/external/opentargets/25.06/association_overall_direct"),
    )
    ap.add_argument(
        "--terms", nargs="*", default=["cardio", "cholesterol", "ldl", "lipid"]
    )
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--disease", type=str, default="cardiovascular")
    ap.add_argument("--out-tsv", type=Path, default=None)
    ap.add_argument("--write-json", action="store_true")
    args = ap.parse_args()

    anchors = build_anchors(args.dir, args.terms, args.top_n)
    out_tsv = args.out_tsv or Path("artifacts/bench") / f"{args.disease}_expanded.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(out_tsv, sep="\t", index=False)
    print(f"[anchors] Wrote expanded anchors -> {out_tsv} ({len(anchors)} genes)")

    if args.write_json:
        bj = Path("benchmarks") / f"{args.disease}_markers.json"
        try:
            data = json.loads(bj.read_text()) if bj.exists() else {"biomarkers": []}
        except Exception:
            data = {"biomarkers": []}
        # Merge unique
        existing = {
            str(it.get("name") or it.get("gene")).upper()
            for it in data.get("biomarkers", [])
        }
        merged = existing | set(anchors["gene"].astype(str).str.upper())
        data["biomarkers"] = [{"name": g} for g in sorted(merged)]
        bj.write_text(json.dumps(data, indent=2))
        print(f"[anchors] Updated {bj}")


if __name__ == "__main__":
    main()
