from __future__ import annotations

"""
Compute ablation metrics (assoc-only vs +priors vs +causal) and a simple
Open Targets baseline correlation if target scores are available.

Inputs:
  - artifacts/promoted.tsv (assoc ordering proxy)
  - artifacts/promoted_full.tsv, artifacts/ranked.tsv (after rerank)
  - data/external/opentargets/LDL_C_targets.tsv (optional: columns name,score)

Outputs:
  - artifacts/bench/ablation.json
  - artifacts/bench/opentargets_correlation.json (optional)
"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np


def _load_names_scores(df: pd.DataFrame, score_col: str) -> pd.Series:
    df = df.dropna(subset=["name"]).copy()
    s = pd.Series(
        df[score_col].astype(float).values, index=df["name"].astype(str).str.upper()
    )
    return s[~s.index.duplicated(keep="first")]


def precision_at_k(y_true: pd.Series, y_score: pd.Series, k: int = 20) -> float:
    idx = y_score.sort_values(ascending=False).index[:k]
    return float(y_true.reindex(idx).fillna(0.0).sum() / max(1, k))


def recall_at_k(y_true: pd.Series, y_score: pd.Series, k: int = 20) -> float:
    denom = float(y_true.sum()) or 1.0
    idx = y_score.sort_values(ascending=False).index[:k]
    return float(y_true.reindex(idx).fillna(0.0).sum() / denom)


def compute_ablation(
    promoted: Path,
    promoted_full: Path,
    ranked: Path,
    bench_list: Path,
    k_list=(5, 10, 20),
) -> Path:
    out = Path("artifacts/bench/ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # assoc-score proxy from promoted order
    prom = pd.read_csv(promoted, sep="\t")
    prom = prom[prom["type"].str.lower() == "gene"].reset_index(drop=True)
    n = max(len(prom), 1)
    prom["assoc_score"] = np.linspace(1.0, 0.0, num=n, endpoint=False)
    assoc = _load_names_scores(prom[["name", "assoc_score"]], "assoc_score")

    # prior+path+total scores
    pf = pd.read_csv(promoted_full, sep="\t") if promoted_full.exists() else prom
    prior = (
        _load_names_scores(pf[["name", "prior_score"]].fillna(0.0), "prior_score")
        if "prior_score" in pf.columns
        else assoc * 0
    )
    path = (
        _load_names_scores(pf[["name", "path_score"]].fillna(0.0), "path_score")
        if "path_score" in pf.columns
        else assoc * 0
    )
    total = (
        _load_names_scores(pf[["name", "total_score"]].fillna(0.0), "total_score")
        if "total_score" in pf.columns
        else assoc
    )

    # benchmark ground-truth as y_true
    if bench_list.suffix == ".json":
        data = json.loads(bench_list.read_text()) if bench_list.exists() else []
        anchors = [
            str(x).upper()
            for x in (data.get("biomarkers", []) if isinstance(data, dict) else data)
        ]
    else:
        b = (
            pd.read_csv(bench_list, sep="\t")
            if bench_list.exists()
            else pd.DataFrame(columns=["gene"])
        )
        col = (
            "gene" if "gene" in b.columns else ("name" if "name" in b.columns else None)
        )
        anchors = b[col].astype(str).str.upper().tolist() if col else []
    y_true = pd.Series(
        0.0, index=sorted(set(assoc.index) | set(total.index) | set(anchors))
    )
    y_true.loc[anchors] = 1.0

    report = {"k": list(k_list), "metrics": {}}
    for name, score in {
        "assoc": assoc,
        "+priors": prior,
        "+causal": path,
        "+priors+causal": total,
    }.items():
        m = {}
        for k in k_list:
            m[f"p@{k}"] = precision_at_k(y_true, score, k)
            m[f"r@{k}"] = recall_at_k(y_true, score, k)
        report["metrics"][name] = m

    out.write_text(json.dumps(report, indent=2))
    return out


def compute_ot_correlation(ranked_full: Path, ot_tsv: Path) -> Path | None:
    if not ranked_full.exists() or not ot_tsv.exists():
        return None
    rf = pd.read_csv(ranked_full, sep="\t")
    ot = pd.read_csv(ot_tsv, sep="\t")
    if "name" not in rf.columns or "total_score" not in rf.columns:
        return None
    # Expect ot columns: name, score
    col = "name" if "name" in ot.columns else ("gene" if "gene" in ot.columns else None)
    sc = "score" if "score" in ot.columns else None
    if not (col and sc):
        return None
    a = rf[["name", "total_score"]].rename(columns={"name": "gene"})
    b = ot[[col, sc]].rename(columns={col: "gene", sc: "ot_score"})
    a["gene"] = a["gene"].astype(str).str.upper()
    b["gene"] = b["gene"].astype(str).str.upper()
    m = a.merge(b, on="gene", how="inner")
    if m.empty:
        return None
    out = Path("artifacts/bench/opentargets_correlation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    corr = float(m["total_score"].corr(m["ot_score"], method="spearman"))
    out.write_text(json.dumps({"n_intersect": int(len(m)), "spearman": corr}, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-json",
        type=Path,
        default=Path("benchmarks/cardiovascular_markers.json"),
    )
    ap.add_argument("--promoted", type=Path, default=Path("artifacts/promoted.tsv"))
    ap.add_argument(
        "--promoted-full", type=Path, default=Path("artifacts/promoted_full.tsv")
    )
    ap.add_argument("--ranked", type=Path, default=Path("artifacts/ranked.tsv"))
    ap.add_argument(
        "--ot-baseline",
        type=Path,
        default=Path("data/external/opentargets/LDL_C_targets.tsv"),
    )
    args = ap.parse_args()

    ab = compute_ablation(
        args.promoted, args.promoted_full, args.ranked, args.bench_json
    )
    oc = compute_ot_correlation(args.promoted_full, args.ot_baseline)
    print(f"[ablation] {ab}")
    if oc:
        print(f"[baseline] {oc}")


if __name__ == "__main__":
    main()
