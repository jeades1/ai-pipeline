from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import math


def precision_at_k(y_true: pd.Series, y_score: pd.Series, k: int = 20) -> float:
    order = y_score.sort_values(ascending=False).index[:k]
    return float(y_true.loc[order].sum() / max(k, 1))


def recall_at_k(y_true: pd.Series, y_score: pd.Series, k: int = 20) -> float:
    order = y_score.sort_values(ascending=False).index[:k]
    denom = float(y_true.sum()) or 1.0
    return float(y_true.loc[order].sum() / denom)


def bootstrap_ci(
    metric_fn, y_true, y_score, k: int, n_boot: int = 500, seed: int = 42
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    base = metric_fn(y_true, y_score, k)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        bs = rng.choice(idx, size=len(idx), replace=True)
        vals.append(metric_fn(y_true.iloc[bs], y_score.iloc[bs], k))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(base), (float(lo), float(hi))


def precision_wilson_ci(
    y_true: pd.Series, y_score: pd.Series, k: int = 20, z: float = 1.96
) -> Tuple[float, Tuple[float, float]]:
    order = y_score.sort_values(ascending=False).index[:k]
    x = float(y_true.loc[order].sum())  # successes (hits in top-k)
    n = float(max(1, k))
    p = x / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (
        z * math.sqrt(max(0.0, (p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n))))
    ) / denom
    return float(p), (float(max(0.0, center - half)), float(min(1.0, center + half)))


def load_promoted(promoted_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(promoted_tsv, sep="\t")
    if not {"name", "type"}.issubset(df.columns):
        raise ValueError("promoted.tsv needs columns: name, type")
    genes = df[df["type"].str.lower() == "gene"].copy()
    # A simple score proxy: earlier rows are higher ranked in this format
    genes["score"] = np.linspace(1.0, 0.0, num=len(genes), endpoint=False)
    genes = genes.drop_duplicates("name").set_index("name")
    # Normalize index to uppercase for case-insensitive alignment
    genes.index = genes.index.astype(str).str.upper()
    return genes


def load_benchmark(bench_tsv: Path) -> pd.Series:
    # TSV with a column "gene" (or name) listing known positives
    df = pd.read_csv(bench_tsv, sep="\t")
    col = "gene" if "gene" in df.columns else ("name" if "name" in df.columns else None)
    if col is None:
        raise ValueError("benchmark TSV must contain 'gene' or 'name' column")
    positives = set(df[col].astype(str).str.upper())
    # Build y_true over union of promoted and benchmark positives
    return pd.Series(1.0, index=sorted(positives))


def align_labels_scores(
    y_true_pos: pd.Series, promoted: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    # Build union index
    idx = sorted(set(promoted.index) | set(y_true_pos.index))
    y_true = pd.Series(0.0, index=idx)
    y_true.loc[y_true_pos.index] = 1.0
    y_score = pd.Series(0.0, index=idx)
    inter = promoted.reindex(idx).fillna({"score": 0.0})
    y_score.loc[inter.index] = inter["score"].values
    return y_true, y_score


def main():
    ap = argparse.ArgumentParser(description="Run benchmarking with bootstrap CIs")
    ap.add_argument("--promoted", type=Path, default=Path("artifacts/promoted.tsv"))
    ap.add_argument("--bench", type=Path, default=Path("benchmarks/markers.json"))
    ap.add_argument(
        "--bench-tsv", type=Path, default=Path("data/benchmarks/benchmarks.tsv")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("artifacts/bench/benchmark_report.json")
    )
    ap.add_argument("--k", type=int, default=25)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    promoted = load_promoted(args.promoted)

    # Prefer TSV if provided; also union with JSON if available
    y_true_pos = pd.Series(dtype=float)
    if args.bench_tsv.exists():
        y_true_pos = load_benchmark(args.bench_tsv)
    # Union JSON anchors if provided
    try:
        if args.bench and Path(args.bench).exists():
            with open(args.bench) as f:
                data = json.load(f)
            if isinstance(data, dict) and "biomarkers" in data:
                genes = [
                    str(it.get("name") or it.get("gene") or "").upper()
                    for it in data["biomarkers"]
                ]
            elif isinstance(data, list):
                genes = [str(x).upper() for x in data]
            else:
                genes = []
            if genes:
                jser = pd.Series(1.0, index=sorted(set(genes)))
                y_true_pos = pd.concat([y_true_pos, jser]).groupby(level=0).max()
    except Exception:
        pass

    y_true, y_score = align_labels_scores(y_true_pos, promoted)

    p_at_k, p_ci = bootstrap_ci(precision_at_k, y_true, y_score, args.k)
    p_at_k_w, p_ci_w = precision_wilson_ci(y_true, y_score, args.k)
    r_at_k, r_ci = bootstrap_ci(recall_at_k, y_true, y_score, args.k)

    report: Dict = {
        "k": args.k,
        "n_candidates": int(len(y_true)),
        "precision_at_k": {"value": p_at_k, "ci95": list(p_ci)},
        "precision_at_k_wilson": {"value": p_at_k_w, "ci95": list(p_ci_w)},
        "recall_at_k": {"value": r_at_k, "ci95": list(r_ci)},
    }

    args.out.write_text(json.dumps(report, indent=2))
    print(f"[bench] Wrote {args.out}")


if __name__ == "__main__":
    main()
