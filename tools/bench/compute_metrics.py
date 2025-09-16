#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List
import argparse
import pandas as pd

from tools.bench.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision_at_k,
    ndcg_at_k,
    mrr,
    hits_at_k,
    coverage,
)


def load_benchmark(bench_json: Path | None, bench_tsv: Path | None) -> List[str]:
    genes: List[str] = []
    # TSV
    if bench_tsv and bench_tsv.exists():
        try:
            df = pd.read_csv(bench_tsv, sep="\t")
            col = "gene" if "gene" in df.columns else df.columns[0]
            genes.extend([str(x) for x in df[col].dropna().astype(str).tolist()])
        except Exception:
            pass
    # JSON
    if bench_json and bench_json.exists():
        try:
            data = json.loads(bench_json.read_text())
            if isinstance(data, dict) and "biomarkers" in data:
                for item in data["biomarkers"]:
                    n = item.get("name") or item.get("gene")
                    if n:
                        genes.append(str(n))
            elif isinstance(data, list):
                genes.extend([str(x) for x in data])
        except Exception:
            pass
    # Deduplicate case-insensitively while preserving original forms
    seen = set()
    uniq: List[str] = []
    for g in genes:
        key = g.upper()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(g)
    return uniq


def build_relevance_vector(promoted: Path, anchors: List[str], max_k: int) -> List[int]:
    df = pd.read_csv(promoted, sep="\t")
    # genes only, keep order
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "gene"]
    names = df["name"].astype(str).tolist()
    anchors_up = set([a.upper() for a in anchors])
    rel = [1 if n.upper() in anchors_up else 0 for n in names]
    return rel[:max_k]


def main():
    ap = argparse.ArgumentParser(
        description="Compute extended ranking metrics for a promoted list vs. anchors"
    )
    ap.add_argument("--promoted", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bench-json", type=str, default="")
    ap.add_argument("--bench-tsv", type=str, default="")
    ap.add_argument("--k-list", type=int, nargs="*", default=[5, 10, 20, 50, 100])
    ap.add_argument(
        "--slices-json",
        type=str,
        default="",
        help="Optional JSON mapping slice name -> list of genes for stratified metrics",
    )
    args = ap.parse_args()

    promoted = Path(args.promoted)
    out = Path(args.out)
    bench_json = Path(args.bench_json) if args.bench_json else None
    bench_tsv = Path(args.bench_tsv) if args.bench_tsv else None

    anchors = load_benchmark(bench_json, bench_tsv)
    total_rel = len(set([a.upper() for a in anchors]))

    # Build relevance vector up to max K we need
    max_k = max(args.k_list) if args.k_list else 100
    rel = build_relevance_vector(promoted, anchors, max_k)

    report = {
        "total_relevant": total_rel,
        "k_list": args.k_list,
        "metrics": {},
        "global": {
            "MRR": mrr(rel),
            "MAP@maxK": average_precision_at_k(rel, len(rel)),
            "coverage": coverage(sum(rel), total_rel) if total_rel else 0.0,
        },
    }

    for k in args.k_list:
        report["metrics"][f"@{k}"] = {
            "precision": precision_at_k(rel, k),
            "recall": recall_at_k(rel, total_rel, k),
            "f1": f1_at_k(rel, total_rel, k),
            "hits": hits_at_k(rel, k),
            "AP": average_precision_at_k(rel, k),
            "nDCG": ndcg_at_k(rel, k),
        }

    # Optional: per-slice metrics for personalized/stratified assessment
    slices_json = Path(args.slices_json) if args.slices_json else None
    if slices_json and slices_json.exists():
        try:
            slices = json.loads(slices_json.read_text())
            report["slices"] = {}
            for sname, sgenes in slices.items():
                sanchors = [str(x) for x in sgenes]
                s_total = len(set([a.upper() for a in sanchors]))
                s_rel_full = build_relevance_vector(promoted, sanchors, max_k)
                s_metrics = {
                    "total_relevant": s_total,
                    "global": {
                        "MRR": mrr(s_rel_full),
                        "MAP@maxK": average_precision_at_k(s_rel_full, len(s_rel_full)),
                        "coverage": (
                            coverage(sum(s_rel_full), s_total) if s_total else 0.0
                        ),
                    },
                    "metrics": {},
                }
                for k in args.k_list:
                    s_metrics["metrics"][f"@{k}"] = {
                        "precision": precision_at_k(s_rel_full, k),
                        "recall": recall_at_k(s_rel_full, s_total, k),
                        "f1": f1_at_k(s_rel_full, s_total, k),
                        "hits": hits_at_k(s_rel_full, k),
                        "AP": average_precision_at_k(s_rel_full, k),
                        "nDCG": ndcg_at_k(s_rel_full, k),
                    }
                report["slices"][sname] = s_metrics
        except Exception as e:
            report["slices_error"] = str(e)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[metrics] wrote {out}")


if __name__ == "__main__":
    main()
