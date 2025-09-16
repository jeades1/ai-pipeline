#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_benchmarks.py

Usage:
  python tools/bench/compare_benchmarks.py \
      --promoted artifacts/promoted.tsv \
      --bench data/benchmarks/sepsis_aki_biomarkers.tsv \
      --out artifacts/bench

Outputs (created in --out):
  - benchmark_report.json
  - benchmark_report.md
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ---------------------------- utils ----------------------------


def norm_symbol(x: str) -> str:
    """Normalize gene/feature symbol for robust matching."""
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def coerce_name_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Ensure a 'name' column exists.
    Accepts 'name', 'gene', or 'symbol' and renames to 'name' if needed.
    """
    candidates = [c for c in ["name", "gene", "symbol"] if c in df.columns]
    if not candidates:
        raise ValueError(f"{label} missing required column: name (or gene/symbol).")
    col = candidates[0]
    if col != "name":
        df = df.rename(columns={col: "name"})
    return df


def ensure_cols(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Guarantee the compare schema: name, layer, type
    - Adds defaults if layer/type are missing
    - Normalizes 'name'
    - Drops duplicates
    """
    df = coerce_name_column(df, label).copy()
    if "layer" not in df.columns:
        df["layer"] = "transcriptomic"
    if "type" not in df.columns:
        df["type"] = "gene"

    # normalize symbol
    df["name"] = df["name"].map(norm_symbol)

    # Canonical order & dedupe
    return df[["name", "layer", "type"]].drop_duplicates()


def precision_at_k(
    ranked_names: List[str], bench_set: set, ks=(5, 10, 20, 50, 100)
) -> Dict[str, float]:
    """
    Compute precision@k for a ranked list of names.
    If ranked_names is shorter than k, precision uses min(k, len(ranked)).
    """
    out = {}
    for k in ks:
        if not ranked_names:
            out[f"p@{k}"] = 0.0
            continue
        cutoff = min(k, len(ranked_names))
        hits = sum(1 for n in ranked_names[:cutoff] if n in bench_set)
        denom = cutoff if cutoff > 0 else 1
        out[f"p@{k}"] = round(hits / denom, 4)
    return out


def f1_score(precision: float, recall: float) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return 0.0
    return round(2 * precision * recall / denom, 4)


# ---------------------------- I/O ----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--promoted",
        required=True,
        help="TSV of promoted candidates (name/layer/type; order = ranking).",
    )
    ap.add_argument(
        "--bench",
        required=True,
        help="TSV of benchmark items (at least name; layer/type optional).",
    )
    ap.add_argument("--out", required=True, help="Output directory for reports.")
    return ap.parse_args()


# ---------------------------- main ----------------------------


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read promoted results
    promoted_raw = pd.read_csv(args.promoted, sep="\t")
    promoted = ensure_cols(promoted_raw, "promoted")

    # Handle both JSON and TSV benchmark formats
    bench_path = Path(args.bench)
    if bench_path.suffix == ".json":
        # Load JSON benchmark with aliases
        with open(bench_path) as f:
            bench_data = json.load(f)

        benchmark_markers = set(bench_data.get("markers", []))
        aliases = bench_data.get("aliases", {})

        # Expand aliases
        for alias, canonical in aliases.items():
            if isinstance(canonical, list):
                benchmark_markers.update(canonical)
            else:
                benchmark_markers.add(canonical)

        bench_set = benchmark_markers
        n_bench = len(bench_set)
    else:
        # TSV format
        bench_raw = pd.read_csv(args.bench, sep="\t")
        bench = ensure_cols(bench_raw, "benchmark")
        bench_set = set(bench["name"])
        n_bench = len(bench_set)

    # Ranked list (current order of rows = ranking)
    ranked_names = promoted["name"].tolist()
    promoted_set = set(ranked_names)

    # Overlap metrics
    hits_set = promoted_set & bench_set
    misses_set = bench_set - promoted_set

    n_prom = len(promoted_set)
    n_hits = len(hits_set)

    recall = round(n_hits / n_bench, 4) if n_bench else None
    # If you want an overall precision (not just p@k), define as hits/|promoted|:
    precision_overall = round(n_hits / n_prom, 4) if n_prom else None

    p_at_k = precision_at_k(ranked_names, bench_set)
    f1 = f1_score(precision_overall or 0.0, recall or 0.0)

    # Write JSON
    report_json = {
        "n_promoted": n_prom,
        "n_benchmark": n_bench,
        "n_found": n_hits,
        "precision_at_k": p_at_k,
        "precision": precision_overall,
        "recall": recall,
        "f1": f1,
        "hits": sorted(hits_set),
        "misses": sorted(misses_set),
        "promoted_preview": ranked_names[:20],
    }
    (outdir / "benchmark_report.json").write_text(json.dumps(report_json, indent=2))

    # Write Markdown
    md_lines: List[str] = []
    md_lines.append("# Benchmark Report\n")
    md_lines.append(f"- **Promoted (unique)**: {n_prom}")
    md_lines.append(f"- **Benchmark (unique)**: {n_bench}")
    md_lines.append(f"- **Hits (overlap)**: {n_hits}")
    md_lines.append("")
    md_lines.append("## Precision@K")
    for k, v in p_at_k.items():
        md_lines.append(f"- **{k}**: {v}")
    md_lines.append("")
    md_lines.append("## Overall")
    md_lines.append(f"- **Precision**: {precision_overall}")
    md_lines.append(f"- **Recall**: {recall}")
    md_lines.append(f"- **F1**: {f1}")
    md_lines.append("")
    md_lines.append("## Sample hits")
    md_lines.append(", ".join(sorted(list(hits_set))[:25]) or "_none_")
    md_lines.append("")
    md_lines.append("## Sample misses")
    md_lines.append(", ".join(sorted(list(misses_set))[:25]) or "_none_")
    md_lines.append("")
    (outdir / "benchmark_report.md").write_text("\n".join(md_lines))

    print(
        f"[bench] Wrote {outdir/'benchmark_report.md'} and {outdir/'benchmark_report.json'}"
    )
    print(f"[bench] Results -> {outdir}")
    print(f"[bench] Found markers: {sorted(hits_set)}")
    print(
        f"[bench] Precision@5: {p_at_k.get('p@5', 0)}, Precision@10: {p_at_k.get('p@10', 0)}"
    )


if __name__ == "__main__":
    main()
