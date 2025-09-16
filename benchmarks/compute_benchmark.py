#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from typing import Dict

# Default benchmarks for different diseases
BENCHMARK_FILES = {
    "default": Path("benchmarks/markers.json"),
    "oncology": Path("benchmarks/oncology_markers.json"),
    "cardiovascular": Path("benchmarks/cardiovascular_markers.json"),
    "aki": Path("benchmarks/aki_markers.json"),  # still supported if present
}


def _load_anchors_and_aliases(p: Path) -> tuple[list[str], dict[str, str]]:
    anchors: list[str] = []
    alias_map: dict[str, str] = {}
    if not p.exists():
        return anchors, alias_map
    data = json.loads(p.read_text())
    # Anchors can be under 'biomarkers' or 'markers' or a plain list
    seq = None
    if isinstance(data, dict) and "biomarkers" in data:
        seq = data.get("biomarkers")
    elif isinstance(data, dict) and "markers" in data:
        seq = data.get("markers")
    else:
        seq = data
    if isinstance(seq, list):
        for item in seq:
            if isinstance(item, dict):
                n = item.get("name") or item.get("gene")
                if n:
                    anchors.append(str(n))
            else:
                anchors.append(str(item))
    # Aliases mapping
    if isinstance(data, dict) and "aliases" in data:
        for k, v in data.get("aliases", {}).items():
            if isinstance(v, list):
                for vv in v:
                    alias_map[str(vv).upper()] = (
                        str(k).upper() if k in anchors else str(vv).upper()
                    )
            else:
                alias_map[str(k).upper()] = str(v).upper()
    return anchors, alias_map


def compute_benchmark(
    promoted_tsv: Path,
    out_json: Path,
    benchmark_file: Path | None = None,
    disease: str = "default",
) -> Path:
    """Compute benchmark metrics for a specific disease."""

    # Select benchmark file
    if benchmark_file is None:
        benchmark_file = BENCHMARK_FILES.get(disease, BENCHMARK_FILES["default"])

    if not benchmark_file.exists():
        print(f"[benchmark] Warning: {benchmark_file} not found, using empty benchmark")
        anchors_raw, aliases = [], {}
    else:
        anchors_raw, aliases = _load_anchors_and_aliases(benchmark_file)

    anchors = [a.upper() for a in anchors_raw if a]

    df = pd.read_csv(promoted_tsv, sep="\t")
    # Handle duplicates by taking the first occurrence of each gene (they should be sorted by score)
    df_unique = df.drop_duplicates(subset=["name"], keep="first")
    names = df_unique["name"].astype(str).str.upper().tolist()
    # Map aliases to canonical names if applicable
    inv_alias = {alias: canon for alias, canon in aliases.items()}
    canonical_names = [inv_alias.get(n, n) for n in names]
    name_to_rank = {n: i + 1 for i, n in enumerate(canonical_names)}
    hits = [a for a in anchors if a in name_to_rank]
    misses = [a for a in anchors if a not in name_to_rank]

    def hits_in_top_k(k: int) -> int:
        return sum(1 for h in hits if name_to_rank[h] <= k)

    def precision_at_k(k: int) -> float:
        # true precision@k: hits among top-k divided by k
        return hits_in_top_k(k) / float(max(1, k))

    def recall_at_k(k: int) -> float:
        # recall within top-k relative to total anchors
        return (
            hits_in_top_k(k) / float(max(1, len(anchors))) if len(anchors) > 0 else 0.0
        )

    out = {
        "disease": disease,
        "benchmark_file": str(benchmark_file),
        "n_promoted": len(names),
        "n_benchmark": len(anchors),
        "n_found": len(hits),
        # Report both true precision@k and recall@k for transparency
        "precision_at_k": {f"p@{k}": precision_at_k(k) for k in [5, 10, 20, 50, 100]},
        "recall_at_k": {f"r@{k}": recall_at_k(k) for k in [5, 10, 20, 50, 100]},
        "precision": round(len(hits) / max(1, len(names)), 4),
        "recall": (
            round(len(hits) / max(1, len(anchors)), 4) if len(anchors) > 0 else 0.0
        ),
        "f1": 0.0,
        "hits": hits,
        "misses": misses,
        "ranks": {h: name_to_rank[h] for h in hits},
        "promoted_preview": names[:20],
    }
    # F1 (treat precision as hits/len(names), recall as above)
    p = out["precision"]
    r = out["recall"]
    out["f1"] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    return out_json


def compute_cross_disease_benchmarks(
    promoted_tsv: Path, out_dir: Path
) -> Dict[str, Path]:
    """Compute benchmarks across multiple diseases."""
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for disease, benchmark_file in BENCHMARK_FILES.items():
        out_json = out_dir / f"{disease}_benchmark_report.json"
        try:
            compute_benchmark(promoted_tsv, out_json, benchmark_file, disease)
            results[disease] = out_json
            print(f"[cross-disease] {disease}: {out_json}")
        except Exception as e:
            print(f"[cross-disease] {disease} failed: {e}")

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--promoted", type=Path, default=Path("artifacts/promoted.tsv"))
    ap.add_argument(
        "--out", type=Path, default=Path("artifacts/bench/benchmark_report.json")
    )
    ap.add_argument(
        "--disease", type=str, default="default", help="Context for benchmark"
    )
    ap.add_argument(
        "--cross-disease", action="store_true", help="Run all disease benchmarks"
    )
    args = ap.parse_args()

    if args.cross_disease:
        results = compute_cross_disease_benchmarks(args.promoted, args.out.parent)
        print(f"[bench] Cross-disease results: {list(results.keys())}")
    else:
        p = compute_benchmark(args.promoted, args.out, disease=args.disease)
        print(f"[bench] wrote {p}")
