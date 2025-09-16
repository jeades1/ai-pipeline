#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import itertools
import subprocess


ART = Path("artifacts")


def run_rerank(
    promoted: Path,
    kg_nodes: Path,
    kg_edges: Path,
    pri_manifest: Path,
    out_ranked: Path,
    weights: Tuple[float, float, float],
    pathway_terms: List[str] | None,
    penalize_immune: bool,
    immune_penalty: float,
) -> None:
    out_full = out_ranked.parent / f"{out_ranked.stem}_full.tsv"
    cmd = [
        "python3",
        "-m",
        "tools.bench.causal_prior_rerank",
        "--promoted",
        str(promoted),
        "--kg-nodes",
        str(kg_nodes),
        "--kg-edges",
        str(kg_edges),
        "--priors-manifest",
        str(pri_manifest),
        "--out-full",
        str(out_full),
        "--out-ranked",
        str(out_ranked),
        "--weights",
        str(weights[0]),
        str(weights[1]),
        str(weights[2]),
    ]
    if pathway_terms:
        cmd += ["--pathway-terms", *pathway_terms]
    if penalize_immune:
        cmd += ["--penalize-immune", "--immune-penalty", str(immune_penalty)]
    subprocess.check_call(cmd)


def eval_metrics(
    promoted_ranked: Path,
    bench_json: Path | None,
    bench_tsv: Path | None,
    k_list: List[int],
) -> Dict:
    out = ART / "bench" / "_tmp_metrics.json"
    cmd = [
        "python3",
        "-m",
        "tools.bench.compute_metrics",
        "--promoted",
        str(promoted_ranked),
        "--out",
        str(out),
        "--k-list",
        *[str(k) for k in k_list],
    ]
    if bench_tsv and bench_tsv.exists():
        cmd += ["--bench-tsv", str(bench_tsv)]
    if bench_json and bench_json.exists():
        cmd += ["--bench-json", str(bench_json)]
    subprocess.check_call(cmd)
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser(
        description="Grid-search assoc/prior/path weights to maximize P@K or MAP"
    )
    ap.add_argument("--disease", type=str, default="cardiovascular")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--promoted", type=Path, default=ART / "promoted.tsv")
    ap.add_argument("--kg-nodes", type=Path, default=ART / "kg_dump/kg_nodes.tsv")
    ap.add_argument("--kg-edges", type=Path, default=ART / "kg_dump/kg_edges.tsv")
    ap.add_argument(
        "--priors-manifest",
        type=Path,
        default=Path("data/processed/priors/manifest.json"),
    )
    ap.add_argument(
        "--bench-json",
        type=Path,
        default=Path("benchmarks/cardiovascular_markers.json"),
    )
    ap.add_argument(
        "--bench-tsv", type=Path, default=ART / "bench/cardiovascular_expanded.tsv"
    )
    ap.add_argument("--out-weights", type=Path, default=ART / "bench/weights.json")
    ap.add_argument(
        "--out-report", type=Path, default=ART / "bench/weight_sweep_report.json"
    )
    ap.add_argument("--penalize-immune", action="store_true")
    ap.add_argument("--immune-penalty", type=float, default=0.20)
    args = ap.parse_args()

    # pathway terms by disease
    pw_terms = {
        "cardiovascular": [
            "lipid",
            "cholesterol",
            "lipoprotein",
            "statin",
            "LDL",
            "HDL",
            "SREBP",
            "LXR",
            "cholesteryl",
            "apolipoprotein",
        ],
        "oncology": ["cell cycle", "apoptosis", "DNA damage", "p53"],
        "aki": ["kidney", "nephron", "renal"],
    }.get(args.disease, None)

    # Small grid
    assoc_grid = [0.2, 0.3]
    prior_grid = [0.4, 0.5, 0.6]
    path_grid = [0.1, 0.2, 0.3]
    combos = [
        (a, p, r) for a, p, r in itertools.product(assoc_grid, prior_grid, path_grid)
    ]

    results = []
    k_list = [5, 10, 20, 50, 100]
    for w in combos:
        out_ranked = ART / "bench" / f"ranked_a{w[0]}_p{w[1]}_r{w[2]}.tsv"
        run_rerank(
            args.promoted,
            args.kg_nodes,
            args.kg_edges,
            args.priors_manifest,
            out_ranked,
            w,
            pw_terms,
            args.penalize_immune,
            args.immune_penalty,
        )
        m = eval_metrics(out_ranked, args.bench_json, args.bench_tsv, k_list)
        p20 = float(m.get("metrics", {}).get("@20", {}).get("precision", 0.0))
        map_k = float(m.get("global", {}).get("MAP@maxK", 0.0))
        results.append({"weights": w, "p@20": p20, "MAP@maxK": map_k, "metrics": m})

    # pick best by P@20, tie-break by MAP
    best = (
        sorted(results, key=lambda x: (x["p@20"], x["MAP@maxK"]), reverse=True)[0]
        if results
        else None
    )

    if best:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(
            json.dumps(
                {
                    "grid": results,
                    "best": best,
                },
                indent=2,
            )
        )
        bw = best["weights"]
        args.out_weights.parent.mkdir(parents=True, exist_ok=True)
        args.out_weights.write_text(
            json.dumps({"assoc": bw[0], "prior": bw[1], "path": bw[2]}, indent=2)
        )
        print(
            f"[weight-sweep] Best weights: {bw}; wrote {args.out_weights} and {args.out_report}"
        )
    else:
        print("[weight-sweep] No results produced")


if __name__ == "__main__":
    main()
