from __future__ import annotations

"""
First-win orchestrator: produce a robust, checkable demo showing the causal-first,
multi-omics differentiator with accepted benchmarks and concise causal traces.

Outputs under artifacts/:
  - bench/benchmark_report.json (precision/recall@k with bootstrap CIs if TSV provided)
  - bench/<disease>_benchmark_report.json (anchor hit summary)
  - audit_multiomics.md (structure + reachability)
  - first_win_report.md (one-stop summary with metrics and top-5 causal traces)

Usage (via CLI wrapper):
  python -m tools.bench.first_win --disease oncology --k 20
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import pandas as pd
import networkx as nx

ART = Path("artifacts")
KG_NODES = ART / "kg_dump" / "kg_nodes.tsv"
KG_EDGES = ART / "kg_dump" / "kg_edges.tsv"
PROMOTED = ART / "promoted.tsv"


def ensure_promoted() -> Path:
    """Build artifacts/promoted.tsv from available outputs if missing."""
    if PROMOTED.exists():
        return PROMOTED
    # best-effort build using existing tool
    from subprocess import check_call

    check_call(["python3", "tools/bench/build_promoted.py", str(ART)])
    if not PROMOTED.exists():
        raise SystemExit("[first-win] promoted.tsv not found and could not be built.")
    return PROMOTED


def _json_to_tsv_markers(json_path: Path, out_tsv: Path) -> Path:
    data = json.loads(json_path.read_text()) if json_path.exists() else []
    genes: List[str] = []
    if isinstance(data, dict) and "biomarkers" in data:
        for item in data["biomarkers"]:
            n = item.get("name") or item.get("gene")
            if n:
                genes.append(str(n))
    elif isinstance(data, list):
        genes = [str(x) for x in data]
    df = pd.DataFrame({"gene": genes})
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    return out_tsv


def run_benchmarks(promoted_tsv: Path, disease: str, k: int) -> Tuple[Path, Path]:
    """Run both CI bootstrap bench and anchor hit summary."""
    bench_out = ART / "bench"
    bench_out.mkdir(parents=True, exist_ok=True)

    # Select benchmark JSON
    disease_map = {
        "default": Path("benchmarks/markers.json"),
        "oncology": Path("benchmarks/oncology_markers.json"),
        "cardiovascular": Path("benchmarks/cardiovascular_markers.json"),
        "aki": Path("benchmarks/aki_markers.json"),
    }
    bench_json = disease_map.get(disease, disease_map["default"])

    # Prefer expanded TSV if present
    bench_tsv = bench_out / f"{disease}_bench.tsv"
    expanded_tsv = bench_out / f"{disease}_expanded.tsv"
    if expanded_tsv.exists():
        bench_tsv = expanded_tsv
    elif bench_json.exists():
        _json_to_tsv_markers(bench_json, bench_tsv)

    # 1) Bootstrap CI report
    ci_json = bench_out / "benchmark_report.json"
    from subprocess import check_call

    args = [
        "python3",
        "benchmarks/run_benchmarks.py",
        "--promoted",
        str(promoted_tsv),
        "--out",
        str(ci_json),
        "--k",
        str(k),
    ]
    if bench_tsv.exists():
        args += ["--bench-tsv", str(bench_tsv)]
    if bench_json.exists():
        args += ["--bench", str(bench_json)]
    check_call(args)

    # 2) Anchor hit summary per disease
    disease_json = bench_out / f"{disease}_benchmark_report.json"
    check_call(
        [
            "python3",
            "benchmarks/compute_benchmark.py",
            "--promoted",
            str(promoted_tsv),
            "--out",
            str(disease_json),
            "--disease",
            str(disease),
        ]
    )

    # 3) Extended metrics report (general and personalized readiness)
    try:
        ext_json = bench_out / "metrics_extended.json"
        # Use the same benchmark TSV/JSON for anchors; allow optional slices file if present
        slices_json = Path("benchmarks/slices.json")
        args_ext = [
            "python3",
            "-m",
            "tools.bench.compute_metrics",
            "--promoted",
            str(promoted_tsv),
            "--out",
            str(ext_json),
            "--k-list",
            "5",
            "10",
            "20",
            "50",
            "100",
        ]
        if bench_tsv.exists():
            args_ext += ["--bench-tsv", str(bench_tsv)]
        if bench_json.exists():
            args_ext += ["--bench-json", str(bench_json)]
        if slices_json.exists():
            args_ext += ["--slices-json", str(slices_json)]
        check_call(args_ext)
    except Exception:
        pass

    return ci_json, disease_json


def load_kg() -> nx.MultiDiGraph | None:
    nodes = (
        pd.read_csv(KG_NODES, sep="\t", low_memory=False)
        if KG_NODES.exists()
        else pd.DataFrame()
    )
    edges = (
        pd.read_csv(KG_EDGES, sep="\t", low_memory=False)
        if KG_EDGES.exists()
        else pd.DataFrame()
    )
    if nodes.empty or edges.empty:
        return None
    G = nx.MultiDiGraph()
    for _, r in nodes.iterrows():
        nid = str(r["id"])
        G.add_node(
            nid,
            kind=r.get("kind", ""),
            layer=r.get("layer", ""),
            name=r.get("name", nid),
        )
    for _, r in edges.iterrows():
        s = str(r["s"])
        o = str(r["o"])
        p = str(r.get("predicate", ""))
        G.add_edge(s, o, key=p, predicate=p, provenance=r.get("provenance", ""))
    return G


def _find_node_by_name(nodes_df: pd.DataFrame, name: str) -> str | None:
    m = nodes_df[nodes_df["name"].astype(str).str.upper() == str(name).upper()]
    if not m.empty:
        return str(m.iloc[0]["id"])
    m2 = nodes_df[nodes_df["id"].astype(str).str.upper() == str(name).upper()]
    if not m2.empty:
        return str(m2.iloc[0]["id"])
    return None


def shortest_path_to_pathway(
    G: nx.MultiDiGraph, nodes_df: pd.DataFrame, gene_name: str, max_hops: int = 3
) -> List[Tuple[str, str, str]]:
    """Return list of (u, predicate, v) edges for a short path to any pathway node; empty if none within hops."""
    src = _find_node_by_name(nodes_df, gene_name)
    if not src or src not in G:
        return []
    # BFS frontier with parent tracking
    from collections import deque

    q = deque([(src, None)])
    parent: Dict[str, Tuple[str, str]] = {
        src: ("", "")
    }  # node -> (prev_node, predicate)
    depth = {src: 0}
    target = None
    for u, _ in q:
        pass  # quiet linter
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if depth[u] >= max_hops:
            continue
        for _, v, k in G.out_edges(u, keys=True):
            if v in parent:
                continue
            parent[v] = (u, str(k))
            depth[v] = depth[u] + 1
            # pathway?
            kind = str(G.nodes[v].get("kind", ""))
            if kind.lower() == "pathway":
                target = v
                dq.clear()
                break
            dq.append(v)
        if target:
            break
    if not target:
        return []
    # Reconstruct path
    path_nodes = [target]
    while path_nodes[-1] != src:
        prev, pred = parent[path_nodes[-1]]
        path_nodes.append(prev)
    path_nodes.reverse()
    edges: List[Tuple[str, str, str]] = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        # pick first predicate observed between u->v
        preds = [d.get("predicate", "") for k, d in G.get_edge_data(u, v).items()]
        pred = preds[0] if preds else ""
        edges.append((u, pred, v))
    return edges


def write_first_win_report(
    disease: str,
    k: int,
    ci_json: Path,
    disease_json: Path,
    audit_md: Path,
    top_traces: List[Tuple[str, List[Tuple[str, str, str]]]],
    nodes_df: pd.DataFrame,
) -> Path:
    out = ART / "first_win_report.md"
    lines: List[str] = []
    lines += ["# First Win — Causal-First Multi-Omics\n"]
    lines += [f"Disease: {disease}\n", f"Top-K: {k}\n", ""]

    # Metrics
    try:
        ci = json.loads(ci_json.read_text())
        lines += ["## Benchmarks\n"]
        lines += [
            f"- Precision@{k}: {ci['precision_at_k']['value']:.3f} (95% CI {ci['precision_at_k']['ci95'][0]:.3f}-{ci['precision_at_k']['ci95'][1]:.3f})"
        ]
        lines += [
            f"- Recall@{k}: {ci['recall_at_k']['value']:.3f} (95% CI {ci['recall_at_k']['ci95'][0]:.3f}-{ci['recall_at_k']['ci95'][1]:.3f})\n"
        ]
    except Exception:
        lines += ["## Benchmarks\n", "- CI report unavailable\n"]
    try:
        dj = json.loads(disease_json.read_text())
        hits = dj.get("hits", [])
        miss = dj.get("misses", [])
        lines += [
            f"- Anchor hits: {len(hits)}/{dj.get('n_benchmark', 0)}",
            f"- Hits: {', '.join(hits) if hits else '—'}",
            f"- Misses: {', '.join(miss) if miss else '—'}\n",
        ]
    except Exception:
        lines += ["- Anchor summary unavailable\n"]

    lines += ["## Structural audit\n", f"See: {audit_md}\n", ""]

    # Ablation and OT correlation (optional)
    ab_json = ART / "bench/ablation.json"
    ot_corr = ART / "bench/opentargets_correlation.json"
    if ab_json.exists():
        try:
            ab = json.loads(ab_json.read_text())
            lines += ["## Ablation (assoc vs +priors vs +causal)\n"]
            for name, m in ab.get("metrics", {}).items():
                parts = [
                    f"{k}: p@{k}={m.get(f'p@{k}',0):.2f}, r@{k}={m.get(f'r@{k}',0):.2f}"
                    for k in ab.get("k", [20])
                ]
                lines += [f"- {name}: " + "; ".join(parts)]
            lines += [""]
        except Exception:
            pass
    if ot_corr.exists():
        try:
            oc = json.loads(ot_corr.read_text())
            lines += [
                "## Open Targets baseline\n",
                f"- Spearman corr (n={oc.get('n_intersect',0)}): {oc.get('spearman',0):.3f}\n",
            ]
        except Exception:
            pass

    # Traces
    lines += ["## Causal/Pathway traces for top candidates (<=3 hops)\n"]
    for gene, edges in top_traces:
        lines += [f"- {gene}"]
        if not edges:
            lines += ["  - (no pathway within 3 hops)"]
            continue
        for u, pred, v in edges:
            u_name = (
                str(nodes_df[nodes_df["id"].astype(str) == str(u)]["name"].iloc[0])
                if not nodes_df.empty
                else u
            )
            v_name = (
                str(nodes_df[nodes_df["id"].astype(str) == str(v)]["name"].iloc[0])
                if not nodes_df.empty
                else v
            )
            lines += [f"  - {u_name} -[{pred}]-> {v_name}"]
    lines += [
        "",
        "_This report is auto-generated. All inputs live under artifacts/._\n",
    ]

    out.write_text("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Run a first-win benchmark + audit with causal traces"
    )
    ap.add_argument(
        "--disease",
        type=str,
        default="oncology",
        help="Benchmark panel to use: oncology|cardiovascular|default|aki",
    )
    ap.add_argument("--k", type=int, default=20, help="Top-K for precision/recall")
    ap.add_argument(
        "--top-trace", type=int, default=5, help="How many top genes to trace"
    )
    args = ap.parse_args()

    prom = ensure_promoted()

    # Optional: rerank with causal+priors if KG and priors exist
    kg_ok = KG_NODES.exists() and KG_EDGES.exists()
    pri_manifest = Path("data/processed/priors/manifest.json")
    if kg_ok and pri_manifest.exists():
        from subprocess import check_call

        # Map disease to pathway terms to emphasize relevant biology in path score
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
        # Optional learned weights
        weights_json = ART / "bench/weights.json"
        weight_args = []
        try:
            if weights_json.exists():
                w = json.loads(weights_json.read_text())
                if all(k in w for k in ("assoc", "prior", "path")):
                    weight_args = [
                        "--weights",
                        str(w["assoc"]),
                        str(w["prior"]),
                        str(w["path"]),
                    ]
            else:
                # Disease-specific sensible defaults when no learned weights available
                if args.disease == "cardiovascular":
                    weight_args = ["--weights", "0.25", "0.55", "0.20"]
                elif args.disease == "aki":
                    weight_args = ["--weights", "0.30", "0.50", "0.20"]
        except Exception:
            weight_args = []

        cmd = [
            "python3",
            "-m",
            "tools.bench.causal_prior_rerank",
            "--promoted",
            str(prom),
            "--kg-nodes",
            str(KG_NODES),
            "--kg-edges",
            str(KG_EDGES),
            "--priors-manifest",
            str(pri_manifest),
            "--out-full",
            str(ART / "promoted_full.tsv"),
            "--out-ranked",
            str(ART / "ranked.tsv"),
        ]
        if pw_terms:
            cmd += ["--pathway-terms", *pw_terms]
        cmd += weight_args
        # Penalize immune artifacts in non-immune disease contexts
        if args.disease in ("cardiovascular", "aki"):
            cmd += ["--penalize-immune", "--immune-penalty", "0.20"]
        check_call(cmd)
        # Overwrite promoted.tsv with ranked order to reflect improved scores
        ranked_p = ART / "ranked.tsv"
        if ranked_p.exists():
            try:
                r = pd.read_csv(ranked_p, sep="\t")
                keep = r[["name", "layer", "type"]].dropna().drop_duplicates()
                keep.to_csv(ART / "promoted.tsv", sep="\t", index=False)
                prom = ART / "promoted.tsv"
            except Exception:
                pass

    # Run benchmarks
    ci_json, disease_json = run_benchmarks(prom, args.disease, args.k)

    # Run audit (reusing existing module)
    from subprocess import check_call

    audit_md = ART / "audit_multiomics.md"
    if KG_NODES.exists() and KG_EDGES.exists():
        check_call(
            [
                "python3",
                "-m",
                "tools.maint.audit_multiomics",
                "--kg-nodes",
                str(KG_NODES),
                "--kg-edges",
                str(KG_EDGES),
                "--ranked",
                (
                    str(ART / "promoted_full.tsv")
                    if (ART / "promoted_full.tsv").exists()
                    else str(prom)
                ),
                "--out",
                str(audit_md),
                "--top-n",
                "100",
            ]
        )
    else:
        audit_md.write_text("KG dump not found; audit skipped.")

    # Optional: ablation/baseline step
    try:
        from subprocess import check_call

        check_call(
            [
                "python3",
                "-m",
                "tools.bench.ablation_and_baseline",
                "--bench-json",
                str(
                    Path("benchmarks")
                    / (
                        f"{args.disease}_markers.json"
                        if (
                            Path("benchmarks") / f"{args.disease}_markers.json"
                        ).exists()
                        else "cardiovascular_markers.json"
                    )
                ),
            ]
        )
    except Exception:
        pass

    # Traces for top-N (skip if KG missing)
    nodes_df = pd.read_csv(KG_NODES, sep="\t") if KG_NODES.exists() else pd.DataFrame()
    traces: List[Tuple[str, List[Tuple[str, str, str]]]] = []
    G = load_kg()
    if G is not None:
        try:
            dfp = pd.read_csv(prom, sep="\t")
            top = dfp[dfp["type"].str.lower() == "gene"]["name"].astype(str).tolist()
            top = top[: args.top_trace]
            for g in top:
                tr = shortest_path_to_pathway(G, nodes_df, g, max_hops=3)
                traces.append((g, tr))
        except Exception:
            pass

    out = write_first_win_report(
        args.disease, args.k, ci_json, disease_json, audit_md, traces, nodes_df
    )
    print(f"[first-win] Report -> {out}")


if __name__ == "__main__":
    main()
