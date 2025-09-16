from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd


def _load_edges(nodes_tsv: Path, edges_tsv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_tsv, sep="\t")
    edges = pd.read_csv(edges_tsv, sep="\t")
    return nodes, edges


def _is_promoted(row: pd.Series) -> bool:
    # If a promoted table is provided, caller can pre-join; otherwise treat 'promoted' via causal_support column when present
    cs = row.get("causal_support")
    if isinstance(cs, str):
        return "promoted" in cs
    if isinstance(cs, dict):
        return cs.get("level") == "promoted"
    return False


def summarize_gaps(
    ranked: pd.DataFrame,
    edges_tsv: Path,
    out_md: Path,
    top_n: int = 50,
) -> Path:
    edges = pd.read_csv(edges_tsv, sep="\t")
    assoc = edges[edges["predicate"] == "associative"]
    causal = edges[
        edges["predicate"].isin(["causes", "inhibits", "activates", "prevents"])
    ]

    # Focus on top candidates
    top = ranked.head(top_n).copy()
    if "causal_support" in top.columns:
        top["is_promoted"] = top.apply(_is_promoted, axis=1)
    else:
        top["is_promoted"] = False

    # Which top candidates lack any causal outgoing edges?
    have_causal = set(causal["s"].astype(str).unique())
    top["has_causal_edge"] = top["name"].astype(str).isin(have_causal)
    top["gap"] = ~top["has_causal_edge"]

    # Summaries
    n = len(top)
    n_gaps = int(top["gap"].sum())
    gap_examples = top[top["gap"]].head(10)[
        ["name", "assoc_score", "effect_size", "p_value"]
    ]

    lines = [
        "# Gap Analysis — Associative vs Causal Coverage",
        "",
        f"- Top-N examined: {n}",
        f"- With no causal edges: {n_gaps} ({(n_gaps/n if n else 0):.1%})",
        "",
        "## Examples lacking causal coverage",
        "",
    ]
    for _, r in gap_examples.iterrows():
        lines.append(
            f"- {r['name']}: assoc={r['assoc_score']:.3f}, eff={r['effect_size']:.3f}, p={r['p_value']:.2e}"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    return out_md


def _load_benchmarks(p: Path) -> List[str]:
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    out = []
    if isinstance(data, dict) and "biomarkers" in data:
        seq = data["biomarkers"]
    else:
        seq = data
    for item in seq:
        if isinstance(item, dict):
            n = item.get("name") or item.get("gene")
            if n:
                out.append(str(n).upper())
        else:
            out.append(str(item).upper())
    return out


def compare_to_benchmark(
    ranked: List[Dict[str, Any]],
    out_path: Path,
    bench_json: Path = Path("benchmarks/markers.json"),
) -> Dict[str, Any]:
    anchors = _load_benchmarks(bench_json)
    name_to_rank = {str(c["name"]).upper(): i + 1 for i, c in enumerate(ranked)}
    hits = {m: name_to_rank.get(m) for m in anchors}
    recall_at_20 = sum(1 for r in hits.values() if r is not None and r <= 20) / max(
        1, len(anchors)
    )

    lines = [
        "# Benchmark Comparison",
        "",
        f"Anchors ({len(anchors)}): {', '.join(anchors)}",
        "",
        "## Ranks in results",
    ]
    for m in anchors:
        r = hits[m]
        lines.append(f"- {m}: {'not found' if r is None else f'rank {r}'}")
    lines += [
        "",
        f"**Recall@20:** {recall_at_20:.2f}",
        "",
        "_Note: curated benchmarks may include literature-derived anchors._",
    ]
    out_path.write_text("\n".join(lines))
    return {"recall_at_20": recall_at_20, "ranks": hits}
