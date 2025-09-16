# reports/compare_to_benchmark.py
import json
from pathlib import Path
from typing import List, Dict, Any


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
    denom = max(1, len(anchors))
    recall_at_20 = sum(1 for r in hits.values() if r is not None and r <= 20) / denom

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
