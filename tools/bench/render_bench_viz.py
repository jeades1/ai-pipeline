#!/usr/bin/env python3
import sys
import json
import pathlib as p

if len(sys.argv) < 2:
    print("Usage: render_bench_viz.py artifacts/bench/benchmark_report.json")
    sys.exit(2)

report_path = p.Path(sys.argv[1])
outdir = report_path.parent

with report_path.open() as fh:
    j = json.load(fh)

tp = j.get("true_positives", [])
fp = j.get("false_positives", [])
fn = j.get("false_negatives", [])
p_at_k = j.get("precision_at_k", {}) or {}
recall = j.get("recall")
f1 = j.get("f1")

md = outdir / "benchmark_quicklook.md"
lines = [
    "# Benchmark quicklook",
    "",
    f"- true positives: {len(tp)}",
    f"- false positives: {len(fp)}",
    f"- false negatives: {len(fn)}",
    f"- precision@10: {p_at_k.get('10')}",
    f"- precision@50: {p_at_k.get('50')}",
    f"- recall: {recall}",
    f"- f1: {f1}",
    "",
    "## Top matched biomarkers",
    "| name | layer | type |",
    "|---|---|---|",
]
for r in tp[:20]:
    name = r.get("name") or r.get("gene") or ""
    layer = r.get("layer", "")
    typ = r.get("type", "")
    lines.append(f"| {name} | {layer} | {typ} |")

md.write_text("\n".join(lines))
print(f"[bench-viz] Wrote {md}")
