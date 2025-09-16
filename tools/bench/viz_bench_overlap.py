#!/usr/bin/env python3
import argparse
import json
import pathlib
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="path to benchmark_report.json")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(pathlib.Path(args.report).read_text())
    s = data["summary"]
    bars = {
        "hits": s["hits"],
        "misses": s["misses"],
        "gaps": s["gaps"],
    }

    # simple bar chart
    plt.figure()
    plt.bar(list(bars.keys()), list(bars.values()))
    plt.title("Benchmark Comparison")
    plt.ylabel("Count")
    fig_path = outdir / "benchmark_bar.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    # small markdown badge
    md = []
    md.append("# Benchmark Viz\n")
    md.append(f"- Hits: **{s['hits']}**")
    md.append(f"- Misses: **{s['misses']}**")
    md.append(f"- Gaps: **{s['gaps']}**")
    md.append(f"- Hit rate: **{s['hit_rate']:.3f}**")
    md.append(f"\n![Benchmark Bar]({fig_path.name})\n")
    (outdir / "benchmark_viz.md").write_text("\n".join(md))

    print(f"[bench] Wrote {fig_path} and {outdir/'benchmark_viz.md'}")


if __name__ == "__main__":
    main()
