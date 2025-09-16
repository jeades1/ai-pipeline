#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report", required=True, help="artifacts/bench/benchmark_report.json"
    )
    ap.add_argument("--out", required=True, help="output directory for plots")
    args = ap.parse_args()

    report_path = Path(args.report)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(report_path.read_text())
    bars = {
        "True Positives": data["true_positives"],
        "False Negatives": data["false_negatives"],
        "False Positives": data["false_positives"],
    }

    # Bar chart
    plt.figure()
    plt.bar(list(bars.keys()), list(bars.values()))
    plt.title("Benchmark Counts")
    plt.ylabel("count")
    fig1 = outdir / "benchmark_counts.png"
    plt.savefig(fig1, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[bench-viz] Wrote {fig1}")

    # Metrics text file
    fig2 = outdir / "metrics.txt"
    with fig2.open("w") as fh:
        fh.write(f"precision={data['precision']:.3f}\n")
        fh.write(f"recall={data['recall']:.3f}\n")
        fh.write(f"f1={data['f1']:.3f}\n")
    print(f"[bench-viz] Wrote {fig2}")


if __name__ == "__main__":
    main()
