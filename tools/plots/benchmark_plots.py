"""
Quick plots for benchmark outputs.

Reads artifacts/bench/benchmark_report.json (if present) and writes a simple
Precision@K bar/line plot to artifacts/pitch/precision_at_k.png.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def precision_at_k_plot(report_json: Path, out_png: Path) -> None:
    report: Dict[str, Any] = json.loads(report_json.read_text())
    # Expect a structure like {"precision_at_k": {"5": 0.6, "10": 0.55, ...}}
    pak = report.get("precision_at_k") or report.get("precision@k") or {}
    if not pak:
        print(f"[bench:plot] No precision@k found in {report_json}")
        return
    # Support both dict (e.g., {"5": 0.6, "p@10": 0.55}) and list of {k, precision}
    items = []
    import re

    if isinstance(pak, dict):
        for k, v in pak.items():
            m = re.search(r"(\d+)", str(k))
            if not m:
                continue
            items.append((int(m.group(1)), float(v)))
    elif isinstance(pak, list):
        for row in pak:
            if isinstance(row, dict):
                k = row.get("k") or row.get("K") or row.get("at")
                m = re.search(r"(\d+)", str(k)) if k is not None else None
                p = row.get("precision") or row.get("p") or row.get("value")
                if m and p is not None:
                    items.append((int(m.group(1)), float(p)))
    items = sorted(items, key=lambda x: x[0])
    if not items:
        print(f"[bench:plot] Could not parse precision@k entries in {report_json}")
        return
    ks = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker="o", color="#2A7DE1", label="Precision@K")
    plt.xticks(ks)
    plt.ylim(0, 1)
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.title("Benchmark Precision@K")
    for x, y in zip(ks, vals):
        plt.text(
            x, min(0.98, y + 0.03), f"{y:.2f}", ha="center", va="bottom", fontsize=8
        )
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[bench:plot] Wrote {out_png}")


if __name__ == "__main__":
    bench_json = Path("artifacts/bench/benchmark_report.json")
    out = Path("artifacts/pitch/precision_at_k.png")
    if bench_json.exists():
        precision_at_k_plot(bench_json, out)
    else:
        print("[bench:plot] benchmark_report.json not found; skip")
