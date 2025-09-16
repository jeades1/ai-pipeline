#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality Precision@K plot from artifacts/bench/benchmark_report.json
Outputs:
  artifacts/bench/precision_plot.png  (PNG, 300 dpi)
  artifacts/bench/precision_plot.svg  (vector)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Deprecated: bench_plots has been removed in favor of conceptual and comparison visuals.
# This file intentionally left blank to avoid import errors if referenced.

INP = Path("artifacts/bench/benchmark_report.json")
OUTDIR = Path("artifacts/bench")
OUTDIR.mkdir(parents=True, exist_ok=True)

report = json.loads(INP.read_text())

# Accept either 'precision_at_k' or 'precision'
prec = report.get("precision_at_k") or report.get("precision") or {}
# Normalize keys like "p@5" -> int 5
pairs = []
for k, v in prec.items():
    try:
        kk = int(str(k).lower().replace("p@", "").strip())
        pairs.append((kk, float(v)))
    except Exception:
        continue
pairs = sorted(pairs, key=lambda x: x[0])

# Typography
plt.rcParams.update(
    {
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.size": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

fig, ax = plt.subplots()
if pairs:
    ks = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    ax.plot(ks, vals, marker="o", linewidth=2)
    for x, y in pairs:
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, max(0.05, min(1.0, max(vals) * 1.15)))
    ax.set_xlim(min(ks), max(ks))
    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K")
    ax.set_title("Benchmark precision@K (higher is better)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
else:
    ax.text(0.5, 0.5, "No precision values found", ha="center", va="center")
    ax.set_axis_off()

fig.tight_layout()
fig.savefig(OUTDIR / "kg_config.svg", format="svg", bbox_inches="tight")
fig.savefig(OUTDIR / "kg_config.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "precision_plot.png", bbox_inches="tight", dpi=300)
print("[bench_plots] Wrote", OUTDIR / "precision_plot.png")
