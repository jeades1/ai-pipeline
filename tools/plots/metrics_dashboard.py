#!/usr/bin/env python3
from __future__ import annotations

"""
Render a compact metrics dashboard and a Markdown table from artifacts/bench/metrics_extended.json.

Outputs:
  - artifacts/pitch/metrics_dashboard.png (and .svg)
  - artifacts/bench/metrics_table.md
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd


EXT_JSON = Path("artifacts/bench/metrics_extended.json")


def load_ext_metrics(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise SystemExit(f"[metrics-dashboard] Missing {p}")
    return json.loads(p.read_text())


def as_frame(d: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    k_list = d.get("k_list", [])
    md = d.get("metrics", {})
    for k in k_list:
        m = md.get(f"@{k}", {})
        row = {
            "K": int(k),
            "precision": float(m.get("precision", 0.0)),
            "recall": float(m.get("recall", 0.0)),
            "f1": float(m.get("f1", 0.0)),
            "hits": int(m.get("hits", 0)),
            "AP": float(m.get("AP", 0.0)),
            "nDCG": float(m.get("nDCG", 0.0)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("K")


def write_md_table(df: pd.DataFrame, d: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# Extended metrics\n")
    lines.append(f"Total anchors: {d.get('total_relevant', 0)}\n")
    lines.append("| K | Precision | Recall | F1 | Hits | AP | nDCG |\n")
    lines.append("|---:|---------:|------:|---:|----:|---:|----:|\n")
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['K'])} | {r['precision']:.2f} | {r['recall']:.4f} | {r['f1']:.4f} | {int(r['hits'])} | {r['AP']:.3f} | {r['nDCG']:.3f} |"
        )
    g = d.get("global", {})
    lines.append("\n## Global\n")
    lines.append(f"- MRR: {float(g.get('MRR', 0.0)):.3f}")
    lines.append(f"- MAP@maxK: {float(g.get('MAP@maxK', 0.0)):.3f}")
    lines.append(f"- Coverage: {float(g.get('coverage', 0.0)):.4f}\n")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[metrics-dashboard] Wrote table -> {out_md}")


def plot_dashboard(df: pd.DataFrame, d: Dict[str, Any], out_png: Path) -> None:
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # Lines over K
    ax1.plot(df["K"], df["precision"], marker="o", color="#2A7DE1")
    ax1.set_title("Precision@K")
    ax1.set_ylim(0, max(0.5, df["precision"].max() + 0.05))
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax2.plot(df["K"], df["recall"], marker="o", color="#8E44AD")
    ax2.set_title("Recall@K")
    ax2.set_ylim(0, max(0.2, df["recall"].max() + 0.01))
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax3.plot(df["K"], df["f1"], marker="o", color="#16A085")
    ax3.set_title("F1@K")
    ax3.set_ylim(0, max(0.3, df["f1"].max() + 0.02))
    ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Global bars
    g = d.get("global", {})
    labels = ["MRR", "MAP@maxK", "Coverage"]
    vals = [
        float(g.get("MRR", 0.0)),
        float(g.get("MAP@maxK", 0.0)),
        float(g.get("coverage", 0.0)),
    ]
    ax4.bar(labels, vals, color=["#F39C12", "#E74C3C", "#27AE60"])
    ax4.set_ylim(0, max(1.0 if max(vals) > 0.8 else 0.6, max(vals) + 0.05))
    for i, v in enumerate(vals):
        ax4.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax4.set_title("Global metrics")

    plt.suptitle("Extended performance dashboard")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.with_suffix(".svg"))
    plt.close()
    print(f"[metrics-dashboard] Wrote {out_png} and {out_png.with_suffix('.svg')}")


def main():
    data = load_ext_metrics(EXT_JSON)
    df = as_frame(data)
    plot_dashboard(df, data, Path("artifacts/pitch/metrics_dashboard.png"))
    write_md_table(df, data, Path("artifacts/bench/metrics_table.md"))


if __name__ == "__main__":
    main()
