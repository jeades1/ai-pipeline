from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def recall_panel(out_png: Path) -> None:
    real_path = Path("artifacts/bench/real_benchmark_report.json")
    legacy_path = Path("artifacts/bench/benchmark_report.json")
    report = {}
    if real_path.exists():
        try:
            report = json.loads(real_path.read_text())
        except Exception:
            pass
    if (not report) and legacy_path.exists():
        try:
            report = json.loads(legacy_path.read_text())
        except Exception:
            pass
    recall_map = report.get("recall_at_k", {})
    ks = [5, 10, 20, 50, 100, 200, 500]
    current = [float(recall_map.get(f"r@{k}", 0.0)) for k in ks]
    target_curve = {
        5: 0.05,
        10: 0.08,
        20: 0.10,
        50: 0.15,
        100: 0.20,
        200: 0.30,
        500: 0.45,
    }
    targets = [target_curve[k] for k in ks]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ks))
    width = 0.38
    ax.bar(
        x - width / 2,
        current,
        width,
        label="Current (Real)",
        color="#8E44AD",
        alpha=0.8,
    )
    ax.bar(x + width / 2, targets, width, label="Target", color="#27AE60", alpha=0.7)
    for i, (c, t) in enumerate(zip(current, targets)):
        ax.text(
            i - width / 2,
            c + 0.005,
            f"{c:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            i + width / 2, t + 0.005, f"{t:.2f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"R@{k}" for k in ks])
    ax.set_ylabel("Recall")
    ax.set_title("Recall@K: Current vs Target")
    ax.set_ylim(0, max(0.5, max(targets + current) * 1.25))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[recall:panel] Wrote {out_png}")


if __name__ == "__main__":
    recall_panel(Path("artifacts/pitch/recall_panel.png"))
