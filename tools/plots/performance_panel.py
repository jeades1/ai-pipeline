from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def performance_panel(out_png: Path) -> None:
    """Render unified performance panel using REAL benchmark if available.

    Priority order for benchmarks:
      1. artifacts/bench/real_benchmark_report.json  (true current performance)
      2. artifacts/bench/benchmark_report.json       (legacy/synthetic)
    We show both current (real) and target aspirational precision@K if real exists.
    """

    real_path = Path("artifacts/bench/real_benchmark_report.json")
    legacy_path = Path("artifacts/bench/benchmark_report.json")
    report = {}
    used_real = False
    if real_path.exists():
        try:
            report = json.loads(real_path.read_text())
            used_real = True
        except Exception:
            pass
    if (not report) and legacy_path.exists():
        report = json.loads(legacy_path.read_text())

    pak = report.get("precision_at_k", {})
    ks = [5, 10, 20, 50, 100]
    current_vals = [float(pak.get(f"p@{k}", 0.0)) for k in ks]
    # Define simple aspirational targets (do not overwrite real)
    target_map = {5: 0.2, 10: 0.2, 20: 0.15, 50: 0.1, 100: 0.06}
    target_vals = [target_map[k] for k in ks]
    # Calibration (if available) - use the fixed calibration results with prioritized order
    calib_png = None
    calib_paths = [
        Path("artifacts/uncertainty_final/calibration_plot.png"),  # Latest fix
        Path("artifacts/uncertainty_fix_test/calibration_plot.png"),  # Test fix
        Path("artifacts/adv_run_calibfix/uncertainty/calibration_plot.png"),
        Path("artifacts/adv_run_enhanced_final/uncertainty/calibration_plot.png"),
        Path("artifacts/adv_run_ltr_final/uncertainty/calibration_plot.png"),
        Path("artifacts/adv_run_uncal_final/uncertainty/calibration_plot.png"),
        Path("artifacts/adv_run_tuned/uncertainty/calibration_plot.png"),
        Path("artifacts/adv_run_mono/uncertainty/calibration_plot.png"),
    ]
    for path in calib_paths:
        if path.exists():
            calib_png = path
            break
    has_calib = calib_png is not None

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    x = np.arange(len(ks))
    width = 0.38 if used_real else 0.6
    if used_real:
        bars_real = ax1.bar(
            x - width / 2,
            current_vals,
            width,
            color="#E74C3C",
            alpha=0.8,
            label="Current (Real)",
        )
        bars_target = ax1.bar(
            x + width / 2,
            target_vals,
            width,
            color="#2ECC71",
            alpha=0.7,
            label="Target",
        )
        for i, (cv, tv) in enumerate(zip(current_vals, target_vals)):
            ax1.text(
                i - width / 2,
                cv + 0.003,
                f"{cv:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            ax1.text(
                i + width / 2,
                tv + 0.003,
                f"{tv:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax1.legend(fontsize=9)
        ax1.set_title("Precision@K (Real vs Target)", fontweight="bold")
        ymax = max(target_vals + current_vals) * 1.25 + 0.01
    else:
        bars = ax1.bar(x, current_vals, width, color="#2E86DE", alpha=0.85)
        for i, v in enumerate(current_vals):
            ax1.text(
                i,
                v + 0.004,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax1.set_title("Precision@K (Benchmark)", fontweight="bold")
        ymax = max(0.2, max(current_vals) * 1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P@{k}" for k in ks])
    ax1.set_ylim(0, max(0.12, ymax))
    ax1.set_ylabel("Precision", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    if has_calib:
        img = plt.imread(str(calib_png))
        ax2.imshow(img)
        ax2.axis("off")
        ax2.set_title("Calibration (ECE) — Latest Run", fontweight="bold")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Calibration plot not found", ha="center", va="center")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[perf:panel] Wrote {out_png}")


if __name__ == "__main__":
    performance_panel(Path("artifacts/pitch/performance_panel.png"))
