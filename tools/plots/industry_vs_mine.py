#!/usr/bin/env python3
"""
Industry target ranges vs. current run precision@k.

Reads:
    - artifacts/bench/cardiovascular_benchmark_report.json (point estimates across K)
    - artifacts/bench/benchmark_report.json (fallback CI if disease-specific CI cannot be derived)

Writes:
    - artifacts/pitch/industry_vs_mine.png
    - artifacts/pitch/industry_vs_mine.svg
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


def _load_ci_report(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_pak_series(p: Path) -> List[Tuple[int, float]]:
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []

    # Check if this is extended metrics format (CV-optimized)
    if "metrics" in data and isinstance(data["metrics"], dict):
        items = []
        for k_str, metrics in data["metrics"].items():
            if k_str.startswith("@") and "precision" in metrics:
                k = int(k_str[1:])  # Remove @ prefix
                precision = float(metrics["precision"])
                items.append((k, precision))
        return sorted(items, key=lambda x: x[0])

    # Check if this is CV-optimized format (single K value)
    if (
        "k" in data
        and "precision_at_k" in data
        and isinstance(data["precision_at_k"], dict)
        and "value" in data["precision_at_k"]
    ):
        k = int(data["k"])
        p_value = float(data["precision_at_k"]["value"])
        return [(k, p_value)]

    # Original format with precision_at_k series
    pak = data.get("precision_at_k", {})
    items = []
    if isinstance(pak, dict):
        for k, v in pak.items():
            m = re.search(r"(\d+)", str(k))
            if not m:
                continue
            items.append((int(m.group(1)), float(v)))
    return sorted(items, key=lambda x: x[0])


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion.

    Returns (lo, hi). If n==0, returns (0.0, 0.0).
    """
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + (z**2) / n
    center = p_hat + (z**2) / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return (lo, hi)


def plot_industry_vs_mine(out_png: Path, disease_json: Path, ci_json: Path) -> None:
    # Industry target bands - ESTIMATED ranges based on publicly available information
    # NOTE: These are approximate estimates extrapolated from limited disclosed metrics
    # and should be interpreted with caution. Actual industry performance may vary.
    # Sources: Public platform disclosures, computational biology literature (2020-2024)
    ks = [5, 10, 20, 50, 100]
    target_bands = {
        5: (0.20, 0.40),  # Estimated from literature reports
        10: (0.15, 0.30),  # Extrapolated from disclosed P@50 values
        20: (0.10, 0.25),  # Conservative estimates for drug discovery
        50: (0.06, 0.15),  # Range from public benchmarks (OpenTargets ~0.08)
        100: (0.03, 0.10),  # Typical precision decay patterns
    }

    mine = _load_pak_series(disease_json)

    # Prefer CI computed from disease-specific point estimates at K (Wilson), fallback to generic CI file
    preferred_k = 20
    ci_k: int = preferred_k
    ci_val: Optional[float] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None

    # Find the y at K (or nearest lower available K) from disease-specific series
    if mine:
        # Build a dict for quick lookup
        mine_dict = {k: v for k, v in mine}
        if preferred_k not in mine_dict:
            # choose the max K <= preferred_k if available
            eligible = [k for k in mine_dict.keys() if k <= preferred_k]
            if eligible:
                ci_k = max(eligible)
        # Use chosen K if present
        if ci_k in mine_dict:
            ci_val = float(mine_dict[ci_k])
            # Derive successes as round(p@K * K)
            successes = int(round(ci_val * ci_k))
            w_lo, w_hi = _wilson_ci(successes, ci_k)
            ci_lo, ci_hi = w_lo, w_hi

    # Fallback to generic CI JSON if we couldn't compute from disease-specific series
    if ci_lo is None or ci_hi is None or ci_val is None:
        ci = _load_ci_report(ci_json)
        ci_k = int(ci.get("k", preferred_k) or preferred_k)
        ci_val = float((ci.get("precision_at_k") or {}).get("value", 0.0))
        ci_ci95 = (ci.get("precision_at_k") or {}).get("ci95", [None, None])
        if isinstance(ci_ci95, list) and len(ci_ci95) == 2:
            ci_lo, ci_hi = ci_ci95[0], ci_ci95[1]

    # Prepare figure
    plt.figure(figsize=(8, 5))
    # Legend proxies for industry visuals
    band_patch = mpatches.Patch(
        color="#c8e6c9", alpha=0.5, label="Industry target range"
    )
    (midline_proxy,) = plt.plot(
        [], [], color="#81c784", linewidth=2, label="Industry midpoint"
    )
    for k in ks:
        lo, hi = target_bands.get(k, (None, None))
        if lo is None or hi is None:
            continue
        lo_f, hi_f = float(lo), float(hi)
        # Draw as a translucent band
        plt.fill_between(
            [k - 0.4, k + 0.4], [lo_f, lo_f], [hi_f, hi_f], color="#c8e6c9", alpha=0.5
        )
        # Midline
        mid = (lo_f + hi_f) / 2.0
        plt.plot([k - 0.4, k + 0.4], [mid, mid], color="#81c784", linewidth=2)

    # Plot our precision@k curve
    if mine:
        xs = [k for k, _ in mine]
        ys = [v for _, v in mine]
        plt.plot(
            xs,
            ys,
            marker="o",
            color="#2A7DE1",
            linewidth=2,
            label="This run (point estimates)",
        )
        for x, y in mine:
            plt.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=9)

    # Add CI at chosen K
    if ci_lo is not None and ci_hi is not None and ci_val is not None:
        plt.errorbar(
            [ci_k],
            [ci_val],
            yerr=[[max(0.0, ci_val - float(ci_lo))], [max(0.0, float(ci_hi) - ci_val)]],
            fmt="s",
            color="#e53935",
            ecolor="#e57373",
            elinewidth=2,
            capsize=5,
            label=f"This run CI@{ci_k}",
        )

    plt.xticks(ks, [f"P@{k}" for k in ks])

    # Dynamic y-axis scaling to accommodate data
    max_y = 0.5  # Default minimum
    if mine:
        max_y = max(max_y, max(v for _, v in mine))
    # Add padding for readability
    plt.ylim(0, max_y * 1.1)

    plt.xlabel("K")
    plt.ylabel("Precision")
    # Add context on benchmark size if available
    n_bench = None
    try:
        dj = json.loads(disease_json.read_text()) if disease_json.exists() else {}
        n_bench = dj.get("n_benchmark")
    except Exception:
        pass
    if n_bench is not None:
        plt.title(
            f"Estimated Industry Ranges vs. This Run\n(n={n_bench} CV benchmark genes)",
            fontsize=12,
            fontweight="bold",
        )
    else:
        plt.title(
            "Estimated Industry Ranges vs. This Run", fontsize=12, fontweight="bold"
        )

    # Add disclaimer
    plt.figtext(
        0.5,
        0.02,
        "Industry ranges are estimates based on public disclosures and literature reports",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    # Ensure industry proxies appear in legend along with plotted series
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [band_patch, midline_proxy] + handles
    labels = ["Estimated industry range", "Estimated industry midpoint"] + labels
    plt.legend(handles, labels)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.with_suffix(".svg"))
    plt.close()
    print(f"[plots] Wrote {out_png} and {out_png.with_suffix('.svg')}")


if __name__ == "__main__":
    # Use CV-optimized results if available, fallback to original
    cv_opt_extended = Path("artifacts/bench/metrics_extended_cv_opt.json")
    cv_opt_report = Path("artifacts/bench/benchmark_report_cv_opt.json")
    disease_report = Path("artifacts/bench/cardiovascular_benchmark_report.json")

    # Prefer extended metrics (has full K series), then single K, then original
    if cv_opt_extended.exists():
        disease_report = cv_opt_extended
    elif cv_opt_report.exists():
        disease_report = cv_opt_report

    plot_industry_vs_mine(
        Path("artifacts/pitch/industry_vs_mine.png"),
        disease_report,
        Path("artifacts/bench/benchmark_report.json"),
    )
