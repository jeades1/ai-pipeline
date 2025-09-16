#!/usr/bin/env python3
"""
Uncertainty quantification for ranking models.
Implements bootstrap confidence intervals and calibration assessment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, List
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def bootstrap_ranking_confidence(
    assoc: pd.DataFrame,
    kg: Any,
    ranking_func: Any,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for gene rankings."""

    # Store bootstrap rankings
    bootstrap_rankings = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_sample = resample(assoc, random_state=i)

        try:
            # Get ranking for this bootstrap sample
            ranked = ranking_func(boot_sample, kg)

            # Store ranks
            ranks = {}
            for idx, row in ranked.iterrows():
                ranks[row["name"]] = idx + 1
            bootstrap_rankings.append(ranks)

        except Exception as e:
            print(f"[bootstrap] Sample {i} failed: {e}")
            continue

    if len(bootstrap_rankings) == 0:
        print("[bootstrap] No successful bootstrap samples")
        return pd.DataFrame()

    # Compute confidence intervals
    all_genes = set()
    for ranks in bootstrap_rankings:
        all_genes.update(ranks.keys())

    confidence_data = []
    alpha = 1 - confidence_level

    for gene in all_genes:
        gene_ranks = [ranks.get(gene, len(assoc)) for ranks in bootstrap_rankings]
        gene_ranks = [r for r in gene_ranks if r is not None]

        if len(gene_ranks) > 0:
            mean_rank = np.mean(gene_ranks)
            std_rank = np.std(gene_ranks)
            lower_ci = np.percentile(gene_ranks, 100 * alpha / 2)
            upper_ci = np.percentile(gene_ranks, 100 * (1 - alpha / 2))

            confidence_data.append(
                {
                    "gene": gene,
                    "mean_rank": mean_rank,
                    "std_rank": std_rank,
                    "ci_lower": lower_ci,
                    "ci_upper": upper_ci,
                    "rank_stability": 1.0
                    / (1.0 + std_rank / mean_rank),  # Higher = more stable
                }
            )

    return pd.DataFrame(confidence_data)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (uniform bins)."""
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= left) & (
            y_prob < right if i < n_bins - 1 else y_prob <= right
        )
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def quantile_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error using quantile (equal-frequency) bins."""
    if len(y_true) == 0:
        return 0.0
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, qs))
    if len(edges) < 3:
        return expected_calibration_error(y_true, y_prob, n_bins=min(5, n_bins))
    inds = np.digitize(y_prob, edges, right=True)
    total = len(y_true)
    ece = 0.0
    for b in range(1, len(edges) + 1):
        mask = inds == b
        if mask.sum() == 0:
            continue
        bucket_true = y_true[mask].mean()
        bucket_pred = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(bucket_true - bucket_pred)
    return float(ece)


def fit_platt_scaling(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    try:
        lr = LogisticRegression(class_weight="balanced", max_iter=500)
        lr.fit(scores.reshape(-1, 1), y_true)
        return lr.predict_proba(scores.reshape(-1, 1))[:, 1]
    except Exception:
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def fit_isotonic(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        return iso.fit_transform(scores, y_true)
    except Exception:
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def calibration_analysis(
    assoc: pd.DataFrame,
    kg: Any,
    ranking_func: Any,
    anchor_genes: List[str],
    out_dir: Path = Path("artifacts/uncertainty"),
    min_pos_rate: float = 0.005,
    max_top_n: int = 1000,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Perform calibration analysis with safeguards for extreme class imbalance.

    Strategy:
      1. Compute ranking and derive binary labels.
      2. If global positive rate < min_pos_rate, restrict to top-N (adaptive) to raise effective positive rate.
      3. Use quantile-based binning fallback if uniform bins yield < 2 non-empty bins.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = ranking_func(assoc, kg)
    anchor_set = set(g.upper() for g in anchor_genes)
    ranked["is_anchor"] = ranked["name"].str.upper().isin(anchor_set)

    y_true_full = ranked["is_anchor"].astype(int).values
    y_prob_full = ranked["assoc_score"].values

    # Normalize to [0,1]
    if y_prob_full.max() > 1.0 or y_prob_full.min() < 0.0:
        y_prob_full = (y_prob_full - y_prob_full.min()) / (
            y_prob_full.max() - y_prob_full.min() + 1e-12
        )

    global_pos_rate = float(y_true_full.mean()) if len(y_true_full) else 0.0

    # Adaptive subset if extremely sparse positives
    effective_df = ranked.copy()
    if global_pos_rate < min_pos_rate and len(anchor_set) > 0:
        # Strategy: Use stratified sampling to avoid clustering all anchors in one bin
        # Take a larger sample but maintain reasonable positive rate
        sample_size = min(max_top_n * 3, len(ranked))  # 3x larger sample
        effective_df = ranked.head(sample_size).copy()

        # Also ensure we have some anchors by including any missing ones
        missing_anchors = ranked[
            ranked["is_anchor"] & ~ranked.index.isin(effective_df.index)
        ]
        if len(missing_anchors) > 0:
            # Add missing anchors but limit to avoid extreme imbalance
            max_additional = min(10, len(missing_anchors))
            effective_df = pd.concat(
                [effective_df, missing_anchors.head(max_additional)]
            ).drop_duplicates()

        # Sort by score for consistent analysis
        effective_df = effective_df.sort_values(
            "assoc_score", ascending=False
        ).reset_index(drop=True)

        # Recompute arrays and ensure numpy conversion
        y_true = effective_df["is_anchor"].astype(int).values
        y_prob = effective_df["assoc_score"].values
        # Convert to numpy arrays explicitly
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        # Normalize again
        if y_prob.max() > 1.0 or y_prob.min() < 0.0:
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
        subset_flag = True
        n_anchors_in_subset = int(y_true.sum())
        print(
            f"[calibration] Using stratified subset with {len(effective_df)} genes, {n_anchors_in_subset} anchors"
        )
    else:
        y_true = y_true_full
        y_prob = y_prob_full
        subset_flag = False

    effective_pos_rate = float(y_true.mean()) if len(y_true) else 0.0

    # Raw calibration metrics
    ece = expected_calibration_error(y_true, y_prob)
    q_ece = quantile_ece(y_true, y_prob)
    try:
        brier = (
            float(brier_score_loss(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
    except Exception:
        brier = float("nan")
    try:
        nll = (
            float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9)))
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
    except Exception:
        nll = float("nan")

    # Primary: uniform bins with Laplace smoothing + kernel smooth curve for extreme imbalance
    fraction_of_positives: List[float] = []
    mean_predicted_value: List[float] = []

    try:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            left, right = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= left) & (
                y_prob < right if i < n_bins - 1 else y_prob <= right
            )
            if mask.sum() == 0:
                continue
            # Laplace smoothing to avoid 0/0 or all-zeros visual collapse
            pos = y_true[mask].sum()
            total = mask.sum()
            frac = (pos + 1) / (total + 2)  # Laplace correction
            fraction_of_positives.append(float(frac))
            mean_predicted_value.append(float(y_prob[mask].mean()))
        # Fallback to quantile if <2 populated bins
        if len(fraction_of_positives) < 2:
            raise ValueError("Too few populated bins after uniform binning")
    except Exception as e:
        print(f"[calibration] uniform binning fallback: {e}")
        try:
            quantiles = np.linspace(0, 1, n_bins + 1)
            q_edges = np.unique(np.quantile(y_prob, quantiles))
            inds = np.digitize(y_prob, q_edges, right=True)
            fop_q: List[float] = []
            mpv_q: List[float] = []
            for b in range(1, len(q_edges)):
                mask = inds == b
                if mask.sum() == 0:
                    continue
                pos = y_true[mask].sum()
                total = mask.sum()
                frac = (pos + 1) / (total + 2)
                fop_q.append(float(frac))
                mpv_q.append(float(y_prob[mask].mean()))
            fraction_of_positives = fop_q
            mean_predicted_value = mpv_q
        except Exception as e2:
            print(f"[calibration] quantile fallback failed: {e2}")

    # Kernel smoothing (Nadaraya-Watson) across probability axis for interpretability under extreme imbalance
    smooth_grid = np.linspace(0, 1, 50)
    sigma = 0.08
    ks_frac = []
    for g in smooth_grid:
        w = np.exp(-0.5 * ((y_prob - g) / sigma) ** 2)
        num = (w * y_true).sum() + 1  # Laplace
        den = w.sum() + 2
        ks_frac.append(float(num / den))

    nonzero_bins = int(sum(f > 0 for f in fraction_of_positives))

    # Advanced calibration (Platt & Isotonic)
    platt_prob = fit_platt_scaling(y_true, y_prob)
    iso_prob = fit_isotonic(y_true, y_prob)

    def metric_bundle(p: np.ndarray) -> Dict[str, float]:
        return {
            "ece_uniform": expected_calibration_error(y_true, p),
            "ece_quantile": quantile_ece(y_true, p),
            "brier": (
                float(brier_score_loss(y_true, p))
                if len(np.unique(y_true)) > 1
                else float("nan")
            ),
            "log_loss": (
                float(log_loss(y_true, np.clip(p, 1e-9, 1 - 1e-9)))
                if len(np.unique(y_true)) > 1
                else float("nan")
            ),
            "mean_conf": float(p.mean()),
            "std_conf": float(p.std()),
        }

    # Brier decomposition helper (quantile bins)
    def brier_decomposition(
        y: np.ndarray, p: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        if len(y) == 0:
            return {
                "reliability": float("nan"),
                "resolution": float("nan"),
                "uncertainty": float("nan"),
            }
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        if len(edges) < 3:
            return {
                "reliability": float("nan"),
                "resolution": float("nan"),
                "uncertainty": float("nan"),
            }
        inds = np.digitize(p, edges, right=True)
        p_bar = y.mean()
        reliability = 0.0
        resolution = 0.0
        for b in range(1, len(edges) + 1):
            m = inds == b
            if m.sum() == 0:
                continue
            o_k = y[m].mean()
            p_k = p[m].mean()
            weight = m.sum() / len(y)
            reliability += weight * (p_k - o_k) ** 2
            resolution += weight * (o_k - p_bar) ** 2
        uncertainty_term = p_bar * (1 - p_bar)
        return {
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty_term),
        }

    raw_brier_decomp = brier_decomposition(y_true, y_prob)

    raw_metrics = {
        "ece_uniform": ece,
        "ece_quantile": q_ece,
        "brier": brier,
        "log_loss": nll,
        "mean_conf": float(y_prob.mean()),
        "std_conf": float(y_prob.std()),
    }
    platt_metrics = metric_bundle(platt_prob)
    iso_metrics = metric_bundle(iso_prob)
    platt_brier_decomp = brier_decomposition(y_true, platt_prob)
    iso_brier_decomp = brier_decomposition(y_true, iso_prob)

    # Plot calibration
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        if fraction_of_positives and mean_predicted_value:
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                label=f"Raw Discrete (ECEq={q_ece:.3f})",
            )
        ax.plot(smooth_grid, ks_frac, "-", color="#E67E22", label="Raw Smoothed")

        # Quantile discrete curves for calibrated variants
        def discrete_curve(p: np.ndarray, label: str, color: str):
            qs = np.linspace(0, 1, 11)
            edges = np.unique(np.quantile(p, qs))
            if len(edges) < 3:
                return
            inds = np.digitize(p, edges, right=True)
            xs, ys = [], []
            for b in range(1, len(edges) + 1):
                m = inds == b
                if m.sum() == 0:
                    continue
                xs.append(float(p[m].mean()))
                pos = y_true[m].sum()
                ys.append(float((pos + 1) / (m.sum() + 2)))
            if xs:
                ax.plot(xs, ys, "o--", color=color, label=label)

        discrete_curve(
            platt_prob, f'Platt (ECEq={platt_metrics["ece_quantile"]:.3f})', "#2E86DE"
        )
        discrete_curve(
            iso_prob, f'Isotonic (ECEq={iso_metrics["ece_quantile"]:.3f})', "#27AE60"
        )
        ax.plot([0, 1], [0, 1], "k:", label="Ideal")
        note = "subset" if subset_flag else "full"
        ax.set_xlabel("Predicted Probability (raw & calibrated)")
        ax.set_ylabel("Fraction of Positives (Laplace)")
        ax.set_title(
            f'Calibration Comparison ({note})\nPos rate={effective_pos_rate:.4f} | Raw ECEq={q_ece:.3f} -> Platt {platt_metrics["ece_quantile"]:.3f}, Iso {iso_metrics["ece_quantile"]:.3f}'
        )
        ax.set_ylim(0, max(0.05, max(ks_frac) * 1.15))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ann_text = "Discrete: quantile bins (Laplace). Calibrated vs raw comparison under extreme imbalance."
        ax.text(
            0.5,
            -0.2,
            ann_text,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
        plt.tight_layout()
        plt.savefig(out_dir / "calibration_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(
            f"[calibration] Saved calibration plot to {out_dir / 'calibration_plot.png'}"
        )
    except Exception as e:
        print(f"[calibration] plot generation failed: {e}")

    # PR metrics (more stable under imbalance)
    try:
        ap = float(average_precision_score(y_true, y_prob))
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        # Save curve
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, label=f"PR (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (effective subset)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        ap = float("nan")
        print(f"[calibration] PR curve failed: {e}")

    metrics = {
        "raw": {**raw_metrics, "brier_decomposition": raw_brier_decomp},
        "platt": {**platt_metrics, "brier_decomposition": platt_brier_decomp},
        "isotonic": {**iso_metrics, "brier_decomposition": iso_brier_decomp},
        "n_calibration_points": len(fraction_of_positives),
        "global_positive_rate": global_pos_rate,
        "effective_positive_rate": effective_pos_rate,
        "used_subset": subset_flag,
        "adaptive_subset_size": (
            int(len(effective_df)) if subset_flag else int(len(ranked))
        ),
        "average_precision": ap,
    }
    # Persist advanced metrics
    try:
        import json as _json

        with open(out_dir / "calibration_metrics.json", "w") as f:
            _json.dump(metrics, f, indent=2)
    except Exception:
        pass
    return metrics


def ensemble_ranking_uncertainty(
    assoc: pd.DataFrame,
    kg: Any,
    ranking_functions: List[Any],
    weights: List[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensemble multiple ranking methods and quantify disagreement."""

    if weights is None:
        weights = [1.0] * len(ranking_functions)
    weights = np.array(weights) / np.sum(weights)

    # Get rankings from each method
    all_rankings = []
    for i, ranking_func in enumerate(ranking_functions):
        try:
            ranked = ranking_func(assoc, kg)
            ranked["method"] = f"method_{i}"
            all_rankings.append(ranked)
        except Exception as e:
            print(f"[ensemble] Method {i} failed: {e}")
            continue

    if len(all_rankings) == 0:
        print("[ensemble] No successful rankings")
        return pd.DataFrame(), pd.DataFrame()

    # Compute ensemble scores
    all_genes = set()
    for ranking in all_rankings:
        all_genes.update(ranking["name"].values)

    ensemble_data = []
    uncertainty_data = []

    for gene in all_genes:
        scores = []
        ranks = []

        for ranking in all_rankings:
            gene_row = ranking[ranking["name"] == gene]
            if not gene_row.empty:
                scores.append(float(gene_row["assoc_score"].iloc[0]))
                ranks.append(int(gene_row.index[0] + 1))
            else:
                scores.append(0.0)
                ranks.append(len(ranking) + 1)

        # Weighted ensemble score
        active_weights = (
            weights[: len(scores)] if weights is not None else [1.0] * len(scores)
        )
        ensemble_score = np.average(scores, weights=active_weights)

        # Uncertainty measures
        score_std = np.std(scores)
        rank_std = np.std(ranks)
        score_range = max(scores) - min(scores)

        ensemble_data.append(
            {
                "name": gene,
                "ensemble_score": ensemble_score,
                "individual_scores": scores,
                "layer": "transcriptomic",
                "type": "gene",
            }
        )

        uncertainty_data.append(
            {
                "gene": gene,
                "score_std": score_std,
                "rank_std": rank_std,
                "score_range": score_range,
                "consensus_strength": 1.0
                / (1.0 + score_std),  # Higher = more consensus
                "rank_stability": 1.0 / (1.0 + rank_std),
            }
        )

    ensemble_df = pd.DataFrame(ensemble_data)
    ensemble_df = ensemble_df.sort_values(
        "ensemble_score", ascending=False
    ).reset_index(drop=True)

    uncertainty_df = pd.DataFrame(uncertainty_data)

    return ensemble_df, uncertainty_df


def ranking_confidence_report(
    assoc: pd.DataFrame,
    kg: Any,
    ranking_func: Any,
    anchor_genes: List[str],
    out_dir: Path = Path("artifacts/uncertainty"),
) -> Dict[str, Any]:
    """Generate comprehensive uncertainty analysis report."""

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[uncertainty] Starting comprehensive analysis...")

    # 1. Bootstrap confidence intervals
    print("[uncertainty] Computing bootstrap confidence intervals...")
    try:
        bootstrap_ci = bootstrap_ranking_confidence(
            assoc, kg, ranking_func, n_bootstrap=50
        )
        bootstrap_ci.to_csv(out_dir / "bootstrap_confidence.csv", index=False)
        print(
            f"[uncertainty] Bootstrap CI saved to {out_dir / 'bootstrap_confidence.csv'}"
        )
    except Exception as e:
        print(f"[uncertainty] Bootstrap analysis failed: {e}")
        bootstrap_ci = pd.DataFrame()

    # 2. Calibration analysis
    print("[uncertainty] Performing calibration analysis...")
    try:
        calibration_metrics = calibration_analysis(
            assoc, kg, ranking_func, anchor_genes, out_dir
        )
        if isinstance(calibration_metrics, dict):
            raw_block = (
                calibration_metrics.get("raw", {})
                if isinstance(calibration_metrics.get("raw", {}), dict)
                else {}
            )
            raw_ece = raw_block.get("ece_quantile", float("nan"))
            try:
                print(f"[uncertainty] Raw quantile ECE: {raw_ece:.3f}")
            except Exception:
                print("[uncertainty] Raw quantile ECE unavailable")
    except Exception as e:
        print(f"[uncertainty] Calibration analysis failed: {e}")
        calibration_metrics = {}

    # 3. Generate summary report
    report = {
        "bootstrap_results": (
            bootstrap_ci.to_dict("records") if not bootstrap_ci.empty else []
        ),
        "calibration_metrics": calibration_metrics,
        "n_genes_analyzed": (
            len(assoc.get("feature", assoc.get("name", assoc.index)).unique())
            if len(assoc) > 0
            else 0
        ),
        "n_anchor_genes": len(anchor_genes),
    }

    # Save report
    import json

    with open(out_dir / "uncertainty_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(
        f"[uncertainty] Complete report saved to {out_dir / 'uncertainty_report.json'}"
    )

    return report


if __name__ == "__main__":
    # Example usage
    print("Uncertainty quantification module - run via demo integration")
