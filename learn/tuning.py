from __future__ import annotations
from typing import Any, Dict, Tuple, List
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Reuse internal helpers from rank_ltr
from .rank_ltr import _feature_table, _train_pointwise_ltr, _load_anchors


def _pointwise_dataset(
    assoc: pd.DataFrame, kg: Any
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    ft = _feature_table(assoc, kg)
    anchors = set(_load_anchors())
    ft["label"] = ft.index.to_series().apply(lambda g: 1 if g in anchors else 0)
    pos = ft[ft["label"] == 1]
    neg = ft[ft["label"] == 0]
    if len(pos) == 0:
        return ft, np.empty((0,)), np.empty((0,)), []
    n_neg = min(len(neg), max(200, 10 * len(pos)))
    neg_s = neg.sample(n=n_neg, random_state=42) if len(neg) > n_neg else neg
    train = pd.concat([pos, neg_s], axis=0)
    feature_cols = [
        "eff",
        "neglogp",
        "deg_in",
        "deg_out",
        "pr",
        "cpdb_deg",
        "tf_out",
        "disease_dist",
        "injury_pathways",
        "repair_pathways",
        "inflammation_pathways",
        "transport_pathways",
        "literature_score",
        "tissue_specificity",
        "network_importance",
        "pathway_relevance",
        "disease_relevance",
        "functional_score",
    ]
    X = train[feature_cols].values
    y = train["label"].values.astype(int)
    return train, X, y, feature_cols


def tune_ltr_params(
    assoc: pd.DataFrame,
    kg: Any,
    method: str = "rf",
    outdir: Path | str | None = None,
    cv_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Lightweight grid search over a small RF/XGB space using CV ROC AUC.

    Returns best params dict. Writes a small JSON report if outdir is provided.
    """
    train, X, y, _ = _pointwise_dataset(assoc, kg)
    if X.size == 0 or len(np.unique(y)) < 2:
        return {}

    grids: List[Dict[str, Any]]
    if method == "xgb":
        grids = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in (100, 200)
            for d in (4, 6, 8)
            for lr in (0.05, 0.1)
        ]
    else:
        # rf default
        grids = [
            {"n_estimators": n, "max_depth": d, "class_weight": "balanced"}
            for n in (100, 200, 400)
            for d in (8, 12, None)
        ]

    best: Dict[str, Any] = {}
    best_score = -1.0
    skf = StratifiedKFold(
        n_splits=max(
            2, min(cv_splits, len(np.unique(y)) if len(y) > cv_splits else cv_splits)
        ),
        shuffle=True,
        random_state=random_state,
    )
    for p in grids:
        try:
            model = _train_pointwise_ltr(
                X, y, method=("xgb" if method == "xgb" else "rf"), params=p
            )
            # Manual CV because model is refit anyway for scoring
            scores: List[float] = []
            for tr, va in skf.split(X, y):
                m = _train_pointwise_ltr(
                    X[tr], y[tr], method=("xgb" if method == "xgb" else "rf"), params=p
                )
                if hasattr(m, "predict_proba"):
                    pr = m.predict_proba(X[va])[:, 1]
                else:
                    pr = m.predict(X[va])
                auc = roc_auc_score(y[va], pr)
                scores.append(float(auc))
            mean_auc = float(np.mean(scores)) if scores else -1.0
            if mean_auc > best_score:
                best_score = mean_auc
                best = dict(p)
        except Exception:
            continue

    if outdir:
        outp = Path(outdir) / "ltr_tuning_report.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps(
                {"method": method, "best_params": best, "best_cv_auc": best_score},
                indent=2,
            )
        )

    return best
