from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

from .rank_ltr import _feature_table, _load_anchors


def ltr_feature_importance(
    assoc: pd.DataFrame, kg: Any, out_dir: Path = Path("artifacts/diagnostics")
) -> pd.DataFrame:
    """Train a lightweight RF on LTR features and compute permutation importances.

    Returns a DataFrame [feature, importance] sorted descending and writes to artifacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = _feature_table(assoc, kg)
    anchors = set(_load_anchors())
    # Convert to numpy array explicitly
    y = (
        ft.index.to_series()
        .apply(lambda g: 1 if g in anchors else 0)
        .astype(int)
        .to_numpy()
    )

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
    X = ft[feature_cols].to_numpy()

    # Simple stratified split (if positives are present)
    if int(y.sum()) == 0:
        df = pd.DataFrame(
            {"feature": feature_cols, "importance": [0.0] * len(feature_cols)}
        )
        df.to_csv(out_dir / "ltr_feature_importance.csv", index=False)
        return df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    try:
        base_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    except Exception:
        base_auc = float("nan")

    pi = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
    importances_mean = getattr(pi, "importances_mean", None)
    if importances_mean is None:
        # Fallback if structure differs
        if isinstance(pi, dict) and "importances_mean" in pi:
            importances_mean = pi["importances_mean"]
        else:
            importances_mean = np.zeros(len(feature_cols))
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances_mean})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(out_dir / "ltr_feature_importance.csv", index=False)

    # Add base metric
    (out_dir / "ltr_feature_importance.meta").write_text(f"base_auc={base_auc}\n")
    return imp_df


def ltr_feature_group_ablation(
    assoc: pd.DataFrame, kg: Any, out_dir: Path = Path("artifacts/diagnostics")
) -> pd.DataFrame:
    """Run quick ablations by feature group and estimate validation AUC deltas.
    Uses anchors as positives; AUC is a proxy for ranking quality.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = _feature_table(assoc, kg)
    anchors = set(_load_anchors())
    y = (
        ft.index.to_series()
        .apply(lambda g: 1 if g in anchors else 0)
        .astype(int)
        .to_numpy()
    )
    if int(y.sum()) == 0:
        df = pd.DataFrame()
        df.to_csv(out_dir / "ltr_feature_group_ablation.csv", index=False)
        return df

    groups: Dict[str, List[str]] = {
        "stat": ["eff", "neglogp"],
        "network": [
            "deg_in",
            "deg_out",
            "pr",
            "cpdb_deg",
            "tf_out",
            "network_importance",
        ],
        "pathway": [
            "injury_pathways",
            "repair_pathways",
            "inflammation_pathways",
            "transport_pathways",
            "pathway_relevance",
        ],
        "disease": ["disease_dist", "literature_score", "disease_relevance"],
        "tissue": ["tissue_specificity", "functional_score"],
    }

    base_cols = sorted({c for cols in groups.values() for c in cols})
    X_full = ft[base_cols].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    ).fit(X_tr, y_tr)
    try:
        base_auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    except Exception:
        base_auc = float("nan")

    rows = []
    for gname, gcols in groups.items():
        keep = [c for c in base_cols if c not in set(gcols)]
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
            ft[keep].to_numpy(), y, test_size=0.3, random_state=42, stratify=y
        )
        clf_g = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ).fit(X_tr2, y_tr2)
        try:
            auc_g = roc_auc_score(y_te2, clf_g.predict_proba(X_te2)[:, 1])
        except Exception:
            auc_g = float("nan")
        rows.append(
            {
                "group": gname,
                "auc": auc_g,
                "delta_vs_base": (
                    (auc_g - base_auc)
                    if (not np.isnan(auc_g) and not np.isnan(base_auc))
                    else float("nan")
                ),
            }
        )

    ab = pd.DataFrame(rows).sort_values("delta_vs_base").reset_index(drop=True)
    ab.to_csv(out_dir / "ltr_feature_group_ablation.csv", index=False)
    (out_dir / "ltr_feature_group_ablation.meta").write_text(f"base_auc={base_auc}\n")
    return ab
