from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb  # type: ignore[import]

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .features_kg import gene_kg_features_cached


def _sanitize_rf_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    allowed: Dict[str, Any] = {}
    if not params:
        return allowed

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return None

    for k, v in params.items():
        if k == "n_estimators":
            iv = _as_int(v)
            if iv is not None:
                allowed[k] = iv
        elif k == "max_depth":
            iv = _as_int(v)
            if iv is not None:
                allowed[k] = iv
        elif k == "max_features":
            if isinstance(v, str) and v in {"sqrt", "log2"}:
                allowed[k] = v
            else:
                fv = _as_float(v)
                if fv is not None:
                    allowed[k] = fv
        elif (
            k == "class_weight"
            and isinstance(v, str)
            and v in {"balanced", "balanced_subsample"}
        ):
            allowed[k] = v
        elif k == "random_state":
            iv = _as_int(v)
            if iv is not None:
                allowed[k] = iv
    return allowed


def _sanitize_xgb_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    allowed: Dict[str, Any] = {}
    if not params:
        return allowed

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return None

    for k, v in params.items():
        if k in {"n_estimators", "max_depth"}:
            iv = _as_int(v)
            if iv is not None:
                allowed[k] = iv
        elif k in {"learning_rate", "subsample", "colsample_bytree"}:
            fv = _as_float(v)
            if fv is not None:
                allowed[k] = fv
        elif k == "random_state":
            iv = _as_int(v)
            if iv is not None:
                allowed[k] = iv
    return allowed


def _load_anchors(bench_json: Path = Path("benchmarks/aki_markers.json")) -> List[str]:
    if not bench_json.exists():
        return []
    data = json.loads(bench_json.read_text())
    if isinstance(data, dict):
        if "biomarkers" in data:
            items = data["biomarkers"]
        else:
            items = data.get("markers", [])
        out: List[str] = []
        for it in items:
            if isinstance(it, dict):
                n = it.get("name") or it.get("gene")
                if n:
                    out.append(str(n).upper())
            else:
                out.append(str(it).upper())
        return out
    if isinstance(data, list):
        out = []
        for it in data:
            if isinstance(it, dict):
                n = it.get("name") or it.get("gene")
                if n:
                    out.append(str(n).upper())
            else:
                out.append(str(it).upper())
        return out
    return []


def _feature_table(
    assoc: pd.DataFrame, kg: Any, use_cache: bool = True
) -> pd.DataFrame:
    assoc = assoc.copy()
    assoc["gene"] = assoc["feature"].astype(str).str.upper()
    assoc["neglogp"] = -np.log10(
        np.clip(
            pd.to_numeric(assoc["p_value"], errors="coerce").fillna(1.0), 1e-300, 1.0
        )
    )
    assoc["eff"] = pd.to_numeric(assoc["effect_size"], errors="coerce").fillna(0.0)
    # Aggregate by gene (max statistical signals across datasets)
    agg = assoc.groupby("gene").agg({"neglogp": "max", "eff": "max"}).reset_index()
    kgf = gene_kg_features_cached(kg, assoc, use_cache=use_cache)
    ft = agg.set_index("gene").join(kgf, how="left").fillna(0.0)

    # Add enhanced composite features
    ft["network_importance"] = ft["deg_in"] + ft["deg_out"] + 10 * ft["pr"]
    ft["pathway_relevance"] = (
        ft["injury_pathways"]
        + ft["inflammation_pathways"]
        + ft["repair_pathways"]
        + ft["transport_pathways"]
    )
    ft["disease_relevance"] = ft["disease_dist"] + 0.1 * ft["literature_score"]
    ft["functional_score"] = ft["tissue_specificity"] + 0.1 * ft["cpdb_deg"]

    return ft


def _create_pairwise_training_data(
    ft: pd.DataFrame, anchors: set
) -> Tuple[np.ndarray, np.ndarray]:
    """Create pairwise training data for ranking."""
    positives = ft[ft.index.isin(anchors)]
    negatives = ft[~ft.index.isin(anchors)]

    if len(positives) == 0 or len(negatives) == 0:
        return np.array([]), np.array([])

    # Enhanced feature set
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

    pairs_X = []
    pairs_y = []

    # Sample balanced pairs
    max_pairs = min(len(positives) * len(negatives), 5000)
    n_neg_per_pos = min(len(negatives), max_pairs // len(positives))

    for pos_idx in positives.index:
        neg_sample = negatives.sample(n=n_neg_per_pos, random_state=42)
        for neg_idx in neg_sample.index:
            # Positive - Negative feature difference
            diff = (
                positives.loc[pos_idx, feature_cols].values
                - negatives.loc[neg_idx, feature_cols].values
            )
            pairs_X.append(diff)
            pairs_y.append(1)  # Positive should rank higher

            # Negative - Positive (reverse pair)
            pairs_X.append(-diff)
            pairs_y.append(0)

    return np.array(pairs_X), np.array(pairs_y)


def _train_pointwise_ltr(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "rf",
    params: Dict[str, Any] | None = None,
) -> Any:
    """Train pointwise ranking model."""
    if method == "xgb" and HAS_XGB:
        base: Dict[str, Any] = dict(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
        base.update(_sanitize_xgb_params(params))
        clf = xgb.XGBClassifier(**base)  # type: ignore[arg-type]
    elif method == "rf":
        base: Dict[str, Any] = dict(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        base.update(_sanitize_rf_params(params))
        clf = RandomForestClassifier(**base)  # type: ignore[arg-type]
    else:
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )

    clf.fit(X, y)
    return clf


def _train_pairwise_ltr(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any] | None = None,
    monotone: List[int] | None = None,
) -> Any:
    """Train pairwise ranking model."""
    if HAS_XGB:
        base: Dict[str, Any] = dict(
            tree_method="hist",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
        base.update(_sanitize_xgb_params(params))
        # Use pairwise ranking objective if available
        try:
            base["objective"] = "rank:pairwise"
        except Exception:
            base["objective"] = "binary:logistic"
        if monotone is not None:
            base["monotone_constraints"] = (
                "(" + ",".join(str(int(m)) for m in monotone) + ")"
            )
        model = xgb.XGBClassifier(**base)  # type: ignore[arg-type]
    else:
        base: Dict[str, Any] = dict(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        base.update(_sanitize_rf_params(params))
    model = RandomForestClassifier(**base)  # type: ignore[arg-type]

    model.fit(X, y)
    return model


def rank_with_ltr(
    assoc: pd.DataFrame,
    kg: Any,
    method: str = "pairwise",
    model_params: Dict[str, Any] | None = None,
    use_cache: bool = True,
    monotone_constraints: List[int] | None = None,
) -> pd.DataFrame:
    """Enhanced learning-to-rank with pairwise training and rich features.

    Args:
        method: "pairwise", "pointwise", or "auto"
    """
    ft = _feature_table(assoc, kg, use_cache=use_cache)
    anchors = set(_load_anchors())
    ft["label"] = ft.index.to_series().apply(lambda g: 1 if g in anchors else 0)

    pos = ft[ft["label"] == 1]
    neg = ft[ft["label"] == 0]

    if len(pos) == 0:
        # Fallback: no training possible; enhanced heuristic scoring
        ft["score"] = (
            0.4 * ft["eff"]
            + 0.3 * ft["neglogp"]
            + 0.1 * ft["network_importance"]
            + 0.1 * ft["pathway_relevance"]
            + 0.1 * ft["disease_relevance"]
        )
        print("[LTR] No positive examples, using enhanced heuristic scoring")
    else:
        # Choose training method
        if method == "auto":
            method = "pairwise" if len(pos) >= 2 and len(neg) >= 10 else "pointwise"

        if method == "pairwise" and len(pos) >= 2:
            # Pairwise training
            X_pairs, y_pairs = _create_pairwise_training_data(ft, anchors)
            if len(X_pairs) > 0:
                model = _train_pairwise_ltr(
                    X_pairs, y_pairs, params=model_params, monotone=monotone_constraints
                )
                print(f"[LTR] Using pairwise ranking with {len(X_pairs)} pairs")

                # For prediction, need to convert back to pointwise features
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
                X_all = ft[feature_cols].values

                # Use model to score individual examples (approximation)
                if hasattr(model, "predict_proba"):
                    ft["score"] = (
                        model.predict_proba(X_all)[:, 1]
                        if X_all.shape[1] == X_pairs.shape[1]
                        else model.predict(X_all)
                    )
                else:
                    ft["score"] = model.predict(X_all)
            else:
                method = "pointwise"  # Fallback

        if method == "pointwise" or method != "pairwise":
            # Pointwise training with enhanced features
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

            clf = _train_pointwise_ltr(X, y, method="rf", params=model_params)
            print(f"[LTR] Using pointwise ranking with {len(X)} examples")

            # Cross-validation assessment
            try:
                cv_score = cross_val_score(
                    clf, X, y, cv=min(3, len(pos)), scoring="roc_auc"
                ).mean()
                print(f"[LTR] Cross-validation AUC: {cv_score:.3f}")
            except Exception:
                pass

            # Score all genes
            X_all = ft[feature_cols].values
            ft["score"] = (
                clf.predict_proba(X_all)[:, 1]
                if hasattr(clf, "predict_proba")
                else clf.predict(X_all)
            )

    # Merge back onto assoc to produce canonical schema
    assoc2 = assoc.copy()
    assoc2["name"] = assoc2["feature"].astype(str)
    assoc2 = assoc2.merge(
        ft[["score"]].reset_index().rename(columns={"gene": "name"}),
        on="name",
        how="left",
    )
    assoc2["assoc_score"] = assoc2["score"].fillna(0.0)
    out = pd.DataFrame(
        {
            "name": assoc2["name"].astype(str),
            "layer": "transcriptomic",
            "type": "gene",
            "assoc_score": pd.to_numeric(assoc2["assoc_score"], errors="coerce").fillna(
                0.0
            ),
            "p_value": pd.to_numeric(assoc2["p_value"], errors="coerce").fillna(1.0),
            "effect_size": pd.to_numeric(assoc2["effect_size"], errors="coerce").fillna(
                0.0
            ),
            "provenance": assoc2["dataset"].astype(str),
        }
    )
    out = out.sort_values(
        ["assoc_score", "effect_size"], ascending=[False, False]
    ).reset_index(drop=True)
    return out
