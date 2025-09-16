from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from typing import Dict, Any

FEATURES = ["feat_creat", "feat_urea", "creat_delta", "creat_roll_mean"]


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    y_true: Any
    y_proba: Any


def train_and_eval(
    features_parquet: str = "data/processed/features.parquet",
    target_col: str = "label",
    test_size: float = 0.2,
    seed: int = 42,
) -> TrainResult:
    df = pd.read_parquet(features_parquet).dropna(
        subset=["feat_creat", "creat_delta", "creat_roll_mean", target_col]
    )
    X = df[FEATURES].fillna(0).to_numpy()
    y = df[target_col].astype(int).to_numpy()
    n = len(df)
    n_test = max(1, int(n * test_size))
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(n_test),
    }
    return TrainResult(metrics=metrics, y_true=y_test, y_proba=proba)
