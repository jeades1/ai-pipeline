from __future__ import annotations

from typing import Iterable, Optional
import math


def precision_at_k(relevant: Iterable[int], k: int) -> float:
    rel = list(relevant)[:k]
    if k <= 0:
        return 0.0
    return sum(rel) / k


def recall_at_k(relevant: Iterable[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    rel = list(relevant)[:k]
    return sum(rel) / total_relevant


def f1_at_k(relevant: Iterable[int], total_relevant: int, k: int) -> float:
    p = precision_at_k(relevant, k)
    r = recall_at_k(relevant, total_relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision_at_k(relevant: Iterable[int], k: int) -> float:
    """Mean of precision@i over i<=k where item i is relevant (MAP if averaged over queries)."""
    rel = list(relevant)[:k]
    if not rel:
        return 0.0
    ap_sum = 0.0
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            ap_sum += hits / i
    if hits == 0:
        return 0.0
    return ap_sum / hits


def dcg_at_k(gains: Iterable[float], k: int) -> float:
    g = list(gains)[:k]
    dcg = 0.0
    for i, val in enumerate(g, start=1):
        denom = math.log2(i + 1)
        dcg += val / denom
    return dcg


def ndcg_at_k(relevant: Iterable[int], k: int) -> float:
    rel = list(relevant)[:k]
    # Binary gains
    ideal = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(rel, k) / idcg


def mrr(relevant: Iterable[int], k: Optional[int] = None) -> float:
    rel = list(relevant)
    upto = len(rel) if k is None else min(k, len(rel))
    for i in range(upto):
        if rel[i]:
            return 1.0 / (i + 1)
    return 0.0


def hits_at_k(relevant: Iterable[int], k: int) -> int:
    return int(sum(list(relevant)[:k]))


def coverage(found_relevant: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return found_relevant / total_relevant
