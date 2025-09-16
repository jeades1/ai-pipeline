# learn/causal.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union, List, Dict, Any

import pandas as pd

# Small score boost for biomarkers present in a benchmark anchor list
DEFAULT_PROMOTION_BONUS = 0.15


def _load_anchors() -> set[str]:
    """
    Load anchor biomarkers (e.g., validated AKI markers) from benchmarks/aki_markers.json.
    Accepts either:
      - a list of strings or dicts with "name" or "gene"
      - an object with key "biomarkers": [...]
    Returns a case-normalized set of names.
    """
    p = Path("benchmarks/aki_markers.json")
    if not p.exists():
        return set()

    try:
        data = json.loads(p.read_text())
    except Exception:
        return set()

    names: List[str] = []
    if isinstance(data, dict) and "biomarkers" in data:
        for item in data.get("biomarkers", []):
            if isinstance(item, dict):
                n = item.get("name") or item.get("gene")
                if n:
                    names.append(str(n))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                n = item.get("name") or item.get("gene")
                if n:
                    names.append(str(n))
            else:
                names.append(str(item))
    return {n.upper() for n in names if n}


ANCHORS: set[str] = _load_anchors()


def promote_causal_edges(
    ranked: Union[pd.DataFrame, Iterable[Dict[str, Any]]],
    kg: Any = None,
    bonus: float = DEFAULT_PROMOTION_BONUS,
) -> pd.DataFrame:
    """
    Add simple causal promotion metadata using benchmark anchors.

    Input:
      - ranked: DataFrame OR iterable of dict-like rows
        Expected columns/keys: 'name' (or 'feature'), 'assoc_score' (float)
      - kg: (unused here, reserved for future KG-aware rules)
      - bonus: score bump applied when a candidate is in ANCHORS

    Output:
      - DataFrame with at least: name, assoc_score, total_score, causal_support
    """
    # Normalize input to list-of-dicts
    if isinstance(ranked, pd.DataFrame):
        rows = ranked.to_dict(orient="records")
    else:
        rows = []
        for r in ranked:
            if isinstance(r, dict):
                rows.append(dict(r))
            elif hasattr(r, "to_dict"):
                rows.append(dict(r.to_dict()))
            else:
                # If someone accidentally passed strings (e.g., column names), skip them
                continue

    out: List[Dict[str, Any]] = []
    for c in rows:
        # Normalize name and handle legacy 'feature'
        name = str(c.get("name") or c.get("feature") or "").upper()
        c["name"] = name

        assoc = float(c.get("assoc_score", 0.0))
        total = assoc

        if name and name in ANCHORS:
            c["causal_support"] = {
                "level": "promoted",
                "source": "benchmark",
                "details": "Listed in benchmarks/aki_markers.json",
            }
            total = assoc + float(bonus)
        else:
            # Preserve any existing causal_support, otherwise add a neutral baseline
            c.setdefault(
                "causal_support",
                {
                    "level": "suggestive",
                    "source": "association",
                    "details": "Association-only evidence",
                },
            )

        c["total_score"] = total
        out.append(c)

    return pd.DataFrame(out)
