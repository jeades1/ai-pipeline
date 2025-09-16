from __future__ import annotations
from typing import Tuple, Any


def map_omnipath(sign: Any, directed: Any) -> Tuple[str, str]:
    """
    Map OmniPath interaction meta to a small predicate set.
    Returns (predicate, sign_attr) where sign_attr ∈ {"+","-",""}.
    """
    try:
        s = str(sign).strip().lower()
    except Exception:
        s = ""
    try:
        d = bool(int(directed)) if isinstance(directed, (int, str)) else bool(directed)
    except Exception:
        d = False

    if s in {"1", "+", "pos", "positive", "activation", "activates"}:
        return ("activates" if d else "interacts_with", "+")
    if s in {"-1", "-", "neg", "negative", "inhibition", "inhibits"}:
        return ("inhibits" if d else "interacts_with", "-")
    return ("interacts_with", "")
