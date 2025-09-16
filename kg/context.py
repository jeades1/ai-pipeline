"""
kg/context.py — minimal, controlled vocabulary helpers for context fields.

We keep this light-weight: no external ontology resolvers. The goal is to
standardize keys used on edges and make it easy to extend later.
"""

from __future__ import annotations
from typing import Any, Dict, Optional


def make_context(
    disease: Optional[str] = None,
    stage: Optional[str] = None,
    timepoint: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    if disease:
        ctx["disease"] = disease  # e.g., "AKI" or an EFO code later
    if stage:
        ctx["stage"] = stage  # e.g., "KDIGO-2"
    if timepoint:
        ctx["timepoint"] = timepoint  # e.g., "admission", "48h"
    if environment:
        ctx["environment"] = environment  # e.g., "human", "mouse", "organoid"
    return ctx
