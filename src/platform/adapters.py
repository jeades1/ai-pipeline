"""
Disease-agnostic adapter configuration loader.

Adapters describe thin, disease-specific customizations without changing the core platform.
They provide ontology anchors, signal weighting, assay module preferences, and outcome targets.

Schema (JSON):
{
  "name": str,
  "description": str,
  "ontology_terms": [str, ...],
  "prioritization_weights": {"omics": float, "imaging": float, "secretome": float, "functional": float},
  "assay_modules": [str, ...],
  "biomarker_archetypes": [str, ...],
  "outcomes": [str, ...],
  "notes": str
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
from pydantic import BaseModel, Field, field_validator
import re


CONF_DIR = Path("conf/platform/adapters")


class PrioritizationWeights(BaseModel):
    omics: float = Field(ge=0, le=1, default=0.7)
    imaging: float = Field(ge=0, le=1, default=0.7)
    secretome: float = Field(ge=0, le=1, default=0.7)
    functional: float = Field(ge=0, le=1, default=0.7)

    @field_validator("omics", "imaging", "secretome", "functional")
    @classmethod
    def non_negative(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class AdapterConfig(BaseModel):
    name: str
    description: str = ""
    ontology_terms: List[str] = []
    prioritization_weights: PrioritizationWeights = PrioritizationWeights()
    assay_modules: List[str] = []
    biomarker_archetypes: List[str] = []
    outcomes: List[str] = []
    notes: str = ""

    def as_weights_dict(self) -> Dict[str, float]:
        return self.prioritization_weights.model_dump()


def _ensure_conf_dir() -> Path:
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    return CONF_DIR


def list_adapters() -> List[str]:
    """List available adapter names based on JSON files in conf/platform/adapters."""
    _ensure_conf_dir()
    return sorted(p.stem for p in CONF_DIR.glob("*.json"))


def load_adapter(name: str) -> AdapterConfig:
    """Load an adapter by name (without .json) from conf/platform/adapters."""
    _ensure_conf_dir()
    path = CONF_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Adapter not found: {path}")
    data = json.loads(path.read_text())
    return AdapterConfig(**data)


def load_all_adapters() -> Dict[str, AdapterConfig]:
    """Load all adapters into a dict keyed by name."""
    return {name: load_adapter(name) for name in list_adapters()}


def merge_weights(
    default: Dict[str, float], adapter: AdapterConfig
) -> Dict[str, float]:
    """Merge default signal weights with adapter-specific overrides (adapter wins)."""
    merged = {**default}
    merged.update(adapter.as_weights_dict())
    return merged


def resolve_assay_type(provenance: str, conf_path: Path | None = None) -> str:
    """Map a dataset/provenance string to an assay type using conf/platform/assays_map.json.
    Returns one of: omics, imaging, secretome, functional (default: omics).
    """
    try:
        path = conf_path or Path("conf/platform/assays_map.json")
        data = json.loads(path.read_text())
        default = str(data.get("default", "omics"))
        pats = data.get("patterns", [])
        pv = str(provenance or "")
        for item in pats:
            pat = str(item.get("match", ""))
            typ = str(item.get("type", ""))
            if pat and re.search(pat, pv, flags=re.IGNORECASE):
                return typ
        return default
    except Exception:
        return "omics"


if __name__ == "__main__":
    # Tiny smoke test when run directly
    print("Available adapters:", list_adapters())
    for name in list_adapters():
        cfg = load_adapter(name)
        print(f"- {name}: assays={cfg.assay_modules}, outcomes={cfg.outcomes}")
