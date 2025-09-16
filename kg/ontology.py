# kg/ontology.py
from typing import Dict

ALIASES = {
    "NGAL": ["LCN2", "Lipocalin-2"],
    "KIM-1": ["HAVCR1"],
    "TIMP-2·IGFBP7": ["TIMP2*IGFBP7", "[TIMP-2][IGFBP7]"],
}


def normalize_label(label: str) -> str:
    label = label.strip()
    for canon, syns in ALIASES.items():
        if label == canon or label in syns:
            return canon
    return label


def canonicalize_map(names) -> Dict[str, str]:
    return {n: normalize_label(n) for n in names}
