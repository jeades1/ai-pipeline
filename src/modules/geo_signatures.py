from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def _read_deg_folder(deg_dir: str) -> pd.DataFrame:
    deg_dir = Path(deg_dir)
    parts = []
    for csv in sorted(deg_dir.glob("DE_*_vs_rest.csv")):
        df = pd.read_csv(csv)
        cols = {c.lower(): c for c in df.columns}
        symbol = cols.get("symbol") or cols.get("gene") or cols.get("genesymbol")
        padj = (
            cols.get("padj")
            or cols.get("adj_pval")
            or cols.get("p_adj")
            or cols.get("fdr")
        )
        logfc = cols.get("logfc") or cols.get("log2fc")
        if not symbol or not padj or not logfc:
            continue
        df = df.rename(columns={symbol: "symbol", padj: "padj", logfc: "logFC"})
        df["contrast"] = csv.stem.replace("DE_", "").replace("_vs_rest", "")
        parts.append(df[["symbol", "logFC", "padj", "contrast"]].dropna())
    if not parts:
        return pd.DataFrame(columns=["symbol", "logFC", "padj", "contrast"])
    return pd.concat(parts, ignore_index=True)


def update_modules_from_geo(
    deg_dir: str,
    modules_json: str,
    padj_thresh: float = 0.05,
    top_n_each: int = 75,
    include_markers: List[str] | None = None,
) -> Dict:
    df = _read_deg_folder(deg_dir)
    if df.empty:
        return {"modules": {}, "diff": {"added": [], "removed": []}}

    df = df[df["padj"] <= padj_thresh].copy()
    df["abs_logFC"] = df["logFC"].abs()
    new_modules = {
        "tubular_injury": (
            df.sort_values(["contrast", "abs_logFC"], ascending=[True, False])
            .groupby("contrast")
            .head(top_n_each)["symbol"]
            .str.upper()
            .unique()
            .tolist()
        )
    }

    include_markers = include_markers or ["HAVCR1", "LCN2"]
    for m in include_markers:
        if m not in new_modules["tubular_injury"]:
            new_modules["tubular_injury"].append(m)

    p = Path(modules_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(p.read_text()) if p.exists() else {}
    out = {**old, **new_modules}
    p.write_text(json.dumps(out, indent=2))

    diff = {
        "added": sorted(
            list(set(out["tubular_injury"]) - set(old.get("tubular_injury", [])))
        ),
        "removed": sorted(
            list(set(old.get("tubular_injury", [])) - set(out["tubular_injury"]))
        ),
    }
    return {"modules": out, "diff": diff}
