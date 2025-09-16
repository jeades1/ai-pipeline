from __future__ import annotations
from pathlib import Path
import json
import scanpy as sc
import pandas as pd

OUT = Path("modeling/modules")
OUT.mkdir(parents=True, exist_ok=True)

POSSIBLE_OBS_KEYS = [
    "cell_type",
    "celltype",
    "celltype_label",
    "cell_type_major",
    "broad_celltype",
]

BUCKETS = {
    "proximal_tubule": ["proximal", "pt", "s1", "s2", "s3"],
    "TAL": ["thick ascending", "tal", "loop of henle"],
    "collecting_duct": [
        "collecting",
        "principal",
        "pc",
        "intercalated",
        "ic",
        "ic-a",
        "ic-b",
    ],
}


def _pick_obs_key(ad) -> str:
    cols = [c.lower() for c in ad.obs.columns]
    mapping = {c.lower(): c for c in ad.obs.columns}
    for k in POSSIBLE_OBS_KEYS:
        if k in cols:
            return mapping[k]
    raise KeyError(
        f"Could not find a cell-type column. Looked for: {POSSIBLE_OBS_KEYS}. Available: {list(ad.obs.columns)}"
    )


def _ensure_logged(ad):
    if "log1p" in ad.uns:
        return
    try:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    except Exception:
        if "counts" in ad.layers:
            ad.X = ad.layers["counts"].copy()
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        else:
            raise


def _markers_from_rank(ad, groupby: str) -> pd.DataFrame:
    if "rank_genes_groups" not in ad.uns:
        sc.tl.rank_genes_groups(ad, groupby=groupby, method="wilcoxon")
    df = sc.get.rank_genes_groups_df(ad, group=None)
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] > 0].copy()
    return df


def _bucket_modules(r: pd.DataFrame, top_n: int) -> dict:
    modules = {}
    r = r.copy()
    r["group_lc"] = r["group"].astype(str).str.lower()

    for mod, kws in BUCKETS.items():
        pat = "|".join([k.lower() for k in kws])
        sub = r[r["group_lc"].str.contains(pat, regex=True, na=False)]
        order_cols = [c for c in ["scores", "pvals_adj"] if c in sub.columns]
        asc = [False if c == "scores" else True for c in order_cols]
        sub = (
            sub.sort_values(order_cols, ascending=asc)
            .drop_duplicates(subset=["names"])
            .head(top_n)
        )
        modules[mod] = sub["names"].astype(str).tolist()

    modules.setdefault("injury", [])
    modules.setdefault("repair", [])
    return modules


def derive_atlas_markers(
    atlas_h5ad: str = "data/external/kidney_atlases/kca_mature.h5ad",
    obs_celltype_key: str | None = None,
    top_n: int = 80,
    obs_where: str | None = None,  # NEW: pandas-query-like filter on ad.obs
) -> Path:
    ad = sc.read_h5ad(atlas_h5ad)
    if obs_where:
        # Safe, local expression on ad.obs columns, e.g. "compartment != 'lymphoid' and compartment != 'myeloid'"
        ad = ad[ad.obs.query(obs_where).index].copy()
    key = obs_celltype_key or _pick_obs_key(ad)
    _ensure_logged(ad)
    r = _markers_from_rank(ad, groupby=key)
    modules = _bucket_modules(r, top_n=top_n)

    out = OUT / "tubular_modules_v1.json"
    with open(out, "w") as f:
        json.dump(modules, f, indent=2)
    return out
