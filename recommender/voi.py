from typing import List, Dict, Any, Optional
import math
from pathlib import Path
import pandas as pd

TISSUE_MODELS = [
    {
        "id": "organoid_IR",
        "name": "Kidney organoid — ischemia-reperfusion",
        "targets": ["NGAL", "KIM-1", "IL-18"],
        "readouts": ["RNA-seq", "ELISA"],
        "effector": "hypoxia/reoxygenation",
        "cost": 3,
        "duration_days": 4,
    },
    {
        "id": "slice_nephrotoxin",
        "name": "Precision-cut kidney slice — cisplatin",
        "targets": ["TIMP-2·IGFBP7", "NGAL", "SERPINA3"],
        "readouts": ["ELISA", "proteomics"],
        "effector": "cisplatin 20 µM",
        "cost": 2,
        "duration_days": 3,
    },
    {
        "id": "MPS_sepsis",
        "name": "Microphysiological system — sepsis cytokine mix",
        "targets": ["C5a", "IL-18", "NGAL"],
        "readouts": ["proteomics"],
        "effector": "TNFα/IL-1β/IL-6 cocktail",
        "cost": 4,
        "duration_days": 5,
    },
]


def build_models_from_invitro(
    metadata_csv: Path,
    readouts_csv: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Construct experiment candidates from in vitro metadata/readouts CSVs.
    - targets are inferred from readout_name per model (uppercased).
    - effector derived from perturbation_name or condition_id.
    - simple defaults for cost/duration; can be refined with columns later.
    """
    md = pd.read_csv(metadata_csv)
    rd = None
    if readouts_csv and Path(readouts_csv).exists():
        rd = pd.read_csv(readouts_csv)

    # Index readouts by model_id -> set of readout_names
    targets_map: Dict[str, List[str]] = {}
    if rd is not None and not rd.empty:
        if "model_id" in rd.columns and "readout_name" in rd.columns:
            g = rd.groupby("model_id")
            for mid, df in g:
                names = df["readout_name"].astype(str).tolist()
                t = [s.upper() for s in names if s and s.lower() not in ("nan", "none")]
                targets_map[str(mid)] = sorted(list(set(t)))

    models: List[Dict[str, Any]] = []
    for _, r in md.iterrows():
        mid = str(r.get("model_id"))
        name = str(r.get("geometry")) or mid
        vasc = str(r.get("vascularization_level", ""))
        cond = str(r.get("condition_id", ""))
        pert = str(r.get("perturbation_name", "")) or str(r.get("perturbation_id", ""))
        effector = pert or cond or "baseline"
        # Simple heuristics for cost and duration
        cost = 3
        duration_days = 4
        if "blood" in vasc.lower() or "lymph" in vasc.lower():
            cost += 1
        if "multi" in str(r.get("geometry", "")).lower():
            cost += 1
            duration_days += 1

        models.append(
            {
                "id": mid,
                "name": f"{name} — {effector}",
                "targets": targets_map.get(mid, []),
                "readouts": ["ELISA", "RNA-seq", "Imaging"],
                "effector": effector,
                "cost": cost,
                "duration_days": duration_days,
                "capabilities": {
                    "vascularization": vasc,
                    "geometry": str(r.get("geometry", "")),
                    "protocol": str(r.get("culture_protocol", "")),
                },
            }
        )
    return models


def _is_promoted(c: Dict[str, Any]) -> bool:
    cs = c.get("causal_support") or {}
    if isinstance(cs, dict):
        return cs.get("level") == "promoted"
    if isinstance(cs, str):
        return "promoted" in cs
    return False


def _eig_binary(p: float, n: int) -> float:
    """Approximate expected information gain for Bernoulli success prob p with n observations.
    Uses a small closed-form surrogate: ΔH ≈ 0.5 * log(1 + n / (p*(1-p) + 1e-6)).
    """
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 0.5 * math.log(1.0 + n / (p * (1.0 - p) + 1e-6))


def recommend_experiments(
    ranked: List[Dict[str, Any]],
    kg,
    top_k: int = 3,
    models: Optional[List[Dict[str, Any]]] = None,
):
    # Focus weight on the top candidates
    weights = {
        str(c["name"]).upper(): 0.5
        + float(c.get("total_score", c.get("assoc_score", 0.0)))
        for c in ranked[:20]
    }

    # Identify candidates that are NOT promoted (bigger identifiability gain when covered)
    not_promoted = {str(c["name"]).upper() for c in ranked[:50] if not _is_promoted(c)}

    pool = models if models is not None and len(models) > 0 else TISSUE_MODELS
    scored = []
    for ex in pool:
        coverage = {t.upper() for t in ex["targets"]} & set(weights.keys())
        voi = sum(weights.get(t, 0.0) for t in coverage)
        ident_gain = sum(1.0 for t in coverage if t in not_promoted)
        # Optional Bayesian EIG (cheap surrogate): assume per-target p ~ sigmoid(score)
        eig = 0.0
        for t in coverage:
            base = weights.get(t, 0.0)
            p = 1.0 / (1.0 + math.exp(-base))
            eig += _eig_binary(p, n=1)

        # Composite: identifiability first, then VoI, then EIG, penalize cost
        score = 0.55 * ident_gain + 0.4 * voi + 0.15 * eig - 0.05 * ex["cost"]
        scored.append(
            {
                **ex,
                "voi": round(voi, 3),
                "eig": round(eig, 3),
                "identifiability_gain": ident_gain,
                "score": round(score, 3),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _load_ranked_from_promoted(promoted_tsv: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(promoted_tsv, sep="\t")
    if not {"name", "type"}.issubset(df.columns):
        return []
    genes = df[df["type"].str.lower() == "gene"].copy()
    # Earlier rows higher; simple linear proxy score
    genes["assoc_score"] = list(
        reversed([i / max(len(genes), 1) for i in range(len(genes))])
    )
    return [
        {"name": n, "assoc_score": float(s)}
        for n, s in zip(genes["name"].astype(str), genes["assoc_score"])
    ]


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Recommend experiments from in vitro models and promoted list"
    )
    ap.add_argument("--promoted", type=Path, default=Path("artifacts/promoted.tsv"))
    ap.add_argument(
        "--metadata", type=Path, default=Path("data/processed/invitro_metadata.csv")
    )
    ap.add_argument(
        "--readouts", type=Path, default=Path("data/processed/invitro_readouts.csv")
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/recommendations.json"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    ranked = _load_ranked_from_promoted(args.promoted)
    models = []
    if args.metadata.exists():
        models = build_models_from_invitro(
            args.metadata, args.readouts if args.readouts.exists() else None
        )

    # No KG object required for current scoring; pass None
    recs = recommend_experiments(ranked, kg=None, top_k=args.top_k, models=models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recs, indent=2))
    print(f"[recommend] Wrote {args.out}")
