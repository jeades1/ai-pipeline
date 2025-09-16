from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, List
import pandas as pd
import numpy as np

FUNCTION_NODES = {
    "BarrierIntegrity": {
        "kind": "Function",
        "layer": "invitro",
        "name": "Barrier Integrity (TEER)",
    },
    "TransportFunction": {
        "kind": "Function",
        "layer": "invitro",
        "name": "Tubular Transport",
    },
    "InflammationResponse": {
        "kind": "Function",
        "layer": "invitro",
        "name": "Inflammation Response",
    },
    "MetabolicStress": {
        "kind": "Function",
        "layer": "invitro",
        "name": "Metabolic Stress Response",
    },
    "CellViability": {"kind": "Function", "layer": "invitro", "name": "Cell Viability"},
    "CellCellContact": {
        "kind": "Function",
        "layer": "invitro",
        "name": "Cell-Cell Contact Formation",
    },
}

CELL_TYPES = {
    "ProximalTubular": {
        "marker_genes": ["SLC34A1", "LRP2"],
        "functions": ["transport", "reabsorption"],
    },
    "DistalTubular": {
        "marker_genes": ["NCCT", "SLC12A3"],
        "functions": ["electrolyte_balance"],
    },
    "Podocyte": {
        "marker_genes": ["NPHS1", "NPHS2"],
        "functions": ["filtration", "barrier"],
    },
    "Endothelial": {
        "marker_genes": ["PECAM1", "VWF"],
        "functions": ["vascular_tone", "permeability"],
    },
    "Immune": {
        "marker_genes": ["CD68", "CD3E"],
        "functions": ["inflammation", "repair"],
    },
}


def load_invitro_signals(
    path: str | Path = "data/processed/invitro/signals.parquet",
) -> pd.DataFrame:
    """Load in vitro signals if present.

    Expected columns:
      gene, function (one of FUNCTION_NODES keys), effect (float), direction (up/down/neutral), assay, dataset
    """
    p = Path(path)
    if not p.exists():
        # try csv
        p = p.with_suffix(".csv")
        if not p.exists():
            return pd.DataFrame(
                columns=["gene", "function", "effect", "direction", "assay", "dataset"]
            )  # empty
        df = pd.read_csv(p)
    else:
        try:
            df = pd.read_parquet(p)
        except Exception:
            df = (
                pd.read_csv(p.with_suffix(".csv"))
                if p.with_suffix(".csv").exists()
                else pd.DataFrame()
            )
    # normalize
    if df.empty:
        return df
    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = [
        c
        for c in ["gene", "function", "effect", "direction", "assay", "dataset"]
        if c in df.columns
    ]
    return df[keep].copy()


def add_invitro_evidence(kg: Any, inv: pd.DataFrame) -> None:
    """Add in vitro function nodes and gene→function edges to KG."""
    if inv is None or len(inv) == 0:
        return
    # Ensure function nodes
    for fid, attrs in FUNCTION_NODES.items():
        kg.ensure_node(fid, **attrs)
    # Add edges
    for r in inv.itertuples(index=False):
        g = str(getattr(r, "gene", "")).strip()
        fn = str(getattr(r, "function", "")).strip()
        if not g or fn not in FUNCTION_NODES:
            continue
        direction = str(getattr(r, "direction", "")).lower()
        sign = "+" if direction == "up" else ("-" if direction == "down" else "")
        eff = None
        try:
            eff = float(getattr(r, "effect"))
        except Exception:
            eff = None
        kg.ensure_node(g, kind="Gene", layer="transcriptomic", name=g)
        kg.add_edge(
            g,
            fn,
            etype="modulates",
            provenance=str(getattr(r, "dataset", getattr(r, "assay", "invitro"))),
            layer="gene→function",
            sign=sign,
            evidence={"effect": eff, "direction": direction},
        )
    # Optional: link function nodes to a phenotype if provided by the caller via another routine
    # Kept disease-agnostic here.


def functional_consistency_scores(inv: pd.DataFrame) -> pd.DataFrame:
    """Compute per-gene functional consistency score from in vitro signals.

    Simple heuristic: average absolute effect per gene across functions.
    Returns DataFrame with columns [name, inv_score] in [0,1] range (min-max scaled if possible).
    """
    if inv is None or len(inv) == 0:
        return pd.DataFrame(columns=["name", "inv_score"])  # empty
    df = inv.copy()
    if "gene" not in df.columns:
        return pd.DataFrame(columns=["name", "inv_score"])  # empty
    eff = pd.to_numeric(
        df["effect"] if "effect" in df.columns else pd.Series([], dtype=float),
        errors="coerce",
    )
    df["effect"] = eff.fillna(0.0).astype(float).abs()
    df["gene_upper"] = df["gene"].astype(str).str.upper()
    agg = df.groupby("gene_upper", as_index=False)["effect"].mean()
    agg["inv_score"] = agg["effect"].astype(float)
    result_df = pd.DataFrame(
        {"name": agg["gene_upper"].values, "inv_score": agg["inv_score"].values}
    )
    # Scale 0-1
    if len(result_df) > 0:
        v = result_df["inv_score"].to_numpy()
        vmax = float(v.max()) if v.size > 0 else 1.0
        if vmax > 0:
            result_df["inv_score"] = result_df["inv_score"] / vmax
    return result_df


def check_existing_invitro_evidence(
    kg: Any, gene: str, perturbation: str, assay_type: str, cell_type: str
) -> bool:
    """Check if experiment combination already exists in KG to prevent redundant recommendations."""
    try:
        # Query KG for existing edges matching criteria
        for _, _, key, eattrs in kg.G.edges(keys=True, data=True):
            if key == "modulates":
                evidence = eattrs.get("evidence", {})
                if (
                    evidence.get("perturbation") == perturbation
                    and evidence.get("assay") == assay_type
                    and evidence.get("cell_type") == cell_type
                ):
                    return True
        return False
    except Exception:
        return False


def generate_synthetic_invitro_data(
    n_genes: int = 50,
    conditions: Optional[List[str]] = None,
    assays: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate realistic synthetic in vitro data for framework testing."""

    if conditions is None:
        conditions = ["control", "toxin", "ischemia", "inflammation", "hypoxia"]
    if assays is None:
        assays = ["TEER", "permeability", "ELISA", "viability", "imaging"]

    # Optionally seed with known injury/stress genes for synthetic generation
    seed_genes = ["HAVCR1", "LCN2", "TIMP2", "CCL2", "CST3", "IGFBP7", "IL18", "UMOD"]

    # Add some random genes
    import random

    random.seed(42)
    additional_genes = [
        f"GENE{i:03d}" for i in range(max(0, n_genes - len(seed_genes)))
    ]
    all_genes = seed_genes + additional_genes

    data = []
    np.random.seed(42)

    for gene in all_genes:
        for condition in conditions:
            for assay in assays:
                # Realistic effect sizes based on literature patterns
                if gene in ["HAVCR1", "LCN2"] and condition != "control":
                    # Strong injury markers
                    effect = np.random.normal(0.8, 0.2)
                    direction = "up"
                elif gene in ["TIMP2", "CCL2"] and condition in [
                    "cisplatin",
                    "ischemia",
                ]:
                    # Moderate injury response
                    effect = np.random.normal(0.5, 0.3)
                    direction = "up" if effect > 0 else "down"
                elif condition == "control":
                    # Baseline variation
                    effect = np.random.normal(0.0, 0.1)
                    direction = "neutral"
                else:
                    # Variable response
                    effect = np.random.normal(0.2, 0.4)
                    direction = (
                        "up"
                        if effect > 0.1
                        else ("down" if effect < -0.1 else "neutral")
                    )

                # Assign function based on assay type
                if assay == "TEER":
                    function = "BarrierIntegrity"
                elif assay == "permeability":
                    function = "BarrierIntegrity"
                elif assay == "ELISA":
                    function = "InflammationResponse"
                elif assay == "viability":
                    function = "CellViability"
                else:
                    function = "MetabolicStress"

                data.append(
                    {
                        "gene": gene,
                        "function": function,
                        "effect": effect,
                        "direction": direction,
                        "assay": assay,
                        "dataset": f"synthetic_{condition}",
                        "condition": condition,
                        "cell_type": "tubular_epithelial",
                    }
                )

    return pd.DataFrame(data)


def add_cellular_interaction_layer(kg: Any) -> None:
    """Add cell-cell interaction nodes and edges to KG."""

    # Add cell type nodes
    for cell_type, props in CELL_TYPES.items():
        kg.ensure_node(cell_type, kind="CellType", layer="cellular", **props)

    # Add interaction edges based on known biology
    interactions = [
        (
            "ProximalTubular",
            "Endothelial",
            "paracrine_signaling",
            {"mediators": ["VEGF", "NO"], "context": "vascular_support"},
        ),
        (
            "Immune",
            "ProximalTubular",
            "cell_cell_contact",
            {"context": "inflammation", "mediators": ["TNF", "IL1B"]},
        ),
        (
            "Podocyte",
            "Endothelial",
            "mechanical_coupling",
            {"context": "filtration", "mediators": ["integrin"]},
        ),
        (
            "DistalTubular",
            "ProximalTubular",
            "paracrine_signaling",
            {"mediators": ["ATP", "adenosine"], "context": "tubuloglomerular_feedback"},
        ),
        (
            "Endothelial",
            "Immune",
            "adhesion",
            {"mediators": ["VCAM1", "ICAM1"], "context": "recruitment"},
        ),
    ]

    for source, target, interaction_type, evidence in interactions:
        kg.add_edge(
            source,
            target,
            etype=interaction_type,
            provenance="literature_curated",
            evidence=evidence,
            layer="cellular",
        )


def link_genes_to_cellular_functions(kg: Any, invitro_data: pd.DataFrame) -> None:
    """Link gene expression to cellular function outcomes with cell-type specificity."""

    if invitro_data.empty:
        return

    # Group by gene and compute functional consistency across assays
    for gene, gene_data in invitro_data.groupby("gene"):
        functional_scores = {}

        # TEER/permeability -> barrier function
        barrier_data = gene_data[gene_data["assay"].isin(["TEER", "permeability"])]
        if not barrier_data.empty:
            functional_scores["BarrierIntegrity"] = barrier_data["effect"].abs().mean()

        # ELISA/secretome -> inflammation function
        inflam_data = gene_data[gene_data["assay"].isin(["ELISA", "secretome"])]
        if not inflam_data.empty:
            functional_scores["InflammationResponse"] = (
                inflam_data["effect"].abs().mean()
            )

        # Viability -> cell survival
        viability_data = gene_data[gene_data["assay"] == "viability"]
        if not viability_data.empty:
            functional_scores["CellViability"] = viability_data["effect"].abs().mean()

        # Add gene->function edges with cell-type context
        cell_types = gene_data["cell_type"].unique()
        for function, score in functional_scores.items():
            for cell_type in cell_types:
                kg.add_edge(
                    gene,
                    function,
                    etype="regulates_function",
                    evidence={"score": score, "cell_type": cell_type},
                    provenance="invitro_synthetic",
                    layer="gene_function",
                )


def enhanced_functional_consistency_scores(inv: pd.DataFrame) -> pd.DataFrame:
    """Enhanced functional consistency scoring with multi-assay integration."""
    if inv is None or len(inv) == 0:
        return pd.DataFrame(columns=["name", "inv_score"])

    df = inv.copy()
    if "gene" not in df.columns:
        return pd.DataFrame(columns=["name", "inv_score"])

    # Multi-dimensional scoring
    scores_by_gene = []

    for gene, gene_data in df.groupby("gene"):
        gene_upper = str(gene).upper()

        # Score by functional category
        function_scores = {}
        for func in FUNCTION_NODES.keys():
            func_data = gene_data[gene_data["function"] == func]
            if not func_data.empty:
                effects = pd.to_numeric(func_data["effect"], errors="coerce").fillna(
                    0.0
                )
                function_scores[func] = effects.abs().mean()

        # Score by assay consistency
        assay_consistency = 0.0
        if len(gene_data["assay"].unique()) > 1:
            assay_effects = gene_data.groupby("assay")["effect"].mean()
            assay_consistency = 1.0 - (
                assay_effects.std() / (assay_effects.abs().mean() + 1e-6)
            )

        # Score by condition responsiveness
        condition_responsiveness = 0.0
        if "condition" in gene_data.columns:
            control_effect = gene_data[gene_data["condition"] == "control"][
                "effect"
            ].mean()
            treatment_effects = gene_data[gene_data["condition"] != "control"]["effect"]
            if not treatment_effects.empty:
                condition_responsiveness = (
                    treatment_effects.abs() - abs(control_effect)
                ).mean()

        # Combined score
        avg_function_score = (
            np.mean(list(function_scores.values())) if function_scores else 0.0
        )
        combined_score = (
            0.5 * avg_function_score
            + 0.3 * max(0, assay_consistency)
            + 0.2 * max(0, condition_responsiveness)
        )

        scores_by_gene.append(
            {
                "name": gene_upper,
                "inv_score": combined_score,
                "function_scores": function_scores,
                "assay_consistency": assay_consistency,
                "condition_responsiveness": condition_responsiveness,
            }
        )

    result = pd.DataFrame(scores_by_gene)

    # Scale to [0,1]
    if len(result) > 0 and "inv_score" in result.columns:
        scores = result["inv_score"].to_numpy()
        score_max = float(np.max(scores)) if scores.size > 0 else 1.0
        if score_max > 0:
            result["inv_score"] = scores / score_max

    return result[["name", "inv_score"]]
