from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import pandas as pd
import networkx as nx


class AdvancedTissueChipKGSchema:
    """
    Extended KG schema for multicellular tissue-chip architecture,
    PDO vascularization, and kinetic analysis metadata.
    """

    # Multicellular Architecture Node Types
    MULTICELLULAR_NODE_TYPES = {
        "MulticellularArchitecture": "multicellular_architecture",
        "TubularGeometry": "tubular_geometry",
        "CellTypeComposition": "cell_type_composition",
        "CellCellSignaling": "cell_cell_signaling",
        "BarrierFunction": "barrier_function",
        "TissueOrganization": "tissue_organization",
    }

    # Vascularization Node Types
    VASCULARIZATION_NODE_TYPES = {
        "PDOVascularization": "pdo_vascularization",
        "VascularNetwork": "vascular_network",
        "MolecularDelivery": "molecular_delivery",
        "PerfusionSystem": "perfusion_system",
        "EndothelialIntegration": "endothelial_integration",
        "VascularPermeability": "vascular_permeability",
    }

    # Kinetic Analysis Node Types
    KINETIC_NODE_TYPES = {
        "RecirculationSystem": "recirculation_system",
        "KineticAnalysis": "kinetic_analysis",
        "BiomarkerKinetics": "biomarker_kinetics",
        "TemporalProfiling": "temporal_profiling",
        "PharmacokineticModeling": "pharmacokinetic_modeling",
        "ClearanceAnalysis": "clearance_analysis",
    }

    # Tissue-Chip Edge Types
    TISSUE_CHIP_EDGE_TYPES = {
        "has_architecture": "multicellular_architecture_relationship",
        "enables_signaling": "cell_cell_signaling_relationship",
        "provides_vascularization": "vascularization_relationship",
        "enhances_delivery": "molecular_delivery_relationship",
        "supports_kinetics": "kinetic_analysis_relationship",
        "extends_viability": "culture_longevity_relationship",
        "improves_sensitivity": "detection_sensitivity_relationship",
        "validates_biomarker": "tissue_chip_validation_relationship",
    }


class KGEvidenceGraph:
    """
    Small helper over a MultiDiGraph that enforces a few conventions:
      - Nodes have at least: id, kind, layer, name
      - Edges have at least: s, predicate (edge type), o, context, layer
      - We store multi-edges keyed by predicate (etype)
    """

    def __init__(self, context: str = ""):
        self.context = context
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()

    # ---------- nodes ----------
    def ensure_node(self, node_id: str, **attrs: Any) -> None:
        """Create/update a node; fill defaults if missing."""
        if not node_id:
            return
        attrs = dict(attrs) if attrs else {}
        attrs.setdefault("kind", "Entity")
        attrs.setdefault("name", node_id)
        # layer is optional on nodes, leave unset if not given
        if node_id in self.G:
            self.G.nodes[node_id].update(attrs)
        else:
            self.G.add_node(node_id, **attrs)

    # ---------- edges ----------
    def add_edge(self, src: str, dst: str, etype: str, **attrs: Any) -> None:
        """
        Add a multi-edge keyed by `etype`. Also store `predicate=etype` as an attribute.
        Fill in defaults for context and layer if absent.
        """
        if src not in self.G:
            self.ensure_node(src)
        if dst not in self.G:
            self.ensure_node(dst)

        attrs = dict(attrs) if attrs else {}
        # normalize required attrs
        attrs.setdefault("predicate", etype)
        attrs.setdefault("context", self.context)  # ensure single source of truth
        attrs.setdefault("layer", "")  # keep optional

        # Never pass 'context' twice; pass only via **attrs
        self.G.add_edge(
            src,
            dst,
            key=etype,  # multi-edge key
            **attrs,
        )

    # ---------- export ----------
    def export_tsv(self, outdir: Path, also_parquet: bool = True) -> None:
        """
        Write two TSVs that tools/kg_smoke.sh expects:
        nodes: id, kind, layer, name
        edges: s, predicate, o, context, direction, evidence, provenance, sign
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # nodes
        nrows = []
        for nid, nattrs in self.G.nodes(data=True):
            nrows.append(
                {
                    "id": nid,
                    "kind": nattrs.get("kind", ""),
                    "layer": nattrs.get("layer", ""),
                    "name": nattrs.get("name", nid),
                }
            )
        nodes_df = pd.DataFrame(nrows, columns=["id", "kind", "layer", "name"])
        nodes_df.to_csv(outdir / "kg_nodes.tsv", sep="\t", index=False)
        if also_parquet:
            try:
                nodes_df.to_parquet(outdir / "kg_nodes.parquet", index=False)
            except Exception:
                pass

        # edges
        erows = []
        for s, o, key, eattrs in self.G.edges(keys=True, data=True):
            ev = eattrs.get("evidence", "")
            if isinstance(ev, (dict, list)):
                ev = json.dumps(ev, ensure_ascii=False)

            ctx = eattrs.get("context", self.context)
            if isinstance(ctx, (dict, list)):
                ctx = json.dumps(ctx, ensure_ascii=False)

            erows.append(
                {
                    "s": s,
                    "predicate": eattrs.get("predicate", key),
                    "o": o,
                    "context": ctx,
                    "direction": eattrs.get("direction", ""),
                    "evidence": ev,
                    "provenance": eattrs.get("provenance", ""),
                    "sign": eattrs.get("sign", ""),
                }
            )
        edges_df = pd.DataFrame(
            erows,
            columns=[
                "s",
                "predicate",
                "o",
                "context",
                "direction",
                "evidence",
                "provenance",
                "sign",
            ],
        )
        edges_df.to_csv(outdir / "kg_edges.tsv", sep="\t", index=False)
        if also_parquet:
            try:
                edges_df.to_parquet(outdir / "kg_edges.parquet", index=False)
            except Exception:
                pass
