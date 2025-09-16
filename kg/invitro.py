from __future__ import annotations
from typing import Optional
import pandas as pd

from .schema import KGEvidenceGraph


def _safe_str(x: object) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def attach_invitro(
    G: KGEvidenceGraph,
    metadata: pd.DataFrame,
    readouts: Optional[pd.DataFrame] = None,
    provenance: str = "invitro",
) -> KGEvidenceGraph:
    """
    Attach in vitro model metadata and assay readouts into the Evidence Graph.

    Expected columns (metadata):
      - donor_id, model_id, geometry, vascularization_level, culture_protocol,
        cell_types (comma-separated), batch_id
    Optional columns:
      - condition_id, perturbation_id, perturbation_name

    Expected columns (readouts):
      - model_id, assay_type, readout_name, value, unit, timepoint, replicate

    Nodes created:
      Donor, OrganoidModel, Geometry, VascularizationLevel, CultureCondition,
      Perturbation, Assay, AssayReadout, ExperimentalBatch

    Edges:
      model->donor (derived_from), model->geometry (has_geometry),
      model->vascularization (has_property), model->condition (under_condition),
      model->perturbation (receives), readout->model (measured_in),
      readout->assay (assessed_by), model->batch (in_batch)
    """

    md = metadata.copy()
    md_cols = {c.lower(): c for c in md.columns}

    def col(name: str) -> str:
        return md_cols.get(name, name)

    for _, r in md.iterrows():
        donor_id = _safe_str(r.get(col("donor_id")))
        model_id = _safe_str(r.get(col("model_id")))
        geometry = _safe_str(r.get(col("geometry")))
        vascular = _safe_str(r.get(col("vascularization_level")))
        protocol = _safe_str(r.get(col("culture_protocol")))
        cell_types = _safe_str(r.get(col("cell_types")))
        batch_id = _safe_str(r.get(col("batch_id")))
        cond_id = _safe_str(r.get(col("condition_id")))
        pert_id = _safe_str(r.get(col("perturbation_id")))
        pert_name = _safe_str(r.get(col("perturbation_name")))

        # Donor and Model
        if donor_id:
            G.ensure_node(f"donor:{donor_id}", kind="Donor", name=donor_id)
        if model_id:
            G.ensure_node(
                f"model:{model_id}",
                kind="OrganoidModel",
                name=model_id,
                layer="physiome",
            )

        # Properties
        if geometry:
            G.ensure_node(f"geometry:{geometry}", kind="Geometry", name=geometry)
            G.add_edge(
                f"model:{model_id}",
                f"geometry:{geometry}",
                "has_geometry",
                provenance=provenance,
            )
        if vascular:
            G.ensure_node(
                f"vascular:{vascular}", kind="VascularizationLevel", name=vascular
            )
            G.add_edge(
                f"model:{model_id}",
                f"vascular:{vascular}",
                "has_property",
                provenance=provenance,
            )
        if protocol:
            G.ensure_node(f"protocol:{protocol}", kind="CultureProtocol", name=protocol)
            G.add_edge(
                f"model:{model_id}",
                f"protocol:{protocol}",
                "cultured_with",
                provenance=provenance,
            )
        if batch_id:
            G.ensure_node(f"batch:{batch_id}", kind="ExperimentalBatch", name=batch_id)
            G.add_edge(
                f"model:{model_id}",
                f"batch:{batch_id}",
                "in_batch",
                provenance=provenance,
            )

        if donor_id and model_id:
            G.add_edge(
                f"model:{model_id}",
                f"donor:{donor_id}",
                "derived_from",
                provenance=provenance,
            )

        if cell_types:
            for ct in [c.strip() for c in cell_types.split(",") if c.strip()]:
                G.ensure_node(f"celltype:{ct}", kind="CellType", name=ct)
                G.add_edge(
                    f"model:{model_id}",
                    f"celltype:{ct}",
                    "contains",
                    provenance=provenance,
                )

        if cond_id:
            G.ensure_node(f"condition:{cond_id}", kind="CultureCondition", name=cond_id)
            G.add_edge(
                f"model:{model_id}",
                f"condition:{cond_id}",
                "under_condition",
                provenance=provenance,
            )

        if pert_id or pert_name:
            pid = pert_id or pert_name
            G.ensure_node(f"pert:{pid}", kind="Perturbation", name=pert_name or pert_id)
            G.add_edge(
                f"model:{model_id}", f"pert:{pid}", "receives", provenance=provenance
            )

    if readouts is not None and not readouts.empty:
        rd = readouts.copy()
        rd_cols = {c.lower(): c for c in rd.columns}

        def colr(name: str) -> str:
            return rd_cols.get(name, name)

        for _, r in rd.iterrows():
            model_id = _safe_str(r.get(colr("model_id")))
            assay_type = _safe_str(r.get(colr("assay_type")))
            readout_name = _safe_str(r.get(colr("readout_name")))
            unit = _safe_str(r.get(colr("unit")))
            timepoint = _safe_str(r.get(colr("timepoint")))
            replicate = _safe_str(r.get(colr("replicate")))

            if assay_type:
                G.ensure_node(f"assay:{assay_type}", kind="Assay", name=assay_type)
            if readout_name:
                rnid = f"readout:{assay_type}:{readout_name}:{timepoint}:{replicate}"
                G.ensure_node(
                    rnid, kind="AssayReadout", name=readout_name, layer="phenome"
                )
                if model_id:
                    G.add_edge(
                        rnid, f"model:{model_id}", "measured_in", provenance=provenance
                    )
                if assay_type:
                    G.add_edge(
                        rnid,
                        f"assay:{assay_type}",
                        "assessed_by",
                        provenance=provenance,
                    )
                if unit:
                    G.G.nodes[rnid]["unit"] = unit
                if timepoint:
                    G.G.nodes[rnid]["timepoint"] = timepoint
                if replicate:
                    G.G.nodes[rnid]["replicate"] = replicate

    return G
