from __future__ import annotations
from pathlib import Path
from kg.context import make_context
from kg.predicates import map_omnipath
from pathlib import Path as _Path
import zipfile as _zipfile
import io as _io
import pandas as _pd
import pandas as pd

# Public API from this module:
#   - add_multiomics_evidence(kg, features, assoc)
#   - add_causal_hints(kg)
#   - add_clinical_context(kg, labels_df=None)


def _coerce_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Return a shallow copy with standardized column names present."""
    out = df.copy()
    out = out.rename(columns=mapping)
    return out


def add_multiomics_evidence(kg, features: pd.DataFrame, assoc: pd.DataFrame) -> None:
    """
    Build feature→phenotype edges using the association table.

    Expected columns:
      features: ['feature','layer','type']
      assoc:    ['feature','effect_size','p_value','dataset','direction']
    """
    # ---- feature nodes ----
    for _, r in features.iterrows():
        kg.ensure_node(
            str(r["feature"]),
            kind=str(r.get("type", "Gene")).title(),
            layer=str(r.get("layer", "")),
            name=str(r["feature"]),
        )

    # ---- Generic phenotype placeholder node (optional; avoid disease-specific default) ----
    # Downstream code should attach to specific phenotype nodes when available.

    # ---- assoc edges ----
    # normalize columns just to be safe
    colmap = {
        "feature": "feature",
        "effect_size": "effect_size",
        "p_value": "p_value",
        "dataset": "dataset",
        "direction": "direction",
    }
    assoc = _coerce_columns(assoc, colmap)

    for _, r in assoc.iterrows():
        f = str(r["feature"])
        direction = str(r.get("direction", "")).lower()
        sign = "+" if direction == "up" else "-" if direction == "down" else ""

        # numbers
        try:
            pv = float(r.get("p_value", "nan"))
        except Exception:
            pv = None
        try:
            eff = float(r.get("effect_size", "nan"))
        except Exception:
            eff = None

        kg.add_edge(
            f,
            "PHENOTYPE",
            etype="associative",
            direction=direction,
            sign=sign,
            evidence={"p_value": pv, "effect_size": eff},
            provenance=str(r.get("dataset", "")),
            context=make_context(environment="human"),
            layer="feature→phenotype",
        )


def add_causal_hints(kg) -> None:
    """
    Add lightweight causal hints from pre-extracted resources if present.
    Currently:
      - CellPhoneDB: receptor_to_transcription_factor.csv (columns: Receptor,TF)
    """
    # ---- CellPhoneDB receptor→TF ----
    f_cpdb = Path(
        "data/external/kg/cellphonedb/cellphonedb_extracted/receptor_to_transcription_factor.csv"
    )
    if f_cpdb.exists():
        try:
            df = pd.read_csv(f_cpdb)
            # Expect columns ['Receptor','TF']
            for _, r in df.iterrows():
                rec = str(r.get("Receptor", "")).strip()
                tf = str(r.get("TF", "")).strip()
                if not rec or not tf:
                    continue
                kg.ensure_node(rec, kind="Protein", layer="protein", name=rec)
                kg.ensure_node(tf, kind="TranscriptionFactor", layer="tf", name=tf)
                kg.add_edge(
                    rec,
                    tf,
                    etype="causes",
                    provenance="CellPhoneDB",
                    context=make_context(environment="human"),
                    layer="protein→tf",
                )
        except Exception:
            # Keep demo resilient
            pass

    # ---- OmniPath (if releases parquet exists) ----
    try:
        op_p = _Path("src/kg/releases/omnipath_interactions.parquet")
        if not op_p.exists():
            op_p = _Path("kg/releases/omnipath_interactions.parquet")
        if op_p.exists():
            op = _pd.read_parquet(op_p)

            # Column resolution: tolerate common OmniPath schemas
            cols = {c.lower(): c for c in op.columns}

            def pick(*names):
                for n in names:
                    c = cols.get(n.lower())
                    if c:
                        return c
                return None

            c_src = pick("source_gene", "source_genesymbol", "source", "src")
            c_dst = pick("target_gene", "target_genesymbol", "target", "dst")
            c_sign = pick(
                "sign", "consensus_direction", "is_stimulation", "is_inhibition"
            )
            c_dir = pick("directed", "is_directed")

            if not (c_src and c_dst):
                raise ValueError(
                    "OmniPath parquet missing recognizable source/target columns"
                )

            for r in op.itertuples(index=False):
                src = (
                    str(getattr(r, c_src)).strip()
                    if getattr(r, c_src, None) is not None
                    else ""
                )
                dst = (
                    str(getattr(r, c_dst)).strip()
                    if getattr(r, c_dst, None) is not None
                    else ""
                )
                if not src or not dst:
                    continue

                # Derive sign and direction robustly
                sign_val = getattr(r, c_sign) if c_sign else ""
                dir_val = (
                    getattr(r, c_dir) if c_dir else True
                )  # assume directed if unknown

                # Map boolean-style stimulation/inhibition to sign
                try:
                    if isinstance(sign_val, (int, float)):
                        s_in = sign_val
                    elif str(sign_val).strip().lower() in {"true", "t", "yes", "y"}:
                        s_in = 1
                    elif str(sign_val).strip().lower() in {"false", "f", "no", "n"}:
                        s_in = 0
                    else:
                        s_in = sign_val
                except Exception:
                    s_in = sign_val

                pred, sign = map_omnipath(s_in, dir_val)
                kg.ensure_node(src, kind="Gene", layer="transcriptomic", name=src)
                kg.ensure_node(dst, kind="Gene", layer="transcriptomic", name=dst)
                kg.add_edge(
                    src,
                    dst,
                    etype=pred,
                    provenance="OmniPath",
                    context=make_context(environment="human"),
                    layer="gene→gene",
                    sign=sign,
                )
    except Exception:
        pass

    # ---- CPDB ligand-receptor (if releases parquet exists) ----
    try:
        cpdb_p = _Path("kg/releases/cellphonedb_lr.parquet")
        if cpdb_p.exists():
            lr = _pd.read_parquet(cpdb_p)
            for r in lr.itertuples(index=False):
                lig = str(getattr(r, "ligand_gene", "")).strip()
                rec = str(getattr(r, "receptor_gene", "")).strip()
                if not lig or not rec:
                    continue
                kg.ensure_node(lig, kind="Gene", layer="transcriptomic", name=lig)
                kg.ensure_node(rec, kind="Gene", layer="transcriptomic", name=rec)
                kg.add_edge(
                    lig,
                    rec,
                    etype="binds",
                    provenance="CellPhoneDB",
                    context=make_context(environment="human"),
                    layer="ligand→receptor",
                )
    except Exception:
        pass

    # You can add more sources here (e.g., Reactome child→parent “is_part_of”, etc.)
    # Make sure to set etype, provenance, and a sensible layer.


def add_gene_protein_bridge(kg) -> None:
    """
    Optional: Add Gene(symbol) → Protein(UNIPROT:ACC) bridging edges if a local
    mapping file is available. This enables cross-level paths Gene→Protein→Pathway.

    Supported sources (first found wins):
      - kg/releases/hgnc_uniprot_map.parquet (columns: symbol, uniprot)
      - data/external/kg/hgnc/hgnc_complete_set.tsv (HGNC complete set; columns include 'symbol' and 'uniprot_ids')
    """
    # Try prebuilt parquet
    try:
        mp = _Path("kg/releases/hgnc_uniprot_map.parquet")
        if mp.exists():
            m = _pd.read_parquet(mp)
            for r in m.itertuples(index=False):
                sym = str(getattr(r, "symbol", "")).strip()
                acc = str(getattr(r, "uniprot", "")).strip()
                if not sym or not acc:
                    continue
                gene = sym.upper()
                prot = f"UNIPROT:{acc}"
                kg.ensure_node(gene, kind="Gene", layer="transcriptomic", name=gene)
                kg.ensure_node(prot, kind="Protein", layer="protein", name=prot)
                kg.add_edge(
                    gene,
                    prot,
                    etype="encodes",
                    provenance="HGNC",
                    context=make_context(environment="human"),
                    layer="gene→protein",
                )
            return
    except Exception:
        pass

    # Try HGNC complete set TSV
    try:
        hp = _Path("data/external/kg/hgnc/hgnc_complete_set.tsv")
        if hp.exists():
            h = _pd.read_csv(hp, sep="\t")
            # Expect 'symbol' and 'uniprot_ids' (pipe- or comma-separated)
            if "symbol" in h.columns and "uniprot_ids" in h.columns:
                for r in h.itertuples(index=False):
                    sym = str(getattr(r, "symbol", "")).strip()
                    accs = str(getattr(r, "uniprot_ids", "")).strip()
                    if not sym or not accs:
                        continue
                    for acc in [
                        a.strip()
                        for a in accs.replace("|", ",").split(",")
                        if a.strip()
                    ]:
                        gene = sym.upper()
                        prot = f"UNIPROT:{acc}"
                        kg.ensure_node(
                            gene, kind="Gene", layer="transcriptomic", name=gene
                        )
                        kg.ensure_node(prot, kind="Protein", layer="protein", name=prot)
                        kg.add_edge(
                            gene,
                            prot,
                            etype="encodes",
                            provenance="HGNC",
                            context=make_context(environment="human"),
                            layer="gene→protein",
                        )
    except Exception:
        # No-op if unavailable; demo stays resilient
        pass

    # Fallback: if we couldn't add any gene->protein edges (no HGNC map), try
    # adding direct Gene(symbol) -> Pathway membership edges from Reactome GMT.
    # This preserves cross-level reachability for transcriptomics.
    try:
        gmt_zip = _Path("data/external/kg/reactome/ReactomePathways.gmt.zip")
        path_p = _Path("data/external/kg/reactome/ReactomePathways.txt")
        if gmt_zip.exists() and path_p.exists():
            # Build pathway name -> id lookup (Homo sapiens only)
            p_df = _pd.read_csv(
                path_p,
                sep="\t",
                header=None,
                names=["pathway_id", "pathway_name", "species"],
                low_memory=False,
            )
            p_df = p_df[p_df["species"].astype(str).str.lower().eq("homo sapiens")]
            name_to_id = {
                str(r.pathway_name): str(r.pathway_id)
                for r in p_df.itertuples(index=False)
            }

            # Load GMT content
            with _zipfile.ZipFile(gmt_zip, "r") as zf:
                # pick the first .gmt entry
                gmt_names = [n for n in zf.namelist() if n.lower().endswith(".gmt")]
                if not gmt_names:
                    raise FileNotFoundError("Reactome GMT not found in zip")
                with zf.open(gmt_names[0], "r") as fh:
                    text = _io.TextIOWrapper(fh, encoding="utf-8")
                    for line in text:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 2:
                            continue
                        pw_name = parts[0].strip()
                        # Heuristic: most GMTs have description as second col; genes start at col>=2
                        gene_syms = parts[2:] if len(parts) >= 3 else parts[1:]
                        if not gene_syms:
                            continue
                        # Resolve pathway id if possible; otherwise create a name-based id
                        pid = name_to_id.get(pw_name)
                        if pid:
                            pw_id = f"REACTOME:{pid}"
                        else:
                            # fallback id based on name
                            safe_name = (
                                "REACTOME_NAME:" + pw_name.replace(" ", "_")[:128]
                            )
                            pw_id = safe_name
                        kg.ensure_node(
                            pw_id, kind="Pathway", layer="pathway", name=pw_name
                        )
                        for sym in gene_syms:
                            g = str(sym).strip()
                            if not g:
                                continue
                            gene = g.upper()
                            kg.ensure_node(
                                gene, kind="Gene", layer="transcriptomic", name=gene
                            )
                            kg.add_edge(
                                gene,
                                pw_id,
                                etype="member_of",
                                provenance="Reactome-GMT",
                                context=make_context(environment="human"),
                                layer="gene→pathway",
                            )
    except Exception:
        pass
    # ---- Reactome UniProt→Pathway: prefer parquet written by ETL; fallback to direct files ----
    try:
        uni_parq = _Path("kg/releases/reactome_uniprot_map.parquet")
        pw_parq = _Path("kg/releases/reactome_pathways.parquet")
        rel_parq = _Path("kg/releases/reactome_relations.parquet")
        if uni_parq.exists() and pw_parq.exists():
            uni = _pd.read_parquet(uni_parq)
            pw = _pd.read_parquet(pw_parq)
            pw_names = {
                r.pathway_id: r.pathway_name for r in pw.itertuples(index=False)
            }
        else:
            uni_p = _Path("data/external/kg/reactome/UniProt2Reactome_All_Levels.txt")
            path_p = _Path("data/external/kg/reactome/ReactomePathways.txt")
            if not (uni_p.exists() and path_p.exists()):
                raise FileNotFoundError("Reactome parquet/text not available")
            uni = _pd.read_csv(
                uni_p,
                sep="\t",
                header=None,
                names=[
                    "uniprot",
                    "pathway_id",
                    "url",
                    "evidence",
                    "species",
                    "pathway_name",
                ],
            )
            pw = _pd.read_csv(
                path_p,
                sep="\t",
                header=None,
                names=["pathway_id", "pathway_name", "species"],
            )
            uni = uni[uni["species"].str.lower().eq("homo sapiens")]
            pw = pw[pw["species"].str.lower().eq("homo sapiens")]
            pw_names = {
                r.pathway_id: r.pathway_name for r in pw.itertuples(index=False)
            }

        for r in uni.itertuples(index=False):
            acc = str(getattr(r, "uniprot", "")).strip()
            pid = str(getattr(r, "pathway_id", "")).strip()
            if not acc or not pid:
                continue
            prot_id = f"UNIPROT:{acc}"
            pw_id = f"REACTOME:{pid}"
            kg.ensure_node(prot_id, kind="Protein", layer="protein", name=prot_id)
            kg.ensure_node(
                pw_id, kind="Pathway", layer="pathway", name=pw_names.get(pid, pw_id)
            )
            kg.add_edge(
                prot_id,
                pw_id,
                etype="participates_in",
                provenance="Reactome",
                context=make_context(environment="human"),
                layer="protein→pathway",
            )

        # Pathway hierarchy edges (child part_of parent)
        try:
            if rel_parq.exists():
                rel = _pd.read_parquet(rel_parq)
            else:
                rel_p = _Path("data/external/kg/reactome/ReactomePathwaysRelation.txt")
                rel = (
                    _pd.read_csv(
                        rel_p, sep="\t", header=None, names=["parent", "child"]
                    )
                    if rel_p.exists()
                    else None
                )
            if rel is not None and len(rel) > 0:
                for rr in rel.itertuples(index=False):
                    parent = str(getattr(rr, "parent", "")).strip()
                    child = str(getattr(rr, "child", "")).strip()
                    if not parent or not child:
                        continue
                    p_id = f"REACTOME:{parent}"
                    c_id = f"REACTOME:{child}"
                    # ensure nodes exist with names if available
                    kg.ensure_node(
                        p_id,
                        kind="Pathway",
                        layer="pathway",
                        name=pw_names.get(parent, p_id),
                    )
                    kg.ensure_node(
                        c_id,
                        kind="Pathway",
                        layer="pathway",
                        name=pw_names.get(child, c_id),
                    )
                    kg.add_edge(
                        c_id,
                        p_id,
                        etype="part_of",
                        provenance="Reactome",
                        context=make_context(environment="human"),
                        layer="pathway_hierarchy",
                    )
        except Exception:
            pass
    except Exception:
        pass


def add_clinical_context(kg, labels_df: pd.DataFrame | None = None) -> None:
    """
    Optionally add subject-level labels if provided (no-op if labels_df is empty).
    Expects columns: ['subject_id','label'] with label ∈ {0,1}.
    """
    if labels_df is None or len(labels_df) == 0:
        return

    if "subject_id" not in labels_df.columns:
        return
    # Phenotype nodes should be provided externally; default to SUBJECT-only edges.

    for _, r in labels_df.iterrows():
        try:
            sid = str(r["subject_id"])
            # Support either 'label' or legacy 'aki_label'
            lbl = int(
                r["label"] if "label" in labels_df.columns else r.get("aki_label", 0)
            )
        except Exception:
            continue
        subj = f"SUBJECT:{sid}"
        kg.ensure_node(subj, kind="Subject", layer="clinical", name=subj)
        kg.add_edge(
            subj,
            f"LABEL:{lbl}",
            etype="has_label",
            provenance="clinical",
            context=make_context(environment="human"),
            layer="clinical",
        )
