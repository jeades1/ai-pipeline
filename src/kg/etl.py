# src/kg/etl.py
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

# ---- Paths ------------------------------------------------------------------

EXTERNAL = Path("data/external/kg")
REACTOME_DIR = EXTERNAL / "reactome"
CPDB_DIR = EXTERNAL / "cellphonedb"

RELEASE = Path("kg/releases")
RELEASE.mkdir(parents=True, exist_ok=True)

# ---- Helpers ----------------------------------------------------------------


def _download(url: str, dest: Path, text: bool = True, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    if text:
        dest.write_text(r.text, encoding="utf-8")
    else:
        dest.write_bytes(r.content)


def _ensure_reactome() -> Dict[str, Path]:
    """Ensure core Reactome text files are available locally; download if missing."""
    files = {
        "pathways": REACTOME_DIR / "ReactomePathways.txt",
        "relations": REACTOME_DIR / "ReactomePathwaysRelation.txt",
        "uniprot_map": REACTOME_DIR / "UniProt2Reactome_All_Levels.txt",
        # Optional: GMT gene sets
        "gmt_zip": REACTOME_DIR / "ReactomePathways.gmt.zip",
    }
    urls = {
        "pathways": "https://reactome.org/download/current/ReactomePathways.txt",
        "relations": "https://reactome.org/download/current/ReactomePathwaysRelation.txt",
        "uniprot_map": "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt",
        "gmt_zip": "https://reactome.org/download/current/ReactomePathways.gmt.zip",
    }
    for key, path in files.items():
        if not path.exists():
            # GMT is optional; skip if it fails
            try:
                _download(urls[key], path, text=(key != "gmt_zip"))
                print(f"[reactome] downloaded {key} -> {path}")
            except Exception as e:
                if key == "gmt_zip":
                    print(f"[reactome] optional {key} not downloaded: {e}")
                else:
                    raise
    return files


# ---- OmniPath ----------------------------------------------------------------

# ---- OmniPath ----------------------------------------------------------------


def _pull_omnipath_interactions(offline: bool = False) -> pd.DataFrame:
    cache = RELEASE / "omnipath_interactions.parquet"
    if offline and cache.exists():
        return pd.read_parquet(cache)

    url = "https://omnipathdb.org/interactions?format=json&fields=extra_attrs,evidences"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.json_normalize(r.json())
        # normalize common columns
        rename = {
            "source_genesymbol": "source_gene",
            "target_genesymbol": "target_gene",
            "is_directed": "directed",
            "n_refs": "n_refs",
            "consensus_direction": "sign",
        }
        for k, v in rename.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        df["source"] = "omnipath_interactions"
        return df
    except Exception as e:
        print(f"[kg][warn] OmniPath interactions fetch failed: {e}")
        if cache.exists():
            print("[kg] Reusing cached OmniPath interactions from releases/")
            return pd.read_parquet(cache)
        print("[kg] No cache available; continuing with empty interactions.")
        return pd.DataFrame(
            columns=[
                "source_gene",
                "target_gene",
                "directed",
                "n_refs",
                "sign",
                "source",
            ]
        )


def _pull_omnipath_intercell(offline: bool = False) -> pd.DataFrame:
    cache = RELEASE / "omnipath_intercell.parquet"
    if offline and cache.exists():
        return pd.read_parquet(cache)

    url = "https://omnipathdb.org/intercell?format=json"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.json_normalize(r.json())
        df["source"] = "omnipath_intercell"
        return df
    except Exception as e:
        print(f"[kg][warn] OmniPath intercell fetch failed: {e}")
        if cache.exists():
            print("[kg] Reusing cached OmniPath intercell from releases/")
            return pd.read_parquet(cache)
        print("[kg] No cache available; continuing with empty intercell.")
        return pd.DataFrame()


# ---- CellPhoneDB -------------------------------------------------------------


def _extract_cpdb_zip_if_needed() -> Path:
    """
    If only a 'cellphonedb.zip' is present, extract it under CPDB_DIR / 'cellphonedb_extracted'.
    Return a path under which CSVs can be found.
    """
    zip_path = CPDB_DIR / "cellphonedb.zip"
    extracted = CPDB_DIR / "cellphonedb_extracted"
    if extracted.exists():
        return extracted
    if zip_path.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted)
        print(f"[cpdb] extracted {zip_path} -> {extracted}")
        return extracted
    # else look for already-unpacked 'cellphonedb' dir
    ready = CPDB_DIR / "cellphonedb"
    return ready if ready.exists() else CPDB_DIR  # fallback; caller will probe


def _load_cpdb_lr() -> pd.DataFrame:
    """
    CellPhoneDB v5 loader:
      1) Read interaction_table.csv (multidata_1_id, multidata_2_id)
      2) Map multidata_id -> id_protein via protein_table.protein_multidata_id
      3) Map id_protein -> gene_name via gene_table
      4) For complexes, expand via complex_composition_table (complex_multidata_id -> protein_multidata_id)
    Falls back to protein_name (e.g., ESR1_HUMAN) if gene_name is missing.
    """
    root = _extract_cpdb_zip_if_needed()

    def read_first(*relative):
        for r in relative:
            p = root / r
            if p.exists():
                return p
        return None

    inter_p = read_first(
        "cellphonedb_extracted/interaction_table.csv",
        "interaction_table.csv",
        "cellphonedb/interaction_table.csv",
    )
    multidata_p = read_first(
        "cellphonedb_extracted/multidata_table.csv",
        "multidata_table.csv",
    )
    protein_p = read_first(
        "cellphonedb_extracted/protein_table.csv",
        "protein_table.csv",
    )
    gene_p = read_first(
        "cellphonedb_extracted/gene_table.csv",
        "gene_table.csv",
    )
    complex_comp_p = read_first(
        "cellphonedb_extracted/complex_composition_table.csv",
        "complex_composition_table.csv",
    )

    if not inter_p or not protein_p or not gene_p:
        print(
            "[cpdb] WARNING: missing interaction/protein/gene table(s). Emitting empty."
        )
        return pd.DataFrame(
            columns=["ligand_gene", "receptor_gene", "annotation", "source"]
        )

    inter = pd.read_csv(inter_p, low_memory=False)
    protein = pd.read_csv(protein_p, low_memory=False)
    gene = pd.read_csv(gene_p, low_memory=False)
    multidata = pd.read_csv(multidata_p, low_memory=False) if multidata_p else None
    complex_comp = (
        pd.read_csv(complex_comp_p, low_memory=False) if complex_comp_p else None
    )

    # Normalize column names to lowercase
    for df in (inter, protein, gene):
        df.columns = [c.strip().lower() for c in df.columns]
    if multidata is not None:
        multidata.columns = [c.strip().lower() for c in multidata.columns]
    if complex_comp is not None:
        complex_comp.columns = [c.strip().lower() for c in complex_comp.columns]

    # Required columns present?
    if not {"multidata_1_id", "multidata_2_id"}.issubset(inter.columns):
        print(
            f"[cpdb] WARNING: expected multidata_1_id/multidata_2_id in {inter_p.name}."
        )
        return pd.DataFrame(
            columns=["ligand_gene", "receptor_gene", "annotation", "source"]
        )

    # --- Build mappings ---
    # protein: id_protein, protein_multidata_id, protein_name (no 'uniprot' in your dump)
    if not {"id_protein", "protein_multidata_id"}.issubset(protein.columns):
        print(
            "[cpdb] WARNING: protein_table is missing id_protein/protein_multidata_id."
        )
        return pd.DataFrame(
            columns=["ligand_gene", "receptor_gene", "annotation", "source"]
        )

    # Map protein_id -> preferred gene symbol (gene_name if present; else hgnc_symbol)
    gene.columns = [c.strip().lower() for c in gene.columns]
    # expected columns: protein_id, gene_name, hgnc_symbol
    if "protein_id" not in gene.columns:
        print("[cpdb] WARNING: gene_table missing protein_id column.")
        return pd.DataFrame(
            columns=["ligand_gene", "receptor_gene", "annotation", "source"]
        )

    gene["gene_pref"] = gene.get("gene_name", pd.Series(index=gene.index, dtype=object))
    if "hgnc_symbol" in gene.columns:
        gene.loc[
            gene["gene_pref"].isna()
            | (gene["gene_pref"].astype(str).str.strip() == ""),
            "gene_pref",
        ] = gene["hgnc_symbol"]

    prot_to_gene = (
        gene.dropna(subset=["protein_id"])
        .astype({"protein_id": "Int64"})
        .groupby("protein_id", as_index=True)["gene_pref"]
        .agg(lambda s: s.dropna().astype(str).unique()[0] if len(s.dropna()) else None)
    )

    # Fallback map: id_protein -> protein_name (e.g., ESR1_HUMAN) if gene symbol missing
    protein_idx = protein.set_index("id_protein")
    prot_to_name = (
        protein_idx["protein_name"] if "protein_name" in protein_idx.columns else None
    )

    def protein_id_to_symbol(pid: int) -> str | None:
        g = prot_to_gene.get(pid, None)
        if pd.notna(g) and str(g).strip():
            return str(g)
        if prot_to_name is not None:
            pn = prot_to_name.get(pid, None)
            if pd.notna(pn) and str(pn).strip():
                return f"PN:{pn}"  # explicit that this is a protein name fallback
        return None

    # Map multidata_id -> protein_id when the multidata refers to a protein
    # In v5, this is via protein.protein_multidata_id
    md_to_protein_id = {}
    if "protein_multidata_id" in protein.columns:
        for r in protein.itertuples(index=False):
            try:
                md = int(getattr(r, "protein_multidata_id"))
                pid = int(getattr(r, "id_protein"))
                md_to_protein_id[md] = pid
            except Exception:
                continue

    # Complex membership: complex_multidata_id -> [protein_id ...]
    complex_members: dict[int, list[int]] = {}
    if complex_comp is not None:
        if {"complex_multidata_id", "protein_multidata_id"}.issubset(
            complex_comp.columns
        ):
            for r in complex_comp.itertuples(index=False):
                try:
                    cm = int(getattr(r, "complex_multidata_id"))
                    pm_md = int(getattr(r, "protein_multidata_id"))
                except Exception:
                    continue
                pid = md_to_protein_id.get(pm_md)
                if pid is not None:
                    complex_members.setdefault(cm, []).append(pid)

    # Resolve a CPDB 'multidata_id' to a set of symbols
    def md_to_symbols(md_id: int) -> list[str]:
        out: list[str] = []
        pid = md_to_protein_id.get(md_id)
        if pid is not None:
            s = protein_id_to_symbol(pid)
            if s:
                out.append(s)
        # Complex case
        if md_id in complex_members:
            for pid in complex_members[md_id]:
                s = protein_id_to_symbol(pid)
                if s:
                    out.append(s)
        # dedupe and drop empties
        return sorted(set([x for x in out if x]))

    # Build pairs
    pairs = []
    ann_cols = [
        c for c in ("classification", "directionality", "source") if c in inter.columns
    ]
    for r in inter.itertuples(index=False):
        m1 = getattr(r, "multidata_1_id")
        m2 = getattr(r, "multidata_2_id")
        if pd.isna(m1) or pd.isna(m2):
            continue
        A = md_to_symbols(int(m1))
        B = md_to_symbols(int(m2))
        if not A or not B:
            continue
        annot = " | ".join(
            [str(getattr(r, c)) for c in ann_cols if pd.notna(getattr(r, c))]
        )
        annot = annot if annot else "cpdb_interaction"
        for a in A:
            for b in B:
                pairs.append(
                    {
                        "ligand_gene": a,
                        "receptor_gene": b,
                        "annotation": annot,
                        "source": "cellphonedb",
                    }
                )

    out = pd.DataFrame(pairs).drop_duplicates()
    return out


# ---- Reactome loaders --------------------------------------------------------


def _load_reactome(files: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    # Pathways: stable_id \t display_name \t species
    pathways = pd.read_csv(
        files["pathways"],
        sep="\t",
        header=None,
        names=["pathway_id", "pathway_name", "species"],
    )
    # Relations: parent \t child
    relations = pd.read_csv(
        files["relations"], sep="\t", header=None, names=["parent", "child"]
    )
    # Mappings: UniProt \t ReactomeID \t URL \t EvidenceCode \t Species \t PathwayName
    uniprot_map = pd.read_csv(
        files["uniprot_map"],
        sep="\t",
        header=None,
        names=["uniprot", "pathway_id", "url", "evidence", "species", "pathway_name"],
    )
    # Normalize human-only (most common for downstream)
    pathways_h = pathways[pathways["species"].str.lower().eq("homo sapiens")]
    uniprot_h = uniprot_map[uniprot_map["species"].str.lower().eq("homo sapiens")]

    return {"pathways": pathways_h, "relations": relations, "uniprot_map": uniprot_h}


# ---- Build orchestration -----------------------------------------------------


def build_kidney_kg(offline: bool = False) -> Path:
    # honor env var too
    offline = offline or os.getenv("KG_OFFLINE") == "1"

    # 1) OmniPath
    print("[kg] Pulling OmniPath…")
    op_inter = _pull_omnipath_interactions(offline=offline)
    op_intercell = _pull_omnipath_intercell(offline=offline)

    if not op_inter.empty:
        op_inter.to_parquet(RELEASE / "omnipath_interactions.parquet", index=False)
        print(
            f"[kg] wrote {RELEASE/'omnipath_interactions.parquet'} ({len(op_inter):,} rows)"
        )
    else:
        print("[kg] skipped writing OmniPath interactions (empty)")

    if not op_intercell.empty:
        op_intercell.to_parquet(RELEASE / "omnipath_intercell.parquet", index=False)
        print(
            f"[kg] wrote {RELEASE/'omnipath_intercell.parquet'} ({len(op_intercell):,} rows)"
        )
    else:
        print("[kg] skipped writing OmniPath intercell (empty)")

    # 2) CellPhoneDB (local) … (keep your existing CPDB v5 loader)
    print("[kg] Loading CellPhoneDB (local)…")
    cpdb_lr = _load_cpdb_lr()
    cpdb_lr.to_parquet(RELEASE / "cellphonedb_lr.parquet", index=False)
    print(f"[kg] wrote {RELEASE/'cellphonedb_lr.parquet'} ({len(cpdb_lr):,} rows)")

    # 3) Reactome: ensure downloads (unless offline and already present), then cache as parquet
    print("[kg] Ensuring Reactome core files…")
    try:
        files = _ensure_reactome()
        rx = _load_reactome(files)
        # Write filtered human pathways and full relations/uniprot map as parquet
        if not rx["pathways"].empty:
            rx["pathways"].to_parquet(
                RELEASE / "reactome_pathways.parquet", index=False
            )
            print(
                f"[kg] wrote {RELEASE/'reactome_pathways.parquet'} ({len(rx['pathways']):,} rows)"
            )
        if not rx["relations"].empty:
            rx["relations"].to_parquet(
                RELEASE / "reactome_relations.parquet", index=False
            )
            print(
                f"[kg] wrote {RELEASE/'reactome_relations.parquet'} ({len(rx['relations']):,} rows)"
            )
        if not rx["uniprot_map"].empty:
            rx["uniprot_map"].to_parquet(
                RELEASE / "reactome_uniprot_map.parquet", index=False
            )
            print(
                f"[kg] wrote {RELEASE/'reactome_uniprot_map.parquet'} ({len(rx['uniprot_map']):,} rows)"
            )
    except Exception as e:
        print(f"[kg][warn] Reactome step failed or offline: {e}")

    # 4) Unified edge set: only concat what we have
    edges = []
    if not op_inter.empty:
        edges.append(
            pd.DataFrame(
                {
                    "source_gene": op_inter.get(
                        "source_gene", pd.Series(dtype=str)
                    ).astype(str),
                    "target_gene": op_inter.get(
                        "target_gene", pd.Series(dtype=str)
                    ).astype(str),
                    "edge_type": "interaction",
                    "source": "omnipath",
                }
            )
        )
    if not cpdb_lr.empty:
        edges.append(
            pd.DataFrame(
                {
                    "source_gene": cpdb_lr.get(
                        "ligand_gene", pd.Series(dtype=str)
                    ).astype(str),
                    "target_gene": cpdb_lr.get(
                        "receptor_gene", pd.Series(dtype=str)
                    ).astype(str),
                    "edge_type": "ligand_receptor",
                    "source": "cellphonedb",
                }
            )
        )

    if edges:
        unified = (
            pd.concat(edges, ignore_index=True)
            .dropna(subset=["source_gene", "target_gene"])
            .drop_duplicates()
        )
    else:
        unified = pd.DataFrame(
            columns=["source_gene", "target_gene", "edge_type", "source"]
        )

    unified.to_parquet(RELEASE / "kg_edges.parquet", index=False)
    print(f"[kg] wrote {RELEASE/'kg_edges.parquet'} ({len(unified):,} rows)")

    return RELEASE
