## Ingestion Summary

Current loaders and schemas:

- Transcriptomics (GEO): `ingest/geo_deg_loader.py`
  - Inputs: CSV/TSV DEG tables (varied schemas), searched under `data/external/geo/deg/`.
  - Output: DataFrame with columns: gene, effect_size (log2FC), p_value, dataset, direction.

- Knowledge Graph build: `kg/build_graph.py`
  - Adds associative edges feature→phenotype; causal hints (CellPhoneDB receptor→TF, OmniPath); cross-level links (Gene→Protein→Pathway); Reactome hierarchy; GMT fallback Gene→Pathway membership.

- Clinical (optional): `ingest/clinical_mimic.py`
  - Best-effort KDIGO-ish labels if MIMIC is available. Can be replaced with your cohort code.

Data relationships:
- Gene symbols (HGNC) can encode UniProt proteins; proteins participate in pathways (Reactome); genes can be members of pathways (GMT fallback). Ligand–receptor and receptor→TF edges enable gene→protein→TF→pathway causal chains when data is present.

Recommended additions:
- Proteomics loader (PRIDE/MassIVE), metabolomics (MetaboLights), single-cell (GEO/SCP), and clinical cohort ETL templates.