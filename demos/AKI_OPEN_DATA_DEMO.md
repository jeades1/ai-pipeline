```markdown
# Open-Data Demonstration: Acute Kidney Injury (ICU) — Tubular Focus (No New Experiments)

**Goal:** Deliver a working, reproducible demo that (1) builds a kidney-focused open Knowledge Graph (KG), (2) derives **tubular functional modules** from public single-cell/spatial data, (3) uses **LINCS L1000** perturbation signatures to simulate “interventions” (reversal analysis), and (4) validates **clinical predictions** and **biomarker ranking** on a **blinded** slice of a public ICU cohort (**MIMIC-IV**).  
**Why AKI:** High-impact outcomes (RRT, mortality), proximal tubule biology, and rich open datasets.  
**No wet lab required.**

---

## Endpoints
- **Primary:** Probability of **renal replacement therapy (RRT)** within **48–72h** of ICU admission (or AKI onset proxy).  
- **Secondary:** MAKE30 (major adverse kidney events at 30d), in-hospital mortality.

---

## Plan (6–8 weeks)

Mermaid source: `docs/diagrams/demo_gantt.mmd`  
To export: `make render-diagrams` (outputs SVG/PNG to `docs/diagrams/exports/`).

---

## Data Sources (all open)
- **Kidney molecular atlas & injury states:** KPMP (sc/snRNA; spatial) & human kidney atlas references.  
- **Tubulointerstitial transcriptomes:** public GEO sets (e.g., NEPTUNE) for tubular signatures.  
- **Perturbational signatures:** **LINCS L1000** (HA1E kidney epithelial line).  
- **Clinical outcomes:** **MIMIC-IV** (ICU EHR) for AKI staging, RRT procedures, urine output, labs, mortality.  
- **Open KG priors:** **Reactome** (pathways), **OmniPath** (signaling; ligand–receptor), **CellPhoneDB** (curated L-R).  
- **Reference biomarkers (face validity):** **KIM-1/HAVCR1**, **NGAL/LCN2**.

> You may need PhysioNet access for MIMIC-IV. Put credentials/paths in `.env`.

---

## Workstreams & Deliverables

### 1) Open kidney Knowledge Graph (KG)
- **Tasks:** ETL from Reactome/OmniPath/CellPhoneDB; normalize IDs; add direction/sign/context; derive tubular modules from KPMP/atlas (barrier/TEER, transport, injury/repair, mito stress/inflammation); register secretome links.
- **Deliverables:**  
  - `kg/releases/kg_kidney_v1.parquet` (or `.graphml`)  
  - `modeling/modules/tubular_modules_v1.json`

### 2) LINCS L1000 reversal (no new experiments)
- **Tasks:** Build injury signatures; compute HA1E connectivity scores; keep mechanistically plausible reversers per KG; assemble mechanism+biomarker shortlist.
- **Deliverables:**  
  - `modeling/modules/lincs_reversal_scores.csv`  
  - `biomarkers/dossiers/tubule_injury_candidates_v1.md`

### 3) Patient Avatars v0 (data-driven)
- **Concept:** latent **tubular function** mediates creatinine kinetics, urine output, acid–base/electrolytes; regularized by KG priors.
- **Deliverables:**  
  - `modeling/twins/avatar_v0.pkl`  
  - `outputs/avatars/per_stay_latents.parquet`

### 4) Clinical prediction & blinded validation
- **Labels:** KDIGO AKI from SCr/UO; RRT via procedures; mortality/MAKE30 via EHR.  
- **Splits:** time-based or unit-based; freeze code & thresholds; independent scorer holds labels.  
- **Metrics:** AUROC/PR-AUC for RRT@48–72h; Brier + calibration curves; decision-curve; NRI vs baseline.
- **Deliverables:**  
  - `outputs/predictions/test_blinded_scores.csv`  
  - `demos/report_aki_demo_v1.pdf`

---

## Reproducibility: Commands

### Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
cp env/.env.example .env

KG Build
python -m kg.etl.build_open_kidney_kg \
  --reactome data/src/reactome/ \
  --omnipath data/src/omnipath/ \
  --cellphonedb data/src/cellphonedb/ \
  --out kg/releases/kg_kidney_v1.parquet

Tubular Modules
python -m modeling.modules.build_tubular_modules \
  --kpmp data/src/kpmp/ \
  --atlas data/src/kidney_atlas/ \
  --out modeling/modules/tubular_modules_v1.json

LINCS Reversal
python -m modeling.modules.lincs_reversal \
  --lincs data/src/lincs/ha1e/ \
  --modules modeling/modules/tubular_modules_v1.json \
  --kg kg/releases/kg_kidney_v1.parquet \
  --out modeling/modules/lincs_reversal_scores.csv

Avatar v0 Training
python -m modeling.twins.train_avatar_v0 \
  --mimic_path $MIMIC_PATH \
  --out modeling/twins/avatar_v0.pkl \
  --latent_out outputs/avatars/per_stay_latents.parquet

Clinical Models & Blinded Test
python -m modeling.predictors.train_eval \
  --mimic_path $MIMIC_PATH \
  --avatar_latents outputs/avatars/per_stay_latents.parquet \
  --modules modeling/modules/tubular_modules_v1.json \
  --split_strategy time \
  --out outputs/predictions/