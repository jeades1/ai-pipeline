# ingest/clinical_mimic.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

MIMIC_DIR = Path("data/mimic")


def _load_csv_gz(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", usecols=usecols)


def label_aki_by_kdigo() -> pd.DataFrame:
    """
    Very simple placeholder: detect AKI vs non-AKI using creatinine trajectories.
    Expects:
      - hosp/admissions.csv.gz
      - icu/icustays.csv.gz
      - hosp/labevents.csv.gz (with itemid for creatinine)
    Returns frame with columns: ['subject_id','hadm_id','icustay_id','aki_label']
    """
    if not MIMIC_DIR.exists():
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "icustay_id", "aki_label"]
        )

    try:
        labs = _load_csv_gz(
            MIMIC_DIR / "hosp" / "labevents.csv.gz",
            usecols=["subject_id", "hadm_id", "itemid", "charttime", "value"],
        )
        icu = _load_csv_gz(
            MIMIC_DIR / "icu" / "icustays.csv.gz",
            usecols=["subject_id", "hadm_id", "icustay_id", "intime", "outtime"],
        )
    except Exception:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "icustay_id", "aki_label"]
        )

    # crude creatinine filter (common ITEMIDs across MIMIC-IV variants; adjust as needed)
    creatinine_itemids = {50912, 1525, 220615, 220587}
    labs = labs[labs["itemid"].isin(creatinine_itemids)].copy()
    # coerce to numeric where possible
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs = labs.dropna(subset=["value"])

    # naive KDIGO-ish: peak creatinine / baseline > 1.5 during ICU stay
    # (replace with your robust implementation later)
    merged = labs.merge(icu, on=["subject_id", "hadm_id"], how="inner")
    grp = merged.groupby(["subject_id", "hadm_id", "icustay_id"])
    feats = grp["value"].agg(["min", "max"]).reset_index()
    feats["ratio"] = feats["max"] / feats["min"].replace(0, pd.NA)
    feats["aki_label"] = (feats["ratio"] >= 1.5).astype(int)
    return feats[["subject_id", "hadm_id", "icustay_id", "aki_label"]]
