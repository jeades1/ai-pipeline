#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

ART = Path("artifacts")
DECK = ART / "deck" / "experiments"
DECK.mkdir(parents=True, exist_ok=True)

prom = pd.read_csv(ART / "promoted.tsv", sep="\t")
genes = sorted(set(prom.loc[prom["layer"] == "transcriptomic", "name"]))

n_written = 0
for g in genes:
    text = f"""# Experiment proposal: validate {g}

**Hypothesis:** {g} differs between AKI vs non-AKI in sepsis and correlates with severity.

**Design:**  
- **Assay:** RT-qPCR or ELISA (depending on availability)  
- **Model:** Existing sepsis biobank samples or ex vivo whole blood stimulation  
- **n / power:** Pilot n≈15–20 per arm (demo), refine with VOI later  
- **Readout:** Log2 fold-change; AUROC for discrimination

**Next step:** Pre-register protocol; collect confounders (age, sex, baseline eGFR).

_(Auto-generated from demo; replace with your preferred design template.)_
"""
    (DECK / f"{g}.md").write_text(text)
    n_written += 1

print(
    f"[experiment_export] Targets: {len(genes)} | Cards copied: {n_written} -> {DECK}"
)
