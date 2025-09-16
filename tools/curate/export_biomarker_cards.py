#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export concise biomarker cards with real evidence.
Inputs:
  artifacts/promoted.tsv   -> gene list (name/layer/type)
  artifacts/assoc.tsv      -> effect_size, p_value, dataset, direction
Outputs:
  artifacts/deck/biomarkers/<GENE>.md
"""
from pathlib import Path
import pandas as pd

ART = Path("artifacts")
DECK = ART / "deck" / "biomarkers"
DECK.mkdir(parents=True, exist_ok=True)

prom = pd.read_csv(ART / "promoted.tsv", sep="\t")
assoc_path = ART / "assoc.tsv"
assoc = pd.read_csv(assoc_path, sep="\t") if assoc_path.exists() else pd.DataFrame()

# If you want to pin to a subset, edit here:
genes = sorted(set(prom.loc[prom["layer"] == "transcriptomic", "name"]))


# Add evidence if available
def evidence_for(gene):
    if assoc.empty:
        return pd.DataFrame()
    sub = assoc[assoc["feature"].astype(str).str.upper() == str(gene).upper()]
    cols = [
        c
        for c in ["dataset", "effect_size", "p_value", "direction"]
        if c in sub.columns
    ]
    return sub[cols].sort_values("p_value").head(5)


n_written = 0
missing = []
for g in genes:
    ev = evidence_for(g)
    status = (
        "benchmark hit"
        if (
            Path("data/benchmarks/sepsis_aki_biomarkers.tsv").exists()
            and (
                pd.read_csv("data/benchmarks/sepsis_aki_biomarkers.tsv", sep="\t")[
                    "name"
                ]
                .astype(str)
                .str.upper()
                .eq(str(g).upper())
                .any()
            )
        )
        else "candidate"
    )

    lines = [
        f"# {g}",
        "",
        "**What it is:** Candidate biomarker (transcriptomic gene)",
        f"**Status:** {status}",
        "",
    ]
    if not ev.empty:
        lines += [
            "**Evidence snapshot (top 5):**",
            "",
            "| dataset | effect_size | p_value | direction |",
            "| --- | ---: | ---: | --- |",
        ]
        for _, r in ev.iterrows():
            lines.append(
                f"| {r.get('dataset','')} | {r.get('effect_size',''):.3f} | {r.get('p_value',''):.2e} | {r.get('direction','')} |"
            )
        lines.append("")
    else:
        lines += [
            "_No per-gene evidence table found in artifacts/assoc.tsv (demo mode)._",
            "",
        ]

    lines += [
        "**Notes (demo):** Derived from public cohorts; association-only; subject to confounding. Pair with experiment card.",
        "",
    ]
    (DECK / f"{g}.md").write_text("\n".join(lines))
    n_written += 1

print(
    f"[biomarker_export] Genes selected: {len(genes)} | Cards copied: {n_written} -> {DECK}"
)
if missing:
    print("[biomarker_export] Missing cards for:", ", ".join(missing))
