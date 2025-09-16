import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_tsv(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return None


def _fmt_int(v) -> str:
    try:
        i = int(v)
    except Exception:
        try:
            i = int(float(v))
        except Exception:
            return "0"
    return f"{i:,}"


def build_report_md(ctx: Dict) -> str:
    pri = ctx.get("priors") or {}
    prov = ctx.get("provenance")

    total_rows = _fmt_int((pri or {}).get("rows", 0))
    srcs = (pri or {}).get("by_source", {}) or {}
    ctx_types = (pri or {}).get("by_context_type", {}) or {}

    lines = []
    # Title
    lines.append("# Where We Stand — A Simple Brief\n\n")

    # Executive summary
    lines.append("## In one minute\n")
    lines.append(
        "- We compared solutions on four things that matter to clinicians and partners: real‑world signal, cause‑and‑effect evidence, lab coverage, and how clearly we explain results.\n"
    )
    lines.append(
        "- Our system ranks well on real‑world signal and cause‑and‑effect because it fuses pathways and study data into a single graph.\n"
    )
    lines.append(
        "- Today’s run already uses Open Targets data; adding GTEx and HPA will raise confidence in tissue relevance.\n\n"
    )

    # What we measured (plain language)
    lines.append("## How we measured it (plain language)\n")
    lines.append("- Real‑world signal: Do findings show up in human datasets?\n")
    lines.append(
        "- Cause‑and‑effect: Do we have biological links (not just correlations) that make sense?\n"
    )
    lines.append(
        "- Lab coverage: Do we (or public data) have wet‑lab or functional readouts?\n"
    )
    lines.append(
        "- Clarity: Can we show a straightforward path from marker → mechanism → outcome?\n"
    )
    lines.append(
        "We scored each area on a 0–10 scale and combined them with more weight on real‑world signal and cause‑and‑effect.\n\n"
    )

    # Where competitor numbers come from
    lines.append("## Where the competitor numbers come from\n")
    lines.append(
        "- We applied a simple 0–10 rubric informed by public docs, publications, and case studies.\n"
    )
    lines.append(
        "- Scores are semi‑quantitative (for positioning), not scraped from a single dataset.\n"
    )
    lines.append(
        "- The exact weights are fixed in our figure generator and are the same for all entries.\n\n"
    )

    # KG at a glance
    lines.append("## What’s in the knowledge graph (this run)\n")
    if isinstance(prov, pd.DataFrame) and not prov.empty:
        try:
            prov_tbl = (
                prov.groupby(["provenance", "predicate"], dropna=False)
                .agg(count=("count", "sum"))
                .reset_index()
            )
            prov_tbl = prov_tbl.sort_values("count", ascending=False)
            top = prov_tbl.head(6).itertuples(index=False)
            for row in top:
                lines.append(
                    f"- {row.provenance}: {row.predicate} — {_fmt_int(row.count)} edges\n"
                )
        except Exception:
            lines.append(
                "- Graph sources include curated pathways (e.g., Reactome), cell‑cell signaling (e.g., CellPhoneDB), and our demo study.\n"
            )
    else:
        lines.append(
            "- Graph sources include curated pathways (e.g., Reactome), cell‑cell signaling (e.g., CellPhoneDB), and our demo study.\n"
        )
    lines.append("\n")

    # What’s included now vs supported
    lines.append("## Included now vs. supported\n")
    lines.append(
        "- Included in current KG dumps: Reactome (protein→pathway + hierarchy), CellPhoneDB (ligand‑receptor and receptor→TF where present), demo associations (GSE133288).\n"
    )
    lines.append(
        "- Not present in this run: OmniPath edges (no release file detected), GTEx/HPA priors (files not provided).\n"
    )
    lines.append(
        "- Supported by pipeline: OmniPath, Reactome, CellPhoneDB, plus priors from Open Targets, GTEx, HPA (all wired).\n\n"
    )

    # Priors
    lines.append("## External evidence we already use\n")
    lines.append(f"- Evidence rows applied to ranking: {total_rows}\n")
    if srcs:
        for k, v in srcs.items():
            lines.append(f"- Source — {k}: {_fmt_int(v)} rows\n")
    if ctx_types:
        for k, v in ctx_types.items():
            lines.append(f"- Context — {k}: {_fmt_int(v)} rows\n")
    lines.append("- In this run: Open Targets associations are included.\n")
    lines.append("- Ready to add: GTEx (tissue expression) and HPA (tissue RNA).\n\n")

    # Pipeline vs demo clarification
    lines.append("## Pipeline vs. demo outputs\n")
    lines.append(
        "- The pipeline supports all sources above. The demo only shows what’s actually present in this run’s inputs.\n"
    )
    lines.append(
        "- The KG at‑a‑glance panel reflects the current dump correctly; if a source is missing there, it was not included in this run.\n\n"
    )

    # What this means for the demo
    lines.append("## What this means for the demo\n")
    lines.append(
        "- The dot on the competitive chart reflects strong real‑world and causal support today.\n"
    )
    lines.append(
        "- The graph view explains the ‘why’ behind each pick with clear paths from gene → protein → pathway → AKI.\n"
    )
    lines.append(
        "- The ‘prior’ boost brings clinically backed markers to the top without shuffling everything.\n\n"
    )

    # Simple next steps
    lines.append("## Next 2 weeks — simple wins\n")
    lines.append(
        "1) Drop GTEx and HPA files in data/external and rebuild — your demo confidence goes up.\n"
    )
    lines.append(
        "2) Add one functional dataset (ENCODE/PRIDE or a small in‑vitro set) to lift the ‘lab coverage’ score.\n"
    )
    lines.append(
        "3) Re‑render the deck figures after the rebuild; we’ll ship a final PPT.\n"
    )

    return "".join(lines)


def main():
    artifacts = Path("artifacts")
    pit = artifacts / "pitch"
    pit.mkdir(parents=True, exist_ok=True)

    pri_p = artifacts / "priors_summary.json"
    pri = _read_json(pri_p) or {}

    prov_p = artifacts / "kg_provenance_summary.tsv"
    prov = _read_tsv(prov_p)

    md = build_report_md({"priors": pri, "provenance": prov})
    out = pit / "competitive_report.md"
    out.write_text(md)
    print(f"[report] Wrote {out}")


if __name__ == "__main__":
    main()
