# reports/make_cards.py
from pathlib import Path
from typing import List, Dict
from jinja2 import Template

BIOMARKER_TMPL = Template(
    """
# {{ name }} — Biomarker Card
**Layer:** {{ layer }} · **Type:** {{ type }}  
**Assoc score:** {{ assoc_score | round(3) }} · **Effect size:** {{ effect_size | round(3) }} · **p-value:** {{ "%.2e" | format(p_value) }}

## Causal Support
- Level: {{ causal_support.level }}
- Source: {{ causal_support.source or "—" }}
- Details: {{ causal_support.details or "—" }}

## Provenance
- Datasets: {{ provenance }}
""".strip()
)

EXPERIMENT_TMPL = Template(
    """
# {{ name }} — Experiment Card
**ID:** {{ id }}  
**Effector:** {{ effector }}  
**Targets:** {{ targets | join(", ") }}  
**Readouts:** {{ readouts | join(", ") }}  
**Estimated cost:** {{ cost }} · **Duration (days):** {{ duration_days }}

## Value of Information
- Expected VoI (relative): {{ voi | round(3) }}
- Identifiability gain (targets covered): {{ identifiability_gain }}
- Overall score (rank): {{ score | round(3) }}
""".strip()
)


def write_biomarker_cards(ranked: List[Dict], kg, outdir: Path):
    for c in ranked:
        p = outdir / f"{c['name'].replace('/', '_')}.md"
        p.write_text(BIOMARKER_TMPL.render(**c))


def write_experiment_cards(exps: List[Dict], kg, outdir: Path):
    for i, ex in enumerate(exps, 1):
        p = outdir / f"{i:02d}_{ex['id']}.md"
        p.write_text(EXPERIMENT_TMPL.render(**ex))
