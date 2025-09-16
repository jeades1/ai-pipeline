from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from jinja2 import Template

DEFAULT_TEMPLATE = """# {{ title }}

**Subjects:** {{ n_subjects }}  
**Positive label prevalence:** {{ label_rate|round(3) }}  
**Mean last creatinine (mg/dL):** {{ cr_mean|round(3) }}  

Top 10 subjects by last creatinine:

| subject_id | last_timestamp | creatinine_mg_dL | label |
|---:|:---:|:---:|:---:|
{% for r in top_rows -%}
| {{ r.subject_id }} | {{ r.timestamp }} | {{ "%.3f"|format(r.creatinine_mg_dL) }} | {{ r.label }} |
{% endfor %}

---

_This is a lightweight report generated from open-access or fallback CSV data.
Replace this template or expand the pipeline to render richer dashboards._
"""


def build_report(
    features_path: Path,
    output_md: Path,
    artifacts_dir: Optional[Path] = None,
    title: str = "Summary Report",
    template_str: Optional[str] = None,
) -> Path:
    features = pd.read_parquet(features_path)
    n_subjects = features["subject_id"].nunique()
    # Prefer a generic 'label' column; fallback to 'aki_flag' for backward compatibility
    if "label" in features.columns:
        label_rate = float(features["label"].mean())
    elif "aki_flag" in features.columns:
        label_rate = float(features["aki_flag"].mean())
    else:
        label_rate = 0.0
    cr_mean = features["creatinine_mg_dL"].mean()

    top_rows = (
        features.sort_values("creatinine_mg_dL", ascending=False)
        .head(10)
        .assign(timestamp=lambda d: d["timestamp"].astype(str))
        .to_dict(orient="records")
    )

    tpl = Template(template_str or DEFAULT_TEMPLATE)
    md = tpl.render(
        title=title,
        n_subjects=n_subjects,
        label_rate=label_rate,
        cr_mean=cr_mean,
        top_rows=top_rows,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md)
    return output_md
