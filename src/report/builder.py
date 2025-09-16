from pathlib import Path
import pandas as pd


def build_report(
    features_parquet: str = "data/processed/features.parquet",
    labs_parquet: str = "data/working/labs.parquet",
    out_md: str = "reports/run_report.md",
) -> str:
    """
    Create a lightweight markdown report with basic dataset stats.
    """
    features_p = Path(features_parquet)
    labs_p = Path(labs_parquet)
    out_p = Path(out_md)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df_labs = pd.read_parquet(labs_p) if labs_p.exists() else pd.DataFrame()
    df_feat = pd.read_parquet(features_p) if features_p.exists() else pd.DataFrame()

    n_rows_labs = len(df_labs)
    n_rows_feat = len(df_feat)
    n_subjects = (
        df_feat["subject_id"].nunique() if "subject_id" in df_feat.columns else 0
    )
    label_col = next(
        (c for c in ("label", "y", "target") if c in df_feat.columns), None
    )
    pos = int(df_feat[label_col].sum()) if label_col else 0

    lines = [
        "# Run Report",
        "",
        "## Overview",
        f"- Labs rows: **{n_rows_labs}**",
        f"- Feature rows: **{n_rows_feat}**",
        f"- Subjects: **{n_subjects}**",
    ]
    if label_col:
        lines.append(f"- Positive labels ({label_col}): **{pos}**")

    # Show a small schema preview
    if not df_feat.empty:
        sample_cols = list(df_feat.columns)[:20]
        lines += [
            "",
            "## Feature columns (first 20)",
            "",
            "```\n" + ", ".join(sample_cols) + "\n```",
        ]

    out_p.write_text("\n".join(lines))
    return str(out_p)
