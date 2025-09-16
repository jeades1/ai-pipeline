#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def load_reports(bench_dir: Path = Path("artifacts/bench")) -> pd.DataFrame:
    rows = []
    # Known default file written by compute_benchmark.py
    jf = bench_dir / "benchmark_report.json"
    if jf.exists():
        try:
            j = json.loads(jf.read_text())
            rows.append(j)
        except Exception:
            pass
    # Future: iterate multiple disease reports in the same folder
    return pd.DataFrame(rows)


def render_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<html><body><h2>No benchmark reports found.</h2></body></html>"
    cols = ["disease", "precision", "recall", "f1"]
    keep = [c for c in cols if c in df.columns]
    table_html = df[keep].to_html(index=False)
    return f"""
    <html>
      <head><title>Cross-Disease Benchmark Dashboard</title></head>
      <body>
        <h2>Cross-Disease Benchmark Dashboard</h2>
        {table_html}
      </body>
    </html>
    """


def main() -> None:
    out = Path("artifacts/bench/cross_disease_dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = load_reports(out.parent)
    out.write_text(render_html(df))
    print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()
