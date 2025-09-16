#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd

HTML_TMPL = """<!doctype html>
<meta charset="utf-8">
<title>AKI Benchmark Report</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:960px;margin:40px auto;padding:0 16px;}
h1{margin-bottom:0} .muted{color:#666}
table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border:1px solid #ddd;padding:8px;font-size:14px}
th{background:#fafafa;text-align:left}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#eee;margin-left:8px}
</style>
<h1>AKI Benchmark Report</h1>
<p class="muted">Source: {report_json}</p>
<h2>Summary</h2>
<ul>
  <li>Bench size: <b>{bench_n}</b></li>
  <li>Promoted overlap: <b>{hit_n}</b></li>
  <li>Recall@promoted: <b>{recall:.1%}</b></li>
</ul>
<h2>Matched Benchmarks <span class="badge">{hit_n}</span></h2>
{hits_table}
<h2>Missed Benchmarks <span class="badge">{miss_n}</span></h2>
{miss_table}
"""


def df_to_html(df: pd.DataFrame):
    if df.empty:
        return "<p class='muted'>None</p>"
    return df.to_html(index=False, escape=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rp = Path(args.report_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_out = out_dir / "benchmark_report.html"

    j = json.loads(rp.read_text())
    bench_n = j.get("bench_size", 0)
    hit = pd.DataFrame(j.get("hits", []))
    miss = pd.DataFrame(j.get("misses", []))

    recall = (len(hit) / bench_n) if bench_n else 0.0

    html = HTML_TMPL.format(
        report_json=rp,
        bench_n=bench_n,
        hit_n=len(hit),
        miss_n=len(miss),
        recall=recall,
        hits_table=df_to_html(hit),
        miss_table=df_to_html(miss),
    )
    html_out.write_text(html, encoding="utf-8")
    print(f"[bench-viz] Wrote {html_out}")


if __name__ == "__main__":
    main()
