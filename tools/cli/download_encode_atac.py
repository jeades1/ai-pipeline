#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
import urllib.request
import shutil

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="data/external/encode/kidney_atac", help="Output directory"
    )
    ap.add_argument("--term", default="Kidney", help="Organ slims term (e.g., Kidney)")
    ap.add_argument("--assay", default="ATAC-seq", help="Assay title")
    ap.add_argument("--limit", type=int, default=1000, help="Max experiments to fetch")
    args = ap.parse_args()

    outdir = Path(args.out)
    n = 0
    base = "https://www.encodeproject.org/search/"
    terms = [args.term]
    seen: set[str] = set()

    # Prefer File search for narrowPeak beds
    for term in terms:
        q = {
            "type": "File",
            "assay_slims": "DNA accessibility",
            "assay_title": args.assay,
            "status": "released",
            "assembly": "GRCh38",
            "biosample_ontology.organ_slims": term,
            "file_format": "bed",
            "file_format_type": "narrowPeak",
            "output_type": "peaks",
            "format": "json",
            "limit": str(args.limit),
        }
        url = base + "?" + urlencode(q)
        try:
            data = _get(url)
        except Exception as e:
            print(f"[encode] query failed for term='{term}': {e}")
            continue
        for f in data.get("@graph", []):
            href = f.get("href") or ""
            if not href or not href.endswith((".narrowPeak.gz", ".narrowPeak")):
                continue
            if href in seen:
                continue
            seen.add(href)
            file_url = "https://www.encodeproject.org" + href
            dst = outdir / Path(href).name
            print(f"[encode] {file_url} -> {dst}")
            _download(file_url, dst)
            n += 1

    print(f"[encode] Downloaded {n} narrowPeak files to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
