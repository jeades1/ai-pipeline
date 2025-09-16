#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import urllib.request
import shutil


UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.read()


def _download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=chunk)


def scrape_and_download(
    base: str, outdir: Path, pattern: str = r"\.parquet$"
) -> list[Path]:
    html = _get(base).decode("utf-8", errors="ignore")
    # Extract hrefs
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    files = [h for h in hrefs if re.search(pattern, h)]
    # Normalize links
    paths: list[Path] = []
    for f in files:
        url = f if f.startswith("http") else base.rstrip("/") + "/" + f.lstrip("/")
        name = url.split("/")[-1]
        dst = outdir / name
        print(f"[opentargets] {url} -> {dst}")
        _download(url, dst)
        paths.append(dst)
    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        default="https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/output/association_overall_direct/",
        help="Base URL directory to scrape",
    )
    p.add_argument(
        "--out",
        default="data/external/opentargets/25.06/association_overall_direct",
        help="Destination directory",
    )
    p.add_argument(
        "--pattern", default=r"\.parquet$", help="Regex for files to download"
    )
    args = p.parse_args()

    outdir = Path(args.out)
    try:
        paths = scrape_and_download(args.base, outdir, args.pattern)
        print(f"[opentargets] Downloaded {len(paths)} files to {outdir}")
        return 0
    except Exception as e:
        print(f"[opentargets] FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
