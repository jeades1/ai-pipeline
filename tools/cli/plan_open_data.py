#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import yaml


def main():
    cfg = Path("conf/datasets/open_access.yaml")
    if not cfg.exists():
        print("No conf/datasets/open_access.yaml found")
        return
    data = yaml.safe_load(cfg.read_text())
    ds = data.get("datasets", [])
    print("=== Open Access Datasets — Acquisition Plan ===")
    for d in ds:
        print(f"- {d['id']}: {d['name']}")
        print(f"  url: {d['url']}")
        print(f"  local: {d['local']}")
        print(f"  purpose: {d['purpose']}")
        print(f"  license: {d.get('license','unknown')}")
        print()
    print("Next steps:")
    print("  1) Download to the listed local paths (create folders as needed)")
    print(
        "  2) For GEO DE tables, drop CSV/TSV under data/external/geo/deg; the loader will pick them up"
    )
    print(
        "  3) Once files are present: run 'make demo' (rebuild assoc), then 'make bench-run' (metrics)"
    )
    print(
        "  4) For LINCS/ENCODE/HPA/GTEx, I can add light parsers to map to gene/pathway priors on request"
    )


if __name__ == "__main__":
    main()
