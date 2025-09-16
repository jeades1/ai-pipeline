#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import json
import shutil
import os
import urllib.request
import yaml


UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36"
}


def _download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=chunk)


def _content_length(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def main():
    cfg = Path("conf/datasets/open_access.yaml")
    if not cfg.exists():
        print("No conf/datasets/open_access.yaml found", file=sys.stderr)
        sys.exit(2)
    data = yaml.safe_load(cfg.read_text())
    ds = data.get("datasets", [])
    report = {"downloaded": [], "manual": []}
    for d in ds:
        mode = d.get("mode", "manual")
        url = d.get("url")
        local = Path(str(d.get("local")))
        if mode == "auto" and url and local:
            try:
                # Guard against very large downloads unless explicitly allowed
                max_bytes_env = os.getenv("OPEN_DATA_MAX_BYTES")
                max_bytes = (
                    int(max_bytes_env) if max_bytes_env else 2 * 1024 * 1024 * 1024
                )  # 2 GiB default
                allow_large = os.getenv("OPEN_DATA_ALLOW_LARGE") == "1"

                size = _content_length(url)
                if size is not None:
                    print(f"[download] {d['id']} size ~{size/1e6:.1f} MB")
                else:
                    print(f"[download] {d['id']} size unknown (no Content-Length)")

                if size is not None and size > max_bytes and not allow_large:
                    reason = (
                        f"skipped: {size} bytes exceeds limit {max_bytes}. "
                        f"Set OPEN_DATA_ALLOW_LARGE=1 or OPEN_DATA_MAX_BYTES to override"
                    )
                    print(f"[download] {d['id']} {reason}")
                    report["manual"].append(
                        {
                            "id": d.get("id"),
                            "url": url,
                            "local": str(local),
                            "notes": (d.get("notes", "") + " " + reason).strip(),
                        }
                    )
                    continue

                print(f"[download] fetching -> {local}")
                _download(url, local)
                report["downloaded"].append({"id": d["id"], "path": str(local)})
            except Exception as e:
                print(f"[download] FAILED {d['id']}: {e}", file=sys.stderr)
                report["manual"].append(
                    {
                        "id": d.get("id"),
                        "url": url,
                        "local": str(local),
                        "notes": (d.get("notes", "") + f" failed: {e}").strip(),
                    }
                )
        else:
            report["manual"].append(
                {
                    "id": d.get("id"),
                    "url": url,
                    "local": str(local),
                    "notes": d.get("notes", ""),
                }
            )

    out = Path("artifacts/open_data_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[download] Report -> {out}")


if __name__ == "__main__":
    main()
