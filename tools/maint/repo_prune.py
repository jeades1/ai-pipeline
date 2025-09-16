#!/usr/bin/env python3
"""
Conservative repo sweeper:
- Seeds: Makefile, README.md, requirements.txt, main.py
- Parses Makefile for referenced files (python scripts, data, plots)
- Walks Python import graph (in-repo modules -> .py files)
- Greps shell scripts for repo-relative calls
- Scans Markdown for local image/file references
Outputs candidate-orphans to stdout and writes tools/maint/repo_prune_report.txt
Use --delete to remove candidates (still conservative).
"""
import argparse
import os
import re
import sys
from pathlib import Path
from collections import deque

REPO = Path(__file__).resolve().parents[2]  # tools/maint/ -> repo root
IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}
TEXT_EXT = {".md", ".mmd", ".txt", ".rst"}
CODE_EXT = {".py", ".sh", ".mk", ".make"}
DATA_EXT = {".tsv", ".csv", ".json", ".yml", ".yaml"}
ASSET_EXT = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf"}

SEED_FILES = {
    "README.md",
    "requirements.txt",
    "Makefile",
    "main.py",
    # known entrypoints/kept tools
    "tools/bench/compare_benchmarks.py",
    "tools/bench/build_ranked.py",
    "tools/plots/kg_config.py",
    "tools/plots/bench_plots.py",
    "tools/plots/kg_schema.py",
    "tools/curate/export_biomarker_cards.py",
    "tools/curate/export_experiment_cards.py",
    "tools/plots/__init__.py",
    "tools/bench/__init__.py",
    "tools/curate/__init__.py",
    # data we know you’re using
    "data/benchmarks/sepsis_aki_biomarkers.tsv",
    "data/benchmarks/sepsis_aki_biomarkers.norm.tsv",
}

PY_IMPORT_RE = re.compile(
    r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import|^\s*import\s+([a-zA-Z0-9_\.]+)",
    re.MULTILINE,
)
SHELL_REF_RE = re.compile(
    r"\bpython\s+([^\s]+\.py)|\bsh\s+([^\s]+\.sh)|\bbash\s+([^\s]+\.sh)"
)
MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")  # [text](path)


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def list_repo_files():
    files = []
    for root, dirs, fnames in os.walk(REPO):
        rootp = Path(root)
        # prune ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in fnames:
            if f.startswith(".DS_Store"):
                continue
            files.append(rootp / f)
    return files


def parse_makefile_refs(text: str):
    refs = set()
    # naive: grab paths that look like repo files
    for m in re.finditer(
        r"(?P<p>(?:[A-Za-z0-9_\-./]+)\.(?:py|sh|tsv|csv|png|jpg|jpeg|svg|mmd|md))", text
    ):
        p = m.group("p")
        if (REPO / p).exists():
            refs.add(p)
    return refs


def python_deps(py_path: Path):
    deps = set()
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return deps
    # direct file refs (open/read_csv/etc.)
    for m in re.finditer(
        r'["\']([^"\']+\.(?:tsv|csv|json|png|jpg|jpeg|svg|mmd|md))["\']', src
    ):
        maybe = m.group(1)
        if (REPO / maybe).exists():
            deps.add(maybe)

    # build module->file mapping on the fly (basic)
    for mm in re.finditer(PY_IMPORT_RE, src):
        mod = (mm.group(1) or mm.group(2) or "").strip()
        if not mod:
            continue
        # only consider in-repo modules (tools.*, kg.*, etc.)
        if "." in mod:
            parts = mod.split(".")
            # try resolve to a file under repo
            candidate = REPO.joinpath(*parts)  # package dir?
            if (candidate.with_suffix(".py")).exists():
                deps.add(rel(candidate.with_suffix(".py")))
            elif candidate.is_dir():
                init = candidate / "__init__.py"
                if init.exists():
                    deps.add(rel(init))
        else:
            # top-level module in repo?
            cand = REPO / f"{mod}.py"
            if cand.exists():
                deps.add(rel(cand))
    return deps


def shell_refs(sh_path: Path):
    refs = set()
    try:
        text = sh_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return refs
    for m in re.finditer(SHELL_REF_RE, text):
        for g in m.groups():
            if not g:
                continue
            p = g.strip()
            if (REPO / p).exists():
                refs.add(p)
    # also catch quoted file assets
    for m in re.finditer(
        r'["\']([^"\']+\.(?:tsv|csv|json|png|jpg|jpeg|svg|mmd|md|py|sh))["\']', text
    ):
        p = m.group(1)
        if (REPO / p).exists():
            refs.add(p)
    return refs


def md_refs(md_path: Path):
    refs = set()
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return refs
    for m in re.finditer(MD_LINK_RE, text):
        p = m.group(1)
        if p.startswith(("http://", "https://")):
            continue
        full = (md_path.parent / p).resolve()
        if full.exists() and REPO in full.parents:
            refs.add(rel(full))
    return refs


def build_graph(all_files):
    # start with seeds that actually exist
    seeds = {f for f in SEED_FILES if (REPO / f).exists()}
    # include Makefile if present
    mk = REPO / "Makefile"
    if mk.exists():
        seeds.add("Makefile")

    used = set(seeds)
    q = deque(seeds)

    def add_ref(pathstr: str):
        p = REPO / pathstr
        if p.exists():
            rp = rel(p)
            if rp not in used:
                used.add(rp)
                q.append(rp)

    while q:
        current = REPO / q.popleft()
        ext = current.suffix.lower()
        # Makefile
        if current.name == "Makefile":
            refs = parse_makefile_refs(
                current.read_text(encoding="utf-8", errors="ignore")
            )
            for r in refs:
                add_ref(r)
        # Python
        elif ext == ".py":
            for r in python_deps(current):
                add_ref(r)
        # Shell
        elif ext == ".sh":
            for r in shell_refs(current):
                add_ref(r)
        # Markdown / Mermaid / docs
        elif ext in TEXT_EXT | {".mmd"}:
            for r in md_refs(current):
                add_ref(r)
        else:
            # no expansion for data/assets
            pass

    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Delete candidate unused files (irreversible).",
    )
    ap.add_argument(
        "--keep-extra",
        nargs="*",
        default=[],
        help="Additional files/dirs to keep (globs allowed).",
    )
    args = ap.parse_args()

    all_files = [rel(p) for p in list_repo_files()]
    used = build_graph(all_files)

    # Always keep top-level config/data directories
    ALWAYS_KEEP_DIRS = {"data", "tools", "demo", "kg", "recommender"}
    # Remove files only if NOT used and not in keep dirs
    candidates = []
    for f in all_files:
        p = Path(f)
        if any(part in IGNORE_DIRS for part in p.parts):  # already pruned
            continue
        if f in used:
            continue
        if p.parts and p.parts[0] in ALWAYS_KEEP_DIRS:
            # keep unknowns in these roots unless explicitly unused assets (e.g., old PNG/SVG not referenced)
            if p.suffix.lower() in ASSET_EXT | {".mmd"}:
                candidates.append(f)  # allow pruning stale diagrams/assets
            continue
        # keep repo meta
        if p.name in {"LICENSE", "pyproject.toml", "setup.cfg", "setup.py"}:
            continue
        candidates.append(f)

    report = REPO / "tools" / "maint" / "repo_prune_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(sorted(candidates)))
    print(f"[prune] Used files: {len(used)}")
    print(f"[prune] Candidates to remove: {len(candidates)}")
    print(f"[prune] Written list -> {rel(report)}")

    if args.delete:
        for f in candidates:
            try:
                (REPO / f).unlink()
                print(f"deleted {f}")
            except IsADirectoryError:
                # skip dirs
                pass
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    sys.exit(main())
