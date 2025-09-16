#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
import urllib.request


API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _get(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


@dataclass
class Project:
    accession: str
    title: str
    description: str
    organisms: list[str]
    tissues: list[str]
    diseases: list[str]
    publicationDate: str | None

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Project":
        return Project(
            accession=d.get("accession", ""),
            title=d.get("title", ""),
            description=d.get("projectDescription", ""),
            organisms=[
                x.get("name", "") if isinstance(x, dict) else str(x)
                for x in d.get("organisms", [])
            ],
            tissues=[
                x.get("name", "") if isinstance(x, dict) else str(x)
                for x in d.get("tissues", [])
            ],
            diseases=[
                x.get("name", "") if isinstance(x, dict) else str(x)
                for x in d.get("diseases", [])
            ],
            publicationDate=d.get("publicationDate"),
        )

    def text(self) -> str:
        return " ".join(
            [
                self.title,
                self.description,
                " ".join(self.organisms),
                " ".join(self.tissues),
                " ".join(self.diseases),
            ]
        )


def search_pride(
    keywords: list[str], page_size: int = 100, max_pages: int = 10
) -> list[Project]:
    results: dict[str, Project] = {}
    for kw in keywords:
        for page in range(0, max_pages):
            q = {"keyword": kw, "pageSize": str(page_size), "page": str(page)}
            url = API_BASE + "?" + urlencode(q)
            try:
                data = _get(url)
            except Exception as e:
                print(f"[pride] query failed for '{kw}' page {page}: {e}")
                break
            # API may return a dict with 'list' or a raw list
            if isinstance(data, dict):
                items = data.get("list", []) or data.get("_embedded", {}).get(
                    "projects", []
                )
            elif isinstance(data, list):
                items = data
            else:
                items = []
            if not items:
                break
            for it in items:
                p = Project.from_json(it)
                if p.accession and p.accession not in results:
                    results[p.accession] = p
    return list(results.values())


def score_project(
    p: Project, inc_terms: list[str], tissue_terms: list[str], require_human: bool
) -> int:
    text = p.text().lower()
    score = 0
    # keyword matches
    for t in inc_terms:
        if t in text:
            score += 2
    # tissue matches
    for t in tissue_terms:
        if t in text:
            score += 1
    # organism preference
    if require_human:
        if any(
            "homo sapiens" in o.lower() or "human" in o.lower() for o in p.organisms
        ):
            score += 2
        else:
            score -= 2
    return score


def matches_filters(
    p: Project, disease_terms: list[str], tissue_terms: list[str], require_human: bool
) -> bool:
    text = p.text().lower()
    if require_human and not any(
        "homo sapiens" in o.lower() or "human" in o.lower() for o in p.organisms
    ):
        return False
    if not any(t in text for t in tissue_terms):
        return False
    if not any(d in text for d in disease_terms):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="artifacts/pride_candidates.csv",
        help="Output CSV of candidate studies",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=[
            # Disease/clinical synonyms and nearby phrases
            "acute kidney injury",
            "aki",
            "acute renal failure",
            "renal failure",
            "kidney injury",
            "sepsis",
            "septic",
            "septic shock",
            "septicemia",
            "bacteremia",
            "urosepsis",
            "systemic inflammatory response",
            "sirs",
            # Broader renal disease catch-alls
            "kidney disease",
            "renal disease",
            "nephropathy",
            "glomerulonephritis",
            "tubular injury",
            # Mechanistic/surrogate phrases
            "ischemia reperfusion",
            "iri",
            "acute tubular necrosis",
            "atn",
            "cisplatin nephropathy",
        ],
        help="Keyword list to search across PRIDE metadata",
    )
    ap.add_argument(
        "--tissues",
        nargs="*",
        default=[
            "kidney",
            "renal",
            "renal cortex",
            "renal medulla",
            "urine",
            "plasma",
            "serum",
        ],
        help="Tissue/sample terms to boost",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Require both tissue and disease term matches (and human if --human)",
    )
    ap.add_argument(
        "--human", action="store_true", help="Prefer human studies in scoring"
    )
    ap.add_argument("--top", type=int, default=50, help="Print preview top-N by score")
    args = ap.parse_args()

    print("[pride] Searching PRIDE for keywords:", ", ".join(args.include))
    projects = search_pride(args.include)
    print(f"[pride] Retrieved {len(projects)} unique projects")

    inc_l = [t.lower() for t in args.include]
    tis_l = [t.lower() for t in args.tissues]
    candidates = []
    for p in projects:
        if args.strict and not matches_filters(p, inc_l, tis_l, args.human):
            continue
        s = score_project(p, inc_l, tis_l, args.human)
        candidates.append((s, p))
    candidates.sort(key=lambda x: x[0], reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "accession",
                "score",
                "title",
                "organisms",
                "tissues",
                "diseases",
                "publicationDate",
            ]
        )
        for s, p in candidates:
            w.writerow(
                [
                    p.accession,
                    s,
                    p.title,
                    ";".join(p.organisms),
                    ";".join(p.tissues),
                    ";".join(p.diseases),
                    p.publicationDate or "",
                ]
            )
        print(f"[pride] Wrote candidates -> {out}")
    print()
    print("Top suggestions:")
    for s, p in candidates[: args.top]:
        print(f"- {p.accession} (score {s}) :: {p.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
