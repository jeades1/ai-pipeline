from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from kg.schema import KGEvidenceGraph
from kg.invitro import attach_invitro


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"donor_id", "model_id"}
    missing = required - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(f"invitro metadata missing required columns: {missing}")
    return df


def load_readouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model_id", "assay_type", "readout_name", "value"}
    missing = required - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(f"invitro readouts missing required columns: {missing}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Attach in vitro models/readouts to Evidence Graph"
    )
    ap.add_argument(
        "--metadata", type=Path, required=True, help="CSV with in vitro model metadata"
    )
    ap.add_argument(
        "--readouts", type=Path, required=False, help="CSV with assay readouts"
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output artifacts directory (kg_dump)",
    )
    ap.add_argument(
        "--context", type=str, default="invitro", help="Context tag for edges"
    )
    args = ap.parse_args()

    md = load_metadata(args.metadata)
    rd = (
        load_readouts(args.readouts)
        if args.readouts and args.readouts.exists()
        else None
    )

    G = KGEvidenceGraph(context=args.context)
    G = attach_invitro(G, md, rd, provenance="invitro")
    G.export_tsv(args.outdir / "kg_dump")
    print(f"[invitro] Wrote KG with in vitro nodes/edges to {args.outdir}/kg_dump")


if __name__ == "__main__":
    main()
