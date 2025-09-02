#!/usr/bin/env bash
set -euo pipefail
outdir="docs/diagrams/exports"
mkdir -p "$outdir"
if ! command -v mmdc >/dev/null; then
  echo "Mermaid CLI (mmdc) not found. Install with: npm i -g @mermaid-js/mermaid-cli"
  exit 1
fi
for f in docs/diagrams/*.mmd; do
  base="$(basename "$f" .mmd)"
  echo "Rendering $f -> $outdir/$base.svg"
  mmdc -i "$f" -o "$outdir/$base.svg"
  mmdc -i "$f" -o "$outdir/$base.png"
done
echo "Done."