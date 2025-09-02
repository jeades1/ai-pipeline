#!/usr/bin/env bash
set -euo pipefail

if ! command -v mmdc >/dev/null 2>&1; then
  echo "Mermaid CLI (mmdc) not found."
  echo "Install with: brew install node && npm i -g @mermaid-js/mermaid-cli"
  exit 1
fi

mkdir -p docs/diagrams/exports

shopt -s nullglob
for f in docs/diagrams/*.mmd; do
  base="$(basename "$f" .mmd)"
  tmp="${TMPDIR:-/tmp}/${base}.sanitized.mmd"
  # Strip Markdown code fences if present; otherwise pass through unchanged
  awk 'BEGIN{on=1} /^```/{on=1-on; next} {if(on) print}' "$f" > "$tmp"
  if [ ! -s "$tmp" ]; then
    cp "$f" "$tmp"
  fi
  echo "Rendering $f -> docs/diagrams/exports/${base}.svg"
  mmdc -i "$tmp" -o "docs/diagrams/exports/${base}.svg"
  mmdc -i "$tmp" -o "docs/diagrams/exports/${base}.png"
done
