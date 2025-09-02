SHELL := /bin/bash

.PHONY: render-diagrams repo-tree update-readme-tree setup

# Render Mermaid .mmd files in docs/diagrams/ -> docs/diagrams/exports
render-diagrams:
	@bash scripts/render_diagrams_make.sh
# Print current repo tree (markdown)
repo-tree:
	@python scripts/update_repo_tree.py --print-only

# Update README repo layout section between TREE markers
update-readme-tree:
	@python scripts/update_repo_tree.py --readme README.md --inplace

# Convenience: create venv and install python deps
setup:
	python -m venv .venv && source .venv/bin/activate && pip install -r env/requirements.txt