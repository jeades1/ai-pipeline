SHELL := /bin/bash

.PHONY: render-diagrams ingest preprocess features report demo help

help:
	@echo "Targets:"
	@echo "  render-diagrams   Render Mermaid diagrams"
	@echo "  ingest            Ingest AKI demo data"
	@echo "  preprocess        Clean & aggregate data"
	@echo "  features          Build feature table"
	@echo "  report            Create tiny Markdown report"
	@echo "  demo              Run end-to-end demo"

render-diagrams:
	./render_diagrams_make.sh

ingest:
	python -m tools.cli.main ingest

preprocess:
	python -m tools.cli.main preprocess

features:
	python -m tools.cli.main features

report:
	python -m tools.cli.main report

demo:
	python -m tools.cli.main demo

.PHONY: data-setup kg lincs mimic demo-full
data-setup:
	python -m tools.cli.main ingest  # keep your tiny CSV demo too
kg:
	python -m tools.cli.main kg_build
lincs:
	python -m tools.cli.main lincs_reversal
mimic:
	python -m tools.cli.main mimic_prepare
demo-full:
	make kg && make lincs && make mimic && python -m tools.cli.main report

.PHONY: kg
kg:
	python -m tools.cli.main kg_build

update-readme-tree:
	@echo "[update-readme-tree] stub"