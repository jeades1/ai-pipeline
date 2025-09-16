# Platform-first Positioning (Disease-agnostic)

This platform integrates knowledge graphs, causal modeling, learning-to-rank, and advanced in vitro models (cells, organoids, MPS, co-cultures) to support ANY disease area via thin, declarative adapters.

- Core: KG + Causal + LTR + Uncertainty + Closed-loop feedback
- In vitro coupling: human-relevant, high-SNR signals to validate and improve hypotheses quickly
- Disease adapters: small JSON configs that set ontology anchors, signal weights, assay preferences, and outcomes
- Products: biomarker discovery, personalized testing, clinical tools, safety/tox, APIs/data products

Adapter configs live in `conf/platform/adapters/*.json` and are loaded by `src/platform/adapters.py`.

Generated visuals in this folder:
- `conceptual_platform_kg.png` – disease-agnostic conceptual graph
- `platform_architecture_overview.png` – layers + disease adapters
- `capabilities_matrix.png` – capability × product grid
- `spinout_pathways.png` – spinout map from the core platform

AKI is retained only as an example use case under `artifacts/pitch/`. 
