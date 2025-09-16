# AI-Guided, Closed-Loop Biomarker Discovery Pipeline

> A self-driving discovery system that learns **causal, multi-scale structure** linking **molecular state → multicellular tissue function → clinical outcomes**, and uses that ⸻

## Quick Demo

### 🚀 **5-Minute Demo**
```bash
# Clone and setup
git clone https://github.com/jeades1/ai-pipeline.git
cd ai-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run enhanced validation demo
python test_enhanced_validation.py

# Start production API server
docker-compose up -d
# Visit http://localhost:8000/docs for API documentation

# Run clinical decision support demo
python test_clinical_decision_support.py

# Generate demo visualizations
python presentation/generate_visualizations.py
```

### 📊 **Interactive Demos**
- **Streamlit Web Interface**: `python streamlit_demo.py`
- **Jupyter Notebooks**: Explore `dashboards/notebooks/`
- **API Testing**: Use `biomarkers/test_api_client.py`
- **Validation Reports**: Check `demo_outputs/validation/`

### 🔬 **Scientific Demos**
- **Causal Discovery**: `python biomarkers/causal_scoring.py`
- **Tissue Chip Integration**: `python test_tissue_chip_integration.py`
- **Multi-Outcome Prediction**: `python test_multi_outcome_prediction.py`
- **Real Data Integration**: `python biomarkers/real_data_integration.py`

⸻

## Initial 90-Day MVPerstanding to design the **next experiments**, generate **patient avatars**, and nominate **robust biomarkers**.

**Key principle:** Functional tissue outcomes (barrier integrity, electrophysiology, contractility, transport, perfusion) are enforced as *mesoscale mediators* between omics and clinical endpoints.

---

## 📋 Navigation Guide & Reading Order

### 🚀 **Quick Start (5 minutes)**
1. **[Getting Started](#getting-started)** - Installation and setup
2. **[Quick Demo](#quick-demo)** - Run the demo pipeline
3. **[Key Features](#key-features)** - What this system does

### 📖 **Complete Understanding (30 minutes)**

**Systematic Guide to Biomarker Discovery Platform** - This structured guide covers our integrated computational-experimental approach to biomarker discovery. For medical researchers and clinicians familiar with biomarker validation but new to machine learning and systems biology approaches.

#### **Phase 1: Scientific Foundation** 
*"Core methodology and biological principles"*

4. **[Mission & Scientific Vision](#mission--outcomes)** - Core objectives and experimental approach
   - Understand our systematic approach to mechanistic biomarker discovery
   - Focus on causal relationships between molecular signatures and tissue function
   - Integration of patient-derived models with computational methods

5. **[PACKAGE_OVERVIEW.md](./PACKAGE_OVERVIEW.md)** - Complete system methodology and validation
   - **Essential reading**: Comprehensive overview of computational methods and experimental validation
   - Details our approach compared to current best practices in biomarker discovery
   - Specific focus on clinical translation and regulatory validation

6. **[System Architecture](#system-architecture)** - Integrated experimental-computational pipeline
   - Data flow from tissue-chip experiments through AI analysis to clinical validation
   - Understanding multi-modal data integration (genomics, proteomics, functional assays)
   - Quality control and validation at each processing stage

#### **Phase 2: Methodological Advances** 
*"What makes this approach scientifically rigorous?"*

7. **[Enhanced Validation Framework](./ENHANCED_VALIDATION_SUMMARY.md)** - Statistical methods and bias control
   - **Why this matters**: 90% of biomarker candidates fail in clinical trials due to poor validation
   - Our solution: Multi-layer statistical validation that catches problems early
   - Read this to understand how we achieve higher success rates than traditional approaches

8. **[Production Deployment](./DEPLOYMENT_SUMMARY.md)** - FastAPI, Docker, Kubernetes deployment
   - **Real-world readiness**: This isn't a research prototype - it's a production system ready for hospital deployment
   - Learn about 24/7 operation, automatic scaling, and integration with hospital IT systems
   - Essential for understanding implementation requirements and timelines

9. **[Clinical Integration](./CLINICAL_EXPANSION_SUMMARY.md)** - Clinical decision support and safety
   - **The payoff**: How AI discoveries translate into better patient care
   - See real scenarios: emergency room diagnosis, personalized treatment selection, drug safety monitoring
   - This bridges the gap between technical capabilities and clinical value

#### **Phase 3: Advanced Features - Where Science Meets Innovation**
*"What makes this system uniquely powerful?"*

10. **[Biomarker Discovery Pipeline](#biomarker-discovery-pipeline)** - Core discovery methodology
    - **The heart of the system**: How we automatically find molecular patterns that predict disease
    - Understand the progression from correlation → causation → clinical validation
    - See how our evidence ladder (E0→E5) ensures robust, reliable biomarkers

11. **[Patient Avatars & Personalization](#patient-avatars--personalization)** - Digital twins and personalization
    - **Precision medicine in action**: Creating digital versions of patients to test treatments virtually
    - Learn how we move from "one-size-fits-all" to "personalized for you" medicine
    - Discover how avatars help doctors choose the best treatment with minimal side effects

12. **[Tissue Chip Integration](./docs/tissue_chip_integration_strategy.md)** - In-vitro system integration
    - **The laboratory revolution**: Using "organs-on-chips" to test treatments before trying them on patients
    - See how miniature human tissues provide better predictions than animal testing
    - Understand how AI guides these experiments for maximum learning efficiency

#### **Phase 4: Technical Deep Dive - Under the Hood**
*"How do we achieve these capabilities technically?"*

13. **[Knowledge Graph](#knowledge-graph)** - Causal reasoning infrastructure
    - **The brain of the system**: A vast network of biological knowledge that grows smarter with each experiment
    - Learn how we connect genes → pathways → tissue function → clinical outcomes
    - Understand how this enables mechanistic explanations, not just black-box predictions

14. **[Evidence Ladder & Validation](#evidence-ladder--validation)** - Validation methodology
    - **Quality control on steroids**: Our systematic approach to ensuring discoveries are real and reliable
    - See the progression from simple correlation to prospective clinical validation
    - Critical for understanding why our biomarkers have higher success rates

15. **[In-Vitro Integration Framework](./docs/invitro_integration_framework.md)** - Laboratory integration
    - **Laboratory automation**: How we orchestrate complex biological experiments using AI
    - Learn about automated assay scheduling, quality control, and adaptive experiment design
    - See how physical laboratory work integrates seamlessly with AI decision-making

#### **Narrative Connections: How It All Fits Together**

**The Complete Story**: Our system creates a virtuous cycle where each component enhances the others:

1. **Tissue chip experiments** (Phase 3, #12) generate high-quality biological data
2. **Enhanced validation** (Phase 2, #7) ensures this data produces reliable insights  
3. **Knowledge graphs** (Phase 4, #13) connect these insights to broader biological understanding
4. **Patient avatars** (Phase 3, #11) personalize these insights for individual patients
5. **Clinical integration** (Phase 2, #9) delivers personalized recommendations to doctors
6. **Production deployment** (Phase 2, #8) makes this available 24/7 across health systems
7. **Evidence ladder validation** (Phase 4, #14) continuously improves the system's reliability

**The Innovation**: Unlike traditional approaches that work in isolation, our system creates **emergent intelligence** where the whole becomes greater than the sum of its parts. Each successful biomarker discovery improves the knowledge graph, which improves future discoveries, which improves patient avatars, which improves clinical decisions.

**The Impact**: This creates a compound effect where the system becomes exponentially more valuable over time, ultimately transforming how we discover, validate, and deploy biomarkers for better patient care.

### 🔬 **Scientific & Research Context**
16. **[Scientific Mission](./docs/scientific/mission.md)** - Research objectives and approach
17. **[Research Gaps Analysis](./docs/scientific/research-gaps.md)** - Current limitations and opportunities
18. **[Competitive Analysis](./docs/quantitative_competitive_analysis.md)** - Industry positioning

### 📊 **Performance & Benchmarks**
19. **[Benchmarking Results](./docs/benchmarks.md)** - Performance comparisons
20. **[Industry Standards](./docs/industry_ranking_standards.md)** - Validation against industry metrics
21. **[Methodology & Metrics](./docs/methodology_metrics.md)** - Technical methodology

### 🏥 **Clinical & Translational**
22. **[Next-Generation Biomarkers](./docs/next_win_personalized_biomarkers.md)** - Personalized medicine approach
23. **[Clinical Outcomes Analysis](./docs/capabilities.md)** - Clinical validation capabilities
24. **[Uncertainty Quantification](./docs/uncertainty_methods_industry.md)** - Risk assessment and uncertainty

### 💡 **Innovation & Future**
25. **[Next Computational Wins](./docs/next_computational_wins_analysis.md)** - Future development priorities
26. **[Advanced Metrics Integration](./docs/advanced_metrics_and_integration_summary.md)** - Next-generation analytics
27. **[Ranking Improvement Plan](./docs/ranking_improvement_plan.md)** - Performance optimization roadmap

---

## 🎯 Key Features

### ✅ **Enhanced Validation Framework**
- **Statistical Rigor**: Advanced statistical testing with multiple correction methods
- **Bias Detection**: Systematic bias identification and mitigation
- **Temporal Stability**: Longitudinal validation and stability assessment
- **Clinical Assessment**: Real-world clinical validation metrics
- **📋 Details**: [Enhanced Validation Summary](./ENHANCED_VALIDATION_SUMMARY.md)

### 🚀 **Production-Ready Deployment**
- **FastAPI REST API**: 12+ endpoints with comprehensive documentation
- **Docker Containers**: Production containerization with Docker Compose
- **Kubernetes Support**: Scalable deployment with K8s manifests
- **Monitoring**: Prometheus/Grafana monitoring and alerting
- **📋 Details**: [Deployment Summary](./DEPLOYMENT_SUMMARY.md)

### 🏥 **Clinical Integration**
- **Decision Support**: Real-time clinical decision support system
- **Safety Assessment**: Automated safety and risk evaluation
- **Multi-Outcome Prediction**: Comprehensive outcome forecasting
- **Regulatory Compliance**: FDA/EMA validation framework
- **📋 Details**: [Clinical Expansion Summary](./CLINICAL_EXPANSION_SUMMARY.md)

### 🧬 **Advanced AI & ML**
- **Causal Discovery**: Automated causal inference and discovery
- **Tissue Chip Integration**: In-vitro system AI integration
- **Federated Learning**: Privacy-preserving multi-institutional learning
- **Digital Twins**: Patient-specific simulation models
- **📋 Details**: [Package Overview](./PACKAGE_OVERVIEW.md)

---

## Table of Contents
1. [Mission & Outcomes](#mission--outcomes)  
2. [System Architecture](#system-architecture)  
3. [Physical Testbeds](#physical-testbeds)  
4. [Sensing & Assays](#sensing--assays)  
5. [Data Ingestion & Metadata](#data-ingestion--metadata)  
6. [Storage & Data Model](#storage--data-model)  
7. [Knowledge Graph](#knowledge-graph)  
8. [Modeling & Simulation](#modeling--simulation)  
9. [Decision & Orchestration](#decision--orchestration)  
10. [Biomarker Discovery Pipeline](#biomarker-discovery-pipeline)  
11. [Patient Avatars & Personalization](#patient-avatars--personalization)  
12. [Evidence Ladder & Validation](#evidence-ladder--validation)  
13. [Governance & Privacy](#governance--privacy)  
14. [KPIs & Dashboards](#kpis--dashboards)  
15. [Quick Demo](#quick-demo)  
16. [Repo Layout](#repo-layout)  
17. [Getting Started](#getting-started)  
18. [Additional Resources](#additional-resources)  
19. [Glossary](#glossary)

---

## Mission & Outcomes
**Mission.** Discover mechanistic, translatable biomarkers by actively learning causal structure across **molecule → cell state → tissue function → clinical outcome**, using patient-derived, perfused, multicellular in-vitro systems.

**Success Outcomes**
- Biomarkers with **interventional evidence** on tissue function, **replication** across donors and models, and **clinical mediation** in harmonized cohorts.
- **Patient avatars** that forecast functional responses and support **personalized biomarkers** and **donor-specific treatment simulations**.
- A **closed loop** that continuously improves knowledge (and the KG) with every data cycle.

---

## System Architecture
> The Mermaid sources live in `docs/diagrams/` so they can be re-used across docs and exported as PNG/SVG.

- **Architecture (overview):** `docs/diagrams/architecture_overview.mmd`  
- **Decision sequence:** `docs/diagrams/decision_sequence.mmd`  
- **Avatars flow:** `docs/diagrams/avatars_flow.mmd`  
- **Evidence ladder:** `docs/diagrams/evidence_ladder.mmd`

<img src="docs/diagrams/exports/architecture_overview.svg" width="900" />

To export images (requires Mermaid CLI):  
```bash
make render-diagrams

Physical Testbeds
	•	Systems: perfused patient-derived organoids, multicellular tubular tissues, vascularized spheroids/organoids; programmable microfluidics; environmental control (O₂, shear, temp, light).
	•	Capabilities: time-sequenced dosing/gradients; sampling ports (media/biopsies); viability guardrails.
	•	Functional outputs: TEER/permeability, electrophysiology (MEA), contractility, calcium/optical physiology, oxygen consumption, perfusion/flow metrics, cilia beat frequency.

⸻

Sensing & Assays
	•	Functional instrumentation; high-content imaging (self-supervised morphology embeddings); sc-multi-omics (scRNA/ATAC/CITE); spatial transcriptomics/proteomics; bulk proteo/phospho/metabolomics; secretome & flux; interventional readouts (perturb-seq, CRISPRi/a).
	•	Orchestration: assay scheduling tied to perturbation scripts; calibration/QC/batch correction; on-the-fly feature extraction.

⸻

Data Ingestion & Metadata
	•	LIMS/ELN for lineage/protocols/telemetry; streaming broker for time-stamped events; experiment versioning; standardized ontologies (donor/consent, assay, device, perturbation, units); automated QC and batch correction with audit trails.

⸻

Storage & Data Model
	•	Lakehouse tiers: raw → bronze (parsed) → silver (QC’d) → gold (analysis-ready).
	•	Typed multi-scale Knowledge Graph (KG); feature store & embedding hub (morphology, cell-state, graph embeddings, twin parameters).
	•	Patient context: epigenetics/lifestyle/exposome; Clinical/RWD via OMOP; federated learning connectors (no raw PHI egress); consent & governance.

⸻

Knowledge Graph

The spine of the platform: a typed, versioned graph that contains priors from open sources and accumulating, provenance-linked evidence from our analyses.
	•	Entities: donor/patient, sample, cell type/state, ligand–receptor pair, pathway/module, gene/protein/variant, perturbation, device, assay, tissue function, environment (flow/O₂), clinical outcome, medication/exposure.
	•	Relations: causal (interventional support), temporal, co-expression/module membership, cell–cell communication, spatial neighborhood, clinical association, provenance.
	•	Evidence: prior score, posterior updates, uncertainty, direction/sign, cell/tissue context, condition tags, source/license, versioning.

Roles in the pipeline
	•	Prior & constraint for modeling (regularizes causal search and module discovery).
	•	Mechanistic scaffold & parameter priors for digital twins.
	•	Decision surface for active learning (identify high-value uncertainty).
	•	Ledger for biomarker evidence and mediation paths.

⸻

Modeling & Simulation
	•	Representation learning: multimodal contrastive/self-supervised across imaging, function, omics.
	•	Module discovery: WGCNA/hdWGCNA-style modules with KG-derived annotations (cell/neighborhood/pathway).
	•	Causal/dynamic modeling: time-series + interventional graphs; mediation enforcing molecular → function → clinical.
	•	Hybrid mechanistic–ML digital twins: ODE/QSP modules + PBPK/transport; ML surrogates for complex subsystems; donor-conditioned parameters; Bayesian uncertainty; counterfactual simulation.

⸻

Decision & Orchestration
	•	Active learning/Bayesian optimization: choose perturbations (dose/order/timing) that maximize expected causal information and reduce uncertainty.
	•	Model predictive control (MPC): enforce viability/safety constraints while exploring.
	•	Scheduler: resource-aware execution across devices/assays with replication and counterbalancing.

⸻

Biomarker Discovery Pipeline
	•	Use cases: diagnostic, prognostic, predictive, monitoring, PD markers.
	•	Scoring: causal impact on functional mediators; cross-model/donor replication; clinical association/mediation; assayability; robustness.
	•	Evidence dossier: mechanism path (graph), interventional evidence, replication, clinical mediation, analytical validation, risk/uncertainty, next steps.

⸻

Patient Avatars & Personalization

A donor-specific hybrid twin calibrated to that donor’s functional and omic trajectories; outputs personalized predictions and unique markers with path-level attribution.

⸻

Evidence Ladder & Validation
	•	E0: Correlation only
	•	E1: Temporal precedence + module linkage
	•	E2: Interventional causality in vitro
	•	E3: Cross-model & cross-donor replication
	•	E4: Clinical association + mediation via function
	•	E5: Prospective predictive performance + analytical validation
	•	Analytical validation: LoD/LoQ, linearity, precision, robustness.
	•	Biological validation: orthogonal assays, rescue experiments, donor hold-outs.
	•	Transportability: explicit checks from in-vitro to clinical domain.

⸻

Governance & Privacy
	•	Consent tracking; provenance & audit; PHI minimization; federated training for multi-site learning; license-aware KG edges; version pinning (KG vX.Y, pipelines vA.B, dataset releases).

⸻

KPIs & Dashboards
	•	Discovery velocity (cycles/week, info gain/experiment)
	•	Causal clarity (fraction of edges with interventional support; mediation effect sizes)
	•	Robustness (cross-donor/model replication)
	•	Translatability (AUROC/PPV on clinical endpoints)
	•	Operations (assay pass rate; cost per E3/E4 candidate)

⸻

Initial 90-Day MVP
	•	Scope: 1 disease area; 2 model systems (e.g., perfused organoid + tubular tissue).
	•	Assays: functional + imaging + targeted scRNA + secretome.
	•	Modeling: single mediation path; one digital-twin module (transport + simple signaling); basic active learner.
	•	Clinical link: one OMOP site, retrospective mediation.
	•	Output: 5–10 ranked candidates at E2–E3 with dossiers + a working avatar for ≥3 donors.

⸻

Repo Layout

Auto-generated by scripts/update_repo_tree.py. Re-run make update-readme-tree after adding folders/files.
<!-- TREE:START -->
```
ai-pipeline/
   └─ Makefile
   └─ README.md
   └─ requirements.txt
  └─ pipeline/
  └─ artifacts/
    └─ demo/
       └─ features.csv
       └─ report.md
  └─ tools/
     └─ __init__.py
    └─ __pycache__/
       └─ __init__.cpython-313.pyc
    └─ cli/
       └─ __init__.py
       └─ cli.py
       └─ main.py
      └─ __pycache__/
         └─ __init__.cpython-313.pyc
         └─ cli.cpython-313.pyc
         └─ main.cpython-313.pyc
  └─ biomarkers/
    └─ dossiers/
    └─ scoring/
  └─ .ruff_cache/
     └─ CACHEDIR.TAG
    └─ 0.5.7/
       └─ 16585389278632218221
       └─ 17199127306083185384
       └─ 18443766002398833481
       └─ 9145442378506862886
  └─ docs/
    └─ diagrams/
       └─ architecture_overview.mmd
       └─ data_flow_demo.mmd
       └─ decision_sequence.mmd
       └─ demo_gantt.mmd
       └─ evidence_ladder.mmd
      └─ exports/
         └─ architecture_overview.png
         └─ architecture_overview.svg
         └─ data_flow_demo.png
         └─ data_flow_demo.svg
         └─ decision_sequence.png
         └─ decision_sequence.svg
         └─ demo_gantt.png
         └─ demo_gantt.svg
         └─ evidence_ladder.png
         └─ evidence_ladder.svg
  └─ modeling/
    └─ twins/
    └─ predictors/
    └─ modules/
    └─ uncertainty/
  └─ env/
  └─ assays/
    └─ protocols/
    └─ qc/
  └─ dashboards/
    └─ notebooks/
  └─ scripts/
     └─ render_diagrams_make.sh
     └─ render_mermaid.sh
     └─ update_repo_tree.py
  └─ kg/
    └─ releases/
    └─ etl/
    └─ schema/
  └─ decision/
    └─ active_learning/
    └─ mpc_scheduler/
  └─ .vscode/
     └─ extensions.json
     └─ launch.json
     └─ settings.json
     └─ tasks.json
  └─ data/
    └─ cache/
    └─ lakehouse/
    └─ working/
       └─ labs.parquet
    └─ processed/
    └─ raw/
       └─ README.md
       └─ aki_labs_demo.csv
       └─ sample.csv
  └─ demos/
     └─ AKI_OPEN_DATA_DEMO.md
  └─ src/
     └─ __init__.py
     └─ features.py
     └─ ingest.py
     └─ preprocess.py
    └─ tools/
    └─ __pycache__/
       └─ __init__.cpython-313.pyc
       └─ ingest.cpython-313.pyc
```
<!-- TREE:END -->

## Additional Resources

### 📚 **Documentation Hub**
- **[Complete Package Overview](./PACKAGE_OVERVIEW.md)** - Comprehensive system documentation
- **[Technical Architecture](./TECHNICAL_ARCHITECTURE.md)** - Detailed technical specifications
- **[API Documentation](http://localhost:8000/docs)** - Interactive API documentation (after `docker-compose up`)

### 🔬 **Scientific Resources**
- **[Scientific Research Mission](./docs/scientific/mission.md)** - Research objectives and methodology
- **[Research Gaps Analysis](./docs/scientific/research-gaps.md)** - Current limitations and opportunities
- **[Experimental Rigor Methodology](./artifacts/experimental_rigor_methodology.md)** - Validation methodology

### 📊 **Performance & Benchmarking**
- **[Benchmarking Analysis](./benchmarking_analysis_report.md)** - Performance analysis report
- **[Industry Benchmarks](./docs/industry_benchmarks_sources.md)** - Industry comparison sources
- **[Competitive Analysis](./docs/quantitative_competitive_analysis.md)** - Quantitative competitive positioning

### 💊 **Clinical & Translational**
- **[Clinical Integration Strategy](./CLINICAL_EXPANSION_SUMMARY.md)** - Clinical workflow integration
- **[Personalized Biomarkers](./docs/next_win_personalized_biomarkers.md)** - Next-generation personalized medicine
- **[Tissue Chip Integration](./docs/tissue_chip_integration_strategy.md)** - In-vitro system integration

### 🛠️ **Development & Deployment**
- **[Enhanced Validation](./ENHANCED_VALIDATION_SUMMARY.md)** - Validation framework overview
- **[Production Deployment](./DEPLOYMENT_SUMMARY.md)** - Deployment strategy and infrastructure
- **[Development Roadmap](./docs/ranking_improvement_plan.md)** - Future development priorities

### 📈 **Analytics & Metrics**
- **[Advanced Metrics](./docs/advanced_metrics_and_integration_summary.md)** - Next-generation analytics
- **[Methodology Assessment](./docs/methodology_metrics.md)** - Technical methodology evaluation
- **[Uncertainty Methods](./docs/uncertainty_methods_industry.md)** - Risk assessment and uncertainty quantification

### 🎯 **Strategic Analysis**
- **[Market Analysis](./MARKET_RESEARCH_VALIDATION.md)** - Market research and validation
- **[Next Computational Wins](./docs/next_computational_wins_analysis.md)** - Future computational priorities
- **[Solution Opportunity](./SOLUTION_OPPORTUNITY_ASSESSMENT.md)** - Strategic opportunity assessment

### 🔧 **Implementation Guides**
- **[Quick Start Guide](./docs/quick-start.md)** - Rapid deployment guide
- **[Feature Explanations](./docs/features_explanation_nontechnical.md)** - Non-technical feature overview
- **[Capabilities Overview](./docs/capabilities.md)** - System capabilities summary

### 📁 **Generated Outputs & Results**
- **[Demo Outputs](./demo_outputs/)** - Generated demonstration results
- **[Benchmarking Results](./artifacts/bench/)** - Benchmarking analysis results
- **[Presentation Materials](./presentation/figures/)** - Visualization and presentation materials
- **[Research Artifacts](./artifacts/)** - Research outputs and analysis results

---

## Getting Started

### 🚀 **Quick Setup (5 minutes)**
```bash
# 1. Clone repository
git clone https://github.com/jeades1/ai-pipeline.git
cd ai-pipeline

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run quick demo
python main.py demo
```

### 🔧 **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-production.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run validation tests
python test_enhanced_validation.py
python test_clinical_decision_support.py
```

### 🐳 **Production Deployment**
```bash
# Start with Docker Compose
docker-compose up -d

# Access API documentation
open http://localhost:8000/docs

# Access monitoring dashboard
open http://localhost:3000  # Grafana
```

### 📊 **Interactive Demos**
```bash
# Web interface
python streamlit_demo.py

# Scientific demos
python biomarkers/causal_scoring.py
python biomarkers/real_data_integration.py

# Visualization generation
python presentation/generate_visualizations.py
```

### 🔬 **Advanced Features**
```bash
# Tissue chip integration
python test_tissue_chip_integration.py

# Multi-outcome prediction
python test_multi_outcome_prediction.py

# Federated learning demo
python biomarkers/federated_learning_final.py
```

### 📋 **Next Steps**
1. **Explore Documentation**: Start with [PACKAGE_OVERVIEW.md](./PACKAGE_OVERVIEW.md)
2. **Run Validations**: Execute validation tests in `demo_outputs/validation/`
3. **Review Results**: Check generated outputs in `demo_outputs/`
4. **Clinical Integration**: Explore clinical decision support features
5. **Advanced Analytics**: Dive into causal discovery and personalization

### 🆘 **Troubleshooting**
- **Dependencies**: Ensure Python 3.8+ and pip are installed
- **Docker Issues**: Check Docker daemon is running
- **Port Conflicts**: Modify ports in `docker-compose.yml` if needed
- **Performance**: See [docs/quick-start.md](./docs/quick-start.md) for optimization tips

### 📚 **Learning Path**
1. **Beginner**: README → Quick Demo → [Features Overview](./docs/features_explanation_nontechnical.md)
2. **Intermediate**: [Package Overview](./PACKAGE_OVERVIEW.md) → [Clinical Integration](./CLINICAL_EXPANSION_SUMMARY.md)
3. **Advanced**: [Enhanced Validation](./ENHANCED_VALIDATION_SUMMARY.md) → [Technical Architecture](./TECHNICAL_ARCHITECTURE.md)
4. **Expert**: [Scientific Mission](./docs/scientific/mission.md) → [Research Gaps](./docs/scientific/research-gaps.md)


---

## Glossary

### 🔬 **Core Concepts**
- **Functional mediator** — tissue-level readout (e.g., TEER, MEA) enforced as the bridge from molecular state to clinical outcome.
- **Digital twin (avatar)** — donor-specific hybrid mechanistic–ML model calibrated to that donor's functional/omic trajectories.
- **Evidence ladder** — progression from correlation → interventional causality → clinical mediation → prospective validation.
- **KG** — typed, versioned graph encoding biology, context, priors, and live evidence with uncertainty and provenance.

### 🧬 **Biomarker Types**
- **Diagnostic** — identifies current disease state or condition
- **Prognostic** — predicts future clinical outcomes
- **Predictive** — forecasts response to specific treatments
- **Monitoring** — tracks disease progression or treatment response
- **Pharmacodynamic (PD)** — measures drug mechanism and effect

### 🎯 **Validation Levels**
- **E0: Correlation** — Statistical association only
- **E1: Temporal** — Time-ordered precedence with module linkage  
- **E2: Interventional** — Causal evidence from in-vitro perturbations
- **E3: Replication** — Cross-model and cross-donor validation
- **E4: Clinical** — Clinical association with functional mediation
- **E5: Prospective** — Prospective validation with analytical rigor

### 🔧 **Technical Terms**
- **TEER** — Trans-epithelial electrical resistance (barrier function)
- **MEA** — Multi-electrode array (electrophysiology)
- **OMOP** — Observational Medical Outcomes Partnership (data standard)
- **PBPK** — Physiologically-based pharmacokinetic modeling
- **QSP** — Quantitative systems pharmacology
- **CRISPRi/a** — CRISPR interference/activation for perturbations

### 📊 **Analytics & AI**
- **Causal Discovery** — Automated identification of cause-effect relationships
- **Federated Learning** — Distributed ML training without data sharing
- **Active Learning** — Intelligent experiment design to maximize information gain
- **Uncertainty Quantification** — Systematic assessment of prediction confidence
- **Multi-modal Fusion** — Integration of diverse data types (omics, imaging, function)

---

## 📄 Version & License

**Version**: 2.0.0 (Enhanced Validation & Clinical Integration)  
**License**: MIT License  
**Repository**: https://github.com/jeades1/ai-pipeline  
**Documentation**: Complete package overview in [PACKAGE_OVERVIEW.md](./PACKAGE_OVERVIEW.md)  
**Last Updated**: September 2025

### �� **Key Achievements**
- ✅ Enhanced statistical validation framework
- ✅ Production-ready deployment infrastructure  
- ✅ Clinical decision support integration
- ✅ Advanced tissue chip AI integration
- ✅ Comprehensive documentation and testing
- ✅ Federated learning capabilities
- ✅ Multi-outcome prediction models

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and:
1. **Issues**: Report bugs or request features on GitHub
2. **Pull Requests**: Submit improvements with tests
3. **Documentation**: Help improve our documentation
4. **Research**: Collaborate on scientific validation

**Contact**: Submit issues on GitHub or check [Scientific Mission](./docs/scientific/mission.md) for research collaboration opportunities.

---

*This AI pipeline represents a comprehensive biomarker discovery platform with enhanced validation, clinical integration, and production deployment capabilities. For complete technical details, see [PACKAGE_OVERVIEW.md](./PACKAGE_OVERVIEW.md).*

