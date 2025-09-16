# In Vitro Integration Framework

## Framework for Enhanced In Vitro Integration

### **Problem Statement**
- Current: Simple toy dataset with 5 genes and basic functional scores
- Need: Comprehensive framework for real multi-cellular, multi-assay integration
- Goal: Prevent recommending experiments when data already exists

### **Proposed Framework Architecture**

#### **1. Multi-Cellular Interaction Data in KG**
```python
# Enhanced KG node types
CELL_TYPES = {
    "ProximalTubular": {"marker_genes": ["SLC34A1", "LRP2"], "functions": ["transport", "reabsorption"]},
    "DistalTubular": {"marker_genes": ["NCCT", "ENaC"], "functions": ["electrolyte_balance"]}, 
    "Podocyte": {"marker_genes": ["NPHS1", "NPHS2"], "functions": ["filtration", "barrier"]},
    "Endothelial": {"marker_genes": ["PECAM1", "VWF"], "functions": ["vascular_tone", "permeability"]},
    "Immune": {"marker_genes": ["CD68", "CD3"], "functions": ["inflammation", "repair"]}
}

# Enhanced edge types for cellular interactions
INTERACTION_TYPES = {
    "cell_cell_contact": {"directionality": "bidirectional", "evidence": ["imaging", "proteomics"]},
    "paracrine_signaling": {"directionality": "source_to_target", "evidence": ["secretome", "functional"]},
    "mechanical_coupling": {"directionality": "bidirectional", "evidence": ["TEER", "contractility"]},
    "metabolic_cross_feeding": {"directionality": "source_to_target", "evidence": ["metabolomics"]}
}
```

#### **2. Assay Response Data Integration**
```python
# Multi-modal assay integration
ASSAY_RESPONSE_SCHEMA = {
    "functional": {
        "TEER": {"units": "ohm*cm2", "direction": "higher_is_better", "cell_types": ["epithelial"]},
        "permeability": {"units": "cm/s", "direction": "lower_is_better", "cell_types": ["epithelial"]},
        "contractility": {"units": "mN", "direction": "context_dependent", "cell_types": ["vascular"]}
    },
    "secretome": {
        "ELISA": {"targets": ["NGAL", "KIM1", "IL18"], "quantitative": True},
        "multiplex": {"targets": "proteome_wide", "semi_quantitative": True}
    },
    "molecular": {
        "scRNA_seq": {"resolution": "single_cell", "coverage": "transcriptome"},
        "bulk_RNA": {"resolution": "population", "coverage": "transcriptome"}, 
        "proteomics": {"resolution": "population", "coverage": "proteome"}
    }
}
```

#### **3. Data Existence Checking**
```python
class ExperimentAvoidanceSystem:
    """Prevent recommending experiments when equivalent data exists."""
    
    def check_existing_evidence(self, gene: str, perturbation: str, 
                               assay_type: str, cell_type: str) -> bool:
        """Check if experiment combination already exists in KG."""
        # Query KG for existing edges matching criteria
        existing = self.kg.query_edges(
            source=gene,
            predicate="responds_to_perturbation", 
            context={"perturbation": perturbation, "assay": assay_type, "cells": cell_type}
        )
        return len(existing) > 0
    
    def filter_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove redundant experiment recommendations."""
        filtered = []
        for rec in recommendations:
            if not self.check_existing_evidence(**rec):
                filtered.append(rec)
            else:
                print(f"Skipping {rec} - equivalent data exists")
        return filtered
```

### **4. Possible Implementation Without Real Data**

#### **Synthetic Data Generation Framework**
```python
def generate_synthetic_invitro_data(n_genes: int = 100, n_conditions: int = 20) -> pd.DataFrame:
    """Generate realistic synthetic in vitro data for framework testing."""
    
    # Base this on known AKI gene signatures and realistic assay ranges
    aki_genes = ["HAVCR1", "LCN2", "TIMP2", "CCL2", "CST3", "IGFBP7", "IL18", "UMOD"]
    
    data = []
    for gene in aki_genes:
        for condition in ["control", "cisplatin", "ischemia", "sepsis"]:
            for assay in ["TEER", "permeability", "ELISA", "scRNA"]:
                # Use realistic effect sizes based on literature
                baseline_effect = np.random.normal(0, 0.1)
                if gene in ["HAVCR1", "LCN2"] and condition != "control":
                    injury_effect = np.random.normal(0.8, 0.2)  # Strong injury markers
                elif condition != "control":
                    injury_effect = np.random.normal(0.3, 0.3)  # Variable response
                else:
                    injury_effect = baseline_effect
                    
                data.append({
                    "gene": gene,
                    "condition": condition, 
                    "assay": assay,
                    "effect_size": injury_effect,
                    "p_value": np.random.exponential(0.1),
                    "cell_type": "tubular_epithelial"
                })
    
    return pd.DataFrame(data)
```

#### **KG Enhancement for Cell-Cell Interactions**
```python
def add_cellular_interaction_layer(kg: KGEvidenceGraph) -> None:
    """Add cell-cell interaction nodes and edges to KG."""
    
    # Add cell type nodes
    for cell_type, props in CELL_TYPES.items():
        kg.ensure_node(cell_type, kind="CellType", layer="cellular", **props)
    
    # Add interaction edges based on literature/databases
    interactions = [
        ("ProximalTubular", "Endothelial", "paracrine_signaling", {"mediators": ["VEGF", "NO"]}),
        ("Immune", "ProximalTubular", "cell_cell_contact", {"context": "inflammation"}),
        ("Podocyte", "Endothelial", "mechanical_coupling", {"context": "filtration"})
    ]
    
    for source, target, interaction_type, evidence in interactions:
        kg.add_edge(source, target, etype=interaction_type, 
                   provenance="literature", evidence=evidence)

def link_genes_to_cell_functions(kg: KGEvidenceGraph, invitro_data: pd.DataFrame) -> None:
    """Link gene expression to cellular function outcomes."""
    
    # Group by gene and compute functional consistency across assays
    for gene, gene_data in invitro_data.groupby("gene"):
        functional_scores = {}
        
        # TEER/permeability -> barrier function
        teer_data = gene_data[gene_data["assay"] == "TEER"]
        if not teer_data.empty:
            functional_scores["barrier_integrity"] = teer_data["effect_size"].mean()
            
        # Secretome -> signaling function  
        elisa_data = gene_data[gene_data["assay"] == "ELISA"]
        if not elisa_data.empty:
            functional_scores["signaling_activity"] = elisa_data["effect_size"].mean()
        
        # Add gene->function edges
        for function, score in functional_scores.items():
            kg.add_edge(gene, function, etype="regulates_function", 
                       evidence={"score": score}, provenance="invitro_synthetic")
```

### **5. Data Integration Priority**

#### **Phase 1: Framework (No Real Data Needed)**
1. **Enhanced KG schema**: Cell types, interaction types, assay response types
2. **Synthetic data generator**: Realistic multi-assay, multi-cellular data
3. **Experiment avoidance system**: Check existing evidence before recommendations

#### **Phase 2: Real Data Integration (When Available)**
1. **Organoid transcriptomics**: scRNA-seq from kidney organoids under injury
2. **Functional readouts**: TEER, permeability, contractility measurements  
3. **Secretome proteomics**: Multiplex protein panels from organoid supernatants
4. **Imaging data**: High-content imaging with morphological features

#### **Phase 3: Advanced Integration**
1. **Temporal dynamics**: Time-series functional and molecular responses
2. **Dose-response modeling**: Concentration-dependent effects
3. **Multi-donor variability**: Account for genetic background effects

### **Implementation Commands**
```bash
# Generate synthetic framework
python -c "from src.invitro.framework import generate_synthetic_invitro_data; generate_synthetic_invitro_data()"

# Test enhanced KG integration
python main.py demo --invitro-boost 0.3 --synthetic-cellular-data

# Validate experiment avoidance
python -c "from src.invitro.framework import ExperimentAvoidanceSystem; ExperimentAvoidanceSystem().test_redundancy_detection()"
```

This framework allows us to build and test the full integration pipeline using realistic synthetic data, while preparing for seamless real data integration when available.
