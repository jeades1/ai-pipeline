# kg/ontology.py - Enhanced with Tissue-Chip Architecture Ontology
from typing import Dict

# Standard biomarker aliases
ALIASES = {
    "NGAL": ["LCN2", "Lipocalin-2"],
    "KIM-1": ["HAVCR1"],
    "TIMP-2·IGFBP7": ["TIMP2*IGFBP7", "[TIMP-2][IGFBP7]"],
}

# Multicellular Architecture Ontology
MULTICELLULAR_ONTOLOGY = {
    "TubularArchitecture": {
        "aliases": ["tubular_geometry", "3D_tubular", "multicellular_tubular"],
        "properties": ["cell_organization", "barrier_function", "tissue_polarity"],
        "cell_types": ["epithelial", "endothelial", "fibroblast", "immune"],
    },
    "CellCellSignaling": {
        "aliases": [
            "intercellular_communication",
            "cell_signaling",
            "paracrine_signaling",
        ],
        "mechanisms": ["paracrine", "juxtacrine", "autocrine", "gap_junction"],
        "validation_markers": ["cytokine_gradients", "growth_factor_networks"],
    },
    "BarrierFunction": {
        "aliases": ["epithelial_barrier", "tissue_barrier", "permeability_barrier"],
        "metrics": ["TEER", "permeability_coefficient", "tight_junction_integrity"],
        "physiological_range": {"TEER_ohm_cm2": [1000, 3000]},
    },
}

# PDO Vascularization Ontology
VASCULARIZATION_ONTOLOGY = {
    "PDOVascularization": {
        "aliases": ["patient_derived_vascularization", "organoid_vascularization"],
        "enhancement_levels": [
            "10x_delivery",
            "100x_delivery",
            "physiological_delivery",
        ],
        "components": ["endothelial_network", "perfusion_chambers", "flow_control"],
    },
    "MolecularDelivery": {
        "aliases": ["drug_delivery", "biomarker_delivery", "molecular_transport"],
        "enhancement_metrics": [
            "penetration_depth",
            "delivery_efficiency",
            "distribution_uniformity",
        ],
        "large_tissue_benefits": [
            "enhanced_penetration",
            "improved_distribution",
            "sustained_levels",
        ],
    },
    "VascularNetwork": {
        "aliases": ["endothelial_network", "capillary_network", "perfusion_network"],
        "properties": ["network_density", "flow_distribution", "permeability_control"],
        "integration_markers": ["CD31", "VE_cadherin", "PECAM1"],
    },
}

# Kinetic Analysis Ontology
KINETIC_ONTOLOGY = {
    "RecirculationKinetics": {
        "aliases": ["biomarker_kinetics", "temporal_kinetics", "dynamic_analysis"],
        "temporal_resolution": ["sub_minute", "minute", "hour"],
        "kinetic_parameters": [
            "secretion_rate",
            "clearance_rate",
            "half_life",
            "steady_state",
        ],
    },
    "PharmacokineticModeling": {
        "aliases": ["PK_modeling", "drug_kinetics", "ADME_analysis"],
        "models": ["one_compartment", "two_compartment", "physiological_based"],
        "parameters": ["absorption", "distribution", "metabolism", "elimination"],
    },
    "TemporalProfiling": {
        "aliases": ["time_course", "dynamic_profiling", "kinetic_profiling"],
        "sampling_methods": ["continuous", "periodic", "automated"],
        "analysis_types": [
            "secretion_kinetics",
            "uptake_kinetics",
            "elimination_kinetics",
        ],
    },
}

# Perfusion Culture Ontology
PERFUSION_ONTOLOGY = {
    "PerfusionCulture": {
        "aliases": ["dynamic_culture", "flow_culture", "perfused_system"],
        "benefits": ["extended_viability", "enhanced_maturation", "improved_function"],
        "longevity_metrics": {
            "standard_days": 7,
            "enhanced_days": 14,
            "optimal_days": 21,
        },
    },
    "FlowControl": {
        "aliases": ["perfusion_control", "flow_regulation", "pressure_control"],
        "parameters": ["flow_rate", "pressure_gradient", "shear_stress"],
        "physiological_ranges": {
            "flow_rate_ul_min": [10, 500],
            "shear_stress_dyn_cm2": [0.1, 10.0],
            "pressure_mmHg": [5, 25],
        },
    },
}

# Tissue-Chip Validation Ontology
VALIDATION_ONTOLOGY = {
    "TissueChipValidation": {
        "aliases": ["chip_validation", "tissue_validation", "multicellular_validation"],
        "validation_levels": ["architecture", "function", "biomarker", "clinical"],
        "success_metrics": [
            "sensitivity",
            "specificity",
            "reproducibility",
            "clinical_correlation",
        ],
    },
    "ClinicalTranslation": {
        "aliases": [
            "clinical_correlation",
            "translational_validation",
            "clinical_relevance",
        ],
        "enhancement_factors": {
            "sensitivity": "10x",
            "specificity": "2x",
            "success_rate": "3x",
        },
        "translation_stages": [
            "preclinical",
            "clinical_validation",
            "regulatory_approval",
        ],
    },
}


def normalize_label(label: str) -> str:
    """Enhanced label normalization including tissue-chip ontology"""
    label = label.strip()

    # Standard biomarker aliases
    for canon, syns in ALIASES.items():
        if label == canon or label in syns:
            return canon

    # Multicellular architecture normalization
    for concept, data in MULTICELLULAR_ONTOLOGY.items():
        if label == concept or label in data.get("aliases", []):
            return concept

    # Vascularization normalization
    for concept, data in VASCULARIZATION_ONTOLOGY.items():
        if label == concept or label in data.get("aliases", []):
            return concept

    # Kinetic analysis normalization
    for concept, data in KINETIC_ONTOLOGY.items():
        if label == concept or label in data.get("aliases", []):
            return concept

    # Perfusion culture normalization
    for concept, data in PERFUSION_ONTOLOGY.items():
        if label == concept or label in data.get("aliases", []):
            return concept

    # Validation normalization
    for concept, data in VALIDATION_ONTOLOGY.items():
        if label == concept or label in data.get("aliases", []):
            return concept

    return label


def canonicalize_map(names) -> Dict[str, str]:
    """Enhanced canonicalization including tissue-chip concepts"""
    return {n: normalize_label(n) for n in names}


def get_tissue_chip_concepts() -> Dict[str, Dict]:
    """Return all tissue-chip related ontology concepts"""
    return {
        "multicellular": MULTICELLULAR_ONTOLOGY,
        "vascularization": VASCULARIZATION_ONTOLOGY,
        "kinetics": KINETIC_ONTOLOGY,
        "perfusion": PERFUSION_ONTOLOGY,
        "validation": VALIDATION_ONTOLOGY,
    }


def validate_tissue_chip_parameters(
    architecture_type: str, vascularization_level: str
) -> bool:
    """Validate tissue-chip configuration parameters"""
    valid_architectures = list(MULTICELLULAR_ONTOLOGY.keys())
    valid_vascularization = list(VASCULARIZATION_ONTOLOGY.keys())

    return (
        architecture_type in valid_architectures
        and vascularization_level in valid_vascularization
    )
