"""
Core Patient Avatar Integration System

Connects MIMIC-IV Avatar v0 model with personalized biomarker discovery.
Implements patient encoding, similarity matching, and biomarker scoring.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatientProfile:
    """Comprehensive patient representation for personalized biomarker discovery"""

    patient_id: str

    # Demographics
    age: int
    sex: str  # 'male', 'female'
    race: str
    bmi: float

    # Clinical history
    comorbidities: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)

    # Laboratory history (temporal data)
    lab_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Risk factors
    genetic_risk_scores: Dict[str, float] = field(default_factory=dict)
    lifestyle_factors: Dict[str, Any] = field(default_factory=dict)
    social_determinants: Dict[str, Any] = field(default_factory=dict)

    # Avatar-specific
    avatar_embedding: Optional[np.ndarray] = None
    avatar_confidence: float = 0.0

    # Clinical context
    admission_type: str = ""
    icu_length_of_stay: float = 0.0
    severity_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class BiomarkerScore:
    """Individual biomarker scoring for a patient"""

    biomarker: str
    population_rank: int
    population_score: float

    # Personalization modifiers
    age_modifier: float = 1.0
    sex_modifier: float = 1.0
    comorbidity_modifier: float = 1.0
    genetic_modifier: float = 1.0
    medication_modifier: float = 1.0

    # Final personalized score
    personalized_score: float = 0.0
    personalized_rank: int = 0
    confidence: float = 0.0

    # Clinical context
    mechanism_relevance: str = ""
    monitoring_priority: str = "standard"  # 'urgent', 'high', 'standard', 'low'


class PatientAvatarEncoder:
    """
    Encodes patient profiles into avatar latent space for similarity matching
    """

    def __init__(self, avatar_model_path: str = "modeling/twins/avatar_v0.pkl"):
        self.avatar_model_path = Path(avatar_model_path)
        self.avatar_model = None
        self.scaler = None
        self.feature_columns = None
        self._load_avatar_model()

    def _load_avatar_model(self):
        """Load pre-trained Avatar v0 model"""
        try:
            if self.avatar_model_path.exists():
                with open(self.avatar_model_path, "rb") as f:
                    model_data = pickle.load(f)
                    self.avatar_model = model_data.get("model")
                    self.scaler = model_data.get("scaler")
                    self.feature_columns = model_data.get("features", [])
                logger.info(f"Loaded Avatar v0 model from {self.avatar_model_path}")
            else:
                logger.warning(
                    f"Avatar model not found at {self.avatar_model_path}, using synthetic encoding"
                )
                self._create_synthetic_encoder()
        except Exception as e:
            logger.error(f"Error loading avatar model: {e}")
            self._create_synthetic_encoder()

    def _create_synthetic_encoder(self):
        """Create synthetic encoder for demonstration purposes"""
        logger.info("Creating synthetic patient encoder")
        # This would be replaced with actual Avatar v0 model
        self.feature_columns = [
            "age",
            "bmi",
            "creatinine_mean",
            "urine_output_mean",
            "sodium_mean",
            "potassium_mean",
            "chloride_mean",
            "diabetes",
            "hypertension",
            "ckd",
            "heart_disease",
        ]

    def encode_patient(self, patient_profile: PatientProfile) -> np.ndarray:
        """
        Encode patient profile into avatar latent space

        Args:
            patient_profile: Comprehensive patient data

        Returns:
            64-dimensional patient embedding vector
        """
        # Extract features for avatar model
        features = self._extract_features(patient_profile)

        if self.avatar_model is not None:
            # Use real Avatar v0 model
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            embedding = self.avatar_model.encode(features_scaled)[0]
        else:
            # Use synthetic encoding for demonstration
            embedding = self._synthetic_encode(features)

        # Update patient profile with embedding
        patient_profile.avatar_embedding = embedding
        patient_profile.avatar_confidence = self._calculate_confidence(features)

        return embedding

    def _extract_features(self, patient_profile: PatientProfile) -> np.ndarray:
        """Extract numerical features from patient profile"""

        # Basic demographics
        features = [
            patient_profile.age,
            patient_profile.bmi,
        ]

        # Laboratory means (from temporal data)
        if not patient_profile.lab_history.empty:
            lab_means = self._calculate_lab_means(patient_profile.lab_history)
            features.extend(lab_means)
        else:
            # Default values if no lab history
            features.extend([1.0, 1200.0, 140.0, 4.0, 100.0])  # Cr, UO, Na, K, Cl

        # Comorbidity indicators
        comorbidity_flags = [
            "diabetes" in patient_profile.comorbidities,
            "hypertension" in patient_profile.comorbidities,
            "ckd" in patient_profile.comorbidities,
            "heart_disease" in patient_profile.comorbidities,
        ]
        features.extend([float(flag) for flag in comorbidity_flags])

        return np.array(features)

    def _calculate_lab_means(self, lab_history: pd.DataFrame) -> List[float]:
        """Calculate mean laboratory values from temporal data"""
        lab_means = []

        # Expected lab columns and defaults
        expected_labs = {
            "creatinine": 1.0,
            "urine_output": 1200.0,
            "sodium": 140.0,
            "potassium": 4.0,
            "chloride": 100.0,
        }

        for lab, default in expected_labs.items():
            if lab in lab_history.columns:
                mean_val = lab_history[lab].mean()
                lab_means.append(mean_val if not pd.isna(mean_val) else default)
            else:
                lab_means.append(default)

        return lab_means

    def _synthetic_encode(self, features: np.ndarray) -> np.ndarray:
        """Create synthetic 64-dimensional embedding"""
        # Deterministic encoding based on features for consistency
        np.random.seed(int(np.sum(features) * 1000) % 10000)

        # Create embedding with realistic structure
        embedding = np.random.randn(64)

        # Add feature-specific components
        embedding[: len(features)] += features * 0.1

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in patient encoding"""
        # Simple heuristic: more complete data = higher confidence
        completeness = np.mean(features != 0)  # Assume 0 means missing
        confidence = min(0.95, 0.5 + completeness * 0.5)
        return confidence


class PatientSimilarityMatcher:
    """
    Finds similar patients based on avatar embeddings for biomarker personalization
    """

    def __init__(self):
        self.patient_database = {}
        self.embeddings_matrix = None
        self.patient_ids = []

    def add_patient(
        self,
        patient_id: str,
        embedding: np.ndarray,
        outcomes: Optional[Dict[str, Any]] = None,
    ):
        """Add patient to similarity database"""
        self.patient_database[patient_id] = {
            "embedding": embedding,
            "outcomes": outcomes or {},
            "added_date": datetime.now(),
        }
        self._rebuild_embedding_matrix()

    def find_similar_patients(
        self, query_embedding: np.ndarray, n_similar: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Find n most similar patients based on avatar embeddings

        Args:
            query_embedding: Patient embedding to match
            n_similar: Number of similar patients to return

        Returns:
            List of (patient_id, similarity_score) tuples
        """
        if self.embeddings_matrix is None:
            return []

        # Calculate cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings_matrix)

        # Get top n similar patients
        top_indices = np.argsort(similarities)[-n_similar:][::-1]

        similar_patients = [
            (self.patient_ids[idx], float(similarities[idx])) for idx in top_indices
        ]

        return similar_patients

    def _rebuild_embedding_matrix(self):
        """Rebuild matrix of all patient embeddings"""
        if not self.patient_database:
            return

        self.patient_ids = list(self.patient_database.keys())
        embeddings = [
            self.patient_database[pid]["embedding"] for pid in self.patient_ids
        ]
        self.embeddings_matrix = np.array(embeddings)

    def _cosine_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and matrix rows"""
        # Normalize query and matrix
        query_norm = query / np.linalg.norm(query)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        # Calculate similarities
        similarities = np.dot(matrix_norm, query_norm)
        return similarities


class PersonalizedBiomarkerEngine:
    """
    Core engine that combines patient avatars with biomarker discovery
    """

    def __init__(self):
        self.encoder = PatientAvatarEncoder()
        self.similarity_matcher = PatientSimilarityMatcher()

        # Load population biomarker rankings (from CV optimization)
        self.population_biomarkers = self._load_population_biomarkers()

        # Biomarker characteristics for personalization
        self.biomarker_profiles = self._load_biomarker_profiles()

        logger.info("Initialized PersonalizedBiomarkerEngine")

    def _load_population_biomarkers(self) -> Dict[str, Dict]:
        """Load population-level biomarker rankings"""
        # These come from your CV optimization results
        return {
            "APOB": {"rank": 1, "precision_20": 0.85, "pathway": "lipid_transport"},
            "HMGCR": {
                "rank": 2,
                "precision_20": 0.82,
                "pathway": "cholesterol_synthesis",
            },
            "LDLR": {"rank": 3, "precision_20": 0.80, "pathway": "ldl_regulation"},
            "PCSK9": {"rank": 4, "precision_20": 0.78, "pathway": "ldl_regulation"},
            "LPL": {
                "rank": 5,
                "precision_20": 0.75,
                "pathway": "triglyceride_metabolism",
            },
            "ADIPOQ": {"rank": 6, "precision_20": 0.73, "pathway": "adipose_signaling"},
            "CETP": {"rank": 7, "precision_20": 0.70, "pathway": "hdl_metabolism"},
            "APOA1": {"rank": 8, "precision_20": 0.68, "pathway": "hdl_metabolism"},
            "LPA": {"rank": 9, "precision_20": 0.65, "pathway": "genetic_lipid"},
            "CRP": {"rank": 10, "precision_20": 0.62, "pathway": "inflammation"},
        }

    def _load_biomarker_profiles(self) -> Dict[str, Dict]:
        """Load biomarker characteristics for personalization"""
        return {
            "APOB": {
                "age_sensitivity": 0.8,
                "sex_effect": "male_higher",
                "comorbidity_modifiers": {"diabetes": 1.3, "ckd": 1.2},
                "medication_modifiers": {"statin": 0.8},
                "volatility": "low",
                "half_life_days": 3,
            },
            "HMGCR": {
                "age_sensitivity": 0.6,
                "sex_effect": "minimal",
                "comorbidity_modifiers": {"hyperlipidemia": 1.4},
                "medication_modifiers": {"statin": 1.5},
                "volatility": "medium",
                "half_life_days": 1,
            },
            "PCSK9": {
                "age_sensitivity": 0.7,
                "sex_effect": "female_higher",
                "comorbidity_modifiers": {"ckd": 1.5, "inflammation": 1.3},
                "medication_modifiers": {"pcsk9_inhibitor": 0.3},
                "volatility": "medium",
                "half_life_days": 7,
            },
            "CRP": {
                "age_sensitivity": 0.9,
                "sex_effect": "female_higher",
                "comorbidity_modifiers": {
                    "diabetes": 1.4,
                    "ckd": 1.6,
                    "inflammation": 2.0,
                },
                "medication_modifiers": {"steroid": 0.5, "nsaid": 0.8},
                "volatility": "high",
                "half_life_days": 0.5,
            },
            "LPA": {
                "age_sensitivity": 0.3,
                "sex_effect": "minimal",
                "comorbidity_modifiers": {"family_history": 2.0},
                "medication_modifiers": {},
                "volatility": "very_low",
                "half_life_days": 365,  # Genetic
            },
        }

    def generate_personalized_scores(
        self, patient_profile: PatientProfile
    ) -> List[BiomarkerScore]:
        """
        Generate personalized biomarker scores for a patient

        Args:
            patient_profile: Comprehensive patient data

        Returns:
            List of BiomarkerScore objects ranked by personalized relevance
        """
        # Encode patient into avatar space
        patient_embedding = self.encoder.encode_patient(patient_profile)

        # Find similar patients for biomarker performance analysis
        similar_patients = self.similarity_matcher.find_similar_patients(
            patient_embedding, n_similar=50
        )

        # Calculate personalized scores for each biomarker
        biomarker_scores = []

        for biomarker, pop_data in self.population_biomarkers.items():
            score = self._calculate_personalized_score(
                biomarker, patient_profile, similar_patients, pop_data
            )
            biomarker_scores.append(score)

        # Rank by personalized score
        biomarker_scores.sort(key=lambda x: x.personalized_score, reverse=True)

        # Update ranks
        for i, score in enumerate(biomarker_scores):
            score.personalized_rank = i + 1

        return biomarker_scores

    def _calculate_personalized_score(
        self,
        biomarker: str,
        patient_profile: PatientProfile,
        similar_patients: List[Tuple[str, float]],
        pop_data: Dict,
    ) -> BiomarkerScore:
        """Calculate personalized score for a single biomarker"""

        base_score = pop_data["precision_20"]
        biomarker_profile = self.biomarker_profiles.get(biomarker, {})

        # Calculate modifiers
        age_modifier = self._calculate_age_modifier(
            biomarker, patient_profile.age, biomarker_profile
        )
        sex_modifier = self._calculate_sex_modifier(
            biomarker, patient_profile.sex, biomarker_profile
        )
        comorbidity_modifier = self._calculate_comorbidity_modifier(
            biomarker, patient_profile.comorbidities, biomarker_profile
        )
        genetic_modifier = self._calculate_genetic_modifier(
            biomarker, patient_profile.genetic_risk_scores
        )
        medication_modifier = self._calculate_medication_modifier(
            biomarker, patient_profile.medications, biomarker_profile
        )

        # Calculate final personalized score
        personalized_score = (
            base_score
            * age_modifier
            * sex_modifier
            * comorbidity_modifier
            * genetic_modifier
            * medication_modifier
        )

        # Calculate confidence based on similar patients and data completeness
        confidence = self._calculate_score_confidence(patient_profile, similar_patients)

        # Determine mechanism relevance
        mechanism_relevance = self._determine_mechanism_relevance(
            biomarker, patient_profile
        )

        return BiomarkerScore(
            biomarker=biomarker,
            population_rank=pop_data["rank"],
            population_score=base_score,
            age_modifier=age_modifier,
            sex_modifier=sex_modifier,
            comorbidity_modifier=comorbidity_modifier,
            genetic_modifier=genetic_modifier,
            medication_modifier=medication_modifier,
            personalized_score=personalized_score,
            confidence=confidence,
            mechanism_relevance=mechanism_relevance,
        )

    def _calculate_age_modifier(self, biomarker: str, age: int, profile: Dict) -> float:
        """Calculate age-based score modifier"""
        age_sensitivity = profile.get("age_sensitivity", 0.5)

        if age < 40:
            return 1.0 - (age_sensitivity * 0.2)
        elif age < 65:
            return 1.0
        else:
            return 1.0 + (age_sensitivity * 0.3)

    def _calculate_sex_modifier(self, biomarker: str, sex: str, profile: Dict) -> float:
        """Calculate sex-based score modifier"""
        sex_effect = profile.get("sex_effect", "minimal")

        if sex_effect == "male_higher":
            return 1.2 if sex == "male" else 0.9
        elif sex_effect == "female_higher":
            return 1.2 if sex == "female" else 0.9
        else:
            return 1.0

    def _calculate_comorbidity_modifier(
        self, biomarker: str, comorbidities: List[str], profile: Dict
    ) -> float:
        """Calculate comorbidity-based score modifier"""
        modifiers = profile.get("comorbidity_modifiers", {})

        modifier = 1.0
        for condition, factor in modifiers.items():
            if condition in comorbidities:
                modifier *= factor

        return modifier

    def _calculate_genetic_modifier(
        self, biomarker: str, genetic_scores: Dict[str, float]
    ) -> float:
        """Calculate genetic risk-based modifier"""
        # Simple heuristic: higher genetic risk increases biomarker relevance
        relevant_scores = {
            "APOB": genetic_scores.get("cardiovascular", 0.5),
            "LPA": genetic_scores.get("cardiovascular", 0.5),
            "PCSK9": genetic_scores.get("cardiovascular", 0.5),
            "CRP": genetic_scores.get("inflammation", 0.5),
        }

        score = relevant_scores.get(biomarker, 0.5)
        return 0.8 + (score * 0.4)  # Range: 0.8 to 1.2

    def _calculate_medication_modifier(
        self, biomarker: str, medications: List[str], profile: Dict
    ) -> float:
        """Calculate medication effect modifier"""
        modifiers = profile.get("medication_modifiers", {})

        modifier = 1.0
        for med, factor in modifiers.items():
            if any(med in medication.lower() for medication in medications):
                modifier *= factor

        return modifier

    def _calculate_score_confidence(
        self, patient_profile: PatientProfile, similar_patients: List[Tuple[str, float]]
    ) -> float:
        """Calculate confidence in personalized score"""
        # Base confidence from avatar encoding
        base_confidence = patient_profile.avatar_confidence or 0.5

        # Boost confidence if we have similar patients
        similarity_boost = min(0.3, len(similar_patients) * 0.01)

        # Reduce confidence if patient is very unusual
        if (
            similar_patients and similar_patients[0][1] < 0.3
        ):  # Low similarity to best match
            similarity_boost *= 0.5

        confidence = min(0.95, base_confidence + similarity_boost)
        return confidence

    def _determine_mechanism_relevance(
        self, biomarker: str, patient_profile: PatientProfile
    ) -> str:
        """Determine why this biomarker is relevant for this patient"""

        relevance_map = {
            "APOB": "Lipid transport dysfunction",
            "HMGCR": "Cholesterol synthesis activity",
            "PCSK9": "LDL receptor regulation",
            "CRP": "Inflammatory cascade activation",
            "LPA": "Genetic predisposition marker",
        }

        base_relevance = relevance_map.get(biomarker, "General biomarker")

        # Add patient-specific context
        if "diabetes" in patient_profile.comorbidities and biomarker in [
            "CRP",
            "ADIPOQ",
        ]:
            return f"{base_relevance} (diabetic complication)"
        elif "ckd" in patient_profile.comorbidities and biomarker in ["PCSK9", "CRP"]:
            return f"{base_relevance} (kidney-cardiovascular axis)"
        elif patient_profile.age > 65 and biomarker == "CRP":
            return f"{base_relevance} (age-related inflammation)"

        return base_relevance


# Demo/testing functions
def create_demo_patients() -> List[PatientProfile]:
    """Create demonstration patient profiles"""

    patients = []

    # Young athletic with family history
    patients.append(
        PatientProfile(
            patient_id="DEMO_001",
            age=35,
            sex="male",
            race="caucasian",
            bmi=22.0,
            comorbidities=["family_history"],
            medications=["multivitamin"],
            genetic_risk_scores={"cardiovascular": 0.7, "inflammation": 0.3},
            lifestyle_factors={"exercise": "high", "smoking": False},
            lab_history=pd.DataFrame(
                {
                    "creatinine": [0.9, 0.95, 0.88],
                    "sodium": [140, 142, 139],
                    "potassium": [4.1, 4.0, 4.2],
                }
            ),
        )
    )

    # Middle-aged with metabolic syndrome
    patients.append(
        PatientProfile(
            patient_id="DEMO_002",
            age=55,
            sex="female",
            race="hispanic",
            bmi=29.5,
            comorbidities=["diabetes", "hypertension", "hyperlipidemia"],
            medications=["metformin", "lisinopril", "atorvastatin"],
            genetic_risk_scores={"cardiovascular": 0.8, "inflammation": 0.7},
            lifestyle_factors={"exercise": "low", "smoking": False},
            lab_history=pd.DataFrame(
                {
                    "creatinine": [1.1, 1.2, 1.15],
                    "sodium": [138, 141, 140],
                    "potassium": [4.5, 4.3, 4.4],
                }
            ),
        )
    )

    # Elderly with multiple comorbidities
    patients.append(
        PatientProfile(
            patient_id="DEMO_003",
            age=75,
            sex="male",
            race="african_american",
            bmi=26.0,
            comorbidities=["ckd", "heart_disease", "diabetes", "inflammation"],
            medications=["insulin", "metoprolol", "furosemide", "prednisone"],
            genetic_risk_scores={"cardiovascular": 0.9, "inflammation": 0.8},
            lifestyle_factors={"exercise": "limited", "smoking": True},
            lab_history=pd.DataFrame(
                {
                    "creatinine": [2.1, 2.3, 2.0],
                    "sodium": [135, 137, 136],
                    "potassium": [5.1, 4.9, 5.0],
                }
            ),
        )
    )

    return patients


def run_avatar_integration_demo():
    """Run demonstration of patient avatar integration"""

    print("\n🧬 PATIENT AVATAR INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Initialize engine
    engine = PersonalizedBiomarkerEngine()

    # Create demo patients
    demo_patients = create_demo_patients()

    for patient in demo_patients:
        print(f"\n👤 Patient: {patient.patient_id}")
        print(f"   Profile: Age {patient.age}, {patient.sex}, BMI {patient.bmi}")
        print(f"   Comorbidities: {', '.join(patient.comorbidities)}")

        # Generate personalized scores
        scores = engine.generate_personalized_scores(patient)

        print(f"   Avatar Confidence: {patient.avatar_confidence:.2f}")
        if patient.avatar_embedding is not None:
            print(f"   Embedding Norm: {np.linalg.norm(patient.avatar_embedding):.2f}")
        else:
            print("   Embedding: Not calculated")

        print("\n   Top 5 Personalized Biomarkers:")
        for i, score in enumerate(scores[:5], 1):
            print(
                f"     {i}. {score.biomarker}: {score.personalized_score:.3f} "
                f"(pop rank #{score.population_rank} → #{score.personalized_rank}) "
                f"[confidence: {score.confidence:.2f}]"
            )
            print(f"        {score.mechanism_relevance}")

        print("\n   Key Modifiers:")
        for score in scores[:3]:
            print(
                f"     {score.biomarker}: age={score.age_modifier:.2f}, "
                f"sex={score.sex_modifier:.2f}, comorbid={score.comorbidity_modifier:.2f}"
            )


if __name__ == "__main__":
    run_avatar_integration_demo()
