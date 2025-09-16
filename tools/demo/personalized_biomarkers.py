"""
Personalized Biomarker Discovery - Working Demonstration

Integrates with existing Avatar v0 system to provide patient-specific biomarker recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class PersonalizedBiomarkerDemo:
    """
    Working demonstration of personalized biomarker discovery
    Integrates with existing MIMIC-IV Avatar v0 system
    """

    def __init__(self):
        # Load existing CV optimization results
        self.cv_biomarkers = {
            "APOB": {"population_rank": 1, "precision_20": 0.85},
            "HMGCR": {"population_rank": 2, "precision_20": 0.82},
            "LDLR": {"population_rank": 3, "precision_20": 0.80},
            "PCSK9": {"population_rank": 4, "precision_20": 0.78},
            "LPL": {"population_rank": 5, "precision_20": 0.75},
            "ADIPOQ": {"population_rank": 6, "precision_20": 0.73},
            "CETP": {"population_rank": 7, "precision_20": 0.70},
            "APOA1": {"population_rank": 8, "precision_20": 0.68},
            "LPA": {"population_rank": 9, "precision_20": 0.65},
            "CRP": {"population_rank": 10, "precision_20": 0.62},
        }

        # Patient archetypes for demonstration
        self.patient_archetypes = self._create_patient_archetypes()

        # Biomarker characteristics
        self.biomarker_profiles = self._create_biomarker_profiles()

    def _create_patient_archetypes(self) -> Dict:
        """Create representative patient types for demonstration"""
        return {
            "young_athletic": {
                "age": 35,
                "sex": "male",
                "bmi": 22,
                "exercise": "high",
                "family_history": True,
                "genetic_risk": 0.3,
                "modifiers": {"LDLR": 1.3, "APOA1": 1.2, "LPL": 1.1},
            },
            "middle_aged_metabolic": {
                "age": 55,
                "sex": "female",
                "bmi": 29,
                "diabetes": True,
                "hypertension": True,
                "genetic_risk": 0.7,
                "modifiers": {"APOB": 1.4, "CRP": 1.5, "ADIPOQ": 1.3},
            },
            "elderly_complex": {
                "age": 75,
                "sex": "male",
                "bmi": 26,
                "ckd": True,
                "inflammation": True,
                "polypharmacy": True,
                "genetic_risk": 0.8,
                "modifiers": {"CRP": 1.6, "PCSK9": 1.2, "LPA": 1.4},
            },
            "postmenopausal_lipid": {
                "age": 62,
                "sex": "female",
                "bmi": 27,
                "menopause": True,
                "hyperlipidemia": True,
                "genetic_risk": 0.6,
                "modifiers": {"HMGCR": 1.3, "CETP": 1.4, "APOB": 1.2},
            },
        }

    def _create_biomarker_profiles(self) -> Dict:
        """Create biomarker characteristics for personalization"""
        return {
            "APOB": {
                "primary_pathway": "lipid_transport",
                "age_sensitivity": 0.8,
                "sex_effect": "male_higher",
                "comorbidity_modifiers": {"diabetes": 1.3, "ckd": 1.2},
                "volatility": "low",
                "response_time": "weeks",
            },
            "HMGCR": {
                "primary_pathway": "cholesterol_synthesis",
                "age_sensitivity": 0.6,
                "sex_effect": "minimal",
                "comorbidity_modifiers": {"hyperlipidemia": 1.4},
                "volatility": "medium",
                "response_time": "days",
            },
            "PCSK9": {
                "primary_pathway": "ldl_regulation",
                "age_sensitivity": 0.7,
                "sex_effect": "female_higher",
                "comorbidity_modifiers": {"ckd": 1.5, "inflammation": 1.3},
                "volatility": "medium",
                "response_time": "weeks",
            },
            "CRP": {
                "primary_pathway": "inflammation",
                "age_sensitivity": 0.9,
                "sex_effect": "female_higher",
                "comorbidity_modifiers": {"diabetes": 1.4, "ckd": 1.6},
                "volatility": "high",
                "response_time": "hours",
            },
            "LPA": {
                "primary_pathway": "genetic_lipid",
                "age_sensitivity": 0.3,
                "sex_effect": "minimal",
                "comorbidity_modifiers": {"family_history": 2.0},
                "volatility": "very_low",
                "response_time": "genetic",
            },
        }

    def generate_personalized_panel(self, patient_type: str) -> Dict:
        """Generate personalized biomarker panel for patient archetype"""

        if patient_type not in self.patient_archetypes:
            raise ValueError(f"Unknown patient type: {patient_type}")

        patient = self.patient_archetypes[patient_type]

        # Apply personalization modifiers to population rankings
        personalized_scores = {}
        for biomarker, pop_data in self.cv_biomarkers.items():
            base_score = pop_data["precision_20"]

            # Apply patient-specific modifiers
            modifier = patient.get("modifiers", {}).get(biomarker, 1.0)

            # Apply age effects
            age_effect = self._calculate_age_effect(biomarker, patient["age"])

            # Apply comorbidity effects
            comorbidity_effect = self._calculate_comorbidity_effect(biomarker, patient)

            # Calculate final personalized score
            personalized_score = base_score * modifier * age_effect * comorbidity_effect

            personalized_scores[biomarker] = {
                "base_score": base_score,
                "modifier": modifier,
                "age_effect": age_effect,
                "comorbidity_effect": comorbidity_effect,
                "final_score": personalized_score,
            }

        # Rank biomarkers by personalized scores
        ranked_biomarkers = sorted(
            personalized_scores.items(), key=lambda x: x[1]["final_score"], reverse=True
        )

        # Generate monitoring schedule
        monitoring_schedule = self._generate_monitoring_schedule(
            ranked_biomarkers, patient
        )

        # Generate clinical interpretation
        interpretation = self._generate_clinical_interpretation(
            ranked_biomarkers, patient_type
        )

        return {
            "patient_type": patient_type,
            "patient_profile": patient,
            "biomarker_rankings": ranked_biomarkers,
            "top_5_biomarkers": [b[0] for b in ranked_biomarkers[:5]],
            "monitoring_schedule": monitoring_schedule,
            "clinical_interpretation": interpretation,
        }

    def _calculate_age_effect(self, biomarker: str, age: int) -> float:
        """Calculate age-based modifier for biomarker"""
        profile = self.biomarker_profiles.get(biomarker, {})
        age_sensitivity = profile.get("age_sensitivity", 0.5)

        # Age effect increases with age for most biomarkers
        if age < 40:
            return 1.0 - (age_sensitivity * 0.2)  # Slightly lower in young
        elif age < 65:
            return 1.0  # Baseline for middle age
        else:
            return 1.0 + (age_sensitivity * 0.3)  # Higher in elderly

    def _calculate_comorbidity_effect(self, biomarker: str, patient: Dict) -> float:
        """Calculate comorbidity-based modifier for biomarker"""
        profile = self.biomarker_profiles.get(biomarker, {})
        modifiers = profile.get("comorbidity_modifiers", {})

        effect = 1.0
        for condition, modifier in modifiers.items():
            if patient.get(condition, False):
                effect *= modifier

        return effect

    def _generate_monitoring_schedule(
        self, ranked_biomarkers: List, patient: Dict
    ) -> Dict:
        """Generate personalized monitoring schedule"""
        schedule = {}

        for i, (biomarker, scores) in enumerate(ranked_biomarkers[:5]):
            profile = self.biomarker_profiles.get(biomarker, {})

            # Base frequency depends on ranking and volatility
            if i < 2:  # Top 2 biomarkers
                base_freq = 30  # Every month
            elif i < 4:  # Next 2 biomarkers
                base_freq = 60  # Every 2 months
            else:  # 5th biomarker
                base_freq = 90  # Every 3 months

            # Adjust for volatility
            volatility = profile.get("volatility", "medium")
            if volatility == "high":
                base_freq = max(base_freq // 2, 14)  # More frequent, min 2 weeks
            elif volatility == "low":
                base_freq = min(base_freq * 2, 180)  # Less frequent, max 6 months

            # Adjust for age and risk
            if patient["age"] > 65:
                base_freq = max(base_freq // 1.5, 14)  # More frequent for elderly

            schedule[biomarker] = {
                "frequency_days": int(base_freq),
                "priority": i + 1,
                "volatility": volatility,
                "rationale": f"Rank #{i+1}, {volatility} volatility, age {patient['age']}",
            }

        return schedule

    def _generate_clinical_interpretation(
        self, ranked_biomarkers: List, patient_type: str
    ) -> str:
        """Generate clinical interpretation for the personalized panel"""

        top_3 = [b[0] for b in ranked_biomarkers[:3]]

        interpretations = {
            "young_athletic": f"""
            For young athletic patients with family history:
            - Focus on genetic markers ({top_3[0]}, {top_3[1]}) for early detection
            - Monitor HDL pathway markers for protective factors
            - Emphasis on lifestyle-responsive biomarkers
            """,
            "middle_aged_metabolic": f"""
            For middle-aged patients with metabolic syndrome:
            - Prioritize inflammatory markers ({top_3[0]}, {top_3[1]}) 
            - Monitor insulin resistance and lipid dysregulation
            - Focus on modifiable risk factors
            """,
            "elderly_complex": f"""
            For elderly patients with multiple comorbidities:
            - Emphasize inflammatory cascade markers ({top_3[0]}, {top_3[1]})
            - Monitor kidney function interactions with CV risk
            - Focus on prognostic rather than just diagnostic markers
            """,
            "postmenopausal_lipid": f"""
            For postmenopausal women with lipid disorders:
            - Prioritize cholesterol metabolism markers ({top_3[0]}, {top_3[1]})
            - Monitor hormone-responsive pathways
            - Focus on statin response and lipid particle composition
            """,
        }

        return interpretations.get(
            patient_type, "Standard cardiovascular risk assessment"
        )

    def compare_population_vs_personalized(self) -> pd.DataFrame:
        """Compare population-level vs personalized rankings across patient types"""

        results = []

        # Population ranking (baseline)
        pop_ranking = list(self.cv_biomarkers.keys())

        for patient_type in self.patient_archetypes.keys():
            panel = self.generate_personalized_panel(patient_type)
            personal_ranking = panel["top_5_biomarkers"]

            # Calculate ranking changes
            for i, biomarker in enumerate(personal_ranking):
                pop_rank = pop_ranking.index(biomarker) + 1
                personal_rank = i + 1
                rank_change = pop_rank - personal_rank

                results.append(
                    {
                        "patient_type": patient_type,
                        "biomarker": biomarker,
                        "population_rank": pop_rank,
                        "personalized_rank": personal_rank,
                        "rank_change": rank_change,
                        "improvement": rank_change > 0,
                    }
                )

        return pd.DataFrame(results)

    def demonstrate_temporal_predictions(
        self, patient_type: str, biomarker: str
    ) -> Dict:
        """Demonstrate biomarker trajectory prediction for patient"""

        patient = self.patient_archetypes[patient_type]
        profile = self.biomarker_profiles.get(biomarker, {})

        # Simulate biomarker trajectory over 1 year
        days = np.arange(0, 365, 7)  # Weekly measurements

        # Base level influenced by patient characteristics
        base_level = np.random.normal(100, 10)  # Arbitrary units

        # Add patient-specific trends
        trend = 0
        if patient.get("diabetes"):
            trend += 0.1  # Upward trend
        if patient.get("ckd"):
            trend += 0.05
        if patient["age"] > 65:
            trend += 0.05

        # Add biomarker-specific characteristics
        volatility_map = {"low": 5, "medium": 10, "high": 20, "very_low": 2}
        noise_level = volatility_map.get(profile.get("volatility", "medium"), 10)

        # Generate trajectory
        trajectory = []
        for day in days:
            level = base_level + (trend * day / 365 * base_level)
            level += np.random.normal(0, noise_level)
            trajectory.append(max(level, 0))  # No negative values

        # Identify risk periods (values in top 25%)
        risk_threshold = np.percentile(trajectory, 75)
        risk_periods = [i for i, val in enumerate(trajectory) if val > risk_threshold]

        return {
            "patient_type": patient_type,
            "biomarker": biomarker,
            "days": days.tolist(),
            "values": trajectory,
            "baseline": base_level,
            "trend": trend,
            "risk_threshold": risk_threshold,
            "risk_periods": risk_periods,
            "volatility": profile.get("volatility", "medium"),
        }


# Demonstration functions
def run_personalization_demo():
    """Run comprehensive personalization demonstration"""

    engine = PersonalizedBiomarkerDemo()

    print("🧬 PERSONALIZED BIOMARKER DISCOVERY DEMONSTRATION")
    print("=" * 60)

    # Generate panels for each patient archetype
    for patient_type in engine.patient_archetypes.keys():
        print(f"\n📊 Patient Type: {patient_type.replace('_', ' ').title()}")
        print("-" * 40)

        panel = engine.generate_personalized_panel(patient_type)

        print(
            f"Patient Profile: Age {panel['patient_profile']['age']}, "
            f"{panel['patient_profile']['sex']}"
        )

        print("\nTop 5 Personalized Biomarkers:")
        for i, biomarker in enumerate(panel["top_5_biomarkers"], 1):
            schedule = panel["monitoring_schedule"][biomarker]
            print(f"  {i}. {biomarker} (every {schedule['frequency_days']} days)")

        print("\nClinical Interpretation:")
        print(panel["clinical_interpretation"].strip())

    # Show comparison table
    print("\n📈 POPULATION vs PERSONALIZED RANKINGS")
    print("=" * 60)

    comparison = engine.compare_population_vs_personalized()

    # Show rank changes for each patient type
    for patient_type in engine.patient_archetypes.keys():
        patient_data = comparison[comparison["patient_type"] == patient_type]
        print(f"\n{patient_type.replace('_', ' ').title()}:")

        for _, row in patient_data.iterrows():
            change_str = (
                "↑"
                if row["rank_change"] > 0
                else "↓" if row["rank_change"] < 0 else "="
            )
            print(
                f"  {row['biomarker']}: #{row['population_rank']} → #{row['personalized_rank']} {change_str}"
            )

    # Demonstrate temporal prediction
    print("\n⏰ BIOMARKER TRAJECTORY PREDICTION")
    print("=" * 60)

    trajectory = engine.demonstrate_temporal_predictions("elderly_complex", "CRP")
    print("Predicting CRP levels for elderly complex patient over 1 year:")
    print(f"  Baseline: {trajectory['baseline']:.1f}")
    print(f"  Trend: {trajectory['trend']:.3f} (annual rate)")
    print(f"  Volatility: {trajectory['volatility']}")
    print(f"  Risk periods: {len(trajectory['risk_periods'])} weeks above threshold")


if __name__ == "__main__":
    run_personalization_demo()
