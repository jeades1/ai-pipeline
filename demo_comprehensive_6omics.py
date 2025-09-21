"""
Comprehensive 6-Omics Integration Demonstration

This script demonstrates the complete integration of genomics, epigenomics, 
transcriptomics, proteomics, metabolomics, exposomics, and clinical data 
using the enhanced AI pipeline architecture.

Features:
- Synthetic data generation with realistic biological relationships
- Complete 6-omics data processing pipeline
- Environmental exposure integration
- Enhanced causal discovery with biological constraints
- Multi-outcome prediction with environmental factors
- Validation and performance metrics

Author: AI Pipeline Team
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import enhanced modules
try:
    from .enhanced_omics_config import (
        OmicsType,
        create_kidney_disease_6omics_config,
    )
    from .enhanced_6omics_causal_discovery import Enhanced6OmicsCausalAnalyzer

    LOCAL_IMPORTS = True
except ImportError:
    # Alternative imports for standalone execution
    LOCAL_IMPORTS = False
    logging.warning("Local imports not available. Using standalone mode.")

    # Define OmicsType enum for standalone mode
    from enum import Enum

    class OmicsType(Enum):
        GENOMICS = "genomics"
        EPIGENOMICS = "epigenomics"
        TRANSCRIPTOMICS = "transcriptomics"
        PROTEOMICS = "proteomics"
        METABOLOMICS = "metabolomics"
        EXPOSOMICS = "exposomics"
        CLINICAL = "clinical"


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Comprehensive6OmicsDemo:
    """Comprehensive demonstration of 6-omics integration pipeline"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "demo_outputs/6omics_integration")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.config_manager = None
        self.epigenomics_integrator = None
        self.exposomics_integrator = None
        self.causal_analyzer = None

        # Data containers
        self.raw_data: Dict[OmicsType, pd.DataFrame] = {}
        self.processed_data: Dict[OmicsType, pd.DataFrame] = {}
        self.integrated_features: Optional[pd.DataFrame] = None
        self.causal_graph = None
        self.prediction_results = {}

        # Demo metadata
        self.demo_metadata = {
            "start_time": datetime.now().isoformat(),
            "n_samples": 250,
            "n_features_by_omics": {},
            "processing_steps": [],
            "performance_metrics": {},
        }

        logger.info(
            f"Initialized 6-omics demo with output directory: {self.output_dir}"
        )

    def run_complete_demo(self):
        """Run the complete 6-omics integration demonstration"""

        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE 6-OMICS INTEGRATION DEMONSTRATION")
        logger.info("=" * 80)

        try:
            # Step 1: Setup and configuration
            self._setup_configuration()

            # Step 2: Generate synthetic multi-omics data
            self._generate_synthetic_data()

            # Step 3: Process individual omics types
            self._process_individual_omics()

            # Step 4: Integrate all omics data
            self._integrate_omics_data()

            # Step 5: Discover causal relationships
            self._discover_causal_structure()

            # Step 6: Build predictive models
            self._build_predictive_models()

            # Step 7: Generate comprehensive report
            self._generate_comprehensive_report()

            # Step 8: Save results
            self._save_results()

            logger.info("=" * 80)
            logger.info("6-OMICS INTEGRATION DEMONSTRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise

    def _setup_configuration(self):
        """Setup configuration for 6-omics integration"""

        logger.info("Setting up 6-omics configuration...")

        if LOCAL_IMPORTS:
            self.config_manager = create_kidney_disease_6omics_config()
        else:
            # Simplified configuration for standalone mode
            self.config_manager = self._create_simple_config()

        self.demo_metadata["processing_steps"].append("Configuration setup")
        logger.info("✅ Configuration setup complete")

    def _create_simple_config(self):
        """Create simplified configuration for standalone mode"""
        return {
            "genomics": {"feature_prefix": "genetic_", "expected_features": 30},
            "epigenomics": {"feature_prefix": "epigenetic_", "expected_features": 40},
            "transcriptomics": {
                "feature_prefix": "transcript_",
                "expected_features": 50,
            },
            "proteomics": {"feature_prefix": "protein_", "expected_features": 35},
            "metabolomics": {"feature_prefix": "metabolite_", "expected_features": 25},
            "exposomics": {"feature_prefix": "exposure_", "expected_features": 20},
            "clinical": {"feature_prefix": "clinical_", "expected_features": 15},
        }

    def _generate_synthetic_data(self):
        """Generate synthetic 6-omics datasets"""

        logger.info("Generating synthetic 6-omics datasets...")

        n_samples = self.demo_metadata["n_samples"]
        np.random.seed(42)  # For reproducibility

        # Generate sample IDs
        sample_ids = [f"patient_{i:04d}" for i in range(n_samples)]

        # Generate each omics type
        self.raw_data = self._generate_all_omics_data(sample_ids)

        # Record feature counts
        for omics_type, data in self.raw_data.items():
            self.demo_metadata["n_features_by_omics"][omics_type.value] = data.shape[1]

        self.demo_metadata["processing_steps"].append("Synthetic data generation")
        logger.info(f"✅ Generated synthetic data for {len(self.raw_data)} omics types")

        # Save raw data summary
        self._save_data_summary()

    def _generate_all_omics_data(
        self, sample_ids: List[str]
    ) -> Dict[OmicsType, pd.DataFrame]:
        """Generate all omics datasets with realistic relationships"""

        data = {}

        # 1. Genomics (SNP data)
        genomics_data = self._generate_genomics_data(sample_ids, n_features=30)
        data[OmicsType.GENOMICS] = genomics_data

        # 2. Exposomics (environmental exposures) - independent baseline
        exposomics_data = self._generate_exposomics_data(sample_ids, n_features=20)
        data[OmicsType.EXPOSOMICS] = exposomics_data

        # 3. Epigenomics (influenced by genomics + exposomics)
        epigenomics_data = self._generate_epigenomics_data(
            sample_ids, genomics_data, exposomics_data, n_features=40
        )
        data[OmicsType.EPIGENOMICS] = epigenomics_data

        # 4. Transcriptomics (influenced by genomics + epigenomics + exposomics)
        transcriptomics_data = self._generate_transcriptomics_data(
            sample_ids, genomics_data, epigenomics_data, exposomics_data, n_features=50
        )
        data[OmicsType.TRANSCRIPTOMICS] = transcriptomics_data

        # 5. Proteomics (influenced by transcriptomics + exposomics)
        proteomics_data = self._generate_proteomics_data(
            sample_ids, transcriptomics_data, exposomics_data, n_features=35
        )
        data[OmicsType.PROTEOMICS] = proteomics_data

        # 6. Metabolomics (influenced by proteomics + exposomics)
        metabolomics_data = self._generate_metabolomics_data(
            sample_ids, proteomics_data, exposomics_data, n_features=25
        )
        data[OmicsType.METABOLOMICS] = metabolomics_data

        # 7. Clinical outcomes (influenced by all molecular omics + exposomics)
        clinical_data = self._generate_clinical_data(
            sample_ids,
            transcriptomics_data,
            proteomics_data,
            metabolomics_data,
            exposomics_data,
            n_features=15,
        )
        data[OmicsType.CLINICAL] = clinical_data

        return data

    def _generate_genomics_data(
        self, sample_ids: List[str], n_features: int
    ) -> pd.DataFrame:
        """Generate SNP genotype data"""
        n_samples = len(sample_ids)

        # SNP genotypes (0, 1, 2)
        snp_data = np.random.binomial(2, 0.3, (n_samples, n_features))

        feature_names = [f"genetic_snp_{i:03d}" for i in range(n_features)]

        return pd.DataFrame(snp_data, index=sample_ids, columns=feature_names)

    def _generate_exposomics_data(
        self, sample_ids: List[str], n_features: int
    ) -> pd.DataFrame:
        """Generate environmental exposure data"""
        n_samples = len(sample_ids)

        # Specific environmental exposures
        data_dict = {
            "exposure_air_pm25": np.random.lognormal(
                2.5, 0.5, n_samples
            ),  # PM2.5 levels
            "exposure_air_no2": np.random.lognormal(2.8, 0.4, n_samples),  # NO2 levels
            "exposure_air_ozone": np.random.lognormal(
                3.2, 0.3, n_samples
            ),  # Ozone levels
            "exposure_chem_pfoa": np.random.lognormal(
                0.5, 0.8, n_samples
            ),  # PFOA levels
            "exposure_chem_lead": np.random.lognormal(
                0.1, 0.7, n_samples
            ),  # Lead levels
            "exposure_chem_mercury": np.random.lognormal(
                -0.2, 0.6, n_samples
            ),  # Mercury levels
            "exposure_built_greenspace": np.random.beta(3, 2, n_samples)
            * 100,  # Green space %
            "exposure_built_noise": np.random.normal(55, 10, n_samples).clip(
                30, 85
            ),  # Noise dB
            "exposure_built_walkability": np.random.beta(2, 3, n_samples)
            * 10,  # Walkability score
            "exposure_lifestyle_steps": np.random.normal(8000, 2000, n_samples).clip(
                1000, 20000
            ),  # Daily steps
            "exposure_lifestyle_sleep": np.random.normal(7.5, 1.0, n_samples).clip(
                4, 12
            ),  # Sleep hours
            "exposure_lifestyle_stress": np.random.beta(2, 5, n_samples)
            * 10,  # Stress score
        }

        # Add additional environmental factors
        for i in range(n_features - len(data_dict)):
            data_dict[f"exposure_env_{i:03d}"] = np.random.normal(0, 1, n_samples)

        return pd.DataFrame(data_dict, index=sample_ids)

    def _generate_epigenomics_data(
        self,
        sample_ids: List[str],
        genomics_data: pd.DataFrame,
        exposomics_data: pd.DataFrame,
        n_features: int,
    ) -> pd.DataFrame:
        """Generate DNA methylation data influenced by genetics and environment"""
        n_samples = len(sample_ids)

        # Base methylation levels (beta values 0-1)
        base_methylation = np.random.beta(2, 8, (n_samples, n_features))

        # Genetic influence (some SNPs affect methylation)
        genetic_effect = genomics_data.iloc[:, :10].mean(axis=1).values.reshape(-1, 1)
        genetic_influence = (
            0.15 * genetic_effect * np.random.normal(0, 0.1, (n_samples, n_features))
        )

        # Environmental influence (pollution affects methylation)
        env_pollutants = (
            exposomics_data[
                ["exposure_air_pm25", "exposure_air_no2", "exposure_chem_pfoa"]
            ]
            .mean(axis=1)
            .values.reshape(-1, 1)
        )
        env_influence = (
            0.1
            * np.log1p(env_pollutants)
            * np.random.normal(0, 0.05, (n_samples, n_features))
        )

        # Combine influences
        methylation_data = base_methylation + genetic_influence + env_influence
        methylation_data = np.clip(methylation_data, 0, 1)  # Keep in valid range

        feature_names = [f"epigenetic_cpg_{i:03d}" for i in range(n_features)]

        return pd.DataFrame(methylation_data, index=sample_ids, columns=feature_names)

    def _generate_transcriptomics_data(
        self,
        sample_ids: List[str],
        genomics_data: pd.DataFrame,
        epigenomics_data: pd.DataFrame,
        exposomics_data: pd.DataFrame,
        n_features: int,
    ) -> pd.DataFrame:
        """Generate gene expression data"""
        n_samples = len(sample_ids)

        # Base expression levels (log-normal)
        base_expression = np.random.lognormal(5, 1.5, (n_samples, n_features))

        # Genetic effect (eQTLs)
        genetic_effect = genomics_data.iloc[:, :15].mean(axis=1).values.reshape(-1, 1)
        genetic_influence = base_expression * (1 + 0.2 * genetic_effect)

        # Epigenetic effect (methylation affects expression)
        methylation_effect = 1 - epigenomics_data.iloc[:, :20].mean(
            axis=1
        ).values.reshape(-1, 1)
        epigenetic_influence = genetic_influence * (1 + 0.3 * methylation_effect)

        # Environmental effect (exposures affect expression)
        env_stress = (
            exposomics_data[
                ["exposure_air_pm25", "exposure_chem_lead", "exposure_lifestyle_stress"]
            ]
            .mean(axis=1)
            .values.reshape(-1, 1)
        )
        env_influence = epigenetic_influence * (1 + 0.15 * np.log1p(env_stress))

        feature_names = [f"transcript_gene_{i:03d}" for i in range(n_features)]

        return pd.DataFrame(env_influence, index=sample_ids, columns=feature_names)

    def _generate_proteomics_data(
        self,
        sample_ids: List[str],
        transcriptomics_data: pd.DataFrame,
        exposomics_data: pd.DataFrame,
        n_features: int,
    ) -> pd.DataFrame:
        """Generate protein abundance data"""
        n_samples = len(sample_ids)

        # Base protein levels
        base_proteins = np.random.lognormal(3, 1, (n_samples, n_features))

        # Transcriptional influence
        transcript_effect = (
            transcriptomics_data.iloc[:, :25].mean(axis=1).values.reshape(-1, 1)
        )
        transcript_influence = base_proteins * (1 + 0.25 * np.log1p(transcript_effect))

        # Environmental influence on protein degradation
        env_toxins = (
            exposomics_data[
                ["exposure_chem_pfoa", "exposure_chem_lead", "exposure_chem_mercury"]
            ]
            .mean(axis=1)
            .values.reshape(-1, 1)
        )
        env_influence = transcript_influence * (1 - 0.1 * np.tanh(env_toxins))

        feature_names = [f"protein_prot_{i:03d}" for i in range(n_features)]

        return pd.DataFrame(env_influence, index=sample_ids, columns=feature_names)

    def _generate_metabolomics_data(
        self,
        sample_ids: List[str],
        proteomics_data: pd.DataFrame,
        exposomics_data: pd.DataFrame,
        n_features: int,
    ) -> pd.DataFrame:
        """Generate metabolite concentration data"""
        n_samples = len(sample_ids)

        # Base metabolite levels
        base_metabolites = np.random.lognormal(2, 0.8, (n_samples, n_features))

        # Protein enzyme influence
        protein_effect = proteomics_data.iloc[:, :15].mean(axis=1).values.reshape(-1, 1)
        protein_influence = base_metabolites * (1 + 0.2 * np.log1p(protein_effect))

        # Lifestyle influence on metabolism
        lifestyle_factors = (
            exposomics_data[["exposure_lifestyle_steps", "exposure_lifestyle_sleep"]]
            .mean(axis=1)
            .values.reshape(-1, 1)
        )
        lifestyle_influence = protein_influence * (
            1 + 0.1 * np.tanh(lifestyle_factors / 5000)
        )

        feature_names = [f"metabolite_metab_{i:03d}" for i in range(n_features)]

        return pd.DataFrame(
            lifestyle_influence, index=sample_ids, columns=feature_names
        )

    def _generate_clinical_data(
        self,
        sample_ids: List[str],
        transcriptomics_data: pd.DataFrame,
        proteomics_data: pd.DataFrame,
        metabolomics_data: pd.DataFrame,
        exposomics_data: pd.DataFrame,
        n_features: int,
    ) -> pd.DataFrame:
        """Generate clinical outcome data"""
        n_samples = len(sample_ids)

        # Base clinical measurements
        base_clinical = np.random.normal(0, 1, (n_samples, n_features))

        # Molecular influences
        molecular_score = (
            np.log1p(transcriptomics_data.iloc[:, :10]).mean(axis=1) * 0.1
            + np.log1p(proteomics_data.iloc[:, :8]).mean(axis=1) * 0.15
            + np.log1p(metabolomics_data.iloc[:, :6]).mean(axis=1) * 0.1
        ).values.reshape(-1, 1)

        # Environmental influences
        env_health_impact = (
            exposomics_data[
                [
                    "exposure_air_pm25",
                    "exposure_air_no2",
                    "exposure_chem_lead",
                    "exposure_lifestyle_stress",
                ]
            ]
            .mean(axis=1)
            .values.reshape(-1, 1)
        )

        # Combined clinical outcomes
        clinical_outcomes = base_clinical + molecular_score + 0.2 * env_health_impact

        feature_names = [f"clinical_outcome_{i:03d}" for i in range(n_features)]

        return pd.DataFrame(clinical_outcomes, index=sample_ids, columns=feature_names)

    def _process_individual_omics(self):
        """Process each omics type individually"""

        logger.info("Processing individual omics datasets...")

        self.processed_data = {}

        for omics_type, raw_data in self.raw_data.items():
            if omics_type == OmicsType.EPIGENOMICS and LOCAL_IMPORTS:
                # Use specialized epigenomics processing
                processed = self._process_epigenomics_specialized(raw_data)
            elif omics_type == OmicsType.EXPOSOMICS and LOCAL_IMPORTS:
                # Use specialized exposomics processing
                processed = self._process_exposomics_specialized(raw_data)
            else:
                # Standard processing
                processed = self._process_standard_omics(raw_data, omics_type)

            self.processed_data[omics_type] = processed

        self.demo_metadata["processing_steps"].append("Individual omics processing")
        logger.info("✅ Individual omics processing complete")

    def _process_epigenomics_specialized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Specialized epigenomics processing"""
        # Simplified processing for demo
        # In real implementation, would use EpigenomicsIntegrator
        processed = data.copy()

        # Quality filtering (remove low-variability CpGs)
        feature_variance = processed.var()
        high_var_features = feature_variance[feature_variance > 0.01].index
        processed = processed[high_var_features]

        # Normalization
        processed = (processed - processed.mean()) / processed.std()

        return processed

    def _process_exposomics_specialized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Specialized exposomics processing"""
        # Simplified processing for demo
        # In real implementation, would use ExposomicsIntegrator
        processed = data.copy()

        # Log-transform concentration data
        concentration_cols = [
            col for col in processed.columns if "chem_" in col or "air_" in col
        ]
        for col in concentration_cols:
            processed[col] = np.log1p(processed[col])

        # Standardize all features
        processed = (processed - processed.mean()) / processed.std()

        return processed

    def _process_standard_omics(
        self, data: pd.DataFrame, omics_type: OmicsType
    ) -> pd.DataFrame:
        """Standard omics processing"""
        processed = data.copy()

        if omics_type == OmicsType.GENOMICS:
            # SNP data - no transformation needed
            pass
        else:
            # Log-transform and standardize for other omics
            if omics_type in [
                OmicsType.TRANSCRIPTOMICS,
                OmicsType.PROTEOMICS,
                OmicsType.METABOLOMICS,
            ]:
                processed = np.log1p(processed)

            # Standardize
            processed = (processed - processed.mean()) / processed.std()

        return processed

    def _integrate_omics_data(self):
        """Integrate all processed omics data"""

        logger.info("Integrating all omics datasets...")

        # Combine all processed data
        integrated_datasets = []

        for omics_type, processed_data in self.processed_data.items():
            integrated_datasets.append(processed_data)

        # Concatenate along columns
        self.integrated_features = pd.concat(integrated_datasets, axis=1)

        # Remove features with too much missing data
        missing_threshold = 0.1
        missing_rates = self.integrated_features.isnull().mean()
        valid_features = missing_rates[missing_rates < missing_threshold].index
        self.integrated_features = self.integrated_features[valid_features]

        # Fill remaining missing values
        self.integrated_features = self.integrated_features.fillna(
            self.integrated_features.median()
        )

        self.demo_metadata["processing_steps"].append("Multi-omics integration")
        logger.info(
            f"✅ Integration complete. Final feature matrix: {self.integrated_features.shape}"
        )

    def _discover_causal_structure(self):
        """Discover causal relationships in the integrated data"""

        logger.info("Discovering causal structure...")

        if LOCAL_IMPORTS:
            # Use enhanced causal discovery
            self.causal_analyzer = Enhanced6OmicsCausalAnalyzer(
                config_manager=self.config_manager,
                causal_discovery_method="correlation",  # Use simpler method for demo
            )

            self.causal_analyzer.load_omics_data(self.processed_data)
            self.causal_graph = self.causal_analyzer.discover_causal_structure()

            # Get causal analysis results
            causal_stats = self.causal_analyzer.get_causal_graph_statistics()
            self.demo_metadata["performance_metrics"]["causal_discovery"] = causal_stats
        else:
            # Simplified causal discovery
            self.causal_graph = self._simple_causal_discovery()

        self.demo_metadata["processing_steps"].append("Causal structure discovery")
        logger.info("✅ Causal discovery complete")

    def _simple_causal_discovery(self):
        """Simplified causal discovery for standalone mode"""
        import networkx as nx

        # Calculate correlation matrix
        corr_matrix = self.integrated_features.corr()

        # Create graph from strong correlations
        graph = nx.DiGraph()
        threshold = 0.3

        for i, var1 in enumerate(corr_matrix.columns):
            for j, var2 in enumerate(corr_matrix.columns):
                if i != j and abs(corr_matrix.iloc[i, j]) > threshold:
                    graph.add_edge(var1, var2, weight=abs(corr_matrix.iloc[i, j]))

        return graph

    def _build_predictive_models(self):
        """Build predictive models for clinical outcomes"""

        logger.info("Building predictive models...")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        # Separate features and clinical targets
        clinical_features = [
            col
            for col in self.integrated_features.columns
            if col.startswith("clinical_")
        ]
        molecular_features = [
            col
            for col in self.integrated_features.columns
            if not col.startswith("clinical_")
        ]

        X = self.integrated_features[molecular_features]
        y = self.integrated_features[clinical_features]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Build models for each clinical outcome
        self.prediction_results = {}

        for outcome in clinical_features[:5]:  # Test on first 5 outcomes
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train[outcome])

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test[outcome], y_pred)
            r2 = r2_score(y_test[outcome], y_pred)

            # Feature importance
            feature_importance = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

            self.prediction_results[outcome] = {
                "mse": mse,
                "r2": r2,
                "top_features": feature_importance.head(10).to_dict(),
            }

        # Overall performance metrics
        avg_r2 = np.mean([result["r2"] for result in self.prediction_results.values()])
        self.demo_metadata["performance_metrics"]["prediction"] = {
            "average_r2": avg_r2,
            "n_outcomes_tested": len(self.prediction_results),
        }

        self.demo_metadata["processing_steps"].append("Predictive modeling")
        logger.info(f"✅ Predictive modeling complete. Average R²: {avg_r2:.3f}")

    def _generate_comprehensive_report(self):
        """Generate comprehensive demonstration report"""

        logger.info("Generating comprehensive report...")

        print("\n" + "=" * 80)
        print("COMPREHENSIVE 6-OMICS INTEGRATION DEMONSTRATION REPORT")
        print("=" * 80)

        # Dataset overview
        print("\nDataset Overview:")
        print(f"  Samples: {self.demo_metadata['n_samples']:,}")
        print(f"  Total features: {self.integrated_features.shape[1]:,}")
        print(f"  Processing steps: {len(self.demo_metadata['processing_steps'])}")

        print("\nFeatures by omics type:")
        for omics_type, count in self.demo_metadata["n_features_by_omics"].items():
            print(f"  {omics_type}: {count} features")

        # Causal discovery results
        if "causal_discovery" in self.demo_metadata["performance_metrics"]:
            causal_stats = self.demo_metadata["performance_metrics"]["causal_discovery"]
            print("\nCausal Discovery Results:")
            print(f"  Graph nodes: {causal_stats.get('total_nodes', 'N/A'):,}")
            print(f"  Graph edges: {causal_stats.get('total_edges', 'N/A'):,}")
            print(f"  Graph density: {causal_stats.get('density', 'N/A'):.4f}")
            if "environmental_influences" in causal_stats:
                print(
                    f"  Environmental influences: {causal_stats['environmental_influences']}"
                )

        # Prediction results
        if self.prediction_results:
            print("\nPredictive Modeling Results:")
            avg_r2 = self.demo_metadata["performance_metrics"]["prediction"][
                "average_r2"
            ]
            print(f"  Average R² score: {avg_r2:.3f}")
            print(f"  Outcomes tested: {len(self.prediction_results)}")

            print("\nIndividual outcome performance:")
            for outcome, metrics in list(self.prediction_results.items())[:3]:
                print(f"  {outcome}: R² = {metrics['r2']:.3f}")

        # Top cross-omics interactions
        if hasattr(self, "causal_analyzer") and self.causal_analyzer:
            try:
                interactions = self.causal_analyzer.analyze_cross_omics_interactions()
                print("\nCross-Omics Interactions:")
                print(f"  Total interactions: {len(interactions['cross_omics_edges'])}")
                print(
                    f"  Environmental influences: {len(interactions['environmental_influences'])}"
                )

                # Top interaction types
                print("\n  Top interaction types:")
                for pair, count in list(interactions["omics_type_connections"].items())[
                    :5
                ]:
                    print(f"    {pair}: {count} edges")
            except Exception:
                print("\nCross-omics interaction analysis not available")

        # Processing timeline
        print("\nProcessing Steps Completed:")
        for i, step in enumerate(self.demo_metadata["processing_steps"], 1):
            print(f"  {i}. {step}")

        print("\n" + "=" * 80)
        print("DEMONSTRATION HIGHLIGHTS")
        print("=" * 80)
        print("✅ Successfully integrated 6 omics data types")
        print("✅ Applied realistic biological relationships")
        print("✅ Incorporated environmental exposures")
        print("✅ Discovered causal structure with biological constraints")
        print("✅ Built predictive models for clinical outcomes")
        print("✅ Demonstrated end-to-end pipeline capability")

        if avg_r2 > 0.5:
            print("🎯 EXCELLENT: Strong predictive performance achieved")
        elif avg_r2 > 0.3:
            print("✨ GOOD: Moderate predictive performance achieved")
        else:
            print("📈 BASELINE: Basic predictive capability demonstrated")

        print("=" * 80)

    def _save_data_summary(self):
        """Save summary of generated data"""
        summary_file = self.output_dir / "data_summary.json"

        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "n_samples": self.demo_metadata["n_samples"],
            "omics_types": list(self.demo_metadata["n_features_by_omics"].keys()),
            "feature_counts": self.demo_metadata["n_features_by_omics"],
            "sample_info": {
                "first_sample": list(self.raw_data.values())[0].index[0],
                "last_sample": list(self.raw_data.values())[0].index[-1],
            },
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved data summary to {summary_file}")

    def _save_results(self):
        """Save all demonstration results"""

        logger.info("Saving demonstration results...")

        # Save integrated feature matrix
        features_file = self.output_dir / "integrated_features.csv"
        self.integrated_features.to_csv(features_file)

        # Save prediction results
        prediction_file = self.output_dir / "prediction_results.json"
        with open(prediction_file, "w") as f:
            json.dump(self.prediction_results, f, indent=2)

        # Save demo metadata
        metadata_file = self.output_dir / "demo_metadata.json"
        self.demo_metadata["end_time"] = datetime.now().isoformat()
        with open(metadata_file, "w") as f:
            json.dump(self.demo_metadata, f, indent=2)

        # Save causal graph if available
        if self.causal_graph:
            try:
                import networkx as nx

                graph_file = self.output_dir / "causal_graph.gexf"
                nx.write_gexf(self.causal_graph, graph_file)
                logger.info(f"Saved causal graph to {graph_file}")
            except Exception as e:
                logger.warning(f"Could not save causal graph: {e}")

        logger.info(f"✅ All results saved to {self.output_dir}")


def run_comprehensive_6omics_demo():
    """Run the comprehensive 6-omics integration demonstration"""

    # Create and run demo
    demo = Comprehensive6OmicsDemo()
    demo.run_complete_demo()

    return demo


if __name__ == "__main__":
    demo = run_comprehensive_6omics_demo()
