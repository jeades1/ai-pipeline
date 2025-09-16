#!/usr/bin/env python3
"""
Tissue-Chip Integration Demo: Synthetic but realistic data to demonstrate 
AI-chip closed-loop biomarker validation.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import random

# Set random seed for reproducible synthetic data
np.random.seed(42)
random.seed(42)


@dataclass
class ChipExperiment:
    """Represents a single tissue-chip experiment."""

    experiment_id: str
    biomarker: str
    perturbation: str
    dose: float
    time_point: int  # hours
    chip_id: str
    replicate: int


@dataclass
class ChipReadout:
    """Tissue-chip measurement results."""

    experiment_id: str
    biomarker_level: float  # fold change vs. control
    viability: float  # % viable cells
    barrier_function: float  # TEER, ohm*cm2
    inflammatory_markers: Dict[str, float]
    confidence_score: float
    p_value: float


@dataclass
class AIHypothesis:
    """AI-generated testable hypothesis."""

    hypothesis_id: str
    biomarker: str
    predicted_direction: str  # "up", "down"
    predicted_magnitude: float  # fold change
    confidence: float
    rationale: str
    priority_score: float


class SyntheticTissueChip:
    """Simulates tissue-chip responses based on realistic cardiovascular biology."""

    def __init__(self):
        # Realistic biomarker baseline levels (relative units)
        self.baseline_levels = {
            "PCSK9": 1.0,
            "APOB": 1.0,
            "HMGCR": 1.0,
            "LDLR": 1.0,
            "CETP": 1.0,
            "LPL": 1.0,
            "ABCA1": 1.0,
            "APOE": 1.0,
        }

        # Realistic perturbation responses (based on literature)
        self.perturbation_effects = {
            "inflammatory_cytokines": {
                "PCSK9": {"direction": "up", "magnitude": 2.5, "noise": 0.3},
                "APOB": {"direction": "up", "magnitude": 1.8, "noise": 0.2},
                "HMGCR": {"direction": "up", "magnitude": 1.4, "noise": 0.25},
                "LDLR": {"direction": "down", "magnitude": 0.6, "noise": 0.2},
                "CETP": {"direction": "up", "magnitude": 1.6, "noise": 0.3},
            },
            "oxidative_stress": {
                "PCSK9": {"direction": "up", "magnitude": 1.9, "noise": 0.25},
                "APOB": {"direction": "up", "magnitude": 2.2, "noise": 0.3},
                "HMGCR": {"direction": "up", "magnitude": 3.1, "noise": 0.4},
                "LPL": {"direction": "down", "magnitude": 0.7, "noise": 0.2},
                "ABCA1": {"direction": "down", "magnitude": 0.5, "noise": 0.3},
            },
            "statin_treatment": {
                "HMGCR": {"direction": "down", "magnitude": 0.3, "noise": 0.15},
                "LDLR": {"direction": "up", "magnitude": 2.8, "noise": 0.2},
                "PCSK9": {"direction": "up", "magnitude": 1.5, "noise": 0.2},
                "APOB": {"direction": "down", "magnitude": 0.4, "noise": 0.25},
            },
            "pcsk9_inhibitor": {
                "PCSK9": {"direction": "down", "magnitude": 0.1, "noise": 0.1},
                "LDLR": {"direction": "up", "magnitude": 4.2, "noise": 0.3},
                "APOB": {"direction": "down", "magnitude": 0.3, "noise": 0.2},
            },
        }

    def run_experiment(self, experiment: ChipExperiment) -> ChipReadout:
        """Simulate tissue-chip experiment with realistic noise and biology."""

        # Get expected response
        if experiment.perturbation in self.perturbation_effects:
            if (
                experiment.biomarker
                in self.perturbation_effects[experiment.perturbation]
            ):
                effect = self.perturbation_effects[experiment.perturbation][
                    experiment.biomarker
                ]

                # Apply dose-response relationship
                dose_factor = min(experiment.dose, 1.0)  # Saturates at high doses
                magnitude = effect["magnitude"] * dose_factor

                # Apply time-dependent effects
                time_factor = 1.0 - np.exp(
                    -experiment.time_point / 12.0
                )  # Hours to steady state

                # Calculate response with noise
                if effect["direction"] == "up":
                    response = magnitude * time_factor
                else:
                    response = 1.0 / (magnitude * time_factor)

                # Add biological noise
                noise = np.random.normal(0, effect["noise"])
                final_response = response * (1 + noise)

                # Calculate confidence and p-value
                confidence = 1.0 - effect["noise"]  # Lower noise = higher confidence
                p_value = (
                    np.random.beta(0.5, 10)
                    if abs(np.log2(final_response)) > 0.5
                    else np.random.beta(2, 2)
                )

            else:
                # No known effect - random noise around baseline
                final_response = 1.0 + np.random.normal(0, 0.1)
                confidence = 0.3
                p_value = np.random.uniform(0.3, 0.9)
        else:
            # Unknown perturbation
            final_response = 1.0 + np.random.normal(0, 0.15)
            confidence = 0.2
            p_value = np.random.uniform(0.5, 0.95)

        # Simulate chip health metrics
        viability = max(0.5, np.random.normal(0.92, 0.05))  # 92% ± 5%
        barrier_function = max(100, np.random.normal(850, 100))  # TEER

        # Simulate inflammatory markers
        inflammatory_markers = {
            "TNF_alpha": np.random.lognormal(0, 0.5),
            "IL1_beta": np.random.lognormal(0, 0.6),
            "IL6": np.random.lognormal(0, 0.4),
        }

        return ChipReadout(
            experiment_id=experiment.experiment_id,
            biomarker_level=final_response,
            viability=viability,
            barrier_function=barrier_function,
            inflammatory_markers=inflammatory_markers,
            confidence_score=confidence,
            p_value=p_value,
        )


class AIBiomarkerPipeline:
    """Simulates AI pipeline generating testable hypotheses."""

    def __init__(self):
        self.knowledge_graph = {}
        self.experimental_evidence = []
        self.model_confidence = 0.4  # Initial confidence

    def generate_hypotheses(self, n_hypotheses: int = 5) -> List[AIHypothesis]:
        """Generate testable hypotheses about biomarker responses."""

        biomarkers = ["PCSK9", "APOB", "HMGCR", "LDLR", "CETP", "LPL"]
        perturbations = [
            "inflammatory_cytokines",
            "oxidative_stress",
            "statin_treatment",
            "pcsk9_inhibitor",
        ]

        hypotheses = []
        for i in range(n_hypotheses):
            biomarker = random.choice(biomarkers)
            perturbation = random.choice(perturbations)

            # Generate prediction based on simple rules + noise
            if perturbation == "inflammatory_cytokines" and biomarker in [
                "PCSK9",
                "APOB",
            ]:
                direction = "up"
                magnitude = np.random.uniform(1.5, 3.0)
                confidence = np.random.uniform(0.7, 0.9)
            elif perturbation == "statin_treatment" and biomarker == "HMGCR":
                direction = "down"
                magnitude = np.random.uniform(0.2, 0.5)
                confidence = np.random.uniform(0.8, 0.95)
            else:
                direction = random.choice(["up", "down"])
                magnitude = np.random.uniform(0.5, 2.5)
                confidence = np.random.uniform(0.3, 0.8)

            hypothesis = AIHypothesis(
                hypothesis_id=f"H_{i+1:03d}",
                biomarker=biomarker,
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                confidence=confidence,
                rationale=f"KG pathway analysis suggests {biomarker} responds to {perturbation}",
                priority_score=confidence * np.random.uniform(0.8, 1.2),
            )
            hypotheses.append(hypothesis)

        return sorted(hypotheses, key=lambda h: h.priority_score, reverse=True)

    def update_from_experiment(
        self, hypothesis: AIHypothesis, result: ChipReadout
    ) -> Dict[str, Any]:
        """Update AI models based on experimental results."""

        # Check if hypothesis was validated
        predicted_fc = (
            hypothesis.predicted_magnitude
            if hypothesis.predicted_direction == "up"
            else 1 / hypothesis.predicted_magnitude
        )
        observed_fc = result.biomarker_level

        # Validation criteria
        direction_correct = (
            hypothesis.predicted_direction == "up" and observed_fc > 1.2
        ) or (hypothesis.predicted_direction == "down" and observed_fc < 0.8)
        magnitude_close = abs(np.log2(predicted_fc) - np.log2(observed_fc)) < 0.5
        significant = result.p_value < 0.05

        validated = direction_correct and magnitude_close and significant

        # Update model confidence
        if validated:
            self.model_confidence = min(0.95, self.model_confidence + 0.05)
        else:
            self.model_confidence = max(0.2, self.model_confidence - 0.02)

        # Store experimental evidence
        evidence = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "biomarker": hypothesis.biomarker,
            "validated": validated,
            "predicted_fc": predicted_fc,
            "observed_fc": observed_fc,
            "p_value": result.p_value,
            "confidence_score": result.confidence_score,
        }
        self.experimental_evidence.append(evidence)

        return {
            "validated": validated,
            "model_confidence": self.model_confidence,
            "evidence": evidence,
        }


class ClosedLoopSystem:
    """Orchestrates AI-chip closed-loop biomarker validation."""

    def __init__(self):
        self.ai_pipeline = AIBiomarkerPipeline()
        self.tissue_chip = SyntheticTissueChip()
        self.experiment_log = []
        self.validation_results = []

    def run_closed_loop_cycle(self, n_cycles: int = 3) -> Dict[str, Any]:
        """Run multiple cycles of hypothesis generation → testing → model update."""

        cycle_results = []

        for cycle in range(n_cycles):
            print(f"\n🔬 === Cycle {cycle + 1} ===")

            # Generate hypotheses
            hypotheses = self.ai_pipeline.generate_hypotheses(n_hypotheses=5)
            print(f"Generated {len(hypotheses)} hypotheses")

            cycle_validations = []

            # Test each hypothesis
            for i, hypothesis in enumerate(hypotheses):
                print(
                    f"Testing hypothesis {i+1}: {hypothesis.biomarker} → {hypothesis.predicted_direction}"
                )

                # Design experiment
                experiment = ChipExperiment(
                    experiment_id=f"EXP_{cycle+1}_{i+1:02d}",
                    biomarker=hypothesis.biomarker,
                    perturbation="inflammatory_cytokines",  # Standard perturbation
                    dose=1.0,
                    time_point=24,  # 24 hours
                    chip_id=f"CHIP_{random.randint(1, 10)}",
                    replicate=1,
                )

                # Run experiment
                result = self.tissue_chip.run_experiment(experiment)

                # Update AI model
                update_result = self.ai_pipeline.update_from_experiment(
                    hypothesis, result
                )

                # Log results
                self.experiment_log.append(
                    {
                        "cycle": cycle + 1,
                        "experiment": asdict(experiment),
                        "hypothesis": asdict(hypothesis),
                        "result": asdict(result),
                        "validation": update_result,
                    }
                )

                cycle_validations.append(update_result["validated"])

                print(
                    f"  Result: {result.biomarker_level:.2f}x, p={result.p_value:.3f}, "
                    f"Validated: {update_result['validated']}"
                )

            cycle_summary = {
                "cycle": cycle + 1,
                "validation_rate": np.mean(cycle_validations),
                "model_confidence": self.ai_pipeline.model_confidence,
                "n_experiments": len(hypotheses),
            }
            cycle_results.append(cycle_summary)

            print(
                f"Cycle {cycle + 1} summary: {np.mean(cycle_validations)*100:.1f}% validation rate, "
                f"Model confidence: {self.ai_pipeline.model_confidence:.2f}"
            )

        return {
            "cycle_results": cycle_results,
            "total_experiments": len(self.experiment_log),
            "overall_validation_rate": np.mean(
                [r["validation_rate"] for r in cycle_results]
            ),
            "final_model_confidence": self.ai_pipeline.model_confidence,
            "experiment_log": self.experiment_log,
        }

    def create_integration_dashboard(
        self, results: Dict[str, Any], output_file: Path
    ) -> None:
        """Create comprehensive dashboard showing AI-chip integration results."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Validation rate over cycles
        cycles = [r["cycle"] for r in results["cycle_results"]]
        validation_rates = [
            r["validation_rate"] * 100 for r in results["cycle_results"]
        ]

        axes[0, 0].plot(
            cycles, validation_rates, "o-", linewidth=3, markersize=8, color="#2ecc71"
        )
        axes[0, 0].set_xlabel("Cycle")
        axes[0, 0].set_ylabel("Validation Rate (%)")
        axes[0, 0].set_title(
            "AI Prediction Validation Rate Over Time", fontweight="bold"
        )
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)

        # 2. Model confidence evolution
        model_confidence = [
            r["model_confidence"] * 100 for r in results["cycle_results"]
        ]

        axes[0, 1].plot(
            cycles, model_confidence, "s-", linewidth=3, markersize=8, color="#3498db"
        )
        axes[0, 1].set_xlabel("Cycle")
        axes[0, 1].set_ylabel("Model Confidence (%)")
        axes[0, 1].set_title("AI Model Confidence Evolution", fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)

        # 3. Biomarker response distribution
        biomarker_responses = [
            exp["result"]["biomarker_level"] for exp in results["experiment_log"]
        ]

        axes[0, 2].hist(
            biomarker_responses, bins=15, alpha=0.7, color="#e74c3c", edgecolor="black"
        )
        axes[0, 2].axvline(
            x=1.0, color="black", linestyle="--", linewidth=2, label="Baseline"
        )
        axes[0, 2].set_xlabel("Fold Change")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_title("Distribution of Biomarker Responses", fontweight="bold")
        axes[0, 2].legend()

        # 4. Validation by biomarker
        biomarkers = list(
            set([exp["hypothesis"]["biomarker"] for exp in results["experiment_log"]])
        )
        validation_by_biomarker = {}

        for biomarker in biomarkers:
            biomarker_exps = [
                exp
                for exp in results["experiment_log"]
                if exp["hypothesis"]["biomarker"] == biomarker
            ]
            validation_rate = np.mean(
                [exp["validation"]["validated"] for exp in biomarker_exps]
            )
            validation_by_biomarker[biomarker] = validation_rate * 100

        bars = axes[1, 0].bar(
            validation_by_biomarker.keys(),
            validation_by_biomarker.values(),
            color="#f39c12",
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("Biomarker")
        axes[1, 0].set_ylabel("Validation Rate (%)")
        axes[1, 0].set_title("Validation Rate by Biomarker", fontweight="bold")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 5. Prediction accuracy scatter
        predicted_fcs = [
            exp["validation"]["evidence"]["predicted_fc"]
            for exp in results["experiment_log"]
        ]
        observed_fcs = [
            exp["validation"]["evidence"]["observed_fc"]
            for exp in results["experiment_log"]
        ]
        validated = [
            exp["validation"]["validated"] for exp in results["experiment_log"]
        ]

        colors = ["#2ecc71" if v else "#e74c3c" for v in validated]
        axes[1, 1].scatter(predicted_fcs, observed_fcs, c=colors, alpha=0.7, s=60)
        axes[1, 1].plot([0.1, 10], [0.1, 10], "k--", alpha=0.5)
        axes[1, 1].set_xlabel("Predicted Fold Change")
        axes[1, 1].set_ylabel("Observed Fold Change")
        axes[1, 1].set_title("Prediction Accuracy", fontweight="bold")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2ecc71",
                markersize=8,
                label="Validated",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#e74c3c",
                markersize=8,
                label="Not Validated",
            ),
        ]
        axes[1, 1].legend(handles=legend_elements)

        # 6. System performance summary
        axes[1, 2].axis("off")
        summary_text = f"""
        🎯 SYSTEM PERFORMANCE SUMMARY
        
        Total Experiments: {results['total_experiments']}
        Overall Validation Rate: {results['overall_validation_rate']*100:.1f}%
        Final Model Confidence: {results['final_model_confidence']*100:.1f}%
        
        🔬 INTEGRATION BENEFITS:
        ✓ Automated hypothesis testing
        ✓ Real-time model updating  
        ✓ Closed-loop optimization
        ✓ Experimental efficiency
        
        📈 NEXT STEPS:
        • Scale to more biomarkers
        • Add dose-response curves
        • Include temporal dynamics
        • Validate on real chips
        """

        axes[1, 2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.suptitle(
            "AI-Tissue Chip Integration Dashboard\nClosed-Loop Biomarker Validation",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
        plt.close()

        print(f"[integration-demo] Generated dashboard: {output_file}")


def main():
    """Run the complete AI-tissue chip integration demonstration."""

    print("🧪 AI-Tissue Chip Integration Demo")
    print("==================================")

    # Initialize closed-loop system
    system = ClosedLoopSystem()

    # Run closed-loop cycles
    results = system.run_closed_loop_cycle(n_cycles=3)

    # Create dashboard
    output_file = Path("artifacts/pitch/ai_chip_integration_demo.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    system.create_integration_dashboard(results, output_file)

    # Save detailed results
    results_file = Path("artifacts/bench/ai_chip_integration_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📊 Results saved to: {results_file}")
    print(f"📈 Dashboard saved to: {output_file}")

    print("\n🎯 DEMO SUMMARY:")
    print(f"Total experiments: {results['total_experiments']}")
    print(f"Overall validation rate: {results['overall_validation_rate']*100:.1f}%")
    print(f"Final model confidence: {results['final_model_confidence']*100:.1f}%")


if __name__ == "__main__":
    main()
