#!/usr/bin/env python3
"""
Interactive Demo Interface
Professional web-based demo for AI Pipeline federated personalization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="AI Pipeline: Federated Personalization Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_demo_data():
    """Load enhanced demo datasets"""

    data_dir = Path("data/demo_enhanced")

    if not data_dir.exists():
        st.error(
            "Enhanced demo data not found. Please run enhanced_synthetic_generator.py first."
        )
        st.stop()

    try:
        patients = pd.read_csv(data_dir / "enhanced_patients.csv")
        biomarkers = pd.read_csv(data_dir / "enhanced_biomarkers.csv")
        outcomes = pd.read_csv(data_dir / "enhanced_outcomes.csv")
        institutions = pd.read_csv(data_dir / "institution_details.csv")

        with open(data_dir / "enhanced_dataset_metadata.json", "r") as f:
            metadata = json.load(f)

        return patients, biomarkers, outcomes, institutions, metadata
    except Exception as e:
        st.error(f"Error loading demo data: {e}")
        st.stop()


def main():
    """Main demo interface"""

    # Header
    st.title("🧬 AI Pipeline: Federated Personalization Demo")
    st.markdown(
        """
    **Revolutionary biomarker discovery through privacy-preserving multi-institutional collaboration**
    
    This demo showcases our competitive advantage in federated personalization - capabilities 
    unavailable to centralized competitors like Tempus Labs and Foundation Medicine.
    """
    )

    # Load data
    patients, biomarkers, outcomes, institutions, metadata = load_demo_data()

    # Sidebar controls
    st.sidebar.header("Demo Controls")

    demo_mode = st.sidebar.selectbox(
        "Demo Mode",
        [
            "Executive Overview",
            "Technical Deep Dive",
            "Competitive Analysis",
            "Institution Explorer",
        ],
    )

    # Main content based on mode
    if demo_mode == "Executive Overview":
        show_executive_overview(patients, biomarkers, outcomes, institutions, metadata)
    elif demo_mode == "Technical Deep Dive":
        show_technical_deep_dive(patients, biomarkers, outcomes, institutions, metadata)
    elif demo_mode == "Competitive Analysis":
        show_competitive_analysis(
            patients, biomarkers, outcomes, institutions, metadata
        )
    else:
        show_institution_explorer(
            patients, biomarkers, outcomes, institutions, metadata
        )


def show_executive_overview(patients, biomarkers, outcomes, institutions, metadata):
    """Executive-level overview"""

    st.header("Executive Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Participating Institutions",
            len(institutions),
            help="Major medical centers in federated network",
        )

    with col2:
        improvement = metadata["federated_advantages"]["rrt_improvement_pct"]
        st.metric(
            "RRT Prediction Improvement",
            f"{abs(improvement):.1f}%",
            delta=f"{improvement:.1f}%",
            help="Improvement in critical care predictions vs traditional approaches",
        )

    with col3:
        st.metric(
            "Exclusive Biomarkers",
            metadata["federated_advantages"]["exclusive_biomarkers"],
            help="Novel biomarkers unavailable to centralized competitors",
        )

    with col4:
        st.metric(
            "Patients Analyzed",
            f"{metadata['dataset_info']['total_patients']:,}",
            help="Scale of federated learning demonstration",
        )

    st.markdown("---")

    # Competitive advantage visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Federated vs Traditional Approach Performance")

        # Performance comparison chart
        performance_data = {
            "Outcome": ["AKI Prediction", "RRT Prediction", "Mortality Prediction"],
            "Traditional Approach": [
                metadata["outcome_rates"]["traditional_aki_rate"],
                metadata["outcome_rates"]["traditional_rrt_rate"],
                metadata["outcome_rates"]["traditional_mortality"],
            ],
            "Federated Approach": [
                metadata["outcome_rates"]["federated_aki_rate"],
                metadata["outcome_rates"]["federated_rrt_rate"],
                metadata["outcome_rates"]["federated_mortality"],
            ],
        }

        fig = px.bar(
            pd.DataFrame(performance_data).melt(id_vars="Outcome"),
            x="Outcome",
            y="value",
            color="variable",
            title="Clinical Outcome Rates (%)",
            color_discrete_map={
                "Traditional Approach": "#ff7f0e",
                "Federated Approach": "#2ca02c",
            },
        )
        fig.update_layout(showlegend=True, yaxis_title="Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Network Effects")

        # Institution network visualization
        fig = px.scatter(
            institutions,
            x="icu_beds",
            y="patients",
            size="patients",
            color="privacy_tier",
            hover_data=["name", "specialty"],
            title="Institutional Network",
            color_discrete_map={"High": "#2ca02c", "Medium": "#ff7f0e"},
        )
        fig.update_layout(xaxis_title="ICU Beds", yaxis_title="Patients in Demo")
        st.plotly_chart(fig, use_container_width=True)

    # Strategic implications
    st.subheader("Strategic Implications")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **🎯 Competitive Moat**
        - Privacy-preserving collaboration
        - Network effects strengthen with scale
        - First-mover advantage in federated medicine
        """
        )

    with col2:
        st.markdown(
            """
        **📈 Market Opportunity**
        - 55% untapped capability space
        - $X billion addressable market
        - Revolutionary vs incremental positioning
        """
        )

    with col3:
        st.markdown(
            """
        **🔒 Regulatory Advantage**
        - HIPAA-compliant by design
        - Privacy-preserving analytics
        - Institutional data sovereignty
        """
        )


def show_technical_deep_dive(patients, biomarkers, outcomes, institutions, metadata):
    """Technical implementation details"""

    st.header("Technical Deep Dive")

    # Biomarker analysis
    st.subheader("Biomarker Discovery Pipeline")

    tab1, tab2, tab3 = st.tabs(
        ["Traditional Biomarkers", "Federated Signatures", "Correlations"]
    )

    with tab1:
        traditional_biomarkers = biomarkers[biomarkers["type"] == "traditional"]

        # Biomarker expression heatmap
        pivot_data = traditional_biomarkers.pivot_table(
            index="patient_id", columns="biomarker", values="expression_log2"
        ).head(
            50
        )  # Show first 50 patients

        fig = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=[f"Patient {i+1}" for i in range(len(pivot_data))],
            title="Traditional Biomarker Expression Patterns (Sample)",
            color_continuous_scale="RdYlBu_r",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Traditional biomarkers** are available to all competitors (Tempus Labs, Foundation Medicine, etc.)"
        )

    with tab2:
        federated_biomarkers = biomarkers[biomarkers["type"] == "federated_exclusive"]

        # Federated biomarker advantage
        fed_pivot = federated_biomarkers.pivot_table(
            index="patient_id", columns="biomarker", values="expression_log2"
        ).head(50)

        fig = px.imshow(
            fed_pivot.values,
            x=fed_pivot.columns,
            y=[f"Patient {i+1}" for i in range(len(fed_pivot))],
            title="Federated-Exclusive Biomarker Signatures (Sample)",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Federated signatures** are only available through our privacy-preserving collaboration platform."
        )

    with tab3:
        # Biomarker correlation analysis
        st.markdown("**Inter-biomarker correlations** improve prediction accuracy")

        # Calculate correlations for traditional biomarkers
        traditional_pivot = traditional_biomarkers.pivot_table(
            index="patient_id", columns="biomarker", values="expression_log2"
        )

        if len(traditional_pivot.columns) > 1:
            correlation_matrix = traditional_pivot.corr()

            fig = px.imshow(
                correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                title="Biomarker Correlation Matrix",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Federated learning architecture
    st.subheader("Federated Learning Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Privacy-Preserving Collaboration**
        - Differential privacy guarantees
        - Secure multiparty computation
        - Local model training
        - Encrypted gradient aggregation
        """
        )

    with col2:
        st.markdown(
            """
        **Technical Implementation**
        - Federated averaging algorithms
        - Heterogeneous data handling
        - Communication-efficient protocols
        - Byzantine fault tolerance
        """
        )


def show_competitive_analysis(patients, biomarkers, outcomes, institutions, metadata):
    """Competitive positioning analysis"""

    st.header("Competitive Analysis")

    # Load competitive analysis data if available
    try:
        import sys

        sys.path.append(".")
        from presentation.application_competitive_analysis import (
            calculate_application_scores,
        )

        results, scorer = calculate_application_scores()

        # 3D competitive visualization
        st.subheader("3D Competitive Landscape")

        plot_data = []
        for company, data in results.items():
            plot_data.append(
                {
                    "Company": company,
                    "Discovery_Capability": data["discovery_capability"],
                    "Clinical_Impact": data["clinical_impact"],
                    "Federated_Personalization": data["individual_scores"][
                        "federated_personalization"
                    ],
                    "Type": (
                        "Target"
                        if "Year 7" in company
                        else (
                            "Our Platform"
                            if "Our Platform" in company
                            else "Competitor"
                        )
                    ),
                }
            )

        df = pd.DataFrame(plot_data)

        fig = px.scatter_3d(
            df,
            x="Discovery_Capability",
            y="Clinical_Impact",
            z="Federated_Personalization",
            color="Type",
            size=[
                20 if "Target" in t else 15 if "Our Platform" in t else 10
                for t in df["Type"]
            ],
            hover_data=["Company"],
            title="3D Competitive Analysis",
            color_discrete_map={
                "Our Platform": "#2ca02c",
                "Target": "#4caf50",
                "Competitor": "#757575",
            },
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Competitive advantages table
        st.subheader("Competitive Advantages")

        advantages_data = []
        for company, data in results.items():
            if "Our Platform" not in company:
                fed_score = data["individual_scores"]["federated_personalization"]
                our_score = results["Our Platform (Year 7 Target)"][
                    "individual_scores"
                ]["federated_personalization"]
                advantage = our_score - fed_score

                advantages_data.append(
                    {
                        "Competitor": company,
                        "Their Federated Score": f"{fed_score:.1f}/10",
                        "Our Target Score": f"{our_score:.1f}/10",
                        "Our Advantage": f"+{advantage:.1f} points",
                    }
                )

        advantages_df = pd.DataFrame(advantages_data)
        st.dataframe(advantages_df, use_container_width=True)

    except Exception:
        st.warning(
            "Competitive analysis data not available. Please run the competitive analysis script first."
        )

        # Fallback competitive summary
        st.subheader("Competitive Positioning Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            **Tempus Labs**
            - Strong genomic sequencing
            - Limited federated capabilities
            - Centralized data model
            """
            )

        with col2:
            st.markdown(
                """
            **Foundation Medicine**
            - Comprehensive genomic profiling
            - No federated personalization
            - Single-institution focus
            """
            )

        with col3:
            st.markdown(
                """
            **Our Platform**
            - Multi-institutional collaboration
            - Privacy-preserving analytics
            - Personalized biomarker discovery
            """
            )


def show_institution_explorer(patients, biomarkers, outcomes, institutions, metadata):
    """Institution-specific analysis"""

    st.header("Institution Explorer")

    # Institution selector
    selected_institution = st.selectbox(
        "Select Institution", options=institutions["name"].tolist()
    )

    # Filter data for selected institution
    inst_patients = patients[patients["institution"] == selected_institution]
    inst_outcomes = outcomes[outcomes["institution"] == selected_institution]
    inst_biomarkers = biomarkers[biomarkers["institution"] == selected_institution]

    # Institution overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patients", len(inst_patients))

    with col2:
        st.metric("Mean Age", f"{inst_patients['age'].mean():.1f}")

    with col3:
        st.metric("Mean APACHE II", f"{inst_patients['apache_ii'].mean():.1f}")

    with col4:
        privacy_tier = inst_patients["privacy_tier"].iloc[0]
        st.metric("Privacy Tier", privacy_tier)

    # Institution-specific performance
    st.subheader("Institution Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Outcome comparison
        traditional_aki = inst_outcomes["traditional_develops_aki"].mean() * 100
        federated_aki = inst_outcomes["federated_develops_aki"].mean() * 100

        outcome_data = {
            "Approach": ["Traditional", "Federated"],
            "AKI Rate (%)": [traditional_aki, federated_aki],
        }

        fig = px.bar(
            pd.DataFrame(outcome_data),
            x="Approach",
            y="AKI Rate (%)",
            title=f"AKI Prediction Performance - {selected_institution}",
            color="Approach",
            color_discrete_map={"Traditional": "#ff7f0e", "Federated": "#2ca02c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Patient characteristics
        fig = px.histogram(
            inst_patients,
            x="age",
            title=f"Age Distribution - {selected_institution}",
            nbins=20,
            color_discrete_sequence=["#2ca02c"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Biomarker patterns
    st.subheader("Biomarker Patterns")

    # Average biomarker expressions
    avg_expressions = (
        inst_biomarkers.groupby(["biomarker", "type"])["expression_log2"]
        .mean()
        .reset_index()
    )

    traditional_avg = avg_expressions[avg_expressions["type"] == "traditional"]
    federated_avg = avg_expressions[avg_expressions["type"] == "federated_exclusive"]

    col1, col2 = st.columns(2)

    with col1:
        if len(traditional_avg) > 0:
            fig = px.bar(
                traditional_avg,
                x="biomarker",
                y="expression_log2",
                title="Traditional Biomarker Levels",
                color_discrete_sequence=["#ff7f0e"],
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if len(federated_avg) > 0:
            fig = px.bar(
                federated_avg,
                x="biomarker",
                y="expression_log2",
                title="Federated Biomarker Signatures",
                color_discrete_sequence=["#2ca02c"],
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
