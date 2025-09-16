"""
Automated Reporting System

This module implements comprehensive automated reporting capabilities for the AI pipeline,
including interactive dashboards, clinical decision support outputs, and automated insights generation.

Key Features:
- Interactive web dashboards with real-time updates
- Clinical decision support reports
- Automated insights generation with natural language summaries
- Performance monitoring and alerts
- Regulatory compliance reporting
- Multi-format output (HTML, PDF, JSON, Excel)

Author: AI Pipeline Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import warnings
from jinja2 import Template, Environment, FileSystemLoader
import base64
from io import BytesIO
import tempfile
import subprocess
import os

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ReportType(Enum):
    """Types of reports that can be generated"""
    CLINICAL_SUMMARY = "clinical_summary"
    PERFORMANCE_DASHBOARD = "performance_dashboard"
    BIOMARKER_ANALYSIS = "biomarker_analysis"
    PATIENT_REPORT = "patient_report"
    REGULATORY_SUBMISSION = "regulatory_submission"
    RESEARCH_SUMMARY = "research_summary"
    MONITORING_ALERT = "monitoring_alert"


class OutputFormat(Enum):
    """Output formats for reports"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    EXCEL = "excel"
    DASHBOARD = "dashboard"


class InsightLevel(Enum):
    """Levels of automated insights"""
    BASIC = "basic"           # Simple statistical summaries
    INTERMEDIATE = "intermediate"  # Pattern recognition and trends
    ADVANCED = "advanced"     # Causal insights and recommendations
    CLINICAL = "clinical"     # Clinical decision support


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    report_type: ReportType
    output_format: OutputFormat
    insight_level: InsightLevel
    
    # Content settings
    include_visualizations: bool = True
    include_statistics: bool = True
    include_recommendations: bool = True
    include_raw_data: bool = False
    
    # Styling
    theme: str = "clinical"
    color_palette: str = "healthcare"
    
    # Clinical settings
    patient_anonymization: bool = True
    include_confidence_intervals: bool = True
    statistical_significance_level: float = 0.05
    
    # Performance settings
    max_figures: int = 20
    figure_dpi: int = 300
    interactive_plots: bool = True


@dataclass
class ReportData:
    """Structured data container for report generation"""
    # Core data
    predictions: Optional[pd.DataFrame] = None
    biomarker_data: Optional[pd.DataFrame] = None
    clinical_outcomes: Optional[pd.DataFrame] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    # Analysis results
    benchmark_results: Optional[Dict[str, Any]] = None
    fusion_results: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    causal_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    patient_ids: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    analysis_version: Optional[str] = None
    
    # Quality metrics
    data_quality_scores: Optional[Dict[str, float]] = None
    confidence_scores: Optional[pd.Series] = None


class AutomatedInsightGenerator:
    """
    Generates automated insights from analysis results using pattern recognition
    and clinical knowledge
    """
    
    def __init__(self, insight_level: InsightLevel = InsightLevel.INTERMEDIATE):
        """Initialize insight generator"""
        self.insight_level = insight_level
        self.clinical_templates = self._load_clinical_templates()
        
        logger.info(f"Initialized Automated Insight Generator (Level: {insight_level.value})")
    
    def generate_insights(self, data: ReportData) -> Dict[str, List[str]]:
        """Generate automated insights from report data"""
        
        logger.info("Generating automated insights")
        
        insights = {
            'key_findings': [],
            'clinical_implications': [],
            'recommendations': [],
            'quality_alerts': [],
            'statistical_insights': []
        }
        
        # Performance insights
        if data.performance_metrics:
            insights['key_findings'].extend(self._analyze_performance(data.performance_metrics))
        
        # Biomarker insights
        if data.biomarker_data is not None:
            insights['clinical_implications'].extend(self._analyze_biomarkers(data.biomarker_data))
        
        # Benchmark insights
        if data.benchmark_results:
            insights['statistical_insights'].extend(self._analyze_benchmarks(data.benchmark_results))
        
        # Data quality insights
        if data.data_quality_scores:
            insights['quality_alerts'].extend(self._analyze_data_quality(data.data_quality_scores))
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations(insights, data)
        
        logger.info(f"Generated {sum(len(v) for v in insights.values())} insights")
        
        return insights
    
    def _analyze_performance(self, metrics: Dict[str, Any]) -> List[str]:
        """Analyze performance metrics and generate insights"""
        
        insights = []
        
        if 'roc_auc' in metrics:
            auc = metrics['roc_auc']
            if auc > 0.9:
                insights.append(f"Excellent discriminative performance achieved (AUC = {auc:.3f})")
            elif auc > 0.8:
                insights.append(f"Good discriminative performance achieved (AUC = {auc:.3f})")
            elif auc > 0.7:
                insights.append(f"Moderate discriminative performance achieved (AUC = {auc:.3f})")
            else:
                insights.append(f"Limited discriminative performance observed (AUC = {auc:.3f})")
        
        if 'sensitivity' in metrics and 'specificity' in metrics:
            sens = metrics['sensitivity']
            spec = metrics['specificity']
            
            if sens > 0.9 and spec > 0.9:
                insights.append("Excellent balance of sensitivity and specificity achieved")
            elif sens > 0.8 or spec > 0.8:
                if sens > spec:
                    insights.append("High sensitivity achieved - suitable for screening applications")
                else:
                    insights.append("High specificity achieved - suitable for confirmatory testing")
        
        return insights
    
    def _analyze_biomarkers(self, biomarker_data: pd.DataFrame) -> List[str]:
        """Analyze biomarker patterns and generate clinical insights"""
        
        insights = []
        
        # Check for missing data patterns
        missing_rates = biomarker_data.isnull().mean()
        high_missing = missing_rates[missing_rates > 0.2]
        
        if not high_missing.empty:
            insights.append(f"High missing data rates detected for {len(high_missing)} biomarkers")
        
        # Check for outliers
        numeric_cols = biomarker_data.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = biomarker_data[col].quantile(0.25)
            Q3 = biomarker_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((biomarker_data[col] < Q1 - 1.5*IQR) | 
                       (biomarker_data[col] > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        if outlier_counts:
            total_outliers = sum(outlier_counts.values())
            insights.append(f"Detected {total_outliers} outlier values across biomarkers")
        
        # Correlation insights
        if len(numeric_cols) > 1:
            corr_matrix = biomarker_data[numeric_cols].corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.8)
            high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) 
                              for i, j in zip(*high_corr) if i < j]
            
            if high_corr_pairs:
                insights.append(f"Strong correlations detected between {len(high_corr_pairs)} biomarker pairs")
        
        return insights
    
    def _analyze_benchmarks(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Analyze benchmark comparison results"""
        
        insights = []
        
        if 'method_performance' in benchmark_results:
            methods = benchmark_results['method_performance']
            best_method = benchmark_results.get('best_method', 'Unknown')
            
            insights.append(f"Best performing method: {best_method}")
            
            # Analyze improvements
            if 'AI Pipeline' in methods:
                ai_auc = methods['AI Pipeline'].get('roc_auc', 0)
                other_aucs = [perf.get('roc_auc', 0) for method, perf in methods.items() 
                             if method != 'AI Pipeline']
                
                if other_aucs:
                    max_other_auc = max(other_aucs)
                    improvement = ai_auc - max_other_auc
                    
                    if improvement > 0.05:
                        insights.append(f"Substantial performance improvement over existing methods (+{improvement:.3f} AUC)")
                    elif improvement > 0.02:
                        insights.append(f"Moderate performance improvement over existing methods (+{improvement:.3f} AUC)")
                    elif improvement > 0:
                        insights.append(f"Modest performance improvement over existing methods (+{improvement:.3f} AUC)")
        
        if 'pairwise_comparisons' in benchmark_results:
            comparisons = benchmark_results['pairwise_comparisons']
            significant_comparisons = sum(1 for comp in comparisons.values() 
                                        if comp.get('significantly_different', False))
            
            insights.append(f"{significant_comparisons}/{len(comparisons)} pairwise comparisons are statistically significant")
        
        return insights
    
    def _analyze_data_quality(self, quality_scores: Dict[str, float]) -> List[str]:
        """Analyze data quality metrics and generate alerts"""
        
        alerts = []
        
        for metric, score in quality_scores.items():
            if score < 0.7:
                alerts.append(f"Low data quality detected for {metric} (score: {score:.2f})")
            elif score < 0.8:
                alerts.append(f"Moderate data quality concern for {metric} (score: {score:.2f})")
        
        # Overall quality assessment
        avg_quality = np.mean(list(quality_scores.values()))
        if avg_quality > 0.9:
            alerts.append(f"Excellent overall data quality (average score: {avg_quality:.2f})")
        elif avg_quality > 0.8:
            alerts.append(f"Good overall data quality (average score: {avg_quality:.2f})")
        else:
            alerts.append(f"Data quality concerns identified (average score: {avg_quality:.2f})")
        
        return alerts
    
    def _generate_recommendations(self, insights: Dict[str, List[str]], data: ReportData) -> List[str]:
        """Generate actionable recommendations based on insights"""
        
        recommendations = []
        
        # Performance-based recommendations
        if data.performance_metrics and 'roc_auc' in data.performance_metrics:
            auc = data.performance_metrics['roc_auc']
            if auc < 0.8:
                recommendations.append("Consider ensemble methods or feature engineering to improve performance")
                recommendations.append("Evaluate additional biomarkers or clinical variables")
        
        # Quality-based recommendations
        if 'quality_alerts' in insights and insights['quality_alerts']:
            recommendations.append("Implement data quality monitoring and validation procedures")
            recommendations.append("Consider data imputation or outlier handling strategies")
        
        # Clinical recommendations
        if data.predictions is not None and len(data.predictions) > 0:
            # Check prediction confidence
            if hasattr(data.predictions, 'prediction_confidence'):
                low_confidence = (data.predictions['prediction_confidence'] < 0.7).sum()
                if low_confidence > 0:
                    recommendations.append(f"Review {low_confidence} predictions with low confidence scores")
        
        # Benchmark-based recommendations
        if data.benchmark_results and 'method_performance' in data.benchmark_results:
            methods = data.benchmark_results['method_performance']
            if len(methods) > 1:
                recommendations.append("Consider ensemble combination of top-performing methods")
        
        return recommendations
    
    def _load_clinical_templates(self) -> Dict[str, str]:
        """Load clinical insight templates"""
        
        templates = {
            'risk_stratification': "Patient classified as {risk_level} risk based on {biomarker_count} biomarkers",
            'treatment_recommendation': "Based on molecular profile, consider {treatment_options}",
            'monitoring_schedule': "Recommend monitoring {biomarkers} every {interval}",
            'follow_up': "Clinical follow-up recommended in {timeframe} based on {indicators}"
        }
        
        return templates


class VisualizationEngine:
    """
    Creates interactive and static visualizations for reports
    """
    
    def __init__(self, config: ReportConfiguration):
        """Initialize visualization engine"""
        self.config = config
        self.color_palette = self._get_color_palette()
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'seaborn-v0_8-whitegrid') else 'default')
        sns.set_palette(self.color_palette)
        
        logger.info("Initialized Visualization Engine")
    
    def create_performance_dashboard(self, data: ReportData) -> Dict[str, Any]:
        """Create comprehensive performance dashboard"""
        
        logger.info("Creating performance dashboard")
        
        figures = {}
        
        if data.performance_metrics:
            # ROC Curve
            if data.predictions is not None and data.clinical_outcomes is not None:
                figures['roc_curve'] = self._create_roc_curve(data)
            
            # Performance metrics bar chart
            figures['metrics_chart'] = self._create_metrics_chart(data.performance_metrics)
        
        if data.biomarker_data is not None:
            # Biomarker distribution
            figures['biomarker_distributions'] = self._create_biomarker_distributions(data.biomarker_data)
            
            # Correlation heatmap
            figures['correlation_heatmap'] = self._create_correlation_heatmap(data.biomarker_data)
        
        if data.benchmark_results:
            # Benchmark comparison
            figures['benchmark_comparison'] = self._create_benchmark_comparison(data.benchmark_results)
        
        if data.temporal_analysis:
            # Temporal trends
            figures['temporal_trends'] = self._create_temporal_trends(data.temporal_analysis)
        
        logger.info(f"Created {len(figures)} dashboard figures")
        
        return figures
    
    def create_patient_summary_plot(self, patient_data: Dict[str, Any]) -> go.Figure:
        """Create patient-specific summary visualization"""
        
        # Create subplots for different aspects - simplified to avoid gauge type issues
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Risk Score', 'Biomarker Profile', 'Prediction Confidence', 'Timeline'],
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "scatter"}]]
        )
        
        # Risk score indicator
        risk_score = patient_data.get('risk_score', 0.5)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=risk_score * 100,
                title={'text': "Risk Score (%)"},
                delta={'reference': 50, 'relative': True}
            ),
            row=1, col=1
        )
        
        # Biomarker profile
        if 'biomarkers' in patient_data:
            biomarkers = patient_data['biomarkers']
            # Take first 5 biomarkers to avoid crowding
            biomarker_items = list(biomarkers.items())[:5]
            fig.add_trace(
                go.Bar(
                    x=[item[0] for item in biomarker_items],
                    y=[item[1] for item in biomarker_items],
                    name="Biomarkers"
                ),
                row=1, col=2
            )
        
        # Prediction confidence
        confidence = patient_data.get('confidence', 0.8)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=confidence * 100,
                title={'text': "Confidence (%)"},
                delta={'reference': 80, 'relative': True}
            ),
            row=2, col=1
        )
        
        # Timeline (if available)
        if 'timeline' in patient_data:
            timeline = patient_data['timeline']
            fig.add_trace(
                go.Scatter(
                    x=timeline.get('dates', []),
                    y=timeline.get('values', []),
                    mode='lines+markers',
                    name="Biomarker Trend"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            title="Patient Summary Dashboard",
            template="plotly_white"
        )
        
        return fig
    
    def _create_roc_curve(self, data: ReportData) -> go.Figure:
        """Create interactive ROC curve"""
        
        from sklearn.metrics import roc_curve, roc_auc_score
        
        # Prepare data
        if (data.clinical_outcomes is not None and 'outcome' in data.clinical_outcomes.columns and 
            data.predictions is not None and 'prediction' in data.predictions.columns):
            y_true = data.clinical_outcomes['outcome']
            y_pred = data.predictions['prediction']
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            
            # Create figure
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            
            # Diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve Analysis',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_white',
                width=600, height=500
            )
            
            return fig
        
        return go.Figure()
    
    def _create_metrics_chart(self, metrics: Dict[str, float]) -> go.Figure:
        """Create performance metrics bar chart"""
        
        # Filter relevant metrics
        display_metrics = {k: v for k, v in metrics.items() 
                          if k in ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc']}
        
        fig = go.Figure(data=go.Bar(
            x=list(display_metrics.keys()),
            y=list(display_metrics.values()),
            marker_color=self.color_palette[0]
        ))
        
        fig.update_layout(
            title='Performance Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )
        
        return fig
    
    def _create_biomarker_distributions(self, biomarker_data: pd.DataFrame) -> go.Figure:
        """Create biomarker distribution plots"""
        
        # Select numeric columns
        numeric_cols = biomarker_data.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
        
        if len(numeric_cols) == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(numeric_cols),
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // 3 + 1
            col_idx = i % 3 + 1
            
            fig.add_trace(
                go.Histogram(
                    x=biomarker_data[col].dropna(),
                    name=col,
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title='Biomarker Distributions',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _create_correlation_heatmap(self, biomarker_data: pd.DataFrame) -> go.Figure:
        """Create biomarker correlation heatmap"""
        
        # Select numeric columns
        numeric_data = biomarker_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Biomarker Correlation Matrix',
            template='plotly_white',
            width=600, height=500
        )
        
        return fig
    
    def _create_benchmark_comparison(self, benchmark_results: Dict[str, Any]) -> go.Figure:
        """Create benchmark comparison chart"""
        
        if 'method_performance' not in benchmark_results:
            return go.Figure()
        
        methods = benchmark_results['method_performance']
        method_names = list(methods.keys())
        auc_scores = [perf.get('roc_auc', 0) for perf in methods.values()]
        
        fig = go.Figure(data=go.Bar(
            x=method_names,
            y=auc_scores,
            marker_color=['red' if name == 'AI Pipeline' else 'blue' for name in method_names]
        ))
        
        fig.update_layout(
            title='Benchmark Comparison (ROC AUC)',
            xaxis_title='Method',
            yaxis_title='ROC AUC',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )
        
        return fig
    
    def _create_temporal_trends(self, temporal_analysis: Dict[str, Any]) -> go.Figure:
        """Create temporal trend visualization"""
        
        # Create placeholder temporal trends
        fig = go.Figure()
        
        if 'biomarker_trends' in temporal_analysis:
            trends = temporal_analysis['biomarker_trends']
            
            for biomarker, trend_data in trends.items():
                fig.add_trace(go.Scatter(
                    x=trend_data.get('timestamps', []),
                    y=trend_data.get('values', []),
                    mode='lines+markers',
                    name=biomarker
                ))
        
        fig.update_layout(
            title='Temporal Biomarker Trends',
            xaxis_title='Time',
            yaxis_title='Biomarker Level',
            template='plotly_white'
        )
        
        return fig
    
    def _get_color_palette(self) -> List[str]:
        """Get color palette for visualizations"""
        
        palettes = {
            'healthcare': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'],
            'clinical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'research': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        }
        
        return palettes.get(self.config.color_palette, palettes['clinical'])
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        
        if risk_score < 0.3:
            return 'green'
        elif risk_score < 0.7:
            return 'yellow'
        else:
            return 'red'


class ReportGenerator:
    """
    Main class for generating automated reports
    """
    
    def __init__(self, 
                 output_dir: Path = Path("reports_output"),
                 config: Optional[ReportConfiguration] = None):
        """Initialize report generator"""
        
        self.output_dir = Path(output_dir)
        self.config = config or ReportConfiguration(
            report_type=ReportType.CLINICAL_SUMMARY,
            output_format=OutputFormat.HTML,
            insight_level=InsightLevel.INTERMEDIATE
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.insight_generator = AutomatedInsightGenerator(self.config.insight_level)
        self.viz_engine = VisualizationEngine(self.config)
        
        # Templates
        self.template_env = self._setup_templates()
        
        logger.info(f"Initialized Report Generator")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Report type: {self.config.report_type.value}")
    
    def generate_report(self, 
                       data: ReportData,
                       report_title: str = "AI Pipeline Analysis Report",
                       custom_sections: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """Generate comprehensive automated report"""
        
        logger.info(f"Generating {self.config.report_type.value} report")
        
        # Generate insights
        insights = self.insight_generator.generate_insights(data)
        
        # Create visualizations
        if self.config.include_visualizations:
            figures = self.viz_engine.create_performance_dashboard(data)
        else:
            figures = {}
        
        # Prepare report context
        report_context = {
            'title': report_title,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'insights': insights,
            'data_summary': self._create_data_summary(data),
            'performance_summary': self._create_performance_summary(data),
            'visualizations': figures,
            'custom_sections': custom_sections or {},
            'config': self.config
        }
        
        # Generate outputs
        output_files = {}
        
        if self.config.output_format == OutputFormat.HTML:
            output_files['html'] = self._generate_html_report(report_context)
        
        if self.config.output_format == OutputFormat.JSON:
            output_files['json'] = self._generate_json_report(report_context)
        
        if self.config.output_format == OutputFormat.EXCEL:
            output_files['excel'] = self._generate_excel_report(report_context, data)
        
        if self.config.output_format == OutputFormat.DASHBOARD:
            output_files['dashboard'] = self._generate_dashboard(report_context)
        
        logger.info(f"Report generation completed. Files: {list(output_files.keys())}")
        
        return output_files
    
    def generate_patient_report(self, 
                              patient_id: str, 
                              patient_data: Dict[str, Any]) -> Path:
        """Generate patient-specific report"""
        
        logger.info(f"Generating patient report for {patient_id}")
        
        # Create patient-specific visualizations
        patient_fig = self.viz_engine.create_patient_summary_plot(patient_data)
        
        # Generate insights for this patient
        patient_insights = self._generate_patient_insights(patient_data)
        
        # Create report context
        context = {
            'patient_id': patient_id,
            'patient_data': patient_data,
            'insights': patient_insights,
            'visualization': patient_fig,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate HTML report
        output_file = self.output_dir / f"patient_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        template = self.template_env.get_template('patient_report.html')
        html_content = template.render(**context)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Patient report saved to {output_file}")
        
        return output_file
    
    def generate_monitoring_alert(self, 
                                alert_data: Dict[str, Any],
                                severity: str = "medium") -> Dict[str, Any]:
        """Generate monitoring alert report"""
        
        logger.info(f"Generating monitoring alert (severity: {severity})")
        
        alert_report = {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'alert_type': alert_data.get('type', 'performance'),
            'message': alert_data.get('message', ''),
            'affected_metrics': alert_data.get('metrics', {}),
            'recommended_actions': alert_data.get('actions', []),
            'threshold_violations': alert_data.get('violations', [])
        }
        
        # Save alert to file
        alert_file = self.output_dir / f"alert_{alert_report['alert_id']}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_report, f, indent=2)
        
        logger.info(f"Alert report saved to {alert_file}")
        
        return alert_report
    
    def _create_data_summary(self, data: ReportData) -> Dict[str, Any]:
        """Create summary of data characteristics"""
        
        summary = {
            'total_samples': 0,
            'biomarker_count': 0,
            'outcome_types': [],
            'data_quality': 'Unknown',
            'completeness': 0.0
        }
        
        if data.predictions is not None:
            summary['total_samples'] = len(data.predictions)
        
        if data.biomarker_data is not None:
            summary['biomarker_count'] = len(data.biomarker_data.columns)
            summary['completeness'] = 1.0 - data.biomarker_data.isnull().mean().mean()
        
        if data.clinical_outcomes is not None:
            summary['outcome_types'] = list(data.clinical_outcomes.columns)
        
        # Overall data quality assessment
        if summary['completeness'] > 0.9:
            summary['data_quality'] = 'Excellent'
        elif summary['completeness'] > 0.8:
            summary['data_quality'] = 'Good'
        elif summary['completeness'] > 0.7:
            summary['data_quality'] = 'Fair'
        else:
            summary['data_quality'] = 'Poor'
        
        return summary
    
    def _create_performance_summary(self, data: ReportData) -> Dict[str, Any]:
        """Create performance summary"""
        
        summary = {
            'overall_performance': 'Unknown',
            'key_metrics': {},
            'benchmark_comparison': 'Not available'
        }
        
        if data.performance_metrics:
            summary['key_metrics'] = data.performance_metrics
            
            # Overall performance assessment
            auc = data.performance_metrics.get('roc_auc', 0.5)
            if auc > 0.9:
                summary['overall_performance'] = 'Excellent'
            elif auc > 0.8:
                summary['overall_performance'] = 'Good'
            elif auc > 0.7:
                summary['overall_performance'] = 'Fair'
            else:
                summary['overall_performance'] = 'Poor'
        
        if data.benchmark_results:
            if 'best_method' in data.benchmark_results:
                best = data.benchmark_results['best_method']
                summary['benchmark_comparison'] = f"Best method: {best}"
        
        return summary
    
    def _generate_html_report(self, context: Dict[str, Any]) -> Path:
        """Generate HTML report"""
        
        logger.info("Generating HTML report")
        
        # Convert plotly figures to HTML
        html_figures = {}
        for name, fig in context['visualizations'].items():
            if hasattr(fig, 'to_html'):
                html_figures[name] = fig.to_html(include_plotlyjs='cdn', div_id=f"plot_{name}")
            else:
                html_figures[name] = str(fig)
        
        context['html_figures'] = html_figures
        
        # Render template
        template = self.template_env.get_template('main_report.html')
        html_content = template.render(**context)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"report_{timestamp}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")
        
        return output_file
    
    def _generate_json_report(self, context: Dict[str, Any]) -> Path:
        """Generate JSON report"""
        
        logger.info("Generating JSON report")
        
        # Convert context to JSON-serializable format
        json_context = {
            'title': context['title'],
            'generated_at': context['generated_at'],
            'insights': context['insights'],
            'data_summary': context['data_summary'],
            'performance_summary': context['performance_summary'],
            'custom_sections': context['custom_sections']
        }
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(json_context, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {output_file}")
        
        return output_file
    
    def _generate_excel_report(self, context: Dict[str, Any], data: ReportData) -> Path:
        """Generate Excel report"""
        
        logger.info("Generating Excel report")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Title', 'Generated At', 'Total Samples', 'Biomarker Count', 'Data Quality'],
                'Value': [
                    context['title'],
                    context['generated_at'],
                    context['data_summary'].get('total_samples', 0),
                    context['data_summary'].get('biomarker_count', 0),
                    context['data_summary'].get('data_quality', 'Unknown')
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Performance metrics
            if context['performance_summary']['key_metrics']:
                metrics_df = pd.DataFrame([context['performance_summary']['key_metrics']])
                metrics_df.to_excel(writer, sheet_name='Performance', index=False)
            
            # Insights
            insights_data = []
            for category, insight_list in context['insights'].items():
                for insight in insight_list:
                    insights_data.append({'Category': category, 'Insight': insight})
            
            if insights_data:
                pd.DataFrame(insights_data).to_excel(writer, sheet_name='Insights', index=False)
            
            # Raw data (if included)
            if self.config.include_raw_data:
                if data.biomarker_data is not None:
                    data.biomarker_data.to_excel(writer, sheet_name='Biomarker_Data')
                
                if data.predictions is not None:
                    data.predictions.to_excel(writer, sheet_name='Predictions')
        
        logger.info(f"Excel report saved to {output_file}")
        
        return output_file
    
    def _generate_dashboard(self, context: Dict[str, Any]) -> Path:
        """Generate interactive dashboard HTML"""
        
        logger.info("Generating interactive dashboard")
        
        # Create combined dashboard with all figures
        dashboard_figures = context['visualizations']
        
        # Create dashboard HTML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"dashboard_{timestamp}.html"
        
        # Use dashboard template
        template = self.template_env.get_template('dashboard.html')
        
        # Convert figures to HTML
        figure_html = {}
        for name, fig in dashboard_figures.items():
            if hasattr(fig, 'to_html'):
                figure_html[name] = fig.to_html(include_plotlyjs='cdn', div_id=f"dash_{name}")
        
        dashboard_context = {
            **context,
            'figure_html': figure_html
        }
        
        html_content = template.render(**dashboard_context)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_file}")
        
        return output_file
    
    def _generate_patient_insights(self, patient_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate patient-specific insights"""
        
        insights = {
            'risk_assessment': [],
            'recommendations': [],
            'monitoring': []
        }
        
        # Risk assessment
        risk_score = patient_data.get('risk_score', 0.5)
        if risk_score > 0.8:
            insights['risk_assessment'].append("High risk patient - immediate attention recommended")
        elif risk_score > 0.5:
            insights['risk_assessment'].append("Moderate risk patient - enhanced monitoring advised")
        else:
            insights['risk_assessment'].append("Low risk patient - routine monitoring sufficient")
        
        # Biomarker-based recommendations
        if 'biomarkers' in patient_data:
            biomarkers = patient_data['biomarkers']
            elevated_markers = [k for k, v in biomarkers.items() if v > 1.5]  # Simplified threshold
            
            if elevated_markers:
                insights['recommendations'].append(f"Elevated biomarkers detected: {', '.join(elevated_markers)}")
        
        # Monitoring recommendations
        confidence = patient_data.get('confidence', 0.8)
        if confidence < 0.7:
            insights['monitoring'].append("Low prediction confidence - consider additional testing")
        
        return insights
    
    def _setup_templates(self) -> Environment:
        """Setup Jinja2 template environment"""
        
        # Create templates directory
        templates_dir = self.output_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Create basic templates
        self._create_default_templates(templates_dir)
        
        # Setup environment
        env = Environment(loader=FileSystemLoader(templates_dir))
        
        return env
    
    def _create_default_templates(self, templates_dir: Path) -> None:
        """Create default HTML templates"""
        
        # Main report template
        main_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .section { margin: 30px 0; }
        .insight { background-color: #e7f3ff; padding: 10px; margin: 10px 0; border-radius: 3px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #f1f1f1; border-radius: 5px; }
        .visualization { margin: 20px 0; text-align: center; }
        .alert { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on: {{ generated_at }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Total Samples:</strong> {{ data_summary.total_samples }}
        </div>
        <div class="metric">
            <strong>Biomarkers:</strong> {{ data_summary.biomarker_count }}
        </div>
        <div class="metric">
            <strong>Data Quality:</strong> {{ data_summary.data_quality }}
        </div>
        <div class="metric">
            <strong>Performance:</strong> {{ performance_summary.overall_performance }}
        </div>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        {% for finding in insights.key_findings %}
        <div class="insight">{{ finding }}</div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Clinical Implications</h2>
        {% for implication in insights.clinical_implications %}
        <div class="insight">{{ implication }}</div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {% for recommendation in insights.recommendations %}
        <div class="insight">{{ recommendation }}</div>
        {% endfor %}
    </div>
    
    {% if insights.quality_alerts %}
    <div class="section">
        <h2>Quality Alerts</h2>
        {% for alert in insights.quality_alerts %}
        <div class="alert">{{ alert }}</div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <h2>Performance Visualizations</h2>
        {% for name, figure_html in html_figures.items() %}
        <div class="visualization">
            <h3>{{ name.replace('_', ' ').title() }}</h3>
            {{ figure_html|safe }}
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Statistical Insights</h2>
        {% for insight in insights.statistical_insights %}
        <div class="insight">{{ insight }}</div>
        {% endfor %}
    </div>
</body>
</html>
        """
        
        with open(templates_dir / "main_report.html", 'w') as f:
            f.write(main_template)
        
        # Patient report template
        patient_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Patient Report - {{ patient_id }}</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .section { margin: 30px 0; }
        .risk-high { color: #dc3545; font-weight: bold; }
        .risk-medium { color: #fd7e14; font-weight: bold; }
        .risk-low { color: #28a745; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Patient Report</h1>
        <h2>Patient ID: {{ patient_id }}</h2>
        <p>Generated on: {{ generated_at }}</p>
    </div>
    
    <div class="section">
        <h2>Risk Assessment</h2>
        {% set risk_score = patient_data.get('risk_score', 0.5) %}
        {% if risk_score > 0.7 %}
        <p class="risk-high">High Risk ({{ "%.0f"|format(risk_score * 100) }}%)</p>
        {% elif risk_score > 0.3 %}
        <p class="risk-medium">Moderate Risk ({{ "%.0f"|format(risk_score * 100) }}%)</p>
        {% else %}
        <p class="risk-low">Low Risk ({{ "%.0f"|format(risk_score * 100) }}%)</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Clinical Insights</h2>
        {% for insight in insights.risk_assessment %}
        <p>• {{ insight }}</p>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {% for recommendation in insights.recommendations %}
        <p>• {{ recommendation }}</p>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Patient Dashboard</h2>
        {{ visualization.to_html(include_plotlyjs='cdn')|safe }}
    </div>
</body>
</html>
        """
        
        with open(templates_dir / "patient_report.html", 'w') as f:
            f.write(patient_template)
        
        # Dashboard template
        dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Pipeline Dashboard</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background-color: #fff; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-container { background-color: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Pipeline Performance Dashboard</h1>
            <p>Generated on: {{ generated_at }}</p>
        </div>
        
        <div class="dashboard-grid">
            {% for name, figure_html in figure_html.items() %}
            <div class="chart-container">
                <h3>{{ name.replace('_', ' ').title() }}</h3>
                {{ figure_html|safe }}
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        """
        
        with open(templates_dir / "dashboard.html", 'w') as f:
            f.write(dashboard_template)


def create_demo_reporting():
    """Create demonstration of the automated reporting system"""
    
    print("\n📊 AUTOMATED REPORTING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = ReportConfiguration(
        report_type=ReportType.CLINICAL_SUMMARY,
        output_format=OutputFormat.HTML,
        insight_level=InsightLevel.ADVANCED,
        include_visualizations=True,
        include_statistics=True,
        include_recommendations=True
    )
    
    print("📊 Initializing reporting system...")
    
    # Initialize report generator
    report_generator = ReportGenerator(
        output_dir=Path("demo_outputs/reports"),
        config=config
    )
    
    print("📊 Creating demonstration data...")
    
    # Create demonstration data
    np.random.seed(42)
    n_samples = 200
    
    # Biomarker data
    biomarker_data = pd.DataFrame({
        'creatinine': np.random.normal(1.2, 0.4, n_samples),
        'bun': np.random.normal(20, 8, n_samples),
        'ngal': np.random.lognormal(4, 0.8, n_samples),
        'cystatin_c': np.random.normal(1.0, 0.3, n_samples),
        'urinary_protein': np.random.exponential(2, n_samples)
    }, index=[f'PATIENT_{i:03d}' for i in range(n_samples)])
    
    # Clinical outcomes
    clinical_outcomes = pd.DataFrame({
        'outcome': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'severity': np.random.choice(['mild', 'moderate', 'severe'], n_samples),
        'days_to_event': np.random.exponential(5, n_samples)
    }, index=biomarker_data.index)
    
    # Predictions
    predictions = pd.DataFrame({
        'prediction': np.random.beta(2, 5, n_samples) * (1 - clinical_outcomes['outcome']) + 
                     np.random.beta(5, 2, n_samples) * clinical_outcomes['outcome'],
        'confidence': np.random.uniform(0.6, 0.95, n_samples)
    }, index=biomarker_data.index)
    
    # Performance metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    binary_pred = (predictions['prediction'] > 0.5).astype(int)
    performance_metrics = {
        'roc_auc': roc_auc_score(clinical_outcomes['outcome'], predictions['prediction']),
        'accuracy': accuracy_score(clinical_outcomes['outcome'], binary_pred),
        'precision': precision_score(clinical_outcomes['outcome'], binary_pred),
        'recall': recall_score(clinical_outcomes['outcome'], binary_pred),
        'f1_score': f1_score(clinical_outcomes['outcome'], binary_pred),
        'sensitivity': recall_score(clinical_outcomes['outcome'], binary_pred),
        'specificity': recall_score(1 - clinical_outcomes['outcome'], 1 - binary_pred)
    }
    
    # Benchmark results
    benchmark_results = {
        'method_performance': {
            'AI Pipeline': {'roc_auc': performance_metrics['roc_auc'], 'accuracy': performance_metrics['accuracy']},
            'Creatinine Only': {'roc_auc': 0.72, 'accuracy': 0.68},
            'Clinical Score': {'roc_auc': 0.75, 'accuracy': 0.71}
        },
        'best_method': 'AI Pipeline',
        'pairwise_comparisons': {
            'AI_Pipeline_vs_Creatinine': {'significantly_different': True, 'p_value': 0.02},
            'AI_Pipeline_vs_Clinical': {'significantly_different': True, 'p_value': 0.01}
        }
    }
    
    # Data quality scores
    data_quality_scores = {
        'completeness': 0.92,
        'accuracy': 0.88,
        'consistency': 0.95,
        'timeliness': 0.87
    }
    
    print(f"   Created data: {len(biomarker_data)} samples, {len(biomarker_data.columns)} biomarkers")
    print(f"   Performance: AUC = {performance_metrics['roc_auc']:.3f}")
    
    print("\n📊 Assembling report data...")
    
    # Create report data
    report_data = ReportData(
        predictions=predictions,
        biomarker_data=biomarker_data,
        clinical_outcomes=clinical_outcomes,
        performance_metrics=performance_metrics,
        benchmark_results=benchmark_results,
        data_quality_scores=data_quality_scores,
        patient_ids=list(biomarker_data.index),
        timestamp=datetime.now(),
        analysis_version="v1.0"
    )
    
    print("\n📊 Generating comprehensive report...")
    
    # Generate main report
    output_files = report_generator.generate_report(
        data=report_data,
        report_title="AI Pipeline Clinical Analysis Report",
        custom_sections={
            'methodology': 'Multi-modal ensemble approach with biomarker fusion',
            'validation': 'Cross-validation with bootstrap confidence intervals'
        }
    )
    
    print(f"   Generated report files: {list(output_files.keys())}")
    
    print("\n👤 Generating patient-specific reports...")
    
    # Generate patient reports for sample patients
    sample_patients = ['PATIENT_001', 'PATIENT_050', 'PATIENT_100']
    patient_reports = []
    
    for patient_id in sample_patients:
        patient_idx = biomarker_data.index.get_loc(patient_id)
        
        patient_data = {
            'risk_score': predictions.loc[patient_id, 'prediction'],
            'confidence': predictions.loc[patient_id, 'confidence'],
            'biomarkers': biomarker_data.loc[patient_id].to_dict(),
            'outcome': clinical_outcomes.loc[patient_id, 'outcome'],
            'timeline': {
                'dates': ['2025-01-01', '2025-02-01', '2025-03-01'],
                'values': [0.3, 0.35, 0.4]  # Simplified static values
            }
        }
        
        patient_report = report_generator.generate_patient_report(patient_id, patient_data)
        patient_reports.append(patient_report)
        
        print(f"   Patient {patient_id}: Risk = {patient_data['risk_score']:.2f}")
    
    print("\n🚨 Generating monitoring alerts...")
    
    # Generate monitoring alerts
    alert_scenarios = [
        {
            'type': 'performance_degradation',
            'message': 'Model performance has decreased below threshold',
            'metrics': {'current_auc': 0.72, 'baseline_auc': 0.85},
            'actions': ['Retrain model', 'Check data quality'],
            'violations': ['AUC below 0.75 threshold']
        },
        {
            'type': 'data_quality',
            'message': 'Increased missing data detected',
            'metrics': {'missing_rate': 0.15, 'threshold': 0.10},
            'actions': ['Check data pipeline', 'Review collection procedures'],
            'violations': ['Missing data rate > 10%']
        }
    ]
    
    alerts = []
    for scenario in alert_scenarios:
        alert = report_generator.generate_monitoring_alert(scenario, severity="high")
        alerts.append(alert)
        print(f"   Alert: {alert['alert_type']} - {alert['message']}")
    
    print("\n📊 Generating different report formats...")
    
    # Generate different formats
    formats_to_test = [
        (OutputFormat.JSON, InsightLevel.BASIC),
        (OutputFormat.EXCEL, InsightLevel.INTERMEDIATE),
        (OutputFormat.DASHBOARD, InsightLevel.ADVANCED)
    ]
    
    additional_reports = {}
    
    for output_format, insight_level in formats_to_test:
        format_config = ReportConfiguration(
            report_type=ReportType.PERFORMANCE_DASHBOARD,
            output_format=output_format,
            insight_level=insight_level
        )
        
        format_generator = ReportGenerator(
            output_dir=Path("demo_outputs/reports"),
            config=format_config
        )
        
        format_outputs = format_generator.generate_report(
            data=report_data,
            report_title=f"AI Pipeline Report ({output_format.value.upper()})"
        )
        
        additional_reports[output_format.value] = format_outputs
        print(f"   Generated {output_format.value.upper()} format")
    
    print("\n📈 Performance summary...")
    
    # Create summary statistics
    total_reports = len(output_files) + len(patient_reports) + len(alerts) + len(additional_reports)
    
    print(f"   📊 Main reports: {len(output_files)}")
    print(f"   👤 Patient reports: {len(patient_reports)}")
    print(f"   🚨 Alert reports: {len(alerts)}")
    print(f"   📋 Format variants: {len(additional_reports)}")
    print(f"   📁 Total outputs: {total_reports}")
    
    print("\n📊 Report insights summary:")
    
    # Generate and display key insights
    insights = report_generator.insight_generator.generate_insights(report_data)
    
    for category, insight_list in insights.items():
        if insight_list:
            print(f"   {category.replace('_', ' ').title()}:")
            for insight in insight_list[:3]:  # Show first 3 insights
                print(f"     • {insight}")
    
    print(f"\n✅ Automated reporting system demonstration completed!")
    print(f"📁 All reports saved to: demo_outputs/reports/")
    
    return report_generator, output_files, patient_reports, alerts


if __name__ == "__main__":
    create_demo_reporting()
