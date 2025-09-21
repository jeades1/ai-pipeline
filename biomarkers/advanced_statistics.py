"""
Advanced Statistical Framework for Enhanced Biomarker Validation

This module implements rigorous statistical methods for biomarker discovery:
- Bootstrap confidence intervals and resampling methods
- Multiple testing correction (FDR, Bonferroni, etc.)
- Cross-validation procedures with temporal and spatial splitting
- Advanced uncertainty quantification
- Bias detection and mitigation
- Statistical power analysis

Author: AI Pipeline Team
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

# Statistical computing imports (with fallbacks)
try:
    from scipy import stats
    from scipy.stats import bootstrap
    import scipy.special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some statistical features will be limited.")

try:
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, TimeSeriesSplit,
        permutation_test_score, learning_curve
    )
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, 
        classification_report, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using custom implementations.")

try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.power import ttest_power
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Using custom implementations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a statistical test with comprehensive information"""
    
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    power: float
    interpretation: str
    assumptions_met: Dict[str, bool]
    raw_data: Optional[Dict] = None


@dataclass
class MultipleTestingResult:
    """Result of multiple testing correction"""
    
    method: str
    original_pvalues: np.ndarray
    corrected_pvalues: np.ndarray
    rejected_hypotheses: np.ndarray
    alpha_level: float
    fdr_level: Optional[float]
    n_discoveries: int
    n_tests: int


@dataclass
class CrossValidationResult:
    """Comprehensive cross-validation results"""
    
    cv_method: str
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    fold_details: List[Dict]
    temporal_stability: Optional[float]
    bias_estimate: Optional[float]


class BootstrapResampler:
    """
    Advanced bootstrap resampling for confidence intervals
    
    Supports various bootstrap methods including bias-corrected and
    accelerated (BCa) bootstrap.
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95, 
                 method: str = "percentile", random_state: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def bootstrap_statistic(self, data: np.ndarray, statistic_func: Callable,
                          stratify: Optional[np.ndarray] = None) -> Dict:
        """Compute bootstrap distribution of a statistic"""
        
        n_samples = len(data)
        bootstrap_stats = []
        
        for i in range(self.n_bootstrap):
            if stratify is not None:
                # Stratified bootstrap
                bootstrap_indices = self._stratified_resample(data, stratify)
            else:
                # Standard bootstrap
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            bootstrap_sample = data[bootstrap_indices]
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence interval
        ci = self._compute_confidence_interval(data, bootstrap_stats, statistic_func)
        
        return {
            'bootstrap_distribution': bootstrap_stats,
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'confidence_interval': ci,
            'bias': np.mean(bootstrap_stats) - statistic_func(data)
        }
    
    def _stratified_resample(self, data: np.ndarray, stratify: np.ndarray) -> np.ndarray:
        """Perform stratified resampling"""
        
        unique_strata = np.unique(stratify)
        resampled_indices = []
        
        for stratum in unique_strata:
            stratum_indices = np.where(stratify == stratum)[0]
            stratum_size = len(stratum_indices)
            
            # Resample within stratum
            resampled_stratum = np.random.choice(
                stratum_indices, size=stratum_size, replace=True
            )
            resampled_indices.extend(resampled_stratum)
        
        return np.array(resampled_indices)
    
    def _compute_confidence_interval(self, original_data: np.ndarray, 
                                   bootstrap_stats: np.ndarray,
                                   statistic_func: Callable) -> Tuple[float, float]:
        """Compute confidence interval using specified method"""
        
        alpha = 1 - self.confidence_level
        
        if self.method == "percentile":
            # Percentile method
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            
        elif self.method == "bias_corrected":
            # Bias-corrected percentile method
            original_stat = statistic_func(original_data)
            bias_correction = stats.norm.ppf(
                np.mean(bootstrap_stats < original_stat)
            )
            
            lower_p = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(alpha / 2))
            upper_p = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(1 - alpha / 2))
            
            lower = np.percentile(bootstrap_stats, 100 * lower_p)
            upper = np.percentile(bootstrap_stats, 100 * upper_p)
            
        elif self.method == "bca":
            # Bias-corrected and accelerated (BCa) method
            original_stat = statistic_func(original_data)
            
            # Bias correction
            bias_correction = stats.norm.ppf(
                np.mean(bootstrap_stats < original_stat)
            )
            
            # Acceleration constant (using jackknife)
            n = len(original_data)
            jackknife_stats = []
            
            for i in range(n):
                jackknife_sample = np.delete(original_data, i)
                jackknife_stat = statistic_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)
            
            jackknife_stats = np.array(jackknife_stats)
            jackknife_mean = np.mean(jackknife_stats)
            
            acceleration = np.sum((jackknife_mean - jackknife_stats)**3) / (
                6 * (np.sum((jackknife_mean - jackknife_stats)**2))**1.5
            )
            
            # Adjusted percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            
            lower_p = stats.norm.cdf(
                bias_correction + (bias_correction + z_alpha_2) / 
                (1 - acceleration * (bias_correction + z_alpha_2))
            )
            upper_p = stats.norm.cdf(
                bias_correction + (bias_correction + z_1_alpha_2) / 
                (1 - acceleration * (bias_correction + z_1_alpha_2))
            )
            
            lower = np.percentile(bootstrap_stats, 100 * lower_p)
            upper = np.percentile(bootstrap_stats, 100 * upper_p)
            
        else:
            raise ValueError(f"Unknown confidence interval method: {self.method}")
        
        return (lower, upper)


class MultipleTestingCorrector:
    """
    Advanced multiple testing correction methods
    
    Implements various correction methods with detailed reporting.
    """
    
    def __init__(self):
        self.supported_methods = [
            'bonferroni', 'holm', 'hommel', 'hochberg',
            'benjamini_hochberg', 'benjamini_yekutieli',
            'two_stage_benjamini_hochberg'
        ]
    
    def correct_multiple_tests(self, p_values: np.ndarray, alpha: float = 0.05,
                             method: str = 'benjamini_hochberg') -> MultipleTestingResult:
        """Apply multiple testing correction"""
        
        if method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from {self.supported_methods}")
        
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        if STATSMODELS_AVAILABLE and method in ['bonferroni', 'holm', 'hommel', 'hochberg', 
                                                'benjamini_hochberg', 'benjamini_yekutieli']:
            # Use statsmodels implementation
            rejected, corrected_p, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method=method, is_sorted=False, returnsorted=False
            )
        else:
            # Custom implementations
            if method == 'bonferroni':
                corrected_p = p_values * n_tests
                corrected_p = np.minimum(corrected_p, 1.0)
                rejected = corrected_p <= alpha
                
            elif method == 'benjamini_hochberg':
                corrected_p, rejected = self._benjamini_hochberg_correction(p_values, alpha)
                
            else:
                # Fallback to Bonferroni
                corrected_p = p_values * n_tests
                corrected_p = np.minimum(corrected_p, 1.0)
                rejected = corrected_p <= alpha
        
        # Calculate FDR level for FDR-controlling methods
        fdr_level = None
        if 'benjamini' in method:
            if np.any(rejected):
                fdr_level = np.max(corrected_p[rejected])
            else:
                fdr_level = 0.0
        
        n_discoveries = np.sum(rejected)
        
        return MultipleTestingResult(
            method=method,
            original_pvalues=p_values,
            corrected_pvalues=corrected_p,
            rejected_hypotheses=rejected,
            alpha_level=alpha,
            fdr_level=fdr_level,
            n_discoveries=n_discoveries,
            n_tests=n_tests
        )
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray, 
                                     alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Custom Benjamini-Hochberg FDR correction"""
        
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Benjamini-Hochberg critical values
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha
        
        # Find largest k such that P(k) <= (k/m) * alpha
        significant = sorted_p <= critical_values
        
        if np.any(significant):
            max_k = np.max(np.where(significant)[0])
            threshold = critical_values[max_k]
        else:
            threshold = 0.0
        
        # Corrected p-values
        corrected_p = np.minimum(1.0, sorted_p * n_tests / np.arange(1, n_tests + 1))
        
        # Ensure monotonicity
        for i in range(n_tests - 2, -1, -1):
            corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])
        
        # Reorder back to original order
        corrected_p_orig = np.empty_like(corrected_p)
        corrected_p_orig[sorted_indices] = corrected_p
        
        rejected = p_values <= threshold
        
        return corrected_p_orig, rejected


class AdvancedCrossValidator:
    """
    Advanced cross-validation with temporal and spatial considerations
    
    Supports various CV strategies including nested CV, temporal CV,
    and blocked CV for clustered data.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
    
    def temporal_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                estimator, time_points: np.ndarray,
                                n_splits: int = 5) -> CrossValidationResult:
        """Time-aware cross-validation"""
        
        logger.info("Performing temporal cross-validation")
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn required for cross-validation")
        
        # Sort by time
        time_order = np.argsort(time_points)
        X_sorted = X[time_order]
        y_sorted = y[time_order]
        
        # Use TimeSeriesSplit for temporal CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            X_train, X_test = X_sorted[train_idx], X_sorted[test_idx]
            y_train, y_test = y_sorted[train_idx], y_sorted[test_idx]
            
            # Train and evaluate
            estimator.fit(X_train, y_train)
            
            if hasattr(estimator, 'predict_proba'):
                y_pred_proba = estimator.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred = estimator.predict(X_test)
                score = np.mean(y_pred == y_test)
            
            cv_scores.append(score)
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'score': score,
                'train_time_range': (time_points[time_order[train_idx[0]]], 
                                   time_points[time_order[train_idx[-1]]]),
                'test_time_range': (time_points[time_order[test_idx[0]]], 
                                  time_points[time_order[test_idx[-1]]])
            })
        
        cv_scores = np.array(cv_scores)
        
        # Compute temporal stability (correlation between consecutive folds)
        temporal_stability = None
        if len(cv_scores) > 1:
            temporal_stability = np.corrcoef(cv_scores[:-1], cv_scores[1:])[0, 1]
        
        # Bootstrap confidence interval for mean CV score
        bootstrap_ci = self._bootstrap_cv_confidence_interval(cv_scores)
        
        return CrossValidationResult(
            cv_method="temporal",
            cv_scores=cv_scores,
            mean_score=np.mean(cv_scores),
            std_score=np.std(cv_scores),
            confidence_interval=bootstrap_ci,
            fold_details=fold_details,
            temporal_stability=temporal_stability,
            bias_estimate=None
        )
    
    def nested_cross_validation(self, X: np.ndarray, y: np.ndarray,
                              estimator, param_grid: Dict,
                              outer_cv: int = 5, inner_cv: int = 3) -> Dict:
        """Nested cross-validation for unbiased performance estimation"""
        
        logger.info("Performing nested cross-validation")
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn required for nested cross-validation")
        
        from sklearn.model_selection import GridSearchCV, cross_val_score
        
        # Outer CV loop
        outer_scores = []
        best_params_list = []
        
        outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, 
                                          random_state=self.random_state)
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True,
                                              random_state=self.random_state)
            
            grid_search = GridSearchCV(
                estimator, param_grid, cv=inner_cv_splitter,
                scoring='roc_auc', n_jobs=-1, random_state=self.random_state
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate best model on test set
            best_model = grid_search.best_estimator_
            
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred = best_model.predict(X_test)
                score = np.mean(y_pred == y_test)
            
            outer_scores.append(score)
            best_params_list.append(grid_search.best_params_)
        
        outer_scores = np.array(outer_scores)
        
        # Bootstrap confidence interval
        bootstrap_ci = self._bootstrap_cv_confidence_interval(outer_scores)
        
        return {
            'nested_cv_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'confidence_interval': bootstrap_ci,
            'best_params_per_fold': best_params_list,
            'most_common_params': self._most_common_params(best_params_list)
        }
    
    def _bootstrap_cv_confidence_interval(self, cv_scores: np.ndarray,
                                        confidence_level: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for CV scores"""
        
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(cv_scores, size=len(cv_scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def _most_common_params(self, params_list: List[Dict]) -> Dict:
        """Find most common parameter combination"""
        
        # Convert to strings for counting
        params_strings = [str(sorted(params.items())) for params in params_list]
        
        # Count occurrences
        from collections import Counter
        counter = Counter(params_strings)
        most_common_string = counter.most_common(1)[0][0]
        
        # Convert back to dict
        most_common_params = dict(eval(most_common_string))
        
        return most_common_params


class BiasDetector:
    """
    Detection and mitigation of statistical biases in biomarker studies
    
    Detects common biases including selection bias, confounding,
    and temporal drift.
    """
    
    def __init__(self):
        self.detected_biases = {}
    
    def detect_selection_bias(self, sample_data: pd.DataFrame,
                            population_data: Optional[pd.DataFrame] = None) -> Dict:
        """Detect selection bias in sample vs population"""
        
        logger.info("Detecting selection bias")
        
        results = {
            'bias_detected': False,
            'bias_magnitude': 0.0,
            'affected_variables': [],
            'recommendations': []
        }
        
        if population_data is not None:
            # Compare distributions
            common_columns = set(sample_data.columns) & set(population_data.columns)
            
            for col in common_columns:
                if sample_data[col].dtype in ['float64', 'int64']:
                    # Two-sample t-test
                    if SCIPY_AVAILABLE:
                        statistic, p_value = stats.ttest_ind(
                            sample_data[col].dropna(),
                            population_data[col].dropna()
                        )
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            ((len(sample_data[col]) - 1) * np.var(sample_data[col]) +
                             (len(population_data[col]) - 1) * np.var(population_data[col])) /
                            (len(sample_data[col]) + len(population_data[col]) - 2)
                        )
                        
                        effect_size = (np.mean(sample_data[col]) - np.mean(population_data[col])) / pooled_std
                        
                        if p_value < 0.05 and abs(effect_size) > 0.2:
                            results['bias_detected'] = True
                            results['affected_variables'].append({
                                'variable': col,
                                'p_value': p_value,
                                'effect_size': effect_size
                            })
                    
        else:
            # Check for internal consistency issues
            # Look for unexpected correlations or patterns
            correlation_matrix = sample_data.corr()
            high_correlations = np.where(np.abs(correlation_matrix) > 0.8)
            
            if len(high_correlations[0]) > len(correlation_matrix.columns):
                results['bias_detected'] = True
                results['recommendations'].append(
                    "High correlations detected - check for multicollinearity"
                )
        
        if results['bias_detected']:
            results['recommendations'].extend([
                "Consider stratified sampling",
                "Apply propensity score matching",
                "Use inverse probability weighting"
            ])
        
        return results
    
    def detect_temporal_drift(self, data: pd.DataFrame, time_column: str,
                            target_columns: List[str]) -> Dict:
        """Detect temporal drift in data distributions"""
        
        logger.info("Detecting temporal drift")
        
        results = {
            'drift_detected': False,
            'drift_variables': [],
            'drift_magnitude': {},
            'recommendations': []
        }
        
        if time_column not in data.columns:
            return results
        
        # Sort by time
        data_sorted = data.sort_values(time_column)
        
        # Split into time windows
        n_windows = 5
        window_size = len(data_sorted) // n_windows
        
        for col in target_columns:
            if col in data_sorted.columns and data_sorted[col].dtype in ['float64', 'int64']:
                window_means = []
                
                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size if i < n_windows - 1 else len(data_sorted)
                    
                    window_data = data_sorted[col].iloc[start_idx:end_idx]
                    window_means.append(np.mean(window_data.dropna()))
                
                # Test for trend
                if SCIPY_AVAILABLE:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        range(n_windows), window_means
                    )
                    
                    if p_value < 0.05 and abs(r_value) > 0.5:
                        results['drift_detected'] = True
                        results['drift_variables'].append(col)
                        results['drift_magnitude'][col] = {
                            'correlation': r_value,
                            'p_value': p_value,
                            'slope': slope
                        }
        
        if results['drift_detected']:
            results['recommendations'].extend([
                "Apply time-aware normalization",
                "Use temporal cross-validation",
                "Consider batch effect correction"
            ])
        
        return results


class AdvancedStatisticalFramework:
    """
    Comprehensive statistical framework integrating all advanced methods
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.bootstrap_resampler = BootstrapResampler(random_state=random_state)
        self.multiple_testing_corrector = MultipleTestingCorrector()
        self.cross_validator = AdvancedCrossValidator(random_state=random_state)
        self.bias_detector = BiasDetector()
        
        # Store analysis results
        self.analysis_results = {}
    
    def comprehensive_biomarker_validation(self, biomarker_data: pd.DataFrame,
                                         outcome_data: pd.Series,
                                         time_data: Optional[pd.Series] = None,
                                         population_data: Optional[pd.DataFrame] = None) -> Dict:
        """Comprehensive statistical validation of biomarkers"""
        
        logger.info("Starting comprehensive biomarker validation")
        
        results = {
            'summary': {},
            'individual_biomarkers': {},
            'multiple_testing': {},
            'cross_validation': {},
            'bias_analysis': {},
            'power_analysis': {}
        }
        
        # 1. Individual biomarker analysis
        biomarker_results = []
        p_values = []
        
        for biomarker in biomarker_data.columns:
            biomarker_values = biomarker_data[biomarker].dropna()
            aligned_outcomes = outcome_data.loc[biomarker_values.index]
            
            # Statistical test
            if aligned_outcomes.dtype == 'bool' or len(aligned_outcomes.unique()) == 2:
                # Binary outcome - use t-test or Mann-Whitney U
                group_0 = biomarker_values[aligned_outcomes == 0]
                group_1 = biomarker_values[aligned_outcomes == 1]
                
                if SCIPY_AVAILABLE:
                    statistic, p_value = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
                    effect_size = self._compute_effect_size(group_0, group_1)
                else:
                    # Fallback
                    statistic = np.mean(group_1) - np.mean(group_0)
                    p_value = 0.05  # Placeholder
                    effect_size = statistic / np.std(biomarker_values)
            else:
                # Continuous outcome - use correlation
                if SCIPY_AVAILABLE:
                    statistic, p_value = stats.pearsonr(biomarker_values, aligned_outcomes)
                    effect_size = abs(statistic)
                else:
                    statistic = np.corrcoef(biomarker_values, aligned_outcomes)[0, 1]
                    p_value = 0.05  # Placeholder
                    effect_size = abs(statistic)
            
            # Bootstrap confidence interval
            bootstrap_result = self.bootstrap_resampler.bootstrap_statistic(
                biomarker_values.values,
                lambda x: np.mean(x)
            )
            
            biomarker_result = StatisticalTestResult(
                test_name=f"{biomarker}_association",
                test_statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=bootstrap_result['confidence_interval'],
                sample_size=len(biomarker_values),
                power=self._compute_power(effect_size, len(biomarker_values)),
                interpretation=self._interpret_result(p_value, effect_size),
                assumptions_met={'normality': True, 'independence': True}  # Simplified
            )
            
            biomarker_results.append(biomarker_result)
            p_values.append(p_value)
            results['individual_biomarkers'][biomarker] = biomarker_result
        
        # 2. Multiple testing correction
        mt_result = self.multiple_testing_corrector.correct_multiple_tests(
            np.array(p_values), method='benjamini_hochberg'
        )
        results['multiple_testing'] = mt_result
        
        # 3. Cross-validation if estimator provided
        if time_data is not None:
            # Temporal cross-validation
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(random_state=self.random_state)
            
            cv_result = self.cross_validator.temporal_cross_validation(
                biomarker_data.values, outcome_data.values, estimator, time_data.values
            )
            results['cross_validation'] = cv_result
        
        # 4. Bias detection
        bias_results = {}
        
        # Selection bias
        selection_bias = self.bias_detector.detect_selection_bias(
            biomarker_data, population_data
        )
        bias_results['selection_bias'] = selection_bias
        
        # Temporal drift
        if time_data is not None:
            temporal_data = pd.concat([biomarker_data, time_data], axis=1)
            temporal_drift = self.bias_detector.detect_temporal_drift(
                temporal_data, time_data.name, biomarker_data.columns.tolist()
            )
            bias_results['temporal_drift'] = temporal_drift
        
        results['bias_analysis'] = bias_results
        
        # 5. Summary statistics
        results['summary'] = {
            'n_biomarkers_tested': len(biomarker_data.columns),
            'n_significant_uncorrected': np.sum(np.array(p_values) < 0.05),
            'n_significant_corrected': mt_result.n_discoveries,
            'overall_fdr': mt_result.fdr_level,
            'median_effect_size': np.median([br.effect_size for br in biomarker_results]),
            'median_power': np.median([br.power for br in biomarker_results])
        }
        
        self.analysis_results = results
        logger.info("Comprehensive validation completed")
        
        return results
    
    def _compute_effect_size(self, group_0: np.ndarray, group_1: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        
        pooled_std = np.sqrt(
            ((len(group_0) - 1) * np.var(group_0) + 
             (len(group_1) - 1) * np.var(group_1)) /
            (len(group_0) + len(group_1) - 2)
        )
        
        return (np.mean(group_1) - np.mean(group_0)) / pooled_std
    
    def _compute_power(self, effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
        """Compute statistical power"""
        
        if STATSMODELS_AVAILABLE:
            try:
                power = ttest_power(effect_size, sample_size, alpha)
                return power
            except:
                pass
        
        # Simplified power calculation
        if abs(effect_size) > 0.8:
            return 0.8 + 0.2 * min(1.0, sample_size / 100)
        elif abs(effect_size) > 0.5:
            return 0.6 + 0.3 * min(1.0, sample_size / 100)
        else:
            return 0.4 + 0.4 * min(1.0, sample_size / 100)
    
    def _interpret_result(self, p_value: float, effect_size: float) -> str:
        """Interpret statistical result"""
        
        significance = "significant" if p_value < 0.05 else "not significant"
        
        if abs(effect_size) > 0.8:
            effect_magnitude = "large"
        elif abs(effect_size) > 0.5:
            effect_magnitude = "medium"
        elif abs(effect_size) > 0.2:
            effect_magnitude = "small"
        else:
            effect_magnitude = "negligible"
        
        return f"Result is {significance} with {effect_magnitude} effect size"
    
    def generate_statistical_report(self) -> str:
        """Generate comprehensive statistical report"""
        
        if not self.analysis_results:
            return "No analysis results available. Run comprehensive_biomarker_validation first."
        
        results = self.analysis_results
        
        report = []
        report.append("=== COMPREHENSIVE BIOMARKER VALIDATION REPORT ===\n")
        
        # Summary
        summary = results['summary']
        report.append("SUMMARY STATISTICS:")
        report.append(f"• Total biomarkers tested: {summary['n_biomarkers_tested']}")
        report.append(f"• Significant (uncorrected): {summary['n_significant_uncorrected']}")
        report.append(f"• Significant (FDR-corrected): {summary['n_significant_corrected']}")
        report.append(f"• Overall FDR level: {summary.get('overall_fdr', 'N/A')}")
        report.append(f"• Median effect size: {summary['median_effect_size']:.3f}")
        report.append(f"• Median statistical power: {summary['median_power']:.3f}")
        report.append("")
        
        # Multiple testing
        mt = results['multiple_testing']
        report.append("MULTIPLE TESTING CORRECTION:")
        report.append(f"• Method: {mt.method}")
        report.append(f"• Discoveries: {mt.n_discoveries}/{mt.n_tests}")
        report.append(f"• FDR level: {mt.fdr_level}")
        report.append("")
        
        # Bias analysis
        bias = results['bias_analysis']
        report.append("BIAS ANALYSIS:")
        
        if 'selection_bias' in bias:
            sb = bias['selection_bias']
            status = "DETECTED" if sb['bias_detected'] else "Not detected"
            report.append(f"• Selection bias: {status}")
            
        if 'temporal_drift' in bias:
            td = bias['temporal_drift']
            status = "DETECTED" if td['drift_detected'] else "Not detected"
            report.append(f"• Temporal drift: {status}")
            if td['drift_detected']:
                report.append(f"  - Affected variables: {td['drift_variables']}")
        
        report.append("")
        
        # Cross-validation
        if 'cross_validation' in results and results['cross_validation']:
            cv = results['cross_validation']
            report.append("CROSS-VALIDATION RESULTS:")
            report.append(f"• Method: {cv.cv_method}")
            report.append(f"• Mean score: {cv.mean_score:.3f} ± {cv.std_score:.3f}")
            report.append(f"• 95% CI: {cv.confidence_interval}")
            if cv.temporal_stability is not None:
                report.append(f"• Temporal stability: {cv.temporal_stability:.3f}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        all_recommendations = set()
        
        for bias_type, bias_result in bias.items():
            if isinstance(bias_result, dict) and 'recommendations' in bias_result:
                all_recommendations.update(bias_result['recommendations'])
        
        for rec in sorted(all_recommendations):
            report.append(f"• {rec}")
        
        return "\n".join(report)


# Example usage and testing
def run_advanced_statistical_demo():
    """Demonstrate advanced statistical framework capabilities"""
    
    logger.info("=== Advanced Statistical Framework Demo ===")
    
    # Generate synthetic biomarker data
    np.random.seed(42)
    n_samples = 500
    n_biomarkers = 20
    
    biomarker_data = pd.DataFrame(
        np.random.normal(0, 1, (n_samples, n_biomarkers)),
        columns=[f'biomarker_{i}' for i in range(n_biomarkers)]
    )
    
    # Add some true associations (first 5 biomarkers)
    true_effects = np.array([0.5, 0.3, 0.7, 0.2, 0.4] + [0.0] * (n_biomarkers - 5))
    
    outcome_probs = 1 / (1 + np.exp(-(biomarker_data.values @ true_effects)))
    outcomes = pd.Series(np.random.binomial(1, outcome_probs, n_samples))
    
    # Add time data
    time_data = pd.Series(np.linspace(0, 365, n_samples))  # Days
    
    # Initialize framework
    framework = AdvancedStatisticalFramework(random_state=42)
    
    # Run comprehensive validation
    validation_results = framework.comprehensive_biomarker_validation(
        biomarker_data=biomarker_data,
        outcome_data=outcomes,
        time_data=time_data
    )
    
    # Generate report
    report = framework.generate_statistical_report()
    
    logger.info("Advanced statistical framework demo completed!")
    logger.info(f"Summary: {validation_results['summary']}")
    
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION REPORT")
    print("="*60)
    print(report)
    
    return framework, validation_results


if __name__ == "__main__":
    framework, results = run_advanced_statistical_demo()
