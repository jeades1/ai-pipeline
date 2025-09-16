"""
Multi-modal Data Fusion Module

This module implements weighted ensemble modeling for integrating molecular signatures,
functional readouts, and clinical metrics into unified biomarker predictions.

Key Features:
- Multi-modal data integration with standardized preprocessing
- Weighted ensemble modeling with adaptive weights
- Feature importance analysis across modalities
- Cross-modal validation and performance assessment
- Uncertainty quantification for ensemble predictions

Author: AI Pipeline Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import json
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
import joblib

# Configure logging
logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class ModalityType(Enum):
    """Types of data modalities"""
    MOLECULAR = "molecular"
    FUNCTIONAL = "functional"
    CLINICAL = "clinical"
    TEMPORAL = "temporal"
    IMAGING = "imaging"


class FusionStrategy(Enum):
    """Strategies for multi-modal fusion"""
    EARLY_FUSION = "early_fusion"  # Concatenate features before modeling
    LATE_FUSION = "late_fusion"    # Combine predictions from separate models
    INTERMEDIATE_FUSION = "intermediate_fusion"  # Fusion at multiple levels
    ADAPTIVE_FUSION = "adaptive_fusion"  # Learned fusion weights


class EnsembleMethod(Enum):
    """Ensemble methods for fusion"""
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    VOTING = "voting"
    BAYESIAN_AVERAGE = "bayesian_average"
    NEURAL_FUSION = "neural_fusion"


class TaskType(Enum):
    """Types of prediction tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_LABEL = "multi_label"


@dataclass
class ModalityData:
    """Data container for a single modality"""
    modality_type: ModalityType
    data: pd.DataFrame
    feature_names: List[str]
    quality_scores: Optional[pd.Series] = None
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    importance_weights: Optional[np.ndarray] = None


@dataclass
class EnsembleModel:
    """Ensemble model container"""
    model_id: str
    base_models: Dict[ModalityType, Any]
    fusion_weights: Dict[ModalityType, float]
    ensemble_method: EnsembleMethod
    fusion_strategy: FusionStrategy
    task_type: TaskType
    performance_metrics: Dict[str, float]
    feature_importance: Dict[ModalityType, np.ndarray]
    uncertainty_estimates: Optional[Dict[str, float]] = None


@dataclass
class FusionResult:
    """Results from multi-modal fusion"""
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray]
    modality_contributions: Dict[ModalityType, np.ndarray]
    uncertainty_scores: np.ndarray
    confidence_intervals: Optional[np.ndarray]
    ensemble_weights: Dict[ModalityType, float]


class MultiModalFusion:
    """
    Multi-modal data fusion for biomarker prediction
    
    This class implements comprehensive multi-modal data fusion capabilities
    including weighted ensemble modeling, cross-modal validation, and uncertainty
    quantification for biomarker discovery and validation.
    """
    
    def __init__(self,
                 fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION,
                 ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
                 task_type: TaskType = TaskType.CLASSIFICATION,
                 output_dir: Path = Path("fusion_outputs")):
        """
        Initialize multi-modal fusion system
        
        Args:
            fusion_strategy: Strategy for combining modalities
            ensemble_method: Method for ensemble learning
            task_type: Type of prediction task
            output_dir: Directory for saving results
        """
        self.fusion_strategy = fusion_strategy
        self.ensemble_method = ensemble_method
        self.task_type = task_type
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.modalities: Dict[ModalityType, ModalityData] = {}
        self.ensemble_models: Dict[str, EnsembleModel] = {}
        self.targets: Optional[pd.Series] = None
        
        # Preprocessing
        self.scalers: Dict[ModalityType, Any] = {}
        self.feature_selectors: Dict[ModalityType, Any] = {}
        
        # Results
        self.fusion_results: Dict[str, FusionResult] = {}
        
        logger.info(f"Initialized Multi-Modal Fusion System")
        logger.info(f"Fusion strategy: {fusion_strategy.value}")
        logger.info(f"Ensemble method: {ensemble_method.value}")
        logger.info(f"Task type: {task_type.value}")
    
    def add_modality(self,
                    modality_type: ModalityType,
                    data: pd.DataFrame,
                    feature_columns: Optional[List[str]] = None,
                    quality_scores: Optional[pd.Series] = None) -> None:
        """Add a data modality to the fusion system"""
        
        logger.info(f"Adding {modality_type.value} modality")
        
        if feature_columns is None:
            feature_columns = list(data.columns)
        
        # Validate data
        if data.empty:
            raise ValueError(f"Empty data provided for {modality_type.value} modality")
        
        # Create modality data container
        modality_data = ModalityData(
            modality_type=modality_type,
            data=data.copy(),
            feature_names=feature_columns,
            quality_scores=quality_scores
        )
        
        self.modalities[modality_type] = modality_data
        
        logger.info(f"Added {modality_type.value} modality: {data.shape[0]} samples, {len(feature_columns)} features")
    
    def set_targets(self, targets: pd.Series) -> None:
        """Set prediction targets"""
        
        logger.info(f"Setting targets: {len(targets)} samples")
        
        if self.task_type == TaskType.CLASSIFICATION:
            logger.info(f"Target classes: {sorted(targets.unique())}")
        else:
            logger.info(f"Target range: {targets.min():.3f} - {targets.max():.3f}")
        
        self.targets = targets.copy()
    
    def preprocess_modalities(self,
                            scaling_method: str = "standard",
                            feature_selection: bool = True,
                            n_features: Optional[int] = None) -> None:
        """Preprocess all modalities with scaling and feature selection"""
        
        logger.info("Preprocessing modalities")
        
        for modality_type, modality_data in self.modalities.items():
            logger.info(f"Preprocessing {modality_type.value} modality...")
            
            # Scaling
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            # Fit scaler and transform data
            scaled_data = scaler.fit_transform(modality_data.data[modality_data.feature_names])
            
            # Update data
            modality_data.data[modality_data.feature_names] = scaled_data
            modality_data.preprocessing_params['scaler'] = scaling_method
            
            # Store scaler
            self.scalers[modality_type] = scaler
            
            # Feature selection
            if feature_selection and self.targets is not None:
                if n_features is None:
                    n_features = min(50, len(modality_data.feature_names) // 2)
                
                n_features = min(n_features, len(modality_data.feature_names))
                
                if self.task_type == TaskType.CLASSIFICATION:
                    selector = SelectKBest(score_func=f_classif, k=n_features)
                else:
                    selector = SelectKBest(score_func=f_regression, k=n_features)
                
                # Align data with targets
                aligned_data, aligned_targets = self._align_data(modality_data.data, self.targets)
                
                if len(aligned_data) > 0:
                    selected_features = selector.fit_transform(
                        aligned_data[modality_data.feature_names], aligned_targets
                    )
                    
                    # Get selected feature names
                    selected_indices = selector.get_support()
                    selected_names = [name for i, name in enumerate(modality_data.feature_names) if selected_indices[i]]
                    
                    # Update feature names
                    modality_data.feature_names = selected_names
                    modality_data.preprocessing_params['feature_selection'] = True
                    modality_data.preprocessing_params['n_selected_features'] = len(selected_names)
                    
                    # Store feature selector
                    self.feature_selectors[modality_type] = selector
                    
                    logger.info(f"Selected {len(selected_names)} features for {modality_type.value}")
            
            logger.info(f"Preprocessed {modality_type.value}: {modality_data.data.shape}")
    
    def train_base_models(self,
                         models_per_modality: Optional[Dict[ModalityType, List[str]]] = None,
                         cv_folds: int = 5) -> Dict[ModalityType, Dict[str, Any]]:
        """Train base models for each modality"""
        
        logger.info("Training base models for each modality")
        
        if self.targets is None:
            raise ValueError("Targets must be set before training models")
        
        if models_per_modality is None:
            # Default models
            if self.task_type == TaskType.CLASSIFICATION:
                default_models = ['random_forest', 'gradient_boosting', 'logistic_regression']
            else:
                default_models = ['random_forest', 'gradient_boosting', 'ridge']
            
            models_per_modality = {modality: default_models for modality in self.modalities.keys()}
        
        base_models = {}
        
        for modality_type, modality_data in self.modalities.items():
            logger.info(f"Training models for {modality_type.value} modality...")
            
            modality_models = {}
            model_names = models_per_modality.get(modality_type, ['random_forest'])
            
            # Align data with targets
            aligned_data, aligned_targets = self._align_data(modality_data.data, self.targets)
            
            if len(aligned_data) == 0:
                logger.warning(f"No aligned data for {modality_type.value} modality")
                continue
            
            X = aligned_data[modality_data.feature_names]
            y = aligned_targets
            
            for model_name in model_names:
                logger.info(f"  Training {model_name}...")
                
                # Create model
                model = self._create_model(model_name)
                
                # Train model
                model.fit(X, y)
                
                # Cross-validate
                if self.task_type == TaskType.CLASSIFICATION:
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
                    metric_name = 'ROC AUC'
                else:
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                    metric_name = 'R²'
                
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                modality_models[model_name] = {
                    'model': model,
                    'cv_score': cv_mean,
                    'cv_std': cv_std,
                    'metric': metric_name
                }
                
                logger.info(f"    {metric_name}: {cv_mean:.3f} ± {cv_std:.3f}")
            
            base_models[modality_type] = modality_models
        
        return base_models
    
    def create_ensemble(self,
                       ensemble_id: str,
                       base_models: Dict[ModalityType, Dict[str, Any]],
                       weight_optimization: bool = True) -> EnsembleModel:
        """Create ensemble model from base models"""
        
        logger.info(f"Creating ensemble model: {ensemble_id}")
        
        # Select best model per modality
        selected_models = {}
        for modality_type, models in base_models.items():
            if models:
                best_model_name = max(models.keys(), key=lambda k: models[k]['cv_score'])
                selected_models[modality_type] = models[best_model_name]['model']
                logger.info(f"Selected {best_model_name} for {modality_type.value} (CV: {models[best_model_name]['cv_score']:.3f})")
        
        # Initialize fusion weights
        if weight_optimization:
            fusion_weights = self._optimize_fusion_weights(selected_models)
        else:
            # Equal weights
            n_modalities = len(selected_models)
            fusion_weights = {modality: 1.0/n_modalities for modality in selected_models.keys()}
        
        # Calculate feature importance
        feature_importance = self._calculate_ensemble_feature_importance(selected_models)
        
        # Evaluate ensemble performance
        performance_metrics = self._evaluate_ensemble(selected_models, fusion_weights)
        
        # Create ensemble model
        ensemble_model = EnsembleModel(
            model_id=ensemble_id,
            base_models=selected_models,
            fusion_weights=fusion_weights,
            ensemble_method=self.ensemble_method,
            fusion_strategy=self.fusion_strategy,
            task_type=self.task_type,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance
        )
        
        self.ensemble_models[ensemble_id] = ensemble_model
        
        logger.info(f"Ensemble created successfully")
        logger.info(f"Fusion weights: {fusion_weights}")
        logger.info(f"Performance: {performance_metrics}")
        
        return ensemble_model
    
    def predict_ensemble(self,
                        ensemble_id: str,
                        data: Optional[Dict[ModalityType, pd.DataFrame]] = None,
                        return_uncertainty: bool = True) -> FusionResult:
        """Make predictions using ensemble model"""
        
        logger.info(f"Making ensemble predictions: {ensemble_id}")
        
        if ensemble_id not in self.ensemble_models:
            raise ValueError(f"Ensemble model {ensemble_id} not found")
        
        ensemble_model = self.ensemble_models[ensemble_id]
        
        # Use training data if no new data provided
        if data is None:
            data = {modality_type: modality_data.data 
                   for modality_type, modality_data in self.modalities.items()}
        
        # Get predictions from each modality
        modality_predictions = {}
        modality_probabilities = {}
        
        for modality_type, model in ensemble_model.base_models.items():
            if modality_type not in data:
                logger.warning(f"No data provided for {modality_type.value} modality")
                continue
            
            modality_data = self.modalities[modality_type]
            X = data[modality_type][modality_data.feature_names]
            
            # Make predictions
            predictions = model.predict(X)
            modality_predictions[modality_type] = predictions
            
            # Get probabilities for classification
            if (self.task_type == TaskType.CLASSIFICATION and 
                hasattr(model, 'predict_proba')):
                probabilities = model.predict_proba(X)
                modality_probabilities[modality_type] = probabilities
        
        # Combine predictions using fusion weights
        if self.fusion_strategy == FusionStrategy.LATE_FUSION:
            final_predictions = self._late_fusion(modality_predictions, ensemble_model.fusion_weights)
            final_probabilities = self._late_fusion_probabilities(modality_probabilities, ensemble_model.fusion_weights)
        else:
            # For now, implement late fusion as default
            final_predictions = self._late_fusion(modality_predictions, ensemble_model.fusion_weights)
            final_probabilities = self._late_fusion_probabilities(modality_probabilities, ensemble_model.fusion_weights)
        
        # Calculate uncertainty scores
        if return_uncertainty:
            uncertainty_scores = self._calculate_prediction_uncertainty(
                modality_predictions, modality_probabilities, ensemble_model.fusion_weights
            )
        else:
            uncertainty_scores = np.zeros(len(final_predictions))
        
        # Calculate modality contributions
        modality_contributions = self._calculate_modality_contributions(
            modality_predictions, ensemble_model.fusion_weights
        )
        
        # Create fusion result
        fusion_result = FusionResult(
            predictions=final_predictions,
            prediction_probabilities=final_probabilities,
            modality_contributions=modality_contributions,
            uncertainty_scores=uncertainty_scores,
            confidence_intervals=None,  # Could be implemented
            ensemble_weights=ensemble_model.fusion_weights
        )
        
        self.fusion_results[ensemble_id] = fusion_result
        
        logger.info(f"Ensemble predictions completed: {len(final_predictions)} samples")
        
        return fusion_result
    
    def validate_ensemble(self,
                         ensemble_id: str,
                         cv_folds: int = 5) -> Dict[str, float]:
        """Validate ensemble model using cross-validation"""
        
        logger.info(f"Validating ensemble model: {ensemble_id}")
        
        if ensemble_id not in self.ensemble_models:
            raise ValueError(f"Ensemble model {ensemble_id} not found")
        
        ensemble_model = self.ensemble_models[ensemble_id]
        
        # Prepare data for cross-validation
        all_data = []
        all_targets = []
        
        for modality_type, modality_data in self.modalities.items():
            if modality_type in ensemble_model.base_models:
                aligned_data, aligned_targets = self._align_data(modality_data.data, self.targets)
                if len(aligned_data) > 0:
                    all_data.append(aligned_data)
                    all_targets.append(aligned_targets)
        
        if not all_data:
            raise ValueError("No aligned data available for validation")
        
        # Use the first aligned dataset for CV splits (they should all be aligned)
        common_indices = all_data[0].index
        y = all_targets[0]
        
        # Cross-validation
        if self.task_type == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        
        for train_idx, test_idx in cv.split(common_indices, y):
            # Split data
            train_data = {}
            test_data = {}
            
            for modality_type, modality_data in self.modalities.items():
                if modality_type in ensemble_model.base_models:
                    modality_df = modality_data.data.loc[common_indices]
                    train_data[modality_type] = modality_df.iloc[train_idx]
                    test_data[modality_type] = modality_df.iloc[test_idx]
            
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Train models on fold
            fold_models = {}
            for modality_type, base_model in ensemble_model.base_models.items():
                # Clone and retrain model
                fold_model = self._clone_model(base_model)
                X_train = train_data[modality_type][self.modalities[modality_type].feature_names]
                fold_model.fit(X_train, y_train)
                fold_models[modality_type] = fold_model
            
            # Make predictions on test set
            modality_predictions = {}
            for modality_type, model in fold_models.items():
                X_test = test_data[modality_type][self.modalities[modality_type].feature_names]
                predictions = model.predict(X_test)
                modality_predictions[modality_type] = predictions
            
            # Combine predictions
            ensemble_predictions = self._late_fusion(modality_predictions, ensemble_model.fusion_weights)
            
            # Calculate score
            if self.task_type == TaskType.CLASSIFICATION:
                if len(np.unique(y)) == 2:  # Binary classification
                    score = roc_auc_score(y_test, ensemble_predictions)
                else:  # Multi-class
                    score = accuracy_score(y_test, ensemble_predictions)
            else:
                score = r2_score(y_test, ensemble_predictions)
            
            cv_scores.append(score)
        
        # Calculate validation metrics
        validation_results = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores,
            'n_folds': cv_folds
        }
        
        logger.info(f"Validation completed: {validation_results['cv_mean']:.3f} ± {validation_results['cv_std']:.3f}")
        
        return validation_results
    
    def analyze_modality_importance(self,
                                  ensemble_id: str) -> Dict[ModalityType, Dict[str, float]]:
        """Analyze importance of each modality in the ensemble"""
        
        logger.info(f"Analyzing modality importance for ensemble: {ensemble_id}")
        
        if ensemble_id not in self.ensemble_models:
            raise ValueError(f"Ensemble model {ensemble_id} not found")
        
        ensemble_model = self.ensemble_models[ensemble_id]
        importance_analysis = {}
        
        for modality_type in ensemble_model.base_models.keys():
            # Get modality weight
            fusion_weight = ensemble_model.fusion_weights.get(modality_type, 0.0)
            
            # Get feature importance from base model
            base_model = ensemble_model.base_models[modality_type]
            if hasattr(base_model, 'feature_importances_'):
                feature_importance = np.mean(base_model.feature_importances_)
            elif hasattr(base_model, 'coef_'):
                feature_importance = np.mean(np.abs(base_model.coef_))
            else:
                feature_importance = 0.0
            
            # Get modality data quality
            modality_data = self.modalities[modality_type]
            if modality_data.quality_scores is not None:
                data_quality = np.mean(modality_data.quality_scores)
            else:
                data_quality = 1.0
            
            # Calculate overall importance score
            overall_importance = fusion_weight * feature_importance * data_quality
            
            importance_analysis[modality_type] = {
                'fusion_weight': fusion_weight,
                'feature_importance': feature_importance,
                'data_quality': data_quality,
                'overall_importance': overall_importance,
                'n_features': len(modality_data.feature_names)
            }
        
        # Sort by overall importance
        sorted_importance = dict(sorted(
            importance_analysis.items(),
            key=lambda x: x[1]['overall_importance'],
            reverse=True
        ))
        
        logger.info("Modality importance ranking:")
        for modality_type, metrics in sorted_importance.items():
            logger.info(f"  {modality_type.value}: {metrics['overall_importance']:.3f}")
        
        return sorted_importance
    
    def generate_fusion_report(self) -> pd.DataFrame:
        """Generate comprehensive fusion analysis report"""
        
        logger.info("Generating fusion analysis report")
        
        report_data = []
        
        for ensemble_id, ensemble_model in self.ensemble_models.items():
            # Basic ensemble info
            row = {
                'ensemble_id': ensemble_id,
                'fusion_strategy': ensemble_model.fusion_strategy.value,
                'ensemble_method': ensemble_model.ensemble_method.value,
                'task_type': ensemble_model.task_type.value,
                'n_modalities': len(ensemble_model.base_models),
                'n_total_features': sum(len(self.modalities[modality].feature_names) 
                                      for modality in ensemble_model.base_models.keys())
            }
            
            # Performance metrics
            for metric_name, metric_value in ensemble_model.performance_metrics.items():
                row[f'performance_{metric_name}'] = metric_value
            
            # Modality weights
            for modality_type, weight in ensemble_model.fusion_weights.items():
                row[f'weight_{modality_type.value}'] = weight
            
            # Individual modality feature counts
            for modality_type in ensemble_model.base_models.keys():
                n_features = len(self.modalities[modality_type].feature_names)
                row[f'n_features_{modality_type.value}'] = n_features
            
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_file = self.output_dir / "fusion_analysis_report.csv"
        report_df.to_csv(report_file, index=False)
        
        logger.info(f"Fusion analysis report saved to {report_file}")
        
        return report_df
    
    def save_ensemble_models(self) -> None:
        """Save ensemble models and metadata"""
        
        logger.info("Saving ensemble models")
        
        # Save fitted models
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for ensemble_id, ensemble_model in self.ensemble_models.items():
            # Save base models
            for modality_type, model in ensemble_model.base_models.items():
                model_file = models_dir / f"{ensemble_id}_{modality_type.value}_model.joblib"
                joblib.dump(model, model_file)
        
        # Save ensemble metadata
        ensemble_data = {}
        for ensemble_id, ensemble_model in self.ensemble_models.items():
            ensemble_data[ensemble_id] = {
                'model_id': ensemble_model.model_id,
                'fusion_strategy': ensemble_model.fusion_strategy.value,
                'ensemble_method': ensemble_model.ensemble_method.value,
                'task_type': ensemble_model.task_type.value,
                'fusion_weights': {k.value: v for k, v in ensemble_model.fusion_weights.items()},
                'performance_metrics': ensemble_model.performance_metrics,
                'modalities': list(ensemble_model.base_models.keys())
            }
        
        ensemble_file = self.output_dir / "ensemble_models.json"
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_data, f, indent=2, default=str)
        
        # Save scalers and preprocessors
        preprocessing_dir = self.output_dir / "preprocessing"
        preprocessing_dir.mkdir(exist_ok=True)
        
        for modality_type, scaler in self.scalers.items():
            scaler_file = preprocessing_dir / f"{modality_type.value}_scaler.joblib"
            joblib.dump(scaler, scaler_file)
        
        for modality_type, selector in self.feature_selectors.items():
            selector_file = preprocessing_dir / f"{modality_type.value}_selector.joblib"
            joblib.dump(selector, selector_file)
        
        logger.info(f"Ensemble models saved to {models_dir}")
        logger.info(f"Metadata saved to {self.output_dir}")
    
    # Private helper methods
    
    def _align_data(self, data: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align data and targets by common indices"""
        
        common_indices = data.index.intersection(targets.index)
        
        if len(common_indices) == 0:
            logger.warning("No common indices found between data and targets")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        aligned_data = data.loc[common_indices]
        aligned_targets = targets.loc[common_indices]
        
        return aligned_data, aligned_targets
    
    def _create_model(self, model_name: str) -> Any:
        """Create model instance based on name"""
        
        if self.task_type == TaskType.CLASSIFICATION:
            if model_name == 'random_forest':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == 'gradient_boosting':
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_name == 'logistic_regression':
                return LogisticRegression(random_state=42, max_iter=1000)
            elif model_name == 'svm':
                return SVC(probability=True, random_state=42)
            elif model_name == 'neural_network':
                return MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        else:  # Regression
            if model_name == 'random_forest':
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == 'gradient_boosting':
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_name == 'ridge':
                return Ridge(random_state=42)
            elif model_name == 'elastic_net':
                return ElasticNet(random_state=42)
            elif model_name == 'svr':
                return SVR()
            elif model_name == 'neural_network':
                return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        
        # Default to random forest
        if self.task_type == TaskType.CLASSIFICATION:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model with same parameters"""
        
        model_params = model.get_params()
        return type(model)(**model_params)
    
    def _optimize_fusion_weights(self, models: Dict[ModalityType, Any]) -> Dict[ModalityType, float]:
        """Optimize fusion weights using validation performance"""
        
        logger.info("Optimizing fusion weights")
        
        if self.targets is None:
            logger.warning("No targets available for weight optimization")
            return {modality: 1.0/len(models) for modality in models.keys()}
        
        if len(models) <= 1:
            return {modality: 1.0 for modality in models.keys()}
        
        # Get validation predictions for each modality
        modality_predictions = {}
        
        for modality_type, model in models.items():
            modality_data = self.modalities[modality_type]
            aligned_data, aligned_targets = self._align_data(modality_data.data, self.targets)
            
            if len(aligned_data) > 0:
                X = aligned_data[modality_data.feature_names]
                predictions = model.predict(X)
                modality_predictions[modality_type] = predictions
        
        if not modality_predictions:
            return {modality: 1.0/len(models) for modality in models.keys()}
        
        # Common targets for all modalities
        common_indices = None
        for modality_type in modality_predictions.keys():
            modality_data = self.modalities[modality_type]
            aligned_data, _ = self._align_data(modality_data.data, self.targets)
            if common_indices is None:
                common_indices = aligned_data.index
            else:
                common_indices = common_indices.intersection(aligned_data.index)
        
        if common_indices is None or len(common_indices) == 0:
            return {modality: 1.0/len(models) for modality in models.keys()}
        
        y_true = self.targets.loc[common_indices]
        
        # Objective function for weight optimization
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate weighted prediction
            weighted_pred = np.zeros(len(y_true))
            for i, modality_type in enumerate(modality_predictions.keys()):
                # Get predictions for common indices
                modality_data = self.modalities[modality_type]
                aligned_data, _ = self._align_data(modality_data.data, self.targets)
                pred_indices = aligned_data.index.intersection(common_indices)
                
                if len(pred_indices) > 0:
                    # Map predictions to common indices
                    pred_values = modality_predictions[modality_type]
                    # Simple mapping - assumes aligned order
                    weighted_pred += weights[i] * pred_values[:len(weighted_pred)]
            
            # Calculate loss
            if self.task_type == TaskType.CLASSIFICATION:
                # Use log loss for classification (simplified)
                return mean_squared_error(y_true, weighted_pred)
            else:
                return mean_squared_error(y_true, weighted_pred)
        
        # Initial weights (equal)
        n_modalities = len(modality_predictions)
        initial_weights = np.ones(n_modalities) / n_modalities
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_modalities)]
        
        try:
            result = minimize(objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x / np.sum(result.x)  # Normalize
                weight_dict = {modality: float(weight) 
                             for modality, weight in zip(modality_predictions.keys(), optimized_weights)}
                logger.info(f"Optimized weights: {weight_dict}")
                return weight_dict
        except Exception as e:
            logger.warning(f"Weight optimization failed: {e}")
        
        # Fallback to equal weights
        return {modality: 1.0/len(models) for modality in models.keys()}
    
    def _calculate_ensemble_feature_importance(self, models: Dict[ModalityType, Any]) -> Dict[ModalityType, np.ndarray]:
        """Calculate feature importance for ensemble"""
        
        feature_importance = {}
        
        for modality_type, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                # Default uniform importance
                n_features = len(self.modalities[modality_type].feature_names)
                importance = np.ones(n_features) / n_features
            
            feature_importance[modality_type] = importance
        
        return feature_importance
    
    def _evaluate_ensemble(self, models: Dict[ModalityType, Any], 
                          fusion_weights: Dict[ModalityType, float]) -> Dict[str, float]:
        """Evaluate ensemble model performance"""
        
        if self.targets is None:
            logger.warning("No targets available for ensemble evaluation")
            return {'error': 1.0}
        
        # Get predictions for each modality
        modality_predictions = {}
        
        for modality_type, model in models.items():
            modality_data = self.modalities[modality_type]
            aligned_data, aligned_targets = self._align_data(modality_data.data, self.targets)
            
            if len(aligned_data) > 0:
                X = aligned_data[modality_data.feature_names]
                predictions = model.predict(X)
                modality_predictions[modality_type] = predictions
        
        if not modality_predictions:
            return {'error': 1.0}
        
        # Combine predictions
        ensemble_predictions = self._late_fusion(modality_predictions, fusion_weights)
        
        # Get true targets (use first modality's aligned targets)
        first_modality = list(modality_predictions.keys())[0]
        modality_data = self.modalities[first_modality]
        _, y_true = self._align_data(modality_data.data, self.targets)
        
        # Calculate metrics
        metrics = {}
        
        if self.task_type == TaskType.CLASSIFICATION:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['accuracy'] = accuracy_score(y_true, ensemble_predictions > 0.5)
                metrics['precision'] = precision_score(y_true, ensemble_predictions > 0.5)
                metrics['recall'] = recall_score(y_true, ensemble_predictions > 0.5)
                metrics['f1'] = f1_score(y_true, ensemble_predictions > 0.5)
                # For ROC AUC, we need probabilities or scores
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, ensemble_predictions)
                except:
                    metrics['roc_auc'] = 0.5
            else:  # Multi-class
                metrics['accuracy'] = accuracy_score(y_true, ensemble_predictions.round())
                metrics['f1_macro'] = f1_score(y_true, ensemble_predictions.round(), average='macro')
        else:  # Regression
            metrics['r2'] = r2_score(y_true, ensemble_predictions)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, ensemble_predictions))
            metrics['mae'] = mean_absolute_error(y_true, ensemble_predictions)
        
        return metrics
    
    def _late_fusion(self, modality_predictions: Dict[ModalityType, np.ndarray],
                    fusion_weights: Dict[ModalityType, float]) -> np.ndarray:
        """Perform late fusion of modality predictions"""
        
        if not modality_predictions:
            return np.array([])
        
        # Find common sample size
        min_samples = min(len(pred) for pred in modality_predictions.values())
        
        # Initialize weighted sum
        weighted_sum = np.zeros(min_samples)
        total_weight = 0.0
        
        for modality_type, predictions in modality_predictions.items():
            weight = fusion_weights.get(modality_type, 0.0)
            weighted_sum += weight * predictions[:min_samples]
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return weighted_sum
    
    def _late_fusion_probabilities(self, modality_probabilities: Dict[ModalityType, np.ndarray],
                                  fusion_weights: Dict[ModalityType, float]) -> Optional[np.ndarray]:
        """Perform late fusion of prediction probabilities"""
        
        if not modality_probabilities:
            return None
        
        # Find common sample and class sizes
        min_samples = min(prob.shape[0] for prob in modality_probabilities.values())
        n_classes = None
        
        for prob in modality_probabilities.values():
            if n_classes is None:
                n_classes = prob.shape[1]
            elif prob.shape[1] != n_classes:
                # Inconsistent class numbers
                return None
        
        if n_classes is None:
            return None
        
        # Initialize weighted sum
        weighted_sum = np.zeros((min_samples, n_classes))
        total_weight = 0.0
        
        for modality_type, probabilities in modality_probabilities.items():
            weight = fusion_weights.get(modality_type, 0.0)
            weighted_sum += weight * probabilities[:min_samples, :]
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return weighted_sum
    
    def _calculate_prediction_uncertainty(self, 
                                        modality_predictions: Dict[ModalityType, np.ndarray],
                                        modality_probabilities: Dict[ModalityType, np.ndarray],
                                        fusion_weights: Dict[ModalityType, float]) -> np.ndarray:
        """Calculate prediction uncertainty scores"""
        
        if not modality_predictions:
            return np.array([])
        
        # Find common sample size
        min_samples = min(len(pred) for pred in modality_predictions.values())
        
        # Calculate variance across modality predictions
        prediction_matrix = np.column_stack([
            pred[:min_samples] for pred in modality_predictions.values()
        ])
        
        # Weight-adjusted variance
        weights = np.array([fusion_weights.get(modality, 0.0) 
                           for modality in modality_predictions.keys()])
        
        if np.sum(weights) == 0:
            return np.ones(min_samples) * 0.5  # Default uncertainty
        
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted variance
        weighted_mean = np.average(prediction_matrix, axis=1, weights=weights)
        uncertainty_scores = np.average(
            (prediction_matrix - weighted_mean.reshape(-1, 1))**2,
            axis=1, weights=weights
        )
        
        return uncertainty_scores
    
    def _calculate_modality_contributions(self,
                                        modality_predictions: Dict[ModalityType, np.ndarray],
                                        fusion_weights: Dict[ModalityType, float]) -> Dict[ModalityType, np.ndarray]:
        """Calculate contribution of each modality to final predictions"""
        
        contributions = {}
        
        if not modality_predictions:
            return contributions
        
        # Find common sample size
        min_samples = min(len(pred) for pred in modality_predictions.values())
        
        total_weight = sum(fusion_weights.get(modality, 0.0) 
                         for modality in modality_predictions.keys())
        
        for modality_type, predictions in modality_predictions.items():
            weight = fusion_weights.get(modality_type, 0.0)
            if total_weight > 0:
                contribution = (weight / total_weight) * predictions[:min_samples]
            else:
                contribution = predictions[:min_samples] / len(modality_predictions)
            
            contributions[modality_type] = contribution
        
        return contributions


def create_demo_multimodal_fusion():
    """Create demonstration of multi-modal data fusion"""
    
    print("\n🔗 MULTI-MODAL DATA FUSION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize fusion system
    fusion_system = MultiModalFusion(
        fusion_strategy=FusionStrategy.LATE_FUSION,
        ensemble_method=EnsembleMethod.WEIGHTED_AVERAGE,
        task_type=TaskType.CLASSIFICATION,
        output_dir=Path("demo_outputs/fusion")
    )
    
    print("📊 Creating multi-modal demonstration data...")
    
    # Create demonstration data for multiple modalities
    np.random.seed(42)
    n_samples = 200
    
    # Molecular data (gene expression, CCI scores)
    molecular_data = pd.DataFrame({
        **{f'GENE_{i:03d}': np.random.normal(0, 1, n_samples) for i in range(50)},
        **{f'CCI_{i:03d}': np.random.uniform(0, 1, n_samples) for i in range(20)}
    }, index=[f'SAMPLE_{i:03d}' for i in range(n_samples)])
    
    # Functional data (assay readouts)
    functional_data = pd.DataFrame({
        'teer_resistance': np.random.normal(1000, 200, n_samples),
        'permeability_coeff': np.random.lognormal(0, 0.5, n_samples),
        'viability_score': np.random.beta(2, 1, n_samples),
        'inflammatory_marker': np.random.gamma(2, 2, n_samples),
        'barrier_function': np.random.normal(0.8, 0.2, n_samples)
    }, index=[f'SAMPLE_{i:03d}' for i in range(n_samples)])
    
    # Clinical data (outcomes, demographics)
    clinical_data = pd.DataFrame({
        'age': np.random.normal(65, 15, n_samples),
        'creatinine_baseline': np.random.normal(1.2, 0.3, n_samples),
        'comorbidity_score': np.random.poisson(2, n_samples),
        'treatment_response': np.random.normal(0.5, 0.3, n_samples),
        'recovery_time': np.random.exponential(48, n_samples)
    }, index=[f'SAMPLE_{i:03d}' for i in range(n_samples)])
    
    # Create target variable (AKI prediction)
    # Make it somewhat dependent on the features
    risk_score = (
        0.3 * (molecular_data['GENE_001'] > 0).astype(int) +
        0.2 * (functional_data['teer_resistance'] < 800).astype(int) +
        0.3 * (clinical_data['creatinine_baseline'] > 1.5).astype(int) +
        0.2 * (clinical_data['age'] > 70).astype(int) +
        np.random.normal(0, 0.1, n_samples)
    )
    targets = pd.Series((risk_score > 0.5).astype(int), 
                       index=[f'SAMPLE_{i:03d}' for i in range(n_samples)],
                       name='aki_outcome')
    
    print(f"   Molecular data: {molecular_data.shape}")
    print(f"   Functional data: {functional_data.shape}")
    print(f"   Clinical data: {clinical_data.shape}")
    print(f"   Targets: {targets.sum()} positive cases / {len(targets)} total")
    
    print("\n🔗 Adding modalities to fusion system...")
    
    # Add modalities
    fusion_system.add_modality(
        modality_type=ModalityType.MOLECULAR,
        data=molecular_data,
        feature_columns=list(molecular_data.columns)
    )
    
    fusion_system.add_modality(
        modality_type=ModalityType.FUNCTIONAL,
        data=functional_data,
        feature_columns=list(functional_data.columns)
    )
    
    fusion_system.add_modality(
        modality_type=ModalityType.CLINICAL,
        data=clinical_data,
        feature_columns=list(clinical_data.columns)
    )
    
    # Set targets
    fusion_system.set_targets(targets)
    
    print(f"   Added {len(fusion_system.modalities)} modalities")
    
    print("\n⚙️  Preprocessing modalities...")
    
    # Preprocess data
    fusion_system.preprocess_modalities(
        scaling_method="standard",
        feature_selection=True,
        n_features=20  # Select top features per modality
    )
    
    for modality_type, modality_data in fusion_system.modalities.items():
        print(f"   {modality_type.value}: {len(modality_data.feature_names)} features selected")
    
    print("\n🤖 Training base models...")
    
    # Train base models
    base_models = fusion_system.train_base_models(
        models_per_modality={
            ModalityType.MOLECULAR: ['random_forest', 'logistic_regression'],
            ModalityType.FUNCTIONAL: ['random_forest', 'gradient_boosting'],
            ModalityType.CLINICAL: ['logistic_regression', 'random_forest']
        },
        cv_folds=5
    )
    
    print(f"   Trained models for {len(base_models)} modalities")
    
    print("\n🎯 Creating ensemble model...")
    
    # Create ensemble
    ensemble_model = fusion_system.create_ensemble(
        ensemble_id="multimodal_aki_predictor",
        base_models=base_models,
        weight_optimization=True
    )
    
    print(f"   Ensemble created with {len(ensemble_model.base_models)} base models")
    print(f"   Fusion weights: {ensemble_model.fusion_weights}")
    
    print("\n🔮 Making ensemble predictions...")
    
    # Make predictions
    fusion_result = fusion_system.predict_ensemble(
        ensemble_id="multimodal_aki_predictor",
        return_uncertainty=True
    )
    
    print(f"   Generated predictions for {len(fusion_result.predictions)} samples")
    print(f"   Mean uncertainty: {np.mean(fusion_result.uncertainty_scores):.3f}")
    
    print("\n✅ Validating ensemble model...")
    
    # Validate ensemble
    validation_results = fusion_system.validate_ensemble(
        ensemble_id="multimodal_aki_predictor",
        cv_folds=5
    )
    
    print(f"   Cross-validation ROC AUC: {validation_results['cv_mean']:.3f} ± {validation_results['cv_std']:.3f}")
    
    print("\n📊 Analyzing modality importance...")
    
    # Analyze modality importance
    importance_analysis = fusion_system.analyze_modality_importance(
        ensemble_id="multimodal_aki_predictor"
    )
    
    print("   Modality importance ranking:")
    for modality_type, metrics in importance_analysis.items():
        print(f"      {modality_type.value}: {metrics['overall_importance']:.3f} "
              f"(weight: {metrics['fusion_weight']:.3f})")
    
    print("\n📋 Generating fusion report...")
    
    # Generate report
    report_df = fusion_system.generate_fusion_report()
    fusion_system.save_ensemble_models()
    
    print(f"\n📊 Multi-Modal Fusion Summary:")
    print(f"   Ensemble models: {len(fusion_system.ensemble_models)}")
    print(f"   Total modalities: {len(fusion_system.modalities)}")
    print(f"   Validation performance: {validation_results['cv_mean']:.3f}")
    print(f"   Best modality: {max(importance_analysis.keys(), key=lambda k: importance_analysis[k]['overall_importance']).value}")
    
    if not report_df.empty:
        print(f"\n🎯 Ensemble Performance:")
        for _, row in report_df.iterrows():
            print(f"   {row['ensemble_id']}: "
                  f"Modalities={row['n_modalities']}, "
                  f"Features={row['n_total_features']}")
    
    print(f"\n✅ Multi-modal data fusion demonstration completed!")
    print(f"📁 Results saved to: demo_outputs/fusion/")
    
    return fusion_system, ensemble_model, fusion_result, importance_analysis


if __name__ == "__main__":
    create_demo_multimodal_fusion()
