"""
Quality Metrics Calculator
Implements all quality metrics for agent evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

logger = logging.getLogger(__name__)


class QualityMetricsCalculator:
    """Calculates quality metrics for agent evaluation."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    # ========== Data Discovery Agent Metrics ==========
    
    def calculate_type_detection_accuracy(
        self, 
        detected_types: Dict[str, str],
        ground_truth_types: Dict[str, str]
    ) -> float:
        """
        Calculate accuracy of data type detection.
        
        Args:
            detected_types: Dictionary mapping column names to detected types
            ground_truth_types: Dictionary mapping column names to actual types
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not detected_types or not ground_truth_types:
            return 0.0
        
        correct = 0
        total = 0
        
        for col, true_type in ground_truth_types.items():
            if col in detected_types:
                detected_type = detected_types[col]
                # Normalize types for comparison
                if self._normalize_type(detected_type) == self._normalize_type(true_type):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _normalize_type(self, dtype: str) -> str:
        """Normalize data type strings for comparison."""
        dtype_lower = dtype.lower()
        if 'int' in dtype_lower:
            return 'numeric'
        elif 'float' in dtype_lower:
            return 'numeric'
        elif 'object' in dtype_lower or 'string' in dtype_lower:
            return 'categorical'
        elif 'datetime' in dtype_lower or 'date' in dtype_lower:
            return 'datetime'
        else:
            return dtype_lower
    
    def check_target_suggestion_relevance(
        self,
        suggested_target: str,
        actual_target: str
    ) -> bool:
        """
        Check if suggested target column matches actual target.
        
        Args:
            suggested_target: Column name suggested as target
            actual_target: Actual target column name
            
        Returns:
            True if match, False otherwise
        """
        return suggested_target.lower() == actual_target.lower()
    
    # ========== EDA Agent Metrics ==========
    
    def calculate_imbalance_detection_rate(
        self,
        eda_results: Dict[str, Any],
        datasets_with_imbalance: List[str]
    ) -> float:
        """
        Calculate rate at which EDA agent detects class imbalance.
        
        Args:
            eda_results: Dictionary mapping dataset names to EDA results
            datasets_with_imbalance: List of dataset names that have imbalance
            
        Returns:
            Detection rate between 0 and 1
        """
        if not datasets_with_imbalance:
            return 1.0  # No imbalanced datasets, so perfect detection
        
        detected_count = 0
        
        for dataset_name in datasets_with_imbalance:
            if dataset_name in eda_results:
                result = eda_results[dataset_name]
                # Check if imbalance was flagged
                if self._imbalance_flagged(result):
                    detected_count += 1
        
        return detected_count / len(datasets_with_imbalance) if datasets_with_imbalance else 0.0
    
    def _imbalance_flagged(self, eda_result: Dict) -> bool:
        """Check if imbalance was flagged in EDA results."""
        # Check various possible fields where imbalance might be reported
        checks = [
            'class_imbalance' in eda_result and eda_result['class_imbalance'].get('flagged', False),
            'imbalance_detected' in eda_result and eda_result['imbalance_detected'],
            'warnings' in eda_result and any('imbalance' in str(w).lower() for w in eda_result['warnings']),
            'issues' in eda_result and any('imbalance' in str(i).lower() for i in eda_result['issues'])
        ]
        return any(checks)
    
    def calculate_missing_value_detection_accuracy(
        self,
        detected_missing: Dict[str, Any],
        actual_missing: Dict[str, float]
    ) -> float:
        """
        Calculate accuracy of missing value detection.
        
        Args:
            detected_missing: Dictionary of detected missing value patterns
            actual_missing: Dictionary mapping columns to actual missing percentages
            
        Returns:
            Detection accuracy between 0 and 1
        """
        if not actual_missing:
            return 1.0  # No missing values, perfect detection
        
        detected_cols = set()
        if isinstance(detected_missing, dict):
            detected_cols = set(detected_missing.keys())
        
        actual_cols_with_missing = {col for col, pct in actual_missing.items() if pct > 0}
        
        if not actual_cols_with_missing:
            return 1.0
        
        # Calculate precision and recall
        true_positives = len(detected_cols & actual_cols_with_missing)
        precision = true_positives / len(detected_cols) if detected_cols else 0.0
        recall = true_positives / len(actual_cols_with_missing) if actual_cols_with_missing else 0.0
        
        # F1 score as accuracy metric
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    
    def calculate_outlier_detection_precision(
        self,
        detected_outliers: Dict[str, List[int]],
        actual_outliers: Dict[str, List[int]]
    ) -> float:
        """
        Calculate precision of outlier detection.
        
        Args:
            detected_outliers: Dictionary mapping columns to lists of outlier indices
            actual_outliers: Dictionary mapping columns to actual outlier indices
            
        Returns:
            Precision score between 0 and 1
        """
        if not detected_outliers:
            return 0.0
        
        total_precision = 0.0
        count = 0
        
        for col, detected_indices in detected_outliers.items():
            if col in actual_outliers:
                actual_indices = set(actual_outliers[col])
                detected_set = set(detected_indices)
                
                if detected_set:
                    true_positives = len(detected_set & actual_indices)
                    precision = true_positives / len(detected_set)
                    total_precision += precision
                    count += 1
        
        return total_precision / count if count > 0 else 0.0
    
    def calculate_correlation_detection_completeness(
        self,
        detected_correlations: List[Dict[str, Any]],
        actual_correlations: List[Dict[str, Any]],
        threshold: float = 0.8
    ) -> float:
        """
        Calculate completeness of correlation detection.
        
        Args:
            detected_correlations: List of detected correlation pairs
            actual_correlations: List of actual high correlation pairs
            threshold: Correlation threshold for "high" correlation
            
        Returns:
            Completeness score between 0 and 1
        """
        if not actual_correlations:
            return 1.0
        
        detected_pairs = set()
        for corr in detected_correlations:
            pair = tuple(sorted([corr.get('feature1'), corr.get('feature2')]))
            detected_pairs.add(pair)
        
        actual_pairs = set()
        for corr in actual_correlations:
            if abs(corr.get('correlation', 0)) >= threshold:
                pair = tuple(sorted([corr.get('feature1'), corr.get('feature2')]))
                actual_pairs.add(pair)
        
        if not actual_pairs:
            return 1.0
        
        detected_count = len(detected_pairs & actual_pairs)
        return detected_count / len(actual_pairs)
    
    # ========== Data Cleaning Agent Metrics ==========
    
    def calculate_data_quality_improvement(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate improvement in data quality after cleaning.
        
        Args:
            before_metrics: Quality metrics before cleaning
            after_metrics: Quality metrics after cleaning
            
        Returns:
            Improvement percentage (can be negative if quality decreased)
        """
        # Use missing value percentage as primary quality metric
        before_missing = before_metrics.get('missing_value_pct', 0)
        after_missing = after_metrics.get('missing_value_pct', 0)
        
        if before_missing == 0:
            return 0.0  # No improvement possible
        
        improvement = (before_missing - after_missing) / before_missing
        return improvement
    
    def check_zero_variance_removal(
        self,
        original_features: List[str],
        cleaned_features: List[str],
        original_data: pd.DataFrame
    ) -> bool:
        """
        Check if zero-variance features were removed.
        
        Args:
            original_features: List of original feature names
            cleaned_features: List of cleaned feature names
            original_data: Original DataFrame
            
        Returns:
            True if zero-variance features were removed, False otherwise
        """
        # Find zero-variance features in original data
        zero_variance_cols = []
        for col in original_features:
            if col in original_data.columns:
                if original_data[col].nunique() <= 1:
                    zero_variance_cols.append(col)
        
        if not zero_variance_cols:
            return True  # No zero-variance features, so removal is correct
        
        # Check if they were removed
        removed_cols = set(original_features) - set(cleaned_features)
        return all(col in removed_cols for col in zero_variance_cols)
    
    # ========== Feature Engineering Agent Metrics ==========
    
    def calculate_feature_usefulness(
        self,
        new_features: List[str],
        data: pd.DataFrame,
        target: str
    ) -> float:
        """
        Calculate average correlation of new features with target.
        
        Args:
            new_features: List of newly created feature names
            data: DataFrame with new features and target
            target: Target column name
            
        Returns:
            Average absolute correlation with target
        """
        if not new_features or target not in data.columns:
            return 0.0
        
        correlations = []
        for feature in new_features:
            if feature in data.columns:
                corr = abs(data[feature].corr(data[target]))
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_multicollinearity_reduction(
        self,
        before_vif: Dict[str, float],
        after_vif: Dict[str, float]
    ) -> float:
        """
        Calculate reduction in VIF scores (multicollinearity).
        
        Args:
            before_vif: VIF scores before feature engineering
            after_vif: VIF scores after feature engineering
            
        Returns:
            Reduction percentage
        """
        if not before_vif:
            return 0.0
        
        before_avg = np.mean(list(before_vif.values()))
        after_avg = np.mean([after_vif.get(k, before_vif[k]) for k in before_vif.keys()])
        
        if before_avg == 0:
            return 0.0
        
        reduction = (before_avg - after_avg) / before_avg
        return max(0.0, reduction)  # Don't return negative
    
    # ========== ML Builder Agent Metrics ==========
    
    def calculate_anti_cheating_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_labels: Optional[List[int]] = None
    ) -> float:
        """
        Calculate anti-cheating score based on specificity, sensitivity, and minority recall.
        Detects if model is "cheating" by always predicting majority class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_labels: List of class labels (if None, inferred)
            
        Returns:
            Anti-cheating score between 0 and 1
        """
        if class_labels is None:
            class_labels = sorted(list(set(y_true) | set(y_pred)))
        
        if len(class_labels) != 2:
            # For multiclass, use macro-averaged metrics
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            return (precision + recall) / 2
        
        # Binary classification
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Find minority class
        class_counts = np.bincount(y_true)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)
        
        # Calculate minority class recall
        if minority_class == 0:
            minority_recall = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        else:
            minority_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Anti-cheating score: average of all three
        score = (specificity + sensitivity + minority_recall) / 3
        
        return score
    
    def check_class_balancing_application(
        self,
        ml_builder_results: Dict[str, Any],
        is_imbalanced: bool
    ) -> bool:
        """
        Check if class balancing was applied when needed.
        
        Args:
            ml_builder_results: Results from ML Builder agent
            is_imbalanced: Whether dataset is imbalanced
            
        Returns:
            True if balancing was applied (or not needed), False if needed but not applied
        """
        if not is_imbalanced:
            return True  # Not needed
        
        # Check various indicators of class balancing
        checks = [
            'class_weight' in str(ml_builder_results).lower(),
            'smote' in str(ml_builder_results).lower(),
            'oversampling' in str(ml_builder_results).lower(),
            'undersampling' in str(ml_builder_results).lower(),
            'balanced' in str(ml_builder_results).lower(),
            'resample' in str(ml_builder_results).lower()
        ]
        
        return any(checks)
    
    def calculate_model_diversity(
        self,
        ml_builder_results: Dict[str, Any]
    ) -> int:
        """
        Count number of different algorithms tried.
        
        Args:
            ml_builder_results: Results from ML Builder agent
            
        Returns:
            Number of different algorithms
        """
        # Common algorithm names to look for
        algorithms = [
            'random_forest', 'rf', 'randomforest',
            'logistic_regression', 'lr', 'logistic',
            'svm', 'support_vector',
            'xgboost', 'xgb',
            'gradient_boosting', 'gbm',
            'naive_bayes', 'nb',
            'knn', 'k_nearest',
            'decision_tree', 'dt'
        ]
        
        results_str = str(ml_builder_results).lower()
        found_algorithms = set()
        
        for algo in algorithms:
            if algo in results_str:
                found_algorithms.add(algo)
        
        return len(found_algorithms)
    
    # ========== Model Evaluation Agent Metrics ==========
    
    def calculate_metric_completeness(
        self,
        computed_metrics: List[str],
        required_metrics: List[str] = None
    ) -> float:
        """
        Calculate completeness of metric computation.
        
        Args:
            computed_metrics: List of metrics that were computed
            required_metrics: List of required metrics (if None, uses default)
            
        Returns:
            Completeness score between 0 and 1
        """
        if required_metrics is None:
            required_metrics = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'confusion_matrix', 'classification_report'
            ]
        
        computed_set = set(m.lower() for m in computed_metrics)
        required_set = set(m.lower() for m in required_metrics)
        
        if not required_set:
            return 1.0
        
        return len(computed_set & required_set) / len(required_set)
    
    def check_imbalance_awareness(
        self,
        evaluation_results: Dict[str, Any],
        is_imbalanced: bool
    ) -> bool:
        """
        Check if evaluation flags imbalance issues.
        
        Args:
            evaluation_results: Results from Model Evaluation agent
            is_imbalanced: Whether dataset is imbalanced
            
        Returns:
            True if imbalance is flagged (or not needed), False if needed but not flagged
        """
        if not is_imbalanced:
            return True  # Not needed
        
        results_str = str(evaluation_results).lower()
        checks = [
            'imbalance' in results_str,
            'minority' in results_str,
            'class_imbalance' in results_str,
            'unbalanced' in results_str,
            'recall' in results_str and 'minority' in results_str
        ]
        
        return any(checks)
    
    def check_overfitting_detection(
        self,
        evaluation_results: Dict[str, Any],
        train_score: Optional[float] = None,
        test_score: Optional[float] = None
    ) -> bool:
        """
        Check if overfitting is detected.
        
        Args:
            evaluation_results: Results from Model Evaluation agent
            train_score: Training score (if available)
            test_score: Test score (if available)
            
        Returns:
            True if overfitting is flagged (or not detected), False if detected but not flagged
        """
        # Check if train/test scores are provided
        if train_score is not None and test_score is not None:
            gap = train_score - test_score
            if gap > 0.1:  # Significant gap indicates overfitting
                # Check if it was flagged
                results_str = str(evaluation_results).lower()
                return any([
                    'overfitting' in results_str,
                    'overfit' in results_str,
                    'gap' in results_str,
                    'generalization' in results_str
                ])
        
        # If no scores provided, check if overfitting is mentioned
        results_str = str(evaluation_results).lower()
        return 'overfitting' in results_str or 'overfit' in results_str
    
    # ========== Technical Reporter Agent Metrics ==========
    
    def calculate_report_completeness(
        self,
        report_content: str,
        required_sections: List[str] = None
    ) -> float:
        """
        Calculate completeness of technical report.
        
        Args:
            report_content: Report content as string
            required_sections: List of required section names
            
        Returns:
            Completeness score between 0 and 1
        """
        if required_sections is None:
            required_sections = [
                'introduction', 'methodology', 'data', 'preprocessing',
                'model', 'results', 'evaluation', 'conclusion'
            ]
        
        content_lower = report_content.lower()
        found_sections = sum(1 for section in required_sections if section in content_lower)
        
        return found_sections / len(required_sections) if required_sections else 1.0
    
    def check_visualization_quality(
        self,
        report_results: Dict[str, Any]
    ) -> bool:
        """
        Check if visualizations are present and informative.
        
        Args:
            report_results: Results from Technical Reporter agent
            
        Returns:
            True if visualizations are present
        """
        # Check for plot files or visualization mentions
        checks = [
            'plots' in str(report_results).lower(),
            'visualization' in str(report_results).lower(),
            'chart' in str(report_results).lower(),
            'figure' in str(report_results).lower(),
            'graph' in str(report_results).lower(),
            'plot' in str(report_results).lower()
        ]
        
        return any(checks)
    
    # ========== Project Manager Agent Metrics ==========
    # (These are evaluated via LLM-as-judge, see llm_judge.py)
    
    def calculate_pm_accuracy_average(
        self,
        explanation_scores: List[float]
    ) -> float:
        """
        Calculate average PM accuracy from LLM-as-judge scores.
        
        Args:
            explanation_scores: List of scores (1-5 scale) from LLM-as-judge
            
        Returns:
            Average score normalized to 0-1 range (target ≥0.85)
        """
        if not explanation_scores:
            return 0.0
        
        # Convert 1-5 scale to 0-1 scale
        normalized_scores = [(s - 1) / 4 for s in explanation_scores]
        return np.mean(normalized_scores)

