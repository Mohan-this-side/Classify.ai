"""
Missing Value Imputation Module

This module provides comprehensive missing value imputation capabilities including:
- Basic imputation strategies (mean, median, mode, constant)
- Advanced imputation strategies (KNN, iterative, forward/backward fill)
- Smart imputation selection based on data characteristics
- Educational explanations for imputation choices
- Validation and quality assessment of imputed values
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Enable experimental iterative imputer
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_AVAILABLE = True
except ImportError:
    ITERATIVE_AVAILABLE = False
    IterativeImputer = None

class MissingValueImputer:
    """
    Comprehensive missing value imputation tool for data cleaning workflows
    """
    
    def __init__(self):
        self.logger = logging.getLogger("missing_value_imputer")
        
        # Imputation strategies
        self.basic_strategies = [
            'mean', 'median', 'mode', 'constant', 'drop'
        ]
        
        self.advanced_strategies = [
            'knn', 'iterative', 'forward_fill', 'backward_fill', 'interpolate'
        ]
        
        # Default parameters
        self.default_params = {
            'knn_neighbors': 5,
            'iterative_max_iter': 10,
            'interpolation_method': 'linear',
            'constant_value': 0,
            'min_samples_for_knn': 3
        }
    
    def impute_missing_values(self, df: pd.DataFrame, 
                            target_column: Optional[str] = None,
                            strategy: Optional[str] = None,
                            custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive missing value imputation
        
        Args:
            df: DataFrame with missing values
            target_column: Optional target column (usually not imputed)
            strategy: Specific strategy to use (if None, auto-selects best strategy)
            custom_params: Custom parameters for imputation methods
            
        Returns:
            Dictionary containing imputation results and analysis
        """
        self.logger.info("Starting comprehensive missing value imputation")
        
        # Merge custom parameters with defaults
        if custom_params is None:
            custom_params = {}
        params = {**self.default_params, **custom_params}
        
        # Analyze missing data patterns
        missing_analysis = self._analyze_missing_patterns(df, target_column)
        
        # Select best strategy if none specified
        if strategy is None:
            strategy = self._select_best_strategy(df, missing_analysis, target_column)
        
        # Perform imputation
        imputation_results = self._perform_imputation(df, strategy, target_column, params)
        
        # Validate imputation quality
        quality_assessment = self._assess_imputation_quality(
            df, imputation_results['imputed_df'], missing_analysis
        )
        
        # Generate educational explanations
        explanations = self._generate_imputation_explanations(
            strategy, missing_analysis, quality_assessment
        )
        
        # Generate recommendations
        recommendations = self._generate_imputation_recommendations(
            missing_analysis, quality_assessment, strategy
        )
        
        return {
            "original_df": df,
            "imputed_df": imputation_results['imputed_df'],
            "strategy_used": strategy,
            "parameters_used": params,
            "missing_analysis": missing_analysis,
            "imputation_details": imputation_results['details'],
            "quality_assessment": quality_assessment,
            "explanations": explanations,
            "recommendations": recommendations,
            "imputation_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_missing_patterns(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        
        analysis = {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "columns_with_missing": [],
            "missing_by_column": {},
            "missing_patterns": {},
            "data_types": {},
            "suitable_strategies": []
        }
        
        # Analyze each column
        for column in df.columns:
            if column == target_column:
                continue
                
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                analysis["columns_with_missing"].append(column)
                analysis["missing_by_column"][column] = {
                    "count": int(missing_count),
                    "percentage": (missing_count / len(df)) * 100,
                    "data_type": str(df[column].dtype),
                    "has_outliers": self._has_outliers(df[column]),
                    "is_categorical": df[column].dtype == 'object' or df[column].nunique() < 20
                }
                
                analysis["data_types"][column] = str(df[column].dtype)
        
        # Identify missing patterns
        analysis["missing_patterns"] = self._identify_missing_patterns(df)
        
        # Suggest suitable strategies
        analysis["suitable_strategies"] = self._suggest_strategies(analysis)
        
        return analysis
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify common missing data patterns"""
        
        patterns = {
            "MCAR": False,  # Missing Completely At Random
            "MAR": False,   # Missing At Random
            "MNAR": False,  # Missing Not At Random
            "systematic": False,
            "random": False
        }
        
        # Check for systematic patterns (missing in specific rows/columns)
        missing_matrix = df.isnull()
        
        # Check if missing values are concentrated in specific rows
        rows_with_missing = missing_matrix.any(axis=1).sum()
        if rows_with_missing > 0:
            missing_percentage = rows_with_missing / len(df)
            if missing_percentage > 0.1:  # More than 10% of rows have missing values
                patterns["systematic"] = True
        
        # Check for random patterns
        if not patterns["systematic"] and missing_matrix.sum().sum() > 0:
            patterns["random"] = True
        
        # Simple heuristic for MCAR vs MAR vs MNAR
        if patterns["random"]:
            patterns["MCAR"] = True  # Assume MCAR for simplicity
        elif patterns["systematic"]:
            patterns["MAR"] = True
        
        return patterns
    
    def _has_outliers(self, series: pd.Series) -> bool:
        """Check if series has outliers using IQR method"""
        if series.dtype in ['object', 'category']:
            return False
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return False
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return ((series < lower_bound) | (series > upper_bound)).any()
    
    def _suggest_strategies(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest appropriate imputation strategies based on data characteristics"""
        
        strategies = []
        
        if analysis["missing_percentage"] < 5:
            strategies.extend(['mean', 'median', 'mode'])
        elif analysis["missing_percentage"] < 20:
            strategies.extend(['mean', 'median', 'knn'])
        else:
            strategies.extend(['knn', 'iterative'])
        
        # Add categorical strategies
        categorical_cols = [col for col, info in analysis["missing_by_column"].items() 
                          if info["is_categorical"]]
        if categorical_cols:
            strategies.extend(['mode', 'constant'])
        
        # Add time series strategies
        if any('datetime' in str(dtype) for dtype in analysis["data_types"].values()):
            strategies.extend(['forward_fill', 'backward_fill', 'interpolate'])
        
        return list(set(strategies))  # Remove duplicates
    
    def _select_best_strategy(self, df: pd.DataFrame, analysis: Dict[str, Any], 
                            target_column: Optional[str] = None) -> str:
        """Automatically select the best imputation strategy"""
        
        missing_percentage = analysis["missing_percentage"]
        suitable_strategies = analysis["suitable_strategies"]
        
        # Decision logic based on missing data characteristics
        if missing_percentage < 1:
            return 'mean'  # Very low missing data
        elif missing_percentage < 5:
            if 'median' in suitable_strategies:
                return 'median'  # Robust to outliers
            else:
                return 'mean'
        elif missing_percentage < 15:
            if 'knn' in suitable_strategies:
                return 'knn'  # More sophisticated
            else:
                return 'median'
        else:
            if 'iterative' in suitable_strategies:
                return 'iterative'  # Best for high missing data
            elif 'knn' in suitable_strategies:
                return 'knn'
            else:
                return 'median'
    
    def _perform_imputation(self, df: pd.DataFrame, strategy: str, 
                          target_column: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual imputation using the selected strategy"""
        
        self.logger.info(f"Performing imputation using strategy: {strategy}")
        
        imputed_df = df.copy()
        details = {
            "strategy": strategy,
            "columns_imputed": [],
            "values_imputed": {},
            "imputation_stats": {}
        }
        
        # Get columns to impute (exclude target column)
        columns_to_impute = [col for col in df.columns if col != target_column and df[col].isnull().any()]
        
        for column in columns_to_impute:
            original_missing = df[column].isnull().sum()
            
            if strategy == 'drop':
                # Drop rows with missing values
                imputed_df = imputed_df.dropna(subset=[column])
                details["columns_imputed"].append(column)
                details["values_imputed"][column] = f"Dropped {original_missing} rows"
                
            elif strategy == 'mean':
                if df[column].dtype in ['int64', 'float64']:
                    mean_value = df[column].mean()
                    imputed_df[column] = df[column].fillna(mean_value)
                    details["columns_imputed"].append(column)
                    details["values_imputed"][column] = f"Filled with mean: {mean_value:.4f}"
                
            elif strategy == 'median':
                if df[column].dtype in ['int64', 'float64']:
                    median_value = df[column].median()
                    imputed_df[column] = df[column].fillna(median_value)
                    details["columns_imputed"].append(column)
                    details["values_imputed"][column] = f"Filled with median: {median_value:.4f}"
                
            elif strategy == 'mode':
                mode_value = df[column].mode()
                if len(mode_value) > 0:
                    imputed_df[column] = df[column].fillna(mode_value[0])
                    details["columns_imputed"].append(column)
                    details["values_imputed"][column] = f"Filled with mode: {mode_value[0]}"
                else:
                    # If no mode, use most frequent value
                    most_frequent = df[column].value_counts().index[0]
                    imputed_df[column] = df[column].fillna(most_frequent)
                    details["columns_imputed"].append(column)
                    details["values_imputed"][column] = f"Filled with most frequent: {most_frequent}"
                
            elif strategy == 'constant':
                constant_value = params['constant_value']
                imputed_df[column] = df[column].fillna(constant_value)
                details["columns_imputed"].append(column)
                details["values_imputed"][column] = f"Filled with constant: {constant_value}"
                
            elif strategy == 'forward_fill':
                imputed_df[column] = df[column].fillna(method='ffill')
                details["columns_imputed"].append(column)
                details["values_imputed"][column] = "Forward filled"
                
            elif strategy == 'backward_fill':
                imputed_df[column] = df[column].fillna(method='bfill')
                details["columns_imputed"].append(column)
                details["values_imputed"][column] = "Backward filled"
                
            elif strategy == 'interpolate':
                if df[column].dtype in ['int64', 'float64']:
                    imputed_df[column] = df[column].interpolate(method=params['interpolation_method'])
                    details["columns_imputed"].append(column)
                    details["values_imputed"][column] = f"Interpolated using {params['interpolation_method']}"
                
            elif strategy == 'knn':
                if len(df) >= params['min_samples_for_knn']:
                    try:
                        # Prepare data for KNN imputation
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if target_column and target_column in numeric_cols:
                            numeric_cols.remove(target_column)
                        
                        if len(numeric_cols) > 0:
                            # Use only numeric columns for KNN
                            knn_data = df[numeric_cols].copy()
                            
                            # Scale the data
                            scaler = StandardScaler()
                            knn_data_scaled = scaler.fit_transform(knn_data)
                            
                            # Apply KNN imputation
                            knn_imputer = KNNImputer(n_neighbors=params['knn_neighbors'])
                            knn_data_imputed = knn_imputer.fit_transform(knn_data_scaled)
                            
                            # Transform back to original scale
                            knn_data_imputed = scaler.inverse_transform(knn_data_imputed)
                            
                            # Update the dataframe
                            for i, col in enumerate(numeric_cols):
                                imputed_df[col] = knn_data_imputed[:, i]
                                if col in columns_to_impute:
                                    details["columns_imputed"].append(col)
                                    details["values_imputed"][col] = f"KNN imputed (k={params['knn_neighbors']})"
                    except Exception as e:
                        self.logger.warning(f"KNN imputation failed for {column}: {e}")
                        # Fallback to median
                        if df[column].dtype in ['int64', 'float64']:
                            median_value = df[column].median()
                            imputed_df[column] = df[column].fillna(median_value)
                            details["columns_imputed"].append(column)
                            details["values_imputed"][column] = f"KNN failed, used median: {median_value:.4f}"
                
            elif strategy == 'iterative':
                if not ITERATIVE_AVAILABLE:
                    self.logger.warning("IterativeImputer not available, falling back to KNN imputation")
                    # Fallback to KNN if iterative is not available
                    try:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if target_column and target_column in numeric_cols:
                            numeric_cols.remove(target_column)
                        
                        if len(numeric_cols) > 0 and len(df) >= params['min_samples_for_knn']:
                            # Use KNN as fallback
                            knn_data = df[numeric_cols].copy()
                            scaler = StandardScaler()
                            knn_data_scaled = scaler.fit_transform(knn_data)
                            
                            knn_imputer = KNNImputer(n_neighbors=params['knn_neighbors'])
                            knn_data_imputed = knn_imputer.fit_transform(knn_data_scaled)
                            knn_data_imputed = scaler.inverse_transform(knn_data_imputed)
                            
                            for i, col in enumerate(numeric_cols):
                                imputed_df[col] = knn_data_imputed[:, i]
                                if col in columns_to_impute:
                                    details["columns_imputed"].append(col)
                                    details["values_imputed"][col] = f"KNN imputed (fallback from iterative, k={params['knn_neighbors']})"
                    except Exception as e:
                        self.logger.warning(f"KNN fallback also failed: {e}")
                        # Final fallback to median
                        for column in columns_to_impute:
                            if df[column].dtype in ['int64', 'float64']:
                                median_value = df[column].median()
                                imputed_df[column] = df[column].fillna(median_value)
                                details["columns_imputed"].append(column)
                                details["values_imputed"][column] = f"Median imputed (iterative and KNN failed): {median_value:.4f}"
                else:
                    try:
                        # Prepare data for iterative imputation
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if target_column and target_column in numeric_cols:
                            numeric_cols.remove(target_column)
                        
                        if len(numeric_cols) > 0:
                            # Use only numeric columns for iterative imputation
                            iter_data = df[numeric_cols].copy()
                            
                            # Apply iterative imputation
                            iter_imputer = IterativeImputer(max_iter=params['iterative_max_iter'], random_state=42)
                            iter_data_imputed = iter_imputer.fit_transform(iter_data)
                            
                            # Update the dataframe
                            for i, col in enumerate(numeric_cols):
                                imputed_df[col] = iter_data_imputed[:, i]
                                if col in columns_to_impute:
                                    details["columns_imputed"].append(col)
                                    details["values_imputed"][col] = f"Iterative imputed (max_iter={params['iterative_max_iter']})"
                    except Exception as e:
                        self.logger.warning(f"Iterative imputation failed: {e}")
                        # Fallback to median
                        for column in columns_to_impute:
                            if df[column].dtype in ['int64', 'float64']:
                                median_value = df[column].median()
                                imputed_df[column] = df[column].fillna(median_value)
                                details["columns_imputed"].append(column)
                                details["values_imputed"][column] = f"Iterative failed, used median: {median_value:.4f}"
        
        # Calculate imputation statistics
        details["imputation_stats"] = {
            "total_values_imputed": sum(len(details["values_imputed"]) for _ in details["values_imputed"]),
            "columns_processed": len(details["columns_imputed"]),
            "remaining_missing": imputed_df.isnull().sum().sum()
        }
        
        return {
            "imputed_df": imputed_df,
            "details": details
        }
    
    def _assess_imputation_quality(self, original_df: pd.DataFrame, imputed_df: pd.DataFrame, 
                                 missing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of imputation"""
        
        assessment = {
            "missing_values_remaining": int(imputed_df.isnull().sum().sum()),
            "missing_values_imputed": int(original_df.isnull().sum().sum() - imputed_df.isnull().sum().sum()),
            "imputation_success_rate": 0,
            "data_integrity_preserved": True,
            "statistical_changes": {},
            "quality_score": 0
        }
        
        # Calculate success rate
        total_original_missing = original_df.isnull().sum().sum()
        if total_original_missing > 0:
            assessment["imputation_success_rate"] = (
                assessment["missing_values_imputed"] / total_original_missing
            ) * 100
        
        # Check data integrity
        if imputed_df.shape[0] != original_df.shape[0]:
            assessment["data_integrity_preserved"] = False
        
        # Analyze statistical changes for numeric columns
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in missing_analysis["missing_by_column"]:
                original_stats = {
                    "mean": original_df[col].mean(),
                    "std": original_df[col].std(),
                    "min": original_df[col].min(),
                    "max": original_df[col].max()
                }
                
                imputed_stats = {
                    "mean": imputed_df[col].mean(),
                    "std": imputed_df[col].std(),
                    "min": imputed_df[col].min(),
                    "max": imputed_df[col].max()
                }
                
                # Calculate percentage changes
                changes = {}
                for stat in ["mean", "std", "min", "max"]:
                    if original_stats[stat] != 0:
                        changes[stat] = ((imputed_stats[stat] - original_stats[stat]) / abs(original_stats[stat])) * 100
                    else:
                        changes[stat] = 0
                
                assessment["statistical_changes"][col] = changes
        
        # Calculate overall quality score
        success_rate = assessment["imputation_success_rate"]
        integrity_score = 100 if assessment["data_integrity_preserved"] else 0
        
        # Penalize large statistical changes
        avg_change = 0
        if assessment["statistical_changes"]:
            all_changes = []
            for col_changes in assessment["statistical_changes"].values():
                all_changes.extend([abs(change) for change in col_changes.values()])
            avg_change = np.mean(all_changes) if all_changes else 0
        
        change_penalty = max(0, 100 - avg_change)
        
        assessment["quality_score"] = (success_rate * 0.4 + integrity_score * 0.3 + change_penalty * 0.3)
        
        return assessment
    
    def _generate_imputation_explanations(self, strategy: str, missing_analysis: Dict[str, Any], 
                                        quality_assessment: Dict[str, Any]) -> Dict[str, str]:
        """Generate educational explanations for imputation"""
        
        explanations = {}
        
        # Strategy explanation
        strategy_explanations = {
            'mean': "Mean imputation replaces missing values with the average of available values. Good for normally distributed data but can be affected by outliers.",
            'median': "Median imputation replaces missing values with the middle value. More robust to outliers than mean imputation.",
            'mode': "Mode imputation replaces missing values with the most frequent value. Best for categorical data or when the most common value is most representative.",
            'constant': "Constant imputation replaces missing values with a fixed value. Simple but may not preserve data relationships.",
            'knn': "K-Nearest Neighbors imputation uses similar rows to estimate missing values. More sophisticated and preserves data patterns.",
            'iterative': "Iterative imputation uses other variables to predict missing values through multiple regression models. Most sophisticated approach.",
            'forward_fill': "Forward fill carries the last known value forward. Good for time series data with temporal patterns.",
            'backward_fill': "Backward fill uses the next known value to fill backwards. Alternative to forward fill for time series.",
            'interpolate': "Interpolation estimates missing values based on surrounding values. Good for ordered data with smooth patterns.",
            'drop': "Dropping rows with missing values. Simple but can result in significant data loss."
        }
        
        explanations["strategy_rationale"] = strategy_explanations.get(strategy, "Custom imputation strategy used.")
        
        # Missing data pattern explanation
        patterns = missing_analysis["missing_patterns"]
        if patterns["MCAR"]:
            explanations["missing_pattern"] = "Missing data appears to be Missing Completely At Random (MCAR), meaning the missingness is not related to any observed or unobserved variables. This is the best case for imputation."
        elif patterns["MAR"]:
            explanations["missing_pattern"] = "Missing data appears to be Missing At Random (MAR), meaning the missingness is related to observed variables but not the missing values themselves. Imputation can work well with this pattern."
        elif patterns["MNAR"]:
            explanations["missing_pattern"] = "Missing data appears to be Missing Not At Random (MNAR), meaning the missingness is related to the missing values themselves. This is the most challenging case for imputation."
        else:
            explanations["missing_pattern"] = "Missing data pattern is unclear. Consider investigating the causes of missingness before imputation."
        
        # Quality assessment explanation
        success_rate = quality_assessment["imputation_success_rate"]
        if success_rate >= 95:
            explanations["quality"] = "Excellent imputation quality with very high success rate. The imputed values should be reliable for analysis."
        elif success_rate >= 80:
            explanations["quality"] = "Good imputation quality with high success rate. The imputed values should be reasonably reliable."
        elif success_rate >= 60:
            explanations["quality"] = "Moderate imputation quality. Consider reviewing the imputation strategy or investigating data quality issues."
        else:
            explanations["quality"] = "Poor imputation quality. The imputation strategy may not be appropriate for this data. Consider alternative approaches."
        
        return explanations
    
    def _generate_imputation_recommendations(self, missing_analysis: Dict[str, Any], 
                                           quality_assessment: Dict[str, Any], 
                                           strategy: str) -> List[str]:
        """Generate recommendations for imputation handling"""
        
        recommendations = []
        
        # Based on missing data percentage
        missing_pct = missing_analysis["missing_percentage"]
        if missing_pct > 50:
            recommendations.append("Very high missing data percentage (>50%). Consider if the dataset is suitable for analysis or if additional data collection is needed.")
        elif missing_pct > 20:
            recommendations.append("High missing data percentage (>20%). Document the imputation strategy carefully and consider sensitivity analysis.")
        elif missing_pct > 5:
            recommendations.append("Moderate missing data percentage (>5%). Monitor the impact of imputation on your analysis results.")
        else:
            recommendations.append("Low missing data percentage (<5%). Imputation should have minimal impact on analysis results.")
        
        # Based on quality assessment
        quality_score = quality_assessment["quality_score"]
        if quality_score < 70:
            recommendations.append("Low imputation quality score. Consider trying alternative imputation strategies or investigating data quality issues.")
        
        # Based on strategy used
        if strategy in ['mean', 'median'] and missing_analysis["missing_patterns"]["systematic"]:
            recommendations.append("Using simple imputation with systematic missing patterns. Consider more sophisticated methods like KNN or iterative imputation.")
        
        if strategy == 'drop' and missing_analysis["missing_percentage"] > 10:
            recommendations.append("Dropping rows resulted in significant data loss. Consider imputation instead of dropping to preserve more data.")
        
        # General recommendations
        recommendations.extend([
            "Always document your imputation strategy and assumptions for reproducibility",
            "Consider performing sensitivity analysis to test how imputation affects your results",
            "Validate imputed values against domain knowledge when possible",
            "For machine learning, consider whether imputed values should be flagged or treated differently"
        ])
        
        return recommendations
