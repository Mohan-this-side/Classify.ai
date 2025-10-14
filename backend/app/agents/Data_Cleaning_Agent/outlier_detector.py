"""
Outlier Detection Module

This module provides comprehensive outlier detection capabilities including:
- IQR (Interquartile Range) method
- Z-score method
- Modified Z-score method
- Isolation Forest method
- Local Outlier Factor (LOF) method
- Statistical analysis and visualization
- Educational explanations for outlier patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    Comprehensive outlier detection tool for data cleaning workflows
    """
    
    def __init__(self):
        self.logger = logging.getLogger("outlier_detector")
        
        # Default thresholds for different methods
        self.default_thresholds = {
            'iqr_multiplier': 1.5,  # IQR multiplier (1.5 is standard)
            'z_score_threshold': 3.0,  # Z-score threshold (3 is standard)
            'modified_z_score_threshold': 3.5,  # Modified Z-score threshold
            'isolation_forest_contamination': 0.1,  # Expected proportion of outliers
            'lof_contamination': 0.1,  # Expected proportion of outliers
            'percentile_lower': 5,  # Lower percentile for capping
            'percentile_upper': 95  # Upper percentile for capping
        }
        
        # Outlier detection methods
        self.available_methods = [
            'iqr', 'z_score', 'modified_z_score', 
            'isolation_forest', 'lof', 'percentile'
        ]
    
    def detect_outliers(self, df: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       methods: Optional[List[str]] = None,
                       thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive outlier detection
        
        Args:
            df: DataFrame to analyze
            target_column: Optional target column for focused analysis
            methods: List of methods to use (default: all methods)
            thresholds: Custom thresholds for methods
            
        Returns:
            Dictionary containing detection results and analysis
        """
        self.logger.info("Starting comprehensive outlier detection")
        
        # Use default methods if none specified
        if methods is None:
            methods = self.available_methods
        
        # Merge custom thresholds with defaults
        if thresholds is None:
            thresholds = {}
        final_thresholds = {**self.default_thresholds, **thresholds}
        
        # Get numeric columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            self.logger.warning("No numeric columns found for outlier detection")
            return self._empty_results()
        
        # Initialize results structure
        detection_results = {
            "outlier_summary": {},
            "method_results": {},
            "column_analysis": {},
            "visualizations": {},
            "explanations": {},
            "recommendations": [],
            "detection_timestamp": datetime.now().isoformat()
        }
        
        # Analyze each numeric column
        for column in numeric_columns:
            if column == target_column:
                continue  # Skip target column for feature analysis
                
            column_results = self._analyze_column_outliers(
                df[column], column, methods, final_thresholds
            )
            detection_results["column_analysis"][column] = column_results
        
        # Generate overall summary
        detection_results["outlier_summary"] = self._generate_outlier_summary(
            detection_results["column_analysis"]
        )
        
        # Generate method comparison
        detection_results["method_results"] = self._compare_methods(
            detection_results["column_analysis"]
        )
        
        # Generate visualizations
        detection_results["visualizations"] = self._create_outlier_visualizations(
            df, numeric_columns, detection_results["column_analysis"]
        )
        
        # Generate educational explanations
        detection_results["explanations"] = self._generate_outlier_explanations(
            detection_results["outlier_summary"], detection_results["method_results"]
        )
        
        # Generate recommendations
        detection_results["recommendations"] = self._generate_outlier_recommendations(
            detection_results["outlier_summary"], detection_results["method_results"]
        )
        
        self.logger.info(f"Outlier detection completed for {len(numeric_columns)} columns")
        
        return detection_results
    
    def _analyze_column_outliers(self, series: pd.Series, column_name: str, 
                               methods: List[str], thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze outliers for a single column using multiple methods"""
        
        results = {
            "column_name": column_name,
            "total_values": len(series),
            "non_null_values": series.count(),
            "null_count": series.isnull().sum(),
            "methods_used": [],
            "outlier_counts": {},
            "outlier_indices": {},
            "outlier_values": {},
            "statistics": {},
            "method_agreement": {},
            "recommendations": []
        }
        
        # Calculate basic statistics
        results["statistics"] = {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
            "iqr": series.quantile(0.75) - series.quantile(0.25)
        }
        
        # Apply each detection method
        method_outliers = {}
        
        for method in methods:
            try:
                if method == 'iqr':
                    outliers = self._detect_iqr_outliers(series, thresholds['iqr_multiplier'])
                elif method == 'z_score':
                    outliers = self._detect_z_score_outliers(series, thresholds['z_score_threshold'])
                elif method == 'modified_z_score':
                    outliers = self._detect_modified_z_score_outliers(series, thresholds['modified_z_score_threshold'])
                elif method == 'isolation_forest':
                    outliers = self._detect_isolation_forest_outliers(series, thresholds['isolation_forest_contamination'])
                elif method == 'lof':
                    outliers = self._detect_lof_outliers(series, thresholds['lof_contamination'])
                elif method == 'percentile':
                    outliers = self._detect_percentile_outliers(series, thresholds['percentile_lower'], thresholds['percentile_upper'])
                else:
                    continue
                
                method_outliers[method] = outliers
                results["methods_used"].append(method)
                results["outlier_counts"][method] = int(outliers.sum())
                results["outlier_indices"][method] = series[outliers].index.tolist()
                results["outlier_values"][method] = series[outliers].tolist()
                
            except Exception as e:
                self.logger.warning(f"Error applying {method} to column {column_name}: {e}")
                continue
        
        # Calculate method agreement
        if len(method_outliers) > 1:
            results["method_agreement"] = self._calculate_method_agreement(method_outliers)
        
        # Generate column-specific recommendations
        results["recommendations"] = self._generate_column_recommendations(
            series, method_outliers, results["statistics"]
        )
        
        return results
    
    def _detect_iqr_outliers(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_z_score_outliers(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def _detect_modified_z_score_outliers(self, series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Detect outliers using Modified Z-score method (more robust)"""
        median = series.median()
        mad = np.median(np.abs(series - median))  # Median Absolute Deviation
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def _detect_isolation_forest_outliers(self, series: pd.Series, contamination: float = 0.1) -> pd.Series:
        """Detect outliers using Isolation Forest method"""
        # Reshape for sklearn
        X = series.dropna().values.reshape(-1, 1)
        
        if len(X) < 2:
            return pd.Series([False] * len(series), index=series.index)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Create boolean series
        outlier_mask = pd.Series([False] * len(series), index=series.index)
        outlier_mask.loc[series.dropna().index] = outlier_labels == -1
        
        return outlier_mask
    
    def _detect_lof_outliers(self, series: pd.Series, contamination: float = 0.1) -> pd.Series:
        """Detect outliers using Local Outlier Factor method"""
        # Reshape for sklearn
        X = series.dropna().values.reshape(-1, 1)
        
        if len(X) < 3:  # LOF needs at least 3 samples
            return pd.Series([False] * len(series), index=series.index)
        
        # Fit LOF
        lof = LocalOutlierFactor(contamination=contamination)
        outlier_labels = lof.fit_predict(X)
        
        # Create boolean series
        outlier_mask = pd.Series([False] * len(series), index=series.index)
        outlier_mask.loc[series.dropna().index] = outlier_labels == -1
        
        return outlier_mask
    
    def _detect_percentile_outliers(self, series: pd.Series, lower_pct: float = 5, upper_pct: float = 95) -> pd.Series:
        """Detect outliers using percentile method"""
        lower_bound = series.quantile(lower_pct / 100)
        upper_bound = series.quantile(upper_pct / 100)
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _calculate_method_agreement(self, method_outliers: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate agreement between different outlier detection methods"""
        methods = list(method_outliers.keys())
        agreement_matrix = {}
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # Avoid duplicates and self-comparison
                    # Calculate Jaccard similarity
                    set1 = set(method_outliers[method1][method_outliers[method1]].index)
                    set2 = set(method_outliers[method2][method_outliers[method2]].index)
                    
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    
                    jaccard_similarity = intersection / union if union > 0 else 0
                    agreement_matrix[f"{method1}_vs_{method2}"] = jaccard_similarity
        
        # Calculate overall agreement
        similarities = list(agreement_matrix.values())
        overall_agreement = np.mean(similarities) if similarities else 0
        
        return {
            "pairwise_agreement": agreement_matrix,
            "overall_agreement": overall_agreement,
            "high_agreement_pairs": [pair for pair, sim in agreement_matrix.items() if sim > 0.7],
            "low_agreement_pairs": [pair for pair, sim in agreement_matrix.items() if sim < 0.3]
        }
    
    def _generate_outlier_summary(self, column_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate overall summary of outlier detection results"""
        
        total_columns = len(column_analysis)
        columns_with_outliers = 0
        total_outliers = 0
        method_usage = {}
        
        for column, analysis in column_analysis.items():
            if any(count > 0 for count in analysis["outlier_counts"].values()):
                columns_with_outliers += 1
                total_outliers += sum(analysis["outlier_counts"].values())
            
            # Track method usage
            for method in analysis["methods_used"]:
                method_usage[method] = method_usage.get(method, 0) + 1
        
        return {
            "total_columns_analyzed": total_columns,
            "columns_with_outliers": columns_with_outliers,
            "columns_without_outliers": total_columns - columns_with_outliers,
            "total_outliers_detected": total_outliers,
            "outlier_percentage": (total_outliers / (total_columns * 1000)) * 100 if total_columns > 0 else 0,  # Assuming ~1000 rows per column
            "method_usage": method_usage,
            "most_common_method": max(method_usage, key=method_usage.get) if method_usage else None
        }
    
    def _compare_methods(self, column_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare different outlier detection methods"""
        
        method_performance = {}
        
        for method in self.available_methods:
            method_counts = []
            method_agreements = []
            
            for column, analysis in column_analysis.items():
                if method in analysis["outlier_counts"]:
                    method_counts.append(analysis["outlier_counts"][method])
                
                if "method_agreement" in analysis and method in analysis["method_agreement"]:
                    method_agreements.append(analysis["method_agreement"].get("overall_agreement", 0))
            
            method_performance[method] = {
                "average_outliers_detected": np.mean(method_counts) if method_counts else 0,
                "total_columns_used": len(method_counts),
                "average_agreement": np.mean(method_agreements) if method_agreements else 0,
                "consistency_score": 1 - np.std(method_counts) / (np.mean(method_counts) + 1e-8) if method_counts else 0
            }
        
        return method_performance
    
    def _create_outlier_visualizations(self, df: pd.DataFrame, numeric_columns: List[str], 
                                     column_analysis: Dict[str, Dict]) -> Dict[str, str]:
        """Create visualizations for outlier detection results"""
        
        visualizations = {}
        
        try:
            # 1. Box plots for each column
            plt.figure(figsize=(15, 10))
            n_cols = min(4, len(numeric_columns))
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            
            for i, column in enumerate(numeric_columns[:8]):  # Limit to 8 columns
                plt.subplot(n_rows, n_cols, i + 1)
                df[column].plot(kind='box')
                plt.title(f'Outliers in {column}')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            boxplot_path = '/app/results/outlier_boxplots.png'
            plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
            plt.close()
            visualizations['boxplots'] = boxplot_path
            
            # 2. Outlier count comparison
            plt.figure(figsize=(12, 6))
            method_counts = {}
            for column, analysis in column_analysis.items():
                for method, count in analysis["outlier_counts"].items():
                    if method not in method_counts:
                        method_counts[method] = []
                    method_counts[method].append(count)
            
            if method_counts:
                methods = list(method_counts.keys())
                counts = [np.mean(method_counts[method]) for method in methods]
                
                plt.bar(methods, counts)
                plt.title('Average Outliers Detected by Method')
                plt.xlabel('Detection Method')
                plt.ylabel('Average Outlier Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                method_comparison_path = '/app/results/outlier_method_comparison.png'
                plt.savefig(method_comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualizations['method_comparison'] = method_comparison_path
            
            # 3. Outlier distribution heatmap
            if len(numeric_columns) > 1:
                plt.figure(figsize=(10, 8))
                outlier_matrix = []
                for column in numeric_columns:
                    if column in column_analysis:
                        row = []
                        for method in self.available_methods:
                            if method in column_analysis[column]["outlier_counts"]:
                                row.append(column_analysis[column]["outlier_counts"][method])
                            else:
                                row.append(0)
                        outlier_matrix.append(row)
                
                if outlier_matrix:
                    sns.heatmap(outlier_matrix, 
                               xticklabels=self.available_methods,
                               yticklabels=numeric_columns,
                               annot=True, fmt='d', cmap='YlOrRd')
                    plt.title('Outlier Detection Results Heatmap')
                    plt.xlabel('Detection Method')
                    plt.ylabel('Column')
                    plt.tight_layout()
                    
                    heatmap_path = '/app/results/outlier_heatmap.png'
                    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    visualizations['heatmap'] = heatmap_path
            
        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_outlier_explanations(self, summary: Dict[str, Any], method_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate educational explanations for outlier detection results"""
        
        explanations = {}
        
        # Overall outlier assessment
        outlier_percentage = summary["outlier_percentage"]
        if outlier_percentage < 1:
            explanations["overall"] = "✅ Very low outlier rate detected. Your data appears to be clean with minimal outliers."
        elif outlier_percentage < 5:
            explanations["overall"] = "⚠️ Moderate outlier rate detected. Some outliers present but within acceptable range."
        elif outlier_percentage < 15:
            explanations["overall"] = "⚠️ High outlier rate detected. Significant number of outliers that may need attention."
        else:
            explanations["overall"] = "🚨 Very high outlier rate detected. Extensive outliers that likely indicate data quality issues."
        
        # Method effectiveness explanation
        if method_results:
            best_method = max(method_results.keys(), 
                            key=lambda x: method_results[x]["average_agreement"])
            explanations["method_effectiveness"] = f"The {best_method} method appears to be most consistent across your data. Different methods may detect different types of outliers."
        
        # Data quality explanation
        columns_with_outliers = summary["columns_with_outliers"]
        total_columns = summary["total_columns_analyzed"]
        outlier_ratio = columns_with_outliers / total_columns if total_columns > 0 else 0
        
        if outlier_ratio < 0.3:
            explanations["data_quality"] = "Most of your columns appear to be free of outliers, indicating good data quality."
        elif outlier_ratio < 0.7:
            explanations["data_quality"] = "Some columns contain outliers, which is normal for real-world data. Review the specific columns with high outlier counts."
        else:
            explanations["data_quality"] = "Many columns contain outliers. This may indicate systematic data collection issues or the need for data cleaning."
        
        return explanations
    
    def _generate_outlier_recommendations(self, summary: Dict[str, Any], method_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for handling outliers"""
        
        recommendations = []
        
        # Overall recommendations based on outlier rate
        outlier_percentage = summary["outlier_percentage"]
        if outlier_percentage > 10:
            recommendations.append("High outlier rate detected - investigate data collection processes and consider data cleaning")
        elif outlier_percentage > 5:
            recommendations.append("Moderate outlier rate - review outliers for data quality issues before analysis")
        else:
            recommendations.append("Low outlier rate - outliers may be legitimate extreme values, handle carefully")
        
        # Method-specific recommendations
        if method_results:
            consistent_methods = [method for method, perf in method_results.items() 
                                if perf["consistency_score"] > 0.8]
            if consistent_methods:
                recommendations.append(f"Use {', '.join(consistent_methods)} methods for reliable outlier detection")
        
        # General recommendations
        recommendations.extend([
            "Consider the context of your data when deciding whether outliers are errors or legitimate extreme values",
            "For machine learning, consider capping outliers rather than removing them to preserve data",
            "Document your outlier handling strategy for reproducibility",
            "Use multiple detection methods to validate outlier identification"
        ])
        
        return recommendations
    
    def _generate_column_recommendations(self, series: pd.Series, method_outliers: Dict[str, pd.Series], 
                                       statistics: Dict[str, float]) -> List[str]:
        """Generate column-specific recommendations"""
        
        recommendations = []
        
        # Check outlier rate
        total_outliers = sum(outliers.sum() for outliers in method_outliers.values())
        outlier_rate = total_outliers / len(series) if len(series) > 0 else 0
        
        if outlier_rate > 0.1:
            recommendations.append("High outlier rate - investigate data quality and collection process")
        elif outlier_rate > 0.05:
            recommendations.append("Moderate outlier rate - review outliers for potential data entry errors")
        
        # Check data distribution
        if statistics["std"] > statistics["mean"] * 2:
            recommendations.append("High variance detected - consider data transformation or normalization")
        
        # Check for extreme values
        if statistics["max"] > statistics["q3"] + 3 * statistics["iqr"]:
            recommendations.append("Extreme high values detected - consider capping or investigation")
        
        if statistics["min"] < statistics["q1"] - 3 * statistics["iqr"]:
            recommendations.append("Extreme low values detected - consider capping or investigation")
        
        return recommendations
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure when no numeric columns found"""
        return {
            "outlier_summary": {
                "total_columns_analyzed": 0,
                "columns_with_outliers": 0,
                "columns_without_outliers": 0,
                "total_outliers_detected": 0,
                "outlier_percentage": 0,
                "method_usage": {},
                "most_common_method": None
            },
            "method_results": {},
            "column_analysis": {},
            "visualizations": {},
            "explanations": {
                "overall": "No numeric columns found for outlier detection",
                "method_effectiveness": "N/A - no numeric data available",
                "data_quality": "N/A - no numeric data available"
            },
            "recommendations": ["No numeric columns available for outlier detection"],
            "detection_timestamp": datetime.now().isoformat()
        }
