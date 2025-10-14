"""
Missing Value Analysis Module

This module provides comprehensive missing value analysis capabilities including:
- Missing value pattern detection
- Missing data visualization
- Statistical analysis of missingness
- Educational explanations for missing data patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MissingValueAnalyzer:
    """
    Comprehensive missing value analysis tool for data cleaning workflows
    """
    
    def __init__(self):
        self.logger = logging.getLogger("missing_value_analyzer")
        
    def analyze_missing_values(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive missing value analysis
        
        Args:
            df: DataFrame to analyze
            target_column: Optional target column for focused analysis
            
        Returns:
            Dictionary containing analysis results and visualizations
        """
        self.logger.info("Starting comprehensive missing value analysis")
        
        # Basic missing value statistics
        missing_stats = self._calculate_missing_statistics(df)
        
        # Missing value patterns
        patterns = self._analyze_missing_patterns(df)
        
        # Missing value correlations
        correlations = self._analyze_missing_correlations(df)
        
        # Target-specific analysis if target column provided
        target_analysis = {}
        if target_column and target_column in df.columns:
            target_analysis = self._analyze_target_missingness(df, target_column)
        
        # Generate visualizations
        visualizations = self._create_missing_visualizations(df, target_column)
        
        # Generate educational explanations
        explanations = self._generate_missing_explanations(missing_stats, patterns, correlations)
        
        # Compile comprehensive report
        analysis_results = {
            "missing_statistics": missing_stats,
            "missing_patterns": patterns,
            "missing_correlations": correlations,
            "target_analysis": target_analysis,
            "visualizations": visualizations,
            "explanations": explanations,
            "recommendations": self._generate_missing_recommendations(missing_stats, patterns),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Missing value analysis completed. Found {missing_stats['total_missing']} missing values")
        
        return analysis_results
    
    def _calculate_missing_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive missing value statistics"""
        
        # Basic counts
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Overall statistics
        total_missing = missing_counts.sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_missing_percentage = (total_missing / total_cells) * 100
        
        # Column-wise analysis
        columns_with_missing = missing_counts[missing_counts > 0]
        columns_without_missing = missing_counts[missing_counts == 0]
        
        # Missing value severity classification
        severity_classification = {}
        for col in df.columns:
            missing_pct = missing_percentages[col]
            if missing_pct == 0:
                severity_classification[col] = "No missing values"
            elif missing_pct < 5:
                severity_classification[col] = "Low (< 5%)"
            elif missing_pct < 20:
                severity_classification[col] = "Moderate (5-20%)"
            elif missing_pct < 50:
                severity_classification[col] = "High (20-50%)"
            else:
                severity_classification[col] = "Very High (> 50%)"
        
        # Row-wise analysis
        rows_with_missing = df.isnull().any(axis=1).sum()
        rows_without_missing = len(df) - rows_with_missing
        complete_rows_percentage = (rows_without_missing / len(df)) * 100
        
        # Missing value distribution
        missing_per_row = df.isnull().sum(axis=1)
        missing_distribution = {
            "mean_missing_per_row": missing_per_row.mean(),
            "max_missing_per_row": missing_per_row.max(),
            "rows_with_all_missing": (missing_per_row == df.shape[1]).sum(),
            "rows_with_some_missing": rows_with_missing,
            "rows_with_no_missing": rows_without_missing
        }
        
        return {
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "overall_missing_percentage": round(overall_missing_percentage, 2),
            "columns_with_missing": len(columns_with_missing),
            "columns_without_missing": len(columns_without_missing),
            "missing_counts": missing_counts.to_dict(),
            "missing_percentages": missing_percentages.to_dict(),
            "severity_classification": severity_classification,
            "complete_rows_percentage": round(complete_rows_percentage, 2),
            "missing_distribution": missing_distribution
        }
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        
        # MCAR (Missing Completely At Random) analysis
        mcar_analysis = self._analyze_mcar_pattern(df)
        
        # MAR (Missing At Random) analysis
        mar_analysis = self._analyze_mar_pattern(df)
        
        # MNAR (Missing Not At Random) analysis
        mnar_analysis = self._analyze_mnar_pattern(df)
        
        # Missing data patterns by column groups
        pattern_groups = self._analyze_missing_groups(df)
        
        return {
            "mcar_analysis": mcar_analysis,
            "mar_analysis": mar_analysis,
            "mnar_analysis": mnar_analysis,
            "pattern_groups": pattern_groups
        }
    
    def _analyze_mcar_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if missing data is Missing Completely At Random"""
        
        # For MCAR, missing values should be randomly distributed
        # We can test this by checking if missing patterns are independent
        
        missing_matrix = df.isnull()
        
        # Calculate missing value correlations
        missing_correlations = missing_matrix.corr()
        
        # MCAR indicators
        high_correlations = (missing_correlations.abs() > 0.3).sum().sum() - len(df.columns)
        independence_score = 1 - (high_correlations / (len(df.columns) * (len(df.columns) - 1)))
        
        return {
            "independence_score": round(independence_score, 3),
            "high_correlations_count": int(high_correlations),
            "likely_mcar": independence_score > 0.7,
            "missing_correlations": missing_correlations.to_dict()
        }
    
    def _analyze_mar_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if missing data is Missing At Random"""
        
        # MAR means missingness depends on observed data
        # We can test this by checking correlations between missing patterns and observed values
        
        missing_matrix = df.isnull()
        mar_indicators = {}
        
        for col in df.columns:
            if missing_matrix[col].sum() > 0:
                # Check if missingness in this column correlates with other observed columns
                other_cols = [c for c in df.columns if c != col and df[c].dtype in ['int64', 'float64']]
                
                if other_cols:
                    correlations = []
                    for other_col in other_cols:
                        # Create binary indicator for missingness
                        missing_indicator = missing_matrix[col].astype(int)
                        # Calculate correlation with observed values
                        corr = df[other_col].corr(missing_indicator)
                        if not pd.isna(corr):
                            correlations.append(abs(corr))
                    
                    mar_indicators[col] = {
                        "max_correlation": max(correlations) if correlations else 0,
                        "mean_correlation": np.mean(correlations) if correlations else 0,
                        "likely_mar": max(correlations) > 0.3 if correlations else False
                    }
        
        return mar_indicators
    
    def _analyze_mnar_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if missing data is Missing Not At Random"""
        
        # MNAR means missingness depends on the missing values themselves
        # This is harder to detect, but we can look for patterns
        
        mnar_indicators = {}
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                # Check if missing values are clustered in certain ranges
                if df[col].dtype in ['int64', 'float64']:
                    # Analyze distribution of missing values
                    non_missing = df[col].dropna()
                    if len(non_missing) > 0:
                        # Check if missing values are more likely in certain ranges
                        q25, q75 = non_missing.quantile([0.25, 0.75])
                        missing_in_lower = df[col].isnull() & (df[col] < q25).fillna(False)
                        missing_in_upper = df[col].isnull() & (df[col] > q75).fillna(False)
                        
                        mnar_indicators[col] = {
                            "missing_in_lower_quartile": missing_in_lower.sum(),
                            "missing_in_upper_quartile": missing_in_upper.sum(),
                            "likely_mnar": abs(missing_in_lower.sum() - missing_in_upper.sum()) > len(df) * 0.1
                        }
        
        return mnar_indicators
    
    def _analyze_missing_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns in column groups"""
        
        # Group columns by data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        group_analysis = {}
        
        for group_name, cols in [("numeric", numeric_cols), ("categorical", categorical_cols), ("datetime", datetime_cols)]:
            if cols:
                group_missing = df[cols].isnull().sum().sum()
                group_total = len(df) * len(cols)
                group_percentage = (group_missing / group_total) * 100
                
                group_analysis[group_name] = {
                    "columns": cols,
                    "missing_count": int(group_missing),
                    "missing_percentage": round(group_percentage, 2),
                    "columns_with_missing": len([c for c in cols if df[c].isnull().sum() > 0])
                }
        
        return group_analysis
    
    def _analyze_missing_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between missing value patterns"""
        
        missing_matrix = df.isnull().astype(int)
        missing_correlations = missing_matrix.corr()
        
        # Find strong correlations between missing patterns
        strong_correlations = []
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # Avoid duplicates
                    corr = missing_correlations.loc[col1, col2]
                    if abs(corr) > 0.3:
                        strong_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(corr, 3),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate"
                        })
        
        return {
            "correlation_matrix": missing_correlations.to_dict(),
            "strong_correlations": strong_correlations,
            "max_correlation": round(missing_correlations.abs().max().max(), 3)
        }
    
    def _analyze_target_missingness(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze missing values specifically related to the target column"""
        
        if target_column not in df.columns:
            return {}
        
        target_missing = df[target_column].isnull().sum()
        target_missing_pct = (target_missing / len(df)) * 100
        
        # Analyze how missing values in other columns relate to target missingness
        target_missing_indicator = df[target_column].isnull()
        
        feature_missing_correlations = {}
        for col in df.columns:
            if col != target_column and df[col].isnull().sum() > 0:
                col_missing_indicator = df[col].isnull()
                corr = target_missing_indicator.corr(col_missing_indicator)
                if not pd.isna(corr):
                    feature_missing_correlations[col] = round(corr, 3)
        
        return {
            "target_missing_count": int(target_missing),
            "target_missing_percentage": round(target_missing_pct, 2),
            "feature_missing_correlations": feature_missing_correlations,
            "high_correlation_features": [col for col, corr in feature_missing_correlations.items() if abs(corr) > 0.3]
        }
    
    def _create_missing_visualizations(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, str]:
        """Create visualizations for missing data analysis"""
        
        visualizations = {}
        
        try:
            # 1. Missing value heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Value Heatmap')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = '/app/results/missing_value_heatmap.png'
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            visualizations['heatmap'] = heatmap_path
            
            # 2. Missing value bar chart
            plt.figure(figsize=(10, 6))
            missing_counts = df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
            
            if len(missing_counts) > 0:
                missing_counts.plot(kind='barh')
                plt.title('Missing Values by Column')
                plt.xlabel('Number of Missing Values')
                plt.tight_layout()
                
                bar_path = '/app/results/missing_value_bars.png'
                plt.savefig(bar_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualizations['bar_chart'] = bar_path
            
            # 3. Missing value percentage chart
            plt.figure(figsize=(10, 6))
            missing_pct = (df.isnull().sum() / len(df)) * 100
            missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
            
            if len(missing_pct) > 0:
                missing_pct.plot(kind='barh')
                plt.title('Missing Value Percentages by Column')
                plt.xlabel('Percentage of Missing Values (%)')
                plt.tight_layout()
                
                pct_path = '/app/results/missing_value_percentages.png'
                plt.savefig(pct_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualizations['percentage_chart'] = pct_path
            
            # 4. Missing value correlation heatmap
            plt.figure(figsize=(10, 8))
            missing_corr = df.isnull().corr()
            sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Missing Value Correlation Matrix')
            plt.tight_layout()
            
            corr_path = '/app/results/missing_value_correlations.png'
            plt.savefig(corr_path, dpi=150, bbox_inches='tight')
            plt.close()
            visualizations['correlation_heatmap'] = corr_path
            
        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_missing_explanations(self, stats: Dict[str, Any], patterns: Dict[str, Any], 
                                     correlations: Dict[str, Any]) -> Dict[str, str]:
        """Generate educational explanations for missing data patterns"""
        
        explanations = {}
        
        # Overall missing data explanation
        total_missing = stats['total_missing']
        overall_pct = stats['overall_missing_percentage']
        
        if total_missing == 0:
            explanations['overall'] = "✅ No missing values found in the dataset. This is excellent for data quality!"
        elif overall_pct < 5:
            explanations['overall'] = f"✅ Low missing data rate ({overall_pct:.1f}%). This is generally acceptable and should not significantly impact analysis."
        elif overall_pct < 20:
            explanations['overall'] = f"⚠️ Moderate missing data rate ({overall_pct:.1f}%). Consider investigating patterns and using appropriate imputation strategies."
        else:
            explanations['overall'] = f"🚨 High missing data rate ({overall_pct:.1f}%). This may significantly impact analysis quality. Consider data collection improvements or advanced imputation methods."
        
        # Pattern explanations
        mcar_score = patterns['mcar_analysis']['independence_score']
        if mcar_score > 0.7:
            explanations['pattern'] = "The missing data appears to be Missing Completely At Random (MCAR), meaning missingness is independent of both observed and unobserved data. This is the best-case scenario for imputation."
        else:
            explanations['pattern'] = "The missing data shows patterns that suggest it may not be completely random. This could indicate Missing At Random (MAR) or Missing Not At Random (MNAR) patterns, which require careful consideration for imputation strategies."
        
        # Correlation explanations
        strong_corrs = len(correlations['strong_correlations'])
        if strong_corrs == 0:
            explanations['correlation'] = "No strong correlations found between missing value patterns across columns. This suggests independent missingness."
        else:
            explanations['correlation'] = f"Found {strong_corrs} strong correlations between missing value patterns. This suggests that missingness in some columns is related to missingness in others, which should be considered in imputation strategies."
        
        # Severity explanations
        high_severity_cols = [col for col, severity in stats['severity_classification'].items() 
                            if severity in ['High (20-50%)', 'Very High (> 50%)']]
        
        if high_severity_cols:
            explanations['severity'] = f"Columns with high missing data rates: {', '.join(high_severity_cols)}. These columns may need special attention or consideration for removal."
        else:
            explanations['severity'] = "No columns have extremely high missing data rates. All columns appear to have manageable missing value levels."
        
        return explanations
    
    def _generate_missing_recommendations(self, stats: Dict[str, Any], patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations for handling missing data"""
        
        recommendations = []
        
        # Overall recommendations
        overall_pct = stats['overall_missing_percentage']
        if overall_pct < 5:
            recommendations.append("Low missing data rate - simple imputation methods (mean, median, mode) should be sufficient.")
        elif overall_pct < 20:
            recommendations.append("Moderate missing data rate - consider advanced imputation methods like KNN or iterative imputation.")
        else:
            recommendations.append("High missing data rate - consider multiple imputation or domain-specific imputation strategies.")
        
        # Pattern-based recommendations
        mcar_score = patterns['mcar_analysis']['independence_score']
        if mcar_score > 0.7:
            recommendations.append("MCAR pattern detected - standard imputation methods should work well.")
        else:
            recommendations.append("Non-random missing patterns detected - consider MAR/MNAR-specific imputation strategies.")
        
        # Column-specific recommendations
        for col, severity in stats['severity_classification'].items():
            if severity == "Very High (> 50%)":
                recommendations.append(f"Consider removing column '{col}' due to very high missing data rate (>50%).")
            elif severity == "High (20-50%)":
                recommendations.append(f"Column '{col}' has high missing data rate - use advanced imputation or consider if column is necessary.")
        
        # Complete rows recommendation
        complete_pct = stats['complete_rows_percentage']
        if complete_pct < 50:
            recommendations.append("Less than 50% of rows are complete - consider listwise deletion carefully as it may significantly reduce sample size.")
        
        return recommendations
