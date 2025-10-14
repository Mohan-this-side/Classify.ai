"""
Educational Explanation Generator

This module provides comprehensive educational explanations for data cleaning processes including:
- Step-by-step explanations for each cleaning operation
- Rationale behind cleaning decisions
- Impact assessment of cleaning actions
- Best practices and recommendations
- Visual summaries and markdown reports
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
import json

class EducationalExplainer:
    """
    Comprehensive educational explanation generator for data cleaning workflows
    """
    
    def __init__(self):
        self.logger = logging.getLogger("educational_explainer")
        
        # Explanation templates and content
        self.explanation_templates = {
            "missing_value_analysis": {
                "title": "Missing Value Analysis",
                "description": "Understanding patterns and extent of missing data in your dataset",
                "key_concepts": [
                    "Missing Completely At Random (MCAR)",
                    "Missing At Random (MAR)", 
                    "Missing Not At Random (MNAR)",
                    "Missing data impact on analysis"
                ]
            },
            "missing_value_imputation": {
                "title": "Missing Value Imputation",
                "description": "Strategies for filling missing values while preserving data integrity",
                "key_concepts": [
                    "Mean/Median/Mode imputation",
                    "K-Nearest Neighbors (KNN) imputation",
                    "Iterative imputation",
                    "Imputation quality assessment"
                ]
            },
            "data_type_validation": {
                "title": "Data Type Validation",
                "description": "Ensuring data types are correct and consistent across your dataset",
                "key_concepts": [
                    "Data type detection",
                    "Type conversion strategies",
                    "Data quality assessment",
                    "Consistency validation"
                ]
            },
            "outlier_detection": {
                "title": "Outlier Detection",
                "description": "Identifying and understanding unusual values in your dataset",
                "key_concepts": [
                    "IQR (Interquartile Range) method",
                    "Z-score method",
                    "Isolation Forest",
                    "Outlier impact on analysis"
                ]
            },
            "data_cleaning_summary": {
                "title": "Data Cleaning Summary",
                "description": "Overview of all cleaning operations performed on your dataset",
                "key_concepts": [
                    "Data quality improvement",
                    "Cleaning impact assessment",
                    "Best practices applied",
                    "Next steps recommendations"
                ]
            }
        }
    
    def generate_comprehensive_explanations(self, cleaning_results: Dict[str, Any], 
                                         original_df: pd.DataFrame,
                                         cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive educational explanations for the entire data cleaning process
        
        Args:
            cleaning_results: Results from all cleaning operations
            original_df: Original dataset before cleaning
            cleaned_df: Final cleaned dataset
            
        Returns:
            Dictionary containing all educational explanations and reports
        """
        self.logger.info("Generating comprehensive educational explanations")
        
        explanations = {
            "overview": self._generate_overview_explanation(cleaning_results, original_df, cleaned_df),
            "step_by_step": self._generate_step_by_step_explanations(cleaning_results),
            "impact_assessment": self._generate_impact_assessment(original_df, cleaned_df, cleaning_results),
            "best_practices": self._generate_best_practices_guide(cleaning_results),
            "recommendations": self._generate_recommendations(cleaning_results),
            "markdown_report": self._generate_markdown_report(cleaning_results, original_df, cleaned_df),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info("Educational explanations generated successfully")
        return explanations
    
    def _generate_overview_explanation(self, cleaning_results: Dict[str, Any], 
                                     original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level overview of the data cleaning process"""
        
        # Calculate basic statistics
        original_shape = original_df.shape
        cleaned_shape = cleaned_df.shape
        rows_removed = original_shape[0] - cleaned_shape[0]
        columns_processed = len(cleaning_results.get('cleaning_actions_taken', []))
        
        # Determine data quality improvement
        quality_improvement = self._assess_quality_improvement(original_df, cleaned_df)
        
        overview = {
            "title": "Data Cleaning Overview",
            "summary": f"Your dataset has been processed through {columns_processed} cleaning operations to improve data quality and prepare it for analysis.",
            "dataset_changes": {
                "original_shape": original_shape,
                "cleaned_shape": cleaned_shape,
                "rows_removed": rows_removed,
                "columns_processed": columns_processed
            },
            "quality_improvement": quality_improvement,
            "key_achievements": self._identify_key_achievements(cleaning_results),
            "overall_impact": self._assess_overall_impact(cleaning_results, quality_improvement)
        }
        
        return overview
    
    def _generate_step_by_step_explanations(self, cleaning_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed explanations for each cleaning step"""
        
        steps = []
        
        # Missing Value Analysis
        if 'missing_value_analysis' in cleaning_results:
            steps.append(self._explain_missing_value_analysis(cleaning_results['missing_value_analysis']))
        
        # Data Type Validation
        if 'data_type_validation' in cleaning_results:
            steps.append(self._explain_data_type_validation(cleaning_results['data_type_validation']))
        
        # Outlier Detection
        if 'outlier_detection' in cleaning_results:
            steps.append(self._explain_outlier_detection(cleaning_results['outlier_detection']))
        
        # Missing Value Imputation
        if 'missing_value_imputation' in cleaning_results:
            steps.append(self._explain_missing_value_imputation(cleaning_results['missing_value_imputation']))
        
        return steps
    
    def _explain_missing_value_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for missing value analysis"""
        
        stats = analysis_results.get('missing_statistics', {})
        patterns = analysis_results.get('missing_patterns', {})
        
        explanation = {
            "step_name": "Missing Value Analysis",
            "purpose": "Understanding the extent and patterns of missing data in your dataset",
            "what_was_done": [
                f"Identified {stats.get('total_missing', 0)} missing values across {stats.get('columns_with_missing', 0)} columns",
                f"Calculated missing percentage: {stats.get('missing_percentage', 0):.2f}%",
                f"Analyzed missing patterns: {', '.join([k for k, v in patterns.items() if v])}"
            ],
            "why_important": [
                "Missing data can bias your analysis results",
                "Understanding missing patterns helps choose appropriate imputation strategies",
                "High missing data percentages may indicate data quality issues"
            ],
            "key_findings": [
                f"Missing data is {'high' if stats.get('missing_percentage', 0) > 20 else 'moderate' if stats.get('missing_percentage', 0) > 5 else 'low'}",
                f"Pattern type: {self._identify_missing_pattern_type(patterns)}",
                f"Most affected columns: {self._get_most_affected_columns(analysis_results)}"
            ],
            "recommendations": analysis_results.get('recommendations', [])[:3],  # Top 3 recommendations
            "next_steps": "Based on this analysis, appropriate imputation strategies will be selected"
        }
        
        return explanation
    
    def _explain_data_type_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for data type validation"""
        
        quality = validation_results.get('quality_assessment', {})
        recommendations = validation_results.get('conversion_recommendations', {})
        
        explanation = {
            "step_name": "Data Type Validation",
            "purpose": "Ensuring all data is stored in the correct format for analysis",
            "what_was_done": [
                f"Validated data types for {len(validation_results.get('type_analysis', {}))} columns",
                f"Identified {quality.get('columns_with_issues', 0)} columns with type issues",
                f"Generated {len(recommendations)} conversion recommendations"
            ],
            "why_important": [
                "Correct data types ensure accurate calculations and analysis",
                "Proper types enable appropriate statistical operations",
                "Type consistency prevents errors in machine learning models"
            ],
            "key_findings": [
                f"Data quality score: {quality.get('overall_quality_score', 0):.1f}/100",
                f"Type consistency: {'Good' if quality.get('consistency_score', 0) > 0.8 else 'Needs improvement'}",
                f"Conversion recommendations: {len(recommendations)} columns need attention"
            ],
            "recommendations": validation_results.get('recommendations', [])[:3],
            "next_steps": "Data types will be converted based on validation results"
        }
        
        return explanation
    
    def _explain_outlier_detection(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for outlier detection"""
        
        summary = detection_results.get('outlier_summary', {})
        method_results = detection_results.get('method_results', {})
        
        explanation = {
            "step_name": "Outlier Detection",
            "purpose": "Identifying unusual values that may be errors or legitimate extreme cases",
            "what_was_done": [
                f"Analyzed {summary.get('total_columns_analyzed', 0)} numeric columns for outliers",
                f"Used {len(method_results)} different detection methods",
                f"Detected {summary.get('total_outliers_detected', 0)} outliers across {summary.get('columns_with_outliers', 0)} columns"
            ],
            "why_important": [
                "Outliers can significantly affect statistical analysis results",
                "Extreme values may indicate data entry errors",
                "Understanding outliers helps in choosing appropriate analysis methods"
            ],
            "key_findings": [
                f"Outlier rate: {summary.get('outlier_percentage', 0):.2f}% of all values",
                f"Best detection method: {summary.get('most_common_method', 'Unknown')}",
                f"Columns with outliers: {summary.get('columns_with_outliers', 0)} out of {summary.get('total_columns_analyzed', 0)}"
            ],
            "recommendations": detection_results.get('recommendations', [])[:3],
            "next_steps": "Outliers will be flagged for review and potential treatment"
        }
        
        return explanation
    
    def _explain_missing_value_imputation(self, imputation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for missing value imputation"""
        
        details = imputation_results.get('imputation_details', {})
        quality = imputation_results.get('quality_assessment', {})
        strategy = imputation_results.get('strategy_used', 'Unknown')
        
        explanation = {
            "step_name": "Missing Value Imputation",
            "purpose": "Filling missing values with appropriate estimates to preserve data for analysis",
            "what_was_done": [
                f"Used {strategy} strategy for imputation",
                f"Imputed {details.get('imputation_stats', {}).get('total_values_imputed', 0)} missing values",
                f"Processed {details.get('imputation_stats', {}).get('columns_processed', 0)} columns"
            ],
            "why_important": [
                "Imputation allows you to use all available data for analysis",
                "Proper imputation preserves data relationships and patterns",
                "Quality imputation reduces bias in your results"
            ],
            "key_findings": [
                f"Imputation success rate: {quality.get('imputation_success_rate', 0):.1f}%",
                f"Quality score: {quality.get('quality_score', 0):.1f}/100",
                f"Data integrity preserved: {'Yes' if quality.get('data_integrity_preserved', False) else 'No'}"
            ],
            "recommendations": imputation_results.get('recommendations', [])[:3],
            "next_steps": "Imputed data is ready for analysis with documented imputation strategy"
        }
        
        return explanation
    
    def _generate_impact_assessment(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                  cleaning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate assessment of cleaning impact on the dataset"""
        
        # Calculate statistical changes
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        
        statistical_changes = {}
        for col in numeric_cols:
            if col in cleaned_df.columns:
                original_stats = {
                    'mean': original_df[col].mean(),
                    'std': original_df[col].std(),
                    'min': original_df[col].min(),
                    'max': original_df[col].max()
                }
                
                cleaned_stats = {
                    'mean': cleaned_df[col].mean(),
                    'std': cleaned_df[col].std(),
                    'min': cleaned_df[col].min(),
                    'max': cleaned_df[col].max()
                }
                
                changes = {}
                for stat in ['mean', 'std', 'min', 'max']:
                    if original_stats[stat] != 0:
                        changes[stat] = ((cleaned_stats[stat] - original_stats[stat]) / abs(original_stats[stat])) * 100
                    else:
                        changes[stat] = 0
                
                statistical_changes[col] = changes
        
        # Assess overall impact
        impact_level = self._assess_impact_level(statistical_changes, original_df, cleaned_df)
        
        assessment = {
            "overall_impact": impact_level,
            "dataset_changes": {
                "rows_removed": original_df.shape[0] - cleaned_df.shape[0],
                "columns_modified": len([col for col in numeric_cols if col in statistical_changes]),
                "missing_values_imputed": self._count_imputed_values(cleaning_results)
            },
            "statistical_changes": statistical_changes,
            "quality_improvements": self._identify_quality_improvements(cleaning_results),
            "data_integrity": self._assess_data_integrity(original_df, cleaned_df),
            "recommendations": self._generate_impact_recommendations(impact_level, statistical_changes)
        }
        
        return assessment
    
    def _generate_best_practices_guide(self, cleaning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate best practices guide based on cleaning results"""
        
        best_practices = {
            "data_quality_monitoring": [
                "Regularly check for missing values and outliers in new data",
                "Validate data types when importing data from external sources",
                "Document data quality issues and their resolution strategies"
            ],
            "imputation_strategies": [
                "Choose imputation methods based on missing data patterns (MCAR, MAR, MNAR)",
                "Use multiple imputation methods and compare results when possible",
                "Always validate imputed values against domain knowledge"
            ],
            "outlier_handling": [
                "Investigate outliers before deciding to remove or transform them",
                "Consider the context and domain knowledge when evaluating outliers",
                "Document outlier treatment decisions for reproducibility"
            ],
            "data_validation": [
                "Implement automated data validation checks in your data pipeline",
                "Use consistent data type standards across all datasets",
                "Regularly audit data quality metrics and set up alerts"
            ],
            "documentation": [
                "Document all data cleaning steps and their rationale",
                "Maintain a data cleaning log with timestamps and decisions",
                "Create reproducible cleaning scripts for future use"
            ]
        }
        
        # Add specific recommendations based on cleaning results
        if 'missing_value_analysis' in cleaning_results:
            missing_pct = cleaning_results['missing_value_analysis'].get('missing_statistics', {}).get('missing_percentage', 0)
            if missing_pct > 20:
                best_practices["high_missing_data"] = [
                    "Consider collecting additional data to reduce missing values",
                    "Investigate the root causes of missing data patterns",
                    "Use multiple imputation methods and sensitivity analysis"
                ]
        
        if 'outlier_detection' in cleaning_results:
            outlier_pct = cleaning_results['outlier_detection'].get('outlier_summary', {}).get('outlier_percentage', 0)
            if outlier_pct > 10:
                best_practices["high_outlier_rate"] = [
                    "Review data collection processes for potential errors",
                    "Consider data transformation techniques for skewed distributions",
                    "Use robust statistical methods that are less sensitive to outliers"
                ]
        
        return best_practices
    
    def _generate_recommendations(self, cleaning_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific recommendations based on cleaning results"""
        
        recommendations = []
        
        # Missing value recommendations
        if 'missing_value_analysis' in cleaning_results:
            missing_stats = cleaning_results['missing_value_analysis'].get('missing_statistics', {})
            missing_pct = missing_stats.get('missing_percentage', 0)
            
            if missing_pct > 30:
                recommendations.append({
                    "category": "Data Collection",
                    "priority": "High",
                    "recommendation": "Consider collecting additional data or investigating data collection processes",
                    "rationale": f"High missing data percentage ({missing_pct:.1f}%) may significantly impact analysis reliability"
                })
            elif missing_pct > 10:
                recommendations.append({
                    "category": "Imputation Strategy",
                    "priority": "Medium",
                    "recommendation": "Use multiple imputation methods and perform sensitivity analysis",
                    "rationale": f"Moderate missing data ({missing_pct:.1f}%) requires careful imputation strategy selection"
                })
        
        # Outlier recommendations
        if 'outlier_detection' in cleaning_results:
            outlier_stats = cleaning_results['outlier_detection'].get('outlier_summary', {})
            outlier_pct = outlier_stats.get('outlier_percentage', 0)
            
            if outlier_pct > 15:
                recommendations.append({
                    "category": "Data Quality",
                    "priority": "High",
                    "recommendation": "Investigate and potentially clean outlier data before analysis",
                    "rationale": f"High outlier rate ({outlier_pct:.1f}%) may indicate data quality issues"
                })
        
        # Data type recommendations
        if 'data_type_validation' in cleaning_results:
            type_quality = cleaning_results['data_type_validation'].get('quality_assessment', {})
            quality_score = type_quality.get('overall_quality_score', 0)
            
            if quality_score < 70:
                recommendations.append({
                    "category": "Data Types",
                    "priority": "Medium",
                    "recommendation": "Review and correct data type issues before analysis",
                    "rationale": f"Low data type quality score ({quality_score:.1f}) may cause analysis errors"
                })
        
        # General recommendations
        recommendations.extend([
            {
                "category": "Documentation",
                "priority": "Low",
                "recommendation": "Document all data cleaning decisions for future reference",
                "rationale": "Good documentation ensures reproducibility and knowledge transfer"
            },
            {
                "category": "Validation",
                "priority": "Medium",
                "recommendation": "Validate cleaning results against domain knowledge",
                "rationale": "Domain expertise helps ensure cleaning decisions are appropriate"
            }
        ])
        
        return recommendations
    
    def _generate_markdown_report(self, cleaning_results: Dict[str, Any], 
                                original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
        """Generate a comprehensive markdown report"""
        
        report = []
        
        # Header
        report.append("# Data Cleaning Report")
        report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overview
        overview = self._generate_overview_explanation(cleaning_results, original_df, cleaned_df)
        report.append("## Overview")
        report.append(overview['summary'])
        report.append("")
        
        # Dataset Changes
        report.append("### Dataset Changes")
        changes = overview['dataset_changes']
        report.append(f"- **Original shape:** {changes['original_shape'][0]} rows × {changes['original_shape'][1]} columns")
        report.append(f"- **Cleaned shape:** {changes['cleaned_shape'][0]} rows × {changes['cleaned_shape'][1]} columns")
        report.append(f"- **Rows removed:** {changes['rows_removed']}")
        report.append(f"- **Columns processed:** {changes['columns_processed']}")
        report.append("")
        
        # Step-by-step explanations
        steps = self._generate_step_by_step_explanations(cleaning_results)
        report.append("## Cleaning Steps")
        
        for i, step in enumerate(steps, 1):
            report.append(f"### {i}. {step['step_name']}")
            report.append(f"**Purpose:** {step['purpose']}")
            report.append("")
            
            report.append("**What was done:**")
            for action in step['what_was_done']:
                report.append(f"- {action}")
            report.append("")
            
            report.append("**Why this is important:**")
            for reason in step['why_important']:
                report.append(f"- {reason}")
            report.append("")
            
            report.append("**Key findings:**")
            for finding in step['key_findings']:
                report.append(f"- {finding}")
            report.append("")
            
            if step.get('recommendations'):
                report.append("**Recommendations:**")
                for rec in step['recommendations']:
                    report.append(f"- {rec}")
                report.append("")
        
        # Impact Assessment
        impact = self._generate_impact_assessment(original_df, cleaned_df, cleaning_results)
        report.append("## Impact Assessment")
        report.append(f"**Overall impact:** {impact['overall_impact']}")
        report.append("")
        
        # Best Practices
        best_practices = self._generate_best_practices_guide(cleaning_results)
        report.append("## Best Practices Applied")
        for category, practices in best_practices.items():
            if isinstance(practices, list):
                report.append(f"### {category.replace('_', ' ').title()}")
                for practice in practices:
                    report.append(f"- {practice}")
                report.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(cleaning_results)
        report.append("## Recommendations")
        for rec in recommendations:
            report.append(f"### {rec['category']} (Priority: {rec['priority']})")
            report.append(f"**Recommendation:** {rec['recommendation']}")
            report.append(f"**Rationale:** {rec['rationale']}")
            report.append("")
        
        return "\n".join(report)
    
    # Helper methods
    def _assess_quality_improvement(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
        """Assess overall quality improvement"""
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        
        if cleaned_missing == 0 and original_missing > 0:
            return "Significant improvement - all missing values addressed"
        elif cleaned_missing < original_missing * 0.5:
            return "Major improvement - missing values reduced by more than 50%"
        elif cleaned_missing < original_missing * 0.8:
            return "Moderate improvement - missing values reduced"
        else:
            return "Minor improvement - some missing values remain"
    
    def _identify_key_achievements(self, cleaning_results: Dict[str, Any]) -> List[str]:
        """Identify key achievements in the cleaning process"""
        achievements = []
        
        if 'missing_value_imputation' in cleaning_results:
            imputed = cleaning_results['missing_value_imputation'].get('imputation_details', {}).get('imputation_stats', {}).get('total_values_imputed', 0)
            if imputed > 0:
                achievements.append(f"Imputed {imputed} missing values")
        
        if 'outlier_detection' in cleaning_results:
            outliers = cleaning_results['outlier_detection'].get('outlier_summary', {}).get('total_outliers_detected', 0)
            if outliers > 0:
                achievements.append(f"Identified {outliers} outliers for review")
        
        if 'data_type_validation' in cleaning_results:
            issues = cleaning_results['data_type_validation'].get('quality_assessment', {}).get('columns_with_issues', 0)
            if issues > 0:
                achievements.append(f"Validated data types for {issues} columns with issues")
        
        return achievements
    
    def _assess_overall_impact(self, cleaning_results: Dict[str, Any], quality_improvement: str) -> str:
        """Assess overall impact of cleaning process"""
        if "Significant" in quality_improvement:
            return "High impact - major data quality improvements achieved"
        elif "Major" in quality_improvement:
            return "Medium-high impact - substantial improvements made"
        elif "Moderate" in quality_improvement:
            return "Medium impact - noticeable improvements achieved"
        else:
            return "Low impact - minor improvements made"
    
    def _identify_missing_pattern_type(self, patterns: Dict[str, bool]) -> str:
        """Identify the type of missing data pattern"""
        if patterns.get('MCAR'):
            return "Missing Completely At Random (MCAR)"
        elif patterns.get('MAR'):
            return "Missing At Random (MAR)"
        elif patterns.get('MNAR'):
            return "Missing Not At Random (MNAR)"
        else:
            return "Unknown pattern"
    
    def _get_most_affected_columns(self, analysis_results: Dict[str, Any]) -> str:
        """Get the most affected columns by missing data"""
        missing_by_column = analysis_results.get('missing_by_column', {})
        if not missing_by_column:
            return "None"
        
        # Sort by missing percentage and get top 3
        sorted_cols = sorted(missing_by_column.items(), 
                           key=lambda x: x[1].get('percentage', 0), reverse=True)
        top_cols = [f"{col} ({info['percentage']:.1f}%)" for col, info in sorted_cols[:3]]
        return ", ".join(top_cols)
    
    def _count_imputed_values(self, cleaning_results: Dict[str, Any]) -> int:
        """Count total imputed values"""
        if 'missing_value_imputation' in cleaning_results:
            return cleaning_results['missing_value_imputation'].get('imputation_details', {}).get('imputation_stats', {}).get('total_values_imputed', 0)
        return 0
    
    def _assess_impact_level(self, statistical_changes: Dict[str, Dict], 
                           original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
        """Assess the level of impact from cleaning"""
        if not statistical_changes:
            return "Minimal impact"
        
        # Calculate average absolute change
        all_changes = []
        for col_changes in statistical_changes.values():
            all_changes.extend([abs(change) for change in col_changes.values()])
        
        avg_change = np.mean(all_changes) if all_changes else 0
        
        if avg_change > 20:
            return "High impact - significant statistical changes"
        elif avg_change > 10:
            return "Medium impact - moderate statistical changes"
        elif avg_change > 5:
            return "Low-medium impact - minor statistical changes"
        else:
            return "Minimal impact - very small statistical changes"
    
    def _identify_quality_improvements(self, cleaning_results: Dict[str, Any]) -> List[str]:
        """Identify specific quality improvements made"""
        improvements = []
        
        if 'missing_value_imputation' in cleaning_results:
            quality = cleaning_results['missing_value_imputation'].get('quality_assessment', {})
            if quality.get('imputation_success_rate', 0) > 80:
                improvements.append("High-quality missing value imputation")
        
        if 'data_type_validation' in cleaning_results:
            quality = cleaning_results['data_type_validation'].get('quality_assessment', {})
            if quality.get('overall_quality_score', 0) > 80:
                improvements.append("Improved data type consistency")
        
        if 'outlier_detection' in cleaning_results:
            summary = cleaning_results['outlier_detection'].get('outlier_summary', {})
            if summary.get('total_outliers_detected', 0) > 0:
                improvements.append("Outlier identification and flagging")
        
        return improvements
    
    def _assess_data_integrity(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
        """Assess data integrity after cleaning"""
        if cleaned_df.shape[0] == original_df.shape[0] and cleaned_df.shape[1] == original_df.shape[1]:
            return "Data integrity preserved - no rows or columns lost"
        elif cleaned_df.shape[0] < original_df.shape[0]:
            return f"Some rows removed ({original_df.shape[0] - cleaned_df.shape[0]}) - review if appropriate"
        else:
            return "Data structure modified - review changes carefully"
    
    def _generate_impact_recommendations(self, impact_level: str, statistical_changes: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on impact level"""
        recommendations = []
        
        if "High impact" in impact_level:
            recommendations.extend([
                "Review statistical changes carefully before proceeding with analysis",
                "Consider sensitivity analysis to understand impact of cleaning decisions",
                "Document all cleaning decisions for reproducibility"
            ])
        elif "Medium impact" in impact_level:
            recommendations.extend([
                "Monitor the impact of cleaning on your analysis results",
                "Consider validating results with original data if possible"
            ])
        else:
            recommendations.append("Cleaning impact is minimal - proceed with confidence")
        
        return recommendations
