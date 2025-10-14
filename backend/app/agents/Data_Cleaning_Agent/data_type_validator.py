"""
Data Type Validation Module

This module provides comprehensive data type validation capabilities including:
- Data type detection and validation
- Type conversion recommendations
- Data quality assessment for each column
- Educational explanations for type issues
- Validation reports and recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

class DataTypeValidator:
    """
    Comprehensive data type validation tool for data cleaning workflows
    """
    
    def __init__(self):
        self.logger = logging.getLogger("data_type_validator")
        
        # Define expected data types and their characteristics
        self.expected_types = {
            'numeric': {
                'dtypes': ['int64', 'float64', 'int32', 'float32'],
                'description': 'Numeric data (integers and floats)',
                'validation_methods': ['is_numeric', 'has_numeric_range']
            },
            'categorical': {
                'dtypes': ['object', 'category'],
                'description': 'Categorical data (strings, categories)',
                'validation_methods': ['is_categorical', 'has_reasonable_cardinality']
            },
            'datetime': {
                'dtypes': ['datetime64[ns]', 'datetime64[ms]'],
                'description': 'Date and time data',
                'validation_methods': ['is_datetime', 'has_valid_date_range']
            },
            'boolean': {
                'dtypes': ['bool'],
                'description': 'Boolean data (True/False)',
                'validation_methods': ['is_boolean', 'has_binary_values']
            }
        }
        
        # Common data type issues patterns
        self.type_issue_patterns = {
            'mixed_numeric': r'^[\d\.,\-\+\s]+$',
            'currency': r'^[\$€£¥]\s*[\d,]+\.?\d*$',
            'percentage': r'^\d+\.?\d*\s*%$',
            'phone': r'^[\d\-\+\(\)\s]+$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'date_formats': [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
                r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
                r'^\d{1,2}/\d{1,2}/\d{4}$',  # M/D/YYYY
                r'^\d{4}/\d{1,2}/\d{1,2}$',  # YYYY/M/D
            ]
        }
    
    def validate_data_types(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data type validation
        
        Args:
            df: DataFrame to validate
            target_column: Optional target column for focused analysis
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        self.logger.info("Starting comprehensive data type validation")
        
        # Basic data type analysis
        type_analysis = self._analyze_current_types(df)
        
        # Type quality assessment
        quality_assessment = self._assess_type_quality(df)
        
        # Type conversion recommendations
        conversion_recommendations = self._generate_conversion_recommendations(df)
        
        # Type consistency analysis
        consistency_analysis = self._analyze_type_consistency(df)
        
        # Target-specific analysis if target column provided
        target_analysis = {}
        if target_column and target_column in df.columns:
            target_analysis = self._analyze_target_type(df, target_column)
        
        # Generate validation report
        validation_report = self._generate_validation_report(
            type_analysis, quality_assessment, conversion_recommendations, consistency_analysis
        )
        
        # Generate educational explanations
        explanations = self._generate_type_explanations(type_analysis, quality_assessment)
        
        # Compile comprehensive results
        validation_results = {
            "type_analysis": type_analysis,
            "quality_assessment": quality_assessment,
            "conversion_recommendations": conversion_recommendations,
            "consistency_analysis": consistency_analysis,
            "target_analysis": target_analysis,
            "validation_report": validation_report,
            "explanations": explanations,
            "recommendations": self._generate_type_recommendations(type_analysis, quality_assessment),
            "validation_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Data type validation completed for {len(df.columns)} columns")
        
        return validation_results
    
    def _analyze_current_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current data types in the dataset"""
        
        type_analysis = {
            "column_types": {},
            "type_distribution": {},
            "type_issues": {},
            "mixed_type_columns": [],
            "inconsistent_types": []
        }
        
        for column in df.columns:
            # Get current data type
            current_dtype = str(df[column].dtype)
            
            # Analyze the column
            column_analysis = self._analyze_column_type(df[column], column)
            
            type_analysis["column_types"][column] = {
                "current_dtype": current_dtype,
                "detected_type": column_analysis["detected_type"],
                "confidence": column_analysis["confidence"],
                "issues": column_analysis["issues"],
                "recommendations": column_analysis["recommendations"]
            }
            
            # Check for mixed types
            if column_analysis["has_mixed_types"]:
                type_analysis["mixed_type_columns"].append(column)
            
            # Check for type inconsistencies
            if column_analysis["confidence"] < 0.7:
                type_analysis["inconsistent_types"].append(column)
        
        # Calculate type distribution
        type_counts = {}
        for col_analysis in type_analysis["column_types"].values():
            detected_type = col_analysis["detected_type"]
            type_counts[detected_type] = type_counts.get(detected_type, 0) + 1
        
        type_analysis["type_distribution"] = type_counts
        
        return type_analysis
    
    def _analyze_column_type(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze a single column's data type"""
        
        analysis = {
            "detected_type": "unknown",
            "confidence": 0.0,
            "has_mixed_types": False,
            "issues": [],
            "recommendations": []
        }
        
        # Get non-null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            analysis["detected_type"] = "empty"
            analysis["confidence"] = 1.0
            analysis["issues"].append("Column is completely empty")
            return analysis
        
        # Test for different data types
        type_scores = {}
        
        # Test for numeric
        numeric_score = self._test_numeric_type(non_null_series)
        type_scores["numeric"] = numeric_score
        
        # Test for datetime
        datetime_score = self._test_datetime_type(non_null_series)
        type_scores["datetime"] = datetime_score
        
        # Test for boolean
        boolean_score = self._test_boolean_type(non_null_series)
        type_scores["boolean"] = boolean_score
        
        # Test for categorical
        categorical_score = self._test_categorical_type(non_null_series)
        type_scores["categorical"] = categorical_score
        
        # Determine the best type
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        analysis["detected_type"] = best_type
        analysis["confidence"] = best_score
        
        # Check for mixed types
        if best_score < 0.8:
            analysis["has_mixed_types"] = True
            analysis["issues"].append("Column appears to have mixed data types")
        
        # Generate specific issues and recommendations
        if best_type == "numeric":
            analysis.update(self._analyze_numeric_issues(series))
        elif best_type == "categorical":
            analysis.update(self._analyze_categorical_issues(series))
        elif best_type == "datetime":
            analysis.update(self._analyze_datetime_issues(series))
        elif best_type == "boolean":
            analysis.update(self._analyze_boolean_issues(series))
        
        return analysis
    
    def _test_numeric_type(self, series: pd.Series) -> float:
        """Test if a series should be numeric type"""
        
        if series.dtype in ['int64', 'float64', 'int32', 'float32']:
            return 1.0
        
        # Check if most values can be converted to numeric
        numeric_count = 0
        total_count = len(series)
        
        for value in series.head(100):  # Sample first 100 values
            try:
                pd.to_numeric(value)
                numeric_count += 1
            except:
                pass
        
        return numeric_count / min(100, total_count)
    
    def _test_datetime_type(self, series: pd.Series) -> float:
        """Test if a series should be datetime type"""
        
        if str(series.dtype).startswith('datetime64'):
            return 1.0
        
        # Check if values match date patterns
        date_matches = 0
        sample_size = min(50, len(series))
        
        for value in series.head(sample_size):
            str_value = str(value)
            if any(re.match(pattern, str_value) for pattern in self.type_issue_patterns['date_formats']):
                date_matches += 1
        
        return date_matches / sample_size
    
    def _test_boolean_type(self, series: pd.Series) -> float:
        """Test if a series should be boolean type"""
        
        if series.dtype == 'bool':
            return 1.0
        
        # Check for boolean-like values
        unique_values = series.astype(str).str.lower().unique()
        
        # Common boolean patterns
        boolean_patterns = ['true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n']
        boolean_like = any(val in boolean_patterns for val in unique_values)
        
        # Check if it's binary
        is_binary = len(unique_values) <= 2
        
        if boolean_like or is_binary:
            return 0.8
        else:
            return 0.0
    
    def _test_categorical_type(self, series: pd.Series) -> float:
        """Test if a series should be categorical type"""
        
        if series.dtype in ['object', 'category']:
            # Check if it has reasonable cardinality for categorical
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.5:  # Less than 50% unique values
                return 0.9
            else:
                return 0.6
        
        return 0.0
    
    def _analyze_numeric_issues(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze issues specific to numeric columns"""
        
        issues = []
        recommendations = []
        
        # Check for non-numeric values
        non_numeric_count = 0
        for value in series.dropna():
            try:
                pd.to_numeric(value)
            except:
                non_numeric_count += 1
        
        if non_numeric_count > 0:
            issues.append(f"Contains {non_numeric_count} non-numeric values")
            recommendations.append("Convert non-numeric values to NaN or appropriate numeric values")
        
        # Check for outliers (only for numeric data)
        if series.dtype in ['int64', 'float64']:
            try:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
                
                if len(outliers) > 0:
                    issues.append(f"Contains {len(outliers)} potential outliers")
                    recommendations.append("Review outliers for data quality issues")
            except Exception:
                pass  # Skip outlier detection if there are issues
        
        # Check for negative values where they shouldn't be (only for numeric data)
        if series.dtype in ['int64', 'float64'] and series.name and any(keyword in series.name.lower() for keyword in ['age', 'count', 'price', 'amount']):
            try:
                if (series < 0).any():
                    issues.append("Contains negative values which may be inappropriate")
                    recommendations.append("Review negative values for data quality")
            except Exception:
                pass  # Skip negative value check if there are issues
        
        return {"issues": issues, "recommendations": recommendations}
    
    def _analyze_categorical_issues(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze issues specific to categorical columns"""
        
        issues = []
        recommendations = []
        
        # Check for high cardinality
        unique_count = series.nunique()
        total_count = len(series)
        cardinality_ratio = unique_count / total_count
        
        if cardinality_ratio > 0.9:
            issues.append(f"Very high cardinality ({unique_count}/{total_count} unique values)")
            recommendations.append("Consider if this should be a categorical column or if it needs encoding")
        
        # Check for inconsistent formatting
        if series.dtype == 'object':
            # Check for case inconsistencies
            unique_values = series.unique()
            case_variations = {}
            for value in unique_values:
                if pd.notna(value):
                    lower_val = str(value).lower()
                    if lower_val not in case_variations:
                        case_variations[lower_val] = []
                    case_variations[lower_val].append(value)
            
            case_inconsistencies = [variations for variations in case_variations.values() if len(variations) > 1]
            if case_inconsistencies:
                issues.append(f"Case inconsistencies found in {len(case_inconsistencies)} groups")
                recommendations.append("Standardize case formatting for consistency")
        
        # Check for leading/trailing whitespace
        if series.dtype == 'object':
            whitespace_issues = series.astype(str).str.strip().ne(series.astype(str)).sum()
            if whitespace_issues > 0:
                issues.append(f"Contains {whitespace_issues} values with leading/trailing whitespace")
                recommendations.append("Remove leading and trailing whitespace")
        
        return {"issues": issues, "recommendations": recommendations}
    
    def _analyze_datetime_issues(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze issues specific to datetime columns"""
        
        issues = []
        recommendations = []
        
        # Check for invalid dates
        if str(series.dtype).startswith('datetime64'):
            invalid_dates = series.isna().sum()
            if invalid_dates > 0:
                issues.append(f"Contains {invalid_dates} invalid or missing dates")
                recommendations.append("Handle invalid dates appropriately")
        
        # Check for future dates where inappropriate
        if series.name and any(keyword in series.name.lower() for keyword in ['birth', 'created', 'start']):
            future_dates = series > pd.Timestamp.now()
            if future_dates.any():
                issues.append("Contains future dates which may be inappropriate")
                recommendations.append("Review future dates for data quality")
        
        return {"issues": issues, "recommendations": recommendations}
    
    def _analyze_boolean_issues(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze issues specific to boolean columns"""
        
        issues = []
        recommendations = []
        
        # Check for missing values
        missing_count = series.isna().sum()
        if missing_count > 0:
            issues.append(f"Contains {missing_count} missing boolean values")
            recommendations.append("Consider how to handle missing boolean values")
        
        # Check for non-boolean values
        if series.dtype == 'object':
            non_boolean_values = series[~series.isin([True, False, 'True', 'False', 'true', 'false', '1', '0', 'T', 'F', 'Y', 'N', 'Yes', 'No'])]
            if len(non_boolean_values) > 0:
                issues.append(f"Contains {len(non_boolean_values)} non-boolean values")
                recommendations.append("Convert non-boolean values to proper boolean format")
        
        return {"issues": issues, "recommendations": recommendations}
    
    def _assess_type_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the overall quality of data types in the dataset"""
        
        quality_metrics = {
            "overall_type_consistency": 0.0,
            "columns_with_issues": 0,
            "type_appropriateness_score": 0.0,
            "conversion_needed": 0,
            "quality_issues": []
        }
        
        total_columns = len(df.columns)
        columns_with_issues = 0
        conversion_needed = 0
        appropriateness_scores = []
        
        for column in df.columns:
            # Analyze column type quality
            column_analysis = self._analyze_column_type(df[column], column)
            
            if column_analysis["confidence"] < 0.8:
                columns_with_issues += 1
                quality_metrics["quality_issues"].append(f"Column '{column}' has type confidence {column_analysis['confidence']:.2f}")
            
            if column_analysis["has_mixed_types"]:
                conversion_needed += 1
            
            appropriateness_scores.append(column_analysis["confidence"])
        
        quality_metrics["overall_type_consistency"] = 1 - (columns_with_issues / total_columns)
        quality_metrics["columns_with_issues"] = columns_with_issues
        quality_metrics["type_appropriateness_score"] = np.mean(appropriateness_scores)
        quality_metrics["conversion_needed"] = conversion_needed
        
        return quality_metrics
    
    def _generate_conversion_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate recommendations for data type conversions"""
        
        recommendations = {
            "immediate_conversions": [],
            "investigation_needed": [],
            "conversion_priority": []
        }
        
        for column in df.columns:
            column_analysis = self._analyze_column_type(df[column], column)
            current_dtype = str(df[column].dtype)
            detected_type = column_analysis["detected_type"]
            confidence = column_analysis["confidence"]
            
            if confidence > 0.9 and current_dtype != detected_type:
                # High confidence conversion
                recommendations["immediate_conversions"].append({
                    "column": column,
                    "current_type": current_dtype,
                    "recommended_type": detected_type,
                    "confidence": confidence,
                    "reason": "High confidence type detection"
                })
            elif confidence > 0.7 and current_dtype != detected_type:
                # Medium confidence - needs investigation
                recommendations["investigation_needed"].append({
                    "column": column,
                    "current_type": current_dtype,
                    "recommended_type": detected_type,
                    "confidence": confidence,
                    "reason": "Medium confidence - investigate data quality"
                })
            elif confidence < 0.5:
                # Low confidence - high priority for investigation
                recommendations["conversion_priority"].append({
                    "column": column,
                    "current_type": current_dtype,
                    "detected_type": detected_type,
                    "confidence": confidence,
                    "reason": "Low confidence - data quality issues likely"
                })
        
        return recommendations
    
    def _analyze_type_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency of data types across the dataset"""
        
        consistency_analysis = {
            "type_consistency_score": 0.0,
            "inconsistent_columns": [],
            "type_patterns": {},
            "recommendations": []
        }
        
        # Analyze type patterns
        type_groups = {}
        for column in df.columns:
            column_analysis = self._analyze_column_type(df[column], column)
            detected_type = column_analysis["detected_type"]
            
            if detected_type not in type_groups:
                type_groups[detected_type] = []
            type_groups[detected_type].append(column)
        
        consistency_analysis["type_patterns"] = type_groups
        
        # Check for inconsistent naming patterns
        for detected_type, columns in type_groups.items():
            if len(columns) > 1:
                # Check if column names follow consistent patterns
                name_patterns = {}
                for col in columns:
                    # Extract common patterns from column names
                    if any(keyword in col.lower() for keyword in ['id', 'key']):
                        pattern = 'identifier'
                    elif any(keyword in col.lower() for keyword in ['date', 'time']):
                        pattern = 'temporal'
                    elif any(keyword in col.lower() for keyword in ['name', 'title']):
                        pattern = 'text'
                    elif any(keyword in col.lower() for keyword in ['count', 'num', 'qty']):
                        pattern = 'count'
                    else:
                        pattern = 'other'
                    
                    if pattern not in name_patterns:
                        name_patterns[pattern] = []
                    name_patterns[pattern].append(col)
                
                # Check for inconsistencies
                for pattern, cols in name_patterns.items():
                    if len(cols) > 1 and detected_type not in ['categorical', 'numeric']:
                        if (pattern == 'identifier' and detected_type != 'numeric') or \
                           (pattern == 'temporal' and detected_type != 'datetime') or \
                           (pattern == 'text' and detected_type != 'categorical'):
                            consistency_analysis["inconsistent_columns"].extend(cols)
        
        # Calculate consistency score
        total_columns = len(df.columns)
        inconsistent_count = len(set(consistency_analysis["inconsistent_columns"]))
        consistency_analysis["type_consistency_score"] = 1 - (inconsistent_count / total_columns)
        
        return consistency_analysis
    
    def _analyze_target_type(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze data type specifically for the target column"""
        
        if target_column not in df.columns:
            return {}
        
        target_series = df[target_column]
        target_analysis = self._analyze_column_type(target_series, target_column)
        
        # Additional target-specific analysis
        target_analysis["is_suitable_for_ml"] = target_analysis["detected_type"] in ['numeric', 'categorical', 'boolean']
        target_analysis["encoding_required"] = target_analysis["detected_type"] == 'categorical'
        target_analysis["scaling_required"] = target_analysis["detected_type"] == 'numeric'
        
        return target_analysis
    
    def _generate_validation_report(self, type_analysis: Dict, quality_assessment: Dict, 
                                  conversion_recommendations: Dict, consistency_analysis: Dict) -> str:
        """Generate a comprehensive validation report"""
        
        report = []
        report.append("=" * 80)
        report.append("🔍 DATA TYPE VALIDATION REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        total_columns = len(type_analysis["column_types"])
        columns_with_issues = quality_assessment["columns_with_issues"]
        
        report.append(f"\n📊 VALIDATION SUMMARY:")
        report.append(f"Total columns analyzed: {total_columns}")
        report.append(f"Columns with type issues: {columns_with_issues}")
        report.append(f"Type consistency score: {consistency_analysis['type_consistency_score']:.2f}")
        report.append(f"Overall type appropriateness: {quality_assessment['type_appropriateness_score']:.2f}")
        
        # Type distribution
        report.append(f"\n📈 TYPE DISTRIBUTION:")
        for data_type, count in type_analysis["type_distribution"].items():
            report.append(f"  {data_type}: {count} columns")
        
        # Immediate conversion recommendations
        if conversion_recommendations["immediate_conversions"]:
            report.append(f"\n⚡ IMMEDIATE CONVERSIONS NEEDED:")
            for conv in conversion_recommendations["immediate_conversions"]:
                report.append(f"  {conv['column']}: {conv['current_type']} → {conv['recommended_type']} (confidence: {conv['confidence']:.2f})")
        
        # Investigation needed
        if conversion_recommendations["investigation_needed"]:
            report.append(f"\n🔍 INVESTIGATION NEEDED:")
            for inv in conversion_recommendations["investigation_needed"]:
                report.append(f"  {inv['column']}: {inv['current_type']} → {inv['recommended_type']} (confidence: {inv['confidence']:.2f})")
        
        # High priority issues
        if conversion_recommendations["conversion_priority"]:
            report.append(f"\n🚨 HIGH PRIORITY ISSUES:")
            for priority in conversion_recommendations["conversion_priority"]:
                report.append(f"  {priority['column']}: {priority['reason']} (confidence: {priority['confidence']:.2f})")
        
        # Quality issues
        if quality_assessment["quality_issues"]:
            report.append(f"\n⚠️ QUALITY ISSUES:")
            for issue in quality_assessment["quality_issues"][:10]:  # Limit to first 10
                report.append(f"  - {issue}")
            if len(quality_assessment["quality_issues"]) > 10:
                report.append(f"  ... and {len(quality_assessment['quality_issues']) - 10} more issues")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _generate_type_explanations(self, type_analysis: Dict, quality_assessment: Dict) -> Dict[str, str]:
        """Generate educational explanations for data type issues"""
        
        explanations = {}
        
        # Overall type quality explanation
        appropriateness_score = quality_assessment["type_appropriateness_score"]
        if appropriateness_score > 0.9:
            explanations["overall"] = "✅ Excellent data type quality! Most columns have appropriate and consistent data types."
        elif appropriateness_score > 0.7:
            explanations["overall"] = "⚠️ Good data type quality with some minor issues. A few columns may need type adjustments."
        elif appropriateness_score > 0.5:
            explanations["overall"] = "⚠️ Moderate data type quality. Several columns have type issues that should be addressed."
        else:
            explanations["overall"] = "🚨 Poor data type quality. Many columns have significant type issues that need immediate attention."
        
        # Type consistency explanation
        consistency_score = type_analysis.get("type_consistency_score", 0.0)
        if consistency_score > 0.8:
            explanations["consistency"] = "Data types are consistent across similar columns. Good naming and type conventions."
        else:
            explanations["consistency"] = "Some inconsistencies found in data types. Consider standardizing column naming and types."
        
        # Mixed types explanation
        mixed_count = len(type_analysis.get("mixed_type_columns", []))
        if mixed_count == 0:
            explanations["mixed_types"] = "No mixed data types detected. All columns have consistent data within each column."
        else:
            explanations["mixed_types"] = f"Found {mixed_count} columns with mixed data types. These columns may need data cleaning before type conversion."
        
        return explanations
    
    def _generate_type_recommendations(self, type_analysis: Dict, quality_assessment: Dict) -> List[str]:
        """Generate recommendations for improving data types"""
        
        recommendations = []
        
        # Overall recommendations based on quality score
        appropriateness_score = quality_assessment["type_appropriateness_score"]
        if appropriateness_score < 0.8:
            recommendations.append("Review and fix data type issues to improve data quality")
        
        # Mixed types recommendations
        mixed_columns = type_analysis.get("mixed_type_columns", [])
        if mixed_columns:
            recommendations.append(f"Clean mixed data types in columns: {', '.join(mixed_columns[:5])}")
        
        # Inconsistent types recommendations
        inconsistent_columns = type_analysis.get("inconsistent_types", [])
        if inconsistent_columns:
            recommendations.append(f"Investigate type inconsistencies in columns: {', '.join(inconsistent_columns[:5])}")
        
        # General recommendations
        recommendations.extend([
            "Use appropriate data types for better memory efficiency and analysis",
            "Convert categorical data to 'category' type for better performance",
            "Ensure datetime columns use proper datetime format",
            "Validate numeric columns for appropriate ranges and formats"
        ])
        
        return recommendations
