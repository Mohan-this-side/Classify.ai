"""
Layer 1 Analysis Methods for Enhanced Data Cleaning Agent

This module contains the Layer 1 hardcoded analysis logic extracted
from the EnhancedDataCleaningAgent for better organization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import asyncio


async def perform_layer1_cleaning(
    df: pd.DataFrame,
    target_column: Optional[str],
    missing_analyzer,
    type_validator,
    outlier_detector,
    missing_imputer
) -> Dict[str, Any]:
    """
    Perform comprehensive Layer 1 data cleaning analysis.

    Args:
        df: Input DataFrame
        target_column: Target column name (optional)
        missing_analyzer: MissingValueAnalyzer instance
        type_validator: DataTypeValidator instance
        outlier_detector: OutlierDetector instance
        missing_imputer: MissingValueImputer instance

    Returns:
        Dictionary with Layer 1 results including:
        - cleaned_data: Cleaned DataFrame
        - missing_analysis: Results from missing value analysis
        - type_validation: Results from type validation
        - outlier_detection: Results from outlier detection
        - imputation_results: Results from imputation
        - data_quality_score: Overall quality score (0-100)
        - cleaning_actions: List of actions taken
    """

    # Create a copy for cleaning
    cleaned_df = df.copy()
    cleaning_actions = []

    # 1. Comprehensive missing value analysis
    missing_analysis = missing_analyzer.analyze_missing_values(df, target_column)
    cleaning_actions.append(
        f"Analyzed missing values: {missing_analysis['missing_statistics']['total_missing']} found"
    )

    # 2. Comprehensive data type validation
    type_validation = type_validator.validate_data_types(df, target_column)
    cleaning_actions.append(
        f"Validated data types: {type_validation['quality_assessment']['columns_with_issues']} columns need attention"
    )

    # 3. Comprehensive outlier detection
    outlier_detection = outlier_detector.detect_outliers(df, target_column)
    cleaning_actions.append(
        f"Detected outliers: {outlier_detection['outlier_summary']['total_outliers_detected']} outliers found"
    )

    # 4. Fix structural errors
    cleaned_df, structural_actions = _fix_structural_errors(cleaned_df)
    cleaning_actions.extend(structural_actions)

    # 5. Intelligent type conversion
    cleaned_df, type_actions = _intelligent_type_conversion(cleaned_df, target_column)
    cleaning_actions.extend(type_actions)

    # 6. Missing value imputation
    imputation_results = missing_imputer.impute_missing_values(cleaned_df, target_column)
    cleaned_df = imputation_results['imputed_df']
    cleaning_actions.append(
        f"Imputed {imputation_results['imputation_details']['imputation_stats']['total_values_imputed']} missing values"
    )

    # 7. Handle duplicates
    cleaned_df, duplicate_actions = _handle_duplicates(cleaned_df)
    cleaning_actions.extend(duplicate_actions)

    # 8. Handle outliers (capping)
    cleaned_df, outlier_actions = _handle_outliers(cleaned_df, target_column)
    cleaning_actions.extend(outlier_actions)

    # 9. Calculate data quality score
    quality_score = _calculate_quality_score(cleaned_df, df)

    return {
        "cleaned_data": cleaned_df,
        "original_data": df,
        "missing_analysis": missing_analysis,
        "type_validation": type_validation,
        "outlier_detection": outlier_detection,
        "imputation_results": imputation_results,
        "data_quality_score": quality_score,
        "cleaning_actions": cleaning_actions,
        "dataset_stats": {
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "rows_removed": df.shape[0] - cleaned_df.shape[0],
            "columns_removed": df.shape[1] - cleaned_df.shape[1]
        }
    }


def _fix_structural_errors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fix structural errors in the dataset"""
    actions = []

    # Remove completely empty rows and columns
    empty_rows_before = df.isnull().all(axis=1).sum()
    empty_cols_before = df.isnull().all(axis=0).sum()

    df = df.dropna(how='all')  # Remove rows that are all NaN
    df = df.dropna(axis=1, how='all')  # Remove columns that are all NaN

    if empty_rows_before > 0:
        actions.append(f"Removed {empty_rows_before} completely empty rows")
    if empty_cols_before > 0:
        actions.append(f"Removed {empty_cols_before} completely empty columns")

    # Reset index
    df = df.reset_index(drop=True)

    return df, actions


def _intelligent_type_conversion(df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Intelligently detect and convert data types"""
    actions = []

    for column in df.columns:
        if column == target_column:
            continue

        original_dtype = str(df[column].dtype)

        # Skip if already properly typed
        if df[column].dtype in ['int64', 'float64', 'bool', 'datetime64[ns]']:
            continue

        # Try to convert to numeric
        if _is_numeric_column(df[column]):
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                actions.append(f"Converted '{column}' to numeric: {original_dtype} → {df[column].dtype}")
            except:
                pass

    return df, actions


def _is_numeric_column(series: pd.Series) -> bool:
    """Check if a column can be converted to numeric"""
    if series.dtype in ['int64', 'float64']:
        return True

    # Check if most values can be converted to numeric
    numeric_count = 0
    total_count = series.dropna().count()

    if total_count == 0:
        return False

    for value in series.dropna().head(100):  # Sample first 100 non-null values
        try:
            pd.to_numeric(value)
            numeric_count += 1
        except:
            pass

    return (numeric_count / min(100, total_count)) > 0.8


def _handle_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Handle duplicate rows"""
    actions = []

    exact_duplicates = df.duplicated().sum()
    if exact_duplicates > 0:
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        actions.append(f"Removed {exact_duplicates} exact duplicate rows")

    return df, actions


def _handle_outliers(df: pd.DataFrame, target_column: Optional[str], outlier_threshold: float = 3.0) -> Tuple[pd.DataFrame, List[str]]:
    """Handle outliers using capping (Winsorization)"""
    actions = []

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        if column == target_column:
            continue

        # Calculate z-scores
        try:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > outlier_threshold

            if outliers.sum() > 0:
                # Cap outliers using 5th and 95th percentiles
                upper_bound = df[column].quantile(0.95)
                lower_bound = df[column].quantile(0.05)

                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

                actions.append(f"Capped {outliers.sum()} outliers in '{column}'")
        except:
            # Skip columns that can't be processed
            pass

    return df, actions


def _calculate_quality_score(cleaned_df: pd.DataFrame, original_df: pd.DataFrame) -> float:
    """
    Calculate overall data quality score (0-100).

    Factors:
    - Completeness: No missing values
    - Consistency: No duplicates
    - Validity: Appropriate data types
    """

    # 1. Completeness (50% weight)
    total_cells = cleaned_df.shape[0] * cleaned_df.shape[1]
    if total_cells > 0:
        missing_ratio = cleaned_df.isnull().sum().sum() / total_cells
        completeness_score = (1 - missing_ratio) * 50
    else:
        completeness_score = 0

    # 2. Consistency (30% weight)
    if len(cleaned_df) > 0:
        duplicate_ratio = cleaned_df.duplicated().sum() / len(cleaned_df)
        consistency_score = (1 - duplicate_ratio) * 30
    else:
        consistency_score = 0

    # 3. Validity (20% weight)
    # Simple heuristic: prefer numeric/datetime types over object types
    total_cols = len(cleaned_df.columns)
    if total_cols > 0:
        non_object_cols = len(cleaned_df.select_dtypes(exclude=['object']).columns)
        validity_score = (non_object_cols / total_cols) * 20
    else:
        validity_score = 0

    total_score = completeness_score + consistency_score + validity_score

    return round(total_score, 2)
