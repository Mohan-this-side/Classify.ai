"""
Prompt Templates for Data Cleaning Layer 2 Code Generation

This module contains prompt templates used to generate adaptive cleaning code
via LLM in the double-layer architecture.
"""

CLEANING_PROMPT_TEMPLATE_V1 = """
# Advanced Data Cleaning Code Generation

You are an expert data scientist tasked with generating Python code for advanced data cleaning based on Layer 1 analysis results.

## Layer 1 Analysis Results

### Missing Values:
- Total missing values: {total_missing}
- Columns with missing values: {columns_with_missing}
- Missing percentages: {missing_percentages}
- Pattern analysis: {missing_pattern}

### Outliers:
- Total outliers detected: {total_outliers}
- Columns with outliers: {outlier_columns}
- Outlier percentage: {outlier_percentage}%

### Data Types:
- Type appropriateness score: {type_score}
- Columns needing conversion: {columns_need_conversion}
- Mixed type columns: {mixed_type_columns}

### Data Quality:
- Overall quality score: {quality_score}/100
- Dataset shape: {dataset_shape}
- Duplicate count: {duplicate_count}

## Your Task

Generate advanced Python code to clean this dataset. Your code should:

1. **Advanced Missing Value Imputation**:
   - Use KNN imputation for numeric columns with moderate missing data (10-30%)
   - Use iterative imputation for columns with relationships
   - Use mode imputation for categorical columns
   - Consider MCAR/MAR/MNAR patterns identified in Layer 1

2. **Intelligent Outlier Handling**:
   - Use IQR method for detecting outliers
   - Apply intelligent capping (Winsorization) rather than removal
   - Consider domain-specific constraints
   - Preserve important data patterns

3. **Smart Data Type Conversion**:
   - Convert string numerics to appropriate numeric types
   - Detect and convert date columns
   - Handle mixed-type columns intelligently
   - Optimize memory usage

4. **Data Standardization**:
   - Standardize categorical values (case, spacing)
   - Handle currency and percentage formats
   - Normalize text fields
   - Fix encoding issues

## Requirements

**Allowed Imports:**
```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

**Code Structure:**
```python
def advanced_clean_data(df):
    \"\"\"
    Perform advanced data cleaning.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    \"\"\"
    # Your advanced cleaning logic here
    cleaned_df = df.copy()

    # Step 1: Advanced missing value handling
    # ... your code ...

    # Step 2: Intelligent outlier treatment
    # ... your code ...

    # Step 3: Type optimization
    # ... your code ...

    # Step 4: Data standardization
    # ... your code ...

    return cleaned_df

# Execute the cleaning
result = advanced_clean_data(df)
```

## Important Constraints

1. **Data Integrity**: Never drop rows unless absolutely necessary (>90% missing)
2. **Reproducibility**: Use random_state=42 for any random operations
3. **Performance**: Keep operations efficient, avoid nested loops on large datasets
4. **Safety**: No file I/O, no network calls, no system operations
5. **Return Format**: Must return a pandas DataFrame with the same structure

## Output

Generate complete, executable Python code following the structure above. Include comments explaining your approach for each step.
"""


CLEANING_PROMPT_TEMPLATE_V2 = """
# Task: Generate Advanced Data Cleaning Code

## Context
You are generating data cleaning code based on Layer 1 analysis. The dataset has already been through basic cleaning, but you can apply more sophisticated techniques.

## Dataset Characteristics
- Shape: {dataset_shape}
- Quality Score: {quality_score}/100
- Primary Issues: {primary_issues}

## Layer 1 Findings

### Missing Data ({total_missing} values):
{missing_details}

### Outliers ({total_outliers} detected):
{outlier_details}

### Data Types:
{type_details}

## Code Generation Instructions

Create a function called `advanced_clean_data(df)` that:

1. **Handles missing values intelligently**:
   - For numeric: Use KNN or iterative imputation
   - For categorical: Use mode or create "Unknown" category
   - Preserve data relationships

2. **Treats outliers appropriately**:
   - Detect using IQR or isolation forest
   - Cap instead of remove (use percentiles)
   - Document outlier treatment

3. **Optimizes data types**:
   - Convert to optimal dtypes for memory efficiency
   - Handle datetime conversions
   - Use categorical dtype for low-cardinality strings

4. **Validates results**:
   - Check no new NaNs introduced
   - Verify shape consistency
   - Ensure data ranges are reasonable

## Allowed Libraries
- pandas, numpy
- sklearn.impute (KNNImputer, SimpleImputer, IterativeImputer)
- sklearn.preprocessing
- scipy.stats (for statistical tests)

## Code Template
```python
def advanced_clean_data(df):
    cleaned_df = df.copy()

    # Your advanced cleaning steps here

    return cleaned_df

result = advanced_clean_data(df)
```

Generate the complete function now.
"""


def create_cleaning_prompt(layer1_results: dict) -> str:
    """
    Create a data cleaning prompt from Layer 1 analysis results.

    Args:
        layer1_results: Dictionary containing Layer 1 analysis results

    Returns:
        Formatted prompt string for LLM code generation
    """

    # Extract key metrics
    missing_analysis = layer1_results.get("missing_analysis", {})
    missing_stats = missing_analysis.get("missing_statistics", {})

    outlier_detection = layer1_results.get("outlier_detection", {})
    outlier_summary = outlier_detection.get("outlier_summary", {})

    type_validation = layer1_results.get("type_validation", {})
    quality_assessment = type_validation.get("quality_assessment", {})

    dataset_stats = layer1_results.get("dataset_stats", {})

    # Format missing value details
    missing_details = []
    for col, pct in missing_stats.get("missing_percentages", {}).items():
        if pct > 0:
            missing_details.append(f"  - {col}: {pct:.1f}% missing")
    missing_details_str = "\n".join(missing_details) if missing_details else "  No missing values"

    # Format outlier details
    outlier_columns = []
    column_analysis = outlier_detection.get("column_analysis", {})
    for col, analysis in column_analysis.items():
        if analysis.get("has_outliers", False):
            outlier_columns.append(f"  - {col}: {analysis.get('outlier_count', 0)} outliers")
    outlier_details_str = "\n".join(outlier_columns) if outlier_columns else "  No outliers detected"

    # Format type details
    type_analysis = type_validation.get("type_analysis", {})
    type_details = []
    for col, info in type_analysis.get("column_types", {}).items():
        if info.get("confidence", 1.0) < 0.9:
            type_details.append(f"  - {col}: Current={info.get('current_dtype')}, Suggested={info.get('detected_type')}")
    type_details_str = "\n".join(type_details) if type_details else "  All types are appropriate"

    # Identify primary issues
    primary_issues = []
    if missing_stats.get("total_missing", 0) > 0:
        primary_issues.append(f"Missing values: {missing_stats['total_missing']}")
    if outlier_summary.get("total_outliers_detected", 0) > 0:
        primary_issues.append(f"Outliers: {outlier_summary['total_outliers_detected']}")
    if quality_assessment.get("columns_with_issues", 0) > 0:
        primary_issues.append(f"Type issues: {quality_assessment['columns_with_issues']} columns")
    primary_issues_str = ", ".join(primary_issues) if primary_issues else "No major issues"

    # Use template V2 (more concise)
    prompt = CLEANING_PROMPT_TEMPLATE_V2.format(
        dataset_shape=dataset_stats.get("cleaned_shape", "Unknown"),
        quality_score=layer1_results.get("data_quality_score", 0),
        primary_issues=primary_issues_str,
        total_missing=missing_stats.get("total_missing", 0),
        missing_details=missing_details_str,
        total_outliers=outlier_summary.get("total_outliers_detected", 0),
        outlier_details=outlier_details_str,
        type_details=type_details_str
    )

    return prompt
