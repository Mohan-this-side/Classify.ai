"""
Prompt Templates for Data Analysis Agents (Discovery and EDA)

These templates are used to generate code via LLM for Layer 2 analysis.
"""

# Version tracking
PROMPT_VERSION = "1.0.0"

DISCOVERY_PROMPT_TEMPLATE = """
You are an expert data scientist analyzing a dataset. Based on the Layer 1 statistical profiling results below, generate Python code to perform ADVANCED analysis that goes beyond basic statistics.

## Layer 1 Analysis Results:

### Basic Information:
- Dataset shape: {shape}
- Total columns: {num_columns}
- Memory usage: {memory_usage_mb:.2f} MB

### Statistical Summary:
{statistical_summary}

### Data Types and Cardinality:
{data_types_summary}

### Correlations (Top 5 pairs):
{top_correlations}

### Missing Value Patterns:
{missing_patterns_summary}

### Detected Column Types:
- ID Columns: {id_columns}
- Datetime Columns: {datetime_columns}
- Categorical Columns: {categorical_columns}
- Continuous Columns: {continuous_columns}

## Your Task:

Generate Python code that performs ADVANCED analysis including:

1. **Normality Testing**: Test numeric columns for normal distribution using Shapiro-Wilk or Anderson-Darling tests
2. **Stationarity Testing**: For time-series-like data, test for stationarity (ADF test)
3. **Pattern Detection**: Identify seasonality, trends, or cyclic patterns in numeric data
4. **Anomaly Detection**: Use statistical methods (Z-score, IQR, Isolation Forest) to detect outliers
5. **Feature Relationships**: Analyze non-linear relationships using mutual information or other methods
6. **Preprocessing Recommendations**: Based on data characteristics, recommend specific transformations
7. **Domain-Specific Insights**: If data patterns suggest specific domains (finance, healthcare, etc.), provide relevant insights

## Code Requirements:

- Use only these imports: pandas, numpy, scipy, sklearn
- Code must be self-contained and executable
- Return results as a dictionary with keys: "normality_tests", "stationarity_tests", "patterns", "anomalies", "relationships", "recommendations", "domain_insights"
- Handle errors gracefully
- Do not use any file I/O, network operations, or system calls
- Assume the DataFrame is available as variable `df`

## Code Structure:

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression

def advanced_discovery_analysis(df):
    results = {{
        "normality_tests": {{}},
        "stationarity_tests": {{}},
        "patterns": {{}},
        "anomalies": {{}},
        "relationships": {{}},
        "recommendations": [],
        "domain_insights": []
    }}

    # Your advanced analysis code here

    return results

# Execute
result = advanced_discovery_analysis(df)
```

Generate the complete implementation now. Focus on providing actionable insights.
"""

EDA_PROMPT_TEMPLATE = """
You are an expert data scientist performing Exploratory Data Analysis (EDA). Based on the Layer 1 statistical analysis below, generate Python code to create ADVANCED visualizations and insights.

## Layer 1 Analysis Results:

### Statistical Summary:
{statistical_summary}

### Correlations:
{correlations_summary}

### Distribution Statistics:
{distribution_stats}

### Target Variable Relationship:
{target_relationship}

## Your Task:

Generate Python code that creates ADVANCED visualizations including:

1. **Interactive Correlation Heatmap**: With p-values and significance indicators
2. **Distribution Analysis**: Overlaid histograms with KDE, box plots, violin plots
3. **Pair Plots**: For top correlated features with target
4. **Outlier Visualization**: Box plots with outlier annotations
5. **Feature Importance Plot**: Using multiple methods
6. **Dimensionality Reduction**: PCA/t-SNE visualizations if applicable

## Code Requirements:

- Use matplotlib, seaborn, or plotly for visualizations
- Create 5-7 high-quality plots
- Save plots to files: plot_1.png, plot_2.png, etc.
- Return plot metadata as dictionary
- Do not use file I/O except for plot saving
- Assume DataFrame is available as `df`

Generate the complete visualization code now.
"""

# Alias for backward compatibility
EDA_VISUALIZATION_PROMPT_TEMPLATE = EDA_PROMPT_TEMPLATE


def get_discovery_prompt(layer1_results: dict) -> str:
    """
    Generate discovery analysis prompt from Layer 1 results.

    Args:
        layer1_results: Layer 1 profiling results

    Returns:
        Formatted prompt string
    """
    basic_info = layer1_results.get("basic_info", {})
    stat_summary = layer1_results.get("statistical_summary", {})
    data_types = layer1_results.get("data_types", {})
    correlations = layer1_results.get("correlations", {})
    missing_patterns = layer1_results.get("missing_patterns", {})
    detected_cols = layer1_results.get("detected_columns", {})

    # Format statistical summary
    stat_summary_str = ""
    for col, stats in list(stat_summary.items())[:5]:  # Show top 5
        stat_summary_str += f"\n  {col}:"
        stat_summary_str += f"\n    Mean: {stats.get('mean', 0):.2f}"
        stat_summary_str += f"\n    Std: {stats.get('std', 0):.2f}"
        stat_summary_str += f"\n    Min: {stats.get('min', 0):.2f}"
        stat_summary_str += f"\n    Max: {stats.get('max', 0):.2f}"

    # Format data types summary
    data_types_summary_str = ""
    for col, info in list(data_types.items())[:10]:  # Show top 10
        data_types_summary_str += f"\n  {col}: {info.get('dtype', 'unknown')} (cardinality: {info.get('cardinality', 0)})"

    # Format top correlations
    top_corr_str = "No strong correlations found"
    if correlations:
        corr_pairs = []
        cols = list(correlations.keys())
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                if col1 in correlations and col2 in correlations[col1]:
                    corr_val = correlations[col1][col2]
                    if abs(corr_val) > 0.5:
                        corr_pairs.append((col1, col2, corr_val))

        if corr_pairs:
            top_corr_str = "\n".join([f"  {c1} <-> {c2}: {val:.3f}" for c1, c2, val in corr_pairs[:5]])

    # Format missing patterns
    missing_summary_str = ""
    cols_with_missing = [(col, info) for col, info in missing_patterns.items() if info.get("count", 0) > 0]
    if cols_with_missing:
        for col, info in cols_with_missing[:5]:
            missing_summary_str += f"\n  {col}: {info.get('percentage', 0):.1f}% ({info.get('count', 0)} values)"
    else:
        missing_summary_str = "  No missing values detected"

    return DISCOVERY_PROMPT_TEMPLATE.format(
        shape=basic_info.get("shape", (0, 0)),
        num_columns=len(basic_info.get("columns", [])),
        memory_usage_mb=basic_info.get("memory_usage_mb", 0),
        statistical_summary=stat_summary_str or "  No numeric columns",
        data_types_summary=data_types_summary_str or "  No columns",
        top_correlations=top_corr_str,
        missing_patterns_summary=missing_summary_str,
        id_columns=", ".join(detected_cols.get("id_columns", [])) or "None",
        datetime_columns=", ".join(detected_cols.get("datetime_columns", [])) or "None",
        categorical_columns=", ".join(detected_cols.get("categorical_columns", []))[:100] or "None",
        continuous_columns=", ".join(detected_cols.get("continuous_columns", []))[:100] or "None"
    )
