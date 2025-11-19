"""
Prompt Templates for Data Analysis Agents (Discovery and EDA)

These templates are used to generate code via LLM for Layer 2 analysis.
"""

# Version tracking
PROMPT_VERSION = "1.0.0"

DISCOVERY_PROMPT_TEMPLATE = """
You are an expert data scientist analyzing a dataset. Generate EXECUTABLE Python code for advanced analysis.

CRITICAL: Your code MUST be syntactically correct and executable. Use proper indentation (4 spaces, no tabs). Start all code at column 0 (no leading spaces before imports).

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

Generate EXECUTABLE Python code that performs ADVANCED analysis including:

1. **Normality Testing**: Test numeric columns for normal distribution using Shapiro-Wilk or Anderson-Darling tests
2. **Stationarity Testing**: For time-series-like data, test for stationarity (ADF test)
3. **Pattern Detection**: Identify seasonality, trends, or cyclic patterns in numeric data
4. **Anomaly Detection**: Use statistical methods (Z-score, IQR, Isolation Forest) to detect outliers
5. **Feature Relationships**: Analyze non-linear relationships using mutual information or other methods
6. **Preprocessing Recommendations**: Based on data characteristics, recommend specific transformations
7. **Domain-Specific Insights**: If data patterns suggest specific domains (finance, healthcare, etc.), provide relevant insights

## STRICT Code Requirements:

1. **Syntax**: Code MUST be syntactically correct Python 3.11
2. **Indentation**: Use exactly 4 spaces for indentation (NO tabs, NO mixed indentation)
3. **Imports**: Use ONLY these imports: pandas, numpy, scipy, sklearn
4. **No markdown**: Return ONLY Python code, NO markdown code fences (```python or ```)
5. **No comments outside code**: Do not include explanatory text outside code blocks
6. **Self-contained**: Code must be executable without modification
7. **Error handling**: Wrap risky operations in try-except blocks
8. **Output format**: Return results as a dictionary with keys: "normality_tests", "stationarity_tests", "patterns", "anomalies", "relationships", "recommendations", "domain_insights"
9. **DataFrame variable**: Assume DataFrame is available as variable `df`
10. **No file I/O**: Do not use file operations except for saving plots (if needed)

## EXACT Code Structure (copy this format exactly):

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression

# CRITICAL: Load the dataset from the sandbox data volume (if dataset is provided)
try:
    df = pd.read_csv('/app/data/dataset')
except FileNotFoundError:
    # If dataset not provided, create empty DataFrame
    df = pd.DataFrame()

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
    
    try:
        # Your advanced analysis code here
        # Test normality for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            try:
                stat, p_value = stats.shapiro(df[col].dropna())
                results["normality_tests"][col] = {{"statistic": stat, "p_value": p_value}}
            except Exception as e:
                results["normality_tests"][col] = {{"error": str(e)}}
        
        # Add more analysis here
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

# Execute the function
result = advanced_discovery_analysis(df)

# IMPORTANT: Print or return the result so it can be captured
print(result)

Generate ONLY the Python code following the exact structure above. Start with imports at column 0.
"""

EDA_PROMPT_TEMPLATE = """
You are an expert data scientist performing Exploratory Data Analysis (EDA). Generate EXECUTABLE Python code to create DIFFERENT types of visualizations.

CRITICAL: Your code MUST be syntactically correct and executable. Use proper indentation (4 spaces, no tabs). Start all code at column 0 (no leading spaces before imports).

## User's Dataset Description:
{user_description}

## Layer 1 Analysis Results:

### Statistical Summary:
{statistical_summary}

### Correlations:
{correlations_summary}

### Distribution Statistics:
{distributions_summary}

### Target Variable Relationship:
{target_relationships_summary}

## Your Task:

Generate EXECUTABLE Python code that creates DIFFERENT types of visualizations:

1. **Correlation Heatmap**: Use seaborn.heatmap() - save as 'plot_1.png'
2. **Distribution Histogram**: Use matplotlib.pyplot.hist() or seaborn.histplot() - save as 'plot_2.png'
3. **Box Plot**: Use seaborn.boxplot() - save as 'plot_3.png'
4. **Scatter Plot**: Use matplotlib.pyplot.scatter() - save as 'plot_4.png'
5. **Violin Plot**: Use seaborn.violinplot() - save as 'plot_5.png'
6. **Pair Plot**: Use seaborn.pairplot() - save as 'plot_6.png' (if applicable)
7. **Feature Importance Bar Chart**: Use matplotlib.pyplot.barh() - save as 'plot_7.png'

## STRICT Code Requirements:

1. **Syntax**: Code MUST be syntactically correct Python 3.11
2. **Indentation**: Use exactly 4 spaces for indentation (NO tabs, NO mixed indentation)
3. **Imports**: Use ONLY: matplotlib, matplotlib.pyplot, seaborn, pandas, numpy
4. **No markdown**: Return ONLY Python code, NO markdown code fences (```python or ```)
5. **No comments outside code**: Do not include explanatory text outside code blocks
6. **Plot saving**: Save each plot to '/app/results/plot_N.png' where N is 1-7
7. **Different plot types**: Create DIFFERENT visualization types (not all the same)
8. **Error handling**: Wrap each plot generation in try-except
9. **DataFrame loading**: You MUST load the dataset using `df = pd.read_csv('/app/data/dataset')` at the start of your code
10. **Target column**: Target column is available as variable `target_col` if provided (check Layer 1 results)
11. **Output**: Print a dictionary with plot metadata at the end

## EXACT Code Structure (copy this format exactly):

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# CRITICAL: Load the dataset from the sandbox data volume
df = pd.read_csv('/app/data/dataset')

# Set style
plt.style.use('default')
sns.set_palette("husl")

plot_metadata = {{}}

try:
    # Plot 1: Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('/app/results/plot_1.png', dpi=150, bbox_inches='tight')
        plt.close()
        plot_metadata['plot_1'] = {{'type': 'correlation_heatmap', 'status': 'success'}}
except Exception as e:
    plot_metadata['plot_1'] = {{'type': 'correlation_heatmap', 'status': 'failed', 'error': str(e)}}

try:
    # Plot 2: Distribution Histogram
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:4]):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
                axes[i].set_title(f'Distribution of {{col}}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('/app/results/plot_2.png', dpi=150, bbox_inches='tight')
        plt.close()
        plot_metadata['plot_2'] = {{'type': 'distribution_histogram', 'status': 'success'}}
except Exception as e:
    plot_metadata['plot_2'] = {{'type': 'distribution_histogram', 'status': 'failed', 'error': str(e)}}

# Add more plots following the same pattern...

# Print metadata
print(plot_metadata)

Generate ONLY the Python code following the exact structure above. Start with imports at column 0. Create DIFFERENT plot types.
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
