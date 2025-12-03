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
3. **Imports**: Use ONLY these imports: pandas, numpy, scipy, sklearn, matplotlib, seaborn
4. **No markdown**: Return ONLY Python code, NO markdown code fences (```python or ```)
5. **No comments outside code**: Do not include explanatory text outside code blocks
6. **Self-contained**: Code must be executable without modification
7. **Error handling**: Wrap risky operations in try-except blocks
8. **Output format**: Return results as a dictionary with keys: "normality_tests", "stationarity_tests", "patterns", "anomalies", "relationships", "recommendations", "domain_insights"
9. **DataFrame variable**: Load DataFrame from '/app/data/dataset' (CSV file in Docker sandbox)
10. **Docker Environment**: Code runs in Docker with read-only filesystem. Save plots to '/app/results/' directory ONLY
11. **Matplotlib Backend**: Use 'Agg' backend for matplotlib (required in Docker): `import matplotlib; matplotlib.use('Agg')`
12. **Print Results**: MUST print the results dictionary at the end so output can be captured

## EXACT Code Structure (copy this format exactly):

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Required for Docker (no display)
import matplotlib.pyplot as plt
import seaborn as sns

# CRITICAL: Load the dataset from the sandbox data volume
# Dataset is provided as CSV file at /app/data/dataset
try:
    df = pd.read_csv('/app/data/dataset')
    if df.empty:
        raise ValueError("Dataset is empty")
except (FileNotFoundError, ValueError) as e:
    # If dataset not available, return empty results
    print({{"error": f"Could not load dataset: {{str(e)}}"}})
    exit(1)

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

# CRITICAL: Print result as JSON string so it can be captured from stdout
import json
print(json.dumps(result, default=str))

# IMPORTANT: All plots must be saved to /app/results/ directory
# Example: plt.savefig('/app/results/my_plot.png')

Generate ONLY the Python code following the exact structure above. Start with imports at column 0.
"""

EDA_PROMPT_TEMPLATE = """
You are an expert data scientist performing Exploratory Data Analysis (EDA). Generate EXECUTABLE Python code to create MEANINGFUL, DATASET-SPECIFIC visualizations that provide actionable insights.

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

Generate EXECUTABLE Python code that creates MEANINGFUL visualizations SPECIFIC to this dataset. Focus on:
1. **Understanding the target variable** - How do features relate to the target?
2. **Identifying key patterns** - What features are most important?
3. **Data quality insights** - Are there outliers, imbalances, or data issues?
4. **Actionable insights** - What can we learn to improve predictions?

Create DIFFERENT types of visualizations with DESCRIPTIVE filenames:

1. **Correlation Heatmap**: Show feature correlations - save as 'correlation_heatmap.png'
2. **Target Distribution**: Show target variable distribution - save as 'target_distribution.png'
3. **Feature vs Target Analysis**: Box plots or violin plots comparing features by target - save as 'feature_target_analysis.png'
4. **Key Feature Distributions**: Histograms of most important numeric features - save as 'key_feature_distributions.png'
5. **Categorical Feature Analysis**: If categorical features exist, show their relationship to target - save as 'categorical_target_analysis.png'
6. **Outlier Detection**: Visualize outliers in key features - save as 'outlier_analysis.png'
7. **Feature Importance**: If available, show feature importance - save as 'feature_importance.png'

## STRICT Code Requirements:

1. **Syntax**: Code MUST be syntactically correct Python 3.11
2. **Indentation**: Use exactly 4 spaces for indentation (NO tabs, NO mixed indentation)
3. **Imports**: Use ONLY: matplotlib, matplotlib.pyplot, seaborn, pandas, numpy
4. **No markdown**: Return ONLY Python code, NO markdown code fences (```python or ```)
5. **No comments outside code**: Do not include explanatory text outside code blocks
6. **Plot saving**: Save each plot to '/app/results/DESCRIPTIVE_NAME.png' (e.g., 'correlation_heatmap.png', NOT 'plot_1.png')
7. **Meaningful plots**: Create visualizations that provide INSIGHTS about THIS SPECIFIC DATASET
8. **Target-focused**: Prioritize plots that show relationships with the target variable
9. **Error handling**: Wrap each plot generation in try-except
10. **DataFrame loading**: You MUST load the dataset using `df = pd.read_csv('/app/data/dataset')` at the start
11. **Target column**: Use the target column from the dataset - identify it from column names or Layer 1 results
12. **Clear titles**: Each plot MUST have a clear, descriptive title explaining what it shows
13. **Output**: Print a dictionary with plot metadata at the end

## EXACT Code Structure (copy this format exactly):

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# CRITICAL: Load the dataset from the sandbox data volume
df = pd.read_csv('/app/data/dataset')

# Identify target column (look for common target column names or use the last column)
target_col = None
possible_targets = ['target', 'label', 'class', 'outcome', 'result', 'HeartDisease', 'heart_disease']
for col in df.columns:
    if col.lower() in [p.lower() for p in possible_targets] or col == df.columns[-1]:
        target_col = col
        break

# Set style
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

plot_metadata = {{}}

try:
    # Plot 1: Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/app/results/correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        plot_metadata['correlation_heatmap'] = {{'type': 'correlation_heatmap', 'status': 'success', 'filename': 'correlation_heatmap.png'}}
except Exception as e:
    plot_metadata['correlation_heatmap'] = {{'type': 'correlation_heatmap', 'status': 'failed', 'error': str(e)}}

try:
    # Plot 2: Target Distribution
    if target_col and target_col in df.columns:
        plt.figure(figsize=(10, 6))
        target_counts = df[target_col].value_counts().sort_index()
        plt.bar(target_counts.index.astype(str), target_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of Target Variable: {{target_col}}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        for i, v in enumerate(target_counts.values):
            plt.text(i, v + max(target_counts.values)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig('/app/results/target_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        plot_metadata['target_distribution'] = {{'type': 'target_distribution', 'status': 'success', 'filename': 'target_distribution.png'}}
except Exception as e:
    plot_metadata['target_distribution'] = {{'type': 'target_distribution', 'status': 'failed', 'error': str(e)}}

try:
    # Plot 3: Feature vs Target Analysis (Box plots for numeric features)
    if target_col and target_col in df.columns and len(numeric_cols) > 0:
        # Select top 4-6 numeric features (excluding target if numeric)
        feature_cols = [col for col in numeric_cols if col != target_col][:6]
        if len(feature_cols) > 0:
            n_features = len(feature_cols)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, col in enumerate(feature_cols):
                if idx < len(axes):
                    sns.boxplot(data=df, x=target_col, y=col, ax=axes[idx])
                    axes[idx].set_title(f'{{col}} by {{target_col}}', fontsize=11, fontweight='bold')
                    axes[idx].set_xlabel(target_col, fontsize=10)
                    axes[idx].set_ylabel(col, fontsize=10)
            
            # Hide extra subplots
            for idx in range(len(feature_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Feature Distributions by Target Variable', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig('/app/results/feature_target_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            plot_metadata['feature_target_analysis'] = {{'type': 'feature_target_analysis', 'status': 'success', 'filename': 'feature_target_analysis.png'}}
except Exception as e:
    plot_metadata['feature_target_analysis'] = {{'type': 'feature_target_analysis', 'status': 'failed', 'error': str(e)}}

# Add more meaningful plots following the same pattern...
# Focus on creating visualizations that provide insights about THIS dataset

# Print metadata
print(plot_metadata)

Generate ONLY the Python code following the exact structure above. Start with imports at column 0. 
Create MEANINGFUL, DATASET-SPECIFIC visualizations with DESCRIPTIVE filenames (NOT plot_1.png, plot_2.png).
Focus on plots that show relationships with the target variable and provide actionable insights.
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
