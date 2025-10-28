"""
🔎 Enhanced Data Discovery Agent with Double-Layer Architecture

Comprehensive data profiling, automatic data type detection, and intelligent feature recommendations.
Provides detailed insights to guide downstream data cleaning and feature engineering steps.

DOUBLE-LAYER ARCHITECTURE:
- Layer 1: Reliable statistical profiling using pandas and numpy
- Layer 2: LLM-generated advanced analysis code with domain-specific insights
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..base_agent import BaseAgent
from ...workflows.state_management import ClassificationState, AgentStatus, state_manager


class DataDiscoveryAgent(BaseAgent):
    """Enhanced Data Discovery Agent with Double-Layer Architecture"""

    def __init__(self) -> None:
        super().__init__(
            agent_name="data_discovery",
            agent_version="3.0.0",
            enable_layer2=True,
            sandbox_timeout=90
        )
        self.logger = logging.getLogger("agent.data_discovery")

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": "Comprehensive data profiling with double-layer architecture",
            "capabilities": [
                "Layer 1: Statistical profiling and correlation analysis",
                "Layer 2: Advanced pattern detection and domain insights",
                "Automatic data type detection",
                "Intelligent feature recommendations",
                "Data quality assessment",
                "Anomaly and seasonality detection"
            ],
            "dependencies": ["data_cleaning"],
            "supports_layer2": True
        }

    def get_dependencies(self) -> list:
        return ["data_cleaning"]

    async def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """
        LAYER 1: Comprehensive statistical profiling using pandas and numpy.

        This provides reliable baseline profiling that includes:
        - Statistical summaries (mean, median, std, quartiles)
        - Data type detection and cardinality analysis
        - Correlation matrix calculation
        - Missing value patterns
        - ID column detection (high cardinality)
        - Date/time column detection

        Args:
            state: Current workflow state

        Returns:
            Dictionary with Layer 1 profiling results
        """
        try:
            self.logger.info("LAYER 1: Performing reliable statistical profiling")

            # Access cleaned dataset via state manager
            df = state_manager.get_dataset(state, "cleaned")
            if df is None:
                df = state_manager.get_dataset(state, "original")

            if df is None:
                raise ValueError("No dataset available for discovery")

            # 1. Basic information
            basic_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            }

            # 2. Statistical summaries
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            statistical_summary = {}

            if numeric_cols:
                desc = df[numeric_cols].describe()
                statistical_summary = {
                    col: {
                        "count": desc.loc["count", col],
                        "mean": desc.loc["mean", col],
                        "std": desc.loc["std", col],
                        "min": desc.loc["min", col],
                        "25%": desc.loc["25%", col],
                        "50%": desc.loc["50%", col],
                        "75%": desc.loc["75%", col],
                        "max": desc.loc["max", col]
                    }
                    for col in numeric_cols
                }

            # 3. Data type detection and cardinality
            data_types_info = {}
            for col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df)

                data_types_info[col] = {
                    "dtype": str(df[col].dtype),
                    "cardinality": unique_count,
                    "cardinality_ratio": unique_count / total_count if total_count > 0 else 0,
                    "is_potential_id": unique_count == total_count and total_count > 0,
                    "is_potential_categorical": unique_count < 20 and unique_count / total_count < 0.1,
                    "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                    "is_datetime": pd.api.types.is_datetime64_any_dtype(df[col])
                }

            # 4. Correlation matrix (numeric columns only)
            correlations = {}
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                correlations = corr_matrix.to_dict()

            # 5. Missing value patterns
            missing_patterns = {
                col: {
                    "count": int(df[col].isnull().sum()),
                    "percentage": float(df[col].isnull().sum() / len(df) * 100)
                }
                for col in df.columns
            }

            # 6. Detect potential ID columns
            id_columns = [
                col for col, info in data_types_info.items()
                if info["is_potential_id"] or (info["cardinality_ratio"] > 0.95 and info["cardinality"] > 100)
            ]

            # 7. Detect datetime columns
            datetime_columns = [
                col for col, info in data_types_info.items()
                if info["is_datetime"] or self._is_potential_datetime(df[col])
            ]

            # 8. Categorical vs continuous detection
            categorical_columns = [
                col for col, info in data_types_info.items()
                if info["is_potential_categorical"]
            ]

            continuous_columns = [
                col for col in numeric_cols
                if col not in categorical_columns and col not in id_columns
            ]

            layer1_results = {
                "basic_info": basic_info,
                "statistical_summary": statistical_summary,
                "data_types": data_types_info,
                "correlations": correlations,
                "missing_patterns": missing_patterns,
                "detected_columns": {
                    "id_columns": id_columns,
                    "datetime_columns": datetime_columns,
                    "categorical_columns": categorical_columns,
                    "continuous_columns": continuous_columns,
                    "numeric_columns": numeric_cols
                },
                "layer": "layer1",
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Layer 1 profiling complete: {len(df)} rows, {len(df.columns)} columns")
            return layer1_results

        except Exception as e:
            self.logger.error(f"Layer 1 analysis failed: {e}")
            raise

    def _comprehensive_data_profiling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data profiling"""
        try:
            profile = {
                "basic_info": {
                    "shape": df.shape,
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "dtypes": df.dtypes.to_dict(),
                    "columns": list(df.columns)
                },
                "statistical_summary": {},
                "data_quality_metrics": {},
                "distribution_analysis": {},
                "correlation_analysis": {}
            }

            # Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                profile["statistical_summary"]["numeric"] = df[numeric_cols].describe().to_dict()
            
            # Statistical summary for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                profile["statistical_summary"]["categorical"] = {}
                for col in categorical_cols:
                    profile["statistical_summary"]["categorical"][col] = {
                        "unique_count": df[col].nunique(),
                        "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                        "frequency": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                    }

            # Data quality metrics
            profile["data_quality_metrics"] = {
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df) * 100)
            }

            # Distribution analysis
            profile["distribution_analysis"] = self._analyze_distributions(df, numeric_cols, categorical_cols)

            # Correlation analysis for numeric columns
            if len(numeric_cols) > 1:
                profile["correlation_analysis"] = {
                    "correlation_matrix": df[numeric_cols].corr().to_dict(),
                    "high_correlations": self._find_high_correlations(df[numeric_cols])
                }

            return profile

        except Exception as e:
            self.logger.error(f"Error in data profiling: {e}")
            return {"error": str(e)}

    def _automatic_data_type_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Automatically detect and validate data types"""
        try:
            detected_types = {}
            confidence_scores = {}
            
            for col in df.columns:
                detected_type, confidence = self._detect_column_type(df[col])
                detected_types[col] = detected_type
                confidence_scores[col] = confidence

            return {
                "detected_types": detected_types,
                "confidence_scores": confidence_scores,
                "recommendations": self._generate_type_recommendations(detected_types, confidence_scores)
            }

        except Exception as e:
            self.logger.error(f"Error in data type detection: {e}")
            return {"error": str(e)}

    def _detect_column_type(self, series: pd.Series) -> Tuple[str, float]:
        """Detect the most likely data type for a column"""
        try:
            # Check for missing values
            null_count = series.isnull().sum()
            non_null_series = series.dropna()
            
            if len(non_null_series) == 0:
                return "unknown", 0.0

            # Check for numeric type
            if pd.api.types.is_numeric_dtype(series):
                # Check if it's integer or float
                if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                    return "integer", 0.9
                else:
                    return "float", 0.9

            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                return "datetime", 0.9

            # Check for boolean
            if series.dtype == 'bool' or series.dtype.name == 'bool':
                return "boolean", 0.9

            # Check for categorical (string with limited unique values)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:  # Less than 10% unique values
                return "categorical", 0.8
            elif unique_ratio < 0.5:  # Less than 50% unique values
                return "categorical", 0.6

            # Check if it's text
            if series.dtype == 'object':
                # Check if all values are strings
                if all(isinstance(x, str) for x in non_null_series.head(100)):
                    return "text", 0.7

            # Default to categorical for object types
            return "categorical", 0.5

        except Exception as e:
            self.logger.error(f"Error detecting type for column {series.name}: {e}")
            return "unknown", 0.0

    def _generate_feature_recommendations(self, df: pd.DataFrame, data_types: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent feature recommendations"""
        try:
            recommendations = {
                "feature_engineering": [],
                "feature_selection": [],
                "data_preprocessing": [],
                "model_recommendations": []
            }

            detected_types = data_types.get("detected_types", {})
            numeric_cols = [col for col, dtype in detected_types.items() if dtype in ["integer", "float"]]
            categorical_cols = [col for col, dtype in detected_types.items() if dtype == "categorical"]
            text_cols = [col for col, dtype in detected_types.items() if dtype == "text"]

            # Feature engineering recommendations
            if len(numeric_cols) > 1:
                recommendations["feature_engineering"].append(
                    "Consider creating interaction features between numeric variables"
                )
                recommendations["feature_engineering"].append(
                    "Consider polynomial features for non-linear relationships"
                )

            if len(categorical_cols) > 0:
                recommendations["feature_engineering"].append(
                    "Apply one-hot encoding or target encoding for categorical variables"
                )

            if len(text_cols) > 0:
                recommendations["feature_engineering"].append(
                    "Consider text preprocessing and feature extraction (TF-IDF, word embeddings)"
                )

            # Feature selection recommendations
            if len(numeric_cols) > 5:
                recommendations["feature_selection"].append(
                    "Consider feature selection techniques to reduce dimensionality"
                )

            # Data preprocessing recommendations
            if len(numeric_cols) > 0:
                recommendations["data_preprocessing"].append(
                    "Apply feature scaling for numeric variables"
                )

            # Model recommendations based on data characteristics
            n_samples, n_features = df.shape
            if n_samples < 1000:
                recommendations["model_recommendations"].append(
                    "Small dataset: consider simple models (Logistic Regression, Naive Bayes)"
                )
            elif n_features > n_samples:
                recommendations["model_recommendations"].append(
                    "High-dimensional data: consider regularization (Ridge, Lasso) or dimensionality reduction"
                )
            else:
                recommendations["model_recommendations"].append(
                    "Consider ensemble methods (Random Forest, Gradient Boosting) for better performance"
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating feature recommendations: {e}")
            return {"error": str(e)}

    def _generate_data_visualizations(self, df: pd.DataFrame, data_types: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data visualization code and recommendations"""
        try:
            visualizations = {
                "plot_code": "",
                "recommended_plots": [],
                "plot_descriptions": {}
            }

            detected_types = data_types.get("detected_types", {})
            numeric_cols = [col for col, dtype in detected_types.items() if dtype in ["integer", "float"]]
            categorical_cols = [col for col, dtype in detected_types.items() if dtype == "categorical"]

            # Generate visualization code
            plot_code = "import matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\n"
            plot_code += "# Set style\nplt.style.use('seaborn-v0_8')\n\n"

            # Numeric columns visualizations
            if len(numeric_cols) > 0:
                plot_code += "# Numeric columns analysis\n"
                plot_code += f"numeric_cols = {numeric_cols}\n\n"
                
                plot_code += "# Distribution plots\n"
                plot_code += "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n"
                plot_code += "axes = axes.ravel()\n\n"
                plot_code += "for i, col in enumerate(numeric_cols[:4]):\n"
                plot_code += "    if i < 4:\n"
                plot_code += "        df[col].hist(bins=30, ax=axes[i], alpha=0.7)\n"
                plot_code += "        axes[i].set_title(f'Distribution of {col}')\n"
                plot_code += "        axes[i].set_xlabel(col)\n"
                plot_code += "        axes[i].set_ylabel('Frequency')\n\n"
                plot_code += "plt.tight_layout()\nplt.show()\n\n"

                visualizations["recommended_plots"].extend([
                    "Distribution plots for numeric variables",
                    "Correlation heatmap",
                    "Box plots for outlier detection"
                ])

            # Categorical columns visualizations
            if len(categorical_cols) > 0:
                plot_code += "# Categorical columns analysis\n"
                plot_code += f"categorical_cols = {categorical_cols}\n\n"
                
                plot_code += "# Bar plots for categorical variables\n"
                plot_code += "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n"
                plot_code += "axes = axes.ravel()\n\n"
                plot_code += "for i, col in enumerate(categorical_cols[:4]):\n"
                plot_code += "    if i < 4:\n"
                plot_code += "        df[col].value_counts().head(10).plot(kind='bar', ax=axes[i])\n"
                plot_code += "        axes[i].set_title(f'Top 10 values in {col}')\n"
                plot_code += "        axes[i].set_xlabel(col)\n"
                plot_code += "        axes[i].set_ylabel('Count')\n"
                plot_code += "        axes[i].tick_params(axis='x', rotation=45)\n\n"
                plot_code += "plt.tight_layout()\nplt.show()\n\n"

                visualizations["recommended_plots"].extend([
                    "Bar plots for categorical variables",
                    "Pie charts for categorical distributions"
                ])

            visualizations["plot_code"] = plot_code
            return visualizations

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            return {"error": str(e)}

    def _assess_data_quality(self, df: pd.DataFrame, data_types: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality"""
        try:
            quality_metrics = {
                "completeness": {},
                "consistency": {},
                "accuracy": {},
                "overall_score": 0.0
            }

            # Completeness assessment
            missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
            quality_metrics["completeness"] = {
                "missing_percentage": missing_percentage,
                "average_completeness": 100 - np.mean(list(missing_percentage.values())),
                "columns_with_missing": [col for col, pct in missing_percentage.items() if pct > 0]
            }

            # Consistency assessment
            duplicate_percentage = (df.duplicated().sum() / len(df) * 100)
            quality_metrics["consistency"] = {
                "duplicate_percentage": duplicate_percentage,
                "duplicate_rows": df.duplicated().sum()
            }

            # Calculate overall quality score
            completeness_score = quality_metrics["completeness"]["average_completeness"] / 100
            consistency_score = max(0, 1 - duplicate_percentage / 100)
            overall_score = (completeness_score + consistency_score) / 2 * 100

            quality_metrics["overall_score"] = round(overall_score, 2)

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return {"error": str(e)}

    def _detect_patterns_and_anomalies(self, df: pd.DataFrame, data_types: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns and anomalies in the data"""
        try:
            patterns = {
                "outliers": {},
                "seasonality": {},
                "trends": {},
                "anomalies": []
            }

            detected_types = data_types.get("detected_types", {})
            numeric_cols = [col for col, dtype in detected_types.items() if dtype in ["integer", "float"]]

            # Detect outliers in numeric columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                patterns["outliers"][col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100,
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                }

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return {"error": str(e)}

    def _generate_comprehensive_recommendations(self, data_profile: Dict, data_types: Dict, 
                                              feature_recommendations: Dict, data_quality: Dict, 
                                              patterns: Dict) -> List[str]:
        """Generate comprehensive recommendations based on all analysis"""
        try:
            recommendations = []

            # Data quality recommendations
            if data_quality.get("overall_score", 0) < 70:
                recommendations.append("Data quality is below 70% - focus on data cleaning and validation")
            
            if data_quality.get("completeness", {}).get("columns_with_missing"):
                recommendations.append("Address missing values in identified columns")

            # Feature engineering recommendations
            recommendations.extend(feature_recommendations.get("feature_engineering", []))

            # Model recommendations
            recommendations.extend(feature_recommendations.get("model_recommendations", []))

            # Outlier recommendations
            outlier_cols = [col for col, info in patterns.get("outliers", {}).items() 
                          if info.get("percentage", 0) > 5]
            if outlier_cols:
                recommendations.append(f"Consider outlier treatment for columns: {', '.join(outlier_cols)}")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

    def _analyze_distributions(self, df: pd.DataFrame, numeric_cols: List[str], 
                             categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze data distributions"""
        try:
            distribution_analysis = {
                "numeric_distributions": {},
                "categorical_distributions": {}
            }

            # Analyze numeric distributions
            for col in numeric_cols:
                distribution_analysis["numeric_distributions"][col] = {
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "is_normal": abs(df[col].skew()) < 0.5 and abs(df[col].kurtosis()) < 0.5
                }

            # Analyze categorical distributions
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                distribution_analysis["categorical_distributions"][col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": value_counts.iloc[0] if not value_counts.empty else 0,
                    "entropy": -sum((value_counts / len(df)) * np.log2(value_counts / len(df) + 1e-10))
                }

            return distribution_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing distributions: {e}")
            return {"error": str(e)}

    def _find_high_correlations(self, df: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find high correlations between numeric columns"""
        try:
            corr_matrix = df.corr()
            high_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        high_correlations.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": corr_value
                        })
            
            return high_correlations

        except Exception as e:
            self.logger.error(f"Error finding correlations: {e}")
            return []

    def _generate_type_recommendations(self, detected_types: Dict[str, str], 
                                     confidence_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on detected data types"""
        try:
            recommendations = []
            
            low_confidence_types = [col for col, conf in confidence_scores.items() if conf < 0.6]
            if low_confidence_types:
                recommendations.append(f"Review data types for columns with low confidence: {', '.join(low_confidence_types)}")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating type recommendations: {e}")
            return []

    def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
        """
        LAYER 2: Generate prompt for LLM to create advanced data discovery code.
        
        Args:
            layer1_results: Results from Layer 1 analysis
            state: Current workflow state
            
        Returns:
            Prompt string for LLM code generation
        """
        self.logger.info("🔧 LAYER 2: Generating LLM code generation prompt for data discovery")
        
        # Extract key insights from Layer 1
        basic_info = layer1_results.get("basic_info", {})
        statistical_summary = layer1_results.get("statistical_summary", {})
        correlations = layer1_results.get("correlations", {})
        data_types = layer1_results.get("data_types", {})
        
        prompt = f"""Generate advanced Python code for comprehensive data discovery and analysis based on the following insights:

## Dataset Overview:
- Shape: {basic_info.get('shape', 'unknown')}
- Columns: {basic_info.get('columns', [])}
- Memory Usage: {basic_info.get('memory_usage_mb', 0):.2f} MB

## Statistical Summary:
{statistical_summary}

## Detected Data Types:
{data_types.get('detected_types', {})}

## Correlation Insights:
{correlations}

## Requirements for Generated Code:
1. Perform advanced statistical analysis beyond basic summaries
2. Detect complex patterns: seasonality, trends, cyclic behavior
3. Identify domain-specific insights based on column names
4. Generate visualizations (charts, heatmaps, distributions)
5. Create feature importance rankings
6. Detect potential data quality issues
7. Use only: pandas, numpy, matplotlib, seaborn
8. Add clear comments explaining each analysis
9. Return structured results (dictionary with findings)

Generate comprehensive, production-ready Python code:"""
        
        return prompt
    
    def process_sandbox_results(
        self,
        sandbox_output: Dict[str, Any],
        layer1_results: Dict[str, Any],
        state: ClassificationState
    ) -> Dict[str, Any]:
        """
        LAYER 2: Process and validate sandbox execution results for data discovery.
        
        Args:
            sandbox_output: Raw output from sandbox execution
            layer1_results: Results from Layer 1 (for comparison)
            state: Current workflow state
            
        Returns:
            Processed and validated discovery results
        """
        self.logger.info("🔍 LAYER 2: Processing sandbox results for data discovery")
        
        # Validate sandbox execution was successful
        if sandbox_output.get("status") != "SUCCESS":
            raise ValueError(f"Sandbox execution failed: {sandbox_output.get('error', 'Unknown error')}")
        
        # Extract discovery results from sandbox output
        discovery_data = sandbox_output.get("output", {})
        
        # Validate the output structure
        if not isinstance(discovery_data, dict):
            raise ValueError("Sandbox output should contain a discovery results dictionary")
        
        # Additional validation can be added here
        # Compare with Layer 1 to ensure quality improvement
        
        result = {
            "advanced_discovery": discovery_data,
            "layer2_success": True,
            "sandbox_execution_time": sandbox_output.get("execution_time", 0)
        }
        
        self.logger.info("✅ LAYER 2: Sandbox results processed and validated")
        return result
    
    def _is_potential_datetime(self, series: pd.Series) -> bool:
        """Check if a series could be datetime"""
        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                return True
            # Try to convert a sample
            pd.to_datetime(series.head(10))
            return True
        except:
            return False


