"""
📊 EDA Agent - Advanced Exploratory Data Analysis with Double-Layer Architecture

DOUBLE-LAYER ARCHITECTURE:
- Layer 1 (Hardcoded): Reliable statistical analysis and basic visualizations
  - Correlation matrix with p-values
  - Distribution statistics
  - Outlier detection (IQR, Z-score)
  - Basic feature importance
  - Target variable relationships
  - Basic plots (histograms, box plots, correlation heatmap)

- Layer 2 (LLM + Sandbox): Advanced visualizations and insights
  - Interactive Plotly visualizations
  - Advanced statistical plots
  - Custom visualizations based on data patterns
  - AI-generated insights
"""

import logging
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import warnings
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest
import json

warnings.filterwarnings('ignore')

from ..base_agent import BaseAgent
from ...workflows.state_management import ClassificationState


class EDAAgent(BaseAgent):
    """
    Advanced EDA Agent with Double-Layer Architecture

    Layer 1: Reliable statistical analysis and basic plots
    Layer 2: LLM-generated advanced visualizations
    """

    def __init__(self) -> None:
        super().__init__(
            agent_name="eda_analysis",
            agent_version="3.0.0",
            enable_layer2=True,
            sandbox_timeout=120,  # Longer timeout for plotting
            sandbox_memory_limit="3g"  # More memory for plots
        )

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create plots directory
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)

        self.logger.info("EDA Agent initialized with double-layer architecture")

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": "Advanced EDA Analysis Agent (Double-Layer)",
            "version": self.agent_version,
            "description": "Comprehensive exploratory data analysis with double-layer architecture",
            "capabilities": [
                "Layer 1: Statistical analysis and basic plots",
                "Layer 2: Advanced interactive visualizations",
                "Correlation analysis with p-values",
                "Distribution statistics and normality tests",
                "Outlier detection (IQR, Z-score)",
                "Feature importance analysis",
                "Target variable analysis",
                "Automated insight generation"
            ],
            "dependencies": ["data_cleaning", "data_discovery"],
            "supports_layer2": True
        }

    def get_dependencies(self) -> list:
        return ["data_cleaning", "data_discovery"]

    # ===== LAYER 1: RELIABLE STATISTICAL ANALYSIS =====

    async def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """
        LAYER 1: Perform reliable statistical analysis and generate basic plots.

        This serves as the fallback if Layer 2 fails.

        Returns:
            Dict containing:
            - correlations: Correlation matrix with p-values
            - distributions: Distribution statistics for all features
            - outliers: Outlier detection results
            - feature_importance: Basic feature importance
            - target_relationships: Relationships with target variable
            - plot_paths: List of generated plot file paths
        """
        self.logger.info("🔬 LAYER 1: Performing reliable statistical analysis...")

        # Get dataset using state_manager (CORRECT WAY)
        from ...workflows.state_management import state_manager
        
        df = None
        
        # Method 1: Get from state_manager using "cleaned" version (after data cleaning)
        try:
            df = state_manager.get_dataset(state, "cleaned")
            if df is not None and hasattr(df, 'shape'):
                self.logger.info(f"  ✅ Found CLEANED dataset via state_manager with shape {df.shape}")
        except Exception as e:
            self.logger.debug(f"  Could not get cleaned dataset: {e}")
        
        # Method 2: Get from state_manager using "original" version
        if df is None or not hasattr(df, 'shape'):
            try:
                df = state_manager.get_dataset(state, "original")
                if df is not None and hasattr(df, 'shape'):
                    self.logger.info(f"  ✅ Found ORIGINAL dataset via state_manager with shape {df.shape}")
            except Exception as e:
                self.logger.debug(f"  Could not get original dataset: {e}")
        
        # Method 3: Try direct access (if stored as DataFrame, not ID)
        if df is None or not hasattr(df, 'shape'):
            for key in ['processed_dataset', 'dataset', 'cleaned_dataset', 'original_dataset']:
                if key in state:
                    potential_df = state[key]
                    if hasattr(potential_df, 'shape'):  # It's a DataFrame!
                        df = potential_df
                        self.logger.info(f"  ✅ Found dataset in state['{key}'] with shape {df.shape}")
                        break
        
        # Final check
        if df is None or not hasattr(df, 'shape'):
            available_keys = [k for k in state.keys() if 'data' in k.lower()]
            self.logger.error(f"  ❌ No DataFrame found! Available data-related keys: {available_keys}")
            self.logger.error(f"  ❌ Checked state_manager.get_dataset(state, 'cleaned' and 'original')")
            self.logger.error(f"  ❌ Checked state keys: processed_dataset, dataset, cleaned_dataset, original_dataset")
            
            # Log what we found in those keys
            for key in ['original_dataset', 'cleaned_dataset', 'dataset', 'processed_dataset']:
                if key in state:
                    val = state[key]
                    self.logger.error(f"  ❌ state['{key}'] = {type(val)} (not a DataFrame)")
            
            return {
                "error": "No DataFrame available for EDA - only found string IDs or None",
                "eda_plots": [],
                "statistical_summary": {},
                "distribution_analysis": {},
                "correlation_matrix": None,
                "available_state_keys": list(state.keys())[:10]  # First 10 keys only
            }

        target_column = state.get("target_column", "")
        session_id = state.get("session_id") or state.get("workflow_id", "unknown")

        # Create session-specific plots directory with proper path
        # Use backend/plots/{workflow_id}/ structure for web serving
        base_plots_dir = Path("backend/plots")
        session_plots_dir = base_plots_dir / session_id
        session_plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"  📁 Created plots directory: {session_plots_dir}")
        self.logger.info(f"  📁 Plots will be accessible at: /api/workflow/plot/{session_id}/...")

        results = {}
        plot_paths = []

        # 1. Correlation Analysis with P-Values
        self.logger.info("  📊 Computing correlation matrix with p-values...")
        correlations = self._compute_correlations_with_pvalues(df, target_column)
        results["correlations"] = correlations

        # Generate correlation heatmap
        corr_plot = self._generate_correlation_heatmap(df, session_plots_dir)
        if corr_plot:
            plot_paths.append(corr_plot)

        # 2. Distribution Statistics
        self.logger.info("  📈 Computing distribution statistics...")
        distributions = self._compute_distribution_statistics(df, target_column)
        results["distributions"] = distributions

        # Generate distribution plots (histograms)
        dist_plots = self._generate_distribution_plots(df, target_column, session_plots_dir)
        plot_paths.extend(dist_plots)

        # 3. Outlier Detection
        self.logger.info("  🎯 Detecting outliers...")
        outliers = self._detect_outliers(df, target_column)
        results["outliers"] = outliers

        # Generate box plots for outlier visualization
        outlier_plots = self._generate_outlier_plots(df, target_column, session_plots_dir)
        plot_paths.extend(outlier_plots)

        # 4. Feature Importance (Basic)
        self.logger.info("  ⭐ Computing basic feature importance...")
        feature_importance = self._compute_basic_feature_importance(df, target_column)
        results["feature_importance"] = feature_importance

        # 5. Target Variable Relationships
        self.logger.info("  🎯 Analyzing target variable relationships...")
        target_relationships = self._analyze_target_relationships(df, target_column)
        results["target_relationships"] = target_relationships

        # Generate target distribution plot
        target_plot = self._generate_target_plot(df, target_column, session_plots_dir)
        if target_plot:
            plot_paths.append(target_plot)

        # 6. Summary Statistics
        results["summary_statistics"] = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum()
        }

        # Convert plot file paths to API-accessible URLs
        plot_list = []
        for plot_path in plot_paths:
            plot_file = Path(plot_path)
            if plot_file.exists():
                # Create API URL: /api/workflow/plot/{workflow_id}/{filename}
                api_url = f"/api/workflow/plot/{session_id}/{plot_file.name}"
                plot_list.append({
                    "title": plot_file.stem.replace("_", " ").title(),
                    "name": plot_file.name,
                    "path": api_url,
                    "url": api_url
                })
                self.logger.info(f"  📊 Plot available: {api_url}")
        
        results["plot_paths"] = plot_paths  # Keep original paths
        results["eda_plots"] = plot_list  # Add structured plot list for API
        results["plots"] = plot_list  # Also add as 'plots' for consistency

        self.logger.info(f"✅ LAYER 1 Complete: Generated {len(plot_list)} plots accessible via API")
        return results

    def _compute_correlations_with_pvalues(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Compute correlation matrix with p-values"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {"error": "Not enough numeric features"}

        # Correlation matrix
        corr_matrix = numeric_df.corr()

        # P-values matrix
        n = len(numeric_df)
        p_values = np.zeros((numeric_df.shape[1], numeric_df.shape[1]))

        for i in range(numeric_df.shape[1]):
            for j in range(i, numeric_df.shape[1]):
                if i == j:
                    p_values[i, j] = 0
                else:
                    col1 = numeric_df.iloc[:, i].dropna()
                    col2 = numeric_df.iloc[:, j].dropna()

                    # Find common indices
                    common_idx = col1.index.intersection(col2.index)
                    if len(common_idx) > 2:
                        corr, p_val = stats.pearsonr(col1[common_idx], col2[common_idx])
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
                    else:
                        p_values[i, j] = 1.0
                        p_values[j, i] = 1.0

        p_values_df = pd.DataFrame(p_values, columns=numeric_df.columns, index=numeric_df.columns)

        # Target correlations
        target_correlations = {}
        if target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].drop(target_column)
            target_p_vals = p_values_df[target_column].drop(target_column)

            target_correlations = {
                "correlations": target_corr.to_dict(),
                "p_values": target_p_vals.to_dict(),
                "significant_features": target_corr[target_p_vals < 0.05].to_dict()
            }

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "p_values_matrix": p_values_df.to_dict(),
            "target_correlations": target_correlations,
            "high_correlation_pairs": self._find_high_correlations(corr_matrix, threshold=0.8)
        }

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Find highly correlated feature pairs"""
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        return pairs

    def _compute_distribution_statistics(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Compute distribution statistics for all numeric features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        dist_stats = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Basic statistics
            stats_dict = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis())
            }

            # Normality test (Shapiro-Wilk for small samples)
            if len(series) <= 5000 and len(series) >= 3:
                try:
                    shapiro_stat, shapiro_p = shapiro(series)
                    stats_dict["shapiro_test"] = {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05
                    }
                except:
                    pass

            dist_stats[col] = stats_dict

        return dist_stats

    def _detect_outliers(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect outliers using IQR and Z-score methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        outliers = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]

            outliers[col] = {
                "iqr_method": {
                    "count": len(iqr_outliers),
                    "percentage": float(len(iqr_outliers) / len(series) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                },
                "zscore_method": {
                    "count": len(z_outliers),
                    "percentage": float(len(z_outliers) / len(series) * 100)
                }
            }

        return outliers

    def _compute_basic_feature_importance(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Compute basic feature importance using correlation"""
        if target_column not in df.columns:
            return {"error": "Target column not found"}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        if not numeric_cols:
            return {"error": "No numeric features"}

        # Compute absolute correlations with target
        importance = {}
        for col in numeric_cols:
            try:
                corr = df[col].corr(df[target_column])
                importance[col] = abs(corr)
            except:
                importance[col] = 0.0

        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "importance_scores": sorted_importance,
            "top_5_features": list(sorted_importance.keys())[:5]
        }

    def _analyze_target_relationships(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze relationships with target variable"""
        if target_column not in df.columns:
            return {"error": "Target column not found"}

        target_series = df[target_column]

        analysis = {
            "unique_values": int(target_series.nunique()),
            "missing_count": int(target_series.isnull().sum()),
            "missing_percentage": float(target_series.isnull().sum() / len(target_series) * 100),
            "value_counts": target_series.value_counts().to_dict()
        }

        # Check if classification or regression
        if target_series.dtype in ['object', 'category'] or target_series.nunique() <= 10:
            analysis["task_type"] = "classification"
            class_counts = target_series.value_counts()
            analysis["class_balance"] = {
                "is_balanced": (class_counts.max() - class_counts.min()) / class_counts.max() < 0.1,
                "balance_ratio": float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0
            }
        else:
            analysis["task_type"] = "regression"
            analysis["target_statistics"] = {
                "mean": float(target_series.mean()),
                "median": float(target_series.median()),
                "std": float(target_series.std())
            }

        return analysis

    def _generate_correlation_heatmap(self, df: pd.DataFrame, plots_dir: Path) -> Optional[str]:
        """Generate correlation heatmap"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return None

            corr_matrix = numeric_df.corr()

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
            plt.title('Feature Correlation Matrix (Layer 1)', fontsize=16)
            plt.tight_layout()

            plot_path = plots_dir / "correlation_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_path)
        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {e}")
            return None

    def _generate_distribution_plots(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> List[str]:
        """Generate histogram plots for numeric features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        if not numeric_cols:
            return []

        plot_paths = []

        try:
            # Create distribution plots
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
                ax = axes[i]
                series = df[col].dropna()

                ax.hist(series, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{col}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')

                # Add mean and median lines
                mean_val = series.mean()
                median_val = series.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
                ax.legend(fontsize=8)

            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle('Feature Distributions (Layer 1)', fontsize=16)
            plt.tight_layout()

            plot_path = plots_dir / "distributions_histograms.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            plot_paths.append(str(plot_path))
        except Exception as e:
            self.logger.error(f"Error generating distribution plots: {e}")

        return plot_paths

    def _generate_outlier_plots(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> List[str]:
        """Generate box plots for outlier visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        if not numeric_cols:
            return []

        plot_paths = []

        try:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
                ax = axes[i]
                series = df[col].dropna()

                box_plot = ax.boxplot(series, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightblue')
                ax.set_title(f'{col}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle('Outlier Detection - Box Plots (Layer 1)', fontsize=16)
            plt.tight_layout()

            plot_path = plots_dir / "outliers_boxplots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            plot_paths.append(str(plot_path))
        except Exception as e:
            self.logger.error(f"Error generating outlier plots: {e}")

        return plot_paths

    def _generate_target_plot(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> Optional[str]:
        """Generate target variable distribution plot"""
        if target_column not in df.columns:
            return None

        try:
            target_series = df[target_column]

            plt.figure(figsize=(12, 5))

            if target_series.dtype in ['object', 'category'] or target_series.nunique() <= 10:
                # Categorical target
                plt.subplot(1, 2, 1)
                target_series.value_counts().plot(kind='bar', color='skyblue', alpha=0.7)
                plt.title(f'Target Distribution: {target_column}')
                plt.xlabel('Class')
                plt.ylabel('Count')
                plt.xticks(rotation=45)

                plt.subplot(1, 2, 2)
                target_series.value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
                plt.title(f'Target Proportions: {target_column}')
                plt.ylabel('')
            else:
                # Numeric target
                plt.subplot(1, 2, 1)
                target_series.hist(bins=30, color='skyblue', alpha=0.7, edgecolor='black')
                plt.title(f'Target Distribution: {target_column}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')

                plt.subplot(1, 2, 2)
                target_series.plot(kind='box', color='lightgreen')
                plt.title(f'Target Box Plot: {target_column}')
                plt.ylabel('Value')

            plt.suptitle(f'Target Variable Analysis (Layer 1)', fontsize=16)
            plt.tight_layout()

            plot_path = plots_dir / "target_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_path)
        except Exception as e:
            self.logger.error(f"Error generating target plot: {e}")
            return None

    # ===== LAYER 2: LLM-GENERATED VISUALIZATIONS =====

    def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
        """
        LAYER 2: Generate prompt for LLM to create advanced visualizations.

        Args:
            layer1_results: Results from Layer 1 analysis
            state: Current workflow state

        Returns:
            Prompt string for LLM code generation
        """
        self.logger.info("📝 LAYER 2: Generating visualization code prompt...")

        # Import the prompt template
        from .prompts import EDA_VISUALIZATION_PROMPT_TEMPLATE

        # Prepare context from Layer 1 results
        context = {
            "statistical_summary": layer1_results.get("summary_statistics", {}),
            "correlations_summary": layer1_results.get("correlations", {}),
            "distributions_summary": layer1_results.get("distributions", {}),
            "outliers_summary": layer1_results.get("outliers", {}),
            "feature_importance_summary": layer1_results.get("feature_importance", {}),
            "target_relationships_summary": layer1_results.get("target_relationships", {}),
            "target_column": state.get("target_column", ""),
            "session_id": state.get("session_id", "unknown")
        }

        # Generate prompt
        prompt = EDA_VISUALIZATION_PROMPT_TEMPLATE.format(**context)

        return prompt

    def process_sandbox_results(
        self,
        sandbox_output: Dict[str, Any],
        layer1_results: Dict[str, Any],
        state: ClassificationState
    ) -> Dict[str, Any]:
        """
        LAYER 2: Process and validate sandbox execution results.

        Validates that:
        - Plots were generated successfully
        - Plot files exist and are valid
        - Plot quality meets standards

        Args:
            sandbox_output: Raw output from sandbox execution
            layer1_results: Results from Layer 1 (for fallback)
            state: Current workflow state

        Returns:
            Processed and validated Layer 2 results

        Raises:
            ValueError: If results are invalid or worse than Layer 1
        """
        self.logger.info("🔍 LAYER 2: Processing and validating sandbox results...")

        # Call parent validation
        results = super().process_sandbox_results(sandbox_output, layer1_results, state)

        # Validate plots were generated
        plot_paths = results.get("plot_paths", [])

        if not plot_paths:
            raise ValueError("No plots were generated by Layer 2")

        # Validate each plot
        from .validators import PlotValidator
        validator = PlotValidator()

        validated_plots = []
        for plot_path in plot_paths:
            validation_result = validator.validate_plot(plot_path)

            if validation_result["is_valid"]:
                validated_plots.append(plot_path)
            else:
                self.logger.warning(f"Plot validation failed for {plot_path}: {validation_result['errors']}")

        if not validated_plots:
            raise ValueError("All Layer 2 plots failed validation")

        results["validated_plot_paths"] = validated_plots
        results["validation_summary"] = {
            "total_plots": len(plot_paths),
            "valid_plots": len(validated_plots),
            "invalid_plots": len(plot_paths) - len(validated_plots)
        }

        self.logger.info(f"✅ LAYER 2 validated: {len(validated_plots)}/{len(plot_paths)} plots passed validation")

        # Merge with Layer 1 results
        merged_results = {**layer1_results, **results}
        merged_results["layer2_plot_paths"] = validated_plots
        merged_results["layer1_plot_paths"] = layer1_results.get("plot_paths", [])
        merged_results["all_plot_paths"] = layer1_results.get("plot_paths", []) + validated_plots

        return merged_results

    def _prepare_sandbox_datasets(self, state: ClassificationState) -> Dict[str, str]:
        """
        Prepare datasets for sandbox execution.

        Override from BaseAgent to provide dataset for visualization.
        """
        df = state.get("cleaned_dataset")
        if df is None:
            df = state.get("original_dataset")

        if df is None:
            return {}

        # Save dataset to temporary file
        temp_dir = Path("/tmp") / "eda_sandbox" / state.get("session_id", "unknown")
        temp_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = temp_dir / "dataset.csv"
        df.to_csv(dataset_path, index=False)

        return {
            "dataset": str(dataset_path)
        }
