"""
📊 EDA Agent - Advanced Exploratory Data Analysis

Performs comprehensive exploratory data analysis with intelligent plot generation
and statistical insights. Thinks like a senior data scientist.
"""

import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent
from ..workflows.state_management import ClassificationState, AgentStatus, state_manager


class EDAAgent(BaseAgent):
    """Advanced EDA Agent that thinks like a senior data scientist"""

    def __init__(self) -> None:
        super().__init__("eda_analysis", "2.0.0")
        self.logger = logging.getLogger("agent.eda")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots directory
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": "Advanced EDA Analysis Agent",
            "version": self.agent_version,
            "description": "Comprehensive exploratory data analysis with intelligent visualizations",
            "capabilities": [
                "Statistical summaries and distributions",
                "Correlation analysis and heatmaps",
                "Target variable analysis",
                "Missing value patterns",
                "Outlier detection and visualization",
                "Feature relationship analysis",
                "Data quality assessment",
                "Interactive plot generation"
            ],
            "dependencies": ["data_cleaning", "data_discovery"],
        }

    def get_dependencies(self) -> list:
        return ["data_cleaning", "data_discovery"]

    async def execute(self, state: ClassificationState) -> ClassificationState:
        try:
            self.logger.info("🧠 Starting advanced EDA analysis - thinking like a data scientist")
            
            # Get cleaned dataset
            df = state_manager.get_dataset(state, "cleaned")
            if df is None:
                df = state_manager.get_dataset(state, "original")
            if df is None:
                raise ValueError("No dataset available for EDA")

            target_column = state.get("target_column", "")
            session_id = state.get("session_id", "unknown")
            
            self.logger.info(f"📊 Analyzing dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"🎯 Target column: {target_column}")
            
            # Create session-specific plots directory
            session_plots_dir = self.plots_dir / session_id
            session_plots_dir.mkdir(exist_ok=True)
            
            # 1. Basic Dataset Overview
            basic_stats = await self._analyze_basic_stats(df, target_column)
            
            # 2. Target Variable Analysis
            target_analysis = await self._analyze_target_variable(df, target_column, session_plots_dir)
            
            # 3. Feature Distribution Analysis
            distribution_analysis = await self._analyze_distributions(df, target_column, session_plots_dir)
            
            # 4. Correlation Analysis
            correlation_analysis = await self._analyze_correlations(df, target_column, session_plots_dir)
            
            # 5. Missing Value Analysis
            missing_analysis = await self._analyze_missing_values(df, session_plots_dir)
            
            # 6. Outlier Analysis
            outlier_analysis = await self._analyze_outliers(df, target_column, session_plots_dir)
            
            # 7. Feature Relationships
            relationship_analysis = await self._analyze_feature_relationships(df, target_column, session_plots_dir)
            
            # 8. Data Quality Assessment
            quality_assessment = await self._assess_data_quality(df, target_column)
            
            # Store all results in state
            state["statistical_summary"] = basic_stats
            state["target_analysis"] = target_analysis
            state["distribution_analysis"] = distribution_analysis
            state["correlation_analysis"] = correlation_analysis
            state["missing_value_analysis"] = missing_analysis
            state["outlier_analysis"] = outlier_analysis
            state["feature_relationship_analysis"] = relationship_analysis
            state["data_quality_assessment"] = quality_assessment
            
            # Store plot paths
            plot_files = list(session_plots_dir.glob("*.png"))
            state["eda_plots"] = [str(p) for p in plot_files]
            
            # Generate comprehensive EDA report
            eda_report = await self._generate_eda_report(
                basic_stats, target_analysis, distribution_analysis, 
                correlation_analysis, missing_analysis, outlier_analysis,
                relationship_analysis, quality_assessment
            )
            state["eda_report"] = eda_report
            
            # Update agent status
            state["agent_statuses"]["eda_analysis"] = AgentStatus.COMPLETED
            state["completed_agents"].append("eda_analysis")
            
            self.logger.info(f"✅ EDA analysis completed - Generated {len(plot_files)} plots")
            self.logger.info(f"📈 Key insights: {len(target_analysis.get('insights', []))} target insights, {len(correlation_analysis.get('insights', []))} correlation insights")
            
            return state

        except Exception as e:
            self.logger.error(f"❌ EDA analysis failed: {e}", exc_info=True)
            state["agent_statuses"]["eda_analysis"] = AgentStatus.FAILED
            state["last_error"] = str(e)
            state["error_count"] += 1
            return state

    async def _analyze_basic_stats(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze basic dataset statistics"""
        self.logger.info("📊 Computing basic statistics...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from numeric if it's there
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        stats = {
            "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "data_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "total": len(df.columns)
            },
            "missing_values": {
                "total": int(df.isnull().sum().sum()),
                "percentage": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
                "by_column": {col: int(df[col].isnull().sum()) for col in df.columns}
            },
            "duplicates": {
                "exact_duplicates": int(df.duplicated().sum()),
                "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100)
            },
            "memory_usage": {
                "total_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
                "per_column_mb": {col: float(df[col].memory_usage(deep=True) / 1024**2) for col in df.columns}
            }
        }
        
        return stats

    async def _analyze_target_variable(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> Dict[str, Any]:
        """Analyze target variable distribution and characteristics"""
        self.logger.info(f"🎯 Analyzing target variable: {target_column}")
        
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in dataset"}
        
        target_series = df[target_column]
        analysis = {}
        
        # Basic target statistics
        analysis["distribution"] = {
            "value_counts": target_series.value_counts().to_dict(),
            "unique_values": int(target_series.nunique()),
            "missing_count": int(target_series.isnull().sum()),
            "missing_percentage": float(target_series.isnull().sum() / len(target_series) * 100)
        }
        
        # Class balance analysis
        if target_series.dtype in ['object', 'category'] or target_series.nunique() <= 10:
            class_counts = target_series.value_counts()
            analysis["class_balance"] = {
                "is_balanced": len(class_counts) > 1 and (class_counts.max() - class_counts.min()) / class_counts.max() < 0.1,
                "balance_ratio": float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0,
                "majority_class": class_counts.index[0],
                "majority_count": int(class_counts.iloc[0]),
                "minority_class": class_counts.index[-1] if len(class_counts) > 1 else None,
                "minority_count": int(class_counts.iloc[-1]) if len(class_counts) > 1 else None
            }
        
        # Generate target distribution plot
        plt.figure(figsize=(12, 8))
        
        if target_series.dtype in ['object', 'category'] or target_series.nunique() <= 10:
            # Categorical target
            plt.subplot(2, 2, 1)
            target_series.value_counts().plot(kind='bar', color='skyblue', alpha=0.7)
            plt.title(f'Target Variable Distribution: {target_column}')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            target_series.value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title(f'Target Variable Proportions: {target_column}')
            plt.ylabel('')
        else:
            # Numeric target
            plt.subplot(2, 2, 1)
            target_series.hist(bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            plt.title(f'Target Variable Distribution: {target_column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 2, 2)
            target_series.plot(kind='box', color='lightgreen')
            plt.title(f'Target Variable Box Plot: {target_column}')
            plt.ylabel('Value')
        
        # Missing values in target
        plt.subplot(2, 2, 3)
        missing_data = target_series.isnull()
        missing_data.value_counts().plot(kind='bar', color=['lightcoral', 'lightgreen'])
        plt.title(f'Missing Values in Target: {target_column}')
        plt.xlabel('Missing')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Not Missing', 'Missing'], rotation=0)
        
        # Target over time (if index suggests time series)
        plt.subplot(2, 2, 4)
        if len(target_series) > 100:
            # Sample for visualization
            sample_size = min(1000, len(target_series))
            sample_indices = np.linspace(0, len(target_series)-1, sample_size, dtype=int)
            target_series.iloc[sample_indices].plot(color='purple', alpha=0.7)
            plt.title(f'Target Variable Sample: {target_column}')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
        else:
            target_series.plot(color='purple', alpha=0.7)
            plt.title(f'Target Variable: {target_column}')
            plt.xlabel('Index')
            plt.ylabel('Value')
        
        plt.tight_layout()
        plot_path = plots_dir / f"target_analysis_{target_column}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        analysis["plot_path"] = str(plot_path)
        
        # Generate insights
        insights = []
        if analysis["distribution"]["missing_percentage"] > 0:
            insights.append(f"⚠️ Target variable has {analysis['distribution']['missing_percentage']:.1f}% missing values")
        
        if "class_balance" in analysis:
            if not analysis["class_balance"]["is_balanced"]:
                insights.append(f"⚠️ Class imbalance detected: {analysis['class_balance']['balance_ratio']:.2f} ratio")
            else:
                insights.append("✅ Target classes are well balanced")
        
        if target_series.nunique() > 20:
            insights.append(f"ℹ️ Target has {target_series.nunique()} unique values - consider if this should be binned")
        
        analysis["insights"] = insights
        return analysis

    async def _analyze_distributions(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> Dict[str, Any]:
        """Analyze feature distributions"""
        self.logger.info("📈 Analyzing feature distributions...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if not numeric_cols:
            return {"message": "No numeric features found for distribution analysis"}
        
        # Create distribution plots
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        distribution_stats = {}
        
        for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
            ax = axes[i]
            series = df[col].dropna()
            
            # Histogram
            ax.hist(series, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution: {col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_val = series.mean()
            median_val = series.median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
            ax.legend()
            
            # Store distribution statistics
            distribution_stats[col] = {
                "mean": float(mean_val),
                "median": float(median_val),
                "std": float(series.std()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "is_normal": abs(series.skew()) < 0.5 and abs(series.kurtosis()) < 0.5
            }
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = plots_dir / "feature_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_path": str(plot_path),
            "distribution_statistics": distribution_stats,
            "insights": self._generate_distribution_insights(distribution_stats)
        }

    async def _analyze_correlations(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> Dict[str, Any]:
        """Analyze feature correlations"""
        self.logger.info("🔗 Analyzing feature correlations...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {"message": "Not enough numeric features for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        plot_path = plots_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze correlations with target
        target_correlations = {}
        if target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].drop(target_column)
            target_correlations = {
                "strong_positive": target_corr[target_corr > 0.7].to_dict(),
                "strong_negative": target_corr[target_corr < -0.7].to_dict(),
                "moderate_positive": target_corr[(target_corr > 0.3) & (target_corr <= 0.7)].to_dict(),
                "moderate_negative": target_corr[(target_corr < -0.3) & (target_corr >= -0.7)].to_dict(),
                "weak": target_corr[(target_corr >= -0.3) & (target_corr <= 0.3)].to_dict()
            }
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return {
            "plot_path": str(plot_path),
            "correlation_matrix": corr_matrix.to_dict(),
            "target_correlations": target_correlations,
            "high_correlation_pairs": high_corr_pairs,
            "insights": self._generate_correlation_insights(target_correlations, high_corr_pairs)
        }

    async def _analyze_missing_values(self, df: pd.DataFrame, plots_dir: Path) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        self.logger.info("🔍 Analyzing missing value patterns...")
        
        missing_data = df.isnull()
        missing_summary = missing_data.sum()
        missing_percentage = (missing_summary / len(df)) * 100
        
        # Create missing value visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Missing values by column
        missing_summary[missing_summary > 0].sort_values(ascending=True).plot(
            kind='barh', ax=axes[0,0], color='lightcoral'
        )
        axes[0,0].set_title('Missing Values by Column')
        axes[0,0].set_xlabel('Count')
        
        # Missing value percentage by column
        missing_percentage[missing_percentage > 0].sort_values(ascending=True).plot(
            kind='barh', ax=axes[0,1], color='orange'
        )
        axes[0,1].set_title('Missing Values Percentage by Column')
        axes[0,1].set_xlabel('Percentage')
        
        # Missing value heatmap (sample)
        sample_size = min(1000, len(df))
        sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
        sns.heatmap(missing_data.iloc[sample_indices], cbar=True, ax=axes[1,0], 
                   cmap='viridis', yticklabels=False)
        axes[1,0].set_title('Missing Value Pattern (Sample)')
        axes[1,0].set_xlabel('Columns')
        
        # Missing value correlation
        missing_corr = missing_data.corr()
        sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Missing Value Correlation')
        
        plt.tight_layout()
        plot_path = plots_dir / "missing_values_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_path": str(plot_path),
            "missing_summary": missing_summary.to_dict(),
            "missing_percentage": missing_percentage.to_dict(),
            "columns_with_missing": missing_summary[missing_summary > 0].to_dict(),
            "missing_correlation": missing_corr.to_dict(),
            "insights": self._generate_missing_value_insights(missing_summary, missing_percentage)
        }

    async def _analyze_outliers(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> Dict[str, Any]:
        """Analyze outliers in numeric features"""
        self.logger.info("🎯 Analyzing outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if not numeric_cols:
            return {"message": "No numeric features for outlier analysis"}
        
        outlier_analysis = {}
        
        # Create outlier plots
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
            
            # Box plot
            box_plot = ax.boxplot(series, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            ax.set_title(f'Outliers: {col}')
            ax.set_ylabel('Value')
            
            # Calculate outlier statistics
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            outlier_analysis[col] = {
                "outlier_count": len(outliers),
                "outlier_percentage": len(outliers) / len(series) * 100,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": outliers.tolist()[:10]  # First 10 outliers
            }
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = plots_dir / "outlier_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_path": str(plot_path),
            "outlier_statistics": outlier_analysis,
            "insights": self._generate_outlier_insights(outlier_analysis)
        }

    async def _analyze_feature_relationships(self, df: pd.DataFrame, target_column: str, plots_dir: Path) -> Dict[str, Any]:
        """Analyze relationships between features and target"""
        self.logger.info("🔗 Analyzing feature relationships...")
        
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found"}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if not numeric_cols:
            return {"message": "No numeric features for relationship analysis"}
        
        # Create scatter plots for top correlated features
        target_corr = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
        top_features = target_corr.head(6).index.tolist()
        
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        relationship_stats = {}
        
        for i, feature in enumerate(top_features[:n_rows*n_cols]):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(df[feature], df[target_column], alpha=0.6, s=20)
            ax.set_xlabel(feature)
            ax.set_ylabel(target_column)
            ax.set_title(f'{feature} vs {target_column}')
            
            # Add trend line
            z = np.polyfit(df[feature].dropna(), df[target_column].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[feature], p(df[feature]), "r--", alpha=0.8)
            
            # Calculate correlation
            corr = df[feature].corr(df[target_column])
            relationship_stats[feature] = {
                "correlation": float(corr),
                "correlation_strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            }
        
        # Hide empty subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = plots_dir / "feature_relationships.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_path": str(plot_path),
            "relationship_statistics": relationship_stats,
            "top_correlated_features": target_corr.head(10).to_dict(),
            "insights": self._generate_relationship_insights(relationship_stats)
        }

    async def _assess_data_quality(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Assess overall data quality"""
        self.logger.info("✅ Assessing data quality...")
        
        quality_metrics = {
            "completeness": 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            "uniqueness": 1 - (df.duplicated().sum() / len(df)),
            "consistency": 0.0,  # Placeholder for consistency checks
            "validity": 0.0  # Placeholder for validity checks
        }
        
        # Calculate consistency (check for data type consistency)
        consistency_score = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    types = [type(x).__name__ for x in sample]
                    consistency_score += len(set(types)) == 1
            else:
                consistency_score += 1
        
        quality_metrics["consistency"] = consistency_score / len(df.columns)
        
        # Calculate validity (check for reasonable value ranges)
        validity_score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not df[col].isna().all():
                # Check for infinite values
                if not np.isinf(df[col]).any():
                    validity_score += 1
        
        if len(numeric_cols) > 0:
            quality_metrics["validity"] = validity_score / len(numeric_cols)
        else:
            quality_metrics["validity"] = 1.0
        
        # Overall quality score
        quality_metrics["overall_score"] = np.mean(list(quality_metrics.values()))
        
        # Generate quality insights
        insights = []
        if quality_metrics["completeness"] < 0.9:
            insights.append(f"⚠️ Low completeness: {quality_metrics['completeness']:.1%}")
        if quality_metrics["uniqueness"] < 0.95:
            insights.append(f"⚠️ High duplicate rate: {(1-quality_metrics['uniqueness']):.1%}")
        if quality_metrics["consistency"] < 0.8:
            insights.append(f"⚠️ Data type inconsistencies detected")
        if quality_metrics["overall_score"] > 0.8:
            insights.append("✅ Good overall data quality")
        else:
            insights.append("⚠️ Data quality needs improvement")
        
        return {
            "quality_metrics": quality_metrics,
            "insights": insights,
            "recommendations": self._generate_quality_recommendations(quality_metrics)
        }

    def _generate_distribution_insights(self, distribution_stats: Dict) -> List[str]:
        """Generate insights from distribution analysis"""
        insights = []
        
        for col, stats in distribution_stats.items():
            if stats["skewness"] > 1:
                insights.append(f"📊 {col} is highly right-skewed (skewness: {stats['skewness']:.2f})")
            elif stats["skewness"] < -1:
                insights.append(f"📊 {col} is highly left-skewed (skewness: {stats['skewness']:.2f})")
            
            if stats["kurtosis"] > 3:
                insights.append(f"📊 {col} has heavy tails (kurtosis: {stats['kurtosis']:.2f})")
            elif stats["kurtosis"] < -1:
                insights.append(f"📊 {col} has light tails (kurtosis: {stats['kurtosis']:.2f})")
            
            if stats["is_normal"]:
                insights.append(f"✅ {col} appears normally distributed")
        
        return insights

    def _generate_correlation_insights(self, target_correlations: Dict, high_corr_pairs: List) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []
        
        if target_correlations.get("strong_positive"):
            insights.append(f"🔗 Strong positive correlations with target: {list(target_correlations['strong_positive'].keys())}")
        
        if target_correlations.get("strong_negative"):
            insights.append(f"🔗 Strong negative correlations with target: {list(target_correlations['strong_negative'].keys())}")
        
        if high_corr_pairs:
            insights.append(f"⚠️ High correlation between features: {len(high_corr_pairs)} pairs (consider removing one)")
        
        if not target_correlations.get("strong_positive") and not target_correlations.get("strong_negative"):
            insights.append("ℹ️ No strong correlations with target found - may need feature engineering")
        
        return insights

    def _generate_missing_value_insights(self, missing_summary: pd.Series, missing_percentage: pd.Series) -> List[str]:
        """Generate insights from missing value analysis"""
        insights = []
        
        high_missing = missing_percentage[missing_percentage > 50]
        if len(high_missing) > 0:
            insights.append(f"⚠️ High missing values (>50%): {list(high_missing.index)}")
        
        moderate_missing = missing_percentage[(missing_percentage > 10) & (missing_percentage <= 50)]
        if len(moderate_missing) > 0:
            insights.append(f"ℹ️ Moderate missing values (10-50%): {list(moderate_missing.index)}")
        
        if missing_percentage.sum() == 0:
            insights.append("✅ No missing values found")
        
        return insights

    def _generate_outlier_insights(self, outlier_analysis: Dict) -> List[str]:
        """Generate insights from outlier analysis"""
        insights = []
        
        for col, stats in outlier_analysis.items():
            if stats["outlier_percentage"] > 5:
                insights.append(f"⚠️ {col} has {stats['outlier_percentage']:.1f}% outliers")
            elif stats["outlier_percentage"] > 1:
                insights.append(f"ℹ️ {col} has {stats['outlier_percentage']:.1f}% outliers")
        
        return insights

    def _generate_relationship_insights(self, relationship_stats: Dict) -> List[str]:
        """Generate insights from feature relationship analysis"""
        insights = []
        
        strong_features = [f for f, stats in relationship_stats.items() if stats["correlation_strength"] == "strong"]
        if strong_features:
            insights.append(f"🔗 Strong relationships with target: {strong_features}")
        
        weak_features = [f for f, stats in relationship_stats.items() if stats["correlation_strength"] == "weak"]
        if len(weak_features) > len(strong_features):
            insights.append("ℹ️ Most features have weak relationships with target - consider feature engineering")
        
        return insights

    def _generate_quality_recommendations(self, quality_metrics: Dict) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if quality_metrics["completeness"] < 0.9:
            recommendations.append("Consider imputation strategies for missing values")
        
        if quality_metrics["uniqueness"] < 0.95:
            recommendations.append("Investigate and remove duplicate records")
        
        if quality_metrics["consistency"] < 0.8:
            recommendations.append("Standardize data types and formats")
        
        if quality_metrics["validity"] < 0.9:
            recommendations.append("Validate data ranges and check for outliers")
        
        return recommendations

    async def _generate_eda_report(self, basic_stats: Dict, target_analysis: Dict, 
                                 distribution_analysis: Dict, correlation_analysis: Dict,
                                 missing_analysis: Dict, outlier_analysis: Dict,
                                 relationship_analysis: Dict, quality_assessment: Dict) -> str:
        """Generate comprehensive EDA report"""
        
        report = []
        report.append("=" * 80)
        report.append("📊 COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Dataset Overview
        report.append(f"\n📋 DATASET OVERVIEW:")
        report.append(f"  Shape: {basic_stats['dataset_shape']['rows']:,} rows × {basic_stats['dataset_shape']['columns']} columns")
        report.append(f"  Memory Usage: {basic_stats['memory_usage']['total_mb']:.2f} MB")
        report.append(f"  Data Types: {basic_stats['data_types']['numeric']} numeric, {basic_stats['data_types']['categorical']} categorical")
        
        # Data Quality
        report.append(f"\n✅ DATA QUALITY ASSESSMENT:")
        quality = quality_assessment['quality_metrics']
        report.append(f"  Completeness: {quality['completeness']:.1%}")
        report.append(f"  Uniqueness: {quality['uniqueness']:.1%}")
        report.append(f"  Consistency: {quality['consistency']:.1%}")
        report.append(f"  Validity: {quality['validity']:.1%}")
        report.append(f"  Overall Score: {quality['overall_score']:.1%}")
        
        # Missing Values
        report.append(f"\n🔍 MISSING VALUES:")
        missing_pct = basic_stats['missing_values']['percentage']
        if missing_pct > 0:
            report.append(f"  Total Missing: {basic_stats['missing_values']['total']:,} ({missing_pct:.1f}%)")
            high_missing = [col for col, pct in missing_analysis['missing_percentage'].items() if pct > 50]
            if high_missing:
                report.append(f"  High Missing (>50%): {high_missing}")
        else:
            report.append("  ✅ No missing values found")
        
        # Target Analysis
        if 'distribution' in target_analysis:
            report.append(f"\n🎯 TARGET VARIABLE ANALYSIS:")
            target_dist = target_analysis['distribution']
            report.append(f"  Unique Values: {target_dist['unique_values']}")
            report.append(f"  Missing: {target_dist['missing_count']} ({target_dist['missing_percentage']:.1f}%)")
            
            if 'class_balance' in target_analysis:
                balance = target_analysis['class_balance']
                report.append(f"  Class Balance: {'✅ Balanced' if balance['is_balanced'] else '⚠️ Imbalanced'}")
                if not balance['is_balanced']:
                    report.append(f"    Balance Ratio: {balance['balance_ratio']:.2f}")
        
        # Key Insights
        report.append(f"\n💡 KEY INSIGHTS:")
        all_insights = []
        all_insights.extend(target_analysis.get('insights', []))
        all_insights.extend(distribution_analysis.get('insights', []))
        all_insights.extend(correlation_analysis.get('insights', []))
        all_insights.extend(missing_analysis.get('insights', []))
        all_insights.extend(outlier_analysis.get('insights', []))
        all_insights.extend(relationship_analysis.get('insights', []))
        all_insights.extend(quality_assessment.get('insights', []))
        
        for i, insight in enumerate(all_insights[:10], 1):  # Top 10 insights
            report.append(f"  {i}. {insight}")
        
        # Recommendations
        report.append(f"\n📝 RECOMMENDATIONS:")
        recommendations = quality_assessment.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report.append(f"  {i}. {rec}")
        
        report.append(f"\n📊 VISUALIZATIONS GENERATED:")
        report.append(f"  - Target variable distribution and analysis")
        report.append(f"  - Feature distribution histograms")
        report.append(f"  - Correlation heatmap")
        report.append(f"  - Missing value patterns")
        report.append(f"  - Outlier analysis")
        report.append(f"  - Feature relationship scatter plots")
        
        report.append("\n" + "=" * 80)
        report.append("🎉 EDA Analysis Complete - Dataset ready for feature engineering!")
        report.append("=" * 80)
        
        return "\n".join(report)
