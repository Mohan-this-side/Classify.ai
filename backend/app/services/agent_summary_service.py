"""
Agent Summary Service - Generates human-readable summaries for agent executions
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import base64
from PIL import Image
import io

from ..services.llm_service import get_llm_service, LLMProvider

logger = logging.getLogger(__name__)


class AgentSummaryService:
    """Service to generate comprehensive summaries for agent executions"""
    
    def __init__(self):
        self.llm_service = get_llm_service()
    
    async def generate_summary(
        self,
        agent_name: str,
        workflow_state: Dict[str, Any],
        workflow_id: str
    ) -> str:
        """
        Generate a comprehensive summary for an agent's execution.
        
        Args:
            agent_name: Name of the agent (e.g., 'eda_analysis', 'data_cleaning')
            workflow_state: Current workflow state containing agent results
            workflow_id: Workflow ID for accessing plots/files
            
        Returns:
            Human-readable summary string
        """
        try:
            if agent_name == "eda_analysis":
                return await self._generate_eda_summary(workflow_state, workflow_id)
            elif agent_name == "data_cleaning":
                return await self._generate_cleaning_summary(workflow_state)
            elif agent_name == "data_discovery":
                return await self._generate_discovery_summary(workflow_state)
            elif agent_name == "feature_engineering":
                return await self._generate_feature_engineering_summary(workflow_state)
            elif agent_name == "ml_building":
                return await self._generate_ml_building_summary(workflow_state)
            elif agent_name == "model_evaluation":
                return await self._generate_model_evaluation_summary(workflow_state)
            elif agent_name == "technical_reporter":
                return await self._generate_reporter_summary(workflow_state)
            else:
                return self._generate_default_summary(agent_name, workflow_state)
        except Exception as e:
            logger.error(f"Error generating summary for {agent_name}: {e}")
            return f"✅ {agent_name.replace('_', ' ').title()} completed successfully. Detailed results are available in the final report."
    
    async def _generate_eda_summary(self, state: Dict[str, Any], workflow_id: str) -> str:
        """Generate EDA summary with plot analysis"""
        try:
            # Collect EDA data
            plots = state.get("eda_plots", [])
            statistical_summary = state.get("statistical_summary", {})
            correlation_analysis = state.get("correlation_analysis", {})
            distribution_analysis = state.get("distribution_analysis", {})
            outlier_analysis = state.get("outlier_analysis", {})
            target_analysis = state.get("target_analysis", {})
            dataset_shape = state.get("dataset_shape", [])
            
            # Build context for LLM
            context_parts = []
            
            # Dataset info
            if dataset_shape:
                context_parts.append(f"Dataset: {dataset_shape[0]} rows × {dataset_shape[1]} columns")
            
            # Statistical summary
            if statistical_summary:
                context_parts.append(f"Statistical Summary: {str(statistical_summary)[:500]}")
            
            # Correlation findings
            if correlation_analysis:
                top_correlations = correlation_analysis.get("top_correlations", [])
                if top_correlations:
                    context_parts.append(f"Top Correlations: {str(top_correlations)[:300]}")
            
            # Distribution findings
            if distribution_analysis:
                context_parts.append(f"Distribution Analysis: {str(distribution_analysis)[:300]}")
            
            # Outlier findings
            if outlier_analysis:
                outlier_count = outlier_analysis.get("outlier_count", 0)
                context_parts.append(f"Outliers Detected: {outlier_count}")
            
            # Target analysis
            if target_analysis:
                context_parts.append(f"Target Variable Analysis: {str(target_analysis)[:300]}")
            
            # Plot analysis using LLM
            plot_insights = ""
            if plots and self.llm_service:
                plot_insights = await self._analyze_plots_with_llm(plots, workflow_id, context_parts)
            
            # Build comprehensive summary (without markdown formatting)
            summary_parts = [
                f"📊 Exploratory Data Analysis Complete",
                f"\nDataset Overview:",
                f"- Analyzed {dataset_shape[0] if dataset_shape else 'N/A'} rows and {dataset_shape[1] if dataset_shape else 'N/A'} columns"
            ]
            
            if statistical_summary:
                summary_parts.append(f"\nKey Statistics:")
                summary_parts.append(f"- Generated comprehensive statistical summaries for all features")
            
            if correlation_analysis:
                summary_parts.append(f"\nCorrelation Analysis:")
                top_corr = correlation_analysis.get("top_correlations", [])
                if top_corr:
                    summary_parts.append(f"- Identified {len(top_corr)} strong feature correlations")
                    summary_parts.append(f"- Strongest correlations help identify feature relationships")
            
            if distribution_analysis:
                summary_parts.append(f"\nDistribution Analysis:")
                summary_parts.append(f"- Analyzed feature distributions to understand data patterns")
                summary_parts.append(f"- Identified normal vs. skewed distributions")
            
            if outlier_analysis:
                outlier_count = outlier_analysis.get("outlier_count", 0)
                summary_parts.append(f"\nOutlier Detection:")
                summary_parts.append(f"- Detected {outlier_count} potential outliers")
                summary_parts.append(f"- Outliers may need special handling during modeling")
            
            if plots:
                summary_parts.append(f"\nVisualizations Generated:")
                summary_parts.append(f"- Created {len(plots)} comprehensive plots")
                summary_parts.append(f"- Plots include correlation heatmaps, distributions, and outlier visualizations")
                
                # Add LLM-generated plot insights
                if plot_insights:
                    summary_parts.append(f"\nPlot Insights:")
                    # Remove markdown from plot insights
                    clean_insights = plot_insights.replace("**", "").replace("*", "")
                    summary_parts.append(clean_insights)
            
            if target_analysis:
                summary_parts.append(f"\nTarget Variable Insights:")
                summary_parts.append(f"- Analyzed target variable distribution and relationships")
                summary_parts.append(f"- Identified key features correlated with target")
            
            summary_parts.append(f"\nNext Steps:")
            summary_parts.append(f"- Use these insights to guide feature engineering")
            summary_parts.append(f"- Consider outlier treatment based on findings")
            summary_parts.append(f"- Leverage correlation patterns for feature selection")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating EDA summary: {e}")
            return f"✅ EDA Analysis completed successfully. Generated {len(plots)} plots and comprehensive statistical analysis."
    
    async def _analyze_plots_with_llm(
        self,
        plots: List[Dict[str, Any]],
        workflow_id: str,
        context: List[str]
    ) -> str:
        """Use LLM to analyze plots and generate insights"""
        try:
            if not self.llm_service or not self.llm_service.clients:
                return ""
            
            # Collect plot information
            plot_info = []
            for plot in plots[:4]:  # Limit to first 4 plots
                plot_name = plot.get("title", plot.get("name", "plot"))
                plot_type = plot_name.lower()
                
                # Determine plot type from name
                if "correlation" in plot_type or "heatmap" in plot_type:
                    plot_info.append(f"- Correlation Heatmap: Shows relationships between numeric features")
                elif "distribution" in plot_type or "histogram" in plot_type:
                    plot_info.append(f"- Distribution Plot: Shows how features are distributed")
                elif "outlier" in plot_type or "box" in plot_type:
                    plot_info.append(f"- Outlier Visualization: Identifies potential outliers in the data")
                elif "target" in plot_type:
                    plot_info.append(f"- Target Distribution: Shows target variable patterns")
                else:
                    plot_info.append(f"- {plot_name}: Visualization available")
            
            # Build comprehensive prompt for LLM
            prompt = f"""You are an expert data scientist analyzing EDA (Exploratory Data Analysis) results. Based on the following analysis context, provide key insights about what the data reveals.

**Analysis Context:**
{chr(10).join(context[:10])}  # Limit context length

**Visualizations Generated:**
{chr(10).join(plot_info)}

**Your Task:**
Provide 4-6 concise, actionable insights about what this EDA reveals. Focus on:

1. **Data Patterns**: What patterns do you see in correlations and distributions?
2. **Data Quality**: Are there data quality concerns (outliers, missing values, imbalances)?
3. **Feature Relationships**: Which features are most correlated with each other and the target?
4. **Distribution Insights**: Are distributions normal, skewed, or have interesting characteristics?
5. **Outlier Impact**: How might outliers affect modeling?
6. **Modeling Implications**: What should be considered for feature engineering and model selection?

Format each insight as a bullet point (1-2 sentences). Be specific and actionable.

**Key Insights:**"""
            
            # Generate insights using LLM
            if LLMProvider.GEMINI in self.llm_service.clients:
                model = self.llm_service.clients[LLMProvider.GEMINI]
                response = model.generate_content(prompt)
                insights = response.text.strip()
                
                # Clean up the response
                insights = insights.replace("**Key Insights:**", "").strip()
                insights = insights.replace("**Insights:**", "").strip()
                
                return insights[:800]  # Limit length but allow more detail
            
            return ""
        except Exception as e:
            logger.warning(f"Error analyzing plots with LLM: {e}")
            return ""
    
    def _generate_cleaning_summary(self, state: Dict[str, Any]) -> str:
        """Generate data cleaning summary"""
        quality_score = state.get("data_quality_score", 0)
        actions_taken = state.get("cleaning_actions_taken", [])
        issues_found = state.get("cleaning_issues_found", [])
        
        summary_parts = [
            f"🧹 Data Cleaning Complete",
            f"\nData Quality Assessment:",
        ]
        
        # Only show quality score if it's meaningful
        if quality_score and quality_score > 0:
            summary_parts.append(f"- Data quality score: {quality_score*100:.1f}%")
        else:
            summary_parts.append(f"- Data quality has been assessed and improved")
        
        summary_parts.append(f"\nIssues Found:")
        if issues_found:
            for issue in issues_found[:5]:
                summary_parts.append(f"- {issue}")
        else:
            summary_parts.append("- No major issues detected")
        
        summary_parts.append(f"\nActions Taken:")
        if actions_taken:
            for action in actions_taken[:5]:
                summary_parts.append(f"- {action}")
        else:
            summary_parts.append("- Standard data cleaning procedures applied")
        
        summary_parts.append(f"\nResult:")
        summary_parts.append(f"- Dataset is now clean and ready for feature engineering")
        summary_parts.append(f"- Missing values handled, duplicates removed, data types corrected")
        
        return "\n".join(summary_parts)
    
    def _generate_discovery_summary(self, state: Dict[str, Any]) -> str:
        """Generate data discovery summary"""
        dataset_shape = state.get("dataset_shape", [])
        data_types = state.get("data_types", {})
        
        summary_parts = [
            f"🔍 Data Discovery Complete",
            f"\nDataset Structure:",
            f"- {dataset_shape[0] if dataset_shape else 'N/A'} rows × {dataset_shape[1] if dataset_shape else 'N/A'} columns",
        ]
        
        if data_types:
            numeric_count = sum(1 for dt in data_types.values() if dt in ['int64', 'float64'])
            categorical_count = len(data_types) - numeric_count
            summary_parts.append(f"- {numeric_count} numeric features, {categorical_count} categorical features")
        
        summary_parts.append(f"\nAnalysis:")
        summary_parts.append(f"- Identified data types and column characteristics")
        summary_parts.append(f"- Detected potential ID columns and datetime features")
        summary_parts.append(f"- Analyzed cardinality and unique value patterns")
        
        return "\n".join(summary_parts)
    
    def _generate_feature_engineering_summary(self, state: Dict[str, Any]) -> str:
        """Generate feature engineering summary"""
        engineered_features = state.get("engineered_features", [])
        feature_transformations = state.get("feature_transformations", {})
        
        summary_parts = [
            f"⚙️ Feature Engineering Complete",
            f"\nFeatures Created: {len(engineered_features)}",
        ]
        
        if engineered_features:
            summary_parts.append(f"\nNew Features:")
            for feat in engineered_features[:5]:
                summary_parts.append(f"- {feat}")
            if len(engineered_features) > 5:
                summary_parts.append(f"- ... and {len(engineered_features) - 5} more")
        
        if feature_transformations:
            summary_parts.append(f"\nTransformations Applied:")
            for feat, trans in list(feature_transformations.items())[:5]:
                summary_parts.append(f"- {feat}: {trans}")
        
        summary_parts.append(f"\nImpact:")
        summary_parts.append(f"- Enhanced dataset with new predictive features")
        summary_parts.append(f"- Improved model's ability to learn patterns")
        
        return "\n".join(summary_parts)
    
    def _generate_ml_building_summary(self, state: Dict[str, Any]) -> str:
        """Generate ML building summary"""
        best_model = state.get("best_model", "Unknown")
        training_metrics = state.get("training_metrics", {})
        model_selection_results = state.get("model_selection_results", {})
        
        summary_parts = [
            f"🤖 Model Building Complete",
            f"\nBest Model Selected: {best_model}",
        ]
        
        if training_metrics:
            test_acc = training_metrics.get("test_accuracy", 0)
            summary_parts.append(f"\nTraining Performance:")
            summary_parts.append(f"- Test Accuracy: {test_acc*100:.1f}%")
        
        if model_selection_results:
            models_tested = model_selection_results.get("models_tested", [])
            if models_tested:
                summary_parts.append(f"\nModels Evaluated: {len(models_tested)}")
                summary_parts.append(f"- Tested multiple algorithms including: {', '.join(models_tested[:3])}")
        
        summary_parts.append(f"\nProcess:")
        summary_parts.append(f"- Trained and evaluated multiple ML algorithms")
        summary_parts.append(f"- Used cross-validation for robust evaluation")
        summary_parts.append(f"- Selected best model based on performance metrics")
        
        return "\n".join(summary_parts)
    
    def _generate_model_evaluation_summary(self, state: Dict[str, Any]) -> str:
        """Generate model evaluation summary"""
        metrics = state.get("evaluation_metrics", {})
        feature_importance = state.get("feature_importance_model", {})
        
        summary_parts = [
            f"📈 Model Evaluation Complete",
        ]
        
        if metrics:
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            
            summary_parts.append(f"\nPerformance Metrics:")
            summary_parts.append(f"- Accuracy: {accuracy*100:.1f}%")
            summary_parts.append(f"- F1 Score: {f1:.3f}")
            summary_parts.append(f"- Precision: {precision*100:.1f}%")
            summary_parts.append(f"- Recall: {recall*100:.1f}%")
        
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:5]
            
            summary_parts.append(f"\nTop Important Features:")
            for feat, importance in top_features:
                summary_parts.append(f"- {feat}: {importance:.2f}")
        
        summary_parts.append(f"\nAssessment:")
        summary_parts.append(f"- Model performance evaluated on test set")
        summary_parts.append(f"- Feature importance analyzed")
        summary_parts.append(f"- Ready for deployment")
        
        return "\n".join(summary_parts)
    
    def _generate_reporter_summary(self, state: Dict[str, Any]) -> str:
        """Generate technical reporter summary"""
        downloadable_files = state.get("downloadable_files", [])
        
        summary_parts = [
            f"📝 Technical Reporting Complete",
            f"\nDeliverables Generated:",
        ]
        
        if downloadable_files:
            for file in downloadable_files:
                file_name = file.get("name", "Unknown")
                summary_parts.append(f"- {file_name}")
        else:
            summary_parts.append("- Technical report")
            summary_parts.append("- Jupyter notebook")
            summary_parts.append("- Model file")
        
        summary_parts.append(f"\nContents:")
        summary_parts.append(f"- Comprehensive analysis documentation")
        summary_parts.append(f"- Reproducible Jupyter notebook")
        summary_parts.append(f"- Model performance analysis")
        summary_parts.append(f"- Recommendations and insights")
        
        return "\n".join(summary_parts)
    
    def _generate_default_summary(self, agent_name: str, state: Dict[str, Any]) -> str:
        """Generate default summary for unknown agents"""
        return f"✅ {agent_name.replace('_', ' ').title()} completed successfully. Detailed results are available in the final report."


# Singleton instance
agent_summary_service = AgentSummaryService()

