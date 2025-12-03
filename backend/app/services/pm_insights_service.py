"""
PM Insights Service - Generates comprehensive insights and action items for completed workflows
"""

import logging
from typing import Dict, Any, List, Optional
from ..services.llm_service import get_llm_service, LLMProvider

logger = logging.getLogger(__name__)


class PMInsightsService:
    """Service to generate comprehensive insights and action items for workflows"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PM insights service.
        
        Args:
            api_key: User-provided API key (optional, uses environment if not provided)
        """
        self.api_key = api_key
        self.llm_service = get_llm_service(api_key=api_key) if api_key else get_llm_service()
    
    async def generate_workflow_summary(self, state: Dict[str, Any]) -> str:
        """
        Generate a comprehensive workflow completion summary with actionable insights.
        
        Args:
            state: Complete workflow state
            
        Returns:
            Human-readable summary with action items
        """
        try:
            # Extract key information
            dataset_shape = state.get("dataset_shape", [])
            target_column = state.get("target_column", "")
            user_description = state.get("user_description", "")
            best_model = state.get("best_model", "")
            metrics = state.get("evaluation_metrics", {})
            
            # ✅ CRITICAL FIX: Normalize metrics to match top bar (same as get_workflow_results)
            if metrics:
                is_binary = metrics.get("is_binary", False)
                if is_binary:
                    # Binary classification: use binary metrics with fallback to weighted
                    if "precision" not in metrics:
                        metrics["precision"] = metrics.get("precision_binary", metrics.get("precision_weighted", 0))
                    if "recall" not in metrics:
                        metrics["recall"] = metrics.get("recall_binary", metrics.get("recall_weighted", 0))
                    if "f1_score" not in metrics:
                        metrics["f1_score"] = metrics.get("f1_binary", metrics.get("f1_weighted", 0))
                else:
                    # Multi-class: use weighted averages
                    if "precision" not in metrics:
                        metrics["precision"] = metrics.get("precision_weighted", metrics.get("precision_macro", 0))
                    if "recall" not in metrics:
                        metrics["recall"] = metrics.get("recall_weighted", metrics.get("recall_macro", 0))
                    if "f1_score" not in metrics:
                        metrics["f1_score"] = metrics.get("f1_weighted", metrics.get("f1_macro", 0))
            
            feature_importance = state.get("feature_importance_model", {})
            eda_plots = state.get("eda_plots", [])
            cleaning_actions = state.get("cleaning_actions_taken", [])
            engineered_features = state.get("engineered_features", [])
            correlation_analysis = state.get("correlation_analysis", {})
            
            # Build comprehensive context for LLM
            context_parts = []
            
            # Dataset info
            if dataset_shape:
                context_parts.append(f"Dataset: {dataset_shape[0]} rows × {dataset_shape[1]} columns")
            if target_column:
                context_parts.append(f"Target Variable: {target_column}")
            if user_description:
                context_parts.append(f"User Goal: {user_description}")
            
            # Model performance - ✅ FIX: Use same keys as top bar (f1_score, precision, recall) with fallbacks
            if metrics:
                accuracy = metrics.get("accuracy", 0)
                f1 = metrics.get("f1_score", metrics.get("f1_weighted", metrics.get("f1_binary", 0)))
                precision = metrics.get("precision", metrics.get("precision_weighted", metrics.get("precision_binary", 0)))
                recall = metrics.get("recall", metrics.get("recall_weighted", metrics.get("recall_binary", 0)))
                context_parts.append(f"Model Performance: Accuracy={accuracy*100:.1f}%, F1={f1:.3f}, Precision={precision*100:.1f}%, Recall={recall*100:.1f}%")
            
            if best_model:
                context_parts.append(f"Best Model: {best_model}")
            
            # Feature importance
            if feature_importance:
                top_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                    reverse=True
                )[:10]
                context_parts.append(f"Top Important Features: {', '.join([f'{feat}({imp:.3f})' for feat, imp in top_features])}")
            
            # EDA findings
            if correlation_analysis:
                top_corr = correlation_analysis.get("top_correlations", [])
                if top_corr:
                    context_parts.append(f"Key Correlations: {str(top_corr)[:300]}")
            
            # Data cleaning
            if cleaning_actions:
                context_parts.append(f"Data Cleaning Actions: {', '.join(cleaning_actions[:5])}")
            
            # Feature engineering
            if engineered_features:
                context_parts.append(f"Engineered Features: {', '.join(engineered_features[:5])}")
            
            context = "\n".join(context_parts)
            
            # Build comprehensive prompt for LLM
            prompt = f"""You are a senior data scientist explaining classification results to a non-expert user. Generate a comprehensive, well-formatted summary with actionable insights.

**Classification Task Context:**
{context}

**Your Task:**
Generate a comprehensive summary that includes:

1. **Executive Summary** (2-3 sentences): What was accomplished
2. **Model Performance Table**: Create a markdown table showing key metrics
3. **Key Insights** (3-5 insights): Each insight should include:
   - Feature name and its impact
   - Specific numbers/percentages
   - Why it matters
   - Actionable recommendation
4. **Top Features Table**: Table showing top 5-10 most important features
5. **Action Items**: Bulleted list of next steps

**Format Requirements:**
- Use markdown formatting (headers, tables, bullet points)
- Use proper markdown table syntax for tables
- Make it readable and professional
- Include specific numbers and percentages
- Use clear section headers (## for main sections, ### for subsections)
- **CRITICAL: F1 Score must ALWAYS be shown as a decimal (e.g., 0.853) NEVER as a percentage (e.g., 85.3%)**
- Accuracy, Precision, and Recall should be shown as percentages (e.g., 85.3%)

**Example Format:**

# Comprehensive Analysis of {target_column.replace('_', ' ').title()} Prediction Model

We have successfully developed and evaluated a machine learning model designed to predict {target_column.replace('_', ' ')} based on {dataset_shape[1] if dataset_shape else 'N/A'} key features.

## Executive Summary

✅ Workflow Complete! We have analyzed your dataset ({dataset_shape[0] if dataset_shape else 'N/A'} rows × {dataset_shape[1] if dataset_shape else 'N/A'} columns) and built a robust classification model using the {best_model if best_model != 'Unknown' else 'selected'} algorithm to predict '{target_column.replace('_', ' ')}'. The final model achieved strong predictive capability across all metrics, with an overall Accuracy of {metrics.get('accuracy', 0)*100:.1f}%, making it a highly reliable tool for supporting early risk detection.

## Model Performance

The {best_model if best_model != 'Unknown' else 'selected'} model demonstrated strong performance in identifying individuals at risk for {target_column.replace('_', ' ')} (Class 1).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | {metrics.get('accuracy', 0)*100:.1f}% | Overall percentage of correct predictions (both disease and normal). |
| **F1 Score** | {metrics.get('f1_score', metrics.get('f1_weighted', metrics.get('f1_binary', 0))):.3f} | The balance between Precision and Recall—a high score indicating excellent overall test performance. |
| **Precision** | {metrics.get('precision', metrics.get('precision_weighted', metrics.get('precision_binary', 0)))*100:.1f}% | Out of all patients predicted to have {target_column.replace('_', ' ')}, {metrics.get('precision', metrics.get('precision_weighted', metrics.get('precision_binary', 0)))*100:.1f}% actually had it (minimizes false alarms). |
| **Recall** | {metrics.get('recall', metrics.get('recall_weighted', metrics.get('recall_binary', 0)))*100:.1f}% | Out of all patients who actually have {target_column.replace('_', ' ')}, {metrics.get('recall', metrics.get('recall_weighted', metrics.get('recall_binary', 0)))*100:.1f}% were correctly identified (minimizes missed cases). |

## Key Insights

### 1. [Feature Name] Impact
[Feature Name] significantly affects {target_column}. [Specific finding with numbers]. [Simple explanation]. 

**Action Item:** [Specific actionable recommendation]

### 2. [Another Feature]
[Description with numbers and impact]

**Action Item:** [Recommendation]

## Top Important Features

| Feature | Importance | Impact |
|---------|------------|--------|
| [Feature 1] | [Value] | [Description] |
| [Feature 2] | [Value] | [Description] |

## Recommended Next Steps

• [Action item 1]
• [Action item 2]
• [Action item 3]

**Important:**
- Use actual feature names from the dataset
- Include specific numbers/percentages when available
- Make explanations simple and accessible
- Provide actionable recommendations
- Format tables properly using markdown syntax
- Use proper markdown headers and formatting

**Generate the summary now:**"""
            
            # Use LLM to generate summary
            if self.llm_service and self.llm_service.clients:
                try:
                    if LLMProvider.GEMINI in self.llm_service.clients:
                        model = self.llm_service.clients[LLMProvider.GEMINI]
                        response = model.generate_content(prompt)
                        summary = response.text.strip()
                        # Clean up the summary: remove any '---' separators and improve formatting
                        summary = summary.replace("---", "")
                        # Remove excessive blank lines
                        import re
                        summary = re.sub(r'\n{3,}', '\n\n', summary)
                        # Ensure proper spacing around headers
                        summary = re.sub(r'\n(##+)\s+', r'\n\n\1 ', summary)
                        summary = summary.strip()
                        return summary
                    elif LLMProvider.OPENAI in self.llm_service.clients:
                        import openai
                        client = self.llm_service.clients[LLMProvider.OPENAI]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1200
                        )
                        summary = response.choices[0].message.content.strip()
                        # Clean up the summary: remove any '---' separators and improve formatting
                        summary = summary.replace("---", "")
                        # Remove excessive blank lines
                        import re
                        summary = re.sub(r'\n{3,}', '\n\n', summary)
                        # Ensure proper spacing around headers
                        summary = re.sub(r'\n(##+)\s+', r'\n\n\1 ', summary)
                        summary = summary.strip()
                        return summary
                except Exception as e:
                    logger.error(f"Error generating workflow summary with LLM: {e}")
            
            # Fallback summary
            return self._generate_fallback_summary(state)
            
        except Exception as e:
            logger.error(f"Error generating workflow summary: {e}")
            return self._generate_fallback_summary(state)
    
    def _generate_fallback_summary(self, state: Dict[str, Any]) -> str:
        """Generate fallback summary without LLM - formatted with markdown"""
        dataset_shape = state.get("dataset_shape", [])
        target_column = state.get("target_column", "")
        metrics = state.get("evaluation_metrics", {})
        
        # ✅ CRITICAL FIX: Normalize metrics to match top bar (same as get_workflow_results)
        if metrics:
            is_binary = metrics.get("is_binary", False)
            if is_binary:
                # Binary classification: use binary metrics with fallback to weighted
                if "precision" not in metrics:
                    metrics["precision"] = metrics.get("precision_binary", metrics.get("precision_weighted", 0))
                if "recall" not in metrics:
                    metrics["recall"] = metrics.get("recall_binary", metrics.get("recall_weighted", 0))
                if "f1_score" not in metrics:
                    metrics["f1_score"] = metrics.get("f1_binary", metrics.get("f1_weighted", 0))
            else:
                # Multi-class: use weighted averages
                if "precision" not in metrics:
                    metrics["precision"] = metrics.get("precision_weighted", metrics.get("precision_macro", 0))
                if "recall" not in metrics:
                    metrics["recall"] = metrics.get("recall_weighted", metrics.get("recall_macro", 0))
                if "f1_score" not in metrics:
                    metrics["f1_score"] = metrics.get("f1_weighted", metrics.get("f1_macro", 0))
        
        feature_importance = state.get("feature_importance_model", {})
        best_model = state.get("best_model", "Unknown")
        cleaning_actions = state.get("cleaning_actions_taken", [])
        
        summary_parts = [
            f"# Comprehensive Analysis of {target_column.replace('_', ' ').title()} Prediction Model",
            "",
            f"We have successfully developed and evaluated a machine learning model designed to predict {target_column.replace('_', ' ')} based on {dataset_shape[1] if dataset_shape else 'N/A'} key features.",
            "",
            "## Executive Summary",
            "",
            f"✅ **Workflow Complete!** We have analyzed your dataset ({dataset_shape[0] if dataset_shape else 'N/A'} rows × {dataset_shape[1] if dataset_shape else 'N/A'} columns) and built a robust classification model using the {best_model if best_model != 'Unknown' else 'selected'} algorithm to predict **{target_column.replace('_', ' ')}**. The final model achieved strong predictive capability across all metrics, with an overall Accuracy of {metrics.get('accuracy', 0)*100:.1f}%, making it a highly reliable tool for supporting early risk detection."
        ]
        
        if metrics:
            accuracy = metrics.get("accuracy", 0)
            # ✅ FIX: Use same keys as top bar (f1_score, precision, recall) with same fallback chain
            f1 = metrics.get("f1_score", metrics.get("f1_weighted", metrics.get("f1_binary", 0)))
            precision = metrics.get("precision", metrics.get("precision_weighted", metrics.get("precision_binary", 0)))
            recall = metrics.get("recall", metrics.get("recall_weighted", metrics.get("recall_binary", 0)))
            
            summary_parts.extend([
                "",
                "## Model Performance",
                "",
                f"The {best_model if best_model != 'Unknown' else 'selected'} model demonstrated strong performance in identifying individuals at risk for {target_column.replace('_', ' ')} (Class 1).",
                "",
                "| Metric | Value | Interpretation |",
                "|--------|-------|----------------|",
                f"| **Accuracy** | {accuracy*100:.1f}% | Overall percentage of correct predictions (both disease and normal). |",
                f"| **F1 Score** | {f1:.3f} | The balance between Precision and Recall—a high score indicating excellent overall test performance. |",
                f"| **Precision** | {precision*100:.1f}% | Out of all patients predicted to have {target_column.replace('_', ' ')}, {precision*100:.1f}% actually had it (minimizes false alarms). |",
                f"| **Recall** | {recall*100:.1f}% | Out of all patients who actually have {target_column.replace('_', ' ')}, {recall*100:.1f}% were correctly identified (minimizes missed cases). |",
                ""
            ])
        
        if best_model and best_model != "Unknown":
            summary_parts.extend([
                f"**Best Model:** {best_model}",
                ""
            ])
        
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:10]
            
            summary_parts.extend([
                "## Top Important Features",
                "",
                "The following features have the strongest predictive power for identifying {target_column.replace('_', ' ')}:",
                "",
                "| Feature | Importance Score |",
                "|---------|------------------|"
            ])
            
            for feat, imp in top_features[:10]:
                summary_parts.append(f"| {feat.replace('_', ' ').title()} | {imp:.3f} |")
            
            summary_parts.extend([
                "",
                "## Key Insights",
                ""
            ])
            
            summary_parts.extend([
                "",
                "## Key Insights and Actionable Recommendations",
                "",
                f"The model highlights specific features that drive the strongest predictive power. Understanding these features is critical for informed decision-making.",
                ""
            ])
            
            for i, (feat, imp) in enumerate(top_features[:5], 1):
                summary_parts.extend([
                    f"### {i}. {feat.replace('_', ' ').title()} Impact",
                    "",
                    f"**Impact:** {feat.replace('_', ' ').title()} is a key factor in predicting {target_column.replace('_', ' ')} with an importance score of {imp:.3f}. This feature significantly influences the model's predictions.",
                    "",
                    f"**Why it Matters:** Understanding the role of {feat.replace('_', ' ').lower()} helps identify critical patterns in the data that drive classification decisions.",
                    "",
                    f"**Action Item:** Monitor and analyze {feat.replace('_', ' ').lower()} values closely, as they are among the most influential factors in the model's predictions.",
                    ""
                ])
        
        if cleaning_actions:
            summary_parts.extend([
                "## Data Quality Improvements",
                ""
            ])
            for action in cleaning_actions[:5]:
                summary_parts.append(f"• {action}")
            summary_parts.append("")
        
        summary_parts.extend([
            "## Recommended Next Steps",
            "",
            "• **Review Model Performance:** Examine the metrics above to understand the model's strengths and limitations",
            "• **Analyze Feature Importance:** Use the feature rankings to identify the most critical factors",
            "• **Download Resources:** Access the trained model, cleaned dataset, and analysis notebook for further exploration",
            "• **Validate Predictions:** Test the model on new data to ensure consistent performance",
            "• **Iterate and Improve:** Consider feature engineering or model tuning based on insights gained"
        ])
        
        # Clean up the summary: remove any '---' separators and improve formatting
        summary_text = "\n".join(summary_parts)
        # Remove markdown horizontal rules (---)
        summary_text = summary_text.replace("---", "")
        # Remove excessive blank lines (more than 2 consecutive)
        import re
        summary_text = re.sub(r'\n{3,}', '\n\n', summary_text)
        # Ensure proper spacing around headers
        summary_text = re.sub(r'\n(##+)\s+', r'\n\n\1 ', summary_text)
        # Clean up any remaining formatting issues
        summary_text = summary_text.strip()
        
        return summary_text
    
    async def generate_feature_insights(self, state: Dict[str, Any], question: str) -> str:
        """
        Generate insights about features based on user question.
        
        Args:
            state: Workflow state
            question: User's question about features
            
        Returns:
            Detailed feature insights
        """
        try:
            feature_importance = state.get("feature_importance_model", {})
            correlation_analysis = state.get("correlation_analysis", {})
            target_column = state.get("target_column", "")
            metrics = state.get("evaluation_metrics", {})
            
            # Build context
            context_parts = []
            
            if feature_importance:
                top_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                    reverse=True
                )[:10]
                context_parts.append(f"Feature Importance Rankings: {', '.join([f'{feat}({imp:.3f})' for feat, imp in top_features])}")
            
            if correlation_analysis:
                top_corr = correlation_analysis.get("top_correlations", [])
                if top_corr:
                    context_parts.append(f"Key Correlations: {str(top_corr)[:300]}")
            
            if target_column:
                context_parts.append(f"Target Variable: {target_column}")
            
            if metrics:
                accuracy = metrics.get("accuracy", 0)
                context_parts.append(f"Model Accuracy: {accuracy*100:.1f}%")
            
            context = "\n".join(context_parts)
            
            prompt = f"""You are a senior data scientist helping a user understand their classification model results.

**Context:**
{context}

**User Question:** {question}

**Your Task:**
- Answer as a senior data scientist would explain to a non-expert
- Use the feature importance and correlation data to provide specific insights
- Explain which features matter most and why
- Provide actionable recommendations
- Use simple language but be specific about numbers and impacts

**Answer:**"""
            
            # Use LLM
            if self.llm_service and self.llm_service.clients:
                try:
                    if LLMProvider.GEMINI in self.llm_service.clients:
                        model = self.llm_service.clients[LLMProvider.GEMINI]
                        response = model.generate_content(prompt)
                        return response.text.strip()
                    elif LLMProvider.OPENAI in self.llm_service.clients:
                        import openai
                        client = self.llm_service.clients[LLMProvider.OPENAI]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500
                        )
                        return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error generating feature insights: {e}")
            
            # Fallback
            return self._generate_fallback_feature_answer(state, question)
            
        except Exception as e:
            logger.error(f"Error generating feature insights: {e}")
            return self._generate_fallback_feature_answer(state, question)
    
    def _generate_fallback_feature_answer(self, state: Dict[str, Any], question: str) -> str:
        """Fallback feature answer"""
        feature_importance = state.get("feature_importance_model", {})
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:5]
            return f"Based on the analysis, the most important features are: {', '.join([f'{feat} (importance: {imp:.3f})' for feat, imp in top_features])}. These features have the strongest impact on predicting {state.get('target_column', 'the target variable')}."
        return "I can help you understand feature importance. Please ask a specific question about your model's features."


# Singleton instance (for backward compatibility, will be created with API key when needed)
_pm_insights_service = None

def get_pm_insights_service(api_key: Optional[str] = None) -> PMInsightsService:
    """
    Get PM insights service instance.
    
    Args:
        api_key: User-provided API key (creates new instance if provided)
    
    Returns:
        PMInsightsService instance
    """
    global _pm_insights_service
    # If API key is provided, create a new instance
    if api_key:
        return PMInsightsService(api_key=api_key)
    # Otherwise use singleton for backward compatibility
    if _pm_insights_service is None:
        _pm_insights_service = PMInsightsService()
    return _pm_insights_service

# Backward compatibility singleton
pm_insights_service = PMInsightsService()

