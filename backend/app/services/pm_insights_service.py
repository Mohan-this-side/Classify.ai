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
            
            # Model performance
            if metrics:
                accuracy = metrics.get("accuracy", 0)
                f1 = metrics.get("f1_weighted", 0)
                precision = metrics.get("precision_weighted", 0)
                recall = metrics.get("recall_weighted", 0)
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
            prompt = f"""You are a senior data scientist explaining classification results to a non-expert user. Generate a comprehensive summary with actionable insights.

**Classification Task Context:**
{context}

**Your Task:**
1. Provide a clear, simple summary of what was accomplished (2-3 sentences)
2. Explain the key findings in plain language that anyone can understand
3. List 3-5 actionable insights about what matters in their dataset
4. For each insight, explain:
   - What feature/pattern was found (use actual feature names from the dataset)
   - How it affects the target variable (use specific numbers/percentages if available)
   - Why it matters (in simple terms)
   - What action they should take (be specific and actionable)

**Format:**
Start with: "✅ Workflow Complete! We've successfully analyzed your dataset and built a classification model..."

Then list insights as:
"• [Feature Name] significantly affects [target variable]. [Specific finding, e.g., 'Employees with master's degrees are X% more likely to leave']. [Simple explanation of why this matters]. Focus on: [Specific actionable recommendation]."

**Example Format:**
"✅ Workflow Complete! We've successfully analyzed your dataset and built a classification model with [accuracy]% accuracy. Here are the key insights:

• Master's education level significantly affects employee leaving decisions. Employees with master's degrees are [X]% more likely to leave than those without. This suggests that higher education may correlate with better job opportunities elsewhere. Focus on: Understanding why higher education correlates with turnover and consider retention strategies specifically for this group, such as career development programs or competitive compensation packages.

• [Another insight with specific feature name and impact]..."

**Important:**
- Use actual feature names from the dataset
- Include specific numbers/percentages when available
- Make explanations simple and accessible
- Provide actionable recommendations
- Focus on what matters most for the classification task

**Generate the summary now:**"""
            
            # Use LLM to generate summary
            if self.llm_service and self.llm_service.clients:
                try:
                    if LLMProvider.GEMINI in self.llm_service.clients:
                        model = self.llm_service.clients[LLMProvider.GEMINI]
                        response = model.generate_content(prompt)
                        summary = response.text.strip()
                        return summary
                    elif LLMProvider.OPENAI in self.llm_service.clients:
                        import openai
                        client = self.llm_service.clients[LLMProvider.OPENAI]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=800
                        )
                        return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error generating workflow summary with LLM: {e}")
            
            # Fallback summary
            return self._generate_fallback_summary(state)
            
        except Exception as e:
            logger.error(f"Error generating workflow summary: {e}")
            return self._generate_fallback_summary(state)
    
    def _generate_fallback_summary(self, state: Dict[str, Any]) -> str:
        """Generate fallback summary without LLM"""
        dataset_shape = state.get("dataset_shape", [])
        target_column = state.get("target_column", "")
        metrics = state.get("evaluation_metrics", {})
        feature_importance = state.get("feature_importance_model", {})
        
        summary_parts = [
            f"✅ Workflow Complete!",
            f"\nWe've successfully analyzed your dataset ({dataset_shape[0]} rows × {dataset_shape[1]} columns) and built a classification model for '{target_column}'."
        ]
        
        if metrics:
            accuracy = metrics.get("accuracy", 0)
            summary_parts.append(f"\nModel Performance: {accuracy*100:.1f}% accuracy")
        
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:5]
            summary_parts.append(f"\nKey Insights:")
            for feat, imp in top_features:
                summary_parts.append(f"• {feat} is a key factor (importance: {imp:.3f})")
        
        summary_parts.append(f"\nYou can now ask me questions about feature importance, model performance, or next steps!")
        
        return "\n".join(summary_parts)
    
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

