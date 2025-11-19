"""
LLM-as-Judge Evaluator
Uses LLM to evaluate qualitative aspects of agent outputs.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import yaml
import re

# Add backend to path to import LLM service
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

try:
    from app.services.llm_service import LLMService, LLMProvider
except ImportError:
    LLMService = None
    LLMProvider = None

logger = logging.getLogger(__name__)


class LLMJudge:
    """Uses LLM to evaluate qualitative aspects of agent outputs."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the LLM-as-judge evaluator."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize LLM service
        if LLMService is None:
            logger.warning("LLMService not available. LLM-as-judge will be disabled.")
            self.llm_service = None
        else:
            provider_name = self.config.get('llm_judge', {}).get('provider', 'gemini')
            provider = LLMProvider.GEMINI if provider_name == 'gemini' else LLMProvider.OPENAI
            self.llm_service = LLMService(primary_provider=provider)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def evaluate_explanation_accuracy(
        self,
        pm_explanation: str,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate accuracy of Project Manager explanation using LLM-as-judge.
        
        Args:
            pm_explanation: Explanation provided by Project Manager agent
            ground_truth: Ground truth information about what happened
            context: Additional context about the situation
            
        Returns:
            Score from 1-5 (will be normalized to 0-1 elsewhere)
        """
        if self.llm_service is None:
            logger.warning("LLM service not available. Returning default score.")
            return 3.0  # Neutral score
        
        prompt = self._build_explanation_accuracy_prompt(
            pm_explanation, ground_truth, context
        )
        
        try:
            # Use LLM to evaluate
            response = await self._query_llm(prompt)
            score = self._extract_score(response, scale=(1, 5))
            return score
        except Exception as e:
            logger.error(f"Error evaluating explanation accuracy: {e}")
            return 3.0  # Default neutral score
    
    async def evaluate_educational_effectiveness(
        self,
        explanation: str,
        target_audience: str = "beginner"
    ) -> float:
        """
        Evaluate educational effectiveness of an explanation.
        
        Args:
            explanation: Explanation to evaluate
            target_audience: Target audience level (beginner/intermediate/advanced)
            
        Returns:
            Score from 1-5
        """
        if self.llm_service is None:
            return 3.0
        
        prompt = f"""You are evaluating an educational explanation for clarity and effectiveness.

Explanation to evaluate:
{explanation}

Target audience: {target_audience}

Rate the explanation on a scale of 1-5 based on:
1. Clarity: Is it easy to understand?
2. Completeness: Does it cover the key concepts?
3. Educational value: Does it help the reader learn?
4. Appropriate level: Is it suitable for the target audience?

Respond with ONLY a number from 1-5, followed by a brief justification (1-2 sentences).
Format: "Score: X\nJustification: ..."
"""
        
        try:
            response = await self._query_llm(prompt)
            score = self._extract_score(response, scale=(1, 5))
            return score
        except Exception as e:
            logger.error(f"Error evaluating educational effectiveness: {e}")
            return 3.0
    
    async def evaluate_decision_reasoning(
        self,
        decision: str,
        reasoning: str,
        alternatives: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate quality of decision reasoning.
        
        Args:
            decision: The decision that was made
            reasoning: Reasoning provided for the decision
            alternatives: Alternative decisions that could have been made
            
        Returns:
            Score from 1-5
        """
        if self.llm_service is None:
            return 3.0
        
        alternatives_str = "\n".join(f"- {alt}" for alt in alternatives) if alternatives else "Not specified"
        
        prompt = f"""You are evaluating the quality of decision reasoning in a machine learning pipeline.

Decision made: {decision}

Reasoning provided: {reasoning}

Alternatives considered: {alternatives_str}

Rate the reasoning quality on a scale of 1-5 based on:
1. Logical soundness: Is the reasoning logically sound?
2. Justification: Is the decision well-justified?
3. Consideration of alternatives: Were alternatives considered?
4. Clarity: Is the reasoning clear and understandable?

Respond with ONLY a number from 1-5, followed by a brief justification (1-2 sentences).
Format: "Score: X\nJustification: ..."
"""
        
        try:
            response = await self._query_llm(prompt)
            score = self._extract_score(response, scale=(1, 5))
            return score
        except Exception as e:
            logger.error(f"Error evaluating decision reasoning: {e}")
            return 3.0
    
    async def evaluate_imputation_appropriateness(
        self,
        imputation_method: str,
        data_context: Dict[str, Any],
        missing_pattern: str
    ) -> float:
        """
        Evaluate appropriateness of missing value imputation method.
        
        Args:
            imputation_method: Method used for imputation
            data_context: Context about the data (types, distribution, etc.)
            missing_pattern: Pattern of missingness (MCAR, MAR, MNAR)
            
        Returns:
            Score from 1-5
        """
        if self.llm_service is None:
            return 3.0
        
        prompt = f"""You are evaluating the appropriateness of a missing value imputation method.

Imputation method used: {imputation_method}

Data context:
- Column types: {data_context.get('column_types', 'Not specified')}
- Missing pattern: {missing_pattern}
- Data distribution: {data_context.get('distribution_info', 'Not specified')}

Rate the appropriateness on a scale of 1-5 based on:
1. Method suitability: Is the method appropriate for the data type?
2. Pattern consideration: Does it account for the missing pattern?
3. Best practices: Does it follow ML best practices?

Respond with ONLY a number from 1-5, followed by a brief justification (1-2 sentences).
Format: "Score: X\nJustification: ..."
"""
        
        try:
            response = await self._query_llm(prompt)
            score = self._extract_score(response, scale=(1, 5))
            return score
        except Exception as e:
            logger.error(f"Error evaluating imputation appropriateness: {e}")
            return 3.0
    
    async def evaluate_encoding_appropriateness(
        self,
        encoding_method: str,
        categorical_info: Dict[str, Any]
    ) -> float:
        """
        Evaluate appropriateness of categorical encoding method.
        
        Args:
            encoding_method: Encoding method used
            categorical_info: Information about categorical features
            
        Returns:
            Score from 1-5
        """
        if self.llm_service is None:
            return 3.0
        
        prompt = f"""You are evaluating the appropriateness of a categorical encoding method.

Encoding method used: {encoding_method}

Categorical feature information:
- Number of categories: {categorical_info.get('n_categories', 'Not specified')}
- Cardinality: {categorical_info.get('cardinality', 'Not specified')}
- Feature type: {categorical_info.get('feature_type', 'Not specified')}

Rate the appropriateness on a scale of 1-5 based on:
1. Method suitability: Is the method appropriate for the cardinality?
2. Information preservation: Does it preserve important information?
3. Best practices: Does it follow ML best practices?

Respond with ONLY a number from 1-5, followed by a brief justification (1-2 sentences).
Format: "Score: X\nJustification: ..."
"""
        
        try:
            response = await self._query_llm(prompt)
            score = self._extract_score(response, scale=(1, 5))
            return score
        except Exception as e:
            logger.error(f"Error evaluating encoding appropriateness: {e}")
            return 3.0
    
    async def evaluate_algorithm_selection(
        self,
        selected_algorithm: str,
        problem_context: Dict[str, Any]
    ) -> float:
        """
        Evaluate appropriateness of algorithm selection.
        
        Args:
            selected_algorithm: Algorithm that was selected
            problem_context: Context about the problem (dataset size, type, etc.)
            
        Returns:
            Score from 1-5
        """
        if self.llm_service is None:
            return 3.0
        
        prompt = f"""You are evaluating the appropriateness of a machine learning algorithm selection.

Algorithm selected: {selected_algorithm}

Problem context:
- Dataset size: {problem_context.get('n_samples', 'Not specified')} samples
- Number of features: {problem_context.get('n_features', 'Not specified')}
- Problem type: {problem_context.get('problem_type', 'Not specified')}
- Class distribution: {problem_context.get('class_distribution', 'Not specified')}

Rate the appropriateness on a scale of 1-5 based on:
1. Algorithm suitability: Is the algorithm appropriate for the problem type?
2. Scale consideration: Does it account for dataset size?
3. Best practices: Does it follow ML best practices?

Respond with ONLY a number from 1-5, followed by a brief justification (1-2 sentences).
Format: "Score: X\nJustification: ..."
"""
        
        try:
            response = await self._query_llm(prompt)
            score = self._extract_score(response, scale=(1, 5))
            return score
        except Exception as e:
            logger.error(f"Error evaluating algorithm selection: {e}")
            return 3.0
    
    def _build_explanation_accuracy_prompt(
        self,
        pm_explanation: str,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for explanation accuracy evaluation."""
        return f"""You are evaluating the accuracy of a Project Manager's explanation in a machine learning pipeline.

Project Manager's Explanation:
{pm_explanation}

Ground Truth (What Actually Happened):
{json.dumps(ground_truth, indent=2)}

Additional Context:
{json.dumps(context, indent=2) if context else 'None'}

Rate the explanation accuracy on a scale of 1-5 based on:
1. Factual correctness: Does it accurately describe what happened?
2. Completeness: Does it cover all important aspects?
3. Clarity: Is it clear and understandable?
4. Educational value: Does it help the user understand ML concepts?

Respond with ONLY a number from 1-5, followed by a brief justification (1-2 sentences).
Format: "Score: X\nJustification: ..."
"""
    
    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt."""
        if self.llm_service is None:
            raise RuntimeError("LLM service not available")
        
        try:
            # Use the LLM service's generate method
            # For evaluation, we'll use a simpler synchronous approach
            # In practice, you might need to adapt this based on your LLM service API
            
            # Try to use Gemini directly
            import google.generativeai as genai
            from app.config import get_settings
            
            settings = get_settings()
            if settings.google_api_key:
                genai.configure(api_key=settings.google_api_key)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text
            else:
                raise RuntimeError("No API key available")
                
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise
    
    def _extract_score(self, response: str, scale: tuple = (1, 5)) -> float:
        """
        Extract numeric score from LLM response.
        
        Args:
            response: LLM response text
            scale: Tuple of (min, max) score range
            
        Returns:
            Extracted score
        """
        min_score, max_score = scale
        
        # Try to find score in various formats
        patterns = [
            r'score:\s*(\d+(?:\.\d+)?)',
            r'score\s*=\s*(\d+(?:\.\d+)?)',
            r'rating:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*\d+',
            r'(\d+(?:\.\d+)?)/\d+',
            r'\b(\d+(?:\.\d+)?)\b'  # Any number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    score = float(matches[0])
                    # Clamp to scale
                    score = max(min_score, min(max_score, score))
                    return score
                except ValueError:
                    continue
        
        # If no score found, return middle of scale
        logger.warning(f"Could not extract score from response: {response[:100]}")
        return (min_score + max_score) / 2


# Convenience functions for common evaluations
async def evaluate_pm_explanation(
    explanation: str,
    ground_truth: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Evaluate a Project Manager explanation comprehensively.
    
    Returns:
        Dictionary with scores for accuracy, educational effectiveness, etc.
    """
    judge = LLMJudge()
    
    accuracy = await judge.evaluate_explanation_accuracy(explanation, ground_truth, context)
    educational = await judge.evaluate_educational_effectiveness(explanation)
    
    return {
        'explanation_accuracy': accuracy,
        'educational_effectiveness': educational,
        'overall_score': (accuracy + educational) / 2
    }

