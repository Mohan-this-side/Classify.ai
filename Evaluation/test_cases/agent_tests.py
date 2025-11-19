"""
Comprehensive Agent Test Suites
Tests for all 8 agents with quality metrics and edge cases.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
import asyncio
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_cases.base_test_framework import BaseAgentTest
from metrics.quality_metrics import QualityMetricsCalculator
from metrics.llm_judge import LLMJudge

# Import agents
from app.agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.agents.ml_pipeline.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_pipeline.ml_builder_agent import MLBuilderAgent
from app.agents.ml_pipeline.model_evaluation_agent import ModelEvaluationAgent
from app.agents.reporting.technical_reporter_agent import TechnicalReporterAgent
from app.agents.coordination.project_manager_agent import ProjectManagerAgent

logger = logging.getLogger(__name__)


class AgentTestSuite:
    """Comprehensive test suite for all agents."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the test suite."""
        self.base_test = BaseAgentTest(config_path)
        self.metrics_calc = QualityMetricsCalculator()
        self.llm_judge = LLMJudge(config_path)
        self.config = self.base_test.config
        
    # ========== Data Discovery Agent Tests ==========
    
    async def test_data_discovery_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test Data Discovery Agent."""
        logger.info(f"Testing Data Discovery Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = DataDiscoveryAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="data_discovery_basic",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        
        # Extract results
        detected_types = result.get('data', {}).get('column_types', {})
        suggested_target = result.get('data', {}).get('suggested_target', None)
        
        # Get ground truth
        ground_truth_types = metadata.get('analysis', {}).get('data_types', {}).get('all', {})
        actual_target = metadata.get('target_column')
        
        # Calculate metrics
        type_accuracy = self.metrics_calc.calculate_type_detection_accuracy(
            detected_types, ground_truth_types
        )
        target_relevance = self.metrics_calc.check_target_suggestion_relevance(
            suggested_target or "", actual_target or ""
        )
        
        # Check thresholds
        threshold = self.config['quality_thresholds']['data_discovery']['type_detection_accuracy']
        passed = type_accuracy >= threshold and target_relevance
        
        return self.base_test.record_test_result(
            test_name="data_discovery_basic",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'type_detection_accuracy': type_accuracy,
                'target_suggestion_relevance': target_relevance
            },
            details={
                'detected_types': detected_types,
                'ground_truth_types': ground_truth_types
            }
        )
    
    # ========== EDA Agent Tests ==========
    
    async def test_eda_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test EDA Agent with focus on imbalance detection."""
        logger.info(f"Testing EDA Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = EDAAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="eda_analysis",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        eda_data = result.get('data', {})
        
        # Check imbalance detection
        imbalance_info = metadata.get('analysis', {}).get('class_imbalance', {})
        is_imbalanced = imbalance_info.get('is_imbalanced', False)
        is_severely_imbalanced = imbalance_info.get('is_severely_imbalanced', False)
        
        # Check if imbalance was flagged
        imbalance_flagged = self.metrics_calc._imbalance_flagged(eda_data)
        
        # Check missing value detection
        missing_info = metadata.get('analysis', {}).get('missing_values', {})
        actual_missing = missing_info.get('percentages', {})
        detected_missing = eda_data.get('missing_values', {})
        
        missing_detection_accuracy = self.metrics_calc.calculate_missing_value_detection_accuracy(
            detected_missing, actual_missing
        )
        
        # Check correlation detection
        actual_correlations = metadata.get('analysis', {}).get('correlations', {}).get('high_correlation_pairs', [])
        detected_correlations = eda_data.get('correlations', [])
        
        correlation_completeness = self.metrics_calc.calculate_correlation_detection_completeness(
            detected_correlations, actual_correlations
        )
        
        # Calculate overall metrics
        imbalance_detection_rate = 1.0 if (not is_imbalanced or imbalance_flagged) else 0.0
        
        # Check thresholds
        thresholds = self.config['quality_thresholds']['eda_analysis']
        passed = (
            (not is_imbalanced or imbalance_detection_rate >= thresholds['imbalance_detection_rate']) and
            missing_detection_accuracy >= thresholds['missing_value_detection_accuracy'] and
            correlation_completeness >= thresholds['correlation_detection_completeness']
        )
        
        # CRITICAL: Check if severely imbalanced dataset was NOT celebrated
        if is_severely_imbalanced:
            # Check if agent flagged it as a problem, not celebrated high accuracy
            result_str = str(eda_data).lower()
            celebrates_accuracy = 'excellent' in result_str and 'accuracy' in result_str and not imbalance_flagged
            if celebrates_accuracy:
                passed = False
        
        return self.base_test.record_test_result(
            test_name="eda_analysis",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'imbalance_detection_rate': imbalance_detection_rate,
                'missing_value_detection_accuracy': missing_detection_accuracy,
                'correlation_detection_completeness': correlation_completeness,
                'imbalance_flagged': imbalance_flagged,
                'is_severely_imbalanced': is_severely_imbalanced
            }
        )
    
    # ========== Data Cleaning Agent Tests ==========
    
    async def test_data_cleaning_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test Data Cleaning Agent."""
        logger.info(f"Testing Data Cleaning Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = EnhancedDataCleaningAgent()
        
        # Calculate before metrics
        before_metrics = {
            'missing_value_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="data_cleaning",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        
        # Get cleaned dataset
        cleaned_df = state_manager.get_dataset(state, "cleaned")
        if cleaned_df is None:
            cleaned_df = df  # Fallback
        
        # Calculate after metrics
        after_metrics = {
            'missing_value_pct': (cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns))) * 100 if len(cleaned_df) > 0 else 0
        }
        
        # Calculate quality improvement
        quality_improvement = self.metrics_calc.calculate_data_quality_improvement(
            before_metrics, after_metrics
        )
        
        # Check zero variance removal
        original_features = [col for col in df.columns if col != target_col]
        cleaned_features = [col for col in cleaned_df.columns if col != target_col]
        zero_variance_removed = self.metrics_calc.check_zero_variance_removal(
            original_features, cleaned_features, df
        )
        
        # Evaluate imputation appropriateness (if missing values were handled)
        imputation_score = 3.0  # Default
        if before_metrics['missing_value_pct'] > 0:
            # Try to extract imputation method from results
            imputation_method = str(result.get('data', {})).lower()
            data_context = {
                'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            # Note: This would ideally use LLM-as-judge, but for now use default
            # In full implementation, would call: await self.llm_judge.evaluate_imputation_appropriateness(...)
        
        # Check thresholds
        thresholds = self.config['quality_thresholds']['data_cleaning']
        passed = (
            quality_improvement >= thresholds.get('data_quality_improvement_min', 0) and
            zero_variance_removed == thresholds.get('zero_variance_removal', True)
        )
        
        return self.base_test.record_test_result(
            test_name="data_cleaning",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'quality_improvement': quality_improvement,
                'zero_variance_removed': zero_variance_removed,
                'imputation_score': imputation_score,
                'before_missing_pct': before_metrics['missing_value_pct'],
                'after_missing_pct': after_metrics['missing_value_pct']
            }
        )
    
    # ========== Feature Engineering Agent Tests ==========
    
    async def test_feature_engineering_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test Feature Engineering Agent."""
        logger.info(f"Testing Feature Engineering Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = FeatureEngineeringAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="feature_engineering",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        
        # Get engineered dataset
        engineered_df = state_manager.get_dataset(state, "engineered")
        if engineered_df is None:
            engineered_df = df  # Fallback
        
        # Check for multicollinearity handling
        has_multicollinearity = metadata.get('analysis', {}).get('correlations', {}).get('has_multicollinearity', False)
        result_str = str(result).lower()
        multicollinearity_detected = any([
            'multicollinearity' in result_str,
            'correlation' in result_str and 'high' in result_str,
            'vif' in result_str
        ])
        
        # Calculate feature usefulness (if new features were created)
        original_feature_count = len([col for col in df.columns if col != target_col])
        engineered_feature_count = len([col for col in engineered_df.columns if col != target_col])
        new_features = [col for col in engineered_df.columns if col not in df.columns]
        
        feature_usefulness = 0.0
        if new_features and target_col in engineered_df.columns:
            feature_usefulness = self.metrics_calc.calculate_feature_usefulness(
                new_features, engineered_df, target_col
            )
        
        # Check Layer 2 success rate
        layer2_success = result.get('layer2_success', False)
        
        # Check thresholds
        thresholds = self.config['quality_thresholds']['feature_engineering']
        passed = (
            (not has_multicollinearity or multicollinearity_detected) and
            feature_usefulness >= thresholds.get('feature_usefulness_correlation_min', 0) if new_features else True
        )
        
        return self.base_test.record_test_result(
            test_name="feature_engineering",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'multicollinearity_detected': multicollinearity_detected,
                'has_multicollinearity': has_multicollinearity,
                'feature_usefulness': feature_usefulness,
                'new_features_count': len(new_features),
                'layer2_success': layer2_success
            }
        )
    
    # ========== ML Builder Agent Tests ==========
    
    async def test_ml_builder_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test ML Builder Agent with anti-cheating validation."""
        logger.info(f"Testing ML Builder Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = MLBuilderAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="ml_builder",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        
        # Check class balancing application
        imbalance_info = metadata.get('analysis', {}).get('class_imbalance', {})
        is_imbalanced = imbalance_info.get('is_imbalanced', False)
        is_severely_imbalanced = imbalance_info.get('is_severely_imbalanced', False)
        
        balancing_applied = self.metrics_calc.check_class_balancing_application(
            result, is_imbalanced
        )
        
        # Check model diversity
        model_diversity = self.metrics_calc.calculate_model_diversity(result)
        
        # Check for data leakage flagging
        has_leakage = 'data_leakage' in metadata.get('analysis', {})
        result_str = str(result).lower()
        leakage_flagged = any([
            'leakage' in result_str,
            'suspicious' in result_str and 'accuracy' in result_str,
            'perfect' in result_str and 'correlation' in result_str
        ]) if has_leakage else True
        
        # Get model predictions for anti-cheating score (if available)
        anti_cheating_score = 0.0
        if 'model' in result.get('data', {}) and target_col in df.columns:
            try:
                # Try to get predictions
                model = result['data'].get('model')
                if model and hasattr(model, 'predict'):
                    X_test = df.drop(columns=[target_col])
                    y_pred = model.predict(X_test)
                    y_true = df[target_col].values
                    anti_cheating_score = self.metrics_calc.calculate_anti_cheating_score(
                        y_true, y_pred
                    )
            except Exception as e:
                logger.warning(f"Could not calculate anti-cheating score: {e}")
        
        # CRITICAL: Check if severely imbalanced dataset triggers balancing
        if is_severely_imbalanced and not balancing_applied:
            passed = False
        else:
            # Check thresholds
            thresholds = self.config['quality_thresholds']['ml_builder']
            passed = (
                balancing_applied and
                model_diversity >= thresholds.get('model_diversity_min', 1) and
                anti_cheating_score >= thresholds.get('anti_cheating_score_min', 0.5) and
                leakage_flagged
            )
        
        return self.base_test.record_test_result(
            test_name="ml_builder",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'balancing_applied': balancing_applied,
                'is_severely_imbalanced': is_severely_imbalanced,
                'model_diversity': model_diversity,
                'anti_cheating_score': anti_cheating_score,
                'leakage_flagged': leakage_flagged,
                'has_leakage': has_leakage
            }
        )
    
    # ========== Model Evaluation Agent Tests ==========
    
    async def test_model_evaluation_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test Model Evaluation Agent."""
        logger.info(f"Testing Model Evaluation Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = ModelEvaluationAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="model_evaluation",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        eval_data = result.get('data', {})
        
        # Check metric completeness
        computed_metrics = list(eval_data.keys())
        metric_completeness = self.metrics_calc.calculate_metric_completeness(computed_metrics)
        
        # Check imbalance awareness
        imbalance_info = metadata.get('analysis', {}).get('class_imbalance', {})
        is_imbalanced = imbalance_info.get('is_imbalanced', False)
        imbalance_awareness = self.metrics_calc.check_imbalance_awareness(
            eval_data, is_imbalanced
        )
        
        # CRITICAL: Check if high accuracy with low minority recall is flagged
        if is_imbalanced:
            accuracy = eval_data.get('accuracy', 0)
            # Check if minority class performance is reported
            result_str = str(eval_data).lower()
            minority_performance_checked = any([
                'minority' in result_str,
                'recall' in result_str and ('class' in result_str or 'minority' in result_str),
                'specificity' in result_str
            ])
            
            # If accuracy is high (>90%) but minority performance not checked, fail
            if accuracy > 0.90 and not minority_performance_checked:
                passed = False
            else:
                passed = True
        else:
            passed = True
        
        # Check thresholds
        thresholds = self.config['quality_thresholds']['model_evaluation']
        passed = passed and (
            metric_completeness >= thresholds.get('metric_completeness', 0.8) and
            imbalance_awareness
        )
        
        return self.base_test.record_test_result(
            test_name="model_evaluation",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'metric_completeness': metric_completeness,
                'imbalance_awareness': imbalance_awareness,
                'computed_metrics': computed_metrics
            }
        )
    
    # ========== Technical Reporter Agent Tests ==========
    
    async def test_technical_reporter_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test Technical Reporter Agent."""
        logger.info(f"Testing Technical Reporter Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = TechnicalReporterAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="technical_reporter",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        report_content = str(result.get('data', {}))
        
        # Check report completeness
        report_completeness = self.metrics_calc.calculate_report_completeness(report_content)
        
        # Check visualization quality
        visualization_quality = self.metrics_calc.check_visualization_quality(result)
        
        # Check thresholds
        thresholds = self.config['quality_thresholds']['technical_reporter']
        passed = (
            report_completeness >= thresholds.get('report_completeness', 0.8) and
            visualization_quality == thresholds.get('visualization_quality', True)
        )
        
        return self.base_test.record_test_result(
            test_name="technical_reporter",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'report_completeness': report_completeness,
                'visualization_quality': visualization_quality
            }
        )
    
    # ========== Project Manager Agent Tests ==========
    
    async def test_project_manager_agent(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict
    ) -> Dict[str, Any]:
        """Test Project Manager Agent with LLM-as-judge."""
        logger.info(f"Testing Project Manager Agent on {dataset_name}")
        
        # Load dataset
        df = self.base_test.load_dataset(dataset_path)
        target_col = metadata.get('target_column')
        
        # Create state
        state = self.base_test.create_state(df, target_col)
        
        # Initialize agent
        agent = ProjectManagerAgent()
        
        # Run agent
        execution_result = await self.base_test.run_agent(agent, state)
        
        if not execution_result['success']:
            return self.base_test.record_test_result(
                test_name="project_manager",
                dataset_name=dataset_name,
                passed=False,
                metrics={'error': execution_result['error']}
            )
        
        result = execution_result['result']
        explanations = result.get('data', {}).get('explanations', {})
        
        # Evaluate explanations using LLM-as-judge
        explanation_scores = []
        for key, explanation in explanations.items():
            if isinstance(explanation, str):
                try:
                    # Create ground truth from metadata
                    ground_truth = {
                        'dataset_name': dataset_name,
                        'issues': metadata.get('expected_issues', [])
                    }
                    
                    score = await self.llm_judge.evaluate_explanation_accuracy(
                        explanation, ground_truth
                    )
                    explanation_scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not evaluate explanation {key}: {e}")
                    explanation_scores.append(3.0)  # Default
        
        # Calculate average PM accuracy
        pm_accuracy = self.metrics_calc.calculate_pm_accuracy_average(explanation_scores)
        
        # Check thresholds
        thresholds = self.config['quality_thresholds']['project_manager']
        passed = pm_accuracy >= thresholds.get('overall_pm_accuracy_min', 0.85)
        
        return self.base_test.record_test_result(
            test_name="project_manager",
            dataset_name=dataset_name,
            passed=passed,
            metrics={
                'pm_accuracy': pm_accuracy,
                'explanation_scores': explanation_scores,
                'num_explanations': len(explanations)
            }
        )
    
    # ========== Run All Tests ==========
    
    async def run_all_agent_tests(self, datasets: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Run all agent tests on all datasets.
        
        Args:
            datasets: List of dictionaries with 'name', 'path', and 'metadata' keys
            
        Returns:
            Dictionary with all test results
        """
        all_results = {}
        
        for dataset_info in datasets:
            dataset_name = dataset_info['name']
            dataset_path = dataset_info['path']
            metadata = dataset_info['metadata']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing all agents on {dataset_name}")
            logger.info(f"{'='*60}")
            
            # Run all agent tests
            results = {
                'data_discovery': await self.test_data_discovery_agent(
                    dataset_path, dataset_name, metadata
                ),
                'eda_analysis': await self.test_eda_agent(
                    dataset_path, dataset_name, metadata
                ),
                'data_cleaning': await self.test_data_cleaning_agent(
                    dataset_path, dataset_name, metadata
                ),
                'feature_engineering': await self.test_feature_engineering_agent(
                    dataset_path, dataset_name, metadata
                ),
                'ml_builder': await self.test_ml_builder_agent(
                    dataset_path, dataset_name, metadata
                ),
                'model_evaluation': await self.test_model_evaluation_agent(
                    dataset_path, dataset_name, metadata
                ),
                'technical_reporter': await self.test_technical_reporter_agent(
                    dataset_path, dataset_name, metadata
                ),
                'project_manager': await self.test_project_manager_agent(
                    dataset_path, dataset_name, metadata
                )
            }
            
            all_results[dataset_name] = results
        
        # Save results
        self.base_test.save_results("Evaluation/results/agent_level/all_agent_tests.json")
        
        return all_results

