"""
Base Test Framework for Agent Evaluation
Provides common functionality for all agent tests.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import json
import yaml
import asyncio

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.workflows.state_management import ClassificationState, state_manager
from app.workflows.classification_workflow import ClassificationWorkflow

logger = logging.getLogger(__name__)


class BaseAgentTest:
    """Base class for all agent tests."""
    
    def __init__(self, config_path: str = "Evaluation/config/evaluation_config.yaml"):
        """Initialize the test framework."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load a dataset from file."""
        return pd.read_csv(dataset_path)
    
    def load_metadata(self, dataset_name: str) -> Dict:
        """Load metadata for a dataset."""
        metadata_path = Path(f"Evaluation/datasets/metadata/{dataset_name}_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def create_state(self, dataset: pd.DataFrame, target_column: str) -> ClassificationState:
        """Create a ClassificationState for testing."""
        from app.workflows.state_management import WorkflowStatus, AgentStatus
        
        dataset_id = f"test_{dataset.shape[0]}_{id(dataset)}"
        
        # Register dataset with state_manager BEFORE creating state
        state_manager.store_dataset(
            {'dataset_id': dataset_id},  # Temporary state for storage
            dataset,
            "original"
        )
        
        state = ClassificationState(
            session_id=f"test_{dataset.shape[0]}",
            dataset_id=dataset_id,
            workflow_status=WorkflowStatus.RUNNING,
            original_dataset=dataset,  # Keep reference for direct access
            target_column=target_column,
            user_description="",
            api_key="",
            dataset_shape=dataset.shape,
            dataset_metadata={},
            data_types={},
            missing_values={},
            duplicate_count=0,
            current_agent=None,
            agent_statuses={},
            completed_agents=[],
            failed_agents=[],
            cleaned_dataset=None,
            cleaning_summary=None,
            data_quality_score=None,
            cleaning_issues_found=[],
            cleaning_actions_taken=[],
            discovery_results=None,
            similar_datasets_found=[],
            research_insights=[],
            recommended_approaches=[],
            domain_knowledge={},
            eda_plots=[],
            statistical_summary=None,
            correlation_matrix=None,
            distribution_analysis=None,
            outlier_analysis=None,
            feature_importance_initial=None,
            engineered_features=[],
            feature_selection_results=None,
            feature_transformations={},
            feature_importance_final=None,
            feature_correlation_analysis=None,
            model_selection_results=None,
            best_model=None,
            model_hyperparameters=None,
            training_metrics=None,
            cross_validation_scores=None,
            model_explanation=None,
            evaluation_metrics=None,
            confusion_matrix=None,
            classification_report=None,
            roc_auc_score=None,
            precision_recall_curve=None,
            model_performance_summary=None,
            technical_report_path=None,
            technical_report_content=None,
            notebook_path=None,
            visualizations_generated=[],
            project_manager_summary=None,
            educational_explanations={},
            user_guidance=[],
            next_steps_recommendations=[],
            warnings=[],
            # Additional required fields
            workflow_progress=0.0,
            progress=0.0,
            estimated_completion_time=None,
            resource_usage={},
            quality_checks_passed=[],
            quality_checks_failed=[],
            errors=[],
            retry_count=0,
            max_retries=3,
            error_count=0,
            last_error=None,
            start_time=None,
            end_time=None,
            total_execution_time=None,
            agent_execution_times={},
            memory_usage={},
            cpu_usage={},
            requires_human_input=False,
            human_input_required=None,
            human_feedback=None,
            user_approvals={},
            output_artifacts={},
            downloadable_files=[],
            model_path=None,
            report_path=None,
            final_report=None,
            executive_summary=None,
            technical_documentation=None,
            recommendations=[],
            limitations=[],
            future_improvements=[],
            roc_curve_data=None,
            feature_importance_model=None,
            model_performance_analysis=None
        )
        return state
    
    async def run_agent(self, agent, state: ClassificationState, timeout: int = 120) -> Dict[str, Any]:
        """Run an agent with timeout and return results."""
        try:
            # Run agent with timeout
            result = await asyncio.wait_for(
                agent.execute(state),
                timeout=timeout
            )
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except asyncio.TimeoutError:
            logger.warning(f"Agent {agent.agent_name} timed out after {timeout}s")
            # Check for partial results in state
            partial_results = self._extract_partial_results(state, agent.agent_name)
            return {
                'success': False,
                'result': partial_results,
                'error': f'Timeout after {timeout}s',
                'partial': True
            }
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            # Try to extract partial results even on error
            partial_results = self._extract_partial_results(state, agent.agent_name)
            return {
                'success': False,
                'result': partial_results,
                'error': str(e),
                'partial': partial_results is not None
            }
    
    def _extract_partial_results(self, state: ClassificationState, agent_name: str) -> Optional[Dict[str, Any]]:
        """Extract partial results from state even if agent failed."""
        partial = {}
        
        # Extract based on agent type
        if agent_name == 'data_discovery':
            if state.get('discovery_results'):
                partial['data'] = state.get('discovery_results', {})
        elif agent_name == 'eda_analysis':
            if state.get('statistical_summary') or state.get('eda_plots'):
                partial['data'] = {
                    'statistical_summary': state.get('statistical_summary'),
                    'correlation_matrix': state.get('correlation_matrix') is not None,
                    'eda_plots': state.get('eda_plots', []),
                    'distribution_analysis': state.get('distribution_analysis'),
                    'outlier_analysis': state.get('outlier_analysis')
                }
        elif agent_name == 'enhanced_data_cleaning':
            if state.get('cleaned_dataset') is not None or state.get('cleaning_summary'):
                partial['data'] = {
                    'cleaning_summary': state.get('cleaning_summary'),
                    'data_quality_score': state.get('data_quality_score'),
                    'cleaning_issues_found': state.get('cleaning_issues_found', [])
                }
        elif agent_name == 'feature_engineering':
            if state.get('engineered_features') or state.get('feature_selection_results'):
                partial['data'] = {
                    'engineered_features': state.get('engineered_features', []),
                    'feature_selection_results': state.get('feature_selection_results')
                }
        elif agent_name == 'ml_builder':
            if state.get('best_model') or state.get('model_selection_results'):
                partial['data'] = {
                    'model_selection_results': state.get('model_selection_results'),
                    'training_metrics': state.get('training_metrics')
                }
        elif agent_name == 'model_evaluation':
            if state.get('evaluation_metrics'):
                partial['data'] = state.get('evaluation_metrics', {})
        elif agent_name == 'technical_reporter':
            if state.get('notebook_path') or state.get('technical_report_path'):
                partial['data'] = {
                    'notebook_path': state.get('notebook_path'),
                    'report_path': state.get('technical_report_path')
                }
        elif agent_name == 'project_manager':
            if state.get('project_manager_summary') or state.get('educational_explanations'):
                partial['data'] = {
                    'explanations': state.get('educational_explanations', {}),
                    'summary': state.get('project_manager_summary')
                }
        
        return partial if partial else None
    
    def record_test_result(
        self,
        test_name: str,
        dataset_name: str,
        passed: bool,
        metrics: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a test result."""
        result = {
            'test_name': test_name,
            'dataset_name': dataset_name,
            'passed': passed,
            'metrics': metrics,
            'details': details or {}
        }
        self.results.append(result)
        return result
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all test results."""
        return self.results
    
    def save_results(self, output_path: str):
        """Save test results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

