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
        
        state = ClassificationState(
            session_id=f"test_{dataset.shape[0]}",
            dataset_id=f"test_{dataset.shape[0]}",
            workflow_status=WorkflowStatus.RUNNING,
            original_dataset=dataset,
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
            next_steps_recommendations=[]
        )
        return state
    
    async def run_agent(self, agent, state: ClassificationState) -> Dict[str, Any]:
        """Run an agent and return results."""
        try:
            result = await agent.execute(state)
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
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

